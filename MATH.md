# The math, step by step

This document follows the order in which the code actually runs, from the lowest-level arithmetic up to the multi-patient planner. Every section names the function it describes, so you can open the code next to it. The worked numbers were produced by running the code itself.

**The pipeline in one picture**

```
cross product ─► orientation sign ─► segment intersection ─► point in polygon
     (§1)          (§2, exact)            (§3)                    (§4)
                                              │                       │
                                              ▼                       ▼
                                   visibility test for one segment pq (§5)
                                              │
                                              ▼
                       visibility graph (§6) ── or ── hospital graph (§9)
                                              │                  │
                                              ▼                  ▼
                                   Dijkstra (§7) / A* (§8)   discretise into cells (§10)
                                                                 │
                                                                 ▼
                                                    space-time A* (§11)
                                                                 │
                                                                 ▼
                                                  prioritized planning (§12)
```

---

## 1. Vectors and the cross product (`geometry.py`: `Vector`, `dot`, `cross`)

A point is a location, and a vector is a displacement: `q - p` is the vector from p to q.

- **Dot product:** u·v = uₓvₓ + u_yv_y = |u||v|cos θ. The code uses it for projections (§3).
- **Cross product (2D):** u × v = uₓv_y − u_yvₓ = |u||v|sin θ.

The cross product is the **signed area of the parallelogram** spanned by u and v. Its sign tells you which way you turn going from u to v:

| u × v | meaning |
|---|---|
| > 0 | v is counter-clockwise from u (left turn) |
| < 0 | v is clockwise from u (right turn) |
| = 0 | u and v are parallel |

**Example:** u = (1, 2), v = (3, −1). Then u × v = 1·(−1) − 2·3 = −7 < 0, so v is clockwise from u. Swapping the arguments flips the sign: v × u = +7.

We never compute θ itself. Taking an `atan2` and comparing angles is slower and adds rounding error, while the sign of a product answers the same question directly.

## 2. Orientation, and making it exact (`orientation`, `orientation_exact`)

### 2.1 The predicate

```
orientation(p, q, r) = sign( (q − p) × (r − p) )
                     = sign( (qₓ−pₓ)(r_y−p_y) − (q_y−p_y)(rₓ−pₓ) )
```

It answers: "walking from p to q, is r on my LEFT (+1), on my RIGHT (−1), or straight ahead (0)?" This one sign is the basis of almost every later decision.

**Example:** p = (0,0), q = (4,0), r = (1,2). (q−p) × (r−p) = (4,0) × (1,2) = 4·2 − 0·1 = 8 > 0, so r is LEFT of the line from p to q. That is correct, since r is above a line that points right.

The magnitude also has a meaning: |8| / 2 = 4 is the area of triangle pqr. That is why `triangle_area` and `signed_area2` reuse the same formula.

### 2.2 Why plain floats are not enough

Floats round. When r is extremely close to the line pq, the true determinant is tiny and rounding can change its sign or turn it into zero. Here is a real case from E1:

- p = (12, 12), q = (24, 24), r = (0.5000000000000001, 0.5). Here rₓ = 0.5 + 2⁻⁵³, the next float above 0.5.
- **Float result:** 12·(0.5 − 12) − 12·(0.5000000000000001 − 12) evaluates to exactly **0.0**, so the naive code says "collinear".
- **Exact result:** −3/2⁵¹ ≈ −1.33 × 10⁻¹⁵ < 0, so r is really to the **right**. That is correct: r is a hair to the right of y = x.

A wrong "collinear" or a flipped sign can make a visibility test go the wrong way, so a path could clip through an obstacle. On the 256 × 256 grid in E1, the naive formula is wrong 8.8% of the time.

### 2.3 The filter (Shewchuk's error bound)

The code computes the two products as floats:

```
detleft  = (qₓ−pₓ)(r_y−p_y)
detright = (q_y−p_y)(rₓ−pₓ)
det      = detleft − detright
```

Each floating-point operation has a relative error of at most u = 2⁻⁵³. Following those errors through the subtractions and multiplications gives a proven bound:

```
|det_computed − det_exact|  ≤  (3u + 16u²) · (|detleft| + |detright|)  =: errbound
```

That bound gives the decision rule:

- If |det_computed| > errbound, the exact value cannot have a different sign, so the code **returns the float sign**. This is the fast path.
- Otherwise the code **recomputes with `fractions.Fraction`**. Every float is a rational number m·2ᵉ, so `Fraction(x)` holds it exactly, and rational +, −, × have no rounding at all. The resulting sign is exact.

**Easy example:** in the example from §2.1, detleft = 8 and detright = 0, so errbound ≈ 3.3 × 10⁻¹⁶ · 8 ≈ 2.7 × 10⁻¹⁵. Since 8 is far above that, the fast path answers.

**Hard example:** in the example from §2.2, det_computed = 0.0 and errbound ≈ 9.2 × 10⁻¹⁴. Since 0 is not above the bound, the code falls back to exact arithmetic and gets −3/2⁵¹, which is RIGHT.

On 100,000 random triples, the exact fallback ran **0 times**. On the adversarial E1 grid it ran for about half the points. So the predicate is exact and, in practice, as fast as plain floats.

`in_circle` works the same way. It is the sign of a 3×3 determinant, with error bound (10u + 96u²) · permanent. It is part of the kernel but is not used in the path planner.

## 3. Segment intersection (`segment_intersection`, `proper_intersection`, `point_on_segment`)

Given segments ab and cd, the code computes four orientations:

```
o1 = orient(a, b, c)    o2 = orient(a, b, d)      ← which side of ab are c and d?
o3 = orient(c, d, a)    o4 = orient(c, d, b)      ← which side of cd are a and b?
```

**Straddle test.** Two segments intersect exactly when each one's endpoints lie on opposite sides of (or on) the other's line:

- **Proper crossing (X shape):** o1·o2 < 0 **and** o3·o4 < 0.
- **No intersection:** o1·o2 > 0 or o3·o4 > 0, meaning both endpoints of one segment are strictly on the same side of the other.
- **Touching:** a zero among o1…o4 means an endpoint lies on the other segment's line.

**Worked example (proper crossing):** a = (0,0), b = (2,2), c = (0,2), d = (2,0).

| | computation | value |
|---|---|---|
| o1 | (2,2) × (0,2) = 2·2 − 2·0 | +4 |
| o2 | (2,2) × (2,0) = 2·0 − 2·2 | −4 |
| o3 | (2,−2) × (0,−2) = 2·(−2) − (−2)·0 | −4 |
| o4 | (2,−2) × (2,0) = 2·0 − (−2)·2 | +4 |

Both products are negative, so this is a proper crossing.

**Finding the crossing point** (`line_intersection`): write the first line as a + t·r with r = b − a, and the second as c + s·u with u = d − c. Setting them equal and crossing both sides with u gives

```
t = ((c − a) × u) / (r × u)
```

Here r = (2,2), u = (2,−2), r × u = −8, and (c − a) × u = (0,2) × (2,−2) = −4. So t = 0.5, and the point is a + 0.5·r = **(1, 1)**. If r × u = 0, the lines are parallel or the same line, and the function returns `None`.

**Degenerate cases, checked in this order:**

1. **Zero-length segment** (a = b): it is just a point, so the answer is "point" if `point_on_segment(a, c, d)`.
   `point_on_segment(p, a, b)` means orient(a, b, p) = 0 **and** p lies inside the bounding box of ab.
2. **All four points collinear** (o1 = o2 = 0): this becomes a 1-D problem. The code sorts the points along the longer axis and intersects the intervals [min, max]. The result is either none, a single shared point (segments touching end to end), or an overlap segment.
   Example: (0,0)–(4,0) and (2,0)–(6,0) give the overlap (2,0)–(4,0).
3. **One zero among o1…o4** (for example, a T-junction): that endpoint itself is the intersection. The code returns the input point, which is exact, and does not recompute it.
4. **Otherwise:** a proper crossing, constructed as above.

## 4. Point in polygon (`point_in_polygon`)

The question is whether point p is INSIDE, on the BOUNDARY of, or OUTSIDE a simple polygon.

1. **Boundary first.** If `point_on_segment(p, edge)` holds for any edge, the answer is BOUNDARY.
2. **Winding number.** Shoot a ray from p to the right and count the edges that cross it, with signs:
   - An edge going **up** across the ray (a_y ≤ p_y < b_y) with p to its **left** adds +1.
   - An edge going **down** across it (b_y ≤ p_y < a_y) with p to its **right** adds −1.
   - The polygon contains p if the total wn ≠ 0.

   The half-open rule (≤ on one end, < on the other) means a ray passing exactly through a vertex is counted once, not twice. Everything uses `orientation`, so the result is exact.

**Example:** square (0,0), (4,0), (4,4), (0,4) and p = (2,2).

| edge | crosses y = 2? | orientation of p | contribution |
|---|---|---|---|
| (0,0)→(4,0) | no (both y = 0) | – | 0 |
| (4,0)→(4,4) | upward | (0,4) × (−2,2) = 8 > 0, LEFT | +1 |
| (4,4)→(0,4) | no | – | 0 |
| (0,4)→(0,0) | downward | (0,−4) × (2,2) = 8 > 0, not RIGHT | 0 |

So wn = 1 and p is **INSIDE**. For p = (5,2), edge (4,0)→(4,4) gives (0,4) × (1,2) = −4, which is not LEFT, so wn = 0 and p is **OUTSIDE**.

## 5. Can I walk straight from p to q? (`visibility.py`: `segment_is_free`)

A segment is allowed if it never enters the **interior** of an obstacle. Touching the boundary is fine, because the agent is modelled as a point.

The test has two steps:

1. **Any proper crossing with an obstacle edge rules the segment out.**
2. **Split at contact points, then test each midpoint.** Collect every parameter t ∈ [0, 1] where pq touches an obstacle boundary (endpoint touches and the ends of collinear overlaps). Here t = `segment_param(x, p, q)` = ((x − p)·(q − p)) / |q − p|², which is the projection from §1. Sort these values.

   Between two consecutive values, the open piece of pq touches no boundary. Moving from outside to inside would require crossing the boundary, so each piece lies entirely inside or entirely outside each obstacle. **One point-in-polygon test at the piece's midpoint therefore decides the whole piece.**

   The midpoint is computed in floats, so it can drift off an edge by about 10⁻¹⁶. To guard against that, a midpoint counts as "inside" only if it is also more than 10⁻⁹ × scale away from the boundary (`strictly_contains`).

**Examples with the square (2,2)–(4,4):**

| segment | what happens | result |
|---|---|---|
| (0,3)→(6,3) | properly crosses edge (2,2)–(2,4) | **blocked** in step 1 |
| (2,2)→(4,4), the diagonal | no proper crossing; contacts only at t = 0, 1; midpoint (3,3) is INSIDE | **blocked** in step 2 |
| (0,2)→(6,2), sliding along the bottom edge | overlap gives t = 1/3 and 2/3; midpoints (1,2) outside, (3,2) boundary, (5,2) outside | **free** |
| (1,3)→(3,5), grazing corner (2,4) | contact at t = 1/2; midpoints (1.5,3.5) and (2.5,4.5) both outside | **free** |

Before any of this, a cheap **bounding-box test** skips obstacles that cannot touch pq at all.

## 6. Building the visibility graph (`build_visibility_graph`)

### 6.1 Why only obstacle vertices are needed

**Claim:** a shortest obstacle-avoiding path from S to G is a polyline whose bends are all at obstacle vertices.

**Proof sketch:** suppose the shortest path bends at a point x that is not an obstacle vertex.

- If x is in free space, a small disk around x is obstacle-free. Replace the bend with a straight chord across it. By the triangle inequality, |ab| < |ax| + |xb|, so the new path is strictly shorter, which is a contradiction.
- If x lies inside an obstacle **edge**, the free space near x is a half-disk. A half-disk is convex, so the chord stays in it, and the same shortcut works.

So bends happen only at vertices, and consecutive bends can see each other. **Every shortest path is a path in the graph** whose nodes are {S, G, all obstacle vertices}, with an edge for every mutually visible pair weighted by Euclidean length. Running Dijkstra or A* on that graph therefore gives the exact shortest path.

**Example** (`test_shortest_path_around_square`): S = (0,3), G = (6,3), and the square from (2,2) to (4,4). The path S → (2,4) → (4,4) → G has length √5 + 2 + √5 = **6.472**. That beats walking around the bottom, which has the same length by symmetry; either is optimal.

### 6.2 Construction and cost

- **Nodes:** all obstacle vertices, minus any buried inside another obstacle, plus S and G. Call that n nodes.
- **Edges:** for each of the n(n−1)/2 pairs, run §5. That test looks at every one of the O(n) obstacle edges.

Total time: **O(n²) pairs × O(n) per test = O(n³)**.

In experiment E2 the measured growth was about n², because the bounding-box test throws away most obstacles for each pair. Lee's rotational sweep achieves O(n² log n) and is listed as future work.

### 6.3 Tangent pruning (`is_tangent`)

A shortest path can only turn at vertex v if it **wraps around** the obstacle there. That requires the line from v to the next node to be tangent to the polygon at v: both polygon neighbours of v, `prev` and `next`, lie on the same side of the line.

```
tangent  ⇔  orient(v, other, prev) · orient(v, other, next) ≥ 0
```

If they lie on opposite sides, the line cuts into the polygon's corner, and nudging the path off v would make it shorter. Such edges can never appear in a shortest path, so they are dropped. In E2 this removed about half the edges (2420 → 1134 at n = 172) without changing any path length. The tests check this.

## 7. Dijkstra (`search.py`: `astar` with h = 0, and `dijkstra_all`)

g(v) is the best distance from the start found so far. The algorithm:

1. Set g(start) = 0 and put the start in a min-priority-queue (`heapq`).
2. Repeatedly pop the node u with the smallest g. Mark it **closed**: its g is now final.
3. **Relax** each neighbour v: if g(u) + w(u,v) < g(v), set g(v) to the new value, record parent(v) = u, and push v.

**Why a popped node is final:** all weights are ≥ 0. Any other route to u must leave the closed set through some node on the queue, and that node already has g ≥ g(u). Extending a route never makes it shorter, so no other route can beat g(u).

Running time is O((V + E) log V) with a binary heap. Stale entries in the queue are skipped when popped ("lazy deletion").

## 8. A* and the Euclidean heuristic (`astar`, `euclidean_heuristic`)

A* is Dijkstra ordered by **f(v) = g(v) + h(v)** instead of g(v), where h(v) is an estimate of the remaining distance. The code uses h(v) = |v G|, the straight-line distance to the goal.

- **Admissible (h ≤ true remaining distance):** no path is shorter than a straight line, so h never overestimates. This guarantees the path A* returns is optimal.
- **Consistent (h(u) ≤ w(u,v) + h(v)):** every edge here is a straight segment with w(u,v) = |uv|. The triangle inequality gives |uG| ≤ |uv| + |vG|. Consistency means f never decreases along a path. As a result, A* behaves exactly like Dijkstra on the reduced weights w′(u,v) = w(u,v) − h(u) + h(v) ≥ 0, so §7's proof carries over: a closed node is final, and when G is popped its g is optimal.

**Why A* expands fewer nodes (geometric picture):** let C* be the optimal path length. A* only expands nodes with f(v) < C*, which means g(v) + |vG| < C*. Since |Sv| ≤ g(v), every expanded node satisfies

```
|S v| + |v G| < C*
```

That is the inside of an **ellipse with foci S and G**. Dijkstra instead expands every node with g(v) < C*, which is a disk of radius C* around S. The ellipse is thin when obstacles are few, so A* explores much less.

**Numbers:**

- Hospital route ER → Ward C: length 45.0. A* expanded 24 nodes, Dijkstra 37.
- E3 random scene with n = 172: A* expanded 21, Dijkstra 172.

The path lengths were identical every time, as the theory says they must be.

## 9. The hospital graph (`hospital.py`: `FloorPlan.validate`)

The floor plan is a hand-made graph: room nodes, door nodes, and corridor waypoints. Weights are Euclidean lengths, so §8 applies unchanged, including consistency. The kernel **proves** the graph is physically walkable:

- For every graph edge and every wall segment, `segment_intersection` (§3) must return `none`. Doors are gaps in the walls, so an edge can only pass through a door.
- Every room node must be INSIDE its room rectangle, checked with `point_in_polygon` (§4).

If either check fails, the program refuses to start.

## 10. Discretising time (`prioritized.py`: `discretize`)

For several patients we need a notion of time. Each edge of length L is cut into

```
k = max(1, round(L / step))
```

cells, and moving one cell takes **one tick**. With step = 1 m, a 6 m corridor segment becomes 6 cells, and the hospital graph becomes 127 cells. Each edge's travel time is off from L/step by at most ½ tick because of rounding.

## 11. Space-time A* (`space_time_astar`)

### 11.1 States and moves

A state is **(cell, t)**. From (u, t), the possible actions are:

- **Move** to a neighbour v, reaching (v, t+1), or
- **Wait**, reaching (u, t+1).

Every action costs 1, so the cost of a plan is simply its **arrival time**.

The search graph is the floor graph copied once per time step, with arrows from layer t to layer t+1. There are at most |cells| × T_max states, with T_max = 500. That bound guarantees the search terminates even when no plan exists.

### 11.2 The rules: conflicts (`ReservationTable.move_allowed`)

Higher-priority patients have reserved cells in space-time. A move u → v from time t to t+1 is forbidden if it causes either conflict:

| conflict | condition | meaning |
|---|---|---|
| **vertex** | someone occupies v at t+1, or is parked at v | two beds in one cell |
| **edge (swap)** | someone moves v → u during the same tick | two beds pass head-on in a single-lane corridor |

Following one cell behind another patient in the same direction is allowed.

### 11.3 The heuristic

h(v) = the exact distance in cells from v to the goal, ignoring everyone else. It is computed once by `dijkstra_all` from the goal; the graph is undirected, so distances from the goal equal distances to it.

- **Admissible:** other patients can only add waits or detours, never shortcuts, so the true remaining time is at least h(v).
- **Consistent:** h(v) ≤ 1 + h(v′) for every move. A wait costs 1 and leaves h unchanged, so it is consistent too.

So space-time A* returns the **earliest possible arrival** given the reservations.

**Tie-breaking:** among plans with equal arrival time, the code prefers fewer **moves**. The priority is the pair (t + h, moves + h), compared lexicographically. The effect is that a patient who has to give way stands still instead of stepping back and forth.

### 11.4 When is the patient done?

Reaching the goal at time t only counts if nobody else passes through the goal cell after t, because the patient stays in the room. The condition is t > `last_reserved_time(goal)`.

### 11.5 Worked example (`tests/test_hospital.py::test_head_on_lower_priority_yields`)

The corridor is cells 0–7, with a side **pocket** off cell 3.

- **hi** (priority 1) goes 0 → 7. **lo** (priority 2) goes 5 → 1, head-on.
- hi is planned first against an empty table: `[0,1,2,3,4,5,6,7]`, arriving at t = 7 with delay 0.
- lo is then planned around hi: `[5, 4, 3, pocket, 3, 2, 1]`.

| t | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| hi | 0 | 1 | 2 | **3** | 4 | 5 | 6 | 7 |
| lo | 5 | 4 | 3 | **pocket** | 3 | 2 | 1 | 1 |

At t = 3, hi needs cell 3, so lo has stepped into the pocket. Between t = 2 and 3, hi moves 2→3 while lo moves 3→pocket; that is not a swap, so it is allowed. lo's time alone would be 4 ticks; it actually takes 6, so its **delay is 2**.

## 12. Prioritized planning and absolute priority (`plan_prioritized`)

```
table ← empty
for each patient in priority order 1, 2, 3, ...:
    path ← space_time_astar(patient, table)     # §11
    table.reserve(path)                          # cells, moves, and the goal from arrival onward
```

### 12.1 Why P1 is never delayed (correct for P1)

P1 is planned against an **empty** table, so space-time A* with no conflicts is just a shortest path. P1's arrival equals its solo time, and its delay is 0.

Later patients only **add** reservations. They never change earlier ones, and every later plan must avoid all earlier reservations. By induction, patient k's plan is the earliest arrival possible given that patients 1…k−1 are fixed.

So higher priority truly dominates: nothing a lower-priority patient does can slow a higher one. `test_priority_is_absolute` checks this by showing that adding patients 2 and 3 leaves P1's path exactly the same.

### 12.2 Why it is not globally optimal

- **Not optimal for total time.** Making P1 one tick slower might save P2 ten ticks. The total time would drop, but absolute priority forbids it. Optimizing the weighted total is what CBS/MAPF solvers do (future work).
- **Not complete.** In `test_prioritized_planning_is_incomplete` (corridor 0–5, pocket at 2, hi 0→5, lo 4→1), lo needs 3 ticks to reach the pocket, but hi occupies the junction at t = 2. A joint plan exists if hi waits a single tick. Prioritized planning never delays hi, so lo gets **no plan** at all.
- **Not even lexicographically optimal in general.** P1 often has several equally fast routes, and planning commits to one of them without considering who comes next. A different, equally fast route for P1 might have let P2 through sooner.

E4 shows the consequences: mean delay grows with priority rank (rank 1 is always 0), and the success rate drops from 100% to 93% at 8 patients. Some of those failures may be truly unsolvable instances.

### 12.3 Cost

Each patient costs one Dijkstra plus one space-time A* over at most |cells| × T states. That works out to O(K · |cells| · T · log(|cells| · T)) for K patients. In practice it took a few milliseconds per scenario (E4).

---

### Summary in one sentence per step

1. Every decision is the **sign of a cross product**.
2. That sign is made **exact** by a float error bound with a rational-arithmetic fallback.
3. Segment intersection is **four orientation signs**, plus explicit handling of degenerate cases.
4. Point-in-polygon is a **winding number** built from orientation signs.
5. A segment is visible if it crosses no edge properly and the **midpoint of every piece between contacts** is outside all obstacles.
6. Shortest paths bend only at **obstacle vertices**, so the visibility graph contains the answer; building it costs O(n³).
7. **Dijkstra** finalizes nodes in order of distance.
8. **A\*** with the straight-line heuristic does the same within an ellipse around S and G and returns the same optimal length.
9. Space-time A* searches over **(cell, time)** with wait moves and forbids vertex and swap conflicts.
10. Planning in priority order makes P1 provably undelayed, but the overall result is neither complete nor optimal for total time.
