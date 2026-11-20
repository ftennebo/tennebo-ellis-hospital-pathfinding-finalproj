# Hospital Path Planning: computational geometry final project

This project finds shortest routes among polygonal obstacles with a visibility graph and then routes patients through a hospital floor plan. When several patients move at once, each has an absolute priority: higher-priority patients are never delayed, and lower-priority patients wait or take a detour around them.

Everything geometric and every search algorithm is written from scratch. Libraries are used only for plotting/GUI (matplotlib) and testing (pytest), as the course rules allow. Python's standard `heapq` (a priority queue) and `fractions` (exact rationals) are used as basic data types.

## Setup and running

```bash
pip install matplotlib pillow pytest      # pillow is only needed to save GIFs

python -m pytest -q                       # 50 tests

python demo_free_plane.py --sample        # interactive visibility graph
python demo_hospital.py route             # click two rooms
python demo_hospital.py patients          # animated multi-patient planning
python experiments.py                     # experiments (~15 s), figures/ + tables
```

Each demo can also render to a file without opening a window (use this for slides or as a backup if the live demo fails):

```bash
python demo_free_plane.py --sample --save figures/free_plane.png
python demo_free_plane.py --sample --tangent --save figures/free_plane_tangent.png
python demo_hospital.py route --start ER --goal "Ward C" --save figures/hospital_route.png
python demo_hospital.py patients --save figures/patients.gif
python demo_hospital.py patients --save figures/patients_snapshot.png --snapshot 14
python demo_hospital.py patients --patients "ER>OR, Ward B>Lab, OR>ER"   # your own scenario
```

Room names: ER, Radiology, Elevator, ICU, OR, Ward A, Ward B, Pharmacy, Lab, Ward C.

### Free-plane demo controls

| input | action |
|---|---|
| left click | add obstacle vertex (mode O), place start (mode S) or goal (mode G) |
| right click / Enter | close the polygon (rejected if it is not simple) |
| `o` `s` `g` | switch mode: Obstacle / Start / Goal |
| `u` | undo last vertex |
| `v` | show/hide visibility graph |
| `t` | tangent-edge pruning on/off |
| `a` | A* ↔ Dijkstra |
| `e` | show nodes expanded by the search |
| `l` / `c` | load sample scene / clear |

The path is recomputed after every change, and the title shows the node and edge counts, build time, path length and number of nodes expanded.

## Files

| file | contents | course part |
|---|---|---|
| `geometry.py` | Point/Vector, dot, cross, distances, **robust orientation**, segment intersection (all degenerate cases), lines, point–line and point–segment distance, triangle area, point-in-triangle, circumcircle, **robust in-circle**, polygon area/orientation, simplicity test, point-in-polygon | Parts 1–3, Robustness |
| `visibility.py` | obstacles, the segment visibility test, visibility-graph construction, tangent pruning | motion planning |
| `search.py` | weighted graph, Dijkstra, A*, single-source Dijkstra | |
| `hospital.py` | floor plan (rooms, walls, doors, corridor graph), checked against the walls using the kernel | |
| `prioritized.py` | time discretisation, reservation table, space-time A*, prioritized planning, independent conflict checker | final system |
| `demo_free_plane.py` | interactive obstacle/start/goal input and display | Part 0, visualisation |
| `demo_hospital.py` | interactive routing and the multi-patient animation | visualisation |
| `experiments.py` | experiments E1–E4 | experiments |
| `tests/` | pytest suites for every module | |
| `MATH.md` | step-by-step explanation of all the math, in execution order | report |
| `figures/` | pre-rendered images, the GIF, and experiment plots | |

## How it works

### 1. Geometry kernel (`geometry.py`)

- **No angles.** Every decision is the sign of a determinant. `orientation(p,q,r)` is the sign of (q − p) × (r − p).
- **Exact predicates.** `orientation` and `in_circle` first compute in floating point with Shewchuk's forward error bound. If |det| is larger than the bound, the sign is guaranteed correct. Otherwise the code recomputes with `Fraction`, which is exact because every float is a rational number. On random input the slow path never runs (0 of 100,000 calls); it runs only near degeneracies.
- **Degeneracies.** `segment_intersection` returns `none`, `point` or `overlap`. It handles collinear overlap, collinear-but-disjoint segments, shared endpoints, T-junctions and zero-length segments, and each case has a test.
- **Point in polygon** uses the winding number, built only from `orientation`, so it is exact too. Boundary points are reported separately.

### 2. Visibility graph (`visibility.py`)

**Theorem (Lozano-Pérez & Wesley 1979).** Among polygonal obstacles, a shortest path bends only at obstacle vertices. So the graph on {start, goal, obstacle vertices}, with an edge between every pair that can see each other, contains the shortest path. Dijkstra or A* on that graph finds it exactly.

**Visibility test for a segment pq:**
1. If pq properly crosses an obstacle edge (an X-crossing), it is blocked.
2. Otherwise pq can still touch obstacle boundaries: it can run along an edge, pass through a vertex, or be a diagonal. The code collects every parameter t where pq touches a boundary. Between two consecutive touches, the piece of pq touches no boundary, so it lies entirely inside or entirely outside each obstacle. One point-in-polygon test at its midpoint decides which. This single argument covers every degenerate case.

**Tangent pruning.** At a vertex v, a shortest path can only turn along a line that is tangent to the obstacle there, meaning both polygon neighbours of v are on the same side. Dropping all other edges removes about half of them without changing the optimum (checked in tests and in E2).

**A\*** uses the straight-line distance as its heuristic. It is admissible (a straight line is shortest) and consistent (triangle inequality), so A* stays optimal and expands far fewer nodes.

### 3. Hospital mode (`hospital.py`)

Ten rooms, two non-walkable blocks, and two horizontal corridors joined by three vertical connectors. The connectors form two loops, so there are alternative routes. Corridor waypoints, door nodes and room nodes form the graph. `FloorPlan.validate()` uses the kernel to show that no graph edge intersects a wall (door gaps are the only way through) and that each room node is inside its room.

### 4. Several patients with absolute priority (`prioritized.py`)

- **Discretisation.** Each corridor edge is cut into cells about 1 m long, and one move takes one tick.
- **Rules.** Corridors are single-lane for beds. There must be no *vertex conflict* (two patients in one cell) and no *swap conflict* (two patients exchanging cells head-on).
- **Prioritized planning.** Patients are planned from highest priority to lowest. Each is planned with **space-time A\*** over states (cell, t), whose actions are "move to a neighbour" or "wait". Reservations made by earlier patients are forbidden. The finished path is then reserved in a space-time table. Patient 1 is planned against an empty table, so it gets its solo-optimal route, which is what *absolute* priority means. A test checks that adding lower-priority patients never changes patient 1's plan.
- **Heuristic.** The exact solo distance to the goal (one Dijkstra from the goal) gives each patient the earliest arrival that is still possible. Ties are broken by fewest moves, so a patient who has to give way waits instead of shuffling back and forth.
- `check_conflicts` re-checks every final plan independently.

Default scenario (`python demo_hospital.py patients`):

| patient | route | alone | actual | what happens |
|---|---|---|---|---|
| P1 critical | ER → OR | 34 | 34 | unaffected (highest priority) |
| P2 urgent | ICU → Radiology | 18 | 27 | head-on with P1: **waits** at the junction and pulls aside |
| P3 stable | Ward C → ICU | 30 | 36 | its short route is busy, so it **detours** around the other loop |
| P4 routine | Elevator → Ward A | 30 | 30 | unaffected |

## Experiments (`python experiments.py`)

| | question | result |
|---|---|---|
| E1 | Does naive float orientation fail? | Yes. On 256×256 points within a few ulps of y = x, it gives the wrong sign 8.8% of the time. The filtered predicate is always right (`figures/e1_robustness.png`). |
| E2 | How big and slow is the visibility graph? | 172 nodes: 2420 edges, 1134 after tangent pruning. Construction is O(n³) in the worst case. Measured growth is about n² because bounding-box rejection makes most pair tests cheap (`figures/e2_e3_visibility.png`). |
| E3 | A* vs Dijkstra | Same path length every time. A* expanded 21 nodes vs 172 for Dijkstra at n = 172. |
| E4 | Prioritized planning as the hospital gets busier | Mean delay grows with priority rank (rank 1 always has zero delay). 100% of instances solved up to 5 patients, 93% with 8 (`figures/e4_prioritized.png`). |

## 15-minute demo script

1. **(2 min) Problem.** Hospital transport: route patients quickly, and an emergency must never be slowed down by routine traffic. In geometric terms: shortest paths among obstacles, then several moving agents.
2. **(3 min) Kernel and robustness.** Show `orientation` and explain why it uses a sign and not an angle. Show `figures/e1_robustness.png`: naive floats give jagged wrong regions, the filtered exact predicate gives a clean diagonal. Make the point that being exact costs almost nothing on normal input.
3. **(4 min) Free-plane demo.** Run `demo_free_plane.py --sample`.
   - State the theorem: shortest paths bend only at obstacle vertices.
   - Draw a new obstacle live and watch the path change.
   - Press `t` to show tangent pruning (fewer edges, same path).
   - Press `a` and `e` to compare A* and Dijkstra (same length, far fewer nodes expanded).
   - Show the degenerate cases: a segment sliding along an edge is allowed, a diagonal is blocked. Explain the split-at-touch-points, test-the-midpoint argument.
4. **(2 min) Hospital routing.** Run `demo_hospital.py route` and click two rooms. The graph is validated against the walls using the kernel's segment intersection test.
5. **(3 min) Several patients.** Run `demo_hospital.py patients`.
   - P1 (red) keeps its optimal route.
   - P2 (orange) meets P1 head-on in a single-lane corridor and waits at the junction.
   - P3 (blue) detours around the other loop.
   - Explain space-time A* and the reservation table.
   - Show E4: delay moves onto lower priorities.
6. **(1 min) Limitations and future work.** See below.

## Limitations and future work

- **Optimizing total travel time.** Prioritized planning is neither complete nor optimal for the sum of travel times. `tests/test_hospital.py::test_prioritized_planning_is_incomplete` shows a case where a joint plan exists but absolute priority rules it out. Conflict-Based Search (CBS) and other MAPF solvers optimize weighted total time and are the natural next step. Some E4 failures may be truly unsolvable instances and not only incompleteness.
- **Faster visibility graph.** Lee's rotational sweep builds it in O(n² log n), and Ghosh–Mount in O(n log n + E).
- **Robot size.** Real beds have a width. Grow the obstacles by the bed's shape (Minkowski sum, configuration space) and plan for a point.
- **Continuous time.** Safe-Interval Path Planning (SIPP) avoids discretising time, and edges could be given different speeds (for example, slower around corners).
- **Multiple floors.** Elevators as time-dependent edges.
