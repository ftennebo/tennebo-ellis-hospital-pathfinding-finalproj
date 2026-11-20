"""
prioritized.py -- multiple patients with ABSOLUTE priority.

Model
-----
* Time is discrete: t = 0, 1, 2, ...  Every corridor edge is subdivided into
  short cells (about `step` units long), so moving one cell takes one tick.
* Corridors are single-lane for beds: two patients may never be in the same
  cell at the same tick (VERTEX conflict) and may never swap cells head-on
  in the same tick (EDGE / swap conflict).
* A patient that reaches its destination room stays there (the room is
  reserved from then on).

Prioritized planning (Erdmann & Lozano-Perez 1987, Silver 2005)
------------------------------------------------------------------
    for patients in order of decreasing priority:
        plan with SPACE-TIME A* that avoids all reservations made so far
        reserve the resulting path in the space-time reservation table

Because the highest-priority patient is planned first against an empty
table, it gets its individually optimal route: priority is ABSOLUTE, a
lower-priority patient can never delay a higher one.  Lower-priority
patients either WAIT (stay in a cell) or DETOUR (take another corridor).

Limitations (good presentation points): prioritized planning is not
complete (a low-priority patient may find no plan even if a joint plan
exists) and not optimal for the SUM of travel times.  Conflict-Based Search
(CBS) / MAPF solvers address that -- future work.
"""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass

from geometry import lerp
from search import Graph, dijkstra_all


# ---------------------------------------------------------------------------
# 1. Discretise a weighted graph into unit-time cells
# ---------------------------------------------------------------------------

def discretize(graph: Graph, step: float = 1.0) -> Graph:
    """
    Return a new graph in which every edge of length L is replaced by
    k = max(1, round(L / step)) unit-weight edges through intermediate
    nodes named (u, v, i).  Original node names are preserved.
    """
    d = Graph()
    for n, p in graph.pos.items():
        d.add_node(n, p)
    for u, v, w in graph.edges():
        k = max(1, round(w / step))
        prev = u
        for i in range(1, k):
            mid = (u, v, i)
            d.add_node(mid, lerp(graph.pos[u], graph.pos[v], i / k))
            d.add_edge(prev, mid, 1.0)
            prev = mid
        d.add_edge(prev, v, 1.0)
    return d


# ---------------------------------------------------------------------------
# 2. Space-time reservation table
# ---------------------------------------------------------------------------

class ReservationTable:
    def __init__(self):
        self.vertex = {}        # (node, t)      -> agent id
        self.edge = {}          # (u, v, t)      -> agent id   (move u->v during [t, t+1])
        self.parked = {}        # node -> (t_from, agent id)  occupied forever from t_from

    def occupant(self, node, t):
        """Who (if anyone) occupies `node` at time t."""
        if (node, t) in self.vertex:
            return self.vertex[(node, t)]
        if node in self.parked and t >= self.parked[node][0]:
            return self.parked[node][1]
        return None

    def move_allowed(self, u, v, t) -> bool:
        """May an agent at u at time t be at v at time t+1?  (u == v is a wait.)"""
        if self.occupant(v, t + 1) is not None:          # vertex conflict
            return False
        if u != v and (v, u, t) in self.edge:            # head-on swap
            return False
        return True

    def last_reserved_time(self, node) -> float:
        """Latest time anybody passes through `node` (inf if somebody parks there)."""
        if node in self.parked:
            return math.inf
        times = [t for (n, t) in self.vertex if n == node]
        return max(times, default=-1)

    def reserve(self, path, agent_id):
        for t, node in enumerate(path):
            self.vertex[(node, t)] = agent_id
            if t + 1 < len(path):
                self.edge[(node, path[t + 1], t)] = agent_id
        self.parked[path[-1]] = (len(path) - 1, agent_id)


# ---------------------------------------------------------------------------
# 3. Space-time A*
# ---------------------------------------------------------------------------

def space_time_astar(graph: Graph, start, goal, table: ReservationTable,
                     h: dict, max_time: int = 500):
    """
    Search over states (node, t).  From (u, t) we may move to any neighbour v
    or WAIT at u, reaching time t + 1, provided the reservation table allows
    it.  Cost = arrival time, so f = t + h(node) where h is the exact
    single-agent distance to the goal (precomputed by Dijkstra from the goal):
    admissible, hence the returned path is the EARLIEST possible arrival
    given the higher-priority reservations.

    A goal state is accepted only if nobody else will pass through the goal
    afterwards (the patient stays in the room).

    Tie-breaking: among plans with the same arrival time we prefer the one
    with the fewest MOVES (lexicographic cost (time, moves)).  So a patient
    who has to give way simply waits instead of shuffling back and forth --
    it looks sensible in the animation and is still time-optimal.

    Returns a list `path` with path[t] = node at time t, or None.
    """
    if table.occupant(start, 0) is not None:
        return None
    goal_free_after = table.last_reserved_time(goal)
    tie = itertools.count()
    h0 = h.get(start, math.inf)
    pq = [(h0, h0, next(tie), 0, start)]      # (f_time, f_moves, tie, t, node)
    parent = {(start, 0): None}
    moves = {(start, 0): 0}                   # best known #moves to reach (node, t)
    closed = set()

    while pq:
        _, _, _, t, u = heapq.heappop(pq)
        if (u, t) in closed:
            continue
        closed.add((u, t))
        if u == goal and t > goal_free_after:
            return _reconstruct(parent, (u, t))
        if t >= max_time:
            continue
        for v in list(graph.adj[u]) + [u]:                 # neighbours + wait
            if (v, t + 1) in closed or not table.move_allowed(u, v, t):
                continue
            hv = h.get(v, math.inf)
            if hv == math.inf:
                continue                                    # cannot reach goal
            m = moves[(u, t)] + (u != v)
            if m < moves.get((v, t + 1), math.inf):
                moves[(v, t + 1)] = m
                parent[(v, t + 1)] = (u, t)
                heapq.heappush(pq, (t + 1 + hv, m + hv, next(tie), t + 1, v))
    return None


def _reconstruct(parent, state):
    path = []
    while state is not None:
        path.append(state[0])
        state = parent[state]
    return path[::-1]


# ---------------------------------------------------------------------------
# 4. Prioritized planning
# ---------------------------------------------------------------------------

@dataclass
class Patient:
    name: str
    start: str
    goal: str
    priority: int            # 1 = most urgent (absolute priority)
    color: str = "tab:red"


@dataclass
class PatientPlan:
    patient: Patient
    path: list | None        # path[t] = cell at time t
    solo_time: int           # arrival time if alone in the hospital
    waits: int = 0           # ticks spent standing still before arrival

    @property
    def arrival(self):
        return None if self.path is None else len(self.path) - 1

    @property
    def delay(self):
        return None if self.path is None else self.arrival - self.solo_time


def plan_prioritized(grid: Graph, patients, max_time: int = 500):
    """
    Plan all patients in priority order (1 first).  `grid` is the
    discretised graph.  Returns a list of PatientPlan in priority order.
    """
    table = ReservationTable()
    plans = []
    for idx, p in enumerate(sorted(patients, key=lambda q: q.priority)):
        h = dijkstra_all(grid, p.goal)           # exact distance-to-goal heuristic
        solo = int(h.get(p.start, math.inf)) if p.start in h else -1
        path = space_time_astar(grid, p.start, p.goal, table, h, max_time)
        if path is not None:
            table.reserve(path, idx)
            waits = sum(1 for a, b in zip(path, path[1:]) if a == b)
        else:
            waits = 0
        plans.append(PatientPlan(p, path, solo, waits))
    return plans


def check_conflicts(plans) -> list:
    """Independent verifier: list every vertex or swap conflict between plans."""
    found = []
    ok = [pl for pl in plans if pl.path]
    horizon = max((len(pl.path) for pl in ok), default=0) + 1

    def at(pl, t):                        # agents stay at their goal afterwards
        return pl.path[min(t, len(pl.path) - 1)]

    for t in range(horizon):
        for i in range(len(ok)):
            for j in range(i + 1, len(ok)):
                a, b = ok[i], ok[j]
                if at(a, t) == at(b, t):
                    found.append(("vertex", a.patient.name, b.patient.name, t))
                if at(a, t) == at(b, t + 1) and at(a, t + 1) == at(b, t) \
                        and at(a, t) != at(a, t + 1):
                    found.append(("swap", a.patient.name, b.patient.name, t))
    return found
