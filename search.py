"""
search.py -- weighted graphs and shortest-path search (Dijkstra and A*).

Written from scratch on top of Python's `heapq` priority queue (a basic data
structure, not a path-finding library).  The same code is used for the
free-plane visibility graph and for the hospital floor plan.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field

from geometry import Point, distance


class Graph:
    """Undirected weighted graph whose nodes have positions in the plane."""

    def __init__(self):
        self.pos: dict = {}          # node -> Point
        self.adj: dict = {}          # node -> {neighbor: weight}

    def add_node(self, node, point: Point):
        self.pos[node] = point
        self.adj.setdefault(node, {})

    def add_edge(self, u, v, weight: float | None = None):
        if weight is None:                       # default: Euclidean length
            weight = distance(self.pos[u], self.pos[v])
        self.adj[u][v] = weight
        self.adj[v][u] = weight

    def neighbors(self, u):
        return self.adj[u].items()

    def edges(self):
        """Each undirected edge once, as (u, v, w)."""
        seen = set()
        for u, nbrs in self.adj.items():
            for v, w in nbrs.items():
                key = frozenset((u, v))
                if key not in seen:
                    seen.add(key)
                    yield u, v, w

    def num_edges(self) -> int:
        return sum(len(n) for n in self.adj.values()) // 2

    def __len__(self):
        return len(self.pos)


@dataclass
class SearchResult:
    path: list | None                 # list of nodes from start to goal (None = unreachable)
    cost: float                       # total path length (inf if unreachable)
    expanded: int                     # number of nodes popped & expanded (for experiments)
    expanded_order: list = field(default_factory=list)   # for visualisation


def euclidean_heuristic(graph: Graph, goal):
    """
    h(n) = straight-line distance to the goal.
    Admissible (never over-estimates, because a straight line is the shortest
    curve) and consistent (triangle inequality), so A* returns optimal paths
    and never needs to re-open a node.
    """
    g = graph.pos[goal]
    return lambda n: distance(graph.pos[n], g)


def astar(graph: Graph, start, goal, heuristic=None) -> SearchResult:
    """
    A* search.  With heuristic=None (h = 0) this IS Dijkstra's algorithm.

    Priority queue ordered by f(n) = g(n) + h(n), where g(n) is the best known
    distance from the start.  Lazy deletion: stale queue entries are skipped.
    """
    if heuristic is None:
        heuristic = lambda n: 0.0
    tie = itertools.count()                 # tie-breaker so nodes are never compared
    g = {start: 0.0}
    parent = {start: None}
    closed = set()
    order = []
    pq = [(heuristic(start), next(tie), start)]

    while pq:
        f, _, u = heapq.heappop(pq)
        if u in closed:                     # stale entry
            continue
        closed.add(u)
        order.append(u)
        if u == goal:                       # goal popped => its g is optimal
            return SearchResult(_reconstruct(parent, goal), g[goal], len(closed), order)
        for v, w in graph.neighbors(u):
            if v in closed:
                continue
            new_g = g[u] + w
            if new_g < g.get(v, float("inf")):     # edge relaxation
                g[v] = new_g
                parent[v] = u
                heapq.heappush(pq, (new_g + heuristic(v), next(tie), v))
    return SearchResult(None, float("inf"), len(closed), order)


def dijkstra(graph: Graph, start, goal) -> SearchResult:
    return astar(graph, start, goal, heuristic=None)


def dijkstra_all(graph: Graph, source) -> dict:
    """Shortest distance from `source` to every reachable node (single-source)."""
    tie = itertools.count()
    dist = {source: 0.0}
    done = set()
    pq = [(0.0, next(tie), source)]
    while pq:
        d, _, u = heapq.heappop(pq)
        if u in done:
            continue
        done.add(u)
        for v, w in graph.neighbors(u):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(pq, (nd, next(tie), v))
    return dist


def _reconstruct(parent, goal):
    path = []
    n = goal
    while n is not None:
        path.append(n)
        n = parent[n]
    return path[::-1]
