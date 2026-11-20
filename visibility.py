"""
visibility.py -- visibility graph among polygonal obstacles.

Key fact (Lozano-Perez & Wesley, 1979):
    Among disjoint polygonal obstacles, a shortest obstacle-avoiding path from
    s to t is a polygonal chain whose interior vertices are obstacle vertices.
So we only need a graph on {s, t} + all obstacle vertices, with an edge
between two nodes whenever the straight segment between them does not pass
through the INTERIOR of any obstacle.  Running Dijkstra/A* on that graph gives
the exact Euclidean shortest path.  (Paths may touch obstacle boundaries and
slide along edges -- a "point robot" model.)

Visibility test for a segment pq (all predicates come from geometry.py):
  1. If pq PROPERLY crosses any obstacle edge, it is blocked.
  2. Otherwise pq may still touch the boundary: pass through a vertex, start
     at a vertex, or run along an edge.  Collect every parameter t in [0, 1]
     where pq meets an obstacle boundary.  Between two consecutive such
     parameters the open piece of pq touches no boundary, so it is either
     entirely inside or entirely outside each obstacle.  Testing ONE point
     (the midpoint) of each piece with point-in-polygon decides it.
  This handles diagonals of a polygon, grazing a vertex, sliding along an
  edge and non-convex obstacles uniformly.

Naive construction is O(n^3): O(n^2) candidate pairs, O(n) edges each.
(Lee's rotational sweep achieves O(n^2 log n) -- mentioned as future work.)
"""

from __future__ import annotations

from geometry import (Point, INSIDE, proper_intersection, segment_intersection,
                      segment_param, point_in_polygon, lerp, make_ccw,
                      distance_point_to_polygon_boundary, orientation)
from search import Graph


class Obstacle:
    """A simple polygon (stored counter-clockwise) plus its bounding box."""

    def __init__(self, vertices):
        self.vertices = make_ccw([Point(*v) for v in vertices])
        xs = [p.x for p in self.vertices]
        ys = [p.y for p in self.vertices]
        self.bbox = (min(xs), min(ys), max(xs), max(ys))

    def edges(self):
        v = self.vertices
        return [(v[i], v[(i + 1) % len(v)]) for i in range(len(v))]

    def bbox_overlaps_segment(self, p: Point, q: Point) -> bool:
        x0, y0, x1, y1 = self.bbox
        return not (max(p.x, q.x) < x0 or min(p.x, q.x) > x1 or
                    max(p.y, q.y) < y0 or min(p.y, q.y) > y1)

    def strictly_contains(self, p: Point, tol: float = 1e-9) -> bool:
        """
        True if p is in the open interior.  The tiny tolerance protects the
        midpoint test against points that should lie exactly ON an edge but
        were moved off it by floating-point rounding in lerp().
        """
        x0, y0, x1, y1 = self.bbox
        if not (x0 <= p.x <= x1 and y0 <= p.y <= y1):
            return False
        if point_in_polygon(p, self.vertices) != INSIDE:
            return False
        scale = max(1.0, abs(x0), abs(y0), abs(x1), abs(y1))
        return distance_point_to_polygon_boundary(p, self.vertices) > tol * scale


def segment_is_free(p: Point, q: Point, obstacles) -> bool:
    """Does the closed segment pq avoid the interiors of all obstacles?"""
    if p == q:
        return not any(o.strictly_contains(p) for o in obstacles)

    params = [0.0, 1.0]
    relevant = []
    for obs in obstacles:
        if not obs.bbox_overlaps_segment(p, q):
            continue                                   # cheap rejection
        relevant.append(obs)
        for a, b in obs.edges():
            if proper_intersection(p, q, a, b):
                return False                           # case 1: X-crossing
            res = segment_intersection(p, q, a, b)
            if res[0] == "point":
                params.append(segment_param(res[1], p, q))
            elif res[0] == "overlap":                  # sliding along an edge
                params.append(segment_param(res[1].a, p, q))
                params.append(segment_param(res[1].b, p, q))

    # Case 2: test the midpoint of every piece between boundary contacts.
    params = sorted(set(min(1.0, max(0.0, t)) for t in params))
    for t0, t1 in zip(params, params[1:]):
        if t1 - t0 < 1e-12:
            continue
        mid = lerp(p, q, (t0 + t1) / 2)
        if any(o.strictly_contains(mid) for o in relevant):
            return False
    return True


def is_tangent(v: Point, prev: Point, nxt: Point, other: Point) -> bool:
    """
    Is the line v->other tangent to the obstacle at vertex v?  (Both polygon
    neighbours of v lie on the same side of it, or on it.)  A shortest path can
    only bend at v along such a line, so non-tangent edges may be pruned.
    """
    return orientation(v, other, prev) * orientation(v, other, nxt) >= 0


def build_visibility_graph(obstacles, extra_points: dict | None = None,
                           tangent_only: bool = False) -> Graph:
    """
    Nodes: every obstacle vertex, named (obstacle_index, vertex_index), that is
    not strictly inside another obstacle, plus each named point in
    `extra_points` (e.g. {"S": start, "G": goal}).
    Edges: all mutually visible pairs, weighted by Euclidean length.
    """
    g = Graph()
    info = {}           # node -> (prev, next) polygon neighbours, for pruning
    for i, obs in enumerate(obstacles):
        n = len(obs.vertices)
        for j, v in enumerate(obs.vertices):
            if any(o.strictly_contains(v) for k, o in enumerate(obstacles) if k != i):
                continue                     # vertex buried inside another obstacle
            g.add_node((i, j), v)
            info[(i, j)] = (obs.vertices[j - 1], obs.vertices[(j + 1) % n])
    for name, p in (extra_points or {}).items():
        g.add_node(name, p)

    nodes = list(g.pos)
    for a in range(len(nodes)):
        for b in range(a + 1, len(nodes)):
            u, v = nodes[a], nodes[b]
            pu, pv = g.pos[u], g.pos[v]
            if pu == pv:
                continue
            if tangent_only:
                if u in info and not is_tangent(pu, *info[u], pv):
                    continue
                if v in info and not is_tangent(pv, *info[v], pu):
                    continue
            if segment_is_free(pu, pv, obstacles):
                g.add_edge(u, v)
    return g
