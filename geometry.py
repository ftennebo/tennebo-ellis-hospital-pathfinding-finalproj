"""
geometry.py -- the geometry kernel.

Everything else in the project (visibility graphs, floor-plan validation,
interactive input) is built only on the routines in this file.

Design principles
-----------------
1. Predicates, not angles.  Every decision ("is r left of pq?", "do these
   segments cross?", "is x inside this polygon?") reduces to the SIGN of a
   small determinant.  We never call atan2/acos to make a decision.

2. Robust predicates.  `orientation` and `in_circle` are *filtered exact*
   predicates: we first evaluate the determinant in floating point together
   with a forward error bound (Shewchuk's "stage A" bounds).  If the result is
   farther from zero than the bound, its sign is provably correct.  Only in the
   rare near-degenerate case do we redo the computation with exact rational
   arithmetic (Python's `fractions.Fraction`, which converts floats exactly).
   So predicates never lie, yet are almost always as fast as plain floats.

3. Degeneracies are first-class.  Collinear points, overlapping segments,
   shared endpoints, parallel lines and zero-length segments all get explicit
   handling and explicit tests (see tests/test_geometry.py).

Constructions (intersection points, circumcenters, distances) necessarily
produce floating-point coordinates; only the predicates are exact.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from fractions import Fraction

# --------------------------------------------------------------------------
# Points and vectors
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Vector:
    """A displacement in the plane."""
    x: float
    y: float

    def __add__(self, other: "Vector") -> "Vector":
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vector") -> "Vector":
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, k: float) -> "Vector":
        return Vector(self.x * k, self.y * k)

    __rmul__ = __mul__

    def __neg__(self) -> "Vector":
        return Vector(-self.x, -self.y)

    def norm(self) -> float:
        return math.hypot(self.x, self.y)


@dataclass(frozen=True)
class Point:
    """A location in the plane.  Point - Point = Vector, Point + Vector = Point."""
    x: float
    y: float

    def __sub__(self, other):
        if isinstance(other, Point):
            return Vector(self.x - other.x, self.y - other.y)
        return Point(self.x - other.x, self.y - other.y)   # Point - Vector

    def __add__(self, v: Vector) -> "Point":
        return Point(self.x + v.x, self.y + v.y)

    def __iter__(self):            # allows  x, y = p  and plotting helpers
        yield self.x
        yield self.y


def dot(u: Vector, v: Vector) -> float:
    return u.x * v.x + u.y * v.y


def cross(u: Vector, v: Vector) -> float:
    """z-component of the 3D cross product = signed area of the parallelogram."""
    return u.x * v.y - u.y * v.x


def distance_squared(p: Point, q: Point) -> float:
    dx, dy = p.x - q.x, p.y - q.y
    return dx * dx + dy * dy


def distance(p: Point, q: Point) -> float:
    return math.hypot(p.x - q.x, p.y - q.y)


def lerp(p: Point, q: Point, t: float) -> Point:
    """Point at parameter t on segment pq (t=0 -> p, t=1 -> q)."""
    return Point(p.x + t * (q.x - p.x), p.y + t * (q.y - p.y))


# --------------------------------------------------------------------------
# Robust orientation predicate
# --------------------------------------------------------------------------

LEFT, COLLINEAR, RIGHT = 1, 0, -1           # return values of orientation()

_EPS = sys.float_info.epsilon / 2           # unit roundoff u = 2^-53
_CCW_ERRBOUND = (3.0 + 16.0 * _EPS) * _EPS  # Shewchuk ccwerrboundA
_ICC_ERRBOUND = (10.0 + 96.0 * _EPS) * _EPS  # Shewchuk iccerrboundA

# Statistics so experiments can report how often the exact fallback is used.
predicate_stats = {"orientation_calls": 0, "orientation_exact": 0,
                   "incircle_calls": 0, "incircle_exact": 0}


def _sign(v) -> int:
    return (v > 0) - (v < 0)


def orientation_exact(p: Point, q: Point, r: Point) -> int:
    """Sign of (q - p) x (r - p) computed with exact rational arithmetic."""
    px, py = Fraction(p.x), Fraction(p.y)
    det = (Fraction(q.x) - px) * (Fraction(r.y) - py) - \
          (Fraction(q.y) - py) * (Fraction(r.x) - px)
    return _sign(det)


def orientation_float(p: Point, q: Point, r: Point) -> int:
    """NAIVE floating-point orientation.  Kept only for the robustness experiment."""
    return _sign((q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x))


def orientation(p: Point, q: Point, r: Point) -> int:
    """
    Where is r relative to the directed line p -> q ?
        LEFT (+1)      counter-clockwise turn p, q, r
        RIGHT (-1)     clockwise turn
        COLLINEAR (0)  on the line
    Sign of (q - p) x (r - p); filtered so the answer is always exact.
    """
    predicate_stats["orientation_calls"] += 1
    detleft = (q.x - p.x) * (r.y - p.y)
    detright = (q.y - p.y) * (r.x - p.x)
    det = detleft - detright
    errbound = _CCW_ERRBOUND * (abs(detleft) + abs(detright))
    if det > errbound or -det > errbound:
        return 1 if det > 0 else -1          # float sign is provably right
    predicate_stats["orientation_exact"] += 1
    return orientation_exact(p, q, r)       # near-degenerate: be exact


def left_turn(p: Point, q: Point, r: Point) -> bool:
    return orientation(p, q, r) == LEFT


def right_turn(p: Point, q: Point, r: Point) -> bool:
    return orientation(p, q, r) == RIGHT


def collinear(p: Point, q: Point, r: Point) -> bool:
    return orientation(p, q, r) == COLLINEAR


# --------------------------------------------------------------------------
# Segments and lines
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    a: Point
    b: Point

    def length(self) -> float:
        return distance(self.a, self.b)

    def is_degenerate(self) -> bool:
        return self.a == self.b


@dataclass(frozen=True)
class Line:
    """The infinite line through two distinct points p and q."""
    p: Point
    q: Point


def point_on_segment(p: Point, a: Point, b: Point) -> bool:
    """True if p lies on the CLOSED segment ab (endpoints included)."""
    if orientation(a, b, p) != COLLINEAR:
        return False
    return (min(a.x, b.x) <= p.x <= max(a.x, b.x) and
            min(a.y, b.y) <= p.y <= max(a.y, b.y))


def segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    """Do the closed segments ab and cd share at least one point?"""
    return segment_intersection(a, b, c, d)[0] != "none"


def proper_intersection(a: Point, b: Point, c: Point, d: Point) -> bool:
    """
    True iff ab and cd cross at a single point interior to BOTH segments
    (an 'X' crossing).  Touching at an endpoint or overlapping collinearly
    is NOT proper.
    """
    o1, o2 = orientation(a, b, c), orientation(a, b, d)
    o3, o4 = orientation(c, d, a), orientation(c, d, b)
    return o1 * o2 < 0 and o3 * o4 < 0


def segment_intersection(a: Point, b: Point, c: Point, d: Point):
    """
    Full classification of the intersection of closed segments ab and cd.

    Returns one of
        ("none",)
        ("point",   Point)      a single common point
        ("overlap", Segment)    collinear segments sharing a sub-segment

    Handles zero-length segments, shared endpoints, T-junctions,
    collinear overlap and collinear-but-disjoint segments.
    """
    # --- zero-length segments are just points --------------------------
    if a == b and c == d:
        return ("point", a) if a == c else ("none",)
    if a == b:
        return ("point", a) if point_on_segment(a, c, d) else ("none",)
    if c == d:
        return ("point", c) if point_on_segment(c, a, b) else ("none",)

    o1, o2 = orientation(a, b, c), orientation(a, b, d)
    o3, o4 = orientation(c, d, a), orientation(c, d, b)

    # --- all four points collinear: 1D interval overlap ----------------
    if o1 == 0 and o2 == 0:
        # Sort points along the dominant axis of ab.
        key = (lambda p: (p.x, p.y)) if abs(b.x - a.x) >= abs(b.y - a.y) \
            else (lambda p: (p.y, p.x))
        s1 = sorted((a, b), key=key)
        s2 = sorted((c, d), key=key)
        lo = max(s1[0], s2[0], key=key)
        hi = min(s1[1], s2[1], key=key)
        if key(lo) > key(hi):
            return ("none",)
        if lo == hi:
            return ("point", lo)
        return ("overlap", Segment(lo, hi))

    # --- general position: straddle test ------------------------------
    if o1 * o2 > 0 or o3 * o4 > 0:
        return ("none",)

    # They intersect in exactly one point.  If that point is an endpoint we
    # return the endpoint itself (exact), otherwise we construct it.
    if o1 == 0:
        return ("point", c)
    if o2 == 0:
        return ("point", d)
    if o3 == 0:
        return ("point", a)
    if o4 == 0:
        return ("point", b)
    return ("point", line_intersection(Line(a, b), Line(c, d)))


def line_intersection(l1: Line, l2: Line):
    """
    Intersection point of two infinite lines, or None if they are parallel
    (this includes the coincident case).
    """
    r = l1.q - l1.p
    s = l2.q - l2.p
    denom = cross(r, s)
    if denom == 0:
        return None
    t = cross(l2.p - l1.p, s) / denom
    return l1.p + r * t


def segment_param(p: Point, a: Point, b: Point) -> float:
    """Parameter t of the projection of p onto line ab (t=0 at a, t=1 at b)."""
    ab = b - a
    return dot(p - a, ab) / dot(ab, ab)


def distance_point_to_line(p: Point, line: Line) -> float:
    d = line.q - line.p
    return abs(cross(d, p - line.p)) / d.norm()


def distance_point_to_segment(p: Point, a: Point, b: Point) -> float:
    if a == b:                                    # zero-length segment
        return distance(p, a)
    t = max(0.0, min(1.0, segment_param(p, a, b)))  # clamp projection
    return distance(p, lerp(a, b, t))


# --------------------------------------------------------------------------
# Triangles and circles
# --------------------------------------------------------------------------

INSIDE, BOUNDARY, OUTSIDE = 1, 0, -1


def signed_area2(p: Point, q: Point, r: Point) -> float:
    """Twice the signed area of triangle pqr (positive if counter-clockwise)."""
    return cross(q - p, r - p)


def triangle_area(p: Point, q: Point, r: Point) -> float:
    return abs(signed_area2(p, q, r)) / 2.0


def point_in_triangle(p: Point, q: Point, r: Point, x: Point) -> int:
    """INSIDE / BOUNDARY / OUTSIDE for point x and triangle pqr (any orientation)."""
    o = orientation(p, q, r)
    if o == COLLINEAR:                       # degenerate triangle = segment(s)
        on = point_on_segment(x, p, q) or point_on_segment(x, q, r) or \
            point_on_segment(x, r, p)
        return BOUNDARY if on else OUTSIDE
    o1, o2, o3 = orientation(p, q, x), orientation(q, r, x), orientation(r, p, x)
    if o1 == -o or o2 == -o or o3 == -o:     # strictly on the wrong side of an edge
        return OUTSIDE
    if 0 in (o1, o2, o3):
        return BOUNDARY
    return INSIDE


def circumcenter(p: Point, q: Point, r: Point):
    """Center of the circle through p, q, r, or None if they are collinear."""
    if orientation(p, q, r) == COLLINEAR:
        return None
    bx, by = q.x - p.x, q.y - p.y
    cx, cy = r.x - p.x, r.y - p.y
    d = 2.0 * (bx * cy - by * cx)
    b2, c2 = bx * bx + by * by, cx * cx + cy * cy
    ux = (cy * b2 - by * c2) / d
    uy = (bx * c2 - cx * b2) / d
    return Point(p.x + ux, p.y + uy)


def circumcircle(p: Point, q: Point, r: Point):
    """(center, radius) of the circumcircle, or None for collinear points."""
    c = circumcenter(p, q, r)
    if c is None:
        return None
    return c, distance(c, p)


def _incircle_det(a, b, c, d):
    adx, ady = a.x - d.x, a.y - d.y
    bdx, bdy = b.x - d.x, b.y - d.y
    cdx, cdy = c.x - d.x, c.y - d.y
    alift = adx * adx + ady * ady
    blift = bdx * bdx + bdy * bdy
    clift = cdx * cdx + cdy * cdy
    det = (alift * (bdx * cdy - cdx * bdy) +
           blift * (cdx * ady - adx * cdy) +
           clift * (adx * bdy - bdx * ady))
    permanent = (alift * (abs(bdx * cdy) + abs(cdx * bdy)) +
                 blift * (abs(cdx * ady) + abs(adx * cdy)) +
                 clift * (abs(adx * bdy) + abs(bdx * ady)))
    return det, permanent


def in_circle(a: Point, b: Point, c: Point, d: Point) -> int:
    """
    For a COUNTER-CLOCKWISE triangle abc:
        +1  d is strictly inside the circumcircle
         0  d is on the circle (cocircular)
        -1  d is strictly outside
    (Signs flip if abc is clockwise.)  Filtered exact, like orientation().
    """
    predicate_stats["incircle_calls"] += 1
    det, permanent = _incircle_det(a, b, c, d)
    errbound = _ICC_ERRBOUND * permanent
    if det > errbound or -det > errbound:
        return 1 if det > 0 else -1
    predicate_stats["incircle_exact"] += 1
    F = [Point(Fraction(p.x), Fraction(p.y)) for p in (a, b, c, d)]
    det, _ = _incircle_det(*F)
    return _sign(det)


# --------------------------------------------------------------------------
# Polygons
# --------------------------------------------------------------------------


def polygon_signed_area(poly) -> float:
    """Shoelace formula.  Positive for counter-clockwise vertex order."""
    s = 0.0
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        s += a.x * b.y - b.x * a.y
    return s / 2.0


def make_ccw(poly):
    """Return the polygon with counter-clockwise vertex order."""
    return list(poly) if polygon_signed_area(poly) > 0 else list(reversed(poly))


def polygon_edges(poly):
    n = len(poly)
    for i in range(n):
        yield poly[i], poly[(i + 1) % n]


def is_simple_polygon(poly) -> bool:
    """
    A polygon is simple if no two non-adjacent edges intersect and adjacent
    edges meet only at their shared vertex.  O(n^2) brute force -- fine for
    hand-drawn obstacles.
    """
    n = len(poly)
    if n < 3 or len(set(poly)) != n:
        return False
    if polygon_signed_area(poly) == 0:
        return False
    edges = list(polygon_edges(poly))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = edges[i]
            c, d = edges[j]
            res = segment_intersection(a, b, c, d)
            if res[0] == "none":
                continue
            adjacent = (j == i + 1) or (i == 0 and j == n - 1)
            if not adjacent:
                return False
            # adjacent edges must only touch at the shared vertex
            if res[0] == "overlap":
                return False
    return True


def point_in_polygon(p: Point, poly) -> int:
    """
    INSIDE / BOUNDARY / OUTSIDE for a simple polygon (convex or not, any
    orientation).  Uses the winding-number rule, implemented only with the
    exact orientation predicate, so the answer is exact.
    """
    # Boundary check first so that boundary points are reported as such.
    for a, b in polygon_edges(poly):
        if point_on_segment(p, a, b):
            return BOUNDARY
    wn = 0
    for a, b in polygon_edges(poly):
        if a.y <= p.y:
            if b.y > p.y and orientation(a, b, p) == LEFT:   # upward crossing
                wn += 1
        else:
            if b.y <= p.y and orientation(a, b, p) == RIGHT:  # downward crossing
                wn -= 1
    return INSIDE if wn != 0 else OUTSIDE


def distance_point_to_polygon_boundary(p: Point, poly) -> float:
    return min(distance_point_to_segment(p, a, b) for a, b in polygon_edges(poly))
