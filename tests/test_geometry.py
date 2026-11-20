"""Tests for the geometry kernel, with explicit degenerate cases."""
import math
import random

import pytest

from geometry import (Point, Vector, Line, LEFT, RIGHT, COLLINEAR, INSIDE, OUTSIDE,
                      BOUNDARY, dot, cross, distance, distance_squared, orientation,
                      orientation_float, orientation_exact, left_turn, right_turn,
                      collinear, point_on_segment, segment_intersection,
                      segments_intersect, proper_intersection, line_intersection,
                      distance_point_to_line, distance_point_to_segment,
                      triangle_area, point_in_triangle, circumcenter, circumcircle,
                      in_circle, polygon_signed_area, make_ccw, point_in_polygon,
                      is_simple_polygon)

P = Point


# ---------------------------------------------------------------- Part 1
def test_vector_basics():
    u, v = Vector(1, 2), Vector(3, -1)
    assert dot(u, v) == 1
    assert cross(u, v) == -7
    assert cross(v, u) == 7
    assert P(1, 1) - P(0, 0) == Vector(1, 1)
    assert P(0, 0) + Vector(2, 3) == P(2, 3)


def test_distances():
    assert distance(P(0, 0), P(3, 4)) == 5
    assert distance_squared(P(0, 0), P(3, 4)) == 25


def test_orientation_clear_cases():
    assert orientation(P(0, 0), P(1, 0), P(0, 1)) == LEFT
    assert orientation(P(0, 0), P(1, 0), P(0, -1)) == RIGHT
    assert orientation(P(0, 0), P(1, 1), P(2, 2)) == COLLINEAR
    assert left_turn(P(0, 0), P(1, 0), P(1, 1))
    assert right_turn(P(0, 0), P(1, 0), P(1, -1))
    assert collinear(P(0, 0), P(0, 5), P(0, -3))


def test_orientation_duplicate_points():
    assert orientation(P(1, 1), P(1, 1), P(5, 7)) == COLLINEAR
    assert orientation(P(1, 1), P(1, 1), P(1, 1)) == COLLINEAR


def test_orientation_large_coordinates():
    big = 1e15
    assert orientation(P(big, big), P(big + 1, big), P(big, big + 1)) == LEFT
    assert orientation(P(-big, 0), P(big, 0), P(0, 1e-3)) == LEFT
    assert orientation(P(-big, 0), P(big, 0), P(0, -1e-3)) == RIGHT


def test_orientation_nearly_collinear_is_exact():
    """
    Classic failure case (Kettner et al., 'Classroom examples of robustness
    problems'): points near the line y = x.  The naive float predicate gives
    wrong / inconsistent signs; our filtered predicate must agree with exact
    rational arithmetic every time.
    """
    p, q = P(12.0, 12.0), P(24.0, 24.0)
    disagreements_naive = 0
    for i in range(64):
        for j in range(64):
            r = P(0.5 + i * 2 ** -53, 0.5 + j * 2 ** -53)
            exact = orientation_exact(p, q, r)
            assert orientation(p, q, r) == exact
            disagreements_naive += orientation_float(p, q, r) != exact
    assert disagreements_naive > 0          # shows the naive predicate DOES fail


def test_orientation_random_agrees_with_exact():
    rng = random.Random(1)
    for _ in range(2000):
        pts = [P(rng.uniform(-1e3, 1e3), rng.uniform(-1e3, 1e3)) for _ in range(3)]
        assert orientation(*pts) == orientation_exact(*pts)


# ---------------------------------------------------------------- Part 2
def test_point_on_segment():
    a, b = P(0, 0), P(4, 4)
    assert point_on_segment(P(2, 2), a, b)
    assert point_on_segment(a, a, b) and point_on_segment(b, a, b)   # endpoints
    assert not point_on_segment(P(5, 5), a, b)                        # collinear, outside
    assert not point_on_segment(P(2, 2.0001), a, b)


def test_segment_proper_crossing():
    r = segment_intersection(P(0, 0), P(2, 2), P(0, 2), P(2, 0))
    assert r == ("point", P(1, 1))
    assert proper_intersection(P(0, 0), P(2, 2), P(0, 2), P(2, 0))


def test_segment_shared_endpoint():
    r = segment_intersection(P(0, 0), P(1, 1), P(1, 1), P(2, 0))
    assert r == ("point", P(1, 1))
    assert not proper_intersection(P(0, 0), P(1, 1), P(1, 1), P(2, 0))


def test_segment_t_junction():
    r = segment_intersection(P(0, 0), P(4, 0), P(2, 0), P(2, 3))
    assert r == ("point", P(2, 0))
    assert not proper_intersection(P(0, 0), P(4, 0), P(2, 0), P(2, 3))


def test_segment_collinear_overlap():
    r = segment_intersection(P(0, 0), P(4, 0), P(2, 0), P(6, 0))
    assert r[0] == "overlap"
    assert {r[1].a, r[1].b} == {P(2, 0), P(4, 0)}
    # one segment contains the other, vertical case
    r = segment_intersection(P(0, 0), P(0, 10), P(0, 7), P(0, 3))
    assert r[0] == "overlap" and {r[1].a, r[1].b} == {P(0, 3), P(0, 7)}


def test_segment_collinear_touching_and_disjoint():
    assert segment_intersection(P(0, 0), P(2, 0), P(2, 0), P(5, 0)) == ("point", P(2, 0))
    assert segment_intersection(P(0, 0), P(2, 0), P(3, 0), P(5, 0)) == ("none",)


def test_segment_parallel_disjoint():
    assert segment_intersection(P(0, 0), P(4, 0), P(0, 1), P(4, 1)) == ("none",)
    assert not segments_intersect(P(0, 0), P(4, 0), P(0, 1), P(4, 1))


def test_segment_zero_length():
    assert segment_intersection(P(1, 1), P(1, 1), P(0, 0), P(2, 2)) == ("point", P(1, 1))
    assert segment_intersection(P(1, 1), P(1, 1), P(0, 1.5), P(2, 1.5)) == ("none",)
    assert segment_intersection(P(1, 1), P(1, 1), P(1, 1), P(1, 1)) == ("point", P(1, 1))
    assert distance_point_to_segment(P(3, 4), P(0, 0), P(0, 0)) == 5


def test_line_intersection():
    assert line_intersection(Line(P(0, 0), P(1, 1)), Line(P(0, 2), P(2, 0))) == P(1, 1)
    assert line_intersection(Line(P(0, 0), P(1, 0)), Line(P(0, 1), P(5, 1))) is None   # parallel
    assert line_intersection(Line(P(0, 0), P(1, 0)), Line(P(2, 0), P(5, 0))) is None   # coincident


def test_point_line_segment_distances():
    assert distance_point_to_line(P(0, 5), Line(P(-1, 0), P(1, 0))) == 5
    assert distance_point_to_segment(P(0, 5), P(-1, 0), P(1, 0)) == 5
    assert distance_point_to_segment(P(4, 4), P(-1, 0), P(1, 0)) == 5   # nearest = endpoint


# ---------------------------------------------------------------- Part 3
def test_triangle_area_and_containment():
    a, b, c = P(0, 0), P(4, 0), P(0, 3)
    assert triangle_area(a, b, c) == 6
    assert point_in_triangle(a, b, c, P(1, 1)) == INSIDE
    assert point_in_triangle(a, b, c, P(2, 0)) == BOUNDARY
    assert point_in_triangle(a, b, c, a) == BOUNDARY
    assert point_in_triangle(a, b, c, P(5, 5)) == OUTSIDE
    assert point_in_triangle(c, b, a, P(1, 1)) == INSIDE            # clockwise input
    # degenerate triangle
    assert point_in_triangle(P(0, 0), P(1, 1), P(2, 2), P(1, 1)) == BOUNDARY


def test_circumcircle():
    c, r = circumcircle(P(1, 0), P(0, 1), P(-1, 0))
    assert math.isclose(c.x, 0, abs_tol=1e-12) and math.isclose(c.y, 0, abs_tol=1e-12)
    assert math.isclose(r, 1)
    assert circumcenter(P(0, 0), P(1, 1), P(2, 2)) is None


def test_in_circle():
    a, b, c = P(1, 0), P(0, 1), P(-1, 0)          # counter-clockwise, unit circle
    assert in_circle(a, b, c, P(0, 0)) == 1
    assert in_circle(a, b, c, P(5, 5)) == -1
    assert in_circle(a, b, c, P(0, -1)) == 0       # exactly cocircular
    assert in_circle(P(0, 0), P(1, 0), P(1, 1), P(0, 1)) == 0  # square corners


def test_in_circle_nearly_cocircular():
    a, b, c = P(0, 0), P(1, 0), P(1, 1)
    assert in_circle(a, b, c, P(0, 1 + 2 ** -50)) == -1
    assert in_circle(a, b, c, P(0, 1 - 2 ** -50)) == 1


# ---------------------------------------------------------------- polygons
SQUARE = [P(0, 0), P(4, 0), P(4, 4), P(0, 4)]
U_SHAPE = [P(0, 0), P(6, 0), P(6, 6), P(4, 6), P(4, 2), P(2, 2), P(2, 6), P(0, 6)]


def test_polygon_orientation():
    assert polygon_signed_area(SQUARE) == 16
    assert polygon_signed_area(list(reversed(SQUARE))) == -16
    assert polygon_signed_area(make_ccw(list(reversed(SQUARE)))) == 16


@pytest.mark.parametrize("poly", [SQUARE, list(reversed(SQUARE))])
def test_point_in_polygon_square(poly):
    assert point_in_polygon(P(2, 2), poly) == INSIDE
    assert point_in_polygon(P(5, 2), poly) == OUTSIDE
    assert point_in_polygon(P(4, 2), poly) == BOUNDARY
    assert point_in_polygon(P(4, 4), poly) == BOUNDARY      # vertex
    assert point_in_polygon(P(-1, 4), poly) == OUTSIDE      # ray through vertices


def test_point_in_polygon_concave():
    assert point_in_polygon(P(3, 4), U_SHAPE) == OUTSIDE    # in the notch
    assert point_in_polygon(P(1, 4), U_SHAPE) == INSIDE
    assert point_in_polygon(P(3, 1), U_SHAPE) == INSIDE
    assert point_in_polygon(P(3, 2), U_SHAPE) == BOUNDARY
    assert point_in_polygon(P(-1, 2), U_SHAPE) == OUTSIDE   # ray along an edge's line


def test_is_simple_polygon():
    assert is_simple_polygon(SQUARE)
    assert is_simple_polygon(U_SHAPE)
    bowtie = [P(0, 0), P(2, 2), P(2, 0), P(0, 2)]
    assert not is_simple_polygon(bowtie)
    assert not is_simple_polygon([P(0, 0), P(1, 1), P(2, 2)])   # collinear
