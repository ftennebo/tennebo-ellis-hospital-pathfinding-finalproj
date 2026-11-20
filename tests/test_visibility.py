"""Visibility tests, including all the 'touching' degeneracies."""
import math

from geometry import Point as P
from search import astar, dijkstra, euclidean_heuristic
from visibility import Obstacle, segment_is_free, build_visibility_graph

SQUARE = Obstacle([(2, 2), (4, 2), (4, 4), (2, 4)])
U = Obstacle([(0, 0), (6, 0), (6, 6), (4, 6), (4, 2), (2, 2), (2, 6), (0, 6)])


def test_segment_through_obstacle_blocked():
    assert not segment_is_free(P(0, 3), P(6, 3), [SQUARE])


def test_segment_missing_obstacle_free():
    assert segment_is_free(P(0, 0), P(6, 1), [SQUARE])


def test_segment_along_edge_is_free():
    assert segment_is_free(P(0, 2), P(6, 2), [SQUARE])          # slides along bottom edge
    assert segment_is_free(P(2, 2), P(4, 2), [SQUARE])          # exactly the edge


def test_diagonal_of_obstacle_blocked():
    assert not segment_is_free(P(2, 2), P(4, 4), [SQUARE])


def test_grazing_a_vertex_is_free():
    assert segment_is_free(P(1, 3), P(3, 5), [SQUARE])          # touches only corner (2,4)


def test_through_two_boundary_points_and_interior_blocked():
    # y = x meets this triangle at the edge point (2.5, 2.5) and the vertex (4, 4);
    # the piece between them is inside, so the segment is blocked
    tri = Obstacle([(2, 3), (3, 2), (4, 4)])
    assert not segment_is_free(P(0, 0), P(6, 6), [tri])


def test_through_vertex_into_interior_blocked():
    # enters through corner (2,2) and continues inside
    assert not segment_is_free(P(1, 1), P(3, 3), [SQUARE])


def test_concave_notch():
    # across the opening of the U: outside the polygon -> visible
    assert segment_is_free(P(2, 6), P(4, 6), [U])
    # between the two arms' outer corners through the arms' interior
    # along both arm tops and across the opening: touches only the boundary
    assert segment_is_free(P(0, 6), P(6, 6), [U])
    assert not segment_is_free(P(1, 3), P(5, 3), [U])


def test_shortest_path_around_square():
    obs = [SQUARE]
    g = build_visibility_graph(obs, {"S": P(0, 3), "G": P(6, 3)})
    r = astar(g, "S", "G", euclidean_heuristic(g, "G"))
    expected = 2 * math.hypot(2, 1) + 2          # S -> corner -> corner -> G
    assert math.isclose(r.cost, expected)


def test_astar_equals_dijkstra_and_tangent_pruning_keeps_optimum():
    obs = [Obstacle([(2, 2), (5, 2), (5, 5), (2, 5)]),
           Obstacle([(7, 1), (10, 3), (8, 6)]),
           Obstacle([(3, 7), (9, 7), (9, 8.2), (4.5, 8.2), (4.5, 11), (3, 11)])]
    pts = {"S": P(0.5, 0.5), "G": P(10, 10)}
    full = build_visibility_graph(obs, pts)
    pruned = build_visibility_graph(obs, pts, tangent_only=True)
    a = astar(full, "S", "G", euclidean_heuristic(full, "G"))
    d = dijkstra(full, "S", "G")
    p = astar(pruned, "S", "G", euclidean_heuristic(pruned, "G"))
    assert math.isclose(a.cost, d.cost) and math.isclose(a.cost, p.cost)
    assert a.expanded <= d.expanded
    assert pruned.num_edges() < full.num_edges()


def test_path_never_enters_obstacles():
    obs = [Obstacle([(2, 2), (5, 2), (5, 5), (2, 5)]), U]
    g = build_visibility_graph(obs, {"S": P(-1, -1), "G": P(3, 5)})
    r = astar(g, "S", "G", euclidean_heuristic(g, "G"))
    assert r.path is not None
    for u, v in zip(r.path, r.path[1:]):
        assert segment_is_free(g.pos[u], g.pos[v], obs)


def test_goal_unreachable():
    box = Obstacle([(0, 0), (10, 0), (10, 10), (0, 10)])
    g = build_visibility_graph([box], {"S": P(-1, 5), "G": P(5, 5)})
    # G is strictly inside the obstacle -> no free segment reaches it
    assert dijkstra(g, "S", "G").path is None
