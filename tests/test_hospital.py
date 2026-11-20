"""Floor plan validity and prioritized multi-patient planning."""
import pytest

from geometry import Point
from hospital import FloorPlan
from prioritized import (Patient, ReservationTable, check_conflicts, discretize,
                         plan_prioritized, space_time_astar)
from search import Graph, astar, dijkstra_all, euclidean_heuristic


@pytest.fixture(scope="module")
def plan():
    return FloorPlan()


@pytest.fixture(scope="module")
def grid(plan):
    return discretize(plan.graph, 1.0)


def test_floor_plan_validates(plan):
    assert plan.validate()
    # every room is reachable from every other room
    rooms = plan.room_names()
    dist = dijkstra_all(plan.graph, rooms[0])
    assert all(r in dist for r in rooms)


def test_route_on_floor_plan(plan):
    g = plan.graph
    r = astar(g, "ER", "Ward C", euclidean_heuristic(g, "Ward C"))
    assert r.path[0] == "ER" and r.path[-1] == "Ward C"
    assert r.cost == pytest.approx(45.0)


def test_discretize_preserves_lengths(plan, grid):
    for a, b in [("ER", "OR"), ("Ward A", "ICU"), ("Lab", "Pharmacy")]:
        assert dijkstra_all(grid, a)[b] == pytest.approx(dijkstra_all(plan.graph, a)[b], abs=2)


def corridor(n, pocket_at=2):
    """A straight single-lane corridor 0-1-2-...-(n-1) with a side pocket."""
    g = Graph()
    for i in range(n):
        g.add_node(i, Point(i, 0))
    for i in range(n - 1):
        g.add_edge(i, i + 1, 1.0)
    g.add_node("pocket", Point(pocket_at, 1))
    g.add_edge(pocket_at, "pocket", 1.0)
    return g


def test_head_on_lower_priority_yields():
    g = corridor(8, pocket_at=3)
    patients = [Patient("hi", 0, 7, 1), Patient("lo", 5, 1, 2)]
    plans = plan_prioritized(g, patients)
    hi, lo = plans
    assert hi.delay == 0                   # absolute priority: no delay for #1
    assert lo.path is not None and lo.delay > 0
    assert "pocket" in lo.path             # it must pull over into the pocket
    assert check_conflicts(plans) == []


def test_prioritized_planning_is_incomplete():
    """Known limitation: 'lo' needs 3 ticks to reach the pocket, 'hi' blocks the
    junction at t=2.  A joint plan EXISTS if 'hi' waits one tick, but
    prioritized planning never delays 'hi', so 'lo' gets no plan."""
    g = corridor(6, pocket_at=2)
    plans = plan_prioritized(g, [Patient("hi", 0, 5, 1), Patient("lo", 4, 1, 2)],
                             max_time=40)
    assert plans[0].delay == 0 and plans[1].path is None


def test_no_conflicts_in_hospital_scenario(grid):
    patients = [Patient("p1", "ER", "OR", 1), Patient("p2", "ICU", "Radiology", 2),
                Patient("p3", "Ward C", "ICU", 3), Patient("p4", "Elevator", "Ward A", 4),
                Patient("p5", "Ward B", "Lab", 5)]
    plans = plan_prioritized(grid, patients)
    assert all(pl.path is not None for pl in plans)
    assert plans[0].delay == 0
    assert check_conflicts(plans) == []
    for pl in plans:                        # every step is a move along an edge or a wait
        for u, v in zip(pl.path, pl.path[1:]):
            assert u == v or v in grid.adj[u]


def test_priority_is_absolute(grid):
    """Adding lower-priority patients never changes a higher one's plan."""
    a = [Patient("p1", "ER", "OR", 1)]
    b = a + [Patient("p2", "ICU", "Radiology", 2), Patient("p3", "OR", "ER", 3)]
    assert plan_prioritized(grid, a)[0].path == plan_prioritized(grid, b)[0].path


def test_space_time_astar_waits_when_needed():
    g = corridor(4)
    table = ReservationTable()
    table.reserve([3, 2, 1, 0], agent_id=0)        # someone walks 3 -> 0 then parks at 0
    h = dijkstra_all(g, 3)
    path = space_time_astar(g, "pocket", 3, table, h)
    # must not stand on 2 at t=1, nor 1... it can only leave after the walker passes
    assert path is not None and path[-1] == 3
    for t, node in enumerate(path):
        assert table.occupant(node, t) is None


def test_unreachable_goal_returns_none():
    g = corridor(3)
    table = ReservationTable()
    table.reserve([1], agent_id=0)                  # someone parks at 1 forever
    assert space_time_astar(g, 0, 2, table, dijkstra_all(g, 2), max_time=30) is None
