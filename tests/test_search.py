"""A* / Dijkstra correctness, compared with brute force on random graphs."""
import itertools
import math
import random

from geometry import Point
from search import Graph, astar, dijkstra, dijkstra_all, euclidean_heuristic


def small_graph():
    g = Graph()
    for n, (x, y) in {"a": (0, 0), "b": (1, 0), "c": (2, 0), "d": (1, 5)}.items():
        g.add_node(n, Point(x, y))
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("a", "d")
    g.add_edge("d", "c")
    return g


def test_simple_path():
    g = small_graph()
    r = dijkstra(g, "a", "c")
    assert r.path == ["a", "b", "c"] and r.cost == 2
    assert astar(g, "a", "c", euclidean_heuristic(g, "c")).cost == 2


def test_unreachable():
    g = small_graph()
    g.add_node("z", Point(9, 9))
    assert dijkstra(g, "a", "z").path is None
    assert math.isinf(dijkstra(g, "a", "z").cost)


def brute_force_shortest(g, s, t):
    """Try every simple path (only for tiny graphs)."""
    best = math.inf
    others = [n for n in g.pos if n not in (s, t)]
    for k in range(len(others) + 1):
        for mid in itertools.permutations(others, k):
            path = (s,) + mid + (t,)
            if all(v in g.adj[u] for u, v in zip(path, path[1:])):
                best = min(best, sum(g.adj[u][v] for u, v in zip(path, path[1:])))
    return best


def test_random_graphs_match_brute_force():
    rng = random.Random(7)
    for trial in range(40):
        g = Graph()
        n = 7
        for i in range(n):
            g.add_node(i, Point(rng.uniform(0, 10), rng.uniform(0, 10)))
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.4:
                    g.add_edge(i, j)          # Euclidean weight -> heuristic admissible
        expected = brute_force_shortest(g, 0, n - 1)
        d = dijkstra(g, 0, n - 1)
        a = astar(g, 0, n - 1, euclidean_heuristic(g, n - 1))
        if math.isinf(expected):
            assert d.path is None and a.path is None
        else:
            assert math.isclose(d.cost, expected) and math.isclose(a.cost, expected)
            assert a.expanded <= d.expanded
            assert math.isclose(dijkstra_all(g, 0)[n - 1], expected)
