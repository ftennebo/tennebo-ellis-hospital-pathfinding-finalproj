"""
experiments.py -- the experiments for the report.  Writes figures to figures/
and prints tables.

    python experiments.py            # all experiments (~1 minute)
    python experiments.py --quick    # smaller sizes

E1  Robustness: naive float orientation vs our filtered exact predicate on
    nearly collinear points.
E2  Visibility graph: size and construction time vs number of obstacle
    vertices, with and without tangent pruning (expect ~O(n^3) time).
E3  Search: nodes expanded by A* vs Dijkstra on the same graphs.
E4  Prioritized planning: delay per priority rank as the hospital gets busier.
"""

import argparse
import math
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import geometry
from geometry import Point, orientation, orientation_exact, orientation_float
from hospital import FloorPlan
from prioritized import Patient, check_conflicts, discretize, plan_prioritized
from search import astar, dijkstra, euclidean_heuristic
from visibility import Obstacle, build_visibility_graph


# ---------------------------------------------------------------- E1
def e1_robustness(n=256):
    p, q = Point(12.0, 12.0), Point(24.0, 24.0)
    naive = [[0] * n for _ in range(n)]
    exact = [[0] * n for _ in range(n)]
    wrong = 0
    geometry.predicate_stats.update(orientation_calls=0, orientation_exact=0)
    for i in range(n):
        for j in range(n):
            r = Point(0.5 + j * 2 ** -53, 0.5 + i * 2 ** -53)
            naive[i][j] = orientation_float(p, q, r)
            exact[i][j] = orientation(p, q, r)
            assert exact[i][j] == orientation_exact(p, q, r)
            wrong += naive[i][j] != exact[i][j]
    st = geometry.predicate_stats
    print(f"E1  {n}x{n} points within a few ulps of the line y = x")
    print(f"    naive float orientation wrong on {wrong} / {n * n} "
          f"({100 * wrong / n / n:.1f}%)")
    print(f"    filtered predicate: {st['orientation_exact']} of {st['orientation_calls']} "
          f"calls needed the exact fallback here (worst case input)")

    # On random, non-degenerate input the fallback is essentially never used.
    geometry.predicate_stats.update(orientation_calls=0, orientation_exact=0)
    rng = random.Random(0)
    for _ in range(100000):
        orientation(*(Point(rng.random(), rng.random()) for _ in range(3)))
    print(f"    on 100000 random triples: {st['orientation_exact']} exact fallbacks")

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    cmap = matplotlib.colors.ListedColormap(["#1f77b4", "#ffffff", "#d62728"])
    for ax, img, title in ((axs[0], naive, "naive floating point"),
                           (axs[1], exact, "filtered exact (ours)")):
        ax.imshow(img, origin="lower", cmap=cmap, vmin=-1, vmax=1)
        ax.set_title(f"orientation((12,12),(24,24),r): {title}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("r = (0.5 + j*2^-53, 0.5 + i*2^-53).  red = left, blue = right, white = collinear")
    fig.savefig("figures/e1_robustness.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- scenes
def random_scene(n_obstacles, rng, size=100.0):
    """Disjoint random convex polygons: one per cell of a k x k grid."""
    k = math.ceil(math.sqrt(n_obstacles))
    cell = size / k
    cells = rng.sample([(i, j) for i in range(k) for j in range(k)], n_obstacles)
    obstacles = []
    for i, j in cells:
        cx, cy = (i + 0.5) * cell, (j + 0.5) * cell
        r = cell * rng.uniform(0.25, 0.42)
        m = rng.randint(3, 6)
        angles = sorted(rng.uniform(0, 2 * math.pi) for _ in range(m))
        obstacles.append(Obstacle([(cx + r * math.cos(a), cy + r * math.sin(a))
                                   for a in angles]))
    s, g = Point(-1.0, -1.0), Point(size + 1, size + 1)
    return obstacles, s, g


# ---------------------------------------------------------------- E2 + E3
def e2_e3_visibility(sizes, reps=2):
    print("\nE2/E3  visibility graph on random scenes")
    print(f"{'obst':>5}{'n':>6}{'edges':>8}{'tangent':>9}{'build s':>9}"
          f"{'tan s':>8}{'Dijk exp':>10}{'A* exp':>8}")
    rows = []
    rng = random.Random(42)
    for m in sizes:
        acc = [0.0] * 7
        for _ in range(reps):
            obs, s, g = random_scene(m, rng)
            pts = {"S": s, "G": g}
            t0 = time.perf_counter()
            full = build_visibility_graph(obs, pts)
            t1 = time.perf_counter()
            tan = build_visibility_graph(obs, pts, tangent_only=True)
            t2 = time.perf_counter()
            d = dijkstra(full, "S", "G")
            a = astar(full, "S", "G", euclidean_heuristic(full, "G"))
            at = astar(tan, "S", "G", euclidean_heuristic(tan, "G"))
            assert math.isclose(d.cost, a.cost) and math.isclose(a.cost, at.cost)
            for k, v in enumerate((len(full), full.num_edges(), tan.num_edges(),
                                   t1 - t0, t2 - t1, d.expanded, a.expanded)):
                acc[k] += v / reps
        rows.append((m, *acc))
        print(f"{m:>5}{acc[0]:>6.0f}{acc[1]:>8.0f}{acc[2]:>9.0f}{acc[3]:>9.2f}"
              f"{acc[4]:>8.2f}{acc[5]:>10.0f}{acc[6]:>8.0f}")

    ns = [r[1] for r in rows]
    # empirical exponent from the two largest sizes
    if len(rows) >= 2:
        k = math.log(rows[-1][4] / rows[-2][4]) / math.log(ns[-1] / ns[-2])
        print(f"    empirical build-time exponent ~ n^{k:.2f} (worst case n^3; bounding-box rejection makes most pair tests cheap)")

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    axs[0].loglog(ns, [r[4] for r in rows], "o-", label="all edges")
    axs[0].loglog(ns, [r[5] for r in rows], "s-", label="tangent only")
    ref = [rows[0][4] * (n / ns[0]) ** 3 for n in ns]
    axs[0].loglog(ns, ref, "k:", label="n^3 reference")
    axs[0].set_xlabel("nodes n")
    axs[0].set_ylabel("build time (s)")
    axs[0].legend()
    axs[0].set_title("E2 construction time")
    axs[1].plot(ns, [r[2] for r in rows], "o-", label="all visible pairs")
    axs[1].plot(ns, [r[3] for r in rows], "s-", label="tangent edges")
    axs[1].set_xlabel("nodes n")
    axs[1].set_ylabel("edges")
    axs[1].legend()
    axs[1].set_title("E2 graph size")
    axs[2].plot(ns, [r[6] for r in rows], "o-", label="Dijkstra")
    axs[2].plot(ns, [r[7] for r in rows], "s-", label="A* (Euclidean h)")
    axs[2].set_xlabel("nodes n")
    axs[2].set_ylabel("nodes expanded")
    axs[2].legend()
    axs[2].set_title("E3 search effort")
    fig.tight_layout()
    fig.savefig("figures/e2_e3_visibility.png", dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- E4
def e4_prioritized(max_patients=8, trials=30):
    print("\nE4  prioritized planning in the hospital (random room pairs)")
    plan = FloorPlan()
    grid = discretize(plan.graph)
    rooms = plan.room_names()
    rng = random.Random(3)
    delay_by_rank = {r: [] for r in range(1, max_patients + 1)}
    print(f"{'patients':>9}{'success':>9}{'mean delay':>12}{'max delay':>11}{'plan ms':>9}")
    summary = []
    for k in range(1, max_patients + 1):
        succ, delays, times = 0, [], []
        for _ in range(trials):
            starts = rng.sample(rooms, k)
            goals = rng.sample(rooms, k)
            pts = [Patient(f"p{i}", s, t, i + 1)
                   for i, (s, t) in enumerate(zip(starts, goals)) if s != t]
            t0 = time.perf_counter()
            plans = plan_prioritized(grid, pts, max_time=300)
            times.append(time.perf_counter() - t0)
            if all(pl.path for pl in plans):
                succ += 1
                assert check_conflicts(plans) == []
                for pl in plans:
                    delays.append(pl.delay)
                    delay_by_rank[pl.patient.priority].append(pl.delay)
        md = sum(delays) / len(delays) if delays else float("nan")
        summary.append((k, succ / trials, md))
        print(f"{k:>9}{100 * succ / trials:>8.0f}%{md:>12.2f}{max(delays, default=0):>11}"
              f"{1000 * sum(times) / trials:>9.1f}")

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    ranks = [r for r in delay_by_rank if delay_by_rank[r]]
    axs[0].bar(ranks, [sum(delay_by_rank[r]) / len(delay_by_rank[r]) for r in ranks],
               color="#1f77b4")
    axs[0].set_xlabel("priority rank (1 = most urgent)")
    axs[0].set_ylabel("mean delay (ticks)")
    axs[0].set_title("E4 delay is pushed onto low priorities")
    axs[1].plot([s[0] for s in summary], [100 * s[1] for s in summary], "o-")
    axs[1].set_xlabel("number of patients")
    axs[1].set_ylabel("instances fully solved (%)")
    axs[1].set_ylim(0, 105)
    axs[1].set_title("E4 success rate (prioritized planning is incomplete)")
    fig.tight_layout()
    fig.savefig("figures/e4_prioritized.png", dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    import os
    os.makedirs("figures", exist_ok=True)
    e1_robustness(128 if args.quick else 256)
    e2_e3_visibility([4, 9, 16] if args.quick else [4, 9, 16, 25, 36])
    e4_prioritized(6 if args.quick else 8, 10 if args.quick else 30)
    print("\nfigures written to figures/")


if __name__ == "__main__":
    main()
