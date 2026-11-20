"""
demo_hospital.py -- hospital floor plan: single routes and multi-patient
prioritized planning.

Single route (interactive):
    python demo_hospital.py route
        click a room -> start, click another room -> goal; A* path is drawn and
        compared with Dijkstra (nodes expanded).  Keys: c = clear.
    python demo_hospital.py route --start ER --goal "Ward C" --save route.png

Several patients with absolute priority (animated):
    python demo_hospital.py patients
    python demo_hospital.py patients --patients "ER>OR, ICU>Radiology, Ward A>ICU"
        (comma separated start>goal pairs, listed from HIGHEST priority down)
    python demo_hospital.py patients --save patients.gif     # write an animated GIF
"""

import argparse

import matplotlib
import matplotlib.pyplot as plt

from geometry import distance, lerp
from hospital import FloorPlan, draw_floor_plan
from prioritized import Patient, check_conflicts, discretize, plan_prioritized
from search import astar, dijkstra, euclidean_heuristic

COLORS = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c", "#9467bd", "#8c564b"]
LABELS = ["critical", "urgent", "stable", "routine", "routine", "routine"]

DEFAULT_PATIENTS = [            # highest priority first
    ("ER", "OR"),               # P1 trauma patient rushed to surgery
    ("ICU", "Radiology"),       # P2 head-on with P1 in the top corridor -> WAITS
    ("Ward C", "ICU"),          # P3 its short route is busy -> DETOURS via the loop
    ("Elevator", "Ward A"),     # P4 unaffected
]


# ---------------------------------------------------------------------------
# Single-route mode
# ---------------------------------------------------------------------------

def route_mode(args):
    plan = FloorPlan()
    g = plan.graph
    fig, ax = plt.subplots(figsize=(11, 7))
    state = {"start": args.start, "goal": args.goal}

    def redraw():
        ax.clear()
        draw_floor_plan(ax, plan)
        title = "Hospital routing: click a start room, then a goal room"
        s, t = state["start"], state["goal"]
        if s:
            ax.plot(*g.pos[s], "o", ms=12, color="#2e7d32", zorder=8)
        if s and t:
            a = astar(g, s, t, euclidean_heuristic(g, t))
            d = dijkstra(g, s, t)
            xs = [g.pos[n].x for n in a.path]
            ys = [g.pos[n].y for n in a.path]
            ax.plot(xs, ys, color="#d32f2f", lw=4, zorder=7)
            ax.plot(*g.pos[t], "*", ms=18, color="#c62828", zorder=8)
            title = (f"{s} -> {t}: length {a.cost:.1f} m | "
                     f"A* expanded {a.expanded} nodes, Dijkstra {d.expanded} "
                     f"(same length {d.cost:.1f})")
        ax.set_title(title, fontsize=10)
        fig.canvas.draw_idle()

    def on_click(ev):
        if ev.inaxes != ax:
            return
        rooms = plan.room_names()
        nearest = min(rooms, key=lambda r: (g.pos[r].x - ev.xdata) ** 2 +
                      (g.pos[r].y - ev.ydata) ** 2)
        if state["start"] is None or state["goal"] is not None:
            state["start"], state["goal"] = nearest, None
        else:
            state["goal"] = nearest
        redraw()

    def on_key(ev):
        if ev.key == "c":
            state["start"] = state["goal"] = None
            redraw()

    redraw()
    if args.save:
        fig.savefig(args.save, dpi=130, bbox_inches="tight")
        print("saved", args.save)
        return
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


# ---------------------------------------------------------------------------
# Multi-patient mode
# ---------------------------------------------------------------------------

def parse_patients(spec):
    if not spec:
        return DEFAULT_PATIENTS
    pairs = []
    for item in spec.split(","):
        s, t = (x.strip() for x in item.split(">"))
        pairs.append((s, t))
    return pairs


def make_patients(pairs):
    return [Patient(f"P{i + 1} ({LABELS[i % len(LABELS)]})", s, t, priority=i + 1,
                    color=COLORS[i % len(COLORS)])
            for i, (s, t) in enumerate(pairs)]


def print_report(plans):
    print(f"{'patient':<16}{'route':<26}{'solo':>6}{'actual':>8}{'delay':>7}{'waits':>7}")
    for pl in plans:
        p = pl.patient
        route = f"{p.start} -> {p.goal}"
        if pl.path is None:
            print(f"{p.name:<16}{route:<26}{pl.solo_time:>6}{'FAILED':>8}")
        else:
            print(f"{p.name:<16}{route:<26}{pl.solo_time:>6}{pl.arrival:>8}"
                  f"{pl.delay:>7}{pl.waits:>7}")
    conflicts = check_conflicts(plans)
    print("conflict check:", "OK, no collisions" if not conflicts else conflicts)


def patients_mode(args):
    plan = FloorPlan()
    grid = discretize(plan.graph, step=args.step)
    rooms = set(plan.room_names())
    pairs = parse_patients(args.patients)
    for s, t in pairs:
        for r in (s, t):
            if r not in rooms:
                raise SystemExit(f"unknown room {r!r}; choose from {sorted(rooms)}")
    patients = make_patients(pairs)
    plans = plan_prioritized(grid, patients)
    print_report(plans)

    ok = [pl for pl in plans if pl.path]
    horizon = max(len(pl.path) for pl in ok) - 1 + 3
    SUB = args.substeps                       # animation frames per tick

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.subplots_adjust(bottom=0.16)
    draw_floor_plan(ax, plan, show_graph=False)
    # cell graph, faint
    for u, v, _ in grid.edges():
        ax.plot([grid.pos[u].x, grid.pos[v].x], [grid.pos[u].y, grid.pos[v].y],
                color="#d0dae3", lw=0.8, zorder=2)

    dots, labels, trails = [], [], []
    for pl in ok:
        c = pl.patient.color
        xs = [grid.pos[n].x for n in pl.path]
        ys = [grid.pos[n].y for n in pl.path]
        ax.plot(xs, ys, color=c, lw=6, alpha=0.18, zorder=3,
                solid_capstyle="round")                      # planned route
        trail, = ax.plot([], [], color=c, lw=2.5, zorder=4)  # route travelled
        dot, = ax.plot([], [], "o", ms=16, color=c, mec="black", zorder=9)
        lab = ax.text(0, 0, str(pl.patient.priority), color="white",
                      ha="center", va="center", fontsize=9, weight="bold", zorder=10)
        dots.append(dot), labels.append(lab), trails.append(trail)
    clock = ax.set_title("", fontsize=12)
    status = fig.text(0.06, 0.03, "", family="monospace", fontsize=9, va="bottom")

    def position(pl, t, frac):
        path = pl.path
        a = grid.pos[path[min(t, len(path) - 1)]]
        b = grid.pos[path[min(t + 1, len(path) - 1)]]
        return lerp(a, b, frac)

    def update(frame):
        t, frac = divmod(frame, SUB)
        frac /= SUB
        lines = []
        for pl, dot, lab, trail in zip(ok, dots, labels, trails):
            p = position(pl, t, frac)
            dot.set_data([p.x], [p.y])
            lab.set_position((p.x, p.y))
            done = pl.path[:min(t, len(pl.path) - 1) + 1]
            trail.set_data([grid.pos[n].x for n in done] + [p.x],
                           [grid.pos[n].y for n in done] + [p.y])
            if t >= pl.arrival:
                st = f"arrived at t={pl.arrival}"
            elif pl.path[t] == pl.path[t + 1]:
                st = "WAITING (yielding to higher priority)"
            else:
                st = "moving"
            lines.append(f"P{pl.patient.priority} {pl.patient.start:>9} -> "
                         f"{pl.patient.goal:<10} solo {pl.solo_time:>3}  "
                         f"actual {pl.arrival:>3}  delay {pl.delay:>3}  | {st}")
        clock.set_text(f"Prioritized planning, absolute priority   t = {t}")
        status.set_text("\n".join(lines))
        return dots + labels + trails + [clock, status]

    if args.save and not args.save.endswith(".gif"):
        update(args.snapshot * SUB)            # .png etc: a single snapshot
        fig.savefig(args.save, dpi=120, bbox_inches="tight")
        print("saved", args.save)
        return

    from matplotlib.animation import FuncAnimation, PillowWriter
    anim = FuncAnimation(fig, update, frames=horizon * SUB,
                         interval=1000 / (SUB * args.speed), blit=False,
                         repeat=not args.save)
    if args.save:
        anim.save(args.save, writer=PillowWriter(fps=SUB * args.speed))
        print("saved", args.save)
        return
    plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("route", help="single shortest route on the floor plan")
    r.add_argument("--start")
    r.add_argument("--goal")
    r.add_argument("--save")
    m = sub.add_parser("patients", help="animated multi-patient planning")
    m.add_argument("--patients", help='e.g. "ER>OR, ICU>Radiology" (highest priority first)')
    m.add_argument("--step", type=float, default=1.0, help="cell length (m) per tick")
    m.add_argument("--substeps", type=int, default=6, help="animation frames per tick")
    m.add_argument("--speed", type=float, default=2.0, help="ticks per second")
    m.add_argument("--save", help="write .gif animation or .png snapshot and exit")
    m.add_argument("--snapshot", type=int, default=12, help="tick for a .png snapshot")
    args = ap.parse_args()
    if getattr(args, "save", None):
        matplotlib.use("Agg")
    for key in list(plt.rcParams):
        if key.startswith("keymap."):
            plt.rcParams[key] = []
    route_mode(args) if args.mode == "route" else patients_mode(args)


if __name__ == "__main__":
    main()
