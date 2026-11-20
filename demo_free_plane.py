"""
demo_free_plane.py -- interactive shortest paths among polygonal obstacles.

    python demo_free_plane.py            # empty canvas
    python demo_free_plane.py --sample   # start with a sample scene
    python demo_free_plane.py --sample --save free_plane.png   # headless render

Mouse / keyboard
    left click      add a vertex to the obstacle being drawn  (mode O)
                    or place start (mode S) / goal (mode G)
    right click / Enter   close the current polygon
    o / s / g       switch mode: Obstacle / Start / Goal
    u               undo last vertex of the polygon being drawn
    v               show / hide the visibility graph
    t               toggle tangent-edge pruning
    a               toggle A* <-> Dijkstra
    e               show / hide the nodes expanded by the search
    l               load the sample scene
    c               clear everything
"""

import argparse
import time

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

from geometry import Point, is_simple_polygon
from search import astar, euclidean_heuristic
from visibility import Obstacle, build_visibility_graph

SAMPLE_OBSTACLES = [
    [(2, 2), (5, 2), (5, 5), (2, 5)],                              # square
    [(7, 1), (10, 3), (8, 6)],                                     # triangle
    [(3, 7), (9, 7), (9, 8.2), (4.5, 8.2), (4.5, 11), (3, 11)],    # L-shape (non-convex)
    [(11, 6), (13.5, 7.5), (13, 10.5), (10.5, 11), (9.8, 8.8)],    # pentagon
    [(6, 9.5), (8, 9.5), (8, 12.5), (6, 12.5)],                    # square, shares a y with L
]
SAMPLE_START = Point(0.5, 0.5)
SAMPLE_GOAL = Point(13.5, 12.5)


class FreePlaneDemo:
    def __init__(self, ax):
        self.ax = ax
        self.obstacles = []          # list of Obstacle
        self.current = []            # vertices of polygon being drawn
        self.start = None
        self.goal = None
        self.mode = "O"
        self.show_graph = True
        self.tangent_only = False
        self.use_astar = True
        self.show_expanded = False
        self.message = ""
        self.result = None
        self.graph = None
        self.stats = ""

    # ----------------------------------------------------------- events
    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        p = Point(round(event.xdata, 2), round(event.ydata, 2))
        if event.button == 3:                       # right click closes polygon
            self.close_polygon()
        elif self.mode == "O":
            self.current.append(p)
            self.message = f"vertex {len(self.current)} added (right-click/Enter closes)"
        elif self.mode in ("S", "G"):
            if any(o.strictly_contains(p) for o in self.obstacles):
                self.message = "That point is inside an obstacle -- rejected."
            elif self.mode == "S":
                self.start = p
                self.message = "start placed"
            else:
                self.goal = p
                self.message = "goal placed"
        self.recompute()

    def on_key(self, event):
        k = event.key
        if k in ("o", "s", "g"):
            self.mode = k.upper()
            self.message = {"O": "Obstacle mode: click vertices",
                            "S": "Start mode: click to place start",
                            "G": "Goal mode: click to place goal"}[self.mode]
        elif k == "enter":
            self.close_polygon()
        elif k == "u" and self.current:
            self.current.pop()
        elif k == "v":
            self.show_graph = not self.show_graph
        elif k == "t":
            self.tangent_only = not self.tangent_only
        elif k == "a":
            self.use_astar = not self.use_astar
        elif k == "e":
            self.show_expanded = not self.show_expanded
        elif k == "l":
            self.load_sample()
        elif k == "c":
            self.__init__(self.ax)
        self.recompute()

    def close_polygon(self):
        if len(self.current) < 3:
            self.message = "A polygon needs at least 3 vertices."
            return
        if not is_simple_polygon(self.current):
            self.message = "Polygon is not simple (edges cross) -- discarded."
        else:
            self.obstacles.append(Obstacle(self.current))
            self.message = f"obstacle #{len(self.obstacles)} added"
        self.current = []

    def load_sample(self):
        self.obstacles = [Obstacle(v) for v in SAMPLE_OBSTACLES]
        self.start, self.goal = SAMPLE_START, SAMPLE_GOAL
        self.current = []
        self.message = "sample scene loaded"

    # ---------------------------------------------------------- compute
    def recompute(self):
        extra = {}
        if self.start:
            extra["S"] = self.start
        if self.goal:
            extra["G"] = self.goal
        t0 = time.perf_counter()
        self.graph = build_visibility_graph(self.obstacles, extra, self.tangent_only)
        t1 = time.perf_counter()
        self.result = None
        if self.start and self.goal:
            h = euclidean_heuristic(self.graph, "G") if self.use_astar else None
            self.result = astar(self.graph, "S", "G", h)
        t2 = time.perf_counter()
        self.stats = (f"{len(self.graph)} nodes, {self.graph.num_edges()} edges "
                      f"(build {1000 * (t1 - t0):.0f} ms)")
        if self.result:
            algo = "A*" if self.use_astar else "Dijkstra"
            if self.result.path:
                self.stats += (f" | {algo}: length {self.result.cost:.2f}, "
                               f"expanded {self.result.expanded} nodes "
                               f"({1000 * (t2 - t1):.1f} ms)")
            else:
                self.stats += f" | {algo}: goal unreachable"
        self.draw()

    # ------------------------------------------------------------- draw
    def draw(self):
        ax = self.ax
        ax.clear()
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 13)
        ax.set_aspect("equal")
        for o in self.obstacles:
            ax.add_patch(MplPolygon([tuple(v) for v in o.vertices], closed=True,
                                    fc="#b0bec5", ec="#37474f", lw=1.5, zorder=2))
        if self.graph and self.show_graph:
            for u, v, _ in self.graph.edges():
                a, b = self.graph.pos[u], self.graph.pos[v]
                ax.plot([a.x, b.x], [a.y, b.y], color="#90caf9", lw=0.7, zorder=1)
        if self.result and self.show_expanded:
            pts = [self.graph.pos[n] for n in self.result.expanded_order]
            ax.scatter([p.x for p in pts], [p.y for p in pts], s=60,
                       facecolors="none", edgecolors="#ff9800", lw=1.5, zorder=4)
        if self.result and self.result.path:
            pts = [self.graph.pos[n] for n in self.result.path]
            ax.plot([p.x for p in pts], [p.y for p in pts], color="#d32f2f",
                    lw=3, zorder=5)
        if self.current:
            ax.plot([p.x for p in self.current], [p.y for p in self.current],
                    "o-", color="#455a64", zorder=6)
        if self.start:
            ax.plot(*self.start, "o", ms=11, color="#2e7d32", zorder=7)
            ax.annotate("S", tuple(self.start), xytext=(6, 6), textcoords="offset points")
        if self.goal:
            ax.plot(*self.goal, "*", ms=16, color="#c62828", zorder=7)
            ax.annotate("G", tuple(self.goal), xytext=(6, 6), textcoords="offset points")
        flags = (f"mode={self.mode}  algo={'A*' if self.use_astar else 'Dijkstra'}  "
                 f"pruning={'tangent' if self.tangent_only else 'off'}")
        ax.set_title(f"Visibility-graph shortest path    [{flags}]\n{self.stats}",
                     fontsize=9)
        ax.text(0.01, 0.01, self.message, transform=ax.transAxes, fontsize=8,
                color="#555")
        ax.figure.canvas.draw_idle()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", action="store_true", help="load the sample scene")
    ap.add_argument("--save", help="render to this image file and exit (no window)")
    ap.add_argument("--tangent", action="store_true", help="start with tangent pruning on")
    args = ap.parse_args()

    if args.save:
        matplotlib.use("Agg")
    # Free the single-letter keys from matplotlib's default shortcuts.
    for key in list(plt.rcParams):
        if key.startswith("keymap."):
            plt.rcParams[key] = []

    fig, ax = plt.subplots(figsize=(9, 8))
    demo = FreePlaneDemo(ax)
    demo.tangent_only = args.tangent
    if args.sample:
        demo.load_sample()
    demo.recompute()

    if args.save:
        fig.savefig(args.save, dpi=130, bbox_inches="tight")
        print(f"saved {args.save}: {demo.stats}")
        return
    fig.canvas.mpl_connect("button_press_event", demo.on_click)
    fig.canvas.mpl_connect("key_press_event", demo.on_key)
    plt.show()


if __name__ == "__main__":
    main()
