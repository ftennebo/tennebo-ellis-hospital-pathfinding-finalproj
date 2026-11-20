"""
hospital.py -- a sample hospital floor plan modelled as a graph.

The floor plan has two kinds of geometry:
  * WALLS: line segments (room outlines with gaps for doors).
  * A ROUTING GRAPH: named waypoints (rooms, doors, corridor junctions) joined
    by corridor edges.

The geometry kernel is used to VALIDATE the graph: no corridor edge may
cross a wall (segment-intersection test), and each room node must lie inside
its room (point-in-polygon).  So the graph is guaranteed to be physically
walkable, and the same Dijkstra / A* from search.py finds routes on it.

Layout (40 x 24 units, think metres):

    +--------+--------+----+--------+--------+
    |   ER   |Radiol. |Elev|  ICU   |   OR   |   rooms, doors face down
    +---D----+---D----+-D--+---D----+---D----+
    |            top corridor  (y = 16)      |
    +-+  +---------------+ +--------------+  +
    | |  |  Central      | |  Surgical    |  |   three vertical connectors
    | |  |  Supply       | |  Prep        |  |   form two loops -> detours
    +-+  +---------------+ +--------------+  +
    |           bottom corridor (y = 6)      |
    +---D----+---D----+-D--+---D----+---D----+
    | Ward A | Ward B |Phar|  Lab   | Ward C |
    +--------+--------+----+--------+--------+
"""

from __future__ import annotations

from geometry import Point, Segment, segment_intersection, point_in_polygon, INSIDE
from search import Graph

DOOR_WIDTH = 1.4

# name: (x0, y0, x1, y1, door_x, door_side)   door_side 'bottom' or 'top'
ROOMS = {
    "ER":          (2, 17, 10, 23, 6, "bottom"),
    "Radiology":   (10, 17, 18, 23, 14, "bottom"),
    "Elevator":    (18, 17, 22, 23, 20, "bottom"),
    "ICU":         (22, 17, 30, 23, 26, "bottom"),
    "OR":          (30, 17, 38, 23, 34, "bottom"),
    "Ward A":      (2, 0, 10, 5, 6, "top"),
    "Ward B":      (10, 0, 18, 5, 14, "top"),
    "Pharmacy":    (18, 0, 22, 5, 20, "top"),
    "Lab":         (22, 0, 30, 5, 26, "top"),
    "Ward C":      (30, 0, 38, 5, 34, "top"),
}

# Non-walkable blocks between the two corridors (no doors).
BLOCKS = {
    "Central Supply": (4, 7, 19, 15),
    "Surgical Prep":  (21, 7, 36, 15),
}

CORRIDOR_TOP_Y = 16
CORRIDOR_BOTTOM_Y = 6
CONNECTOR_X = (3, 20, 37)        # x of the three vertical corridors


def rect_polygon(x0, y0, x1, y1):
    return [Point(x0, y0), Point(x1, y0), Point(x1, y1), Point(x0, y1)]


def _rect_walls(x0, y0, x1, y1, door_x=None, door_side=None):
    """Four walls of a rectangle; the wall on `door_side` gets a gap."""
    walls = []
    bottom = [(Point(x0, y0), Point(x1, y0))]
    top = [(Point(x0, y1), Point(x1, y1))]
    half = DOOR_WIDTH / 2
    if door_side == "bottom":
        bottom = [(Point(x0, y0), Point(door_x - half, y0)),
                  (Point(door_x + half, y0), Point(x1, y0))]
    elif door_side == "top":
        top = [(Point(x0, y1), Point(door_x - half, y1)),
               (Point(door_x + half, y1), Point(x1, y1))]
    walls += bottom + top
    walls += [(Point(x0, y0), Point(x0, y1)), (Point(x1, y0), Point(x1, y1))]
    return [Segment(a, b) for a, b in walls]


class FloorPlan:
    def __init__(self):
        self.rooms = ROOMS
        self.blocks = BLOCKS
        self.walls: list[Segment] = []
        self.graph = Graph()
        self._build()
        self.validate()

    # ---------------------------------------------------------------
    def _build(self):
        g = self.graph
        # Walls --------------------------------------------------------
        for name, (x0, y0, x1, y1, dx, side) in ROOMS.items():
            self.walls += _rect_walls(x0, y0, x1, y1, dx, side)
        for name, (x0, y0, x1, y1) in BLOCKS.items():
            self.walls += _rect_walls(x0, y0, x1, y1)

        # Corridor waypoints: every door x and every connector x, on both
        # corridors.  Consecutive waypoints along a corridor are joined.
        xs = sorted({spec[4] for spec in ROOMS.values()} | set(CONNECTOR_X))
        for y, tag in ((CORRIDOR_TOP_Y, "T"), (CORRIDOR_BOTTOM_Y, "B")):
            prev = None
            for x in xs:
                node = f"{tag}{x}"
                g.add_node(node, Point(x, y))
                if prev:
                    g.add_edge(prev, node)
                prev = node
        # Vertical connectors (with a midpoint node, handy for animation)
        for x in CONNECTOR_X:
            mid = f"V{x}"
            g.add_node(mid, Point(x, (CORRIDOR_TOP_Y + CORRIDOR_BOTTOM_Y) / 2))
            g.add_edge(f"T{x}", mid)
            g.add_edge(mid, f"B{x}")

        # Rooms: room node -- door node -- corridor node
        for name, (x0, y0, x1, y1, dx, side) in ROOMS.items():
            door_y = y0 if side == "bottom" else y1
            inner_y = y0 + 2.5 if side == "bottom" else y1 - 2.5
            corridor = f"T{dx}" if side == "bottom" else f"B{dx}"
            g.add_node(name, Point(dx, inner_y))
            g.add_node(f"{name} door", Point(dx, door_y))
            g.add_edge(name, f"{name} door")
            g.add_edge(f"{name} door", corridor)

    # ---------------------------------------------------------------
    def validate(self):
        """Use the geometry kernel to prove the graph respects the walls."""
        for u, v, _ in self.graph.edges():
            a, b = self.graph.pos[u], self.graph.pos[v]
            for w in self.walls:
                if segment_intersection(a, b, w.a, w.b)[0] != "none":
                    raise ValueError(f"corridor edge {u}-{v} hits a wall at {w}")
        for name, (x0, y0, x1, y1, *_rest) in ROOMS.items():
            if point_in_polygon(self.graph.pos[name], rect_polygon(x0, y0, x1, y1)) != INSIDE:
                raise ValueError(f"room node {name} is not inside its room")
        return True

    def room_names(self):
        return list(ROOMS)


# ---------------------------------------------------------------------------
# Plotting helper (shared by the demos)
# ---------------------------------------------------------------------------

def draw_floor_plan(ax, plan: FloorPlan, show_graph=True, label_rooms=True):
    import matplotlib.patches as patches
    ax.add_patch(patches.Rectangle((0, 0), 40, 24, color="#eef2f5", zorder=0))
    for name, (x0, y0, x1, y1, *_r) in plan.rooms.items():
        ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                       color="#ffffff", zorder=1))
        if label_rooms:
            ty = y1 - 1.2 if _r[1] == "bottom" else y0 + 1.0
            ax.text((x0 + x1) / 2, ty, name, ha="center", va="center",
                    fontsize=9, color="#333", zorder=5)
    for name, (x0, y0, x1, y1) in plan.blocks.items():
        ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                       color="#c9d1d9", zorder=1))
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, name, ha="center", va="center",
                fontsize=9, color="#555", style="italic", zorder=5)
    for w in plan.walls:
        ax.plot([w.a.x, w.b.x], [w.a.y, w.b.y], color="#2b2b2b", lw=2, zorder=3)
    if show_graph:
        g = plan.graph
        for u, v, _ in g.edges():
            ax.plot([g.pos[u].x, g.pos[v].x], [g.pos[u].y, g.pos[v].y],
                    color="#9db4c8", lw=1, ls="--", zorder=2)
        xs = [p.x for p in g.pos.values()]
        ys = [p.y for p in g.pos.values()]
        ax.scatter(xs, ys, s=12, color="#6b8aa6", zorder=4)
    ax.set_xlim(-0.5, 40.5)
    ax.set_ylim(-0.5, 24.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
