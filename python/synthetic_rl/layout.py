"""Interface constants and index mappings, DERIVED from the real environment definition.

Sources of truth inspected:
  python/mini_motorways_env.py   NUM_TILE_FEATURES=15, NUM_GLOBAL_FEATURES=11, _ROAD_STATE_ORDINAL, _CONTENT_TYPE_ORDINAL,
                                 encode_observation (tiles row-major: y outer, x inner; then globals)
  game_plugin/.../Entrypoint.cs  ActionIndexToDescriptor / DescriptorToActionIndex: 0=WAIT, then per tile in row-major
                                 (y then x) order, dir in [East, South], kind in [Add, Remove]; upgrade slots after the tiles.
                                 Observation bounds == action bounds == RectInt(xMin=-18, yMin=-10, 37 x 21).
Nothing is hard-coded to 11,666 / 3,111: both are computed below and asserted against the documented values in tests."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import mini_motorways_env as _native  # noqa: E402  (imports only constants/encoder; no game process is started)

XMIN, YMIN, W, H = -18, -10, 37, 21                 # native bounds (HELLO / observation bounds)
N_TILES = W * H
NUM_TILE_FEATURES = _native.NUM_TILE_FEATURES        # 15
NUM_GLOBAL_FEATURES = _native.NUM_GLOBAL_FEATURES    # 11
UPGRADE_SLOTS = 2                                    # UpgradeChoiceSlots in Entrypoint.cs
OBS_DIM = N_TILES * NUM_TILE_FEATURES + NUM_GLOBAL_FEATURES
N_ACTIONS = 1 + N_TILES * 4 + UPGRADE_SLOTS
UPGRADE_BASE = 1 + N_TILES * 4

# tile feature indices (documented in mini_motorways_env.py)
F_BUILDABLE, F_WATER, F_MOUNTAIN, F_RAIL = 0, 1, 2, 3
F_ROAD = {"N": 4, "E": 5, "S": 6, "W": 7}
F_LIGHT, F_ROUNDABOUT, F_CONTENT, F_GROUP, F_DEMAND, F_OVERCROWD, F_VEHICLES = 8, 9, 10, 11, 12, 13, 14
ROAD_ACTIVE = _native._ROAD_STATE_ORDINAL["Active"] / 3.0
CONTENT_HOUSE = _native._CONTENT_TYPE_ORDINAL["House"] / 2.0
CONTENT_DEST = _native._CONTENT_TYPE_ORDINAL["Destination"] / 2.0

# global feature indices
G_HOUR, G_DAY, G_WEEK, G_SCORE, G_CONCRETE = 0, 1, 2, 3, 4      # 5..9 bridges/tunnels/motorways/lights/roundabouts, 10 pending upgrades

# directions: native South is y-1 ("South is y-1 on this map", phase11a_runner.py)
DIRS = {"N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0)}
OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}


def tile_index(x: int, y: int) -> int:
    return (y - YMIN) * W + (x - XMIN)


def in_bounds(x: int, y: int) -> bool:
    return XMIN <= x < XMIN + W and YMIN <= y < YMIN + H


def action_index(kind: str, x: int, y: int, d: str) -> int:
    """kind in {'ADD','REMOVE'}, d in {'E','S'} (the only two directions the action space exposes)."""
    return 1 + tile_index(x, y) * 4 + (0 if d == "E" else 1) * 2 + (0 if kind == "ADD" else 1)


def decode_action(a: int):
    """-> ('WAIT',) | ('ADD'|'REMOVE', x, y, d) | ('UPGRADE', option)"""
    if a == 0:
        return ("WAIT",)
    if a >= UPGRADE_BASE:
        return ("UPGRADE", a - UPGRADE_BASE)
    r = a - 1
    t, sub = divmod(r, 4)
    y, x = divmod(t, W)
    return ("ADD" if sub % 2 == 0 else "REMOVE", XMIN + x, YMIN + y, "E" if sub // 2 == 0 else "S")


def neighbor(x: int, y: int, d: str):
    dx, dy = DIRS[d]
    return x + dx, y + dy


def edge_key(x: int, y: int, d: str):
    """Canonical undirected edge between (x,y) and its d-neighbour: (x0,y0,'E'|'S') of the E/S representative."""
    if d in ("E", "S"):
        return (x, y, d)
    nx, ny = neighbor(x, y, d)
    return (nx, ny, OPP[d])


def edge_endpoints(e):
    x, y, d = e
    return (x, y), neighbor(x, y, d)
