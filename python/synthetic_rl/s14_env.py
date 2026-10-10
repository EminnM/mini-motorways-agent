"""S14: the S13 multi-house environment with the v2 (group-aware) observation. Subclass only: s13_env.py is unchanged and still produces v1 observations.
Slots follow obs_v2.GroupSlotTracker (order of first appearance of a colour group among VISIBLE structures, row-major tie-break) -- the same rule the native v2 encoder applies to the
plugin's per-tile groupIndex, so the two encoders agree on equivalent visible states (checked by s14_tests)."""
import numpy as np
from gymnasium import spaces

from . import layout as L
from . import obs_v2 as V
from .s13_env import MultiHouseEnv


class MultiHouseEnvV2(MultiHouseEnv):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(V.OBS_DIM_V2,), dtype=np.float32)
        self.tracker = V.GroupSlotTracker()
        self._obs_v2 = np.zeros(V.OBS_DIM_V2, dtype=np.float32)

    def tile_group_labels(self):
        """row-major native-style labels: the colour group of every House / Destination tile that carries a model, None elsewhere (carparks and model-less destination tiles have none)."""
        labels = [None] * L.N_TILES
        for p, c in self.content.items():
            if c == "House":
                labels[L.tile_index(*p)] = self.houses[p]["group"]
            elif c == "Destination":
                labels[L.tile_index(*p)] = self.group.get(p)
        return labels

    def _reset_state(self, sc):
        self.tracker.reset()
        super()._reset_state(sc)

    def _build_obs(self):
        super()._build_obs()                                     # v1 vector in self._obs (unchanged semantics)
        self._obs_v2 = V.extend_v1(self._obs, V.slot_grid_from_labels(self.tile_group_labels(), self.tracker))

    def reset(self, *, seed=None, options=None):
        _, info = super().reset(seed=seed, options=options)
        return self._obs_v2.copy(), info

    def step(self, action):
        _, r, te, tr, info = super().step(action)
        return self._obs_v2.copy(), r, te, tr, info

    def to_native_raw(self):
        """Render the current synthetic state as a native-style PolicyObservation dict (the subset of fields the v1/v2 encoders read), for encoder-agreement tests."""
        water = self.water
        tiles = []
        state_name = {"A": "Active", "M": "Mothballed"}
        for y in range(L.YMIN, L.YMIN + L.H):
            for x in range(L.XMIN, L.XMIN + L.W):
                p = (x, y)
                road = {}
                for d, (dx, dy) in L.DIRS.items():
                    e = L.edge_key(x, y, d)
                    road[d] = state_name.get(self.edges.get(e), "None")
                c = self.content.get(p)
                ct = "House" if c == "House" else ("Destination" if c == "Destination" else ("Carpark" if c == "Carpark" else "None"))
                g = self.houses[p]["group"] if c == "House" else (self.group.get(p) if c == "Destination" else None)
                dest = None
                if c == "Destination":
                    dd = next((d for d in self.dests if p in d["tiles"]), None)
                    if dd is not None:
                        dest = {"destinationType": "Destination", "totalDemand": dd["demand"], "isOvercrowding": dd["demand"] >= 7, "overcrowdingTime": 0.0, "isUpgraded": False}
                tiles.append({"coord": [x, y], "buildable": True, "overWater": p in water, "underMountain": False, "overRail": False, "road": road, "hasTrafficLight": False, "hasRoundabout": False,
                              "motorwayDirections": [], "contentType": ct, "groupIndex": g, "destination": dest, "visibleVehicleCount": 0})
        hour = int(1.2 * self.t) + 1
        day = hour // 24
        g = {"hour": hour, "day": min(day, 365), "week": min(day // 7, 52), "score": self.score, "concreteAvailable": self.concrete, "concreteTotal": self.total, "bridgesAvailable": 1, "bridgesTotal": 1,
             "tunnelsAvailable": 0, "tunnelsTotal": 0, "motorwaysAvailable": 0, "motorwaysTotal": 0, "trafficLightsAvailable": 0, "trafficLightsTotal": 0, "roundaboutsAvailable": 0, "roundaboutsTotal": 0, "pendingUpgradeChoices": 0}
        return {"simulationTime": self.t, "bounds": {"xMin": L.XMIN, "yMin": L.YMIN, "width": L.W, "height": L.H}, "tiles": tiles, "globals": g}
