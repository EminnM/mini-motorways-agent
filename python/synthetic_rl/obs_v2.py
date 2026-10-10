"""S14 observation encoding v2 = the frozen v1 layout (15 tile channels + 11 globals, python/mini_motorways_env.encode_observation, UNCHANGED) followed by K_GROUPS extra per-tile one-hot
channels carrying the colour-group identity of house / destination tiles. No existing channel is modified or repurposed.

  tile layout v2 (21 floats):   [v1 channels 0..14] + [group slot 0 .. 5]
  obs_v2 = 777 x 21 + 11 = 16,328 floats

Group SLOTS, not native group indices: the native integer groupIndex is an arbitrary label (native order of first appearance is e.g. 0,1,3,2). Slot = rank of the group's FIRST APPEARANCE
among the structures visible in the current episode (first group seen -> slot 0, next new group -> slot 1, ...). Properties: invariant under any consistent relabeling of the native ids; stable within an
episode (a slot is never re-assigned); new groups take the next free slot; uses only what is visible now or was visible earlier in the episode (a human remembers which colours appeared first);
nothing about future spawns. K_GROUPS = 6 = MotorwaysThemeDatabase colour-group capacity (GenerateDemandProcess._allocatedColorGroups = new bool[6]).
Tie-break when several NEW groups appear in the same observation: row-major order of their first tile (deterministic, label-free)."""
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mini_motorways_env as _native  # noqa: E402  (frozen v1 encoder; only called, never modified)
from . import layout as L  # noqa: E402

K_GROUPS = 6
NUM_TILE_FEATURES_V2 = L.NUM_TILE_FEATURES + K_GROUPS          # 21
OBS_DIM_V2 = L.N_TILES * NUM_TILE_FEATURES_V2 + L.NUM_GLOBAL_FEATURES   # 16,328
F_GROUP_SLOT0 = L.NUM_TILE_FEATURES                              # first of the K one-hot channels
VERSION = "obs_v2.1"


class GroupSlotTracker:
    """Per-episode map native groupIndex -> slot, in order of first appearance. reset() at every episode start."""

    def __init__(self):
        self.slots = {}

    def reset(self):
        self.slots = {}

    def assign(self, groups_in_row_major_order):
        """groups: iterable of native group labels in row-major tile order (None ignored). Returns the slot map."""
        for g in groups_in_row_major_order:
            if g is None or g in self.slots:
                continue
            if len(self.slots) >= K_GROUPS:
                continue                                         # beyond capacity: tile keeps has-group=1 (v1) but no slot; counted by the caller if it ever happens
            self.slots[g] = len(self.slots)
        return self.slots


def slot_grid_from_labels(labels, tracker: GroupSlotTracker):
    """labels: array (N_TILES,) of native group labels in row-major order (object array; None for no group). Returns (N_TILES, K) one-hot float32."""
    tracker.assign(labels)
    out = np.zeros((L.N_TILES, K_GROUPS), dtype=np.float32)
    for i, g in enumerate(labels):
        if g is not None and g in tracker.slots:
            out[i, tracker.slots[g]] = 1.0
    return out


def extend_v1(obs_v1: np.ndarray, slot_onehot: np.ndarray) -> np.ndarray:
    """Pure layout operation: v1 vector + (N_TILES, K) one-hot -> v2 vector. The v1 channels are copied bit-for-bit."""
    tiles = obs_v1[: L.N_TILES * L.NUM_TILE_FEATURES].reshape(L.N_TILES, L.NUM_TILE_FEATURES)
    out = np.empty(OBS_DIM_V2, dtype=np.float32)
    out[: L.N_TILES * NUM_TILE_FEATURES_V2] = np.concatenate([tiles, slot_onehot.astype(np.float32)], axis=1).reshape(-1)
    out[L.N_TILES * NUM_TILE_FEATURES_V2:] = obs_v1[L.N_TILES * L.NUM_TILE_FEATURES:]
    return out


def strip_to_v1(obs_v2: np.ndarray) -> np.ndarray:
    tiles = obs_v2[: L.N_TILES * NUM_TILE_FEATURES_V2].reshape(L.N_TILES, NUM_TILE_FEATURES_V2)[:, : L.NUM_TILE_FEATURES]
    return np.concatenate([tiles.reshape(-1), obs_v2[L.N_TILES * NUM_TILE_FEATURES_V2:]]).astype(np.float32)


def encode_native_v2(raw: dict, tracker: GroupSlotTracker) -> np.ndarray:
    """Native PolicyObservation JSON -> v2 vector. v1 part = the frozen encoder; slots from the per-tile groupIndex the plugin already emits (Entrypoint.cs BuildPolicyObservation)."""
    v1 = _native.encode_observation(raw)
    labels = [t.get("groupIndex") for t in raw["tiles"]]          # the plugin iterates row-major (y outer, x inner) == layout.tile_index order
    return extend_v1(v1, slot_grid_from_labels(labels, tracker))


def s7_v1_to_v2(obs_v1: np.ndarray) -> np.ndarray:
    """S7 long-route tasks have exactly one colour group: every tile with the v1 has-group flag is slot 0. Derivable from the v1 vector alone."""
    tiles = obs_v1[: L.N_TILES * L.NUM_TILE_FEATURES].reshape(L.N_TILES, L.NUM_TILE_FEATURES)
    onehot = np.zeros((L.N_TILES, K_GROUPS), dtype=np.float32)
    onehot[:, 0] = (tiles[:, L.F_GROUP] > 0.5).astype(np.float32)
    return extend_v1(obs_v1, onehot)


class V2ModelOnV1Env:
    """Adapter so the frozen S7 evaluators (which hand out v1 observations) can score a v2 model: predict(obs_v1, ...) -> model.predict(v1->v2)."""

    def __init__(self, model, convert=s7_v1_to_v2):
        self.model, self.convert = model, convert

    def predict(self, obs, action_masks=None, deterministic=True):
        return self.model.predict(self.convert(np.asarray(obs, dtype=np.float32)), action_masks=action_masks, deterministic=deterministic)
