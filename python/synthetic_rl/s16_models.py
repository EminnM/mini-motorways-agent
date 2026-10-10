"""S16 diagnostic models (supervised, isolated from the production policy classes).
 A  existing S8-C trunk: 3x3 convs (32,64,64), dilation 1 -> receptive field 7x7; per-tile 1x1 head -> 4 logits per tile (same layout as the policy's tile head)
 B  dilated trunk: 3x3 convs (32,48,48,48) with dilations 1,2,4,8 -> receptive field 31x31 (covers the 21-row map and 31 of 37 columns); comparable parameter count
 C  = B with the six group channels zeroed at the input (group-blind control)
Output: logits over the 3,108 tile actions in native order (index = 1 + tile*4 + dir*2 + kind); WAIT / upgrade logits are not modelled."""
import torch
import torch.nn as nn

from . import layout as L
from . import obs_v2 as V


class TileNet(nn.Module):
    def __init__(self, channels, dilations, blind=False):
        super().__init__()
        self.blind = blind
        c = V.NUM_TILE_FEATURES_V2
        layers = []
        for out, d in zip(channels, dilations):
            layers += [nn.Conv2d(c, out, 3, padding=d, dilation=d), nn.ReLU()]
            c = out
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Conv2d(c, 4, 1)
        self.rf = 1 + 2 * sum(dilations)

    def tiles(self, obs):
        b = obs.shape[0]
        t = obs[:, : L.N_TILES * V.NUM_TILE_FEATURES_V2].reshape(b, L.H, L.W, V.NUM_TILE_FEATURES_V2).permute(0, 3, 1, 2)
        if self.blind:
            t = torch.cat([t[:, : L.NUM_TILE_FEATURES], torch.zeros_like(t[:, L.NUM_TILE_FEATURES:])], dim=1)
        return t

    def forward(self, obs):
        b = obs.shape[0]
        return self.head(self.trunk(self.tiles(obs))).permute(0, 2, 3, 1).reshape(b, L.N_TILES * 4)


def make(name):
    if name == "A":
        return TileNet((32, 64, 64), (1, 1, 1))
    if name == "B":
        return TileNet((32, 48, 48, 48), (1, 2, 4, 8))
    if name == "C":
        return TileNet((32, 48, 48, 48), (1, 2, 4, 8), blind=True)
    raise ValueError(name)


def effective_rf(model, n_probe=8):
    """empirical receptive field: largest Chebyshev distance between an output tile and an input tile with non-zero gradient (random inputs, random weights as-is)."""
    model.eval()
    x = torch.rand(1, V.OBS_DIM_V2, requires_grad=True)
    out = model(x).reshape(L.H, L.W, 4)
    cy, cx = L.H // 2, L.W // 2
    out[cy, cx].sum().backward()
    g = x.grad[0, : L.N_TILES * V.NUM_TILE_FEATURES_V2].reshape(L.H, L.W, V.NUM_TILE_FEATURES_V2).abs().sum(-1)
    nz = (g > 0).nonzero()
    return int(max(max(abs(int(p[0]) - cy), abs(int(p[1]) - cx)) for p in nz) * 2 + 1)
