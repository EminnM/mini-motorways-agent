"""S20 compact goal-conditioned spatial policy (one model, trained from scratch; no S8-C weights).
Input: 23 tile channels (v2's 21 + goal house + goal destinations) on the 21x37 grid + 11 globals.
Trunk: the S8-C pattern (3x3 convs, full resolution, translation equivariant) extended with dilated 3x3 layers so the receptive field (27x27)
covers a 14-edge route between frontier and goal: channels (32,64,64,64,64,64), dilations (1,1,2,4,4,1).
Heads: 1x1 conv -> 4 logits per tile in native order (E-add, E-remove, S-add, S-remove) == action indices 1..3108; [mean-pool, max-pool, globals]
-> MLP -> WAIT, 2 upgrade logits (always masked in the synthetic env) and DONE.  Output = 3112 logits in the layout [WAIT, tiles, upg0, upg1, DONE]."""
import torch
import torch.nn as nn

import common as C
from common import L

CHANNELS = (32, 64, 64, 64, 64, 64)
DILATIONS = (1, 1, 2, 4, 4, 1)


class GoalPolicy(nn.Module):
    def __init__(self, n_tile=C.N_TILE_CH, channels=CHANNELS, dilations=DILATIONS, head_hidden=64):
        super().__init__()
        layers, c = [], n_tile
        for out, d in zip(channels, dilations):
            layers += [nn.Conv2d(c, out, 3, padding=d, dilation=d), nn.ReLU()]
            c = out
        self.trunk = nn.Sequential(*layers)
        self.tile_head = nn.Conv2d(c, 4, 1)
        self.global_branch = nn.Sequential(nn.Linear(2 * c + L.NUM_GLOBAL_FEATURES, head_hidden), nn.ReLU())
        self.wait_upg_done_head = nn.Linear(head_hidden, 1 + L.UPGRADE_SLOTS + 1)
        nn.init.normal_(self.tile_head.weight, std=0.01)
        nn.init.zeros_(self.tile_head.bias)
        nn.init.normal_(self.wait_upg_done_head.weight, std=0.01)
        nn.init.zeros_(self.wait_upg_done_head.bias)

    def forward(self, tiles_u8, glob):
        """tiles_u8: (B, N_TILES, 23) uint8 (value*6) or float already scaled; glob: (B, 11) float -> logits (B, 3112)"""
        b = tiles_u8.shape[0]
        x = tiles_u8.float() / C.TILE_SCALE if tiles_u8.dtype == torch.uint8 else tiles_u8
        x = x.reshape(b, L.H, L.W, C.N_TILE_CH).permute(0, 3, 1, 2)
        fmap = self.trunk(x)
        pooled = torch.cat([fmap.mean(dim=(2, 3)), fmap.amax(dim=(2, 3)), glob], dim=1)
        g = self.global_branch(pooled)
        wud = self.wait_upg_done_head(g)                                            # [wait, upg0, upg1, done]
        tile_logits = self.tile_head(fmap).permute(0, 2, 3, 1).reshape(b, L.N_TILES * 4)
        return torch.cat([wud[:, :1], tile_logits, wud[:, 1:]], dim=1)

    @property
    def receptive_field(self):
        return 1 + 2 * sum(DILATIONS)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def masked_logits(logits, mask_bool):
    return logits.masked_fill(~mask_bool, -1e9)


if __name__ == "__main__":
    m = GoalPolicy()
    print("params", m.n_params(), "receptive field", m.receptive_field)
    t = torch.zeros(2, L.N_TILES, C.N_TILE_CH, dtype=torch.uint8)
    print(m(t, torch.zeros(2, 11)).shape)
