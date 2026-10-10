"""S17 hybrid: S8-C conv trunk (v2 input, 21 channels) -> ONE residual global self-attention block over the 777 tile positions -> the existing per-tile action head and pooled WAIT/upgrade/value heads.
Insertion point: between the trunk's last ReLU (feature map B x 64 x 21 x 37) and the heads. Tokens = the 64-d feature of each tile + a learned positional embedding (777 x 64); LayerNorm -> 4-head
self-attention (dim 64, torch scaled_dot_product_attention) -> output projection -> added back to the feature map (residual). The output projection is ZERO-initialised, so at init the block
contributes exactly 0 and the hybrid reproduces the migrated S8-C policy bit-for-bit (checked by s17_benchmark.py). `use_attention=False` disables the residual path (functional ablation)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from . import layout as L
from . import obs_v2 as V


class GlobalAttentionBlock(nn.Module):
    def __init__(self, dim=64, heads=4, n_tokens=L.N_TILES):
        super().__init__()
        self.dim, self.heads = dim, heads
        self.pos = nn.Parameter(torch.randn(1, n_tokens, dim) * 0.02)
        self.ln = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, fmap):                                     # fmap: B x C x H x W
        b, c, h, w = fmap.shape
        x = fmap.flatten(2).transpose(1, 2)                      # B x N x C
        t = self.ln(x + self.pos)
        q, k, v = self.qkv(t).chunk(3, dim=-1)
        q, k, v = (z.reshape(b, -1, self.heads, c // self.heads).transpose(1, 2) for z in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v)             # B x heads x N x d
        a = a.transpose(1, 2).reshape(b, -1, c)
        y = x + self.out(a)                                      # residual; out is zero at init
        return y.transpose(1, 2).reshape(b, c, h, w)


class SpatialExtractorV3(nn.Module):
    def __init__(self, n_tile_features=V.NUM_TILE_FEATURES_V2, channels=(32, 64, 64), head_hidden=64, heads=4, use_attention=True):
        super().__init__()
        self.n_tile_features = n_tile_features
        self.use_attention = use_attention
        c = n_tile_features
        layers = []
        for out in channels:
            layers += [nn.Conv2d(c, out, 3, padding=1), nn.ReLU()]
            c = out
        self.trunk = nn.Sequential(*layers)
        self.attn = GlobalAttentionBlock(c, heads)
        self.tile_head = nn.Conv2d(c, 4, 1)
        g = 2 * c + L.NUM_GLOBAL_FEATURES
        self.global_branch = nn.Sequential(nn.Linear(g, head_hidden), nn.ReLU())
        self.wait_upg_head = nn.Linear(head_hidden, 1 + L.UPGRADE_SLOTS)
        self.value_head = nn.Sequential(nn.Linear(head_hidden, head_hidden), nn.ReLU(), nn.Linear(head_hidden, 1))
        self.latent_dim_pi = L.N_ACTIONS
        self.latent_dim_vf = 1
        nn.init.normal_(self.tile_head.weight, std=0.01)
        nn.init.zeros_(self.tile_head.bias)
        nn.init.normal_(self.wait_upg_head.weight, std=0.01)
        nn.init.zeros_(self.wait_upg_head.bias)

    def features(self, obs):
        b = obs.shape[0]
        nf = self.n_tile_features
        tiles = obs[:, : L.N_TILES * nf].reshape(b, L.H, L.W, nf).permute(0, 3, 1, 2)
        fmap = self.trunk(tiles)
        if self.use_attention:
            fmap = self.attn(fmap)
        return fmap

    def _encode(self, obs):
        fmap = self.features(obs)
        glob = obs[:, L.N_TILES * self.n_tile_features:]
        pooled = torch.cat([fmap.mean(dim=(2, 3)), fmap.amax(dim=(2, 3)), glob], dim=1)
        return fmap, self.global_branch(pooled)

    def tile_logits(self, fmap):
        b = fmap.shape[0]
        return self.tile_head(fmap).permute(0, 2, 3, 1).reshape(b, L.N_TILES * 4)

    def forward_actor(self, obs):
        fmap, gvec = self._encode(obs)
        wu = self.wait_upg_head(gvec)
        return torch.cat([wu[:, :1], self.tile_logits(fmap), wu[:, 1:]], dim=1)

    def forward_critic(self, obs):
        _, gvec = self._encode(obs)
        return self.value_head(gvec)

    def forward(self, obs):
        fmap, gvec = self._encode(obs)
        wu = self.wait_upg_head(gvec)
        return torch.cat([wu[:, :1], self.tile_logits(fmap), wu[:, 1:]], dim=1), self.value_head(gvec)


class SpatialMaskablePolicyV3(MaskableActorCriticPolicy):
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = SpatialExtractorV3()

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        self.action_net = nn.Identity()
        self.value_net = nn.Identity()
        nn.init.zeros_(self.mlp_extractor.attn.out.weight)       # SB3's orthogonal init (ortho_init=True) runs in super()._build and would overwrite the zero residual
        nn.init.zeros_(self.mlp_extractor.attn.out.bias)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)


def migrate_v2_to_v3(sd_v2: dict, sd_v3: dict):
    """copy every v2 tensor by name (all exist in v3 with identical shapes); attention tensors keep their init (out projection = 0)."""
    out = dict(sd_v3)
    copied, new = [], []
    for k in sd_v3:
        if k in sd_v2:
            if tuple(sd_v2[k].shape) != tuple(sd_v3[k].shape):
                raise ValueError(f"shape mismatch {k}")
            out[k] = sd_v2[k].clone()
            copied.append(k)
        else:
            new.append(k)
    return out, {"copied": len(copied), "new_attention_tensors": new}


class SupervisedTileNet(nn.Module):
    """S16/S17 supervised head on the policy extractor: logits over the 3,108 tile actions = exactly the policy's per-tile head applied to the (attention-augmented) feature map."""

    def __init__(self, extractor: SpatialExtractorV3, blind=False):
        super().__init__()
        self.ex = extractor
        self.blind = blind

    def forward(self, obs):
        if self.blind:
            b = obs.shape[0]
            t = obs[:, : L.N_TILES * V.NUM_TILE_FEATURES_V2].reshape(b, L.N_TILES, V.NUM_TILE_FEATURES_V2).clone()
            t[:, :, L.NUM_TILE_FEATURES:] = 0.0
            obs = torch.cat([t.reshape(b, -1), obs[:, L.N_TILES * V.NUM_TILE_FEATURES_V2:]], dim=1)
        return self.ex.tile_logits(self.ex.features(obs))
