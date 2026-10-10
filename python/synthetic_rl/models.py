"""Spatial CNN policy for MaskablePPO on the NATIVE observation/action interface (architecture transfer: the same class
works on the real environment because it only assumes the documented layout: 777 tiles x 15 features row-major, 11
globals, and the action indexing 0=WAIT, 1+tile*4+dir*2+kind, 2 upgrade slots).

  obs[11666] -> tile tensor (B,15,21,37) + globals (B,11)
  trunk: three 3x3 convolutions (receptive field 7x7), full resolution, translation equivariant
  action head: 1x1 conv -> 4 logits per tile in native order (E-add, E-remove, S-add, S-remove) -> flatten == action
               indices 1..3108; WAIT logit from [mean-pool, max-pool, globals]; the two upgrade logits likewise.
  value head: [mean-pool, max-pool, globals] -> MLP -> scalar.
Output logits have the exact layout of the native Discrete(3111) space and are masked by the native legal mask by
MaskablePPO. No privileged information is used; nothing about the task is baked into the weights."""
import torch
import torch.nn as nn
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from . import layout as L


class SpatialExtractor(nn.Module):
    def __init__(self, channels=(32, 64, 64), head_hidden=64):
        super().__init__()
        c = L.NUM_TILE_FEATURES
        layers = []
        for out in channels:
            layers += [nn.Conv2d(c, out, 3, padding=1), nn.ReLU()]
            c = out
        self.trunk = nn.Sequential(*layers)
        self.tile_head = nn.Conv2d(c, 4, 1)                         # per-tile logits: dir*2 + kind
        g = 2 * c + L.NUM_GLOBAL_FEATURES
        self.global_branch = nn.Sequential(nn.Linear(g, head_hidden), nn.ReLU())
        self.wait_upg_head = nn.Linear(head_hidden, 1 + L.UPGRADE_SLOTS)
        self.value_head = nn.Sequential(nn.Linear(head_hidden, head_hidden), nn.ReLU(), nn.Linear(head_hidden, 1))
        self.latent_dim_pi = L.N_ACTIONS                            # latent_pi IS the logit vector (action_net = Identity)
        self.latent_dim_vf = 1                                      # latent_vf IS the value (value_net = Identity)
        nn.init.normal_(self.tile_head.weight, std=0.01)
        nn.init.zeros_(self.tile_head.bias)
        nn.init.normal_(self.wait_upg_head.weight, std=0.01)
        nn.init.zeros_(self.wait_upg_head.bias)

    def _encode(self, obs):
        b = obs.shape[0]
        tiles = obs[:, : L.N_TILES * L.NUM_TILE_FEATURES].reshape(b, L.H, L.W, L.NUM_TILE_FEATURES).permute(0, 3, 1, 2)
        glob = obs[:, L.N_TILES * L.NUM_TILE_FEATURES:]
        fmap = self.trunk(tiles)
        pooled = torch.cat([fmap.mean(dim=(2, 3)), fmap.amax(dim=(2, 3)), glob], dim=1)
        return fmap, self.global_branch(pooled)

    def forward_actor(self, obs):
        fmap, gvec = self._encode(obs)
        b = obs.shape[0]
        tile_logits = self.tile_head(fmap).permute(0, 2, 3, 1).reshape(b, L.N_TILES * 4)   # (B,H,W,4) row-major == native order
        wu = self.wait_upg_head(gvec)                                                      # [wait, upg0, upg1]
        return torch.cat([wu[:, :1], tile_logits, wu[:, 1:]], dim=1)

    def forward_critic(self, obs):
        _, gvec = self._encode(obs)
        return self.value_head(gvec)

    def forward(self, obs):
        fmap, gvec = self._encode(obs)
        b = obs.shape[0]
        tile_logits = self.tile_head(fmap).permute(0, 2, 3, 1).reshape(b, L.N_TILES * 4)
        wu = self.wait_upg_head(gvec)
        return torch.cat([wu[:, :1], tile_logits, wu[:, 1:]], dim=1), self.value_head(gvec)


class SpatialMaskablePolicy(MaskableActorCriticPolicy):
    """MaskablePPO policy with the spatial extractor; plug in via policy_kwargs-free `policy=SpatialMaskablePolicy`."""

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = SpatialExtractor()

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        self.action_net = nn.Identity()
        self.value_net = nn.Identity()
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
