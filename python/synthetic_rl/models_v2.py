"""S14: the S1 spatial policy with a configurable number of input tile channels (default obs v2: 21) and the S8-C -> v2 checkpoint migration.
Architecture identical to synthetic_rl/models.py except the first convolution's input width; every other tensor has the same name and shape, so pretrained weights copy 1:1.
Migration: conv1 weight[:, :15] <- pretrained, weight[:, 15:] <- 0, bias <- pretrained. With zero weights on the new channels the migrated network computes EXACTLY the old network's
function of the v1 channels (verified numerically by s14_tests.py), so pretrained behaviour is preserved at step 0 and learning on the group channels starts from zero."""
import torch
import torch.nn as nn
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from . import layout as L
from . import obs_v2 as V


class SpatialExtractorV2(nn.Module):
    def __init__(self, n_tile_features=V.NUM_TILE_FEATURES_V2, channels=(32, 64, 64), head_hidden=64):
        super().__init__()
        self.n_tile_features = n_tile_features
        c = n_tile_features
        layers = []
        for out in channels:
            layers += [nn.Conv2d(c, out, 3, padding=1), nn.ReLU()]
            c = out
        self.trunk = nn.Sequential(*layers)
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

    def _encode(self, obs):
        b = obs.shape[0]
        nf = self.n_tile_features
        tiles = obs[:, : L.N_TILES * nf].reshape(b, L.H, L.W, nf).permute(0, 3, 1, 2)
        glob = obs[:, L.N_TILES * nf:]
        fmap = self.trunk(tiles)
        pooled = torch.cat([fmap.mean(dim=(2, 3)), fmap.amax(dim=(2, 3)), glob], dim=1)
        return fmap, self.global_branch(pooled)

    def forward_actor(self, obs):
        fmap, gvec = self._encode(obs)
        b = obs.shape[0]
        tile_logits = self.tile_head(fmap).permute(0, 2, 3, 1).reshape(b, L.N_TILES * 4)
        wu = self.wait_upg_head(gvec)
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


class SpatialMaskablePolicyV2(MaskableActorCriticPolicy):
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = SpatialExtractorV2()

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        self.action_net = nn.Identity()
        self.value_net = nn.Identity()
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)


def migrate_state_dict(sd_src: dict, sd_dst: dict) -> dict:
    """Copy every v1 tensor into the v2 state dict; widen the first conv (mlp_extractor.trunk.0.weight) with zero columns for the new channels. Raises on any other mismatch."""
    out = {}
    report = {"copied": [], "widened": [], "zero_initialised_slice": None}
    if set(sd_src) != set(sd_dst):
        raise ValueError(f"parameter names differ: {sorted(set(sd_src) ^ set(sd_dst))}")
    for k, v_dst in sd_dst.items():
        v_src = sd_src[k]
        if tuple(v_src.shape) == tuple(v_dst.shape):
            out[k] = v_src.clone()
            report["copied"].append(k)
        elif k == "mlp_extractor.trunk.0.weight" and v_src.shape[0] == v_dst.shape[0] and v_src.shape[2:] == v_dst.shape[2:] and v_src.shape[1] < v_dst.shape[1]:
            w = torch.zeros_like(v_dst)
            w[:, : v_src.shape[1]] = v_src
            out[k] = w
            report["widened"].append(k)
            report["zero_initialised_slice"] = [k, int(v_src.shape[1]), int(v_dst.shape[1])]
        else:
            raise ValueError(f"cannot migrate {k}: {tuple(v_src.shape)} -> {tuple(v_dst.shape)}")
    return out, report
