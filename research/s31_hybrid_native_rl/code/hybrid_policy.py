"""S31 policy: the S1/S8-C spatial extractor on the v2 observation (SpatialExtractorV2, unchanged trunk/tile head/value head) with the pooled head widened from 3 logits (WAIT, upg0, upg1)
to 4 (WAIT, upg0, upg1, PLAN). Initialised from the migrated S8-C v2 checkpoint; the PLAN logit's weights are zero and its bias is set so that the initial policy delegates to planner B
with high probability (the experiment asks for deviations FROM planner B, so training starts at the strong baseline). Not a new architecture: one extra output unit."""
import torch
import torch.nn as nn
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from synthetic_rl import layout as L
from synthetic_rl.models_v2 import SpatialExtractorV2

PLAN_INIT_BIAS = 6.0


class HybridExtractor(SpatialExtractorV2):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.wait_upg_head = nn.Linear(64, 1 + L.UPGRADE_SLOTS + 1)   # WAIT, upg0, upg1, PLAN
        self.latent_dim_pi = L.N_ACTIONS + 1
        nn.init.normal_(self.wait_upg_head.weight, std=0.01)
        nn.init.zeros_(self.wait_upg_head.bias)

    def forward_actor(self, obs):
        fmap, gvec = self._encode(obs)
        wu = self.wait_upg_head(gvec)
        return torch.cat([wu[:, :1], self.tile_logits(fmap), wu[:, 1:]], dim=1)   # [WAIT, tiles..., upg0, upg1, PLAN]

    def forward(self, obs):
        fmap, gvec = self._encode(obs)
        wu = self.wait_upg_head(gvec)
        return torch.cat([wu[:, :1], self.tile_logits(fmap), wu[:, 1:]], dim=1), self.value_head(gvec)

    def tile_logits(self, fmap):
        b = fmap.shape[0]
        return self.tile_head(fmap).permute(0, 2, 3, 1).reshape(b, L.N_TILES * 4)


class HybridMaskablePolicy(MaskableActorCriticPolicy):
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = HybridExtractor()

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        self.action_net = nn.Identity()
        self.value_net = nn.Identity()
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)


def init_from_migrated_v2(model, path):
    """copy every tensor of the migrated S8-C v2 policy; widen wait_upg_head (3->4 outputs): rows 0..2 copied, PLAN row zero weight + PLAN_INIT_BIAS."""
    from sb3_contrib import MaskablePPO
    from synthetic_rl.models_v2 import SpatialMaskablePolicyV2  # noqa: F401
    src = MaskablePPO.load(path, device="cpu").policy.state_dict()
    dst = model.policy.state_dict()
    out = dict(dst)
    report = {"copied": 0, "widened": []}
    for k, v in dst.items():
        if k not in src:
            continue
        if src[k].shape == v.shape:
            out[k] = src[k].clone()
            report["copied"] += 1
        elif k == "mlp_extractor.wait_upg_head.weight":
            w = torch.zeros_like(v)
            w[:3] = src[k]
            out[k] = w
            report["widened"].append(k)
        elif k == "mlp_extractor.wait_upg_head.bias":
            b = torch.zeros_like(v)
            b[:3] = src[k]
            b[3] = PLAN_INIT_BIAS
            out[k] = b
            report["widened"].append(k)
    model.policy.load_state_dict(out, strict=True)
    return report
