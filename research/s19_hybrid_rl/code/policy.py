"""S19 candidate-conditioned, permutation-equivariant policy.
  shared MLP encodes [candidate features ; global context] -> one logit per candidate slot (masked);
  WAIT logit from the global context alone;
  value head from global context + masked mean/max pooled candidate encodings.
Permuting the candidate slots permutes the candidate logits identically and leaves the WAIT logit and the value unchanged
(verified numerically by check_equivariance())."""
import os
import sys
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from strategic_env import CAND_DIM, GLOBAL_DIM, K  # noqa: E402

NEG = -1e9


def mlp(i, h, o, n=2):
    layers, d = [], i
    for _ in range(n):
        layers += [nn.Linear(d, h), nn.Tanh()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class CandidatePolicy(nn.Module):
    def __init__(self, gdim: int = GLOBAL_DIM, cdim: int = CAND_DIM, hidden: int = 64):
        super().__init__()
        self.gdim, self.cdim, self.hidden = gdim, cdim, hidden
        self.g_enc = mlp(gdim, hidden, hidden, 1)
        self.c_enc = mlp(cdim + hidden, hidden, hidden, 2)
        self.c_logit = nn.Linear(hidden, 1)
        self.wait_logit = mlp(hidden, hidden, 1, 1)
        self.value = mlp(3 * hidden, hidden, 1, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.c_logit.weight, gain=0.01)
        nn.init.orthogonal_(self.wait_logit[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.value[-1].weight, gain=1.0)

    def forward(self, g: torch.Tensor, c: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """g (B,G), c (B,K,C), mask (B,K+1) bool -> masked logits (B,K+1), value (B,)"""
        B, Kc, _ = c.shape
        hg = torch.tanh(self.g_enc(g))                                   # (B,H)
        hc = torch.tanh(self.c_enc(torch.cat([c, hg.unsqueeze(1).expand(B, Kc, self.hidden)], -1)))  # (B,K,H)
        lc = self.c_logit(hc).squeeze(-1)                                # (B,K)
        lw = self.wait_logit(hg)                                         # (B,1)
        logits = torch.cat([lc, lw], -1).masked_fill(~mask, NEG)
        cm = mask[:, :Kc].float().unsqueeze(-1)                          # (B,K,1) pooled over LEGAL candidates only
        n = cm.sum(1).clamp(min=1.0)
        mean = (hc * cm).sum(1) / n
        mx = torch.where(cm.bool(), hc, torch.full_like(hc, -1e4)).max(1).values
        mx = torch.where(cm.sum(1) > 0, mx, torch.zeros_like(mx))
        v = self.value(torch.cat([hg, mean, mx], -1)).squeeze(-1)
        return logits, v

    @torch.no_grad()
    def act(self, obs: Dict[str, np.ndarray], deterministic: bool = False, gen: torch.Generator = None) -> Tuple[int, float, float]:
        g = torch.as_tensor(obs["global"]).unsqueeze(0)
        c = torch.as_tensor(obs["cand"]).unsqueeze(0)
        m = torch.as_tensor(obs["mask"]).unsqueeze(0)
        logits, v = self.forward(g, c, m)
        logp = torch.log_softmax(logits, -1)
        if deterministic:
            a = int(torch.argmax(logits, -1).item())
        else:
            a = int(torch.multinomial(logp.exp(), 1, generator=gen).item())
        return a, float(logp[0, a].item()), float(v.item())


@torch.no_grad()
def check_equivariance(seed: int = 0, trials: int = 20, B: int = 4) -> Dict[str, float]:
    """numerical check: permuting slots permutes candidate logits identically; WAIT logit and value invariant."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pol = CandidatePolicy()
    worst_logit, worst_wait, worst_value = 0.0, 0.0, 0.0
    for _ in range(trials):
        g = torch.randn(B, GLOBAL_DIM)
        c = torch.randn(B, K, CAND_DIM)
        mask = torch.as_tensor(rng.random((B, K + 1)) < 0.6)
        mask[:, K] = True
        perm = torch.as_tensor(rng.permutation(K))
        lg, v = pol(g, c, mask)
        lg2, v2 = pol(g, c[:, perm], torch.cat([mask[:, :K][:, perm], mask[:, K:]], 1))
        worst_logit = max(worst_logit, float((lg[:, :K][:, perm] - lg2[:, :K]).abs().max()))
        worst_wait = max(worst_wait, float((lg[:, K] - lg2[:, K]).abs().max()))
        worst_value = max(worst_value, float((v - v2).abs().max()))
    return {"max_abs_logit_diff": worst_logit, "max_abs_wait_diff": worst_wait, "max_abs_value_diff": worst_value, "pass": max(worst_logit, worst_wait, worst_value) < 1e-5}


if __name__ == "__main__":
    print(check_equivariance())
