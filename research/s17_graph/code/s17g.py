"""S17-G: isolated graph-neural-network benchmark for colour-aware routing (supervised, S16 frozen dataset, no PPO, no native game).
Writes ONLY under research/s17_graph/. Reads S16 data + helpers read-only (experiments/s16_train.py: load / set_loss / metrics / paired_bootstrap).
   set PYTHONIOENCODING=utf-8 && python research/s17_graph/code/s17g.py
"""
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import s16_train as S16  # noqa: E402  (read-only helpers; importing does not run training)
from synthetic_rl import layout as L  # noqa: E402
from synthetic_rl import obs_v2 as V  # noqa: E402

OUT = os.path.join(ROOT, "research", "s17_graph")
os.makedirs(OUT, exist_ok=True)
SEED, LR, BATCH, EPOCHS = 17, 1e-3, 64, 30
TIME_PER_MODEL_S = 300            # 2 arms x 300 s + evaluation/ablations must fit the 15-min cap
GLOBAL_CAP_S = 15 * 60
HIDDEN, LAYERS = 64, 3
HOLDOUT_SLOT = S16.HOLDOUT_SLOT   # 2
NT, FT, K = L.N_TILES, V.NUM_TILE_FEATURES_V2, V.K_GROUPS
N_ACT = NT * 4

# ----------------------------------------------------------------------------------------------------------------------
# 1. Graph schema (fixed physical skeleton; per-sample features derived from the v2 observation ONLY)
# ----------------------------------------------------------------------------------------------------------------------
# Candidate physical edges = every in-bounds 4-neighbour pair, represented once by its E/S representative (x,y,d) exactly as the
# native action space does: ADD(x,y,'E') joins (x,y)-(x+1,y), ADD(x,y,'S') joins (x,y)-(x,y-1) (South is y-1).
_cu, _cv, _cd, _cact = [], [], [], []
for y in range(L.YMIN, L.YMIN + L.H):
    for x in range(L.XMIN, L.XMIN + L.W):
        for d in ("E", "S"):
            nx, ny = L.neighbor(x, y, d)
            if L.in_bounds(nx, ny):
                _cu.append(L.tile_index(x, y)); _cv.append(L.tile_index(nx, ny)); _cd.append(0 if d == "E" else 1)
                _cact.append(L.action_index("ADD", x, y, d) - 1)        # logit index = native action id - 1
CAND_U, CAND_V, CAND_D, CAND_ACT = (np.array(a, dtype=np.int64) for a in (_cu, _cv, _cd, _cact))
N_CAND = len(CAND_U)                                                 # 36*21 + 37*20 = 1496
# directed physical message edges (both directions of every candidate edge) with the direction of the SENDER relative to the receiver
# E edge: v=(x+1,y) is East of u; u is West of v.  S edge: v=(x,y-1) is South of u; u is North of v.   dir index: N=0,E=1,S=2,W=3
DIR_SRC = np.concatenate([CAND_V, CAND_U]); DIR_DST = np.concatenate([CAND_U, CAND_V])
DIR_DIR = np.concatenate([np.where(CAND_D == 0, 1, 2), np.where(CAND_D == 0, 3, 0)])
DIR_CAND = np.concatenate([np.arange(N_CAND), np.arange(N_CAND)])    # which candidate edge a directed message edge belongs to
_xs = np.arange(NT) % L.W; _ys = np.arange(NT) // L.W
COORD = np.stack([_xs / (L.W - 1) * 2 - 1, _ys / (L.H - 1) * 2 - 1], 1).astype(np.float32)

NODE_FEAT_NAMES = ["x_norm", "y_norm", "buildable", "water", "mountain", "rail", "road_N", "road_E", "road_S", "road_W", "house", "destination", "has_group", "demand", "overcrowding", "vehicles"] + [f"slot{k}" for k in range(K)]
EDGE_FEAT_NAMES = ["active_road", "mothballed_road", "potential_construction", "both_non_water"]
F_NODE, F_EDGE = len(NODE_FEAT_NAMES), len(EDGE_FEAT_NAMES)


def graph_from_obs(obs_vec):
    """v2 observation vector (16328,) -> graph dict. Deterministic; uses no label / meta / oracle information (signature takes nothing else)."""
    tiles = np.asarray(obs_vec, dtype=np.float32)[: NT * FT].reshape(NT, FT)
    water, mountain = tiles[:, L.F_WATER] > 0.5, tiles[:, L.F_MOUNTAIN] > 0.5
    content = tiles[:, L.F_CONTENT]
    house, dest = np.abs(content - L.CONTENT_HOUSE) < 0.1, content > 0.9
    slots = tiles[:, V.F_GROUP_SLOT0: V.F_GROUP_SLOT0 + K] > 0.5
    node = np.concatenate([COORD, tiles[:, [L.F_BUILDABLE, L.F_WATER, L.F_MOUNTAIN, L.F_RAIL, L.F_ROAD["N"], L.F_ROAD["E"], L.F_ROAD["S"], L.F_ROAD["W"]]],
                          house[:, None].astype(np.float32), dest[:, None].astype(np.float32), tiles[:, [L.F_GROUP, L.F_DEMAND, L.F_OVERCROWD, L.F_VEHICLES]],
                          slots.astype(np.float32)], axis=1)
    # physical edge features: road state along the candidate edge (max of both endpoint channels), buildability
    road_u = np.where(CAND_D == 0, tiles[CAND_U, L.F_ROAD["E"]], tiles[CAND_U, L.F_ROAD["S"]])
    road_v = np.where(CAND_D == 0, tiles[CAND_V, L.F_ROAD["W"]], tiles[CAND_V, L.F_ROAD["N"]])
    road = np.maximum(road_u, road_v)
    active, moth = (road > 0.5) & (road < 0.9), road > 0.9
    nonwater = ~water[CAND_U] & ~water[CAND_V]
    potential = nonwater & ~mountain[CAND_U] & ~mountain[CAND_V] & (road < 0.01)
    edge = np.stack([active, moth, potential, nonwater], 1).astype(np.float32)
    # semantic edges: every VISIBLE house tile <-> every VISIBLE destination tile of the same slot (type flag separates them from physical edges)
    ss, dd = [], []
    for k in range(K):
        hs, ds = np.nonzero(house & slots[:, k])[0], np.nonzero(dest & slots[:, k])[0]
        if len(hs) and len(ds):
            a, b = np.meshgrid(hs, ds, indexing="ij")
            ss.append(a.ravel()); dd.append(b.ravel())
    sem = (np.concatenate(ss), np.concatenate(dd)) if ss else (np.zeros(0, np.int64), np.zeros(0, np.int64))
    return {"node": node, "edge": edge, "gate": nonwater.astype(np.float32), "sem_house": sem[0].astype(np.int64), "sem_dest": sem[1].astype(np.int64)}


class GraphSet:
    """All graphs of a split, stored as stacked tensors + CSR-style semantic edge lists."""

    def __init__(self, obs):
        t0 = time.time()
        gs = [graph_from_obs(o) for o in obs.numpy()]
        self.conv_s_per_sample = (time.time() - t0) / len(gs)
        self.node = torch.as_tensor(np.stack([g["node"] for g in gs]))
        self.edge = torch.as_tensor(np.stack([g["edge"] for g in gs]))
        self.gate = torch.as_tensor(np.stack([g["gate"] for g in gs]))
        self.sem = [(torch.as_tensor(g["sem_house"]), torch.as_tensor(g["sem_dest"])) for g in gs]
        self.n = len(gs)

    def batch(self, idx, use_sem=True, blind=False):
        idx = idx.tolist() if torch.is_tensor(idx) else list(idx)
        node = self.node[idx].clone()
        if blind:
            node[:, :, F_NODE - K:] = 0.0
        src, dst = [], []
        if use_sem:
            for b, i in enumerate(idx):
                h, d = self.sem[i]
                src += [h + b * NT, d + b * NT]; dst += [d + b * NT, h + b * NT]       # undirected: house->dest and dest->house
        src = torch.cat(src) if src else torch.zeros(0, dtype=torch.long)
        dst = torch.cat(dst) if dst else torch.zeros(0, dtype=torch.long)
        return node, self.edge[idx], self.gate[idx], src, dst


# ----------------------------------------------------------------------------------------------------------------------
# 2. Model: node projection, LAYERS message-passing layers (separate physical / semantic weights, residual), edge scoring head
# ----------------------------------------------------------------------------------------------------------------------
class MPLayer(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.phys = nn.Linear(h + F_EDGE + 4, h)      # message from neighbour: [h_v, edge feats, direction one-hot]
        self.sem = nn.Linear(h, h)                    # message along a semantic (same-colour house<->destination) edge
        self.upd = nn.Sequential(nn.Linear(3 * h, h), nn.ReLU(), nn.Linear(h, h))

    def forward(self, hf, ef_dir, gate_dir, src_f, dst_f, sem_src, sem_dst):
        """hf: (B*NT, H) flattened nodes; src_f/dst_f: flattened directed physical edges (B*2*N_CAND,); ef_dir/gate_dir flattened likewise."""
        msg = torch.relu(self.phys(torch.cat([hf[src_f], ef_dir], -1))) * gate_dir[:, None]
        agg_p = torch.zeros_like(hf).index_add_(0, dst_f, msg) / 4.0
        agg_s = torch.zeros_like(hf)
        if sem_src.numel():
            agg_s = agg_s.index_add_(0, sem_dst, torch.relu(self.sem(hf[sem_src])))
            deg = torch.zeros(hf.shape[0], device=hf.device).index_add_(0, sem_dst, torch.ones(len(sem_dst))).clamp(min=1.0)
            agg_s = agg_s / deg[:, None]
        return hf + self.upd(torch.cat([hf, agg_p, agg_s], -1))


class RoutingGNN(nn.Module):
    def __init__(self, hidden=HIDDEN, layers=LAYERS):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(F_NODE, hidden), nn.ReLU())
        self.layers = nn.ModuleList([MPLayer(hidden) for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(2 * hidden + F_EDGE + 2, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.register_buffer("src", torch.as_tensor(DIR_SRC)); self.register_buffer("dst", torch.as_tensor(DIR_DST))
        self.register_buffer("dir1h", torch.nn.functional.one_hot(torch.as_tensor(DIR_DIR), 4).float())
        self.register_buffer("dcand", torch.as_tensor(DIR_CAND))
        self.register_buffer("cu", torch.as_tensor(CAND_U)); self.register_buffer("cv", torch.as_tensor(CAND_V))
        self.register_buffer("cd1h", torch.nn.functional.one_hot(torch.as_tensor(CAND_D), 2).float())
        self.register_buffer("cact", torch.as_tensor(CAND_ACT))

    def forward(self, node, edge, gate, sem_src, sem_dst):
        B = node.shape[0]
        ef_dir = torch.cat([edge[:, self.dcand], self.dir1h.expand(B, -1, -1)], -1).reshape(B * 2 * N_CAND, -1)
        gate_dir = gate[:, self.dcand].reshape(-1)
        off = (torch.arange(B, device=node.device) * NT)[:, None]
        src_f, dst_f = (self.src[None] + off).reshape(-1), (self.dst[None] + off).reshape(-1)
        hf = self.proj(node).reshape(B * NT, -1)
        for lay in self.layers:
            hf = lay(hf, ef_dir, gate_dir, src_f, dst_f, sem_src, sem_dst)
        h = hf.reshape(B, NT, -1)
        z = torch.cat([h[:, self.cu], h[:, self.cv], edge, self.cd1h.expand(B, -1, -1)], -1)
        score = self.head(z).squeeze(-1)                                   # (B, N_CAND)
        logits = torch.full((B, N_ACT), -1e9, device=node.device)
        logits[:, self.cact] = score                                       # candidate ADD actions only; REMOVE / non-candidate stay -1e9
        return logits


@torch.no_grad()
def predict(model, gs, add_mask, use_sem=True, blind=False, bs=64):
    model.eval()
    out = []
    for i in range(0, gs.n, bs):
        idx = list(range(i, min(i + bs, gs.n)))
        lg = model(*gs.batch(idx, use_sem, blind)).masked_fill(~add_mask[idx], -1e9)
        out.append(lg.argmax(dim=1))
    return torch.cat(out)


# ----------------------------------------------------------------------------------------------------------------------
# 3. Correctness checks on the conversion
# ----------------------------------------------------------------------------------------------------------------------
def conversion_checks(splits):
    chk = {}
    # (1) semantic edges connect only visible house/destination tiles of the same slot (independent recomputation from the raw obs)
    # (2) candidate edges obey the spatial conventions (E: x+1, S: y-1; in bounds; one representative per unordered pair)
    # (3) every label action and every legal ADD action maps to a candidate edge; legal ADD edges never touch water
    # (4) counterfactual twins differ in semantic edges (and in slot node features), identical physical edges
    # (5) no oracle/label data in the graph: graph is a pure function of obs (re-conversion of equal obs equal; feature list enumerated)
    # (6) test geometry unseen: no test observation equals any train observation
    act_to_cand = {int(a): k for k, a in enumerate(CAND_ACT)}
    ok_sem, ok_spatial = True, True
    for k in range(N_CAND):
        u, v = CAND_U[k], CAND_V[k]
        xu, yu = u % L.W + L.XMIN, u // L.W + L.YMIN
        xv, yv = v % L.W + L.XMIN, v // L.W + L.YMIN
        ok_spatial &= (CAND_D[k] == 0 and (xv, yv) == (xu + 1, yu)) or (CAND_D[k] == 1 and (xv, yv) == (xu, yu - 1))
        ok_spatial &= L.decode_action(int(CAND_ACT[k]) + 1) == ("ADD", xu, yu, "E" if CAND_D[k] == 0 else "S")
    chk["candidate_edges_spatial_convention_ok"] = bool(ok_spatial)
    chk["n_candidate_edges"] = int(N_CAND)
    chk["candidate_edges_unique"] = bool(len(set(zip(CAND_U.tolist(), CAND_V.tolist()))) == N_CAND and len(set(CAND_ACT.tolist())) == N_CAND)
    for name, (meta, obs, add_mask, tgt, oth, gs) in splits.items():
        n_label, n_label_ok, n_legal, n_legal_ok, n_legal_water = 0, 0, 0, 0, 0
        for i in range(gs.n):
            o = obs[i].numpy(); tiles = o[: NT * FT].reshape(NT, FT)
            house = np.abs(tiles[:, L.F_CONTENT] - L.CONTENT_HOUSE) < 0.1; dest = tiles[:, L.F_CONTENT] > 0.9
            sl = tiles[:, V.F_GROUP_SLOT0:V.F_GROUP_SLOT0 + K].argmax(1); has = tiles[:, V.F_GROUP_SLOT0:V.F_GROUP_SLOT0 + K].max(1) > 0.5
            hs, ds = gs.sem[i]
            for a, b in zip(hs.tolist(), ds.tolist()):
                ok_sem &= bool(house[a] and dest[b] and has[a] and has[b] and sl[a] == sl[b])
            expected = sum(int((house & has & (sl == k)).sum()) * int((dest & has & (sl == k)).sum()) for k in range(K))
            ok_sem &= len(hs) == expected
            for a in meta[i]["valid"]:
                n_label += 1; n_label_ok += int((a - 1) in act_to_cand)
            legal = np.nonzero(add_mask[i].numpy())[0]
            n_legal += len(legal)
            for a in legal:
                if int(a) in act_to_cand:
                    n_legal_ok += 1
                    n_legal_water += int(gs.gate[i, act_to_cand[int(a)]] == 0)
        geoms = {}
        for i, m in enumerate(meta):
            if m["colour_dependent"]:
                geoms.setdefault(m["geom"], []).append(i)
        pairs = [v for v in geoms.values() if len(v) == 2]
        sem_diff = sum(1 for a, b in pairs if not (torch.equal(gs.sem[a][0], gs.sem[b][0]) and torch.equal(gs.sem[a][1], gs.sem[b][1])))
        slot_diff = sum(1 for a, b in pairs if not torch.equal(gs.node[a, :, F_NODE - K:], gs.node[b, :, F_NODE - K:]))
        phys_same = sum(1 for a, b in pairs if torch.equal(gs.edge[a], gs.edge[b]) and torch.equal(gs.node[a, :, :F_NODE - K], gs.node[b, :, :F_NODE - K]))
        chk[name] = {"n": gs.n, "semantic_edges_only_visible_matching_slots": bool(ok_sem), "label_actions": n_label, "label_actions_on_candidate_edges": n_label_ok,
                     "legal_add_actions": n_legal, "legal_add_on_candidate_edges": n_legal_ok, "legal_add_touching_water": n_legal_water,
                     "cf_pairs": len(pairs), "cf_pairs_semantic_edges_differ": sem_diff, "cf_pairs_slot_features_differ": slot_diff, "cf_pairs_physical_graph_identical": phys_same,
                     "mean_semantic_edges_per_graph": float(np.mean([len(h) for h, _ in gs.sem])), "conversion_s_per_sample": gs.conv_s_per_sample}
    # (5) determinism / purity: re-convert a few observations and compare; enumerate the feature sources
    g1, g2 = graph_from_obs(splits["test"][1][0].numpy()), graph_from_obs(splits["test"][1][0].numpy())
    chk["conversion_deterministic"] = bool(all(np.array_equal(g1[k], g2[k]) for k in g1))
    chk["graph_inputs"] = "graph_from_obs(obs_vec) only; node features = " + ", ".join(NODE_FEAT_NAMES) + "; edge features = " + ", ".join(EDGE_FEAT_NAMES) + "; no meta/label/mask field is read"
    tr_h = {hash(o.numpy().tobytes()) for o in splits["train"][1]}
    chk["test_obs_identical_to_a_train_obs"] = int(sum(1 for o in splits["test"][1] if hash(o.numpy().tobytes()) in tr_h))
    chk["test_train_geom_id_overlap_note"] = "splits are by geometry seed (S16); geom ids are per-split counters, so identity is checked on the observations themselves"
    return chk


# ----------------------------------------------------------------------------------------------------------------------
# 4. Main
# ----------------------------------------------------------------------------------------------------------------------
def main():
    torch.set_num_threads(4)
    torch.manual_seed(SEED); np.random.seed(SEED)
    t_all = time.time()
    splits = {}
    t_conv = time.time()
    for name in ("train", "val", "test"):
        meta, obs, add_mask, tgt, oth = S16.load(name)
        splits[name] = (meta, obs, add_mask, tgt, oth, GraphSet(obs))
    conv_total = time.time() - t_conv
    checks = conversion_checks(splits)
    checks["conversion_total_s_all_splits"] = round(conv_total, 2)
    print("CHECKS", json.dumps(checks, default=str), flush=True)
    json.dump(checks, open(os.path.join(OUT, "conversion_checks.json"), "w"), indent=1, default=str)

    tr_meta, tr_obs, tr_mask, tr_tgt, tr_oth, tr_gs = splits["train"]
    keep = [i for i, m in enumerate(tr_meta) if m["query_slot"] != HOLDOUT_SLOT]
    va_meta, va_obs, va_mask, va_tgt, va_oth, va_gs = splits["val"]
    te_meta, te_obs, te_mask, te_tgt, te_oth, te_gs = splits["test"]
    results = {"checks": checks, "config": {"seed": SEED, "lr": LR, "batch": BATCH, "epochs_max": EPOCHS, "time_cap_per_model_s": TIME_PER_MODEL_S, "hidden": HIDDEN, "layers": LAYERS,
                                            "optimizer": "Adam", "loss": "S16 set_loss: -log sum_valid softmax over legal ADD actions", "train_samples_used": len(keep),
                                            "train_slot2_dropped": len(tr_meta) - len(keep), "holdout_slot": HOLDOUT_SLOT, "threads": 4}, "models": {}}
    t_train_eval = time.time()
    hits, preds, models = {}, {}, {}
    for arm in ("G", "G0"):
        use_sem, blind = (arm == "G"), (arm == "G0")
        torch.manual_seed(SEED)
        model = RoutingGNN()
        n_par = sum(p.numel() for p in model.parameters())
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        curve, best, ep, t0, n_seen = [], (0.0, None), 0, time.time(), 0
        while ep < EPOCHS and time.time() - t0 < TIME_PER_MODEL_S and time.time() - t_all < GLOBAL_CAP_S - 150:
            model.train()
            perm = torch.as_tensor(keep)[torch.randperm(len(keep))]
            tl = 0.0
            for i in range(0, len(perm), BATCH):
                j = perm[i:i + BATCH]
                loss = S16.set_loss(model(*tr_gs.batch(j, use_sem, blind)), tr_mask[j], tr_tgt[j])
                opt.zero_grad(); loss.backward(); opt.step()
                tl += float(loss) * len(j); n_seen += len(j)
            ep += 1
            t_ep = time.time() - t0
            pv = predict(model, va_gs, va_mask, use_sem, blind)
            mv, _ = S16.metrics(va_meta, pv, va_tgt, va_oth)
            curve.append({"epoch": ep, "train_loss": round(tl / len(perm), 4), "val_acc": round(mv["all"]["valid_set_acc"], 4), "val_cd_acc": round(mv["colour_dependent"]["valid_set_acc"], 4),
                          "val_cf_pair_acc": mv["cf_pairs_both_correct"]["acc"], "s": round(time.time() - t0, 1)})
            if mv["all"]["valid_set_acc"] >= best[0]:
                best = (mv["all"]["valid_set_acc"], {k: v.clone() for k, v in model.state_dict().items()})
            print(arm, curve[-1], flush=True)
        train_s = time.time() - t0
        model.load_state_dict(best[1])
        torch.save(model.state_dict(), os.path.join(OUT, f"model_{arm}.pt"))
        models[arm] = model
        pt = predict(model, te_gs, te_mask, use_sem, blind)
        mt, hit = S16.metrics(te_meta, pt, te_tgt, te_oth)
        hits[arm], preds[arm] = hit, pt
        fails = [{"geom": m["geom"], "cf": m["cf"], "query": m["query"], "query_slot": m["query_slot"], "decoy_slot": m["decoy_slot"], "within_rf": m["within_rf"], "cost_correct": m["cost_correct"], "cost_decoy": m["cost_decoy"],
                  "pred": L.decode_action(int(pt[i]) + 1), "valid": [L.decode_action(a) for a in m["valid"]], "chose_decoy_target": bool(te_oth[i, pt[i]])} for i, m in enumerate(te_meta) if not hit[i] and m["colour_dependent"]][:6]
        results["models"][arm] = {"params": n_par, "group_aware": use_sem, "epochs_done": ep, "train_s": round(train_s, 1), "train_samples_per_s": round(n_seen / train_s, 1), "best_val_acc": best[0],
                                  "curve": curve, "test": mt, "failure_examples": fails}
        print(arm, json.dumps({k: v for k, v in results["models"][arm].items() if k not in ("curve", "failure_examples")}), flush=True)
    # paired bootstrap G - G0
    sub = {"all": np.arange(len(te_meta)), "colour_dependent": np.array([i for i, m in enumerate(te_meta) if m["colour_dependent"]]), "beyond_rf": np.array([i for i, m in enumerate(te_meta) if not m["within_rf"]]),
           "within_rf": np.array([i for i, m in enumerate(te_meta) if m["within_rf"]]), "unseen_slot": np.array([i for i, m in enumerate(te_meta) if m["query_slot"] == HOLDOUT_SLOT])}
    results["paired"] = {"G_minus_G0": {k: S16.paired_bootstrap(hits["G"][v], hits["G0"][v]) for k, v in sub.items() if len(v)}}
    # inference-only ablations of the trained G
    abl = {}
    for name, (use_sem, blind) in (("a_no_semantic_edges", (False, False)), ("b_zero_group_features", (True, True)), ("c_both", (False, True))):
        pa = predict(models["G"], te_gs, te_mask, use_sem, blind)
        ma, ha = S16.metrics(te_meta, pa, te_tgt, te_oth)
        abl[name] = {"test": {k: ma[k]["valid_set_acc"] for k in ("all", "colour_dependent", "beyond_rf", "unseen_slot", "cf_twins", "originals_of_cf")}, "cf_pairs_both_correct": ma["cf_pairs_both_correct"]["acc"],
                     "delta_vs_G": {k: round(ma[k]["valid_set_acc"] - results["models"]["G"]["test"][k]["valid_set_acc"], 4) for k in ("all", "colour_dependent", "beyond_rf", "unseen_slot")},
                     "n_test_predictions_changed": int((pa != preds["G"]).sum()), "n_changed_colour_dependent": int((pa != preds["G"])[sub["colour_dependent"]].sum()),
                     "paired_vs_G_colour_dependent": S16.paired_bootstrap(ha[sub["colour_dependent"]], hits["G"][sub["colour_dependent"]])}
    results["ablations_of_G"] = abl
    # runtime costs
    model = models["G"].eval()
    with torch.no_grad():
        for _ in range(3):
            model(*te_gs.batch([0]))
        t = time.time()
        for i in range(50):
            model(*te_gs.batch([i]))
        lat1 = (time.time() - t) / 50
        t = time.time()
        for _ in range(5):
            model(*te_gs.batch(list(range(64))))
        lat64 = (time.time() - t) / 5
    t = time.time()
    for i in range(100):
        graph_from_obs(te_obs[i].numpy())
    conv1 = (time.time() - t) / 100
    results["cost"] = {"params": results["models"]["G"]["params"], "conversion_ms_per_sample": round(conv1 * 1e3, 3), "inference_ms_batch1_per_sample": round(lat1 * 1e3, 2), "inference_ms_batch64_per_sample": round(lat64 / 64 * 1e3, 2),
                       "inference_ms_batch64_total": round(lat64 * 1e3, 1), "train_samples_per_s": {a: results["models"][a]["train_samples_per_s"] for a in ("G", "G0")},
                       "s16_cnn_A_params_reference": 61764}
    results["compute_s_train_eval"] = round(time.time() - t_train_eval, 1)
    results["compute_s_total_incl_conversion"] = round(time.time() - t_all, 1)
    json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=str)
    print("PAIRED", json.dumps(results["paired"]))
    print("ABLATIONS", json.dumps(abl, default=str))
    print("COST", json.dumps(results["cost"]))
    print("TOTAL train+eval", results["compute_s_train_eval"], "s; incl conversion", results["compute_s_total_incl_conversion"], "s")


if __name__ == "__main__":
    main()
