"""S20 behaviour cloning of the S18-H construction skill (one model, one configuration, no sweep; 2 CPU threads; ~20 min total incl. optional DAgger).
Loss: cross-entropy of the planner action over LEGAL actions (illegal logits masked to -1e9).  Metrics per epoch (val): loss, imitation accuracy
(masked argmax == planner action), valid-set accuracy (masked argmax in the oracle valid set; for terminal states the valid set is {DONE}),
DONE precision / recall.  Best-val (imitation accuracy) checkpoint.  Optional DAgger round: greedy autonomous rollouts of the learned policy
on train scenes, every visited state relabelled by a fresh GoalExecutor plan (demos.label_state), appended, brief retraining -- performed only
if the autonomous completion rate on train scenes is clearly below the imitation accuracy (decision rule and outcome written to train_log.json)."""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import demos as D  # noqa: E402
import rollout as R  # noqa: E402
from model import GoalPolicy  # noqa: E402

torch.set_num_threads(2)
DATA = os.path.join(C.S20, "data")
RUN = os.path.join(C.S20, "runs")
BATCH, LR, SEED = 128, 1e-3, 0
BC_BUDGET_S = 600            # fixed wall budget for the BC phase -> fixed epoch count computed once from the measured step time
DAGGER_SCENES = 250
DAGGER_BUDGET_S = 300
TOTAL_BUDGET_S = 1200


def load(split):
    z = np.load(os.path.join(DATA, f"demos_{split}.npz"))
    d = {k: z[k] for k in z.files}
    d["mask"] = np.unpackbits(d["mask"], axis=1)[:, : C.N_ACT].astype(bool)
    n = len(d["action"])
    vs = np.zeros((n, C.N_ACT), dtype=bool)
    for i in range(n):
        vs[i, d["vflat"][d["voff"][i]:d["voff"][i + 1]]] = True
    d["validset"] = vs
    return d


def to_tensors(d):
    return (torch.from_numpy(d["tiles"]), torch.from_numpy(d["glob"]), torch.from_numpy(d["mask"]), torch.from_numpy(d["action"].astype(np.int64)), torch.from_numpy(d["validset"]))


def evaluate(model, T, bs=256):
    tiles, glob, mask, act, vs = T
    model.eval()
    n = len(act)
    loss = 0.0
    correct = valid = 0
    done_tp = done_fp = done_fn = 0
    with torch.no_grad():
        for i in range(0, n, bs):
            lo = model(tiles[i:i + bs], glob[i:i + bs]).masked_fill(~mask[i:i + bs], -1e9)
            y = act[i:i + bs]
            loss += F.cross_entropy(lo, y, reduction="sum").item()
            p = lo.argmax(1)
            correct += (p == y).sum().item()
            valid += vs[i:i + bs][torch.arange(len(p)), p].sum().item()
            done_tp += ((p == C.A_DONE) & (y == C.A_DONE)).sum().item()
            done_fp += ((p == C.A_DONE) & (y != C.A_DONE)).sum().item()
            done_fn += ((p != C.A_DONE) & (y == C.A_DONE)).sum().item()
    model.train()
    return {"loss": loss / n, "imitation_acc": correct / n, "valid_acc": valid / n, "done_precision": done_tp / max(1, done_tp + done_fp), "done_recall": done_tp / max(1, done_tp + done_fn), "n": n}


def train_epochs(model, opt, Ttr, Tval, epochs, log, tag, best, ckpt):
    tiles, glob, mask, act, vs = Ttr
    n = len(act)
    g = torch.Generator().manual_seed(SEED + len(log))
    for ep in range(epochs):
        t0 = time.time()
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        nb = 0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            lo = model(tiles[idx], glob[idx]).masked_fill(~mask[idx], -1e9)
            loss = F.cross_entropy(lo, act[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        ev = evaluate(model, Tval)
        row = {"phase": tag, "epoch": ep, "train_loss": tot / nb, **{f"val_{k}": v for k, v in ev.items() if k != "n"}, "epoch_s": round(time.time() - t0, 1)}
        log.append(row)
        print(json.dumps(row), flush=True)
        if ev["imitation_acc"] > best["acc"]:
            best.update({"acc": ev["imitation_acc"], "row": row})
            torch.save(model.state_dict(), ckpt)
    return best


def dagger_collect(model, scenes, n_scenes, rng):
    """greedy autonomous rollouts on train scenes; every visited state (incl. the start) relabelled by the planner; returns rows + rollout success."""
    pol = R.LearnedPolicy(model, "greedy", seed=1)
    rows, succ = [], 0
    for i in rng.permutation(len(scenes))[:n_scenes]:
        sc = scenes[int(i)]
        goal = tuple(sc["meta"]["goal"])
        env = C.restore_env(sc["state"])
        budget = max(6, 3 * sc["meta"]["n_adds"])
        for step in range(budget + 1):
            tiles, glob = C.goal_obs_u8(env, goal)
            mask = C.legal_mask_ext(env)
            a_lab, _ = D.label_state(env, goal)
            valid, _ = C.oracle_valid(env, goal)
            rows.append({"tiles": tiles, "glob": glob, "mask": mask, "action": a_lab, "concrete": env.concrete, "done": a_lab == C.A_DONE, "valid": sorted(valid), "step": step, "scene": int(i)})
            if a_lab == C.A_DONE:
                succ += 1
                break
            if step == budget:
                break
            a = pol(env, goal)
            if a == C.A_DONE or not env.action_masks()[a]:
                break
            env.step(a)
    return rows, succ / n_scenes


def main():
    os.makedirs(RUN, exist_ok=True)
    T0 = time.time()
    torch.manual_seed(SEED)
    tr, va = load("train"), load("val")
    Ttr, Tval = to_tensors(tr), to_tensors(va)
    model = GoalPolicy()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    # fixed epoch count from one measured step (one configuration; no sweep)
    t = time.time()
    lo = model(Ttr[0][:BATCH], Ttr[1][:BATCH]).masked_fill(~Ttr[2][:BATCH], -1e9)
    F.cross_entropy(lo, Ttr[3][:BATCH]).backward()
    opt.zero_grad()
    step_s = time.time() - t
    steps_per_epoch = int(np.ceil(len(tr["action"]) / BATCH))
    epochs = int(max(3, min(30, BC_BUDGET_S / (step_s * steps_per_epoch * 1.15))))
    info = {"n_train": int(len(tr["action"])), "n_val": int(len(va["action"])), "params": model.n_params(), "receptive_field": model.receptive_field, "batch": BATCH, "lr": LR, "step_s": round(step_s, 3),
            "steps_per_epoch": steps_per_epoch, "bc_epochs": epochs, "train_wait_labels": int((tr["action"] == 0).sum()), "train_done_labels": int((tr["action"] == C.A_DONE).sum())}
    print(json.dumps(info), flush=True)
    log = []
    best = {"acc": -1.0, "row": None}
    ckpt = os.path.join(RUN, "model_bc.pt")
    best = train_epochs(model, opt, Ttr, Tval, epochs, log, "bc", best, ckpt)
    model.load_state_dict(torch.load(ckpt))
    bc_wall = time.time() - T0
    info["bc_wall_s"] = round(bc_wall, 1)
    info["bc_best"] = best["row"]
    # ---------------- DAgger decision: autonomous greedy completion on a train subset vs imitation accuracy
    scenes = json.load(open(os.path.join(DATA, "scenes_train.json")))
    rng = np.random.default_rng(SEED)
    pol = R.LearnedPolicy(model, "greedy", seed=2)
    probe_idx = rng.permutation(len(scenes))[:100]
    probe = [R.run_task(scenes[int(i)], pol, max(6, 3 * scenes[int(i)]["meta"]["n_adds"])) for i in probe_idx]
    auto = float(np.mean([r["success"] for r in probe]))
    info["dagger_probe"] = {"scenes": len(probe), "autonomous_completion_train": auto, "imitation_acc_val": best["acc"], "rule": "DAgger if completion < imitation_acc - 0.10"}
    do_dagger = auto < best["acc"] - 0.10 and (time.time() - T0) < TOTAL_BUDGET_S - DAGGER_BUDGET_S
    info["dagger_done"] = bool(do_dagger)
    print("DAgger probe", json.dumps(info["dagger_probe"]), "->", do_dagger, flush=True)
    final = ckpt
    if do_dagger:
        t = time.time()
        rows, dsucc = dagger_collect(model, scenes, DAGGER_SCENES, rng)
        packed = D.pack(rows)
        np.savez_compressed(os.path.join(DATA, "dagger_round1.npz"), **packed)
        packed["mask"] = np.unpackbits(packed["mask"], axis=1)[:, : C.N_ACT].astype(bool)
        vs = np.zeros((len(rows), C.N_ACT), dtype=bool)
        for i, r in enumerate(rows):
            vs[i, r["valid"]] = True
        packed["validset"] = vs
        merged = {k: np.concatenate([tr[k], packed[k]]) for k in ("tiles", "glob", "mask", "action", "validset")}
        Tm = to_tensors(merged)
        collect_s = time.time() - t
        remaining = TOTAL_BUDGET_S - (time.time() - T0) - 30
        steps_m = int(np.ceil(len(merged["action"]) / BATCH))
        d_epochs = int(max(1, min(10, remaining / (step_s * steps_m * 1.15))))
        info["dagger"] = {"relabelled_states": len(rows), "rollout_completion_during_collection": dsucc, "collect_s": round(collect_s, 1), "retrain_epochs": d_epochs, "merged_n": int(len(merged["action"]))}
        opt = torch.optim.Adam(model.parameters(), lr=LR / 2)
        ckpt2 = os.path.join(RUN, "model_dagger.pt")
        torch.save(model.state_dict(), ckpt2)
        best2 = {"acc": best["acc"], "row": best["row"]}
        best2 = train_epochs(model, opt, Tm, Tval, d_epochs, log, "dagger", best2, ckpt2)
        info["dagger"]["best"] = best2["row"]
        final = ckpt2
        model.load_state_dict(torch.load(ckpt2))
        pol = R.LearnedPolicy(model, "greedy", seed=2)
        probe2 = [R.run_task(scenes[int(i)], pol, max(6, 3 * scenes[int(i)]["meta"]["n_adds"])) for i in probe_idx]
        info["dagger"]["autonomous_completion_train_after"] = float(np.mean([r["success"] for r in probe2]))
    info["final_checkpoint"] = os.path.basename(final)
    info["final_val"] = evaluate(model, Tval)
    info["total_wall_s"] = round(time.time() - T0, 1)
    json.dump({"info": info, "curve": log}, open(os.path.join(RUN, "train_log.json"), "w"), indent=1)
    print(json.dumps(info), flush=True)


def resume(budget_s):
    """continue the SAME run (same model, batch, lr; fresh Adam moments) for the remaining wall budget; re-applies the DAgger decision rule."""
    T0 = time.time()
    torch.manual_seed(SEED + 1)
    prev = json.load(open(os.path.join(RUN, "train_log.json")))
    info, log = prev["info"], prev["curve"]
    tr, va = load("train"), load("val")
    Ttr, Tval = to_tensors(tr), to_tensors(va)
    model = GoalPolicy()
    ckpt = os.path.join(RUN, "model_bc.pt")
    model.load_state_dict(torch.load(ckpt))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    step_s = info["step_s"]
    epochs = int(max(1, min(30, (budget_s - 90) / (step_s * info["steps_per_epoch"] * 1.0))))
    best = {"acc": info["bc_best"]["val_imitation_acc"], "row": info["bc_best"]}
    best = train_epochs(model, opt, Ttr, Tval, epochs, log, "bc_resume", best, ckpt)
    model.load_state_dict(torch.load(ckpt))
    info["resume"] = {"epochs": epochs, "wall_s": round(time.time() - T0, 1), "reason": "first BC phase was CPU-starved by concurrent checks (6 epochs in 600 s); same configuration continued for the remaining time box"}
    info["bc_best"] = best["row"]
    scenes = json.load(open(os.path.join(DATA, "scenes_train.json")))
    rng = np.random.default_rng(SEED)
    pol = R.LearnedPolicy(model, "greedy", seed=2)
    probe_idx = rng.permutation(len(scenes))[:100]
    probe = [R.run_task(scenes[int(i)], pol, max(6, 3 * scenes[int(i)]["meta"]["n_adds"])) for i in probe_idx]
    auto = float(np.mean([r["success"] for r in probe]))
    info["dagger_probe_after_resume"] = {"scenes": 100, "autonomous_completion_train": auto, "imitation_acc_val": best["acc"], "rule": "DAgger if completion < imitation_acc - 0.10", "decision": bool(auto < best["acc"] - 0.10)}
    info["dagger_done"] = False
    info["final_checkpoint"] = "model_bc.pt"
    info["final_val"] = evaluate(model, Tval)
    info["total_wall_s"] = round(info["total_wall_s"] + time.time() - T0, 1)
    json.dump({"info": info, "curve": log}, open(os.path.join(RUN, "train_log.json"), "w"), indent=1)
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--resume":
        resume(float(sys.argv[2]))
    else:
        main()
