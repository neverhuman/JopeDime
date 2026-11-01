"""H163 single-attempt CPU evaluator; predictions are locked before each A outcome read."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import analysis as stat

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
H146 = ROOT / "research/flywheel_selfplay/h146_lookahead_exec"
H160 = ROOT / "research/flywheel_selfplay/h160_trajectory_subspace"
DATA = Path("/home/ubuntu/jope-data/h160")
OUT_DEFAULT = Path("/home/ubuntu/jope-data/h163")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parent_api():
    sys.path.insert(0, str(H146))
    spec = importlib.util.spec_from_file_location("h163_parent", H146 / "runner.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, sys.modules["runtime"]


def model_costs(net, parent, rt, pools, tokens, masks, rows):
    out = {}
    sizes = {}
    for row in rows:
        cid = row["context_sha256"]
        sizes.setdefault(len(pools[cid]["fitness"]), []).append(cid)
    with __import__("torch").no_grad():
        for size, cids in sorted(sizes.items()):
            for cid in cids:
                score = parent.predict(net, pools, tokens, masks, [cid],
                                       [np.arange(size)]).detach().cpu().numpy()[0]
                if score.shape != (size,) or not np.isfinite(score).all():
                    raise RuntimeError("invalid candidate cost vector")
                out[cid] = score.astype(np.float64)
    return out


def row_utility(pool, choice):
    return float(-pool["fitness"][choice] if pool["gate"][choice] else -2.0)


def elo(rows, panel):
    """H145 400-iteration Bradley-Terry MM, equal-family context weights."""
    x = np.asarray(panel, dtype=np.float64)
    fam = np.asarray([r["family_id"] for r in rows])
    unique, inverse = np.unique(fam, return_inverse=True)
    weights = 1.0 / (len(unique) * np.bincount(inverse)[inverse])
    m = x.shape[1]
    wins = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(i + 1, m):
            d = x[:, i] - x[:, j]
            w = float(weights[d > 1e-4].sum() + 0.5 * weights[np.abs(d) <= 1e-4].sum())
            wins[i, j], wins[j, i] = w, 1.0 - w
    p = np.ones(m, dtype=np.float64)
    for _ in range(400):
        nxt = np.ones(m, dtype=np.float64)
        for i in range(m):
            denom = sum(1.0 / (p[i] + p[j]) for j in range(m) if j != i)
            nxt[i] = max(wins[i].sum() / max(denom, 1e-12), 1e-12)
        p = nxt / p[0]
    return (1500.0 + (400.0 / np.log(10.0)) * np.log(p)).tolist()


def panel_utility(rows, pools, predictions, arm, weight=0.5):
    values, choices = [], []
    for row in rows:
        cid = row["context_sha256"]
        costs = np.stack([predictions[m][cid][None, :] for m in ("R882", "R859", "R785", "R782")])
        choice = int(stat.choose_indices(stat.arm_costs(costs, arm, weight))[0])
        choices.append(choice)
        values.append(row_utility(pools[cid], choice))
    return np.asarray(values, dtype=np.float64), np.asarray(choices, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--launch", type=Path)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = args.output.resolve()
    out.mkdir()  # One attempt; never reuse or resume output.
    pins = json.loads((HERE / "manifest.json").read_text())
    for p, expected in pins["parent_sources"].items():
        if sha(p) != expected:
            raise RuntimeError("H146 parent source hash mismatch: " + p)
    for p, expected in pins["assets"].items():
        if sha(p) != expected:
            raise RuntimeError("pinned inference asset hash mismatch: " + p)
    if not args.smoke:
        if args.launch is None:
            raise RuntimeError("--launch required")
        launch = json.loads(args.launch.read_text())
        if launch.get("manifest_sha256") != sha(HERE / "manifest.json"):
            raise RuntimeError("launch manifest hash mismatch")
        if not launch.get("merged_source_verified") or not launch.get("redteam_review_pass"):
            raise RuntimeError("merged source and red-team review PASS required")
        import subprocess
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        if launch.get("source_commit") != head:
            raise RuntimeError("launch source commit mismatch")
    for name, item in pins["checkpoint_inputs"].items():
        if sha(item["path"]) != item["sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {name}")
    if sha(pins["selected_metadata_path"]) != pins["selected_metadata_sha256"]:
        raise RuntimeError("selected cohort metadata hash mismatch")
    if sha(pins["feature_cache_path"]) != pins["feature_cache_sha256"]:
        raise RuntimeError("feature cache hash mismatch")
    if sha(pins["baseline_reference"]["path"]) != pins["baseline_reference"]["sha256"]:
        raise RuntimeError("baseline reference hash mismatch")
    parent, rt = parent_api()
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    train, val = parent.selected()
    if len(train) != 1219 or len({r["family_id"] for r in train}) != 205 or len(val) != 91:
        raise RuntimeError("cohort inventory mismatch")
    expected_families = json.loads((HERE / "fold_manifest.json").read_text())["families"]
    if stat.folds(train) != expected_families:
        raise RuntimeError("fixed family fold identity mismatch")
    net, guard, names, frozen, fused = rt.build(pins["checkpoint_inputs"]["R882"]["path"])
    if rt.DOPE != pins["dope_composite_sha256"]:
        raise RuntimeError("DOPE composite mismatch")
    rt.check_frozen(net, guard, names, frozen)
    states = {}
    for name, item in pins["checkpoint_inputs"].items():
        state = torch.load(item["path"], map_location="cpu", weights_only=True)["state_dict"]
        if set(state) != set(net.state_dict()):
            raise RuntimeError("checkpoint state key set mismatch")
        for key, value in net.state_dict().items():
            if key not in names and not torch.equal(state[key], value):
                raise RuntimeError("non-adapter tensor changed: " + key)
        states[name] = state
    adapter_parameters = sum(p.numel() for n, p in net.named_parameters() if n in names)
    if adapter_parameters != 29_211:
        raise RuntimeError("expected exact 29,211 adapter parameters")
    if args.smoke:
        (out / "smoke.json").write_text(json.dumps({"status": "PASS", "forward_calls": 0,
            "optimizer_updates": 0, "adapter_parameters": adapter_parameters,
            "models_checked": sorted(states), "dope_composite_sha256": rt.DOPE,
            "frozen_sha256": frozen, "torch_threads": torch.get_num_threads()}, indent=2) + "\n")
        return
    allowed = {x["path"] for x in pins["checkpoint_inputs"].values()}
    allowed |= {str(rt.INPUT / "selected.json"), str(rt.CACHE), pins["baseline_reference"]["path"]}
    allowed |= {str(rt.INPUT / "pools" / f"{r['context_sha256']}.npz") for r in train + val}
    audit = rt.Audit(allowed, out)
    sys.addaudithook(audit.check)
    pools = {}
    projection = fused.get_w1_sign_matrix() / np.sqrt(412.0)

    def load_pools(rows):
        for row in rows:
            cid = row["context_sha256"]
            path = rt.INPUT / "pools" / f"{cid}.npz"
            if rt.sha(path) != row["pool_sha256"]:
                raise RuntimeError("pool identity mismatch")
            with np.load(path, allow_pickle=False) as z:
                pools[cid] = {"program": z["program"].astype(np.float32),
                              "fitness": z["fitness"].astype(np.float64),
                              "gate": z["gate"].astype(bool)}
            pools[cid]["proj16"] = (pools[cid]["program"].astype(np.float64) @ projection).astype(np.float32)

    with np.load(rt.CACHE, allow_pickle=False) as z:
        ids = list(map(str, z["cids"]))
        if set(ids) != {r["context_sha256"] for r in train + val} or len(ids) != len(train) + len(val):
            raise RuntimeError("feature cache closure mismatch")
        tokens = {c: torch.as_tensor(z["tokens_10x32"][i], dtype=torch.float32) for i, c in enumerate(ids)}
        masks = {c: torch.as_tensor(z["mask_10"][i], dtype=torch.bool) for i, c in enumerate(ids)}
    # Stage 1 opens only DL-Val outcome pools and verifies exact R882 per-context parity.
    load_pools(val)
    model_preds_val = {}
    initial = {k: v.detach().clone() for k, v in net.state_dict().items()}
    for name in ("R882", "R859", "R785", "R782"):
        state = torch.load(pins["checkpoint_inputs"][name]["path"], map_location="cpu", weights_only=True)["state_dict"]
        if set(state) != set(initial):
            raise RuntimeError("checkpoint key mismatch")
        for key, value in initial.items():
            if key not in names and not torch.equal(state[key], value):
                raise RuntimeError("non-adapter checkpoint tensor differs")
        net.load_state_dict(state, strict=True)
        model_preds_val[name] = model_costs(net, parent, rt, pools, tokens, masks, val)
        rt.check_frozen(net, guard, names, frozen)
    # The first scored operation is exact per-context R882 DL-Val; TRAIN outcome pools remain unopened.
    baseline, baseline_choices = panel_utility(val, pools, model_preds_val, "R882")
    ref_path = DATA / "inputs/best_dev_checkpoint.json"
    reference = json.loads(ref_path.read_text())
    gate = parent.baseline_gate(baseline.tolist(), reference)
    (out / "baseline.json").write_text(json.dumps({"gate": gate, "choices": baseline_choices.tolist()}, indent=2) + "\n")
    # TRAIN outcome pools are opened only after the exact secondary baseline passes.
    load_pools(train)
    model_preds = dict(model_preds_val)
    for name in ("R882", "R859", "R785", "R782"):
        state = torch.load(pins["checkpoint_inputs"][name]["path"], map_location="cpu", weights_only=True)["state_dict"]
        net.load_state_dict(state, strict=True)
        model_preds[name].update(model_costs(net, parent, rt, pools, tokens, masks, train))
        rt.check_frozen(net, guard, names, frozen)
    all_rows = train + val
    offsets = [0]
    flat = {name: [] for name in ("R882", "R859", "R785", "R782")}
    for row in all_rows:
        cid = row["context_sha256"]
        n = len(pools[cid]["fitness"])
        for name in flat:
            if model_preds[name][cid].shape != (n,):
                raise RuntimeError("candidate order/score vector mismatch")
            flat[name].append(model_preds[name][cid])
        offsets.append(offsets[-1] + n)
    np.savez_compressed(out / "prediction_panels.npz",
                        context_ids=np.asarray([r["context_sha256"] for r in all_rows]),
                        candidate_index=np.concatenate([np.arange(len(pools[r["context_sha256"]]["fitness"]), dtype=np.int32) for r in all_rows]),
                        offsets=np.asarray(offsets, dtype=np.int64),
                        **{f"cost_{name}": np.concatenate(parts).astype(np.float32) for name, parts in flat.items()})
    base_train, base_train_choices = panel_utility(train, pools, model_preds, "R882")
    folds = stat.folds(train)
    fold_index = {f: i for i, group in enumerate(folds) for f in group}
    by_cid = {r["context_sha256"]: i for i, r in enumerate(train)}
    results = {arm: np.full((len(stat.SEEDS), len(train)), np.nan) for arm in stat.ARMS}
    choices_oof = {arm: np.full((len(stat.SEEDS), len(train)), -1, dtype=np.int64) for arm in stat.ARMS}
    secondary = {arm: np.full((len(stat.SEEDS), 5, len(val)), np.nan) for arm in stat.ARMS}
    secondary_choices = {arm: np.full((len(stat.SEEDS), 5, len(val)), -1, dtype=np.int64) for arm in stat.ARMS}
    decisions = []
    for outer in range(5):
        select_fold = (outer + 1) % 5
        fit = [r for r in train if fold_index[r["family_id"]] not in (outer, select_fold)]
        select = [r for r in train if fold_index[r["family_id"]] == select_fold]
        assess = [r for r in train if fold_index[r["family_id"]] == outer]
        fit_fams, sel_fams, out_fams = ({r["family_id"] for r in group} for group in (fit, select, assess))
        if fit_fams & sel_fams or fit_fams & out_fams or sel_fams & out_fams:
            raise RuntimeError("family leakage")
        weight_scores = {}
        for w in (0.5, 0.7):
            vals, _ = panel_utility(fit, pools, model_preds, "ENS4_W", w)
            fams, means = stat.family_means(fit, vals)
            weight_scores[str(w)] = float(means.mean())
        weight = 0.5 if weight_scores["0.5"] >= weight_scores["0.7"] else 0.7
        base_sel, _ = panel_utility(select, pools, model_preds, "R882")
        for si, seed in enumerate(stat.SEEDS):
            locked = {}
            for arm in ("ENS2", "ENS4", "ENS4_W", "RANKAVG4"):
                vals, _ = panel_utility(select, pools, model_preds, arm, weight)
                stat_gate = stat.family_bootstrap(select, vals, base_sel, 10_000, seed + outer * 1000)
                locked[arm] = {"admit": stat.admit(stat_gate), "selector": stat_gate}
            decision = {"seed": seed, "outer_fold": outer, "selection_fold": select_fold,
                        "ens4_w": weight, "fit_weight_utility": weight_scores, "arms": locked}
            # Persist the lock before assessing A outcomes.
            lock_path = out / f"A{outer}-{seed}-selection-lock.json"
            lock_path.write_text(json.dumps(decision, indent=2, allow_nan=False) + "\n")
            for arm in stat.ARMS:
                use_arm = arm == "R882" or locked[arm]["admit"]
                vals, choices = panel_utility(assess, pools, model_preds, arm if use_arm else "R882", weight)
                for row, v, choice in zip(assess, vals, choices, strict=True):
                    idx = by_cid[row["context_sha256"]]
                    results[arm][si, idx] = v
                    choices_oof[arm][si, idx] = choice
                val_values, val_choices = panel_utility(val, pools, model_preds, arm if use_arm else "R882", weight)
                secondary[arm][si, outer] = val_values
                secondary_choices[arm][si, outer] = val_choices
            decisions.append(decision)
    if any(not np.isfinite(v).all() for v in results.values()):
        raise RuntimeError("incomplete outer assessment panels")
    arm_names = list(stat.ARMS)
    panel = np.stack([results[a] for a in arm_names], axis=-1)
    summary = {}
    for j, arm in enumerate(arm_names):
        per_seed = []
        for si, seed in enumerate(stat.SEEDS):
            s = stat.family_bootstrap(train, results[arm][si], results["R882"][si], 10_000, stat.BOOT_SEED + seed)
            per_seed.append(s)
        delta = [s["delta"] for s in per_seed]
        summary[arm] = {
            "per_seed": per_seed,
            "median_gain": float(np.median(delta)),
            "median_ci_low_95": float(np.median([s["ci_low_95"] for s in per_seed])),
            "median_ci_high_95": float(np.median([s["ci_high_95"] for s in per_seed])),
            "elo_vs_r882": [e[j] for e in [elo(train, results_for_seed) for results_for_seed in np.stack([results[a] for a in arm_names], axis=-1)]],
            "changed_choices_per_seed": [int(np.sum(choices_oof[arm][si] != choices_oof["R882"][si])) for si in range(len(stat.SEEDS))],
            "minus2_per_seed": [int(np.sum(results[arm][si] <= -2.0 + 1e-12)) for si in range(len(stat.SEEDS))],
        }
    for arm in arm_names[1:]:
        summary[arm]["verdict"] = "ADVANCE" if summary[arm]["median_gain"] > .001 and summary[arm]["median_ci_low_95"] > 0 else "NO_ADVANCE"
    for arm in arm_names:
        summary[arm]["dlval_selected_policy"] = {
            "mean_utility_per_seed": [float(secondary[arm][si].mean()) for si in range(len(stat.SEEDS))],
            "mean_utility_median": float(np.median([secondary[arm][si].mean() for si in range(len(stat.SEEDS))])),
            "elo_per_seed": [elo(val, np.column_stack([baseline, secondary[arm][si].mean(axis=0)]))[1] for si in range(len(stat.SEEDS))],
            "changed_choices_per_seed_of_455": [int(np.sum(secondary_choices[arm][si] != baseline_choices[None, :])) for si in range(len(stat.SEEDS))],
            "minus2_per_seed_of_455": [int(np.sum(secondary[arm][si] <= -2.0 + 1e-12)) for si in range(len(stat.SEEDS))],
        }
    report = {"status": "COMPLETE", "baseline_gate": gate, "optimizer_updates": 0,
              "dope_composite_sha256": rt.DOPE, "decisions": decisions,
              "summary": summary,
              "train_oof_utility": {a: v.tolist() for a, v in results.items()},
              "train_oof_choices": {a: v.tolist() for a, v in choices_oof.items()},
              "dlval_selected_policy_utility": {a: v.tolist() for a, v in secondary.items()},
              "dlval_selected_policy_choices": {a: v.tolist() for a, v in secondary_choices.items()},
              "baseline_train_utility": base_train.tolist(), "baseline_val_utility": baseline.tolist(),
              "baseline_val_choices": baseline_choices.tolist(), "baseline_train_choices": base_train_choices.tolist()}
    (out / "panels.json").write_text(json.dumps(report, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
