"""H157 executor: H142 TR_LR candidate generation with strict inner selection."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import random
import resource
import subprocess
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["RAYON_NUM_THREADS"] = "1"

import numpy as np

import science

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "h146_lookahead_exec"
H142_RUNNER = Path("/home/ubuntu/jope-rescue-20260921/h142-trust-region-listwise/run_h142_cpu.py")
H142_RUNNER_SHA = "d5215234c2659d2a470ac95c0900827bfad79704b7fb5ac7fe952cdb633d4a8e"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def guard_resources(output):
    import psutil

    rss = psutil.Process().memory_info().rss
    threads = len(list(Path("/proc/self/task").iterdir()))
    fs = os.statvfs(output)
    free = fs.f_bavail * fs.f_frsize
    disk = os.statvfs("/")
    used_pct = 100.0 * (disk.f_blocks - disk.f_bfree) / disk.f_blocks
    load = os.getloadavg()[0]
    available = psutil.virtual_memory().available
    if rss >= 6_000_000_000 or threads > 4 or free < 40_000_000_000:
        raise RuntimeError("H157 hard RSS/thread/free-disk guard")
    if used_pct >= 95.0 or load > 85.0 or available < 12 * 1024**3:
        raise RuntimeError("H157 host load/disk/memory guard")
    return dict(rss_bytes=rss, threads=threads, free_bytes=free,
                disk_used_percent=used_pct, load1=load, mem_available_bytes=available)


def import_parent():
    sys.path.insert(0, str(PARENT))
    try:
        spec = importlib.util.spec_from_file_location("h157_parent_runner", PARENT / "runner.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        runtime = sys.modules["runtime"]
    finally:
        sys.path.pop(0)
    return module, runtime


def verify(pins):
    for path, expected in pins["inputs"].items():
        if sha(path) != expected:
            raise RuntimeError("input identity mismatch: " + path)
    for path, expected in pins["assets"].items():
        if sha(path) != expected:
            raise RuntimeError("asset identity mismatch: " + path)
    for name, version in pins["packages"].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError("package version mismatch: " + name)
    for name, expected in pins["source_sha256"].items():
        if sha(HERE / name) != expected:
            raise RuntimeError("source identity mismatch: " + name)
    for name, expected in pins["parent_source_sha256"].items():
        if sha(PARENT / name) != expected:
            raise RuntimeError("H146 parent source identity mismatch: " + name)
    for name, expected in pins["reference_source_sha256"].items():
        if sha((HERE / name).resolve()) != expected:
            raise RuntimeError("H142/H148 reference source mismatch: " + name)
    if (str(H142_RUNNER) != pins["h142_runner_reference"]
            or H142_RUNNER_SHA != pins["h142_runner_reference_sha256"]
            or sha(H142_RUNNER) != H142_RUNNER_SHA):
        raise RuntimeError("pinned H142 runner source mismatch")


def family_means(rows, values):
    return np.asarray(
        [
            np.asarray(values, dtype=np.float64)[
                np.asarray([r["family_id"] == family for r in rows])
            ].mean()
            for family in sorted({r["family_id"] for r in rows})
        ],
        dtype=np.float64,
    )


def h142_loss(prediction, fitness, gate):
    import torch
    import torch.nn.functional as F

    targets = torch.where(gate, fitness.clamp(-4.0, 2.0), torch.full_like(fitness, 2.0))
    smooth = F.smooth_l1_loss(prediction, targets, reduction="none", beta=1.0).mean(dim=1)
    kl = F.kl_div(
        F.log_softmax(-prediction / 0.2, dim=-1),
        F.softmax(-targets / 0.2, dim=-1),
        reduction="none",
    ).sum(dim=1)
    return (smooth + kl).mean()


def training_plan(fit_rows, pools, seed):
    """H142 20x2 effective updates with H157 family-bootstrap resampling."""
    families = sorted({r["family_id"] for r in fit_rows})
    if len(families) != 164:
        raise ValueError("H157 fit family count")
    by_family = {
        family: tuple(r["context_sha256"] for r in fit_rows if r["family_id"] == family)
        for family in families
    }
    family_rng = np.random.default_rng(seed + 157140)
    sampled = family_rng.choice(families, size=len(families), replace=True)
    expanded = tuple(cid for family in sampled for cid in by_family[str(family)])
    if not expanded:
        raise ValueError("empty family-bootstrap roster")
    plan = []
    sampler = seed * 1_000_003 + 10_007
    for rnd in range(1, 21):
        round_seed = int((sampler + rnd * 104729) % 2**32)
        np.random.seed(round_seed)
        rot = rnd % len(expanded)
        rotated = expanded[rot:] + expanded[:rot]
        for step in range(8):
            stream = int.from_bytes(
                hashlib.sha256(f"flywheel-batch/v1:{sampler}:{rnd}:{step}".encode()).digest()[:8],
                "little",
            )
            cids = tuple(random.Random(stream).choices(rotated, k=16))
            indexes = []
            for cid in cids:
                order = np.argsort(pools[cid]["fitness"])
                k = min(48, 192, len(order))
                rest = order[k:] if len(order) > k else order
                extra = np.random.choice(rest, 192 - k, replace=len(rest) < 192 - k)
                indexes.append(np.concatenate([order[:k], extra]).astype(np.int64))
            if step < 2:
                plan.append((stream, cids, tuple(indexes)))
    if len(plan) != 40:
        raise RuntimeError("H142 plan must contain 40 effective updates")
    return plan


def validate_launch(launch, head, manifest_sha256):
    if (
        launch.get("source_commit") != head
        or launch.get("manifest_sha256") != manifest_sha256
        or launch.get("merged_source_verified") is not True
        or launch.get("cross_review_pass") is not True
        or launch.get("reviewer") != "luna_v1_rnk11"
        or not launch.get("review_pr")
        or not launch.get("reviewed_source_commit")
        or not launch.get("merged_pr")
    ):
        raise RuntimeError("merged source and luna_v1_rnk11 exact-head cross-review required")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--launch", type=Path)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if os.uname().nodename != "xbabe2":
        raise RuntimeError("H157 is authorized only on xbabe2")
    if os.getpriority(os.PRIO_PROCESS, 0) < 19:
        raise RuntimeError("launch with nice -n 19")
    out = args.output.resolve()
    if not out.is_relative_to(Path("/home/ubuntu/jope-data/h157")):
        raise RuntimeError("H157 outputs must remain in external jope-data custody")
    out.mkdir()  # Exclusive attempt directory; no resume or automatic retry.
    pins = json.loads((HERE / "manifest.json").read_text())
    verify(pins)
    parent, rt = import_parent()
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if torch.cuda.is_initialized() or torch.cuda.is_available():
        raise RuntimeError("CUDA must remain unavailable")
    guard_resources(out)
    net, guard, names, frozen, fused = rt.build(pins["checkpoint"])
    if args.smoke:
        save(
            out / "smoke.json",
            dict(status="PASS", scored_contexts=0, optimizer_updates=0,
                 trainable_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad),
                 adapter_tensors=len(names), dope_sha256=rt.DOPE, frozen_sha256=frozen),
        )
        return
    if not args.launch:
        raise RuntimeError("merged-source launch receipt required")
    launch_path = args.launch.resolve()
    if not launch_path.is_relative_to(Path("/home/ubuntu/jope_plan/evidence/H157")):
        raise RuntimeError("launch receipt must remain in external H157 evidence custody")
    launch = json.loads(launch_path.read_text())
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
    validate_launch(launch, head, sha(HERE / "manifest.json"))
    sealed_manifest = (HERE / pins["h150_sealed_identity_manifest"]).resolve()
    if sha(sealed_manifest) != pins["h150_sealed_identity_sha256"]:
        raise RuntimeError("sealed identity-only deny-list mismatch")
    h150 = json.loads(sealed_manifest.read_text())
    packet_path = Path(pins["inputs"]["/home/ubuntu/jope-data/h50/input/selected.json"])
    packet = json.loads(packet_path.read_text())
    source_manifest = json.loads(Path(pins["inputs"]["/home/ubuntu/jope-data/h50/input/source_manifest.json"]).read_text())
    if (source_manifest.get("source_manifest_sha256") != "3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3"
            or source_manifest.get("excluded_dl_holdout_count") != 93
            or source_manifest.get("excluded_payloads_opened") != 0
            or source_manifest.get("outer_rows_delivered") != 0):
        raise RuntimeError("H50 TRAIN source/holdout custody contract mismatch")
    train, val = packet["train"], packet["dev"]
    if len(train) != 1219 or len(val) != 91:
        raise RuntimeError("selected TRAIN/VAL roster size mismatch")
    if {r["family_id"] for r in train} & set(h150["families"]):
        raise RuntimeError("TRAIN family overlaps known sealed identity deny-list")
    if {r["family_id"] for r in train} & {r["family_id"] for r in val}:
        raise RuntimeError("TRAIN/VAL family overlap")
    for rows, split in ((train, "DL-Train"), (val, "DL-Val")):
        if any(r.get("dl_split") != split or r.get("task") != "regression" or
               r.get("has_meta_test") is True or r.get("source_split") not in ("train", "inner")
               for r in rows):
            raise RuntimeError("row outside pinned regression split")
    if {r["family_id"] for r in train} != set(sum(json.loads((HERE / "fold_manifest.json").read_text())["groups"], [])):
        raise RuntimeError("fold manifest family closure mismatch")
    manifest_folds = science.folds(r["family_id"] for r in train)
    fold_doc = json.loads((HERE / "fold_manifest.json").read_text())
    if manifest_folds != fold_doc["groups"]:
        raise RuntimeError("deterministic H157 fold assignment mismatch")
    inner_families, fit_families = set(manifest_folds[0]), set(sum(manifest_folds[1:], []))
    fit_rows = [r for r in train if r["family_id"] in fit_families]
    inner_rows = [r for r in train if r["family_id"] in inner_families]
    if len({r["family_id"] for r in inner_rows}) != 41 or len({r["family_id"] for r in fit_rows}) != 164:
        raise RuntimeError("family partition counts")

    allowed = set(pins["inputs"])
    allowed.update(str(rt.INPUT / "pools" / f"{r['context_sha256']}.npz") for r in train + val)
    audit = rt.Audit(allowed, out)
    sys.addaudithook(audit.check)
    started = time.monotonic()
    updates = 0
    try:
        resource_log = (out / "resources.jsonl").open("x")
        def sample_resources():
            sample = guard_resources(out)
            sample["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            resource_log.write(json.dumps(sample, sort_keys=True) + "\n")
            resource_log.flush()
        sample_resources()
        # Close the exact unchanged-r882 regression gate before any update.
        pools, tokens, masks = load_features(parent, rt, fused, train, val, torch)
        def evaluate(rows):
            sample_resources()
            result = parent.evaluate(net, pools, tokens, masks, rows)
            rt.check_frozen(net, guard, names, frozen)
            sample_resources()
            return result

        census = rt.dropout_census(net)
        if census["adapter_training"] or census["dope_training"] or census["adapter_dropout_modules"]:
            raise RuntimeError("unexpected effective model mode or dropout")
        eval_base = evaluate(val)
        ref = json.loads(Path(pins["reference"]).read_text())
        contract = ref["evaluation_contract"]
        if (
            hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
            != pins["evaluation_identity"]
            or [c["cid"] for c in contract["contexts"]]
            != [r["context_sha256"] for r in val]
        ):
            raise RuntimeError("DL-Val evaluation identity mismatch")
        expected = np.asarray(ref["utility"]["per_context"], dtype=np.float64)
        actual = np.asarray(eval_base["values"], dtype=np.float64)
        if actual.shape != (91,) or not np.isfinite(actual).all() or np.max(np.abs(actual - expected)) > 1e-7:
            raise RuntimeError("r882 91/91 native regression gate failed")
        if abs(float(actual.mean()) - 1.1377941898164525) > 1e-7:
            raise RuntimeError("r882 mean differs from frozen H157 reference")
        del net
        net, guard, names, frozen, fused = rt.build(pins["checkpoint"])
        def evaluate(rows):
            sample_resources()
            result = checked_evaluate(parent, rt, net, guard, names, frozen, pools, tokens, masks, rows)
            sample_resources()
            return result
        baseline_b = evaluate(val)
        if not np.array_equal(np.asarray(baseline_b["values"]), actual):
            raise RuntimeError("independent r882 copies are not exactly equal")
        rt.check_frozen(net, guard, names, frozen)
        save(out / "baseline.json", dict(status="PASS", context_count=91, family_count=19,
             mean=float(actual.mean()), per_context=actual.tolist(),
             two_copy_max_abs_diff=float(np.max(np.abs(np.asarray(baseline_b["values"]) - actual))),
             reference_max_abs_diff=float(np.max(np.abs(actual - expected))),
             dope_sha256=rt.DOPE, frozen_sha256=frozen))
        initial = {k: v.detach().clone() for k, v in net.state_dict().items()}
        parameters = [p for p in net.parameters() if p.requires_grad]
        initial_vec = torch.cat([initial[n].reshape(-1) for n in names])
        family_ids = [r["family_id"] for r in inner_rows]
        val_baseline = eval_base["values"]
        inner_baseline = evaluate(inner_rows)["values"]
        inner_base_family = family_means(inner_rows, inner_baseline)
        plan_receipts, chosen_states = [], {}

        for seed_index, seed in enumerate(science.SEEDS):
            arm_outputs = {}
            for candidate_index in range(9):
                arm = "TR1" if candidate_index == 0 else f"bok-{candidate_index:02d}"
                sampler_seed = seed * 100 + candidate_index
                net.load_state_dict(initial, strict=True)
                rt.check_frozen(net, guard, names, frozen)
                optimizer = torch.optim.AdamW(parameters, lr=1e-5, weight_decay=1e-4)
                initial_adapter = torch.cat([dict(net.named_parameters())[n].detach().reshape(-1) for n in names])
                losses, batches = [], []
                plan = training_plan(fit_rows, pools, sampler_seed)
                allowed_fit = {r["context_sha256"] for r in fit_rows}
                for step, (stream, cids, indexes) in enumerate(plan, 1):
                    sample_resources()
                    if not set(cids) <= allowed_fit:
                        raise RuntimeError("fit family/context leakage")
                    torch.manual_seed(stream)
                    fitness = torch.as_tensor(np.stack([pools[c]["fitness"][ix] for c, ix in zip(cids, indexes, strict=True)]), dtype=torch.float32)
                    gate = torch.as_tensor(np.stack([pools[c]["gate"][ix] for c, ix in zip(cids, indexes, strict=True)]), dtype=torch.bool)
                    optimizer.zero_grad(set_to_none=True)
                    prediction = parent.predict(net, pools, tokens, masks, cids, indexes)
                    objective = h142_loss(prediction, fitness, gate)
                    if not torch.isfinite(objective):
                        raise RuntimeError("nonfinite H142 objective")
                    objective.backward()
                    torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    updates += 1
                    rt.check_frozen(net, guard, names, frozen)
                    sample_resources()
                    losses.append(float(objective.detach()))
                    batches.append(dict(step=step, stream=stream, context_ids=list(cids), indices_sha256=hashlib.sha256(np.asarray(indexes, dtype="<i8").tobytes()).hexdigest()))
                final_vec = torch.cat([dict(net.named_parameters())[n].detach().reshape(-1) for n in names])
                drift = float(torch.linalg.vector_norm(final_vec - initial_adapter))
                linf = float(torch.max(torch.abs(final_vec - initial_adapter)))
                if drift <= 0:
                    raise RuntimeError("candidate did not update adapter parameters")
                inner_result = evaluate(inner_rows)
                delta = np.asarray(inner_result["values"]) - np.asarray(inner_baseline)
                inner_deltas = family_means(inner_rows, delta)
                state = {n: net.state_dict()[n].detach().clone() for n in names}
                state_sha = rt.tensor_hash(state)
                candidate_file = out / f"seed{seed}-{arm}.npz"
                np.savez_compressed(candidate_file, **{n: t.cpu().numpy() for n, t in state.items()})
                arm_outputs[arm] = dict(
                    family_deltas=inner_deltas,
                    state=state,
                    state_sha256=state_sha,
                    checkpoint_sha256=sha(candidate_file),
                    drift_l2=drift,
                    drift_linf=linf,
                    losses=losses,
                    batches=batches,
                )
                if candidate_index == 0:
                    tr1_score = science.bootstrap_median(inner_deltas, 157141 + 100 * seed_index)
                    tr1_admit = science.admits(tr1_score)
                    tr1_choice = "TR1" if tr1_admit else "R882"
                print(f"CANDIDATE seed={seed} id={arm} updates=40 drift_l2={drift:.8f} inner_median={float(np.median(inner_deltas)):.8f}", flush=True)
            bok_candidates = {f"bok-{i:02d}": arm_outputs[f"bok-{i:02d}"]["family_deltas"] for i in range(1, 9)}
            bok_decision = science.choose_winner(bok_candidates, 157140 + 100 * seed_index)
            for candidate_id, item in arm_outputs.items():
                score = tr1_score if candidate_id == "TR1" else bok_decision["candidates"][candidate_id]
                candidate_receipt = out / f"seed{seed}-{candidate_id}.json"
                candidate_index = 0 if candidate_id == "TR1" else int(candidate_id[-2:])
                save(candidate_receipt, dict(
                    candidate_id=candidate_id, seed=seed,
                    sampler_seed=seed * 100 + candidate_index, updates=40,
                    inner_family_gains=item["family_deltas"].tolist(), inner_bootstrap=score,
                    adapter_state_sha256=item["state_sha256"],
                    checkpoint_sha256=item["checkpoint_sha256"],
                    adapter_l2_drift=item["drift_l2"], adapter_linf_drift=item["drift_linf"],
                    losses=item["losses"], batches=item["batches"],
                ))
            lock = dict(seed=seed, tr1=dict(score=tr1_score, selected=tr1_choice), bok8=bok_decision,
                        checkpoint_hashes={k: v["checkpoint_sha256"] for k, v in arm_outputs.items()},
                        adapter_state_hashes={k: v["state_sha256"] for k, v in arm_outputs.items()})
            lock_path = out / f"seed{seed}-selection.json"
            save(lock_path, lock)
            selected = {"TR1": tr1_choice, "BOK8": bok_decision["selected"]}
            for treatment, candidate in selected.items():
                chosen_states[(seed, treatment)] = None if candidate == "R882" else arm_outputs[candidate]["state"]
            plan_receipts.append(dict(
                lock=lock, lock_sha256=sha(lock_path),
                candidate_summaries={k: dict(inner_median=float(np.median(v["family_deltas"])),
                    drift_l2=v["drift_l2"], drift_linf=v["drift_linf"],
                    checkpoint_sha256=v["checkpoint_sha256"], adapter_state_sha256=v["state_sha256"])
                    for k, v in arm_outputs.items()},
            ))

        # All three seed choices and hashes are durable before any treatment DL-Val read.
        train_baseline = evaluate(train)["values"]
        val_panels = {"R882": [], "TR1": [], "BOK8": []}
        train_panels = {"R882": [], "TR1": [], "BOK8": []}
        for seed in science.SEEDS:
            for treatment in ("TR1", "BOK8"):
                net.load_state_dict(initial, strict=True)
                candidate = chosen_states[(seed, treatment)]
                if candidate is not None:
                    with torch.no_grad():
                        for name, tensor in candidate.items():
                            dict(net.named_parameters())[name].copy_(tensor)
                rt.check_frozen(net, guard, names, frozen)
                val_panels[treatment].append(evaluate(val)["values"])
                train_panels[treatment].append(evaluate(train)["values"])
            val_panels["R882"].append(val_baseline)
            train_panels["R882"].append(train_baseline)

        val_families = [r["family_id"] for r in val]
        family_panels = {}
        for arm, panels in val_panels.items():
            family_panels[arm] = np.stack([family_means(val, panel) for panel in panels])
        primary = science.primary_summary(family_panels["R882"], {"BOK8": family_panels["BOK8"], "TR1": family_panels["TR1"]})
        bok = primary["treatments"]["BOK8"]
        verdict = "ADVANCE" if bok["median_gain"] > 0.001 and bok["holm_lcb95"] > 0 else "RULE_NEUTRAL"
        elo = {}
        win_panels = []
        for si, seed in enumerate(science.SEEDS):
            values = np.stack([np.asarray(val_panels[a][si], dtype=np.float64) for a in ("R882", "TR1", "BOK8")], axis=1)
            fw = science.family_wins(values, val_families)
            win_panels.append(fw)
            elo[str(seed)] = science.bt_ratings(fw.sum(axis=0), anchor=0).tolist()
        win_panels = np.asarray(win_panels, dtype=np.float64)
        elo_draw_ix = np.random.default_rng(157143).integers(0, 19, (1000, 19))
        elo_draws = science.bt_ratings(win_panels[:, elo_draw_ix].sum(axis=2), anchor=0)
        elo_draws = np.median(elo_draws, axis=0)
        train_elo = {}
        for si, seed in enumerate(science.SEEDS):
            values = np.stack([np.asarray(train_panels[a][si], dtype=np.float64)
                               for a in ("R882", "TR1", "BOK8")], axis=1)
            family_wins = science.family_wins(values, [r["family_id"] for r in train])
            train_elo[str(seed)] = science.bt_ratings(family_wins.sum(axis=0), anchor=0).tolist()
        train_family_means = {
            arm: [float(family_means(train, panel).mean()) for panel in panels]
            for arm, panels in train_panels.items()
        }
        metrics = dict(
            wave="H157", status="COMPLETE", verdict=verdict, optimizer_updates=updates,
            baseline=dict(context_mean=float(np.mean(val_baseline)), context_count=91,
                          family_count=19, expected_mean=1.1377941898164525, match=True),
            primary=primary, elo_by_seed=elo, elo_arms=["R882", "TR1", "BOK8"],
            median_elo=np.median(np.asarray(list(elo.values())), axis=0).tolist(),
            elo_bootstrap_ci95=np.quantile(elo_draws, [0.025, 0.975], axis=0).T.tolist(),
            descriptive_train=dict(contexts=1219, families=205, historical_r882_train_exposure=True,
                                   equal_family_mean_by_seed=train_family_means,
                                   equal_family_gain_vs_r882_by_seed={arm: [x - y for x, y in zip(means, train_family_means["R882"], strict=True)]
                                       for arm, means in train_family_means.items() if arm != "R882"},
                                   elo_by_seed=train_elo,
                                   median_elo=np.median(np.asarray(list(train_elo.values())), axis=0).tolist()),
            selections=plan_receipts, source_commit=head, manifest_sha256=sha(HERE / "manifest.json"),
            dope_sha256=rt.DOPE, frozen_sha256=frozen, serving_promotion=False,
        )
        if updates != 1080:
            raise RuntimeError("H157 update count mismatch")
        verify(pins)
        save(out / "metrics.json", metrics)
    finally:
        if "resource_log" in locals() and not resource_log.closed:
            resource_log.close()
        save(out / "audit.json", dict(**audit.receipt(), optimizer_updates=updates,
             elapsed_seconds=time.monotonic()-started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
             thread_count=len(list(Path("/proc/self/task").iterdir())), cuda_initialized=torch.cuda.is_initialized()))


def load_features(parent, rt, fused, train, val, torch):
    import hashlib

    rows = train + val
    pools = {}
    projection = fused.get_w1_sign_matrix() / np.sqrt(412.0)
    for row in rows:
        cid = row["context_sha256"]
        path = rt.INPUT / "pools" / f"{cid}.npz"
        if rt.sha(path) != row["pool_sha256"]:
            raise RuntimeError("pool hash mismatch")
        with np.load(path, allow_pickle=False) as data:
            program = data["program"].astype(np.float32)
            pools[cid] = dict(program=program,
                              proj16=(program.astype(np.float64) @ projection).astype(np.float32),
                              fitness=data["fitness"].astype(np.float64),
                              gate=data["gate"].astype(bool))
    with np.load(rt.CACHE, allow_pickle=False) as data:
        ids = list(map(str, data["cids"]))
        if set(ids) != set(pools) or len(ids) != len(pools):
            raise RuntimeError("feature-cache closure mismatch")
        tokens_raw, masks_raw = data["tokens_10x32"], data["mask_10"]
        tokens = {cid: torch.as_tensor(tokens_raw[i], dtype=torch.float32) for i, cid in enumerate(ids)}
        masks = {cid: torch.as_tensor(masks_raw[i], dtype=torch.bool) for i, cid in enumerate(ids)}
    return pools, tokens, masks


def checked_evaluate(parent, rt, net, guard, names, frozen, pools, tokens, masks, rows):
    result = parent.evaluate(net, pools, tokens, masks, rows)
    rt.check_frozen(net, guard, names, frozen)
    return result


if __name__ == "__main__":
    main()
