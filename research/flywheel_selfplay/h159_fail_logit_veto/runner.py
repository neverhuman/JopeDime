"""Single H159 zero-update CPU execution: TRAIN-calibrated fail-logit veto at selection time on frozen r882."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

import runtime as rt
import science as stat

HERE = Path(__file__).resolve().parent


def verify(pins):
    for kind in ("inputs", "assets"):
        for p, h in pins[kind].items():
            if rt.sha(p) != h:
                raise RuntimeError("identity mismatch: " + p)
    for p, h in pins["executor_sha256"].items():
        if rt.sha(HERE / p) != h:
            raise RuntimeError("executor identity mismatch: " + p)
    if rt.sha(HERE / "PREREG.md") != pins["prereg_sha256"]:
        raise RuntimeError("PREREG.md sha256 mismatch")
    if rt.sha(HERE / "fold_manifest.json") != pins["fold_manifest_sha256"]:
        raise RuntimeError("fold_manifest.json sha256 mismatch")
    for name, version in pins["packages"].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError("package version mismatch: " + name)


def selected():
    packet = json.loads((rt.INPUT / "selected.json").read_text())
    train = packet["train"]
    val = packet["dev"]
    if len(train) != 1219 or len(val) != 91:
        raise RuntimeError("incomplete cohort")
    for rows, split in [(train, "DL-Train"), (val, "DL-Val")]:
        if len({r["context_sha256"] for r in rows}) != len(rows):
            raise RuntimeError("duplicate context")
        if any(
            r["dl_split"] != split
            or r["task"] != "regression"
            or r.get("has_meta_test") is True
            or r.get("source_split") not in ("train", "inner")
            for r in rows
        ):
            raise RuntimeError("protected row")
    if {r["family_id"] for r in train} & {r["family_id"] for r in val}:
        raise RuntimeError("family overlap between DL-Train and DL-Val")
    return train, val


def load_pools(rows):
    pools = {}
    for r in rows:
        cid = r["context_sha256"]
        p = rt.INPUT / "pools" / f"{cid}.npz"
        if rt.sha(p) != r["pool_sha256"]:
            raise RuntimeError("pool sha mismatch: " + cid)
        with np.load(p) as z:
            pools[cid] = dict(
                program=z["cand_prog_feat412"].astype(np.float32),
                proj16=z["cand_proj16"].astype(np.float32),
                fitness=z["cand_fitness_f"].astype(np.float32),
                gate=z["cand_gate"].astype(bool),
            )
    return pools


def load_tokens(rows):
    import torch

    with np.load(rt.CACHE) as z:
        ids = list(z["context_ids"])
        ix = {str(c): i for i, c in enumerate(ids)}
        toks = z["mm_tokens_10x32"].astype(np.float32)
        masks = z["mm_mask_10"].astype(np.float32)
    t_out, m_out = {}, {}
    for r in rows:
        c = r["context_sha256"]
        i = ix[c]
        t_out[c] = torch.from_numpy(toks[i])
        m_out[c] = torch.from_numpy(masks[i])
    return t_out, m_out


def predict_batch(net, pools, tokens, masks, cids, size):
    import torch

    progs = np.stack([pools[c]["program"] for c in cids])
    proj = np.stack([pools[c]["proj16"] for c in cids])
    b, k = progs.shape[:2]
    if k != size:
        raise RuntimeError("candidate size mismatch")
    out = net.adapter_10tok.predict_costs(
        ctx_feat=torch.zeros((b, 584)),
        prog_412=torch.as_tensor(progs),
        proj_16=torch.as_tensor(proj),
        base_prior_cost=torch.zeros((b, k)),
        scale=torch.full((b,), 0.4208),
        sos_prior=torch.zeros((b, k, 2)),
        mm_tokens_10x32=torch.stack([tokens[c] for c in cids]),
        mm_mask_10=torch.stack([masks[c] for c in cids]),
        revealed_indices=torch.arange(min(4, k)).reshape(1, -1).expand(b, -1),
        revealed_raw_f=torch.as_tensor(
            np.stack([pools[c]["fitness"][: min(4, k)] for c in cids]),
            dtype=torch.float32,
        ),
        revealed_gate=torch.as_tensor(
            np.stack([pools[c]["gate"][: min(4, k)] for c in cids]),
            dtype=torch.float32,
        ),
    )
    total_cost = out["total_cost"].detach().cpu().float().numpy()
    fail_logit = out["fail_logit"].detach().cpu().float().numpy()
    if not np.isfinite(total_cost).all() or not np.isfinite(fail_logit).all():
        raise RuntimeError("nonfinite total_cost or fail_logit predictions")
    return total_cost, fail_logit


def score_panel(net, pools, tokens, masks, rows):
    import torch

    groups = {}
    for r in rows:
        cid = r["context_sha256"]
        groups.setdefault(len(pools[cid]["fitness"]), []).append(cid)
    by_cid = {}
    with torch.no_grad():
        for size, cids in sorted(groups.items()):
            for start in range(0, len(cids), 8):
                batch = cids[start : start + 8]
                costs, logits = predict_batch(net, pools, tokens, masks, batch, size)
                for cid, cost_vec, logit_vec in zip(batch, costs, logits, strict=True):
                    by_cid[cid] = (cost_vec, logit_vec)
    records = []
    for r in rows:
        cid = r["context_sha256"]
        cost_vec, logit_vec = by_cid[cid]
        p = pools[cid]
        records.append(
            dict(
                context_sha256=cid,
                family_id=r["family_id"],
                total_cost=cost_vec,
                fail_logit=logit_vec,
                fitness=p["fitness"],
                gate=p["gate"],
            )
        )
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=False, exist_ok=False)
    rt.guards(out)
    pins = json.loads((HERE / "manifest.json").read_text())
    verify(pins)

    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    train, val = selected()
    if len({r["family_id"] for r in train}) != 205:
        raise RuntimeError("TRAIN family inventory mismatch")
    groups = stat.folds(train)
    if groups != json.loads((HERE / "fold_manifest.json").read_text())["families"]:
        raise RuntimeError("fold_manifest identity mismatch")

    launch_path = (out.parent / "launch.json").resolve()
    allowed = set(pins["inputs"])
    allowed.add(str(launch_path))
    allowed.update(
        str(rt.INPUT / "pools" / f"{r['context_sha256']}.npz") for r in train + val
    )
    audit = rt.Audit(allowed, out)
    sys.addaudithook(audit.check)
    started = time.monotonic()
    scored_contexts = 0
    try:
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA forbidden")
        ckpt_path = "/home/ubuntu/jope-data/h159/inputs/round-882.pt"
        ref_path = "/home/ubuntu/jope-data/h159/inputs/best_dev_checkpoint.json"
        net, guard, names, frozen, adapter_hash, _ = rt.build(ckpt_path)
        census = rt.dropout_census(net)
        if (
            census["adapter_dropout_modules"]
            or any(m["p"] != 0 for m in census["adapter_attention_dropout"])
            or census["effective_encoder"] != "_sync_free_encode_context_and_candidates"
            or census["adapter_training"]
            or census["dope_training"]
        ):
            raise RuntimeError("effective inference/training mode changed")
        rt.check_frozen(net, guard, names, frozen, adapter_hash)

        if args.smoke:
            rt.save(
                out / "smoke.json",
                dict(
                    status="PASS",
                    optimizer_updates=0,
                    scored_contexts=0,
                    adapter_tensors=len(names),
                    adapter_parameters=sum(net.state_dict()[n].numel() for n in names),
                    trainable_parameters=sum(
                        p.numel() for p in net.parameters() if p.requires_grad
                    ),
                    census=census,
                    frozen_sha256=frozen,
                    adapter_sha256=adapter_hash,
                    dope_sha256=rt.DOPE,
                    packages=pins["packages"],
                ),
            )
            return

        if not launch_path.exists():
            raise RuntimeError("launch.json required before scored run")
        launch = json.loads(launch_path.read_text())
        if launch.get("manifest_sha256") != rt.sha(
            HERE / "manifest.json"
        ) or not launch.get("merged_source_verified"):
            raise RuntimeError("merged preregistration verification required")

        ref = json.loads(Path(ref_path).read_text())
        contract = ref["evaluation_contract"]
        expected = [
            dict(
                cid=r["context_sha256"],
                pool_sha256=r["pool_sha256"],
                training_table_sha256=r["train_only_jtt1_sha256"],
            )
            for r in val
        ]
        if (
            contract["contexts"] != expected
            or hashlib.sha256(
                json.dumps(
                    contract["contexts"], sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
            != contract["contexts_sha256"]
        ):
            raise RuntimeError("DL-Val contract mismatch")

        all_rows = train + val
        pools = load_pools(all_rows)
        tokens, masks = load_tokens(all_rows)

        # 1. Score 91-ctx DL-Val and verify exact R882 regression gate first
        val_records = score_panel(net, pools, tokens, masks, val)
        scored_contexts += len(val_records)
        r882_val_eval = stat.evaluate_threshold_on_records(val_records, None)
        ref_u = [ref["per_context"][r["context_sha256"]]["utility"] for r in val]
        err = np.abs(np.asarray(r882_val_eval["utilities"]) - np.asarray(ref_u))
        sanity = dict(
            matches=int(np.sum(err == 0.0)),
            max_context_error=float(err.max()),
            mean_error=float(
                abs(r882_val_eval["context_mean_utility"] - ref["mean_utility"])
            ),
            mean=float(r882_val_eval["context_mean_utility"]),
            equal_family_mean=float(r882_val_eval["equal_family_mean_utility"]),
            invalid_picks=int(r882_val_eval["invalid_picks"]),
            status="PASS" if float(err.max()) == 0.0 else "FAIL",
        )
        rt.save(out / "baseline.json", sanity)
        if sanity["status"] != "PASS" or sanity["matches"] != 91:
            raise RuntimeError("R882 91/91 baseline regression gate failed")
        rt.guards(out)

        # 2. Score 1,219-ctx DL-Train
        train_records = score_panel(net, pools, tokens, masks, train)
        scored_contexts += len(train_records)
        rt.check_frozen(net, guard, names, frozen, adapter_hash)
        rt.guards(out)

        # 3. Run H159 analysis across R882, VETO_FIX, VETO_CV, and ORACLE_INFO
        analysis = stat.analyze_h159(train_records, val_records)
        metrics = dict(
            wave="H159",
            status="COMPLETE",
            verdict=analysis["verdict"],
            r882_verdict=analysis["r882_verdict"],
            rule_neutral=analysis["rule_neutral"],
            rule_neutral_across_entire_grid=analysis["rule_neutral_across_entire_grid"],
            optimizer_updates=0,
            scored_contexts=scored_contexts,
            baseline=sanity,
            primary_gates=analysis["primary_gates"],
            r882_baseline=analysis["r882_baseline"],
            primary_veto_fix_dl_val=analysis["primary_veto_fix_dl_val"],
            secondary_veto_cv_dl_train=analysis["secondary_veto_cv_dl_train"],
            grid_profile=analysis["grid_profile"],
            oracle_info=analysis["oracle_info"],
            seeds=list(stat.SEEDS),
            fold_seed=stat.FOLD_SEED,
            bootstrap_seed=stat.BOOT_SEED,
            grid_t=list(stat.GRID_T),
            source_commit=launch.get("source_commit"),
            manifest_sha256=rt.sha(HERE / "manifest.json"),
            dope_sha256=rt.DOPE,
            frozen_sha256=frozen,
            adapter_sha256=adapter_hash,
            historical_dl_val_exposure=True,
            serving_promotion=False,
            wall_seconds=time.monotonic() - started,
        )
        np.savez_compressed(
            out / "panels.npz",
            val_context_ids=np.array([r["context_sha256"] for r in val_records]),
            train_context_ids=np.array([r["context_sha256"] for r in train_records]),
            val_r882_utilities=np.asarray(r882_val_eval["utilities"], dtype=np.float64),
            val_r882_choices=np.asarray(r882_val_eval["choices"], dtype=np.int32),
            val_winner_fail_probs=np.asarray(
                [
                    float(
                        stat.sigmoid(r["fail_logit"])[int(np.argmin(r["total_cost"]))]
                    )
                    for r in val_records
                ],
                dtype=np.float64,
            ),
            train_winner_fail_probs=np.asarray(
                [
                    float(
                        stat.sigmoid(r["fail_logit"])[int(np.argmin(r["total_cost"]))]
                    )
                    for r in train_records
                ],
                dtype=np.float64,
            ),
        )
        verify(pins)
        rt.save(out / "metrics.json", metrics)
    finally:
        rt.save(
            out / "audit.json",
            dict(
                **audit.receipt(),
                optimizer_updates=0,
                scored_contexts=scored_contexts,
                wall_seconds=time.monotonic() - started,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024,
                threads_at_end=len(list(Path("/proc/self/task").iterdir())),
                cuda_initialized=torch.cuda.is_initialized(),
            ),
        )


if __name__ == "__main__":
    main()
