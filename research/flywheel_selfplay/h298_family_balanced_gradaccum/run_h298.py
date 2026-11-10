"""H298 (h298_family_balanced_gradaccum): execute PAIRED_CONTROL, FAMBAL_ACC1, and FAMBAL_ACC4 on xbabe1."""

from __future__ import annotations

import copy
import datetime
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analysis_h298 as a298  # noqa: E402

ROOT = Path("/home/ubuntu/JopeDime/.agent/worktrees/flywheel-gp-fused")
if (ROOT / "research/flywheel_selfplay").exists():
    sys.path.insert(0, str(ROOT / "research/flywheel_selfplay"))

DATA_DIR = Path(os.environ.get("H298_DATA_DIR", "/home/ubuntu/jope-data/H298-rd_argon"))
DOPE_SHA = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
EXPECTED_CKPT_SHA = "1d6ea900ebed8d2a05c3b3f37576adc15aadd2b53c585bfd5cb71a103e9ff97b"
EXPECTED_EVAL_ID = "e79a7715c1531ccb8c4055ab96971101cd4034ec3a71d83bca6d25b0e604ca83"
EXPECTED_SELECTED_SHA = "981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1"
EXPECTED_PREREG_BATCHES_SHA = "51e93dff787ee5d38f12fc41c2fb4b8021929ed59926a708922c5340bd359f95"


def sha256_file(p: Path | str) -> str:
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def state_sha(values: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name in sorted(values):
        t = values[name].detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(t.dtype).encode())
        h.update(np.asarray(t.shape, dtype=np.int64).tobytes())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".new")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def log_event(events_path: Path, kind: str, **kw: Any) -> None:
    rec = {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "event": kind,
        **kw,
    }
    with events_path.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def check_host_guard(events_path: Path) -> None:
    mem_avail_gib = psutil.virtual_memory().available / (1024**3)
    if mem_avail_gib < 12.0:
        raise RuntimeError(f"MemAvailable {mem_avail_gib:.2f} GiB < 12.0 GiB floor; aborting")
    load1 = os.getloadavg()[0]
    if load1 > 85.0:
        log_event(events_path, "pause_load", load1=load1)
        while os.getloadavg()[0] > 80.0:
            time.sleep(5.0)
        log_event(events_path, "resume_load", load1=os.getloadavg()[0])


def run_experiment() -> dict[str, Any]:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)

    import run_fused_closed_loop as base
    from dev_incumbent import warm_restart_adapter, batch_seed
    from automatic_promotion import training_indices
    from jope._telemetry import FullFlywheelReporter

    events_path = DATA_DIR / "events_h298.jsonl"
    t0 = time.time()

    # Verify regression receipt and preregistered batches exist and match before any arm runs
    reg_path = DATA_DIR / "regression_receipt.json"
    assert reg_path.exists(), "Regression check must pass before running arms"
    reg = json.loads(reg_path.read_text())
    assert reg["status"] == "PASS"
    assert reg["r3175_replay"]["mean"] == a298.R3175_DEV91_MEAN
    assert reg["r3175_replay"]["selected_programs_match_91_of_91"] is True
    assert reg["t1845_control_184501_replay"]["round_10_mean"] == 1.1354387884
    assert reg["t1845_control_184501_replay"]["selected_programs_match_91_of_91"] is True

    batches_path = DATA_DIR / "preregistered_batches.json"
    assert sha256_file(batches_path) == EXPECTED_PREREG_BATCHES_SHA
    prereg_batches = json.loads(batches_path.read_text())

    assert sha256_file(DATA_DIR / "checkpoints/round-3175.pt") == EXPECTED_CKPT_SHA
    assert sha256_file(DATA_DIR / "input/selected.json") == EXPECTED_SELECTED_SHA
    sel = json.loads((DATA_DIR / "input/selected.json").read_text())
    assert len(sel["train"]) == 1219 and len(sel["dev"]) == 91

    contract = {
        "protocol": "fixed-dev-cost-argmin-negative-fitness/v1",
        "selected_packet_sha256": sha256_file(DATA_DIR / "input/selected.json"),
        "dope_sha256": DOPE_SHA,
        "source_sha256": sha256_file(DATA_DIR / "input/cpu_exact_fused.py"),
        "model_source_sha256": sha256_file("/home/ubuntu/JopeDimeMicro/python/model.py"),
        "adapter_source_sha256": sha256_file("/home/ubuntu/JopeDimeMicro/python/phase12_multimodal_model.py"),
        "contexts": [
            {
                "cid": r["context_sha256"],
                "pool_sha256": r["pool_sha256"],
                "training_table_sha256": r["train_only_jtt1_sha256"],
            }
            for r in sel["dev"]
        ],
    }
    eval_id = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    assert eval_id == EXPECTED_EVAL_ID, eval_id

    spec = importlib.util.spec_from_file_location("h298_copy_fused", DATA_DIR / "input/cpu_exact_fused.py")
    fused = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fused)
    net = fused.FusedJopeNetwork(device=torch.device("cpu"), adapter_variant="h40b")
    for p in net.parameters():
        p.requires_grad_(False)
    for n, p in net.adapter_10tok.named_parameters():
        if not n.startswith("anchor."):
            p.requires_grad_(True)
    base.guard.lock_dope_weights_and_eval_mode(net)
    ckpt = torch.load(DATA_DIR / "checkpoints/round-3175.pt", map_location="cpu", weights_only=True)
    net.load_state_dict(ckpt["state_dict"], strict=True)
    assert ckpt["dope_sha256"] == DOPE_SHA
    assert base.guard.snapshot_dope_weights(net).composite_sha256 == DOPE_SHA

    trainable = {n: p for n, p in net.named_parameters() if p.requires_grad}
    params = list(trainable.values())
    assert sum(p.numel() for p in params) == 29211
    baseline_state = {n: t.detach().clone() for n, t in net.state_dict().items()}
    frozen_fn = lambda: {n: t for n, t in net.state_dict().items() if n not in trainable}
    frozen_sha = state_sha(frozen_fn())

    tokens = torch.load(DATA_DIR / "tokens.pt", map_location="cpu", weights_only=True)
    w1_mat = base.evaluator.get_w1_sign_matrix() / np.sqrt(412)
    pools: dict[str, dict[str, Any]] = {}
    for r in sel["train"] + sel["dev"]:
        cid = r["context_sha256"]
        p_path = DATA_DIR / "input/pools" / f"{cid}.npz"
        assert sha256_file(p_path) == r["pool_sha256"]
        with np.load(p_path, allow_pickle=False) as z:
            prog = z["program"].astype(np.float32)
            pools[cid] = {
                "rec": r,
                "program": prog,
                "pids": [v.decode() if isinstance(v, bytes) else str(v) for v in z["program_sha256"].tolist()],
                "fitness": z["fitness"].astype(np.float64),
                "gate": z["gate"].astype(bool),
                "proj16": prog.astype(np.float64) @ w1_mat,
            }

    fam_ids = sorted({r["family_id"] for r in sel["train"]})
    fam_cids = [sorted(r["context_sha256"] for r in sel["train"] if r["family_id"] == f) for f in fam_ids]
    assert len(fam_ids) == 205 and sum(len(x) for x in fam_cids) == 1219
    dev_families = [r["family_id"] for r in sel["dev"]]
    assert len(set(dev_families)) == 19

    scorer_dir = DATA_DIR / "scorer"
    scorer_dir.mkdir(exist_ok=True)
    reporter = FullFlywheelReporter(DATA_DIR / "input/selected.json", scorer_dir, scorer_dir / "history.jsonl")

    def predict(cids: list[str], indexes: list[np.ndarray]) -> dict[str, torch.Tensor]:
        prog = np.stack([pools[c]["program"][ix] for c, ix in zip(cids, indexes)])
        proj = np.stack([pools[c]["proj16"][ix] for c, ix in zip(cids, indexes)])
        b, m = prog.shape[:2]
        histf = np.stack([pools[c]["fitness"][ix[:min(4, m)]] for c, ix in zip(cids, indexes)])
        histg = np.stack([pools[c]["gate"][ix[:min(4, m)]] for c, ix in zip(cids, indexes)])
        return net.adapter_10tok.predict_costs(
            ctx_feat=torch.zeros((b, 584)),
            prog_412=torch.as_tensor(prog),
            proj_16=torch.as_tensor(proj),
            base_prior_cost=torch.zeros((b, m)),
            scale=torch.full((b,), 0.4208),
            sos_prior=torch.zeros((b, m, 2)),
            mm_tokens_10x32=torch.stack([tokens[c]["token"] for c in cids]),
            mm_mask_10=torch.stack([tokens[c]["mask"] for c in cids]),
            revealed_indices=torch.arange(min(4, m)).reshape(1, -1).expand(b, -1),
            revealed_raw_f=torch.as_tensor(histf, dtype=torch.float32),
            revealed_gate=torch.as_tensor(histg, dtype=torch.float32),
        )

    def score_terminal_dev() -> dict[str, Any]:
        rows = []
        net.eval()
        for r in sel["dev"]:
            c = r["context_sha256"]
            p = pools[c]
            with torch.no_grad():
                out = predict([c], [np.arange(len(p["pids"]))])
                cost = out["total_cost"][0].float().numpy().copy()
                choice = int(np.argmin(cost))
            utility = -float(p["fitness"][choice]) if p["gate"][choice] else -2.0
            rows.append({"cid": c, "selected_program": p["pids"][choice], "choice": choice, "utility": utility})
        cs = [r["cid"] for r in rows]
        fit = [float(pools[r["cid"]]["fitness"][r["choice"]]) for r in rows]
        gate = [bool(pools[r["cid"]]["gate"][r["choice"]]) for r in rows]
        u = reporter.utility(cs, fit, gate, [r["selected_program"] for r in rows], "dev")
        return {
            "mean": float(u["mean"]),
            "python_sum_mean": float(sum(r["utility"] for r in rows) / len(rows)),
            "selected_programs": [r["selected_program"] for r in rows],
            "per_context": [float(r["utility"]) for r in rows],
            "canonical_utility": u,
        }

    r3175_base = json.loads((DATA_DIR / "TRIAL_BASELINE.json").read_text())
    assert r3175_base["mean"] == a298.R3175_DEV91_MEAN

    arms_dir = DATA_DIR / "arms"
    arms_dir.mkdir(exist_ok=True)
    arm_results: dict[str, dict[int, dict[str, Any]]] = {
        "PAIRED_CONTROL": {},
        "FAMBAL_ACC1": {},
        "FAMBAL_ACC4": {},
    }

    for seed in a298.REPLICATION_SEEDS:
        rnd = seed
        seed_prereg = prereg_batches[str(seed)]

        # Reconstruct and verify exact preregistered batches for this seed
        np.random.seed(seed)
        ctrl_batches = []
        for step in range(8):
            order_seed = batch_seed(seed, rnd, step)
            cids = list(reporter.training_order(order_seed, 8192, False))
            ix = np.stack([training_indices(pools[c], 192) for c in cids])
            exp = seed_prereg["PAIRED_CONTROL"][step]
            assert hashlib.sha256(json.dumps(cids).encode()).hexdigest() == exp["order_sha256"]
            assert hashlib.sha256(ix.tobytes()).hexdigest() == exp["slate_sha256"]
            ctrl_batches.append((cids, ix))

        acc1_batches = []
        acc4_batches = []
        for step in range(8):
            order_seed = batch_seed(seed, rnd, step)
            cids4 = a298.sample_fambal_cids(fam_ids, fam_cids, order_seed, 32768)
            cids1 = cids4[:8192]
            np.random.seed(order_seed & 0xFFFFFFFF)
            ix4 = np.stack([training_indices(pools[c], 192) for c in cids4])
            ix1 = ix4[:8192]
            exp1 = seed_prereg["FAMBAL_ACC1"][step]
            exp4 = seed_prereg["FAMBAL_ACC4"][step]
            assert hashlib.sha256(json.dumps(cids1).encode()).hexdigest() == exp1["order_sha256"]
            assert hashlib.sha256(ix1.tobytes()).hexdigest() == exp1["slate_sha256"]
            assert hashlib.sha256(json.dumps(cids4).encode()).hexdigest() == exp4["order_sha256"]
            assert hashlib.sha256(ix4.tobytes()).hexdigest() == exp4["slate_sha256"]
            acc1_batches.append((cids1, ix1))
            acc4_batches.append((cids4, ix4))

        for arm_name, batches in [
            ("PAIRED_CONTROL", ctrl_batches),
            ("FAMBAL_ACC1", acc1_batches),
            ("FAMBAL_ACC4", acc4_batches),
        ]:
            out_json = DATA_DIR / f"{arm_name}-{seed}.json"
            if out_json.exists():
                cached = json.loads(out_json.read_text())
                arm_results[arm_name][seed] = cached
                log_event(events_path, "arm_cached_loaded", arm=arm_name, seed=seed, mean=cached["mean"])
                continue

            check_host_guard(events_path)
            net.load_state_dict(baseline_state, strict=True)
            opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
            opt.load_state_dict(copy.deepcopy(ckpt["optimizer"]))
            torch.manual_seed(seed + 999)
            restart = warm_restart_adapter(net, opt, seed, rnd)
            perturb_l2 = float(
                torch.sqrt(sum(torch.sum((p - baseline_state[n]) ** 2) for n, p in trainable.items()))
            )
            initial_sha = state_sha(trainable)
            step_before = max(int(v["step"]) for v in opt.state.values() if "step" in v)
            losses, norms = [], []

            log_event(
                events_path,
                "arm_start",
                arm=arm_name,
                seed=seed,
                sigma=restart["sigma"],
                lr=restart["lr"],
                perturbation_l2=perturb_l2,
                updates=len(batches),
                presentations_per_update=len(batches[0][0]),
            )

            for step_idx, (cids, ix) in enumerate(batches):
                t_step = time.time()
                net.train()
                opt.zero_grad(set_to_none=True)
                total_loss = 0.0
                n_pres = len(cids)
                for off in range(0, n_pres, 32):
                    if off % 2048 == 0:
                        check_host_guard(events_path)
                    cc = cids[off : off + 32]
                    ii = ix[off : off + 32]
                    pred = predict(cc, list(ii))["total_cost"].float()
                    target = torch.as_tensor(
                        np.stack([pools[c]["fitness"][j] for c, j in zip(cc, ii)]),
                        dtype=torch.float32,
                    )
                    gate = torch.as_tensor(np.stack([pools[c]["gate"][j] for c, j in zip(cc, ii)]))
                    target = torch.where(gate, target, torch.full_like(target, 2.0))
                    loss = F.smooth_l1_loss(pred, target) + 0.1 * F.kl_div(
                        F.log_softmax(-pred / 0.2, dim=-1),
                        F.softmax(-target / 0.2, dim=-1),
                        reduction="batchmean",
                    )
                    assert torch.isfinite(loss), f"Non-finite loss at {arm_name} seed={seed} step={step_idx}"
                    weight = len(cc) / n_pres
                    (loss * weight).backward()
                    total_loss += float(loss.detach()) * weight

                norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
                assert np.isfinite(norm), f"Non-finite grad norm at {arm_name} seed={seed} step={step_idx}"
                opt.step()
                # Enforce dope_freeze_guard on EVERY update
                if state_sha(frozen_fn()) != frozen_sha:
                    raise RuntimeError("Frozen non-adapter state mutated during update!")
                if base.guard.snapshot_dope_weights(net).composite_sha256 != DOPE_SHA:
                    raise RuntimeError("DOPE weights checksum mismatch during update!")
                losses.append(total_loss)
                norms.append(norm)
                log_event(
                    events_path,
                    "optimizer_update",
                    arm=arm_name,
                    seed=seed,
                    step=step_idx + 1,
                    presentations=n_pres,
                    loss=total_loss,
                    grad_norm=norm,
                    step_sec=round(time.time() - t_step, 3),
                )

            candidate_sha = state_sha(trainable)
            assert candidate_sha != initial_sha
            step_after = max(int(v["step"]) for v in opt.state.values() if "step" in v)
            assert step_after == step_before + 8

            ckpt_out = arms_dir / f"{arm_name}-{seed}.pt"
            torch.save(
                {
                    "state_dict": net.state_dict(),
                    "optimizer": opt.state_dict(),
                    "seed": seed,
                    "arm": arm_name,
                    "dope_sha256": DOPE_SHA,
                    "adapter_sha256": candidate_sha,
                },
                ckpt_out,
            )
            ckpt_sha = sha256_file(ckpt_out)
            log_event(
                events_path,
                "candidate_frozen_before_dev",
                arm=arm_name,
                seed=seed,
                checkpoint_sha256=ckpt_sha,
                adapter_sha256=candidate_sha,
            )

            # Touch dev91 ONCE per final candidate (terminal score only)
            dev_res = score_terminal_dev()
            dev_res.update(
                {
                    "seed": seed,
                    "arm": arm_name,
                    "sigma": restart["sigma"],
                    "lr": restart["lr"],
                    "perturbation_l2": perturb_l2,
                    "initial_adapter_sha256": initial_sha,
                    "adapter_sha256": candidate_sha,
                    "checkpoint": str(ckpt_out),
                    "checkpoint_sha256": ckpt_sha,
                    "optimizer_step_before": step_before,
                    "optimizer_step_after": step_after,
                    "optimizer_updates": 8,
                    "presentations_per_update": len(batches[0][0]),
                    "training_losses": losses,
                    "gradient_norms": norms,
                    "changed_selections_from_r3175": int(
                        sum(a != b for a, b in zip(dev_res["selected_programs"], r3175_base["selected_programs"]))
                    ),
                    "frozen_state_sha256": frozen_sha,
                    "dope_sha256": DOPE_SHA,
                    "dope_match": True,
                }
            )
            write_json(out_json, dev_res)
            arm_results[arm_name][seed] = dev_res
            log_event(
                events_path,
                "terminal_result",
                arm=arm_name,
                seed=seed,
                utility=dev_res["mean"],
                lift_vs_r3175=dev_res["mean"] - r3175_base["mean"],
                changed_contexts=dev_res["changed_selections_from_r3175"],
            )

    # Evaluate primary (FAMBAL_ACC4), secondary (FAMBAL_ACC1), and PAIRED_CONTROL
    acc4_evals = [
        a298.evaluate_seed_comparison(
            arm_results["FAMBAL_ACC4"][s],
            arm_results["PAIRED_CONTROL"][s],
            r3175_base,
            dev_families,
        )
        for s in a298.REPLICATION_SEEDS
    ]
    acc1_evals = [
        a298.evaluate_seed_comparison(
            arm_results["FAMBAL_ACC1"][s],
            arm_results["PAIRED_CONTROL"][s],
            r3175_base,
            dev_families,
        )
        for s in a298.REPLICATION_SEEDS
    ]
    ctrl_evals = [
        a298.evaluate_seed_comparison(
            arm_results["PAIRED_CONTROL"][s],
            arm_results["PAIRED_CONTROL"][s],
            r3175_base,
            dev_families,
        )
        for s in a298.REPLICATION_SEEDS
    ]

    acc4_summary = a298.summarize_arm_across_seeds(acc4_evals)
    acc1_summary = a298.summarize_arm_across_seeds(acc1_evals)
    ctrl_summary = a298.summarize_arm_across_seeds(ctrl_evals)

    verdict = "DEV_ADVANCE" if acc4_summary["all_3_of_3_pass"] else "NOT_ADVANCED"

    summary_payload = {
        "wave_id": "H298",
        "slug": "h298_family_balanced_gradaccum",
        "agent": "rd_argon",
        "verdict": verdict,
        "elapsed_sec": round(time.time() - t0, 3),
        "r3175_baseline_mean": a298.R3175_DEV91_MEAN,
        "strict_advance_threshold": a298.STRICT_THRESHOLD,
        "evaluation_identity": EXPECTED_EVAL_ID,
        "dope_sha256": DOPE_SHA,
        "round_3175_sha256": EXPECTED_CKPT_SHA,
        "preregistered_batches_sha256": EXPECTED_PREREG_BATCHES_SHA,
        "regression_receipt": reg,
        "primary_arm_FAMBAL_ACC4": acc4_summary,
        "secondary_arm_FAMBAL_ACC1": acc1_summary,
        "paired_control_PAIRED_CONTROL": ctrl_summary,
    }
    write_json(DATA_DIR / "metrics_h298.json", summary_payload)
    log_event(events_path, "experiment_complete", verdict=verdict, elapsed_sec=summary_payload["elapsed_sec"])
    return summary_payload


if __name__ == "__main__":
    run_experiment()
