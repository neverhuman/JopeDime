"""H205 GPU runner on xbabe1 RTX 4090: fresh vs static data & static teacher across 50 rounds from r882.

Executes inside a single GPU worker process on xbabe1 cuda:0 alongside live flywheel PID 2297929:
- Stage 0: Native r882 parity 91/91 exact (1.1377941898164525) on DL-Val before any candidate forward.
- Stage 1: STATIC-DATA (50 rounds from r882 on fixed 1,219 contexts / 205 families with lockstep H178
  per-round 205-family pooled cross-fit B-fold selector admission) + frozen-r882 teacher identity snapshot,
  flushed to stage1_static_data_b_panel.npz before Stage 2.
- Stage 2: Gated on luna_v1_tel's JRB-01 ADMISSION PASS receipt and verified fresh D4 regression pool
  with native-Rust-CV (`bootstrap.s1main --fit-s1`) label/program/feature receipts; ingests verified
  disjoint fresh tokens, masks, programs, and labels into the engine via `ingest_verified_fresh_rows_into_engine`,
  then executes STATIC-TEACHER and FRESH on the identical fresh-family schedule (50 rounds each from r882
  under equal compute and lockstep H178 per-round 205-family pooled B-fold selector admission).
- Stage 3: Writes and fsyncs `pooled_b_selection_lock.json` (including per-cell checkpoint parameter digests)
  BEFORE any held-out A-fold evaluation (`evaluate_heldout_a_panel` enforces `verify_b_lock_persisted_before_a_eval`).
"""

import time

PROCESS_START_MONOTONIC = time.monotonic()

import argparse  # noqa: E402 -- capture startup before imports
from datetime import datetime, timezone  # noqa: E402 -- capture startup before imports
import hashlib  # noqa: E402 -- capture startup before imports
import importlib.util  # noqa: E402 -- capture startup before imports
import json  # noqa: E402 -- capture startup before imports
import math  # noqa: E402 -- capture startup before imports
import os  # noqa: E402 -- capture startup before imports
from pathlib import Path  # noqa: E402 -- capture startup before imports
import re  # noqa: E402 -- capture startup before imports
import struct  # noqa: E402 -- capture startup before imports
import subprocess  # noqa: E402 -- capture startup before imports
import sys  # noqa: E402 -- capture startup before imports
import tempfile  # noqa: E402 -- capture startup before imports

import numpy as np  # noqa: E402 -- capture startup before imports
import torch  # noqa: E402 -- capture startup before imports

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import analysis as stat  # noqa: E402
import monitor as mon  # noqa: E402


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def place(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: place(v, device) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(place(v, device) for v in value)
    if isinstance(value, list):
        return [place(v, device) for v in value]
    return value


def parameter_digest(named_tensors):
    """Compute a deterministic SHA256 over ordered parameter names and float32/float64 bytes."""
    h = hashlib.sha256()
    for name, tensor in sorted(named_tensors.items()):
        h.update(name.encode("utf-8"))
        arr = tensor.detach().cpu().numpy()
        h.update(arr.tobytes())
    return h.hexdigest()


def ingest_verified_fresh_rows_into_engine(engine, fresh_rows, historical_cids):
    """Ingest verified disjoint fresh D4 rows (`tokens_10x32`, `mask_10`, `program`, `fitness`, `gate`) into `engine`.

    Enforces:
    1. Historical cache closure on `historical_cids` (`1,310` contexts) is intact before ingestion.
    2. Every fresh CID is strictly disjoint from `historical_cids` and existing `engine.pools`.
    3. `stat.extract_and_verify_fresh_row_payload(row)` validates inline and/or `pool_npz_path` arrays
       against `native_evaluator_receipt` digests (`labels_sha256`, `programs_sha256`, `features_sha256`).
    4. Populates `engine.tokens[cid]`, `engine.masks[cid]`, and `engine.pools[cid]`, then executes a
       24-candidate `engine.batch([cid], [np.arange(24)])` probe to prove batching succeeds without
       `KeyError` and returns bit-identical `fitness`, `gate`, `tokens`, and `masks` tensors.
    """
    hist_set = set(historical_cids)
    if not hist_set.issubset(set(engine.tokens)) or not hist_set.issubset(
        set(engine.masks)
    ):
        raise RuntimeError(
            "historical feature cache closure violated prior to fresh ingestion"
        )
    if not fresh_rows:
        raise RuntimeError(
            "ingest_verified_fresh_rows_into_engine requires non-empty fresh_rows"
        )

    ingested = {}
    slate_ix = np.arange(stat.SLATE_CANDIDATES, dtype=np.int64)
    for row in fresh_rows:
        payload = stat.extract_and_verify_fresh_row_payload(row)
        cid = payload["context_sha256"]
        if cid in hist_set:
            raise RuntimeError(f"fresh row {cid} collides with historical cohort cache")
        if cid in ingested:
            raise RuntimeError(f"duplicate fresh row {cid} in ingestion batch")

        prog = payload["program"]
        proj16 = (prog.astype(np.float64) @ engine.projection).astype(np.float32)
        tok_t = torch.as_tensor(payload["tokens_10x32"], dtype=torch.float32)
        mask_t = torch.as_tensor(payload["mask_10"], dtype=torch.bool)

        engine.tokens[cid] = tok_t
        engine.masks[cid] = mask_t
        engine.pools[cid] = dict(
            program=prog,
            proj16=proj16,
            fitness=payload["fitness"],
            gate=payload["gate"],
            labels_sha256=payload["labels_sha256"],
            programs_sha256=payload["programs_sha256"],
            features_sha256=payload["features_sha256"],
        )

        kwargs, b_tok, b_mask, b_fit, b_gate = engine.batch([cid], [slate_ix])
        expected_fit_f32 = payload["fitness"][slate_ix].astype(np.float32)
        expected_gate = payload["gate"][slate_ix]
        if (
            b_tok.shape != (1, stat.TOKEN_STEPS, stat.TOKEN_DIM)
            or b_mask.shape != (1, stat.TOKEN_STEPS)
            or b_fit.detach().cpu().numpy().tobytes() != expected_fit_f32.tobytes()
            or b_gate.detach().cpu().numpy().tobytes() != expected_gate.tobytes()
            or kwargs["prog_412"].detach().cpu().numpy().tobytes()
            != prog[slate_ix].tobytes()
        ):
            raise RuntimeError(
                f"fresh engine.batch verification failed for {cid}: consumed batch differs from verified receipt"
            )
        ingested[cid] = dict(
            dataset_id=payload["dataset_id"],
            labels_sha256=payload["labels_sha256"],
            programs_sha256=payload["programs_sha256"],
            features_sha256=payload["features_sha256"],
        )

    return dict(
        status="PASS",
        ingested_contexts=len(ingested),
        historical_contexts_preserved=len(hist_set),
        digests_by_cid=ingested,
    )


def verify_b_lock_persisted_before_a_eval(lock_path, expected_checkpoint_digests):
    """Enforce that `pooled_b_selection_lock.json` is durably persisted on disk and binds every cell checkpoint BEFORE A-fold scoring."""
    p = Path(lock_path)
    if not p.is_file():
        raise RuntimeError(
            f"stage-order violation: held-out A evaluation requires persisted {p}"
        )
    lock_obj = json.loads(p.read_text())
    if lock_obj.get("wave") != stat.WAVE or "decisions" not in lock_obj:
        raise RuntimeError(
            "stage-order violation: pooled_b_selection_lock.json missing wave/decisions"
        )
    persisted_digests = lock_obj.get("selected_checkpoint_digests")
    if (
        not isinstance(persisted_digests, dict)
        or persisted_digests != expected_checkpoint_digests
    ):
        raise RuntimeError(
            "stage-order violation: persisted selected_checkpoint_digests mismatch"
        )
    return lock_obj


def evaluate_heldout_a_panel(
    engine,
    restore_adapter,
    cells,
    r882_train_values,
    lock_path,
    expected_checkpoint_digests,
    arm_name,
):
    """Evaluate held-out A_0..A_4 folds ONLY AFTER verifying `pooled_b_selection_lock.json` on disk."""
    lock_obj = verify_b_lock_persisted_before_a_eval(
        lock_path, expected_checkpoint_digests
    )
    arm_digests = lock_obj["selected_checkpoint_digests"].get(arm_name, {})
    a_panel = np.broadcast_to(
        r882_train_values[None, :], (len(stat.SEEDS), len(r882_train_values))
    ).copy()
    for (si, outer), cell in sorted(cells.items()):
        cell_key = f"seed_{cell['seed']}_outer_{outer}"
        actual_digest = parameter_digest(cell["incumbent_state"])
        if arm_digests.get(cell_key) != actual_digest:
            raise RuntimeError(
                f"stage-order violation: cell {arm_name}/{cell_key} incumbent digest differs from persisted B lock"
            )
        restore_adapter(cell["incumbent_state"])
        a_eval, _ = engine.evaluate(cell["a_rows"], native=False)
        a_panel[si, cell["a_indices"]] = np.asarray(a_eval["values"], dtype=np.float64)
    return a_panel


def load_h170_parent():
    path = ROOT / "research/flywheel_selfplay/h170_onecycle/runner.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("h205_h170_parent", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_launch(launch, manifest_sha, process_start_monotonic=None):
    if launch.get("wave") != stat.WAVE:
        raise RuntimeError(f"launch wave mismatch: {launch.get('wave')!r}")
    if launch.get("authority") != stat.AUTHORITY_STRING:
        raise RuntimeError("launch.json missing exact verbatim t2022 authority string")
    if launch.get("manifest_sha256") != manifest_sha:
        raise RuntimeError("launch.json manifest_sha256 mismatch")
    if (
        int(launch.get("numeric_threads", 0)) > mon.MAX_NUMERIC_THREADS
        or int(launch.get("numeric_threads", 0)) < 1
    ):
        raise RuntimeError("launch.json numeric_threads must be in 1..4")
    if not bool(launch.get("cuda_runtime_threads_exempt", False)):
        raise RuntimeError("launch.json must record cuda_runtime_threads_exempt=True")
    if int(launch.get("live_flywheel_pid_untouched", 0)) != mon.LIVE_FLYWHEEL_PID:
        raise RuntimeError(
            "launch.json must record live_flywheel_pid_untouched=2297929"
        )
    start_mono = (
        float(process_start_monotonic)
        if process_start_monotonic is not None
        else float(
            os.environ.get("H205_PROCESS_START_MONOTONIC", PROCESS_START_MONOTONIC)
        )
    )
    prior_elapsed = float(
        launch.get(
            "prior_attempts_elapsed_seconds",
            os.environ.get(
                "H205_PRIOR_ELAPSED_SECONDS", stat.PRIOR_ATTEMPTS_ELAPSED_SECONDS
            ),
        )
    )
    cap_seconds = float(
        launch.get(
            "campaign_wall_budget_cap_seconds", stat.CAMPAIGN_WALL_BUDGET_CAP_SECONDS
        )
    )
    req_rem = launch.get(
        "remaining_wall_budget_seconds",
        os.environ.get("H205_REMAINING_BUDGET_SECONDS"),
    )
    env_hard = os.environ.get("H205_HARD_DEADLINE_MONOTONIC")
    deadline_spec = mon.compute_shared_process_deadline(
        start_mono,
        prior_elapsed_seconds=prior_elapsed,
        campaign_cap_seconds=cap_seconds,
        requested_remaining_seconds=req_rem,
        env_hard_deadline_monotonic=env_hard,
    )
    if "fresh_wait_seconds" in launch:
        fw = float(launch["fresh_wait_seconds"])
        if fw <= 0.0 or fw > deadline_spec["remaining_budget_seconds"] + 1e-9:
            raise RuntimeError(
                f"launch.json fresh_wait_seconds ({fw}) must be in (0, {deadline_spec['remaining_budget_seconds']}]"
            )
    return deadline_spec


def configure_threads(numeric_threads):
    n = max(1, min(mon.MAX_NUMERIC_THREADS, int(numeric_threads)))
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    bad = mon.check_numeric_env_and_torch(
        env=os.environ,
        torch_intra_op=torch.get_num_threads(),
        torch_inter_op=torch.get_num_interop_threads(),
    )
    if bad:
        raise RuntimeError(f"numeric thread configuration violation: {bad}")


def run_gpu_wave(out, launch, manifest, deadline_spec=None):
    if deadline_spec is None:
        deadline_spec = mon.compute_shared_process_deadline(
            float(os.environ.get("H205_PROCESS_START_MONOTONIC", time.monotonic())),
            prior_elapsed_seconds=launch.get(
                "prior_attempts_elapsed_seconds", stat.PRIOR_ATTEMPTS_ELAPSED_SECONDS
            ),
            campaign_cap_seconds=launch.get(
                "campaign_wall_budget_cap_seconds",
                stat.CAMPAIGN_WALL_BUDGET_CAP_SECONDS,
            ),
            requested_remaining_seconds=launch.get("remaining_wall_budget_seconds"),
            env_hard_deadline_monotonic=os.environ.get("H205_HARD_DEADLINE_MONOTONIC"),
        )
    mon.enforce_shared_process_deadline(
        time.monotonic(), deadline_spec, stage="run_gpu_wave_entry"
    )
    host = os.uname().nodename.split(".")[0]
    if host != "xbabe1" or os.getpriority(os.PRIO_PROCESS, 0) != 19:
        raise RuntimeError(
            f"H205 GPU execution requires xbabe1 at nice 19 (got {host!r})"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device cuda:0 required on xbabe1 RTX 4090")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(0.18, 0)

    h170 = load_h170_parent()
    rt = h170.runtime()
    ckpt_path = Path(manifest["checkpoint"])
    ref_path = Path(manifest["reference"])
    if sha(ckpt_path) != stat.R882_SHA256:
        raise RuntimeError("r882 checkpoint SHA256 mismatch")

    net, guard, names, frozen, fused = rt.build(ckpt_path)
    net.to("cuda:0")
    orig_predict = net.adapter_10tok.predict_costs

    def gpu_predict_costs(**kwargs):
        return place(orig_predict(**place(kwargs, "cuda:0")), "cpu")

    net.adapter_10tok.predict_costs = gpu_predict_costs
    rt.check_frozen(net, guard, names, frozen)

    initial_state = {
        n: p.detach().cpu().clone()
        for n, p in net.named_parameters()
        if p.requires_grad
    }
    initial_teacher_digest = parameter_digest(initial_state)

    device_receipt = dict(
        pid=os.getpid(),
        host=os.uname().nodename,
        device=str(next(net.parameters()).device),
        gpu=torch.cuda.get_device_name(0),
        dope_sha256=rt.DOPE,
        r882_sha256=stat.R882_SHA256,
        teacher_param_sha256=initial_teacher_digest,
        trainable_parameters=sum(
            p.numel() for p in net.parameters() if p.requires_grad
        ),
        numeric_threads=torch.get_num_threads(),
        cuda_runtime_threads_exempt=True,
        memory_fraction=0.18,
        tf32=False,
        live_flywheel_pid_untouched=mon.LIVE_FLYWHEEL_PID,
        prior_attempts_elapsed_seconds=stat.PRIOR_ATTEMPTS_ELAPSED_SECONDS,
        remaining_wall_budget_seconds=stat.REMAINING_WALL_BUDGET_SECONDS,
        utc=datetime.now(timezone.utc).isoformat(),
    )
    save(out / "gpu-device.json", device_receipt)
    print("GPU DEVICE", json.dumps(device_receipt), flush=True)

    train, val = h170.cohorts(rt)
    historical_cids = {r["context_sha256"] for r in (train + val)}

    class GPUEngine(h170.Engine):
        def anchored(self, kwargs):
            return place(super().anchored(place(kwargs, "cuda:0")), "cpu")

        def forward(self, kwargs, tok, mask, anchor):
            return (
                super()
                .forward(
                    place(kwargs, "cuda:0"),
                    place(tok, "cuda:0"),
                    place(mask, "cuda:0"),
                    place(anchor, "cuda:0"),
                )
                .cpu()
            )

        def check(self):
            rt.check_frozen(self.net, self.guard, self.names, self.frozen)
            if self.net.dope_model.training or self.net.adapter_10tok.anchor.training:
                raise RuntimeError("frozen mode violation")
            if not torch.cuda.is_initialized() or any(
                p.device.type != "cuda" for p in self.net.parameters()
            ):
                raise RuntimeError("xbabe1 CUDA parameter placement required")

    engine = GPUEngine(net, rt, names, frozen, guard, fused, train + val)
    engine.load(val)

    # Stage 0: Native r882 91/91 exact parity gate before any candidate forward
    val_eval, _ = engine.evaluate(val, native=True, prove_identity=True)
    ref_payload = json.loads(ref_path.read_text())
    if isinstance(ref_payload, dict) and "utility" in ref_payload:
        ref_vals = ref_payload["utility"]["per_context"]
    elif isinstance(ref_payload, dict):
        ref_vals = ref_payload["values"]
    else:
        ref_vals = ref_payload
    parity = stat.verify_r882_native_parity(val_eval["values"], ref_vals)
    save(out / "stage0_r882_parity.json", parity)
    print(
        f"STAGE0_R882_PARITY_PASS 91/91 bit-exact mean={parity['mean']:.16f}",
        flush=True,
    )

    # Stage 1: Load 1,219 DL-Train pools & evaluate baseline r882
    engine.load(train)
    train_base_eval, _ = engine.evaluate(train, native=True, prove_identity=False)
    r882_train_values = np.asarray(train_base_eval["values"], dtype=np.float64)
    gt_orders = {
        r["context_sha256"]: np.argsort(
            np.where(
                engine.pools[r["context_sha256"]]["gate"],
                engine.pools[r["context_sha256"]]["fitness"],
                2.0,
            ),
            kind="mergesort",
        )
        for r in train
    }

    def restore_adapter(state_cpu):
        with torch.no_grad():
            for n, p in net.named_parameters():
                if n in state_cpu:
                    p.copy_(state_cpu[n].to(p.device))

    teacher_identity_check = stat.verify_frozen_teacher_identity(
        teacher_checkpoint_sha256=stat.R882_SHA256,
        dope_sha256=rt.DOPE,
        initial_param_sha256=initial_teacher_digest,
        current_param_sha256=parameter_digest(initial_state),
        teacher_training_mode=False,
        any_requires_grad=False,
    )
    save(
        out / "static_teacher_infra.json",
        dict(
            status="READY",
            teacher_checkpoint_sha256=stat.R882_SHA256,
            teacher_param_sha256=initial_teacher_digest,
            teacher_identity_verified=teacher_identity_check["immutable"],
            r882_train_mean=float(r882_train_values.mean()),
            utc=datetime.now(timezone.utc).isoformat(),
        ),
    )

    cid_to_idx = {r["context_sha256"]: i for i, r in enumerate(train)}
    updates = 0
    round_log = []

    def run_arm_lockstep_pooled(
        arm_name,
        orders_map,
        fresh_rows_for_arm=None,
        teacher_cost_by_cid=None,
    ):
        """Run 50 rounds across all 3 seeds x 5 outer folds in lockstep with H178 pooled 205-family B-fold selection.

        Does NOT evaluate held-out A_0..A_4 folds; held-out A evaluation is deferred until after all
        three arms complete B-fold selection and `pooled_b_selection_lock.json` is written and fsynced.
        """
        nonlocal updates
        incumbent_b_panel = np.broadcast_to(
            r882_train_values[None, :], (len(stat.SEEDS), len(train))
        ).copy()
        cells = {}
        for si, seed in enumerate(stat.SEEDS):
            for outer in range(5):
                fit_rows, b_rows, a_rows = stat.partition(train, outer)
                if fresh_rows_for_arm is not None:
                    fresh_fit = stat.partition_fresh_rows(fresh_rows_for_arm, outer)
                    slates = stat.h177_analysis.fit_slates(
                        fresh_fit, orders_map, seed, outer
                    )
                    plan = stat.build_equal_compute_round_plan(
                        fit_rows, seed, outer, fresh_fit_rows=fresh_fit
                    )[arm_name]
                else:
                    slates = stat.h177_analysis.fit_slates(
                        fit_rows, orders_map, seed, outer
                    )
                    plan = stat.build_equal_compute_round_plan(fit_rows, seed, outer)[
                        arm_name
                    ]

                fit_cids = sorted(slates.keys())
                slate_anchors = {}
                for start in range(0, len(fit_cids), 32):
                    chunk = fit_cids[start : start + 32]
                    kw_chunk, _, _, _, _ = engine.batch(
                        chunk, [slates[c] for c in chunk]
                    )
                    anc = engine.anchored(kw_chunk)
                    for idx_c, cid_c in enumerate(chunk):
                        slate_anchors[cid_c] = (
                            anc[0][idx_c].clone(),
                            anc[1][idx_c].clone(),
                        )

                cells[(si, outer)] = dict(
                    seed=seed,
                    outer=outer,
                    b_rows=b_rows,
                    a_rows=a_rows,
                    b_indices=[cid_to_idx[r["context_sha256"]] for r in b_rows],
                    a_indices=[cid_to_idx[r["context_sha256"]] for r in a_rows],
                    slates=slates,
                    slate_anchors=slate_anchors,
                    plan=plan,
                    incumbent_state={k: v.clone() for k, v in initial_state.items()},
                    candidate_state={k: v.clone() for k, v in initial_state.items()},
                    opt_state=None,
                )

        admitted_rounds = 0
        for r_idx in range(1, stat.ROUNDS + 1):
            mon.enforce_shared_process_deadline(
                time.monotonic(), deadline_spec, stage=f"{arm_name}_round_{r_idx}"
            )
            for (si, outer), cell in cells.items():
                restore_adapter(cell["candidate_state"])
                params = [p for n, p in net.named_parameters() if p.requires_grad]
                opt = torch.optim.AdamW(
                    params, lr=3e-5, betas=(0.9, 0.999), weight_decay=0.01
                )
                if cell["opt_state"] is not None:
                    opt.load_state_dict(cell["opt_state"])
                r_info = cell["plan"][r_idx - 1]
                opt.zero_grad(set_to_none=True)
                for mb in r_info["microbatches"]:
                    cids = mb["cids"]
                    ixs = [cell["slates"][c] for c in cids]
                    kwargs, tok, mask, fitness, gate = engine.batch(cids, ixs)
                    anchor = (
                        torch.stack([cell["slate_anchors"][c][0] for c in cids]),
                        torch.stack([cell["slate_anchors"][c][1] for c in cids]),
                    )
                    pred = engine.forward(kwargs, tok, mask, anchor)
                    if arm_name == "STATIC-TEACHER":
                        t_cost = torch.stack(
                            [
                                torch.as_tensor(
                                    teacher_cost_by_cid[c][ix], dtype=torch.float32
                                )
                                for c, ix in zip(cids, ixs, strict=True)
                            ]
                        )
                        t_gate = torch.ones_like(gate, dtype=torch.bool)
                        loss_val = rt.loss(pred, t_cost, t_gate) / len(
                            r_info["microbatches"]
                        )
                    else:
                        loss_val = rt.loss(pred, fitness, gate) / len(
                            r_info["microbatches"]
                        )
                    loss_val.backward()
                opt.step()
                updates += 1
                cell["candidate_state"] = {
                    n: p.detach().cpu().clone()
                    for n, p in net.named_parameters()
                    if p.requires_grad
                }
                cell["opt_state"] = opt.state_dict()

            if r_idx % 10 == 0 or r_idx == stat.ROUNDS:
                mon.enforce_shared_process_deadline(
                    time.monotonic(),
                    deadline_spec,
                    stage=f"{arm_name}_round_{r_idx}_b_eval",
                )
                cand_b_panel = incumbent_b_panel.copy()
                for (si, outer), cell in cells.items():
                    restore_adapter(cell["candidate_state"])
                    b_eval, _ = engine.evaluate(cell["b_rows"], native=False)
                    cand_b_panel[si, cell["b_indices"]] = np.asarray(
                        b_eval["values"], dtype=np.float64
                    )

                pooled_decision = stat.admit_round_checkpoint_pooled(
                    train,
                    incumbent_b_panel,
                    cand_b_panel,
                    seed=stat.BOOT_SEED + r_idx,
                )
                round_log.append(
                    dict(
                        arm=arm_name,
                        round=r_idx,
                        admit=pooled_decision["admit"],
                        gain=pooled_decision["gain"],
                        lcb95=pooled_decision["lcb95"],
                        harm_rate=pooled_decision["harm_rate"],
                        p_raw=pooled_decision["p_raw"],
                    )
                )
                if pooled_decision["admit"]:
                    incumbent_b_panel = cand_b_panel
                    admitted_rounds += 1
                    for cell in cells.values():
                        cell["incumbent_state"] = {
                            k: v.clone() for k, v in cell["candidate_state"].items()
                        }
                else:
                    for cell in cells.values():
                        cell["candidate_state"] = {
                            k: v.clone() for k, v in cell["incumbent_state"].items()
                        }

            save(
                out / "progress.json",
                dict(
                    wave=stat.WAVE,
                    stage="STAGE1_STATIC_DATA"
                    if arm_name == "STATIC-DATA"
                    else "STAGE2_FRESH_AND_TEACHER",
                    arm=arm_name,
                    round=r_idx,
                    total_rounds=stat.ROUNDS,
                    optimizer_updates=updates,
                    admitted_rounds=admitted_rounds,
                    r882_parity_verified=True,
                    utc=datetime.now(timezone.utc).isoformat(),
                ),
            )

        cell_digests = {
            f"seed_{cell['seed']}_outer_{outer}": parameter_digest(
                cell["incumbent_state"]
            )
            for (si, outer), cell in sorted(cells.items())
        }
        return incumbent_b_panel, cells, admitted_rounds, cell_digests

    # Execute Stage 1: STATIC-DATA B-fold selection and durably flush B panel before Stage 2 wait
    static_b_panel, static_cells, static_adm, static_digests = run_arm_lockstep_pooled(
        "STATIC-DATA", gt_orders
    )
    np.savez_compressed(
        out / "stage1_static_data_b_panel.npz",
        static_b_panel=static_b_panel,
        r882_train_values=r882_train_values,
    )

    # Stage 2: Gate STATIC-TEACHER and FRESH on JRB-01 ADMISSION PASS + verified fresh D4 pool
    # Shares the single process-start remaining deadline (never adds a fresh budget after Stage 1).
    jrb_receipt_path = Path(
        launch.get(
            "jrb01_admission_receipt",
            "/home/ubuntu/jope_plan/evidence/JRB-01/ADMISSION-PASS.json",
        )
    )
    hold_file = out / "hold_before_fresh.json"
    wait_deadline = deadline_spec["hard_deadline_monotonic"]
    if "fresh_wait_seconds" in launch:
        wait_deadline = min(
            wait_deadline,
            deadline_spec["process_start_monotonic"]
            + float(launch["fresh_wait_seconds"]),
        )
    fresh_bundle = None
    while time.monotonic() < wait_deadline:
        mon.enforce_shared_process_deadline(
            time.monotonic(), deadline_spec, stage="stage2_wait_jrb01"
        )
        if hold_file.is_file():
            raise RuntimeError(
                f"FRESH boundary hold requested by owner file: {hold_file}"
            )
        try:
            fresh_bundle = stat.verify_and_load_fresh_d4_pool(
                jrb_receipt_path, train, eval_rows=train + val
            )
            break
        except RuntimeError as exc:
            now_wait = time.monotonic()
            save(
                out / "progress.json",
                dict(
                    wave=stat.WAVE,
                    stage="STAGE2_WAITING_FOR_JRB01_ADMISSION_PASS",
                    optimizer_updates=updates,
                    jrb01_receipt_path=str(jrb_receipt_path),
                    reason=str(exc),
                    r882_parity_verified=True,
                    stage1_static_data_b_panel_flushed=True,
                    static_data_admitted_rounds=static_adm,
                    elapsed_seconds=now_wait - deadline_spec["process_start_monotonic"],
                    cumulative_elapsed_seconds=deadline_spec["prior_elapsed_seconds"]
                    + (now_wait - deadline_spec["process_start_monotonic"]),
                    remaining_budget_seconds=wait_deadline - now_wait,
                    utc=datetime.now(timezone.utc).isoformat(),
                ),
            )
            sleep_for = min(5.0, max(0.0, wait_deadline - time.monotonic()))
            if sleep_for <= 0.0:
                break
            time.sleep(sleep_for)

    if fresh_bundle is None:
        raise RuntimeError(
            f"FRESH and STATIC-TEACHER arms blocked: verified fresh D4 pool not available at {jrb_receipt_path}"
        )

    fresh_rows = fresh_bundle["records"]
    ingestion_receipt = ingest_verified_fresh_rows_into_engine(
        engine, fresh_rows, historical_cids=historical_cids
    )
    save(out / "fresh_engine_ingestion.json", ingestion_receipt)

    restore_adapter(initial_state)
    stat.verify_frozen_teacher_identity(
        teacher_checkpoint_sha256=stat.R882_SHA256,
        dope_sha256=rt.DOPE,
        initial_param_sha256=initial_teacher_digest,
        current_param_sha256=parameter_digest(
            {n: p.detach().cpu() for n, p in net.named_parameters() if p.requires_grad}
        ),
        teacher_training_mode=bool(
            net.dope_model.training or net.adapter_10tok.training
        ),
        any_requires_grad=False,
    )
    _, fresh_teacher_raw = engine.evaluate(fresh_rows, native=False)
    fresh_teacher_orders = {
        cid: np.argsort(cost, kind="mergesort")
        for cid, cost in fresh_teacher_raw.items()
    }
    fresh_gt_orders = {
        r["context_sha256"]: np.argsort(
            np.where(
                engine.pools[r["context_sha256"]]["gate"],
                engine.pools[r["context_sha256"]]["fitness"],
                2.0,
            ),
            kind="mergesort",
        )
        for r in fresh_rows
    }

    teacher_b_panel, teacher_cells, teacher_adm, teacher_digests = (
        run_arm_lockstep_pooled(
            "STATIC-TEACHER",
            fresh_teacher_orders,
            fresh_rows_for_arm=fresh_rows,
            teacher_cost_by_cid=fresh_teacher_raw,
        )
    )
    fresh_b_panel, fresh_cells, fresh_adm, fresh_digests = run_arm_lockstep_pooled(
        "FRESH",
        fresh_gt_orders,
        fresh_rows_for_arm=fresh_rows,
    )

    # Stage 3: Write and fsync pooled_b_selection_lock.json BEFORE any held-out A-fold evaluation
    selected_checkpoint_digests = {
        "STATIC-DATA": static_digests,
        "STATIC-TEACHER": teacher_digests,
        "FRESH": fresh_digests,
    }
    b_lock = stat.h178_round_select(
        train,
        r882_train_values,
        {
            "STATIC-DATA": static_b_panel,
            "STATIC-TEACHER": teacher_b_panel,
            "FRESH": fresh_b_panel,
        },
    )
    b_lock["selected_checkpoint_digests"] = selected_checkpoint_digests
    lock_path = out / "pooled_b_selection_lock.json"
    save(lock_path, b_lock)

    static_a_panel = evaluate_heldout_a_panel(
        engine,
        restore_adapter,
        static_cells,
        r882_train_values,
        lock_path,
        selected_checkpoint_digests,
        "STATIC-DATA",
    )
    teacher_a_panel = evaluate_heldout_a_panel(
        engine,
        restore_adapter,
        teacher_cells,
        r882_train_values,
        lock_path,
        selected_checkpoint_digests,
        "STATIC-TEACHER",
    )
    fresh_a_panel = evaluate_heldout_a_panel(
        engine,
        restore_adapter,
        fresh_cells,
        r882_train_values,
        lock_path,
        selected_checkpoint_digests,
        "FRESH",
    )

    gate_summary = stat.evaluate_h205_common_gate(
        val_eval["values"],
        ref_vals,
        train,
        r882_train_values,
        {
            "STATIC-DATA": static_a_panel,
            "STATIC-TEACHER": teacher_a_panel,
            "FRESH": fresh_a_panel,
        },
    )
    metrics = dict(
        wave=stat.WAVE,
        status="COMPLETE",
        authority=stat.AUTHORITY_STRING,
        optimizer_updates=updates,
        scored_contexts=len(train) + len(val),
        jrb01_admission=dict(
            status=fresh_bundle["status"],
            admitted_rows=fresh_bundle["admitted_rows"],
            admitted_families=fresh_bundle["admitted_families"],
            manifest_sha256=fresh_bundle["manifest_sha256"],
        ),
        fresh_engine_ingestion=dict(
            status=ingestion_receipt["status"],
            ingested_contexts=ingestion_receipt["ingested_contexts"],
            historical_contexts_preserved=ingestion_receipt[
                "historical_contexts_preserved"
            ],
        ),
        r882_parity=parity,
        pooled_b_selection=b_lock["decisions"],
        common_gate=gate_summary,
        admitted_rounds_summary=dict(
            STATIC_DATA=static_adm,
            STATIC_TEACHER=teacher_adm,
            FRESH=fresh_adm,
        ),
        round_log=round_log,
        verdict=gate_summary["verdict"],
        utc=datetime.now(timezone.utc).isoformat(),
    )
    save(out / "metrics.json", metrics)
    return metrics


def _dsl_feature_dict(dsl):
    """Canonical 412D DSL feature dictionary matching `rank_prepare.features`."""
    ops = re.findall(r"\(([^()]+)\)", dsl)
    result = {"steps": float(len(ops)), "dsl_characters": float(len(dsl))}
    for pos, content in enumerate(ops):
        tokens = content.split()
        op = tokens[0]
        result["count." + op] = result.get("count." + op, 0) + 1
        result[f"position{pos}.{op}"] = 1.0
        if pos == len(ops) - 1:
            result["estimator." + op] = 1.0
        for token in tokens[1:]:
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            if key == "seed":
                continue
            try:
                value = float(value)
            except ValueError:
                if value not in ("true", "false"):
                    continue
                value = float(value == "true")
            value = float(np.sign(value) * np.log1p(abs(value)))
            result[f"parameter.{op}.{key}.sum"] = (
                result.get(f"parameter.{op}.{key}.sum", 0) + value
            )
            result[f"parameter.{op}.{key}.count"] = (
                result.get(f"parameter.{op}.{key}.count", 0) + 1
            )
            result[f"position{pos}.{key}"] = value
    return result


def _w1_sign_projection_matrix():
    cols = []
    for k in range(16):
        col = [
            1
            if (
                hashlib.sha256(
                    b"jope-context-cross-v1\0" + struct.pack("<II", i, k)
                ).digest()[0]
                & 1
            )
            else -1
            for i in range(stat.PROGRAM_DIM)
        ]
        cols.append(col)
    return np.asarray(cols, dtype=np.float64).T / math.sqrt(float(stat.PROGRAM_DIM))


def featurize_dsl_412_and_proj16(dsl_list, feature_names=None, projection_mat=None):
    """Featurize GP DSL strings into `(N, 412)` float32 `program` and `(N, 16)` float32 `proj16`."""
    if feature_names is None:
        if sha(stat.PROGRAM_FEATURE_NAMES_PATH) != stat.PROGRAM_FEATURE_NAMES_SHA256:
            raise RuntimeError("program-feature-names.json SHA256 mismatch")
        feature_names = json.loads(
            stat.PROGRAM_FEATURE_NAMES_PATH.read_text(encoding="utf-8")
        )
    if len(feature_names) != stat.PROGRAM_DIM:
        raise RuntimeError(
            f"expected {stat.PROGRAM_DIM} program feature names, got {len(feature_names)}"
        )
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    prog_mat = np.zeros((len(dsl_list), stat.PROGRAM_DIM), dtype=np.float32)
    for row_i, dsl in enumerate(dsl_list):
        for k, v in _dsl_feature_dict(dsl).items():
            idx = name_to_idx.get(k)
            if idx is not None:
                prog_mat[row_i, idx] = float(v)
    proj_mat = (
        projection_mat if projection_mat is not None else _w1_sign_projection_matrix()
    )
    proj16 = (prog_mat.astype(np.float64) @ proj_mat).astype(np.float32)
    return prog_mat, proj16


def extract_frozen_dope_tokens_10x32(
    net, ws, x_train_np, y_train_np, device=torch.device("cpu"), deadline_spec=None
):
    """Extract frozen Stage-1A/1B `tokens_10x32`, `mask_10`, `dope629`, `lm48_z`, and `ctx_feat_584` on TRAIN rows only."""
    if deadline_spec is not None:
        mon.enforce_shared_process_deadline(
            time.monotonic(), deadline_spec, stage="extract_frozen_dope_tokens"
        )
    sys.path.insert(0, str(ROOT / "research/flywheel_selfplay"))
    import run_fused_closed_loop as base_loop  # noqa: E402

    x_arr = np.asarray(x_train_np, dtype=np.float64)
    y_arr = np.asarray(y_train_np, dtype=np.float64)
    if x_arr.ndim != 2 or y_arr.ndim != 1 or x_arr.shape[0] != y_arr.shape[0]:
        raise RuntimeError(
            f"invalid training matrix/target shapes: X={x_arr.shape}, y={y_arr.shape}"
        )
    if not np.isfinite(y_arr).all():
        raise RuntimeError(
            "non-finite target values in extract_frozen_dope_tokens_10x32"
        )

    yn = ((y_arr - y_arr.mean()) / max(float(y_arr.std()), 1e-8)).astype(np.float32)
    vec, cols, _ = ws.fit_transform_and_extract_bundle_high(
        base_loop.landmark_float32(x_arr), np.ascontiguousarray(yn)
    )
    colbuf = np.zeros((64, 17), np.float32)
    ncols = min(len(cols), 64)
    colbuf[:ncols, : min(cols.shape[1], 17)] = cols[:ncols, :17]

    with torch.no_grad():
        d629, lm48 = net._run_dope_mlc_v3_and_lm48(
            torch.as_tensor(vec, device=device).reshape(1, 256),
            torch.as_tensor(colbuf, device=device).reshape(1, 64, 17),
            use_landmarks=True,
        )
        tok, mask, lm48_z = net._project_stage1b_tokens_batched_gpu(lm48, d629)

    d629_np = d629[0].detach().cpu().numpy().astype(np.float32)
    lm48_z_np = lm48_z[0].detach().cpu().numpy().astype(np.float32)
    tok_np = tok[0].detach().cpu().numpy().astype(np.float32)
    mask_np = mask[0].detach().cpu().numpy().astype(bool)
    ctx_584 = np.zeros((584,), dtype=np.float32)
    ctx_584[88:200] = d629_np[:112]
    ctx_584[536:584] = lm48_z_np
    return dict(
        tokens_10x32=tok_np,
        mask_10=mask_np,
        dope629=d629_np,
        lm48_z=lm48_z_np,
        ctx_feat_584=ctx_584,
        train_rows=int(x_arr.shape[0]),
        feature_columns=int(x_arr.shape[1]),
        outer_test_rows_decoded=0,
    )


def _default_same_train_candidate_records(feature_columns=5):
    """Build 24 deterministic canonical `(pipe X ...)` GP candidate program records for same-TRAIN evaluation."""
    ncols = max(1, int(feature_columns))
    dsls = [
        "(pipe X (ridge))",
        "(pipe X (ridge_gcv))",
        "(pipe X (standardize) (ridge))",
        "(pipe X (standardize) (ridge_gcv))",
        "(pipe X (robust_scale) (ridge))",
        "(pipe X (robust_scale) (ridge_gcv))",
        "(pipe X (rank) (ridge))",
        "(pipe X (blom) (ridge))",
        "(pipe X (winsor_clip q=0.05) (standardize) (ridge))",
        "(pipe X (winsor_clip q=0.1) (robust_scale) (ridge_gcv))",
        "(pipe X (standardize) (poly_interact) (ridge))",
        "(pipe X (standardize) (poly_interact) (ridge_gcv))",
        f"(pipe X (standardize) (pair_prod topk={min(4, ncols)}) (ridge))",
        "(pipe X (standardize) (row_stats) (ridge))",
        "(pipe X (standardize) (cheb degree=2) (ridge))",
        "(pipe X (standardize) (cheb degree=3) (ridge_gcv))",
        "(pipe X (standardize) (hinge knots=3) (ridge))",
        "(pipe X (standardize) (quantile_bins bins=4) (ridge))",
        "(pipe X (standardize) (rff_cos gamma=0.5 n=16 seed=1) (ridge))",
        "(pipe X (standardize) (rff_sin gamma=0.5 n=16 seed=2) (ridge))",
        f"(pipe X (standardize) (rand_proj k={min(4, ncols)} seed=3) (ridge))",
        "(pipe X (standardize) (elasticnet alpha=0.01 l1_ratio=0.5))",
        "(pipe X (standardize) (ksmooth gamma=1.0 k=5))",
        "(pipe X (standardize) (histgbm depth=3 lr=0.1 rounds=20))",
    ]
    records = []
    for birth_seq, dsl in enumerate(dsls):
        psha = hashlib.sha256(dsl.encode("utf-8")).hexdigest()
        records.append(
            dict(
                program=dict(
                    dsl=dsl,
                    program_sha256=psha,
                    birth_seq=int(birth_seq),
                    parents=[],
                    operator="init_slate24",
                )
            )
        )
    records.sort(key=lambda r: r["program"]["program_sha256"])
    return records


def evaluate_slate24_live_s1main(
    s1_input_jtt1_path,
    candidate_records,
    worker_bin=stat.EVALUATOR_BINARY_PATH,
    max_workers=2,
    deadline_spec=None,
    verify_worker_sha256=True,
):
    """Run live `bootstrap.s1main --fit-s1` on EVERY candidate GP program (`len >= 24`) and compute/verify inner-CV labels."""
    bin_p = Path(worker_bin)
    if verify_worker_sha256 and sha(bin_p) != stat.EVALUATOR_BINARY_SHA256:
        raise RuntimeError(
            "bootstrap.s1main SHA256 mismatch in evaluate_slate24_live_s1main"
        )
    s1_p = Path(s1_input_jtt1_path)
    if not s1_p.is_file():
        raise RuntimeError(f"s1-input.jtt1 missing: {s1_p}")

    if len(candidate_records) < stat.SLATE_CANDIDATES:
        raise RuntimeError(
            f"at least {stat.SLATE_CANDIDATES} candidates required for live s1main verification"
        )

    max_diffs = []
    gate_matches = []
    live_fitness = []
    live_logratio = []
    live_gate = []
    with tempfile.TemporaryDirectory(prefix="h205_live_s1_") as td:
        td_path = Path(td)
        env = {
            "PATH": "/usr/bin:/bin",
            "RAYON_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "TMPDIR": str(td_path),
            "HOME": str(td_path),
        }
        for idx, rec in enumerate(candidate_records):
            if deadline_spec is not None:
                mon.enforce_shared_process_deadline(
                    time.monotonic(),
                    deadline_spec,
                    stage=f"live_s1main_candidate_{idx}",
                )
            prog = dict(rec["program"])
            prog.setdefault("birth_seq", int(idx))
            prog.setdefault("parents", [])
            prog.setdefault("operator", "init_slate24")
            dsl = prog["dsl"]
            cand_path = td_path / f"cand_{idx:04d}.json"
            out_path = td_path / f"eval_{idx:04d}.json"
            cand_path.write_text(json.dumps(prog, sort_keys=True), encoding="utf-8")
            proc = _run_bounded_subprocess(
                [
                    "nice",
                    "-n",
                    "19",
                    str(bin_p),
                    "--fit-s1",
                    str(s1_p),
                    str(cand_path),
                    str(out_path),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=90,
                env=env,
                check=False,
                deadline_spec=deadline_spec,
            )
            if proc.returncode != 0 or not out_path.is_file():
                raise RuntimeError(
                    f"live bootstrap.s1main --fit-s1 failed on candidate {idx}: rc={proc.returncode}, stderr={proc.stderr.strip()}"
                )
            ev = json.loads(out_path.read_text(encoding="utf-8"))
            if (
                ev.get("status") != "completed"
                or ev.get("fold_protocol") != "jope.s1-two-fold/v1"
                or "official_test" in ev
            ):
                raise RuntimeError(f"unexpected s1main record on candidate {idx}: {ev}")
            inner_cv = ev.get("inner_cv", ())
            if len(inner_cv) != 2:
                raise RuntimeError(f"expected 2 inner_cv folds, got {len(inner_cv)}")
            ratios = []
            wins = 0
            for fold_rec in inner_cv:
                m_rmse = float(fold_rec["metrics"]["rmse"])
                b_rmse = float(fold_rec["baseline_metrics"]["rmse"])
                r = max(m_rmse / b_rmse, 1e-12)
                ratios.append(r)
                if r <= 1.0:
                    wins += 1
            lr0, lr1 = math.log(ratios[0]), math.log(ratios[1])
            mean_lr = 0.5 * (lr0 + lr1)
            tail_lr = max(lr0, lr1)
            sd_lr = 0.5 * abs(lr0 - lr1)
            geomean_ratio = math.exp(mean_lr)
            comp = dsl.count("(")
            gate = (wins == 2) and (geomean_ratio <= 0.90)
            if gate:
                fit_live = 0.8 * mean_lr + 0.1 * tail_lr + 0.1 * sd_lr + 0.0005 * comp
            else:
                fit_live = 100.0 + mean_lr + tail_lr + sd_lr
            live_fitness.append(float(fit_live))
            live_logratio.append(float(mean_lr))
            live_gate.append(bool(gate))
            if "fitness" in rec and "cv_gate_passed" in rec:
                max_diffs.append(abs(float(fit_live) - float(rec["fitness"])))
                gate_matches.append(bool(gate) == bool(rec["cv_gate_passed"]))
            else:
                max_diffs.append(0.0)
                gate_matches.append(True)

    max_diff = float(max(max_diffs)) if max_diffs else 0.0
    all_gate = bool(all(gate_matches))
    if max_diff > 1e-9 or not all_gate:
        raise RuntimeError(
            f"live s1main parity check failed: max_diff={max_diff}, all_gate={all_gate}"
        )
    mon.enforce_shared_process_deadline(
        time.monotonic(), deadline_spec, stage="after_native_or_s3_verification"
    )
    return dict(
        status="PASS",
        evaluator=stat.NATIVE_RUST_EVALUATOR_ID,
        evaluator_binary_sha256=sha(bin_p),
        evaluated_candidates=len(candidate_records),
        max_abs_fitness_diff=max_diff,
        gate_exact_match=all_gate,
        fitness_array=np.asarray(live_fitness, dtype=np.float64),
        logratio_array=np.asarray(live_logratio, dtype=np.float64),
        gate_array=np.asarray(live_gate, dtype=bool),
    )


def consume_single_fresh_context_in_inherited_engine(
    engine, net, rt, fresh_row, historical_cids, deadline_spec=None
):
    """Ingest one verified fresh regression context into `engine` and execute 1 bounded AdamW step on `adapter_10tok`."""
    if deadline_spec is not None:
        mon.enforce_shared_process_deadline(
            time.monotonic(),
            deadline_spec,
            stage="consume_single_fresh_context_entry",
        )
    ing = ingest_verified_fresh_rows_into_engine(
        engine, [fresh_row], historical_cids=historical_cids
    )
    cid = fresh_row["context_sha256"]
    if cid in set(historical_cids):
        raise RuntimeError("fresh context collides with historical 1,310 cohort")

    gt_order = np.argsort(
        np.where(
            engine.pools[cid]["gate"],
            engine.pools[cid]["fitness"],
            2.0,
        ),
        kind="mergesort",
    )
    slates = stat.h177_analysis.fit_slates(
        [fresh_row], {cid: gt_order}, stat.SEEDS[0], 0
    )
    slate_ix = slates[cid]
    if len(slate_ix) != stat.SLATE_CANDIDATES:
        raise RuntimeError(f"expected 24-candidate slate, got {len(slate_ix)}")

    initial_trainable = {
        n: p.detach().cpu().clone()
        for n, p in net.named_parameters()
        if p.requires_grad
    }
    trainable_count = sum(p.numel() for p in initial_trainable.values())
    if trainable_count != 29211:
        raise RuntimeError(
            f"expected 29,211 trainable adapter_10tok parameters, got {trainable_count}"
        )
    adapter_before = parameter_digest(initial_trainable)
    rt.check_frozen(net, engine.guard, engine.names, engine.frozen)
    dope_before = rt.DOPE

    kwargs, tok, mask, fitness, gate = engine.batch([cid], [slate_ix])
    anchor = engine.anchored(kwargs)
    pred = engine.forward(kwargs, tok, mask, anchor)

    params = [p for _, p in net.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=3e-5, betas=(0.9, 0.999), weight_decay=0.01)
    opt.zero_grad(set_to_none=True)
    loss_val = rt.loss(pred, fitness, gate)
    if not torch.isfinite(loss_val).all():
        raise RuntimeError(f"non-finite loss on fresh context {cid}: {loss_val}")
    if deadline_spec is not None:
        mon.enforce_shared_process_deadline(
            time.monotonic(), deadline_spec, stage="before_fresh_context_backward"
        )
    loss_val.backward()
    if deadline_spec is not None:
        mon.enforce_shared_process_deadline(
            time.monotonic(), deadline_spec, stage="before_fresh_context_optimizer_step"
        )
    opt.step()

    after_trainable = {
        n: p.detach().cpu().clone()
        for n, p in net.named_parameters()
        if p.requires_grad
    }
    adapter_after = parameter_digest(after_trainable)
    rt.check_frozen(net, engine.guard, engine.names, engine.frozen)
    dope_after = rt.DOPE

    # Restore initial adapter weights after recording the 1-step delta proof
    with torch.no_grad():
        for n, p in net.named_parameters():
            if n in initial_trainable:
                p.copy_(initial_trainable[n].to(p.device))

    if adapter_before == adapter_after:
        raise RuntimeError("adapter parameter digest did not change after 1 AdamW step")
    if deadline_spec is not None:
        mon.enforce_shared_process_deadline(
            time.monotonic(),
            deadline_spec,
            stage="consume_single_fresh_context_exit",
        )

    return dict(
        status="PASS",
        ingested_context_sha256=cid,
        dataset_id=fresh_row["dataset_id"],
        family_id=fresh_row["family_id"],
        used_old_1310_fallback=False,
        historical_contexts_preserved=ing["historical_contexts_preserved"],
        slate_candidates=int(len(slate_ix)),
        revealed_indices=[int(x) for x in slate_ix[:4]],
        unrevealed_indices=[int(x) for x in slate_ix[4:]],
        loss_value=float(loss_val.detach().cpu().item()),
        optimizer_steps=1,
        trainable_parameters=int(trainable_count),
        adapter_sha256_before=adapter_before,
        adapter_sha256_after=adapter_after,
        dope_sha256_before=dope_before,
        dope_sha256_after=dope_after,
    )


def _run_bounded_subprocess(command, *, deadline_spec, timeout=90.0, **kwargs):
    """Bound one native/AWS operation by the existing shared deadline, never a fresh window."""
    limit = mon.remaining_subprocess_timeout(time.monotonic(), deadline_spec, timeout)
    try:
        return subprocess.run(command, timeout=limit, **kwargs)
    finally:
        mon.enforce_shared_process_deadline(
            time.monotonic(), deadline_spec, stage="after_native_or_aws_subprocess"
        )


def _s3_get_object_with_version(
    bucket, key, dest_path, aws_profile="veox-jepson-prod", *, deadline_spec=None
):
    dest_p = Path(dest_path)
    dest_p.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["AWS_PROFILE"] = aws_profile
    cmd = [
        "aws",
        "s3api",
        "get-object",
        "--bucket",
        bucket,
        "--key",
        key,
        str(dest_p),
    ]
    proc = _run_bounded_subprocess(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        check=True,
        deadline_spec=deadline_spec,
    )
    meta = json.loads(proc.stdout)
    mon.enforce_shared_process_deadline(
        time.monotonic(), deadline_spec, stage="after_native_or_s3_verification"
    )
    return dict(
        bucket=bucket,
        key=key,
        version_id=str(meta.get("VersionId", "")),
        size_bytes=int(dest_p.stat().st_size),
        sha256=sha(dest_p),
    )


def _s3_put_create_only_and_verify(
    bucket,
    key,
    src_path,
    aws_profile="veox-jepson-prod",
    replay_dest_path=None,
    deadline_spec=None,
):
    src_p = Path(src_path)
    local_sha = sha(src_p)
    env = os.environ.copy()
    env["AWS_PROFILE"] = aws_profile
    put_cmd = [
        "aws",
        "s3api",
        "put-object",
        "--bucket",
        bucket,
        "--key",
        key,
        "--body",
        str(src_p),
        "--if-none-match",
        "*",
    ]
    put_proc = _run_bounded_subprocess(
        put_cmd,
        capture_output=True,
        text=True,
        env=env,
        check=True,
        deadline_spec=deadline_spec,
    )
    put_meta = json.loads(put_proc.stdout)
    ver_id = str(put_meta.get("VersionId", "")).strip()
    if not ver_id:
        raise RuntimeError(
            f"put-object returned empty VersionId for s3://{bucket}/{key}"
        )

    if replay_dest_path is not None:
        dl_p = Path(replay_dest_path)
        dl_p.parent.mkdir(parents=True, exist_ok=True)
        get_cmd = [
            "aws",
            "s3api",
            "get-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--version-id",
            ver_id,
            str(dl_p),
        ]
        _run_bounded_subprocess(
            get_cmd,
            capture_output=True,
            text=True,
            env=env,
            check=True,
            deadline_spec=deadline_spec,
        )
        dl_sha = sha(dl_p)
        dl_size = int(dl_p.stat().st_size)
    else:
        with tempfile.NamedTemporaryFile(prefix="h205_s3_verify_", delete=True) as tf:
            get_cmd = [
                "aws",
                "s3api",
                "get-object",
                "--bucket",
                bucket,
                "--key",
                key,
                "--version-id",
                ver_id,
                tf.name,
            ]
            _run_bounded_subprocess(
                get_cmd,
                capture_output=True,
                text=True,
                env=env,
                check=True,
                deadline_spec=deadline_spec,
            )
            dl_sha = sha(tf.name)
            dl_size = int(Path(tf.name).stat().st_size)

    if dl_sha != local_sha:
        raise RuntimeError(
            f"S3 exact-VersionId readback SHA256 mismatch for {key}: {dl_sha} != {local_sha}"
        )
    mon.enforce_shared_process_deadline(
        time.monotonic(), deadline_spec, stage="after_native_or_s3_verification"
    )
    return dict(
        name=src_p.name,
        bucket=bucket,
        key=key,
        version_id=ver_id,
        size_bytes=dl_size,
        local_sha256=local_sha,
        downloaded_version_sha256=dl_sha,
    )


def verify_final_bundle_before_summary(
    *,
    s3_bucket,
    s3_prefix,
    bundle_dir,
    replay_dir,
    deadline_spec,
    _s3_put_create_only_and_verify,
    stat,
    mon,
    time,
    nq0_enc,
    expected_eval_bin_sha,
    roster,
):
    """Verify final replay and deadline before the success summary is built."""
    custody_upload = _s3_put_create_only_and_verify(
        s3_bucket,
        f"{s3_prefix}/s3_custody_receipt.json",
        bundle_dir / "s3_custody_receipt.json",
        replay_dest_path=replay_dir / "s3_custody_receipt.json",
        deadline_spec=deadline_spec,
    )
    verified_bundle = stat.verify_single_context_regression_seam_bundle(
        replay_dir,
        expected_train_sha256=nq0_enc["csv_sha256"],
        expected_train_matrix_sha256=nq0_enc["train_matrix_sha256"],
        expected_train_rows=nq0_enc["nt"],
        expected_evaluator_binary_sha256=expected_eval_bin_sha,
        forbidden_cids=roster["forbidden_cids"],
        forbidden_families=roster["forbidden_families"],
        forbidden_datasets=roster["forbidden_datasets"],
    )
    mon.enforce_shared_process_deadline(
        time.monotonic(), deadline_spec, stage="after_final_bundle_verification"
    )
    return custody_upload, verified_bundle


def construct_and_consume_single_regression_context(
    out,
    launch,
    manifest,
    deadline_spec=None,
    worker_bin=stat.EVALUATOR_BINARY_PATH,
    verify_worker_sha256=True,
    candidate_records=None,
    *,
    use_train_padding_bridge=True,
    require_hold_release_for_live=False,
):
    """Execute the H205 single-context regression data construction + pre-consumption S3 replay + inherited-engine consumption milestone.

    Enforces:
    1. Real exclusion schemas (`JRB01_SEALED_FAMILY_EXCLUSION.json`, `sealed_family_manifest.json`, `d4-role-candidate-manifest.json`,
       historical `selected.json`/`cohort.json`) and transitive family/duplicate closure.
    2. Exact `manifest.json` (`files.train.has_header`, `rows`, `columns`, `shape.features`, `sha256`) and same-`TRAIN`
       physical row/matrix/tensor identity across `train.csv`, `features.npz`, `pool.npz`, `native_evaluator_receipt.json`,
       and `context_manifest.json`.
    3. Raw `nq == 0` protocol probe (`probe_s1main_nq0_protocol_blocker`) retained as a negative control (`use_train_padding_bridge=False`
       stops fail-closed with `status = "BLOCKED_NQ0_PROTOCOL"` and `optimizer_steps = 0`), while the prospective H206-compatible
       `TRAIN`-only `nq=0 -> nq=10` padding bridge (`use_train_padding_bridge=True`) constructs `padded_jtt1_nq10` from the
       exact `nq = 0` `TRAIN` body with 10 cyclic `TRAIN` feature rows and zero query targets (`padding_examples_added_to_learner = 0`).
    4. Create-only S3 persistence (`--if-none-match "*"`) and exact-`VersionId` replay into `verified_s3_replay/` verified by
       `stat.verify_pre_consumption_s3_replay_bundle` BEFORE `consume_single_fresh_context_in_inherited_engine` is invoked.
    """
    t_start = (
        float(deadline_spec["process_start_monotonic"])
        if deadline_spec is not None
        else time.monotonic()
    )
    if deadline_spec is None:
        deadline_spec = validate_launch(
            launch, sha(HERE / "manifest.json"), process_start_monotonic=t_start
        )
    mon.enforce_shared_process_deadline(
        time.monotonic(), deadline_spec, stage="single_regression_context_entry"
    )
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        try:
            os.nice(19 - os.getpriority(os.PRIO_PROCESS, 0))
        except OSError:
            pass

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cid = stat.TARGET_REGRESSION_CONTEXT_SHA256
    ds_id = stat.TARGET_REGRESSION_DATASET_ID
    fam_id = stat.TARGET_REGRESSION_FAMILY_ID
    fam_key = stat.TARGET_REGRESSION_FAMILY_KEY
    bundle_dir = out / cid
    bundle_dir.mkdir(parents=True, exist_ok=True)

    if require_hold_release_for_live:
        hold_rel_path = launch.get("hold_release_receipt")
        hold_released = False
        if hold_rel_path and Path(hold_rel_path).is_file():
            rel_obj = json.loads(Path(hold_rel_path).read_text(encoding="utf-8"))
            hold_released = (
                isinstance(rel_obj, dict)
                and rel_obj.get("status") == "RELEASED"
                and rel_obj.get("wave") == stat.WAVE
            )
        if not hold_released:
            elapsed_wall = float(time.monotonic() - t_start)
            hold_summary = dict(
                status="HOLD_ACTIVE_PRE_SCORING",
                seam_qualified=False,
                wave=stat.WAVE,
                mode="construct_single_regression_context",
                dataset_id=ds_id,
                family_id=fam_id,
                family_key=fam_key,
                context_sha256=cid,
                train_padding_bridge=dict(
                    schema_version=stat.TRAIN_PADDING_BRIDGE_SCHEMA,
                    query_padding_rule=stat.TRAIN_PADDING_QUERY_RULE,
                    protocol_sha256=stat.TRAIN_PADDING_PROTOCOL_SHA256,
                    original_jtt1_nq0_sha256=stat.TARGET_REGRESSION_NQ0_JTT1_SHA256,
                    padded_jtt1_nq10_sha256=stat.TARGET_REGRESSION_PADDED_NQ10_JTT1_SHA256,
                    cut_sha256=stat.TARGET_REGRESSION_CUT_SHA256,
                    padding_examples_added_to_learner=0,
                    padding_families_added=0,
                ),
                h206_readiness_claimed_from_tokens_alone=False,
                native_calls_total=0,
                native_failures_total=0,
                optimizer_steps=0,
                adapter_updated=False,
                elapsed_wall_seconds=elapsed_wall,
                prior_attempts_elapsed_seconds=deadline_spec["prior_elapsed_seconds"],
                cumulative_elapsed_wall_seconds=deadline_spec["prior_elapsed_seconds"]
                + elapsed_wall,
                remaining_wall_budget_seconds=deadline_spec["remaining_budget_seconds"]
                - elapsed_wall,
                utc=datetime.now(timezone.utc).isoformat(),
            )
            save(out / "single_regression_context_summary.json", hold_summary)
            print(json.dumps(hold_summary), flush=True)
            return hold_summary

    # 1. Freeze source-bound allowed regression TRAIN roster and verify target candidate
    roster = stat.build_allowed_regression_train_roster(verify_hashes=True)
    mon.enforce_shared_process_deadline(
        time.monotonic(), deadline_spec, stage="after_roster_construction"
    )
    roster_serializable = dict(
        status=roster["status"],
        wave=roster["wave"],
        fpool_disjoint_datasets_count=roster["fpool_disjoint_datasets_count"],
        transitive_disjoint_datasets_count=roster["transitive_disjoint_datasets_count"],
        allowed_datasets_count=roster["allowed_datasets_count"],
        allowed_families_count=roster["allowed_families_count"],
        skipped_missing_family_count=roster["skipped_missing_family_count"],
        skipped_sibling_duplicate_count=roster["skipped_sibling_duplicate_count"],
        forbidden_datasets_count=roster["forbidden_datasets_count"],
        forbidden_family_aliases_count=roster["forbidden_family_aliases_count"],
        forbidden_cids_count=roster["forbidden_cids_count"],
        exclusion_source_counts=roster["exclusion_source_counts"],
        roster_sha256=roster["roster_sha256"],
        target_dataset_id=ds_id,
        target_family_id=fam_id,
        target_context_sha256=cid,
        records=roster["records"],
    )
    save(out / "allowed_regression_train_roster.json", roster_serializable)

    cand_check = stat.verify_single_regression_train_candidate(
        dict(
            dataset_id=ds_id,
            family_id=fam_id,
            family_key=fam_key,
            context_sha256=cid,
            task="regression",
            role="regression_train",
            source_split="train",
            dl_split="DL-Train",
            has_meta_test=False,
        ),
        roster,
    )

    # 2. Fetch ONE permitted original S3 TRAIN object (never test.csv) & encode according to manifest.json
    raw_intake_dir = out / "raw_s3_intake" / ds_id
    raw_intake_dir.mkdir(parents=True, exist_ok=True)
    ds_manifest_s3 = _s3_get_object_with_version(
        "veox-jopedime-datasets-use2",
        f"datasets/{ds_id}/manifest.json",
        raw_intake_dir / "manifest.json",
        deadline_spec=deadline_spec,
    )
    train_csv_s3 = _s3_get_object_with_version(
        "veox-jopedime-datasets-use2",
        f"blobs/sha256/{stat.TARGET_REGRESSION_TRAIN_SHA256}",
        raw_intake_dir / "train.csv",
        deadline_spec=deadline_spec,
    )
    if train_csv_s3["sha256"] != stat.TARGET_REGRESSION_TRAIN_SHA256:
        raise RuntimeError("downloaded S3 train.csv SHA256 mismatch")
    mon.enforce_shared_process_deadline(
        time.monotonic(), deadline_spec, stage="after_s3_train_intake"
    )

    nq0_jtt1_path = raw_intake_dir / f"{ds_id}_train_only_nq0.jtt1"
    nq0_enc = stat.encode_train_only_jtt1_from_csv(
        raw_intake_dir / "train.csv",
        out_jtt1_path=nq0_jtt1_path,
        expected_csv_sha256=stat.TARGET_REGRESSION_TRAIN_SHA256,
        manifest_path=raw_intake_dir / "manifest.json",
    )

    # 3. Load frozen r882 + DOPE network and extract frozen Stage-1A/1B tokens on the exact S3 train.csv rows
    h170 = load_h170_parent()
    rt = h170.runtime()
    ckpt_path = Path(manifest["checkpoint"])
    if sha(ckpt_path) != stat.R882_SHA256:
        raise RuntimeError("r882 checkpoint SHA256 mismatch")
    net, guard, names, frozen, fused = rt.build(ckpt_path)
    rt.check_frozen(net, guard, names, frozen)

    sys.path.insert(0, str(ROOT / "research/flywheel_selfplay"))
    import run_fused_closed_loop as base_loop  # noqa: E402

    ws = base_loop.FusedLandmarkWorkspace(
        num_threads=int(launch.get("numeric_threads", 2))
    )

    raw_csv_tokens = extract_frozen_dope_tokens_10x32(
        net,
        ws,
        nq0_enc["X_train"],
        nq0_enc["y_train"],
        device=torch.device("cpu"),
        deadline_spec=deadline_spec,
    )
    raw_csv_features_sha = hashlib.sha256(
        raw_csv_tokens["tokens_10x32"].tobytes() + raw_csv_tokens["mask_10"].tobytes()
    ).hexdigest()

    # 4. Probe bootstrap.s1main --fit-s1 on the exact 1,000-row nq=0 S3 train.csv JTT1 table as a negative control.
    #    When `use_train_padding_bridge=False`, fail closed (`status = "BLOCKED_NQ0_PROTOCOL"`, `optimizer_steps = 0`)
    #    if nq=0 is rejected; NEVER substitute the local 1,600-row s1-prod fa480de3... context!
    if not use_train_padding_bridge:
        probe_timeout = mon.remaining_subprocess_timeout(
            time.monotonic(), deadline_spec, 30.0
        )
        nq0_probe = stat.probe_s1main_nq0_protocol_blocker(
            nq0_jtt1_path,
            timeout_seconds=probe_timeout,
            worker_bin=worker_bin,
            expected_worker_sha256=(
                stat.EVALUATOR_BINARY_SHA256 if verify_worker_sha256 else None
            ),
        )
        mon.enforce_shared_process_deadline(
            time.monotonic(), deadline_spec, stage="after_nq0_protocol_probe"
        )
    else:
        nq0_probe = dict(
            evaluator=stat.NATIVE_RUST_EVALUATOR_ID,
            evaluator_binary_sha256=stat.EVALUATOR_BINARY_SHA256,
            jtt1_sha256=nq0_enc["jtt1_sha256"],
            header=dict(
                nt=nq0_enc["nt"], nq=nq0_enc["nq"], p=nq0_enc["p"], k=nq0_enc["k"]
            ),
            returncode=1,
            nq0_refused_by_header_guard=True,
        )
    if (not use_train_padding_bridge) and nq0_probe["returncode"] != 0:
        elapsed_wall = float(time.monotonic() - t_start)
        blocker_summary = dict(
            status="BLOCKED_NQ0_PROTOCOL",
            seam_qualified=False,
            wave=stat.WAVE,
            mode="construct_single_regression_context",
            dataset_id=ds_id,
            family_id=fam_id,
            family_key=fam_key,
            context_sha256=cid,
            fpool_disjoint_datasets_count=roster["fpool_disjoint_datasets_count"],
            transitive_disjoint_datasets_count=roster[
                "transitive_disjoint_datasets_count"
            ],
            allowed_roster_datasets=roster["allowed_datasets_count"],
            allowed_roster_families=roster["allowed_families_count"],
            roster_sha256=roster["roster_sha256"],
            source_train_s3=train_csv_s3,
            source_manifest_s3=ds_manifest_s3,
            pure_s3_train_csv_nq0_protocol_check=dict(
                csv_sha256=nq0_enc["csv_sha256"],
                train_matrix_sha256=nq0_enc["train_matrix_sha256"],
                jtt1_nq0_sha256=nq0_enc["jtt1_sha256"],
                cut_sha256=nq0_enc["cut_sha256"],
                nt=nq0_enc["nt"],
                nq=nq0_enc["nq"],
                p=nq0_enc["p"],
                k=nq0_enc["k"],
                has_header=nq0_enc["has_header"],
                raw_csv_features_sha256=raw_csv_features_sha,
                s1main_probe=nq0_probe,
            ),
            local_s1prod_fallback_used=False,
            s3_persisted_and_replayed_before_optimizer_step=False,
            optimizer_steps=0,
            adapter_updated=False,
            elapsed_wall_seconds=elapsed_wall,
            prior_attempts_elapsed_seconds=deadline_spec["prior_elapsed_seconds"],
            cumulative_elapsed_wall_seconds=deadline_spec["prior_elapsed_seconds"]
            + elapsed_wall,
            remaining_wall_budget_seconds=deadline_spec["remaining_budget_seconds"]
            - elapsed_wall,
            utc=datetime.now(timezone.utc).isoformat(),
        )
        save(bundle_dir / "nq0_protocol_blocker_receipt.json", blocker_summary)
        save(out / "single_regression_context_summary.json", blocker_summary)
        print(json.dumps(blocker_summary), flush=True)
        return blocker_summary

    # 5. Construct the H206-compatible TRAIN-only padded native JTT1 (`nq = 10`) from the exact `nq = 0` TRAIN body
    #    (when `use_train_padding_bridge=True`), retaining all original TRAIN rows/folds unchanged and hashing the
    #    padded native input separately (`padded_jtt1_nq10_sha256 != original_jtt1_nq0_sha256`).
    if use_train_padding_bridge:
        padded_nq10_jtt1_path = raw_intake_dir / f"{ds_id}_train_only_padded_nq10.jtt1"
        padding_bridge = stat.build_train_only_padded_native_jtt1(
            nq0_jtt1_path,
            out_padded_jtt1_path=padded_nq10_jtt1_path,
            expected_nq0_sha256=nq0_enc["jtt1_sha256"],
            expected_cut_sha256=nq0_enc["cut_sha256"],
            expected_train_matrix_sha256=nq0_enc["train_matrix_sha256"],
            expected_nt=nq0_enc["nt"],
            expected_p=nq0_enc["p"],
            worker_bin=worker_bin,
            expected_worker_sha256=(
                stat.EVALUATOR_BINARY_SHA256 if verify_worker_sha256 else None
            ),
        )
        derived_cid = stat.derive_same_s3_train_context_sha256(
            dataset_id=ds_id,
            family_id=fam_id,
            train_sha256=nq0_enc["csv_sha256"],
            train_matrix_sha256=nq0_enc["train_matrix_sha256"],
            original_jtt1_nq0_sha256=padding_bridge["original_jtt1_nq0_sha256"],
            cut_sha256=padding_bridge["cut_sha256"],
            padded_jtt1_nq10_sha256=padding_bridge["padded_jtt1_nq10_sha256"],
            protocol_sha256=padding_bridge["protocol_sha256"],
        )
        if derived_cid != cid:
            cid = derived_cid
            bundle_dir = out / cid
            bundle_dir.mkdir(parents=True, exist_ok=True)
        native_eval_jtt1_path = padded_nq10_jtt1_path
        s1_input_sha = padding_bridge["padded_jtt1_nq10_sha256"]
    else:
        padding_bridge = None
        native_eval_jtt1_path = nq0_jtt1_path
        s1_input_sha = nq0_enc["jtt1_sha256"]

    cands_to_eval = (
        list(candidate_records)
        if candidate_records is not None
        else _default_same_train_candidate_records(feature_columns=nq0_enc["p"])
    )
    ordered_records = sorted(
        cands_to_eval, key=lambda r: r["program"]["program_sha256"]
    )
    ordered_pshas = [r["program"]["program_sha256"] for r in ordered_records]
    dsl_list = [r["program"]["dsl"] for r in ordered_records]
    prog_mat, proj16_mat = featurize_dsl_412_and_proj16(dsl_list)

    live_s1_check = evaluate_slate24_live_s1main(
        native_eval_jtt1_path,
        ordered_records,
        worker_bin=worker_bin,
        deadline_spec=deadline_spec,
        verify_worker_sha256=verify_worker_sha256,
    )
    fitness_arr = live_s1_check.pop("fitness_array")
    logratio_arr = live_s1_check.pop("logratio_array")
    gate_arr = live_s1_check.pop("gate_array")

    tok_10x32 = raw_csv_tokens["tokens_10x32"]
    mask_10 = raw_csv_tokens["mask_10"]

    pool_npz_path = bundle_dir / "pool.npz"
    features_npz_path = bundle_dir / "features.npz"
    np.savez(
        pool_npz_path,
        program=prog_mat,
        proj16=proj16_mat,
        fitness=fitness_arr,
        logratio=logratio_arr,
        gate=gate_arr,
        program_sha256=np.asarray(ordered_pshas),
        tokens_10x32=tok_10x32,
        mask_10=mask_10,
        train_rows=np.asarray(nq0_enc["nt"], dtype=np.int64),
        train_sha256=np.asarray(nq0_enc["csv_sha256"]),
        train_matrix_sha256=np.asarray(nq0_enc["train_matrix_sha256"]),
    )
    np.savez(
        features_npz_path,
        cids=np.asarray([cid]),
        tokens_10x32=tok_10x32,
        mask_10=mask_10,
        dope629=raw_csv_tokens["dope629"],
        lm48_z=raw_csv_tokens["lm48_z"],
        ctx_feat_584=raw_csv_tokens["ctx_feat_584"],
        train_rows=np.asarray(nq0_enc["nt"], dtype=np.int64),
        train_sha256=np.asarray(nq0_enc["csv_sha256"]),
        train_matrix_sha256=np.asarray(nq0_enc["train_matrix_sha256"]),
    )

    labels_sha = hashlib.sha256(fitness_arr.tobytes() + gate_arr.tobytes()).hexdigest()
    programs_sha = hashlib.sha256(prog_mat.tobytes()).hexdigest()
    features_sha = hashlib.sha256(tok_10x32.tobytes() + mask_10.tobytes()).hexdigest()
    pool_sha = sha(pool_npz_path)
    feat_npz_sha = sha(features_npz_path)

    native_eval_receipt = dict(
        status="PASS",
        evaluator=stat.NATIVE_RUST_EVALUATOR_ID,
        evaluator_binary_sha256=live_s1_check["evaluator_binary_sha256"],
        context_sha256=cid,
        dataset_id=ds_id,
        family_id=fam_id,
        family_key=fam_key,
        train_sha256=nq0_enc["csv_sha256"],
        train_matrix_sha256=nq0_enc["train_matrix_sha256"],
        train_rows=int(nq0_enc["nt"]),
        has_header=bool(nq0_enc["has_header"]),
        feature_columns=int(nq0_enc["p"]),
        original_jtt1_nq0_sha256=nq0_enc["jtt1_sha256"],
        cut_sha256=nq0_enc["cut_sha256"],
        padded_jtt1_nq10_sha256=(
            padding_bridge["padded_jtt1_nq10_sha256"]
            if padding_bridge is not None
            else ""
        ),
        padding_queries_sha256=(
            padding_bridge["padding_queries_sha256"]
            if padding_bridge is not None
            else ""
        ),
        train_padding_bridge_protocol_sha256=(
            padding_bridge["protocol_sha256"] if padding_bridge is not None else ""
        ),
        native_rust_source_sha256=stat.NATIVE_RUST_SOURCE_SHA256,
        metric_source_sha256=stat.METRIC_EVALUATOR_SOURCE_SHA256,
        s1_input_sha256=s1_input_sha,
        training_block_sha256=nq0_enc["jtt1_sha256"],
        labels_sha256=labels_sha,
        programs_sha256=programs_sha,
        features_sha256=features_sha,
        dope_sha256=stat.DOPE_SHA256,
        r882_sha256=stat.R882_SHA256,
        candidates_count=len(ordered_records),
        evaluated_candidates_count=int(live_s1_check["evaluated_candidates"]),
        native_calls_total=int(live_s1_check["evaluated_candidates"]) + 1,
        native_failures_total=1 if nq0_probe["returncode"] != 0 else 0,
        cache_hits=0,
        cache_misses=int(live_s1_check["evaluated_candidates"]),
        cache_hit_bytes=0,
        cache_miss_bytes=0,
        family_novelty_is_new_independent_family=True,
        padding_examples_added_to_learner=0,
        padding_families_added=0,
        outer_test_rows_decoded=0,
        train_padding_bridge=padding_bridge,
        live_s1main_slate24_verification=live_s1_check,
        pure_s3_train_csv_nq0_protocol_check=dict(
            csv_sha256=nq0_enc["csv_sha256"],
            train_matrix_sha256=nq0_enc["train_matrix_sha256"],
            jtt1_nq0_sha256=nq0_enc["jtt1_sha256"],
            cut_sha256=nq0_enc["cut_sha256"],
            nt=nq0_enc["nt"],
            nq=nq0_enc["nq"],
            p=nq0_enc["p"],
            k=nq0_enc["k"],
            has_header=nq0_enc["has_header"],
            raw_csv_features_sha256=raw_csv_features_sha,
            s1main_probe=nq0_probe,
        ),
        utc=datetime.now(timezone.utc).isoformat(),
    )
    save(bundle_dir / "native_evaluator_receipt.json", native_eval_receipt)

    context_manifest = dict(
        wave=stat.WAVE,
        status="PASS",
        dataset_id=ds_id,
        family_id=fam_id,
        family_key=fam_key,
        family_aliases=sorted(
            {fam_id, fam_key, f"dedupe:{fam_id}", f"dedupe:{fam_key}"}
        ),
        context_sha256=cid,
        task="regression",
        role="regression_train",
        source_split="train",
        dl_split="DL-Train",
        has_meta_test=False,
        outer_test_rows_decoded=0,
        train_sha256=nq0_enc["csv_sha256"],
        train_matrix_sha256=nq0_enc["train_matrix_sha256"],
        train_rows=int(nq0_enc["nt"]),
        has_header=bool(nq0_enc["has_header"]),
        feature_columns=int(nq0_enc["p"]),
        original_jtt1_nq0_sha256=nq0_enc["jtt1_sha256"],
        cut_sha256=nq0_enc["cut_sha256"],
        padded_jtt1_nq10_sha256=(
            padding_bridge["padded_jtt1_nq10_sha256"]
            if padding_bridge is not None
            else ""
        ),
        padding_queries_sha256=(
            padding_bridge["padding_queries_sha256"]
            if padding_bridge is not None
            else ""
        ),
        train_padding_bridge_protocol_sha256=(
            padding_bridge["protocol_sha256"] if padding_bridge is not None else ""
        ),
        padding_examples_added_to_learner=0,
        padding_families_added=0,
        family_novelty_is_new_independent_family=True,
        s1_input_sha256=s1_input_sha,
        training_block_sha256=nq0_enc["jtt1_sha256"],
        pool_sha256=pool_sha,
        features_npz_sha256=feat_npz_sha,
        labels_sha256=labels_sha,
        programs_sha256=programs_sha,
        features_sha256=features_sha,
        roster_sha256=roster["roster_sha256"],
        candidate_admission=cand_check,
        utc=datetime.now(timezone.utc).isoformat(),
    )
    save(bundle_dir / "context_manifest.json", context_manifest)

    # 6. Persist data & native receipts create-only to S3 and replay by exact VersionId into verified_s3_replay/
    #    BEFORE any inherited-engine consumption or optimizer step!
    s3_bucket = "veox-jopedime-learning-use2"
    utc_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    s3_prefix = (
        f"experiments/rd_breakthrough_20261101/H205/regression-seam/{cid}-{utc_tag}"
    )
    replay_dir = bundle_dir / "verified_s3_replay"
    replay_dir.mkdir(parents=True, exist_ok=True)

    pre_consumption_uploaded = []
    for fname in (
        "context_manifest.json",
        "native_evaluator_receipt.json",
        "pool.npz",
        "features.npz",
    ):
        mon.enforce_shared_process_deadline(
            time.monotonic(), deadline_spec, stage=f"s3_put_pre_{fname}"
        )
        pre_consumption_uploaded.append(
            _s3_put_create_only_and_verify(
                s3_bucket,
                f"{s3_prefix}/{fname}",
                bundle_dir / fname,
                replay_dest_path=replay_dir / fname,
                deadline_spec=deadline_spec,
            )
        )

    pre_custody = dict(
        status="PASS",
        wave=stat.WAVE,
        stage="pre_consumption_s3_replay",
        persisted_and_replayed_before_optimizer_step=True,
        source_dataset_manifest=ds_manifest_s3,
        source_train_object=train_csv_s3,
        test_object_fetched=False,
        learning_store_bucket=s3_bucket,
        learning_store_prefix=s3_prefix,
        uploaded_bundle_objects=pre_consumption_uploaded,
        utc=datetime.now(timezone.utc).isoformat(),
    )
    save(bundle_dir / "s3_pre_consumption_custody_receipt.json", pre_custody)
    pre_custody_upload = _s3_put_create_only_and_verify(
        s3_bucket,
        f"{s3_prefix}/s3_pre_consumption_custody_receipt.json",
        bundle_dir / "s3_pre_consumption_custody_receipt.json",
        replay_dest_path=replay_dir / "s3_pre_consumption_custody_receipt.json",
        deadline_spec=deadline_spec,
    )

    expected_eval_bin_sha = (
        stat.EVALUATOR_BINARY_SHA256
        if verify_worker_sha256
        else live_s1_check["evaluator_binary_sha256"]
    )
    mon.enforce_shared_process_deadline(
        time.monotonic(), deadline_spec, stage="pre_consumption_s3_replay_verify"
    )
    pre_replay_verified = stat.verify_pre_consumption_s3_replay_bundle(
        replay_dir,
        expected_train_sha256=nq0_enc["csv_sha256"],
        expected_train_matrix_sha256=nq0_enc["train_matrix_sha256"],
        expected_train_rows=nq0_enc["nt"],
        expected_evaluator_binary_sha256=expected_eval_bin_sha,
        forbidden_cids=roster["forbidden_cids"],
        forbidden_families=roster["forbidden_families"],
        forbidden_datasets=roster["forbidden_datasets"],
    )

    # 7. ONLY AFTER S3 create-only upload + exact-VersionId replay verification succeeds,
    #    consume the replayed S3 bundle in the inherited h170.Engine!
    replayed_cm = json.loads((replay_dir / "context_manifest.json").read_text())
    replayed_nr = json.loads((replay_dir / "native_evaluator_receipt.json").read_text())
    train_hist, val_hist = h170.cohorts(rt)
    historical_cids = {r["context_sha256"] for r in (train_hist + val_hist)}
    engine = h170.Engine(net, rt, names, frozen, guard, fused, train_hist + val_hist)
    fresh_row = dict(
        replayed_cm,
        pool_npz_path=str(replay_dir / "pool.npz"),
        native_evaluator_receipt=replayed_nr,
    )
    eng_receipt = consume_single_fresh_context_in_inherited_engine(
        engine,
        net,
        rt,
        fresh_row,
        historical_cids=historical_cids,
        deadline_spec=deadline_spec,
    )
    eng_receipt["s3_persisted_and_replayed_before_optimizer_step"] = True
    eng_receipt["consumed_pool_s3_version_id"] = pre_replay_verified[
        "version_ids_by_name"
    ]["pool.npz"]
    eng_receipt["consumed_features_s3_version_id"] = pre_replay_verified[
        "version_ids_by_name"
    ]["features.npz"]
    eng_receipt["pre_consumption_s3_replay_sha256"] = sha(
        replay_dir / "s3_pre_consumption_custody_receipt.json"
    )
    save(bundle_dir / "engine_consumption_receipt.json", eng_receipt)

    eng_upload = _s3_put_create_only_and_verify(
        s3_bucket,
        f"{s3_prefix}/engine_consumption_receipt.json",
        bundle_dir / "engine_consumption_receipt.json",
        replay_dest_path=replay_dir / "engine_consumption_receipt.json",
        deadline_spec=deadline_spec,
    )
    uploaded_objects = list(pre_consumption_uploaded) + [
        pre_custody_upload,
        eng_upload,
    ]

    elapsed_wall = float(time.monotonic() - t_start)
    s3_custody = dict(
        status="PASS",
        wave=stat.WAVE,
        source_dataset_manifest=ds_manifest_s3,
        source_train_object=train_csv_s3,
        test_object_fetched=False,
        learning_store_bucket=s3_bucket,
        learning_store_prefix=s3_prefix,
        uploaded_bundle_objects=uploaded_objects,
        elapsed_wall_seconds=elapsed_wall,
        prior_attempts_elapsed_seconds=deadline_spec["prior_elapsed_seconds"],
        cumulative_elapsed_wall_seconds=deadline_spec["prior_elapsed_seconds"]
        + elapsed_wall,
        remaining_wall_budget_seconds=deadline_spec["remaining_budget_seconds"]
        - elapsed_wall,
        campaign_wall_budget_cap_seconds=deadline_spec["campaign_cap_seconds"],
        utc=datetime.now(timezone.utc).isoformat(),
    )
    save(bundle_dir / "s3_custody_receipt.json", s3_custody)

    # 8. Run end-to-end fail-closed bundle verification on the replayed S3 bundle
    # and enforce the shared deadline before any qualified summary is constructed.
    custody_upload, verified_bundle = verify_final_bundle_before_summary(
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        bundle_dir=bundle_dir,
        replay_dir=replay_dir,
        deadline_spec=deadline_spec,
        _s3_put_create_only_and_verify=_s3_put_create_only_and_verify,
        stat=stat,
        mon=mon,
        time=time,
        nq0_enc=nq0_enc,
        expected_eval_bin_sha=expected_eval_bin_sha,
        roster=roster,
    )
    # Final invocation accounting and a successful worker exit are also required;
    # this earlier scientific/custody summary cannot certify its own transport tail.
    elapsed_wall = float(time.monotonic() - t_start)
    summary = dict(
        status="PENDING_FINAL_ACCOUNTING",
        operation_succeeded=True,
        seam_qualified=False,
        qualification_requires_final_accounting=True,
        accounting_invocation_id=deadline_spec.get("accounting_invocation_id"),
        wave=stat.WAVE,
        mode="construct_single_regression_context",
        dataset_id=ds_id,
        family_id=fam_id,
        family_key=fam_key,
        context_sha256=cid,
        fpool_disjoint_datasets_count=roster["fpool_disjoint_datasets_count"],
        transitive_disjoint_datasets_count=roster["transitive_disjoint_datasets_count"],
        allowed_roster_datasets=roster["allowed_datasets_count"],
        allowed_roster_families=roster["allowed_families_count"],
        roster_sha256=roster["roster_sha256"],
        source_train_s3=train_csv_s3,
        source_manifest_s3=ds_manifest_s3,
        pure_s3_train_csv_nq0_protocol_check=native_eval_receipt[
            "pure_s3_train_csv_nq0_protocol_check"
        ],
        pre_consumption_s3_replay=pre_replay_verified,
        s1_prod_context_bundle=verified_bundle,
        engine_consumption=eng_receipt,
        h206_readiness_claimed_from_tokens_alone=False,
        family_novelty_ledger=dict(
            family_novelty_is_new_independent_family=True,
            padding_examples_added_to_learner=0,
            padding_families_added=0,
        ),
        s3_bundle_custody_upload=custody_upload,
        uploaded_bundle_objects=uploaded_objects,
        elapsed_wall_seconds=elapsed_wall,
        prior_attempts_elapsed_seconds=deadline_spec["prior_elapsed_seconds"],
        cumulative_elapsed_wall_seconds=deadline_spec["prior_elapsed_seconds"]
        + elapsed_wall,
        remaining_wall_budget_seconds=deadline_spec["remaining_budget_seconds"]
        - elapsed_wall,
        utc=datetime.now(timezone.utc).isoformat(),
    )
    mon.enforce_shared_process_deadline(
        time.monotonic(), deadline_spec, stage="before_qualified_summary_publication"
    )
    save(out / "single_regression_context_summary.json", summary)
    mon.enforce_shared_process_deadline(
        time.monotonic(), deadline_spec, stage="after_summary_write"
    )
    print(json.dumps(summary), flush=True)
    mon.enforce_shared_process_deadline(
        time.monotonic(), deadline_spec, stage="after_summary_publication"
    )
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--launch", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--construct-single-regression-context", action="store_true")
    args = ap.parse_args()

    manifest_path = HERE / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_sha = sha(manifest_path)
    launch = json.loads(args.launch.read_text())
    if args.construct_single_regression_context:
        if (
            not os.environ.get("H205_ACCOUNTING_INVOCATION_ID")
            or os.environ.get("H205_ACCOUNTING_PARENT_PID") != str(os.getppid())
            or os.environ.get("H205_ACCOUNTING_BOOT_ID")
            != Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        ):
            raise RuntimeError(
                "construction requires the existing accounted monitor; standalone worker is unqualified"
            )
    deadline_spec = validate_launch(launch, manifest_sha)
    deadline_spec["accounting_invocation_id"] = os.environ.get(
        "H205_ACCOUNTING_INVOCATION_ID"
    )
    configure_threads(launch.get("numeric_threads", 2))

    args.output.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        mon.enforce_shared_process_deadline(
            time.monotonic(), deadline_spec, stage="smoke"
        )
        smoke_payload = dict(
            wave=stat.WAVE,
            status="PASS",
            mode="smoke",
            authority=stat.AUTHORITY_STRING,
            manifest_sha256=manifest_sha,
            numeric_threads=torch.get_num_threads(),
            cuda_runtime_threads_exempt=True,
            prior_attempts_elapsed_seconds=deadline_spec["prior_elapsed_seconds"],
            remaining_wall_budget_seconds=deadline_spec["remaining_budget_seconds"],
            campaign_wall_budget_cap_seconds=deadline_spec["campaign_cap_seconds"],
            optimizer_updates=0,
            scored_contexts=0,
        )
        save(args.output / "smoke.json", smoke_payload)
        print(json.dumps(smoke_payload), flush=True)
        return

    if args.construct_single_regression_context:
        construct_and_consume_single_regression_context(
            args.output,
            launch,
            manifest,
            deadline_spec=deadline_spec,
            require_hold_release_for_live=True,
        )
        return

    run_gpu_wave(args.output, launch, manifest, deadline_spec=deadline_spec)


if __name__ == "__main__":
    main()
