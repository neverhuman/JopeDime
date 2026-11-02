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

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

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
        else float(os.environ.get("H205_PROCESS_START_MONOTONIC", time.monotonic()))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--launch", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    manifest_path = HERE / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_sha = sha(manifest_path)
    launch = json.loads(args.launch.read_text())
    deadline_spec = validate_launch(launch, manifest_sha)
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

    run_gpu_wave(args.output, launch, manifest, deadline_spec=deadline_spec)


if __name__ == "__main__":
    main()
