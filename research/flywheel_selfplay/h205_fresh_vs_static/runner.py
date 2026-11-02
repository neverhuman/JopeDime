"""H205 GPU runner on xbabe1 RTX 4090: fresh vs static data & static teacher across 50 rounds from r882.

Executes inside a single GPU worker process on xbabe1 cuda:0 alongside live flywheel PID 2297929:
- Stage 0: Native r882 parity 91/91 exact (1.1377941898164525) on DL-Val before any candidate forward.
- Stage 1: STATIC-DATA (50 rounds from r882 on fixed 1,219 contexts / 205 families with H178 per-round
  pooled cross-fit selector admission) + STATIC-TEACHER frozen-r882 teacher infrastructure initialization.
- Stage 2: Gated on luna_v1_tel's JRB-01 ADMISSION PASS receipt; executes STATIC-TEACHER and FRESH
  (50 rounds each from r882 under equal compute and H178 per-round pooled selector admission).
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


def load_h170_parent():
    path = ROOT / "research/flywheel_selfplay/h170_onecycle/runner.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("h205_h170_parent", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_launch(launch, manifest_sha):
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


def run_gpu_wave(out, launch, manifest):
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

    device_receipt = dict(
        pid=os.getpid(),
        host=os.uname().nodename,
        device=str(next(net.parameters()).device),
        gpu=torch.cuda.get_device_name(0),
        dope_sha256=rt.DOPE,
        r882_sha256=stat.R882_SHA256,
        trainable_parameters=sum(
            p.numel() for p in net.parameters() if p.requires_grad
        ),
        numeric_threads=torch.get_num_threads(),
        cuda_runtime_threads_exempt=True,
        memory_fraction=0.18,
        tf32=False,
        live_flywheel_pid_untouched=mon.LIVE_FLYWHEEL_PID,
        utc=datetime.now(timezone.utc).isoformat(),
    )
    save(out / "gpu-device.json", device_receipt)
    print("GPU DEVICE", json.dumps(device_receipt), flush=True)

    train, val = h170.cohorts(rt)

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

    # Stage 1: Load 1,219 DL-Train pools, evaluate baseline r882 & cache STATIC-TEACHER rankings
    engine.load(train)
    train_base_eval, train_base_raw = engine.evaluate(
        train, native=True, prove_identity=False
    )
    r882_train_values = np.asarray(train_base_eval["values"], dtype=np.float64)
    teacher_orders = {
        cid: np.argsort(cost, kind="mergesort") for cid, cost in train_base_raw.items()
    }
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
    save(
        out / "static_teacher_infra.json",
        dict(
            status="READY",
            teacher_checkpoint_sha256=stat.R882_SHA256,
            cached_contexts=len(teacher_orders),
            r882_train_mean=float(r882_train_values.mean()),
            utc=datetime.now(timezone.utc).isoformat(),
        ),
    )

    initial_state = {
        n: p.detach().cpu().clone()
        for n, p in net.named_parameters()
        if p.requires_grad
    }

    def restore_adapter(state_cpu):
        with torch.no_grad():
            for n, p in net.named_parameters():
                if n in state_cpu:
                    p.copy_(state_cpu[n].to(p.device))

    cid_to_idx = {r["context_sha256"]: i for i, r in enumerate(train)}
    static_b_panel = np.broadcast_to(
        r882_train_values[None, :], (len(stat.SEEDS), len(train))
    ).copy()
    static_a_panel = np.broadcast_to(
        r882_train_values[None, :], (len(stat.SEEDS), len(train))
    ).copy()
    teacher_b_panel = np.broadcast_to(
        r882_train_values[None, :], (len(stat.SEEDS), len(train))
    ).copy()
    teacher_a_panel = np.broadcast_to(
        r882_train_values[None, :], (len(stat.SEEDS), len(train))
    ).copy()

    updates = 0
    round_log = []

    def run_arm_cell(arm_name, si, seed, outer, orders_map):
        nonlocal updates
        fit_rows, b_rows, a_rows = stat.partition(train, outer)
        slates = stat.h177_analysis.fit_slates(fit_rows, orders_map, seed, outer)
        plan = stat.build_equal_compute_round_plan(fit_rows, seed, outer)[arm_name]
        restore_adapter(initial_state)
        params = [p for n, p in net.named_parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=3e-5, betas=(0.9, 0.999), weight_decay=0.01)

        incumbent_state = {
            n: p.detach().cpu().clone()
            for n, p in net.named_parameters()
            if p.requires_grad
        }
        b_indices = [cid_to_idx[r["context_sha256"]] for r in b_rows]
        a_indices = [cid_to_idx[r["context_sha256"]] for r in a_rows]
        incumbent_b_vals = r882_train_values[b_indices].copy()
        admitted_rounds = 0

        fit_cids = sorted(slates.keys())
        slate_anchors = {}
        for start in range(0, len(fit_cids), 32):
            chunk = fit_cids[start : start + 32]
            kw_chunk, _, _, _, _ = engine.batch(chunk, [slates[c] for c in chunk])
            anc = engine.anchored(kw_chunk)
            for idx_c, cid_c in enumerate(chunk):
                slate_anchors[cid_c] = (anc[0][idx_c].clone(), anc[1][idx_c].clone())

        for r_info in plan:
            r_idx = r_info["round"]
            opt.zero_grad(set_to_none=True)
            for mb in r_info["microbatches"]:
                cids = mb["cids"]
                ixs = [slates[c] for c in cids]
                kwargs, tok, mask, fitness, gate = engine.batch(cids, ixs)
                anchor = (
                    torch.stack([slate_anchors[c][0] for c in cids]),
                    torch.stack([slate_anchors[c][1] for c in cids]),
                )
                pred = engine.forward(kwargs, tok, mask, anchor)
                if arm_name == "STATIC-TEACHER":
                    t_cost = torch.stack(
                        [
                            torch.as_tensor(train_base_raw[c][ix], dtype=torch.float32)
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

            if r_idx % 10 == 0 or r_idx == stat.ROUNDS:
                b_eval, _ = engine.evaluate(b_rows, native=False)
                cand_b_vals = np.asarray(b_eval["values"], dtype=np.float64)
                if (
                    float(cand_b_vals.mean())
                    > float(incumbent_b_vals.mean()) + stat.MIN_GAIN_MARGIN
                ):
                    incumbent_b_vals = cand_b_vals
                    incumbent_state = {
                        n: p.detach().cpu().clone()
                        for n, p in net.named_parameters()
                        if p.requires_grad
                    }
                    admitted_rounds += 1
                else:
                    restore_adapter(incumbent_state)

            if r_idx % 5 == 0 or r_idx == 1:
                save(
                    out / "progress.json",
                    dict(
                        wave=stat.WAVE,
                        stage="STAGE1_STATIC_ARMS",
                        arm=arm_name,
                        seed=seed,
                        outer_fold=outer,
                        round=r_idx,
                        total_rounds=stat.ROUNDS,
                        optimizer_updates=updates,
                        admitted_rounds=admitted_rounds,
                        r882_parity_verified=True,
                        utc=datetime.now(timezone.utc).isoformat(),
                    ),
                )

        restore_adapter(incumbent_state)
        a_eval, _ = engine.evaluate(a_rows, native=False)
        return (
            incumbent_b_vals,
            np.asarray(a_eval["values"], dtype=np.float64),
            b_indices,
            a_indices,
            admitted_rounds,
        )

    for si, seed in enumerate(stat.SEEDS):
        for outer in range(5):
            b_vals, a_vals, b_ix, a_ix, adm = run_arm_cell(
                "STATIC-DATA", si, seed, outer, gt_orders
            )
            static_b_panel[si, b_ix] = b_vals
            static_a_panel[si, a_ix] = a_vals
            round_log.append(
                dict(arm="STATIC-DATA", seed=seed, outer=outer, admitted_rounds=adm)
            )

            tb_vals, ta_vals, b_ix, a_ix, t_adm = run_arm_cell(
                "STATIC-TEACHER", si, seed, outer, teacher_orders
            )
            teacher_b_panel[si, b_ix] = tb_vals
            teacher_a_panel[si, a_ix] = ta_vals
            round_log.append(
                dict(
                    arm="STATIC-TEACHER", seed=seed, outer=outer, admitted_rounds=t_adm
                )
            )

    # Stage 2: Gate FRESH arm on JRB-01 ADMISSION PASS receipt
    jrb_receipt_path = Path(
        launch.get(
            "jrb01_admission_receipt",
            "/home/ubuntu/jope_plan/evidence/JRB-01/ADMISSION-PASS.json",
        )
    )
    wait_deadline = time.monotonic() + float(launch.get("fresh_wait_seconds", 18000))
    jrb_gate = None
    while time.monotonic() < wait_deadline:
        try:
            jrb_gate = stat.verify_fresh_admission_gate(jrb_receipt_path)
            break
        except RuntimeError as exc:
            save(
                out / "progress.json",
                dict(
                    wave=stat.WAVE,
                    stage="STAGE2_WAITING_FOR_JRB01_ADMISSION_PASS",
                    optimizer_updates=updates,
                    jrb01_receipt_path=str(jrb_receipt_path),
                    reason=str(exc),
                    r882_parity_verified=True,
                    static_arms_complete=True,
                    utc=datetime.now(timezone.utc).isoformat(),
                ),
            )
            time.sleep(5.0)

    if jrb_gate is None:
        raise RuntimeError(
            f"FRESH arm blocked: JRB-01 ADMISSION PASS receipt not available at {jrb_receipt_path}"
        )

    fresh_b_panel = np.broadcast_to(
        r882_train_values[None, :], (len(stat.SEEDS), len(train))
    ).copy()
    fresh_a_panel = np.broadcast_to(
        r882_train_values[None, :], (len(stat.SEEDS), len(train))
    ).copy()
    for si, seed in enumerate(stat.SEEDS):
        for outer in range(5):
            fb_vals, fa_vals, b_ix, a_ix, f_adm = run_arm_cell(
                "FRESH", si, seed, outer, gt_orders
            )
            fresh_b_panel[si, b_ix] = fb_vals
            fresh_a_panel[si, a_ix] = fa_vals
            round_log.append(
                dict(arm="FRESH", seed=seed, outer=outer, admitted_rounds=f_adm)
            )

    b_lock = stat.h178_round_select(
        train,
        r882_train_values,
        {
            "STATIC-DATA": static_b_panel,
            "STATIC-TEACHER": teacher_b_panel,
            "FRESH": fresh_b_panel,
        },
    )
    save(out / "pooled_b_selection_lock.json", b_lock)

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
        jrb01_admission_gate=jrb_gate,
        pooled_b_selection=b_lock,
        common_gate=gate_summary,
        round_log=round_log,
        verdict=gate_summary["verdict"],
        utc=datetime.now(timezone.utc).isoformat(),
    )
    save(out / "metrics.json", metrics)
    print("COMPLETE", metrics["verdict"], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--launch", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    manifest = json.loads((HERE / "manifest.json").read_text())
    launch = json.loads(args.launch.read_text())
    validate_launch(launch, sha(HERE / "manifest.json"))
    configure_threads(launch.get("numeric_threads", 2))

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        save(
            out / "smoke.json",
            dict(
                wave=stat.WAVE,
                status="PASS",
                authority=stat.AUTHORITY_STRING,
                optimizer_updates=0,
                scored_contexts=0,
                numeric_threads=torch.get_num_threads(),
                cuda_runtime_threads_exempt=True,
                imported_h177_rule_identity=bool(
                    stat.h178_calibration.h177_analysis.pooled_b_select is not None
                ),
            ),
        )
        return

    run_gpu_wave(out, launch, manifest)


if __name__ == "__main__":
    main()
