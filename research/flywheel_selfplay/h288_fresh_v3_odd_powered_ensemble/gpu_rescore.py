"""H286 GPU scoring on xbabe1 (hw_t2325): H283's scorer with only the checkpoint sha256 and round taken as
arguments (as H282 did), so the same path scores r3175 (primary) and r782/r785 (descriptive). H283 docstring follows.

H283 GPU scoring on xbabe1 (hw_t2305): H281's scorer, unchanged on its DL-Val91 path, followed IN THE SAME PROCESS
by the same per-context writer-path forward on the H283 fresh contexts. The fresh phase runs only if the r3175 DL-Val91
regression is exact (Outcome A). Changes from H281: load_dev takes the input root; a --fresh-dir phase; the record says
H283 and stores the fresh costs. Everything else is H281's text.

H281 docstring:

Forward-only replica of the live writer's own dev scoring path (research/flywheel_selfplay/run_fused_closed_loop.py at
486b709c3, which carries evaluation identity e79a7715...): the pinned CPU-exact fused source (sha b9f1468a...) built on
cuda:0 with adapter h40b, the r3175 state_dict loaded strict from a read-only COPY, multimodal tokens recomputed on the
GPU from the train-only tables exactly as the writer prepares them, then `utility()`: per context, the legacy
`predict([c], [arange(pool)])` cost vector, argmin, utility -fitness if gated else -2.0. No optimizer, no gradients,
eval mode, PyTorch defaults as in the writer (TF32 matmul off, cuDNN TF32 default). The writer process, its GPU
context, its lock (~/jope_plan/qualification.lock), outputs and custody are never touched."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import struct  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
MICRO = Path("/home/ubuntu/JopeDimeMicro")
INPUT = Path("/home/ubuntu/jope-data/h50/input")
CKPT_SHA256 = "1d6ea900ebed8d2a05c3b3f37576adc15aadd2b53c585bfd5cb71a103e9ff97b"
DOPE = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
SOURCE_SHA256 = "b9f1468a61aa4dc888c8f2931f9c759f7d4f057793a77ccb05eed6300fb4f9ab"
SELECTED_SHA256_PREFIX = "981960ad628f5029"
WRITER_PID = 1402163
GIB = 2**30
NEED_BYTES = int(
    1.5 * GIB
)  # declared peak need: CUDA context + 7 MB weights + 1-context activations (CPU estimate)
HEADROOM_BYTES = 3 * GIB
ALLOCATOR_CAP_BYTES = (
    2 * GIB
)  # torch.cuda.set_per_process_memory_fraction cap for the caching allocator
SEED = 20261029  # the writer's default seed; scoring is deterministic and draws no randomness


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gpu_memory():
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    free, total = (int(x) * 2**20 for x in out.strip().splitlines()[0].split(","))
    return free, total


def writer_state():
    """Read-only liveness of the writer: /proc state and presence in the GPU compute-app list."""
    status = Path(f"/proc/{WRITER_PID}/status")
    state = None
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("State:"):
                state = line.split()[1]
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).split()
    return dict(
        alive=state is not None,
        state=state,
        on_gpu=str(WRITER_PID) in apps,
        ok=state in ("R", "S", "D") and str(WRITER_PID) in apps,
    )


def admit(free_bytes, need=NEED_BYTES, headroom=HEADROOM_BYTES):
    return free_bytes >= need + headroom


def classify(per_equal, picks_equal, mean, stored_mean, n=91, tol=1e-12):
    """Outcome A only with every context's utility and pick equal and the mean equal to 1e-12."""
    return (
        "A_EXACT"
        if per_equal == n and picks_equal == n and abs(mean - stored_mean) <= tol
        else "B_NOT_EXACT"
    )


def landmark_tokens(net, fused_ws, landmark_float32, tables, cids, device):
    """The writer's token preparation (run_fused_closed_loop.py lines 377-399 at 486b709c3), dev contexts only."""
    token, mask = {}, {}
    for cid in cids:
        x, y = tables[cid]
        yn = (
            (y.astype(np.float64) - np.mean(y.astype(np.float64)))
            / max(float(np.std(y.astype(np.float64))), 1e-8)
        ).astype(np.float32)
        v, col, _ = fused_ws.fit_transform_and_extract_bundle_high(
            landmark_float32(x), np.ascontiguousarray(yn)
        )
        cl = np.zeros((64, 17), np.float32)
        n = min(len(col), 64)
        cl[:n, : min(col.shape[1], 17)] = col[:n, :17]
        with torch.no_grad():
            d, lm = net._run_dope_mlc_v3_and_lm48(
                torch.as_tensor(v, device=device).reshape(1, 256),
                torch.as_tensor(cl, device=device).reshape(1, 64, 17),
                use_landmarks=True,
            )
            tok, msk, _ = net._project_stage1b_tokens_batched_gpu(lm, d)
        token[cid] = tok[0].detach()
        mask[cid] = msk[0].detach()
    return token, mask


def load_dev(evaluator, rows, root=INPUT):
    """The writer's load_packet fields for the DL-Val91 rows (pool and train-only table hashes enforced)."""
    pools, tables = {}, {}
    for rec in rows:
        cid = rec["context_sha256"]
        pp = root / "pools" / f"{cid}.npz"
        if sha(pp) != rec["pool_sha256"]:
            raise RuntimeError("pool hash mismatch")
        with np.load(pp, allow_pickle=False) as z:
            prog = z["program"].astype(np.float32)
            pools[cid] = {
                "pids": [
                    v.decode() if isinstance(v, bytes) else str(v)
                    for v in z["program_sha256"].tolist()
                ],
                "program": prog,
                "proj16": prog.astype(np.float64)
                @ evaluator.get_w1_sign_matrix()
                / np.sqrt(412),
                "fitness": z["fitness"].astype(np.float64),
                "gate": z["gate"].astype(bool),
            }
        path = root / "tables" / f"{cid}.jtt1"
        if sha(path) != rec["train_only_jtt1_sha256"]:
            raise RuntimeError("training input hash mismatch")
        raw = path.read_bytes()
        nt, nq, width = struct.unpack_from("<III", raw, 4)
        if nq or raw[:4] != b"JTT1" or raw[16] != 2:
            raise RuntimeError("requires train-only 2-fold input")
        a = (
            np.frombuffer(raw, dtype="<f8", offset=17 + nt)
            .reshape(nt, width + 1)
            .copy()
        )
        tables[cid] = (a[:, :-1], a[:, -1])
    return pools, tables


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--stored", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fresh-dir", type=Path, required=True)
    ap.add_argument("--round", type=int, required=True)  # H286: as in H282
    ap.add_argument("--sha256", required=True)  # H286: as in H282
    ap.add_argument(
        "--preflight",
        action="store_true",
        help="hashes, imports and inputs only; CUDA is never initialised",
    )
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir()  # exclusive: single attempt
    t0 = time.time()
    record = dict(
        wave="H286",
        step="gpu_rescore_then_fresh",
        round=args.round,
        started_utc=t0,
        host=os.uname().nodename,
    )

    def finish(status, **kw):
        record.update(status=status, finished_utc=time.time(), **kw)
        (out / "rescore.json").write_text(
            json.dumps(record, indent=1, allow_nan=False) + "\n"
        )
        print(status, json.dumps(kw)[:600], flush=True)

    if (
        os.uname().nodename.split(".")[0] != "xbabe1"
        or os.getpriority(os.PRIO_PROCESS, 0) < 10
    ):
        raise RuntimeError("xbabe1 nice>=10 required")
    if sha(args.ckpt) != args.sha256:
        raise RuntimeError("r3175 copy hash")
    stored = json.loads(args.stored.read_text())["dev_utility"]
    w0 = writer_state()
    free, total = gpu_memory()
    record.update(
        writer_before=w0,
        vram_free_before=free,
        vram_total=total,
        need_bytes=NEED_BYTES,
        headroom_bytes=HEADROOM_BYTES,
    )
    if not w0["ok"]:
        return finish("ABORT_WRITER_STATE")
    if not admit(free):
        return finish("NOT_ADMITTED_VRAM")
    sys.path.insert(0, str(MICRO / "python"))
    import rust_evaluator as evaluator
    from hpc.landmark_workspace import FusedLandmarkWorkspace

    spec = importlib.util.spec_from_file_location(
        "freeze_guard", HERE / "dope_freeze_guard.py"
    )
    guard = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = guard
    spec.loader.exec_module(guard)
    sys.path.insert(0, str(HERE))
    from device_pool_cache import landmark_float32

    selected = json.loads((INPUT / "selected.json").read_text())
    if not sha(INPUT / "selected.json").startswith(SELECTED_SHA256_PREFIX):
        raise RuntimeError("selected packet hash")
    dev_rows = selected["dev"]
    if len(dev_rows) != 91 or any(
        r["dl_split"] != "DL-Val"
        or r.get("task") != "regression"
        or r.get("has_meta_test") is True
        for r in dev_rows
    ):
        raise RuntimeError("dev closure")
    dev = [r["context_sha256"] for r in dev_rows]
    pools, tables = load_dev(evaluator, dev_rows)
    source = INPUT / "cpu_exact_fused.py"
    if sha(source) != SOURCE_SHA256:
        raise RuntimeError("CPU-exact fused source drift")
    fs = importlib.util.spec_from_file_location("cpu_exact_fused", source)
    fused = importlib.util.module_from_spec(fs)
    fs.loader.exec_module(fused)
    if args.preflight:
        return finish(
            "PREFLIGHT_OK",
            contexts=len(dev),
            cuda_initialized=torch.cuda.is_initialized(),
        )
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        return finish("ABORT_NO_GPU")
    torch.cuda.set_per_process_memory_fraction(ALLOCATOR_CAP_BYTES / total)
    try:
        net = fused.FusedJopeNetwork(device=device, adapter_variant="h40b")
        for p in net.parameters():
            p.requires_grad_(False)
        guard.lock_dope_weights_and_eval_mode(net)
        before = guard.snapshot_dope_weights(net)
        if before.composite_sha256 != DOPE:
            raise RuntimeError("DOPE before")
        state = torch.load(args.ckpt, map_location=device, weights_only=True)
        if int(state["round"]) != args.round or state.get("admission") == "rejected":
            raise RuntimeError("checkpoint identity")
        net.load_state_dict(state["state_dict"], strict=True)
        del state
        guard.verify_dope_frozen_unchanged(net, before)
        ws = FusedLandmarkWorkspace(num_threads=4)
        token, mask = landmark_tokens(net, ws, landmark_float32, tables, dev, device)

        def predict(cids, selections):
            b = len(cids)
            m = len(selections[0])
            prog = np.stack([pools[c]["program"][s] for c, s in zip(cids, selections)])
            proj = np.stack([pools[c]["proj16"][s] for c, s in zip(cids, selections)])
            prog_tensor = torch.as_tensor(prog, device=device)
            proj_tensor = torch.as_tensor(proj, device=device)
            history_f = torch.as_tensor(
                np.stack(
                    [
                        pools[c]["fitness"][idx[: min(4, m)]]
                        for c, idx in zip(cids, selections)
                    ]
                ),
                dtype=torch.float32,
                device=device,
            )
            history_gate = torch.as_tensor(
                np.stack(
                    [
                        pools[c]["gate"][idx[: min(4, m)]]
                        for c, idx in zip(cids, selections)
                    ]
                ),
                dtype=torch.float32,
                device=device,
            )
            return net.adapter_10tok.predict_costs(
                ctx_feat=torch.zeros((b, 584), device=device),
                prog_412=prog_tensor,
                proj_16=proj_tensor,
                base_prior_cost=torch.zeros((b, m), device=device),
                scale=torch.full((b,), 0.4208, device=device),
                sos_prior=torch.zeros((b, m, 2), device=device),
                mm_tokens_10x32=torch.stack([token[c] for c in cids]),
                mm_mask_10=torch.stack([mask[c] for c in cids]),
                revealed_indices=torch.arange(min(4, m), device=device)
                .reshape(1, -1)
                .expand(b, -1),
                revealed_raw_f=history_f,
                revealed_gate=history_gate,
            )["total_cost"].float()

        net.eval()
        values, choices, costs, idx = [], [], {}, []
        with torch.no_grad():
            for c in dev:
                p = pools[c]
                pred = predict([c], [np.arange(len(p["pids"]))])[0].cpu().numpy()
                i = int(np.argmin(pred))
                costs[c] = pred.astype(np.float32)
                idx.append(i)
                choices.append(p["pids"][i])
                values.append(float(-p["fitness"][i]) if p["gate"][i] else -2.0)
        guard.verify_dope_frozen_unchanged(net, before)
        dl_mean = float(np.mean(values))
        dl_exact = classify(
            int(sum(a == b for a, b in zip(values, stored["per_context"]))),
            int(sum(a == b for a, b in zip(choices, stored["selected_programs"]))),
            dl_mean,
            stored["mean"],
        )
        fresh_record = dict(status="NOT_RUN_REGRESSION_NOT_EXACT")
        if dl_exact == "A_EXACT":
            fresh_rows = json.loads((args.fresh_dir / "selected.json").read_text())[
                "fresh"
            ]
            fresh_ids = [r["context_sha256"] for r in fresh_rows]
            fpools, ftables = load_dev(evaluator, fresh_rows, root=args.fresh_dir)
            pools.update(fpools)
            ftoken, fmask = landmark_tokens(
                net, ws, landmark_float32, ftables, fresh_ids, device
            )
            token.update(ftoken)
            mask.update(fmask)
            net.eval()
            fcosts, fslot0 = {}, []
            with torch.no_grad():
                for c in fresh_ids:
                    p = pools[c]
                    pred = predict([c], [np.arange(len(p["pids"]))])[0].cpu().numpy()
                    i = int(np.argmin(pred))
                    fcosts[c] = pred.astype(np.float32)
                    fslot0.append(float(-p["fitness"][i]) if p["gate"][i] else -2.0)
            guard.verify_dope_frozen_unchanged(net, before)
            np.savez_compressed(out / "fresh-r3175-gpu-costs.npz", **fcosts)
            fresh_record = dict(
                status="COMPLETE",
                contexts=len(fresh_ids),
                costs_sha256=sha(out / "fresh-r3175-gpu-costs.npz"),
                slot0_mean=float(np.mean(fslot0)),
                writer_after_fresh=writer_state(),
            )
        after = guard.snapshot_dope_weights(net).composite_sha256
        peak = dict(
            allocated=torch.cuda.max_memory_allocated(),
            reserved=torch.cuda.max_memory_reserved(),
        )
    except torch.cuda.OutOfMemoryError as error:
        return finish("ABORT_CUDA_OOM", error=str(error)[:400])
    np.savez_compressed(out / "r3175-gpu-costs.npz", **costs)
    mean = float(np.mean(values))
    per_equal = int(sum(a == b for a, b in zip(values, stored["per_context"])))
    picks_equal = int(sum(a == b for a, b in zip(choices, stored["selected_programs"])))
    diffs = [
        dict(
            index=i,
            cid=c,
            gpu=values[i],
            stored=stored["per_context"][i],
            gpu_pick=choices[i],
            stored_pick=stored["selected_programs"][i],
        )
        for i, c in enumerate(dev)
        if values[i] != stored["per_context"][i]
        or choices[i] != stored["selected_programs"][i]
    ]
    finish(
        "COMPLETE",
        outcome=classify(per_equal, picks_equal, mean, stored["mean"]),
        gpu_mean=mean,
        stored_mean=stored["mean"],
        abs_mean_residual=abs(mean - stored["mean"]),
        per_context_equal=per_equal,
        picks_equal=picks_equal,
        differing_contexts=diffs,
        argmin_index=idx,
        costs_sha256=sha(out / "r3175-gpu-costs.npz"),
        dope_before=before.composite_sha256,
        dope_after=after,
        ckpt_sha256_after=sha(args.ckpt),
        peak_gpu=peak,
        writer_after=writer_state(),
        vram_free_after=gpu_memory()[0],
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu_name=torch.cuda.get_device_name(0),
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        optimizer_updates=0,
        fresh=fresh_record,
    )


if __name__ == "__main__":
    main()
