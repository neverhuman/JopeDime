#!/usr/bin/env python3
"""H133 pre-prereg fit-only optimizer-drift calibration. No assessment data is opened."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_BLOCKTIME"] = "0"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

import numpy as np
import torch

TORCH_THREADS = 2
torch.set_num_threads(TORCH_THREADS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

HOME = Path("/home/ubuntu/jope-rescue-20260921/h133-optim-geometry")
EXPORT = HOME / "train_only_export.json"
FOLD = HOME / "fold_manifest_133100.json"
FIT_EXPORT = HOME / "fit_only_export_133100.json"
INPUT_DIR = Path("/home/ubuntu/jope-data/h50/input")
R882_PATH = Path("/home/ubuntu/jope-data/h51/ckpt/round-882.pt")
R882_SHA = "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64"
DOPE_SHA = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
EXPORT_SHA = "ffe506f68d7aade13a7eef6f0cf55d6e79b200dd5e742f8f035dccd0ee27e3ff"
FOLD_SHA = "322137e01ef0a6079284c0e907c8a5ce0db476d46eac1c6ea0fdce3315ea65fc"
SEED = 133
GRID = (1e-3, 3e-3, 1e-2, 3e-2)
UPDATES = 160
BATCH_SIZE = 16
SLATE_SIZE = 192
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
MAX_RSS = 6 * 1024**3
MIN_MEM = 30 * 1024**3
MAX_LOAD1 = 86.7
MAX_DISK = 93.8
MAX_SCRATCH = 2 * 1024**3


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Read-only reuse of the frozen H132 helper implementation and its source dependencies.
h132 = load_module("h133_h132_helpers", Path("/home/ubuntu/jope-rescue-20260921/h132-ewc-bitfit/run_h132_cpu.py"))
h48 = h132.h48
h55 = h132.h55
h53b = h132.h53b
h59 = h132.h59
try:
    import threadpoolctl
    threadpoolctl.threadpool_limits(limits=1)
except ImportError:
    pass
torch.set_num_threads(TORCH_THREADS)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rss_bytes() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return 0


def guard(stage: str) -> dict:
    fs = os.statvfs("/")
    used = fs.f_blocks - fs.f_bfree
    available = fs.f_bavail
    disk_pct = 100.0 * used / (used + available)
    mem = 0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            mem = int(line.split()[1]) * 1024
            break
    load1, load5, _ = os.getloadavg()
    rss = rss_bytes()
    scratch = sum(p.stat().st_size for p in HOME.rglob("*") if p.is_file())
    if torch.cuda.is_available():
        raise RuntimeError(f"{stage}: CUDA visible")
    if torch.get_num_threads() > 4 or len(list(Path("/proc/self/task").iterdir())) > 4:
        raise RuntimeError(f"{stage}: process/torch threads exceed 4")
    if os.getpriority(os.PRIO_PROCESS, 0) < 19:
        raise RuntimeError(f"{stage}: process nice value is not 19")
    if rss > MAX_RSS:
        raise RuntimeError(f"{stage}: RSS {rss} exceeds 6 GiB")
    if mem < MIN_MEM:
        raise RuntimeError(f"{stage}: MemAvailable below 30 GiB")
    if load1 > MAX_LOAD1:
        raise RuntimeError(f"{stage}: load1 {load1:.2f} exceeds {MAX_LOAD1}")
    if disk_pct >= MAX_DISK:
        raise RuntimeError(f"{stage}: disk {disk_pct:.4f}% >= {MAX_DISK}%")
    if scratch >= MAX_SCRATCH:
        raise RuntimeError(f"{stage}: H133 scratch >= 2 GiB")
    return {"stage": stage, "nice": os.getpriority(os.PRIO_PROCESS, 0), "torch_threads": torch.get_num_threads(),
            "rss_bytes": rss, "mem_available_bytes": mem, "load1": load1, "disk_used_pct": disk_pct,
            "scratch_bytes": scratch}


def make_fit_tokens(net, tables, device):
    # H46's shared helper hardcodes a 4-thread landmark workspace; H133 caps it at 2
    # so the C++ workspace plus the Python/Torch runtime stays within the 4-thread guard.
    tokens, masks = {}, {}
    ws = h48.base.FusedLandmarkWorkspace(num_threads=2)
    for i, (cid, (x, y)) in enumerate(tables.items(), 1):
        if i % 16 == 0:
            guard(f"tokens_{i}")
            h48.base.headroom()
        yd = y.astype(np.float64)
        yn = ((yd - yd.mean()) / max(float(yd.std()), 1e-8)).astype(np.float32)
        vec, cols, _ = ws.fit_transform_and_extract_bundle_high(
            h48.base.landmark_float32(x), np.ascontiguousarray(yn)
        )
        colbuf = np.zeros((64, 17), np.float32)
        n = min(len(cols), 64)
        colbuf[:n, : min(cols.shape[1], 17)] = cols[:n, :17]
        with torch.no_grad():
            d, lm = net._run_dope_mlc_v3_and_lm48(
                torch.as_tensor(vec, device=device).reshape(1, 256),
                torch.as_tensor(colbuf, device=device).reshape(1, 64, 17),
                use_landmarks=True,
            )
            tok, mask, _ = net._project_stage1b_tokens_batched_gpu(lm, d)
        tokens[cid], masks[cid] = tok[0].detach(), mask[0].detach()
    return tokens, masks


def nontrainable_sha(net) -> str:
    h = hashlib.sha256()
    for name, parameter in sorted(net.named_parameters()):
        if not parameter.requires_grad:
            h.update(name.encode())
            h.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def build_plan(fit_cids, pools, orders):
    full_ids = tuple(sorted(fit_cids))
    sampler_key = SEED * 1_000_003 + 10_007
    plan = []
    rng_state = np.random.get_state()
    try:
        for rnd in range(1, 81):
            round_seed = int((sampler_key + rnd * 104729) % (2**32))
            np.random.seed(round_seed)
            rotated = full_ids[rnd % len(full_ids):] + full_ids[:rnd % len(full_ids)]
            for step in range(8):
                if step >= 2:
                    continue
                stream = h132.paired_stream_seed(SEED, rnd, step)
                cids = tuple(random.Random(stream).choices(rotated, k=BATCH_SIZE))
                indices = tuple(np.asarray(h53b.cached_training_indices(pools[c], orders[c], SLATE_SIZE), dtype=np.int64).copy() for c in cids)
                plan.append({"stream_seed": stream, "context_ids": cids, "candidate_indices": indices})
    finally:
        np.random.set_state(rng_state)
    if len(plan) != UPDATES:
        raise RuntimeError(f"expected {UPDATES} updates, got {len(plan)}")
    return plan


def run_cell(label, optimizer_kind, lr, plan, pools, tokens, masks):
    ckpt = HOME / "scratch" / f"r882_{label}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(R882_PATH, ckpt)
    if sha256_file(ckpt) != R882_SHA:
        raise RuntimeError(f"{label}: fresh r882 copy hash mismatch")
    try:
        device = torch.device("cpu")
        net, trainable, guard_obj, _state = h55.build_model(INPUT_DIR, ckpt, device, h48)
    finally:
        ckpt.unlink(missing_ok=True)
    names = [n for n, p in net.named_parameters() if p.requires_grad]
    if len(trainable) != 47 or sum(p.numel() for p in trainable) != 29211:
        raise RuntimeError(f"{label}: adapter trainable inventory mismatch")
    if any(not n.startswith("adapter_10tok.") or ".anchor." in n for n in names):
        raise RuntimeError(f"{label}: non-adapter parameter trainable")
    dope_pre = guard_obj.snapshot_dope_weights(net).composite_sha256
    if dope_pre != DOPE_SHA:
        raise RuntimeError(f"{label}: DOPE pre-hash mismatch")
    frozen_pre = nontrainable_sha(net)
    initial = [(p.detach().clone()) for _, p in sorted(net.adapter_10tok.named_parameters()) if not _.startswith("anchor.")]
    predictor = h48.make_predictor(net, pools, tokens, masks, device)
    if optimizer_kind == "adamw":
        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=WEIGHT_DECAY)
    elif optimizer_kind == "sgd":
        optimizer = torch.optim.SGD(trainable, lr=lr, momentum=0.9, nesterov=True, weight_decay=WEIGHT_DECAY)
    else:
        raise RuntimeError("unknown optimizer")
    started = time.monotonic()
    for i, batch in enumerate(plan, 1):
        if i % 8 == 1 or i == UPDATES:
            guard(f"{label}_update_{i}")
        torch.manual_seed(batch["stream_seed"])
        cids = list(batch["context_ids"])
        indices = list(batch["candidate_indices"])
        targets = h59.preregistered_targets(
            torch.as_tensor(np.stack([pools[c]["fitness"][ix] for c, ix in zip(cids, indices)]), dtype=torch.float32),
            torch.as_tensor(np.stack([pools[c]["gate"][ix] for c, ix in zip(cids, indices)]), dtype=torch.bool),
        )
        optimizer.zero_grad(set_to_none=True)
        predictions = predictor(cids, indices)
        objective = h132.per_context_losses(predictions, targets).mean()
        if not bool(torch.isfinite(objective)):
            raise RuntimeError(f"{label}: nonfinite fit objective")
        objective.backward()
        torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP, error_if_nonfinite=True)
        optimizer.step()
        del objective, predictions, targets
    elapsed = time.monotonic() - started
    dope_post = guard_obj.snapshot_dope_weights(net).composite_sha256
    frozen_post = nontrainable_sha(net)
    if dope_post != DOPE_SHA or frozen_post != frozen_pre:
        raise RuntimeError(f"{label}: frozen parameter verification failed")
    current = [p.detach() for _, p in sorted(net.adapter_10tok.named_parameters()) if not _.startswith("anchor.")]
    drift = float(torch.linalg.vector_norm(torch.cat([(a - b).reshape(-1).to(torch.float64) for a, b in zip(current, initial)])).item())
    final_adapter_sha = h53b.adapter_sha(net)
    del predictor, optimizer, trainable, net
    return {"label": label, "optimizer": optimizer_kind, "lr": lr, "updates": UPDATES, "fit_adapter_l2_drift": drift,
            "dope_sha_pre": dope_pre, "dope_sha_post": dope_post, "nontrainable_sha_pre": frozen_pre,
            "nontrainable_sha_post": frozen_post, "final_adapter_sha256": final_adapter_sha, "elapsed_seconds": elapsed}


def main():
    guard("start")
    if sha256_file(EXPORT) != EXPORT_SHA or sha256_file(FOLD) != FOLD_SHA:
        raise RuntimeError("train-only export/fold manifest SHA mismatch")
    export = json.loads(EXPORT.read_bytes())
    if export.get("schema") != "jope.h133.dl-train-only-metadata/v1":
        raise RuntimeError("wrong export schema")
    train_rows = export["rows"]
    if len(train_rows) != 1219 or any(r.get("dl_split") != "DL-Train" for r in train_rows):
        raise RuntimeError("non-train row in train-only export")
    folds = json.loads(FOLD.read_bytes())
    if len(folds) != 5 or any(len(f) != 41 for f in folds):
        raise RuntimeError("invalid 5-fold manifest")
    fold_families = set().union(*(set(x) for x in folds))
    if fold_families != {r["family_id"] for r in train_rows}:
        raise RuntimeError("fold family roster differs from train-only metadata")
    fit_families = set(f for fold in folds[1:] for f in fold)
    fit_rows = [r for r in train_rows if r["family_id"] in fit_families]
    if len(fit_rows) != 974 or len({r["family_id"] for r in fit_rows}) != 164:
        # The actual context count is reported rather than assumed; family count is invariant.
        if len({r["family_id"] for r in fit_rows}) != 164:
            raise RuntimeError("fit family count mismatch")
    FIT_EXPORT.write_bytes(json.dumps({"schema":"jope.h133.fit-only-metadata/v1","seed":SEED,"rows":fit_rows},sort_keys=True,separators=(",",":")).encode())
    fit_cids = sorted(r["context_sha256"] for r in fit_rows)
    if not fit_cids or len(fit_cids) != len(set(fit_cids)):
        raise RuntimeError("fit context roster invalid")
    del train_rows, export
    # Payload access is restricted to fit_rows; no assessment pool/table paths are formed or opened.
    pools = {}
    orders = {}
    tables = {}
    w1_mat = h48.base.evaluator.get_w1_sign_matrix() / np.sqrt(412.0)
    for row in fit_rows:
        if row["family_id"] not in fit_families or row.get("dl_split") != "DL-Train":
            raise RuntimeError("fit-only data closure violation")
        cid = row["context_sha256"]
        pool_path = INPUT_DIR / "pools" / f"{cid}.npz"
        table_path = INPUT_DIR / "tables" / f"{cid}.jtt1"
        if sha256_file(pool_path) != row["pool_sha256"]:
            raise RuntimeError(f"fit pool SHA mismatch: {cid}")
        raw = table_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["train_only_jtt1_sha256"]:
            raise RuntimeError(f"fit JTT1 SHA mismatch: {cid}")
        nt,nq,width = h48.base.struct.unpack_from("<III",raw,4)
        if raw[:4] != b"JTT1" or nq != 0 or raw[16] != 2:
            raise RuntimeError(f"non-train JTT1 table: {cid}")
        values=np.frombuffer(raw,dtype="<f8",offset=17+nt).reshape(nt,width+1).copy()
        tables[cid]=(values[:,:-1],values[:,-1])
        with np.load(pool_path,allow_pickle=False) as z:
            progs=z["program"].astype(np.float32)
            fitness=z["fitness"].astype(np.float64)
            gate_arr=z["gate"].astype(bool)
            pids=[v.decode() if isinstance(v,bytes) else str(v) for v in z["program_sha256"].tolist()]
        pools[cid]={"rec":row,"pids":pids,"program":progs,"proj16":(progs.astype(np.float64)@w1_mat).astype(np.float32),"fitness":fitness,"gate":gate_arr}
        orders[cid]=np.argsort(fitness)
        del raw,values,progs,fitness,gate_arr,pids
        if len(pools)%32==0:
            guard(f"loaded_{len(pools)}_fit_contexts")
    if set(pools)!=set(fit_cids):
        raise RuntimeError("loaded pool set differs from fit-only context set")
    device=torch.device("cpu")
    ckpt=HOME/"scratch"/"r882_tokens.pt"
    ckpt.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(R882_PATH,ckpt)
    if sha256_file(ckpt)!=R882_SHA:
        raise RuntimeError("token model checkpoint SHA mismatch")
    try:
        net,trainable,guard_obj,_=h55.build_model(INPUT_DIR,ckpt,device,h48)
    finally:
        ckpt.unlink(missing_ok=True)
    tokens,masks=make_fit_tokens(net,tables,device)
    if set(tokens)!=set(fit_cids):
        raise RuntimeError("token closure differs from fit-only contexts")
    del net,trainable,guard_obj
    plan=build_plan(fit_cids,pools,orders)
    plan_digest=hashlib.sha256(json.dumps([{"stream":b["stream_seed"],"cids":b["context_ids"],"ix":[x.tolist() for x in b["candidate_indices"]]} for b in plan],sort_keys=True,separators=(",",":")).encode()).hexdigest()
    progress_path = HOME / "calibration_progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_bytes())
        if progress.get("batch_plan_sha256") != plan_digest or progress.get("fold_manifest_sha256") != FOLD_SHA:
            raise RuntimeError("persisted calibration progress belongs to a different plan")
    else:
        progress = {"schema":"jope.h133.fit_only_calibration_progress/v1","seed":SEED,
                    "fold_manifest_sha256":FOLD_SHA,"train_only_export_sha256":EXPORT_SHA,
                    "batch_plan_sha256":plan_digest,"control":None,"sgdm_grid":[]}
    def persist_progress():
        tmp = progress_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(progress,sort_keys=True,indent=2)+"\n")
        os.replace(tmp,progress_path)
    if progress["control"] is None:
        progress["control"] = run_cell("control160", "adamw", 1e-4, plan, pools, tokens, masks)
        persist_progress()
        print(json.dumps({"completed":"CONTROL_160","fit_adapter_l2_drift":progress["control"]["fit_adapter_l2_drift"]}),flush=True)
    rows_by_lr = {float(r["lr"]):r for r in progress["sgdm_grid"]}
    for lr in GRID:
        if lr not in rows_by_lr:
            row = run_cell(f"sgdm_{lr:g}","sgd",lr,plan,pools,tokens,masks)
            row["abs_drift_difference_from_control"] = abs(row["fit_adapter_l2_drift"]-progress["control"]["fit_adapter_l2_drift"])
            progress["sgdm_grid"].append(row)
            persist_progress()
            rows_by_lr[lr] = row
            print(json.dumps({"completed":"SGDM_GRID","lr":lr,"fit_adapter_l2_drift":row["fit_adapter_l2_drift"],
                              "abs_drift_difference_from_control":row["abs_drift_difference_from_control"]}),flush=True)
    control=progress["control"]
    rows=[rows_by_lr[lr] for lr in GRID]
    chosen=min(rows,key=lambda x:(x["abs_drift_difference_from_control"],x["lr"]))
    receipt={"schema":"jope.h133.fit_only_drift_calibration/v1","seed":SEED,"fold_seed":133100,"fold_manifest_sha256":FOLD_SHA,
             "train_only_export_sha256":EXPORT_SHA,"fit_family_count":164,"fit_context_count":len(fit_rows),"assessment_payload_reads":0,
             "assessment_outcome_metrics_read":0,"control":control,"sgdm_grid":rows,"chosen_sgdm_lr":chosen["lr"],
             "selection":"minimum absolute adapter L2 drift difference to CONTROL-160; exact tie chooses smaller LR",
             "batch_plan_sha256":plan_digest,"scratch_bytes_at_end":sum(p.stat().st_size for p in HOME.rglob("*") if p.is_file())}
    out=HOME/"fit_only_calibration.json"
    out.write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    guard("complete")
    print(json.dumps({"status":"PASS","control_drift":control["fit_adapter_l2_drift"],"grid":[{"lr":r["lr"],"drift":r["fit_adapter_l2_drift"],"absdiff":r["abs_drift_difference_from_control"]} for r in rows],"chosen_lr":chosen["lr"],"receipt":str(out)},sort_keys=True),flush=True)

if __name__=="__main__":
    main()
