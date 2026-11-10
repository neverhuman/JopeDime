"""H296 (hw_t0240, rd_opus): SWA / Polyak averaging of offline TRAIN-only adapter trajectories from an r3175 COPY.

The harness is the t1845 plateau experiment (~/jope_plan/evidence/fw-plateau-t1845-20261102/{experiment,trials}.py),
reused function by function with three changes: its own copy directory (D), helper modules pinned from the t1845 source
commit e5c5b852 (pinned/: dev_incumbent 81c85296 = the t1845 pin, automatic_promotion, device_pool_cache,
dope_freeze_guard), and the H296 trajectory/averaging arms. CPU only (CUDA hidden), 4 threads, nice 19.

Stages (each refuses to run twice):
  prepare  copy the t1845 input/scorer/tokens and the r3175/r3100 checkpoints into D; verify every sha256.
  regress  (1) r3175 copy replays dev91 identity e79a7715 at exactly 1.1389572403748425 with 91/91 choices;
           (2) the t1845 control seed 184501 (2 updates) reproduces 1.1354387884248622 and its 91 selections.
  run      (after the prereg is on origin/v1/next) per seed 29601-29603: warm restart exactly as t1845's control, then 16
           TRAIN-only updates (each a full pass over the 1219 TRAIN contexts, slate 192, microbatch 32, SmoothL1 + 0.1 KL,
           clip 1, AdamW) with 4 cosine-restart cycles of 4 updates (lr_max = the warm-restart LR, lr_min = 0.1 lr_max;
           within a cycle lr = lr_min + 0.5 (lr_max - lr_min)(1 + cos(pi * pos / 3)), pos = 0..3). No dev gradients, no
           intermediate dev selection, no rollback. Candidates (all frozen and hashed before any dev score):
             CONTROL     the 16th iterate (paired control, no averaging);
             SWA_CYCLE   uniform average of the 4 cycle-end iterates (updates 4, 8, 12, 16; r3175 itself excluded);
             EMA_POLYAK  EMA with decay 0.8 over the 16 iterates (initialised at iterate 1);
           plus SOUP_RET once: uniform average of the r3100 and r3175 adapters (zero training).
           Each candidate touches dev91 exactly once (terminal score, t1845's native canonical scorer).
Averaging acts on the 29,211 trainable adapter parameters only; every frozen tensor and DOPE are checked unchanged."""

import argparse
import copy
import datetime
import hashlib
import importlib.util
import json
import math
import os
import shutil
import struct
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
D = Path(os.environ.get("H296_DATA", "/home/ubuntu/jope-data/H296-opus"))
OUT = D / "run"
MICRO = Path("/home/ubuntu/JopeDimeMicro")
T1845_D = Path("/home/ubuntu/jope-data/fw-plateau-t1845-20261102")
T1845_O = Path("/home/ubuntu/jope_plan/evidence/fw-plateau-t1845-20261102")
RUN = Path("/home/ubuntu/jope-data/flywheel-full-20261030/run-20261030")  # read-only: copies only
DOPE = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
R3175_SHA = "1d6ea900ebed8d2a05c3b3f37576adc15aadd2b53c585bfd5cb71a103e9ff97b"
R3100_SHA = "df0cd494786e48a1d2ebda8b57613081317851f22126950985e19819c27f2854"
IDENTITY = "e79a7715c1531ccb8c4055ab96971101cd4034ec3a71d83bca6d25b0e604ca83"
STORED_MEAN = 1.1389572403748425
T1845_CONTROL_SEED = 184501
T1845_CONTROL_MEAN = 1.1354387884248622
SEEDS = (29601, 29602, 29603)
N_UPDATES, PER_CYCLE, LR_MIN_FRAC, EMA_DECAY = 16, 4, 0.1, 0.8
DEADLINE = 1793769300.0  # 2026-11-04T05:15:00Z (compute; closeout by 05:45Z)
MAX_RSS = 4 * 2**30
MIN_AVAIL = 12 * 2**30
N_TRAINABLE = 29211

THREADS = int(
    os.environ.get("H296_THREADS", "4")
)  # run stage: 1 per forked seed worker (3 workers + idle parent <= 4 cores)
torch.set_num_threads(THREADS)
torch.set_num_interop_threads(1)
sys.path.insert(0, str(MICRO / "python"))
sys.path.insert(0, str(HERE / "pinned"))
import rust_evaluator as evaluator  # noqa: E402
from automatic_promotion import training_indices  # noqa: E402
from dev_incumbent import batch_seed, warm_restart_adapter  # noqa: E402
from device_pool_cache import landmark_float32  # noqa: E402
from hpc.landmark_workspace import FusedLandmarkWorkspace  # noqa: E402

_gs = importlib.util.spec_from_file_location("freeze_guard", HERE / "pinned" / "dope_freeze_guard.py")
guard = importlib.util.module_from_spec(_gs)
sys.modules[_gs.name] = guard
_gs.loader.exec_module(guard)
proc = psutil.Process()
PARENT = [os.getpid()]
_last = [0.0]


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(p, d):
    p = Path(p)
    q = p.with_suffix(p.suffix + ".new")
    q.write_text(json.dumps(d, indent=2, allow_nan=False))
    os.replace(q, p)


def event(kind, **kw):
    r = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "event": kind, **kw}
    with (OUT / "events.jsonl").open("a") as f:
        f.write(json.dumps(r) + "\n")
    print(json.dumps(r), flush=True)


def mem_available():
    return int(next(line.split()[1] for line in open("/proc/meminfo") if line.startswith("MemAvailable"))) * 1024


def res_guard(force=False):
    """t1845's guard plus H296's limits: hard deadline, RSS <= 4 GiB (abort), pause on host CPU > 85% or MemAvailable < 12 GiB."""
    if time.time() > DEADLINE:
        raise RuntimeError("H296 deadline reached")
    if os.getpid() == PARENT[0] and proc.memory_info().rss > MAX_RSS:
        raise RuntimeError("H296 parent RSS limit")
    if not force and time.monotonic() - _last[0] < 1:
        return
    _last[0] = time.monotonic()
    cpu, avail = psutil.cpu_percent(None), mem_available()
    if cpu > 85 or avail < MIN_AVAIL:
        event("pause", cpu_percent=cpu, mem_available=avail)
        while True:
            if time.time() > DEADLINE:
                raise RuntimeError("deadline while paused")
            time.sleep(5)
            cpu, avail = psutil.cpu_percent(None), mem_available()
            if cpu <= 80 and avail >= MIN_AVAIL:
                break
        event("resume", cpu_percent=cpu, mem_available=avail)
    with (OUT / "resource-samples.jsonl").open("a") as f:
        f.write(
            json.dumps(
                {"epoch": time.time(), "cpu_percent": cpu, "mem_available": avail, "rss_bytes": proc.memory_info().rss}
            )
            + "\n"
        )


def state_sha(values):
    h = hashlib.sha256()
    for name in sorted(values):
        t = values[name].detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(t.dtype).encode())
        h.update(np.asarray(t.shape, dtype=np.int64).tobytes())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def prepare():
    """Fresh copy dir: t1845 input (rehashed against selected.json), scorer, tokens, r3175 and r3100 copies."""
    assert not (D / "PREPARED.json").exists(), "prepare already done"
    for sub in ("input/pools", "input/tables", "checkpoints", "scorer", "trial-scorer", "arms", "run", "reference"):
        (D / sub).mkdir(parents=True, exist_ok=True)
    pins = json.loads((T1845_O / "PREREGISTRATION.json").read_text())["source_and_input_pins"]
    copied = []
    for name in ("selected.json", "source_manifest.json", "cpu_exact_fused.py"):
        shutil.copyfile(T1845_D / "input" / name, D / "input" / name)
        assert sha(D / "input" / name) == pins[str(T1845_D / "input" / name)]
        copied.append(dict(path=f"input/{name}", sha256=sha(D / "input" / name)))
    sel = json.loads((D / "input/selected.json").read_text())
    assert len(sel["train"]) == 1219 and len(sel["dev"]) == 91
    for r in sel["train"] + sel["dev"]:
        res_guard()
        assert (
            r["source_split"] in ("train", "inner")
            and r["dl_split"] in ("DL-Train", "DL-Val")
            and r["task"] == "regression"
        )
        for folder, suffix, key in (("pools", ".npz", "pool_sha256"), ("tables", ".jtt1", "train_only_jtt1_sha256")):
            rel = Path(folder) / (r["context_sha256"] + suffix)
            shutil.copyfile(T1845_D / "input" / rel, D / "input" / rel)
            assert sha(D / "input" / rel) == r[key]
    for f in ("scorer", "trial-scorer"):
        for p in (T1845_D / f).iterdir():
            if p.is_file() and p.name != "history.jsonl":
                shutil.copyfile(p, D / f / p.name)
    shutil.copyfile(T1845_D / "tokens.pt", D / "tokens.pt")
    for name, want in (("round-3175.pt", R3175_SHA), ("round-3100.pt", R3100_SHA)):
        shutil.copyfile(RUN / name, D / "checkpoints" / name)
        assert sha(D / "checkpoints" / name) == want
        shutil.copyfile(RUN / name.replace(".pt", ".json"), D / "checkpoints" / name.replace(".pt", ".json"))
    shutil.copyfile(T1845_O / "control_v2-184501.json", D / "reference/t1845-control_v2-184501.json")
    size = sum(p.stat().st_size for p in D.rglob("*") if p.is_file())
    assert size < 3 * 10**9
    write(D / "PREPARED.json", dict(contexts=1310, copied_header_files=copied, bytes=size, tokens_sha256=sha(D / "tokens.pt"),
                                    r3175_sha256=sha(D / "checkpoints/round-3175.pt"), r3100_sha256=sha(D / "checkpoints/round-3100.pt"),
                                    sealed_opened=0, outer_opened=0))  # fmt: skip
    event("prepared", bytes=size)


def build(ckpt_path, expected_sha):
    res_guard(True)
    assert sha(ckpt_path) == expected_sha
    spec = importlib.util.spec_from_file_location("plateau_copy_fused", D / "input/cpu_exact_fused.py")
    fused = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fused)
    net = fused.FusedJopeNetwork(device=torch.device("cpu"), adapter_variant="h40b")
    for p in net.parameters():
        p.requires_grad_(False)
    for n, p in net.adapter_10tok.named_parameters():
        if not n.startswith("anchor."):
            p.requires_grad_(True)
    guard.lock_dope_weights_and_eval_mode(net)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    net.load_state_dict(ckpt["state_dict"], strict=True)
    assert ckpt["dope_sha256"] == DOPE and guard.snapshot_dope_weights(net).composite_sha256 == DOPE
    assert sum(p.numel() for p in net.parameters() if p.requires_grad) == N_TRAINABLE
    return net, ckpt


def load_context(rec):
    cid = rec["context_sha256"]
    p = D / "input/pools" / (cid + ".npz")
    assert sha(p) == rec["pool_sha256"]
    with np.load(p, allow_pickle=False) as z:
        prog = z["program"].astype(np.float32)
        pool = {"rec": rec, "program": prog, "pids": [v.decode() if isinstance(v, bytes) else str(v) for v in z["program_sha256"].tolist()],
                "fitness": z["fitness"].astype(np.float64), "gate": z["gate"].astype(bool)}  # fmt: skip
    pool["proj16"] = prog.astype(np.float64) @ evaluator.get_w1_sign_matrix() / np.sqrt(412)
    return pool


def tokens_for(net, recs):
    cache = D / "tokens.pt"
    prior = torch.load(cache, map_location="cpu", weights_only=True)
    ws, added = FusedLandmarkWorkspace(num_threads=4), 0
    for r in recs:
        res_guard()
        cid = r["context_sha256"]
        if cid in prior:
            continue
        p = D / "input/tables" / (cid + ".jtt1")
        assert sha(p) == r["train_only_jtt1_sha256"]
        raw = p.read_bytes()
        nt, nq, width = struct.unpack_from("<III", raw, 4)
        assert nq == 0 and raw[:4] == b"JTT1" and raw[16] == 2
        a = np.frombuffer(raw, dtype="<f8", offset=17 + nt).reshape(nt, width + 1).copy()
        x, y = a[:, :-1], a[:, -1]
        yd = y.astype(np.float64)
        yn = ((yd - yd.mean()) / max(float(yd.std()), 1e-8)).astype(np.float32)
        v, col, _ = ws.fit_transform_and_extract_bundle_high(landmark_float32(x), np.ascontiguousarray(yn))
        cl = np.zeros((64, 17), np.float32)
        n = min(len(col), 64)
        cl[:n, : min(col.shape[1], 17)] = col[:n, :17]
        with torch.no_grad():
            d, lm = net._run_dope_mlc_v3_and_lm48(
                torch.as_tensor(v).reshape(1, 256), torch.as_tensor(cl).reshape(1, 64, 17), use_landmarks=True
            )
            tok, msk, _ = net._project_stage1b_tokens_batched_gpu(lm, d)
        prior[cid] = {"token": tok[0].detach(), "mask": msk[0].detach()}
        added += 1
    if added:
        raise RuntimeError(f"tokens.pt from t1845 lacked {added} contexts")  # the copy must be complete
    return prior


def predict(net, pools, tokens, cids, indexes):
    prog = np.stack([pools[c]["program"][ix] for c, ix in zip(cids, indexes)])
    proj = np.stack([pools[c]["proj16"][ix] for c, ix in zip(cids, indexes)])
    b, m = prog.shape[:2]
    histf = np.stack([pools[c]["fitness"][ix[: min(4, m)]] for c, ix in zip(cids, indexes)])
    histg = np.stack([pools[c]["gate"][ix[: min(4, m)]] for c, ix in zip(cids, indexes)])
    return net.adapter_10tok.predict_costs(
        ctx_feat=torch.zeros((b, 584)), prog_412=torch.as_tensor(prog), proj_16=torch.as_tensor(proj), base_prior_cost=torch.zeros((b, m)),
        scale=torch.full((b,), 0.4208), sos_prior=torch.zeros((b, m, 2)), mm_tokens_10x32=torch.stack([tokens[c]["token"] for c in cids]),
        mm_mask_10=torch.stack([tokens[c]["mask"] for c in cids]), revealed_indices=torch.arange(min(4, m)).reshape(1, -1).expand(b, -1),
        revealed_raw_f=torch.as_tensor(histf, dtype=torch.float32), revealed_gate=torch.as_tensor(histg, dtype=torch.float32),
    )  # fmt: skip


def evaluate(net, recs, pools, tokens):
    rows = []
    net.eval()
    for r in recs:
        res_guard()
        c = r["context_sha256"]
        p = pools[c]
        with torch.no_grad():
            out = predict(net, pools, tokens, [c], [np.arange(len(p["pids"]))])
        cost = out["total_cost"][0].float().numpy().copy()
        choice = int(np.argmin(cost))
        utility = -float(p["fitness"][choice]) if p["gate"][choice] else -2.0
        rows.append({"cid": c, "family_id": r["family_id"], "choice": choice, "selected_program": p["pids"][choice], "utility": utility,
                     "invalid": not bool(p["gate"][choice]), "minimum_ties": int((cost == cost[choice]).sum())})  # fmt: skip
    return {"mean": sum(x["utility"] for x in rows) / len(rows), "selected_programs": [x["selected_program"] for x in rows],
            "per_context": [x["utility"] for x in rows], "rows": rows}  # fmt: skip


def canonical(net, recs, pools, tokens, reporter):
    """t1845's canonical scorer: native FullFlywheelReporter aggregation of the first-argmin choices."""
    result = evaluate(net, recs, pools, tokens)
    cs = [x["cid"] for x in result["rows"]]
    fit = [float(pools[x["cid"]]["fitness"][x["choice"]]) for x in result["rows"]]
    gate = [bool(pools[x["cid"]]["gate"][x["choice"]]) for x in result["rows"]]
    u = reporter.utility(cs, fit, gate, result["selected_programs"], "dev")
    result["python_sum_mean"] = result["mean"]
    result["mean"] = u["mean"]
    return result


def identity(sel):
    contract = {"protocol": "fixed-dev-cost-argmin-negative-fitness/v1", "selected_packet_sha256": sha(D / "input/selected.json"), "dope_sha256": DOPE,
                "source_sha256": sha(D / "input/cpu_exact_fused.py"), "model_source_sha256": sha(MICRO / "python/model.py"),
                "adapter_source_sha256": sha(MICRO / "python/phase12_multimodal_model.py"),
                "contexts": [{"cid": r["context_sha256"], "pool_sha256": r["pool_sha256"], "training_table_sha256": r["train_only_jtt1_sha256"]} for r in sel["dev"]]}  # fmt: skip
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()


TELEMETRY_SO = Path(
    "/home/ubuntu/.venvs/flywheel-gp-fused/lib/python3.12/site-packages/jope/_telemetry.cpython-312-x86_64-linux-gnu.so"
)
TELEMETRY_SHA = "93acd3087e04b5b26d6ff74a47752d463bd7104b3ad1c32b5642e992676bc76f"  # the t1845 pin


def load_reporter():
    """t1845's scorer: the pinned jope._telemetry extension (sha 93acd308), loaded explicitly under the jope package."""
    import importlib.machinery

    import jope

    assert sha(TELEMETRY_SO) == TELEMETRY_SHA
    loader = importlib.machinery.ExtensionFileLoader("jope._telemetry", str(TELEMETRY_SO))
    spec = importlib.util.spec_from_file_location("jope._telemetry", TELEMETRY_SO, loader=loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    sys.modules["jope._telemetry"] = mod
    jope._telemetry = mod
    return mod.FullFlywheelReporter


def setup():
    FullFlywheelReporter = load_reporter()

    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_available()
    sel = json.loads((D / "input/selected.json").read_text())
    assert identity(sel) == IDENTITY
    net, ckpt = build(D / "checkpoints/round-3175.pt", R3175_SHA)
    trainable = {n: p for n, p in net.named_parameters() if p.requires_grad}
    baseline = {n: t.detach().clone() for n, t in net.state_dict().items()}
    frozen_sha = state_sha({n: t for n, t in net.state_dict().items() if n not in trainable})
    tokens = tokens_for(net, sel["train"] + sel["dev"])
    pools = {}
    for r in sel["train"] + sel["dev"]:
        res_guard()
        pools[r["context_sha256"]] = load_context(r)
    reporter = FullFlywheelReporter(D / "input/selected.json", D / "trial-scorer", D / "trial-scorer/history.jsonl")
    return dict(
        sel=sel,
        net=net,
        ckpt=ckpt,
        trainable=trainable,
        baseline=baseline,
        frozen_sha=frozen_sha,
        tokens=tokens,
        pools=pools,
        reporter=reporter,
    )


def check_frozen(s):
    net = s["net"]
    assert state_sha({n: t for n, t in net.state_dict().items() if n not in s["trainable"]}) == s["frozen_sha"]
    assert guard.snapshot_dope_weights(net).composite_sha256 == DOPE


def batches_for(s, seed, rnd, steps):
    np.random.seed(seed)
    out = []
    for step in range(steps):
        cids = s["reporter"].training_order(batch_seed(seed, rnd, step), 8192, False)
        assert set(cids) == {r["context_sha256"] for r in s["sel"]["train"]}
        ix = np.stack([training_indices(s["pools"][c], 192) for c in cids])
        out.append((cids, ix))
        event(
            "batch_preregistered",
            seed=seed,
            step=step,
            order_sha256=hashlib.sha256(json.dumps(cids).encode()).hexdigest(),
            slate_sha256=hashlib.sha256(ix.tobytes()).hexdigest(),
        )
    return out


def start_trajectory(s, seed, rnd):
    """t1845 control start: incumbent weights and Adam moments, then warm_restart_adapter (perturbation + LR)."""
    net, params = s["net"], list(s["trainable"].values())
    net.load_state_dict(s["baseline"], strict=True)
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    opt.load_state_dict(copy.deepcopy(s["ckpt"]["optimizer"]))
    torch.manual_seed(seed + 999)
    restart = warm_restart_adapter(net, opt, seed, rnd)
    return opt, restart


def update(s, opt, cids, ix):
    net, pools, tokens, params = s["net"], s["pools"], s["tokens"], list(s["trainable"].values())
    net.train()
    opt.zero_grad(set_to_none=True)
    total, n = 0.0, len(cids)
    for off in range(0, n, 32):
        res_guard()
        cc, ii = cids[off : off + 32], ix[off : off + 32]
        pred = predict(net, pools, tokens, cc, list(ii))["total_cost"].float()
        target = torch.as_tensor(np.stack([pools[c]["fitness"][j] for c, j in zip(cc, ii)]), dtype=torch.float32)
        gate = torch.as_tensor(np.stack([pools[c]["gate"][j] for c, j in zip(cc, ii)]))
        target = torch.where(gate, target, torch.full_like(target, 2.0))
        loss = F.smooth_l1_loss(pred, target) + 0.1 * F.kl_div(
            F.log_softmax(-pred / 0.2, dim=-1), F.softmax(-target / 0.2, dim=-1), reduction="batchmean"
        )
        assert torch.isfinite(loss)
        w = len(cc) / n
        (loss * w).backward()
        total += float(loss.detach()) * w
    norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
    assert np.isfinite(norm)
    opt.step()
    check_frozen(s)
    return total, norm


def score(s, name, adapter_state, extra):
    """Load an adapter state (trainables only), save + hash the candidate BEFORE its single dev score, then score."""
    net = s["net"]
    net.load_state_dict(s["baseline"], strict=True)
    with torch.no_grad():
        for n, p in s["trainable"].items():
            p.copy_(adapter_state[n])
    check_frozen(s)
    path = D / "arms" / f"{name}.pt"
    assert not path.exists()
    torch.save({"state_dict": net.state_dict(), "arm": name, "dope_sha256": DOPE}, path)
    ck_sha = sha(path)
    a_sha = state_sha(s["trainable"])
    event("candidate_frozen_before_dev", arm=name, checkpoint_sha256=ck_sha, adapter_sha256=a_sha)
    return dict(name=name, checkpoint=str(path), checkpoint_sha256=ck_sha, adapter_sha256=a_sha, **extra)


def regress():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / "REGRESSION.json").exists()
    s = setup()
    b = canonical(s["net"], s["sel"]["dev"], s["pools"], s["tokens"], s["reporter"])
    stored = json.loads((D / "checkpoints/round-3175.json").read_text())["dev_utility"]
    r1 = dict(
        mean=b["mean"],
        stored=STORED_MEAN,
        exact=b["mean"] == STORED_MEAN == stored["mean"]
        and b["selected_programs"] == stored["selected_programs"]
        and b["per_context"] == stored["per_context"],
    )
    event("regression_r3175", **r1)
    assert r1["exact"], "r3175 copy does not replay dev91 exactly"
    write(OUT / "BASELINE.json", b)
    batches = batches_for(s, T1845_CONTROL_SEED, T1845_CONTROL_SEED, 2)
    opt, restart = start_trajectory(s, T1845_CONTROL_SEED, T1845_CONTROL_SEED)
    for cids, ix in batches:
        update(s, opt, cids, ix)
    c = canonical(s["net"], s["sel"]["dev"], s["pools"], s["tokens"], s["reporter"])
    ref = json.loads((D / "reference/t1845-control_v2-184501.json").read_text())
    r2 = dict(
        mean=c["mean"],
        target=T1845_CONTROL_MEAN,
        exact=c["mean"] == T1845_CONTROL_MEAN == ref["mean"] and c["selected_programs"] == ref["selected_programs"],
        lr=restart["lr"],
        sigma=restart["sigma"],
    )
    event("regression_t1845_control", **r2)
    write(
        OUT / "REGRESSION.json",
        dict(
            r3175=r1,
            t1845_control_184501=r2,
            identity=IDENTITY,
            optimizer_updates=2,
            note="the regression trajectory is discarded; no H296 arm is scored here",
        ),
    )
    assert r2["exact"], "t1845 control seed 184501 did not reproduce"


def run_seed(s, seed, batches):
    """One seed's 16-update trajectory in a forked worker; freezes CONTROL / SWA_CYCLE / EMA_POLYAK (no dev scoring here)."""
    names = list(s["trainable"])
    opt, restart = start_trajectory(s, seed, seed)
    lr_max = float(restart["lr"])
    lr_min = LR_MIN_FRAC * lr_max
    cycle_ends, ema, log = [], None, []
    event("trajectory_start", seed=seed, lr_max=lr_max, sigma=restart["sigma"], pid=os.getpid())
    for t, (cids, ix) in enumerate(batches):
        pos = t % PER_CYCLE
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * pos / (PER_CYCLE - 1)))
        for g in opt.param_groups:
            g["lr"] = lr
        loss, norm = update(s, opt, cids, ix)
        it = {n: s["trainable"][n].detach().clone() for n in names}
        ema = (
            {n: it[n].clone() for n in names}
            if ema is None
            else {n: EMA_DECAY * ema[n] + (1 - EMA_DECAY) * it[n] for n in names}
        )
        if pos == PER_CYCLE - 1:
            cycle_ends.append(it)
        log.append(dict(update=t + 1, lr=lr, loss=loss, grad_norm=norm, adapter_sha256=state_sha(it)))
        event("optimizer_update", seed=seed, update=t + 1, lr=lr, loss=loss, grad_norm=norm)
    assert len(cycle_ends) == 4
    final = {n: s["trainable"][n].detach().clone() for n in names}
    swa = {n: torch.stack([c[n] for c in cycle_ends]).mean(dim=0) for n in names}
    info = dict(seed=seed, lr_max=lr_max, lr_min=lr_min, sigma=restart["sigma"], trajectory=log)
    for name, state, extra in (
        (f"CONTROL-{seed}", final, info),
        (f"SWA_CYCLE-{seed}", swa, dict(seed=seed)),
        (f"EMA_POLYAK-{seed}", ema, dict(seed=seed)),
    ):
        write(OUT / f"{name}.frozen.json", score(s, name, state, extra))


def load_adapter(s, path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return {n: payload["state_dict"][n] for n in s["trainable"]}


def score_frozen(s, name, state):
    net = s["net"]
    net.load_state_dict(s["baseline"], strict=True)
    with torch.no_grad():
        for n, p in s["trainable"].items():
            p.copy_(state[n])
    check_frozen(s)
    r = canonical(net, s["sel"]["dev"], s["pools"], s["tokens"], s["reporter"])
    event("terminal_result", arm=name, mean=r["mean"])
    return dict(
        mean=r["mean"],
        per_context=r["per_context"],
        selected_programs=r["selected_programs"],
        invalid=[x["invalid"] for x in r["rows"]],
    )


def run(launch):
    assert (OUT / "REGRESSION.json").exists() and not (OUT / "RESULTS.json").exists()
    assert THREADS == 1, "run stage uses one torch thread per forked seed worker"
    launch = json.loads(Path(launch).read_text())
    assert launch["prereg_published"] and launch["code_sha256"] == sha(__file__)
    s = setup()
    write(OUT / "RUN_LAUNCH.json", dict(launch, started_utc=time.time(), stage="before any H296 update or score"))
    batches = {seed: batches_for(s, seed, seed, N_UPDATES) for seed in SEEDS}
    pids = {}
    for seed in SEEDS:
        pid = os.fork()
        if pid == 0:
            global proc
            proc = psutil.Process()
            try:
                run_seed(s, seed, batches[seed])
                os._exit(0)
            except BaseException as e:  # noqa: BLE001
                write(OUT / f"FAILED-{seed}.json", dict(error=repr(e)))
                os._exit(1)
        pids[pid] = seed
    status = {}
    while pids:
        pid, st = os.waitpid(-1, 0)
        status[pids.pop(pid)] = os.waitstatus_to_exitcode(st)
        event("worker_exit", seed=list(status)[-1], code=list(status.values())[-1])
    results = []
    for seed in SEEDS:
        if status[seed] != 0:
            continue
        for arm in ("CONTROL", "SWA_CYCLE", "EMA_POLYAK"):
            frozen = json.loads((OUT / f"{arm}-{seed}.frozen.json").read_text())
            assert sha(frozen["checkpoint"]) == frozen["checkpoint_sha256"]
            frozen.update(score_frozen(s, frozen["name"], load_adapter(s, frozen["checkpoint"])))
            write(OUT / f"{frozen['name']}.json", frozen)
            results.append({k: v for k, v in frozen.items() if k != "trajectory"})
    soup_net, _ = build(D / "checkpoints/round-3100.pt", R3100_SHA)
    r3100 = {n: p.detach().clone() for n, p in soup_net.named_parameters() if p.requires_grad}
    del soup_net
    soup = {n: 0.5 * (s["baseline"][n] + r3100[n]) for n in s["trainable"]}
    cand = score(s, "SOUP_RET", soup, dict(members=["round-3100.pt", "round-3175.pt"]))
    cand.update(score_frozen(s, "SOUP_RET", soup))
    write(OUT / "SOUP_RET.json", cand)
    results.append(cand)
    check_frozen(s)
    write(OUT / "RESULTS.json", dict(results=results, worker_status=status, baseline=json.loads((OUT / "BASELINE.json").read_text())["per_context"],
                                     dev_families=[r["family_id"] for r in s["sel"]["dev"]], identity=IDENTITY, dope_sha256=DOPE,
                                     frozen_state_sha256=s["frozen_sha"], live_changes=False))  # fmt: skip
    event("complete", candidates=len(results), worker_status=status)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("prepare", "regress", "run"))
    ap.add_argument("--launch")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"prepare": prepare, "regress": regress, "run": lambda: run(a.launch)}[a.stage]()
