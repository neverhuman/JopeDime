"""H75-closed-loop runner (PREREG.md / protocol.json in the parent directory are the authority).

Stages:
  plan    -- verify pins and input closure, freeze rosters (acquisition + prospective contexts per fold), 0 fits.
  pilot   -- fit-only timing/identity pilot: fold 0, seed 701, cells A and D, cycle 1 only (no held outcomes).
  shard   -- one (fold, seed): cells A/B/C/D x 2 acquisition cycles, C/D persistent training, replay_B/replay_D,
             then (only after every learner is frozen) the legacy held-family panel, the prospective U8 search
             panel (A/B/C/D/G/R) and the common 801-803 confirmation.
Every native call goes through one ledger (fits, child CPU s, wall s, GPU s, deadline). Outputs are OBJ02 outcome
records (jope.h75-closed-loop.outcome/v1) and GOV03 score records (jope.h75-closed-loop.score/v1); this runner does
not apply the seven-contrast decision.

Interpretations fixed here (stated in the README, not chosen after outcomes):
  * "history" for a cell is its four revealed anchors. A/C (and every cell in cycle 1) use the context's four
    hash-ordered pool anchors (storage positions 0..3). From cycle 2, B/D (and G/R in the search panel) use the four
    best completed programs among {those anchors} U {their own completed fresh evaluations} (utility desc, program
    SHA tie-break). Parents for the proposer are the context's champion(s) in cycle 1 for every policy; from cycle 2
    B/D/G/R use that same best-four history as parents.
  * The proposer's champion-retrieval bank excludes every family in the fold's held (scoring) set.
  * The pre-cycle known best for the apparent-HWM flag is the best passing fitness among the static pool and the
    cell's completed fresh rows (the live promotion semantics).
"""
from __future__ import annotations

import argparse
import calendar
import fcntl
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WAVE_DIR = HERE.parent
REPO = HERE.parents[3]  # HERE = runner/; parents: h75_closed_loop_evolution, flywheel_selfplay, research, repo
INPUT = Path.home() / "jope-data/h50/input"
CACHE = Path.home() / "jope-data/h58/cache"
FOLDS = Path.home() / "jope-data/w1/fold_manifest.json"
R882 = Path.home() / "jope-data/h51/ckpt/round-882.pt"
MICRO = Path.home() / "JopeDimeMicro"
WORKER = MICRO / "bin/bootstrap.s1main"
QUAL_LOCK = Path("/home/ubuntu/jope_plan/qualification.lock")

PINS = {
    "fold_manifest": "dd211134bba127960b1612687478064969dca422e35dfa889790760f316961c1",
    "selected": "981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1",
    "source_manifest": "99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4",
    "r882": "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64",
    "dope": "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814",
    "fused_source": "b9f1468a61aa4dc888c8f2931f9c759f7d4f057793a77ccb05eed6300fb4f9ab",
    "worker": "e3be938644722d14a1c1a0e82168518838c4d174cfe031076a339745c48870a8",
    "micro_rust_evaluator": "53422343699172b61ed69ed707015558f7cf1a344e5449f30e1eb9d708aef900",
    "champions_featurized": "ea020d489ec9a813e4dfb2b6fe45c126ff9f8d8a1b59d0ff9f8242cfe5e6efc0",
    "context_features": "75704e8279c19fee8a7843ccbea801e3cd4b67e0ee08d282b070ddae7ab82faf",
    "pool_priors": "6db567ef2c10d2c67c375f7500305ec693a700c4e71eb23003b4cd312ee51cbe",
}
CELLS = {"A": (False, False), "B": (False, True), "C": (True, False), "D": (True, True)}  # (updates, fresh)
SEARCH_POLICIES = ("A", "B", "C", "D", "G", "R")
CONFIRM_SEEDS = (801, 802, 803)
MARGIN = 1e-6
SCALE = 0.4208


# ----------------------------------------------------------------------------- utilities

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def hkey(*parts) -> str:
    return hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest()


def hseed(*parts) -> int:
    return int(hkey(*parts)[:15], 16)


def utility(status: str, gate: bool, fitness: float) -> float:
    return -float(fitness) if status == "completed" and gate else -2.0


def load_protocol() -> dict:
    p = json.loads((WAVE_DIR / "protocol.json").read_text())
    if p.get("wave_id") != "H75-closed-loop":
        raise SystemExit("protocol wave id mismatch")
    return p


# ----------------------------------------------------------------------------- rosters (pure, testable)

def acquisition_roster(fold: int, fit_families: dict[str, list[str]], kind: str, k: int = 4) -> list[str]:
    """PREREG: sort families by SHA256('H75-closed-loop:<kind>:<fold>:<family_id>'), first k families, smallest
    context SHA in each. kind is 'fit' (acquisition) or 'eval' (prospective held contexts)."""
    fams = sorted(fit_families, key=lambda f: hkey("H75-closed-loop", kind, fold, f))
    if len(fams) < k:
        raise SystemExit(f"fold {fold}: fewer than {k} eligible {kind} families; refusing plan")
    return [min(fit_families[f]) for f in fams[:k]]


def select_proposals(costs: np.ndarray, rng: np.random.Generator) -> tuple[list[int], list[float]]:
    """DL selection: two lowest-cost, then two uniform without replacement from the remainder.
    Returns indices and conditional inclusion probabilities."""
    n = len(costs)
    if n < 4:
        raise ValueError("fewer than four distinct proposals")
    order = list(np.argsort(costs, kind="stable"))
    top, rest = order[:2], order[2:]
    pick = list(rng.choice(rest, 2, replace=False))
    probs = [1.0, 1.0] + [2.0 / len(rest)] * 2
    return [int(i) for i in top + pick], probs


def select_uniform(n: int, rng: np.random.Generator) -> tuple[list[int], list[float]]:
    if n < 4:
        raise ValueError("fewer than four distinct proposals")
    return [int(i) for i in rng.choice(n, 4, replace=False)], [4.0 / n] * 4


def best_history(rows: list[dict], k: int = 4) -> list[dict]:
    """Best k completed rows by utility (desc), program SHA tie-break."""
    done = [r for r in rows if r["status"] == "completed"]
    return sorted(done, key=lambda r: (-r["utility"], r["program_sha256"]))[:k]


def stratified_match(rows_b: list[dict], rows_d: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """Common-learner replay: match counts and pass/fail proportions by deterministic hash subsampling within each
    status stratum to the common minimum."""
    out_b, out_d, report = [], [], {}
    for gate in (True, False):
        sb = sorted([r for r in rows_b if r["gate"] is gate], key=lambda r: hkey("replay", r["record_id"]))
        sd = sorted([r for r in rows_d if r["gate"] is gate], key=lambda r: hkey("replay", r["record_id"]))
        m = min(len(sb), len(sd))
        out_b += sb[:m]; out_d += sd[:m]
        report["pass" if gate else "fail"] = dict(B=len(sb), D=len(sd), kept=m)
    return out_b, out_d, report


# ----------------------------------------------------------------------------- native evaluation + ledger

def s1_metrics(dsl: str, inner_cv: list[dict]) -> dict:
    """Verbatim logic of rust_evaluator.compute_s1_metrics_from_inner_cv."""
    if len(inner_cv) != 2:
        return {"status": "failed", "fitness": None, "logratio": None, "gate": False}
    ratios, wins = [], 0
    for rec in inner_cv:
        m = (rec.get("metrics") or {}).get("rmse")
        b = (rec.get("baseline_metrics") or {}).get("rmse")
        if m is None or b is None or not math.isfinite(m) or not math.isfinite(b) or b <= 1e-12:
            return {"status": "failed", "fitness": None, "logratio": None, "gate": False}
        r = max(float(m) / float(b), 1e-12)
        ratios.append(r)
        wins += r <= 1.0
    lr0, lr1 = math.log(ratios[0]), math.log(ratios[1])
    mean_lr, tail_lr, sd_lr = 0.5 * (lr0 + lr1), max(lr0, lr1), 0.5 * abs(lr0 - lr1)
    gate = wins == 2 and math.exp(mean_lr) <= 0.90
    comp = dsl.count("(")
    fit = (0.8 * mean_lr + 0.1 * tail_lr + 0.1 * sd_lr + 0.0005 * comp) if gate else (
        100.0 + max(mean_lr, 0.0) + max(tail_lr, 0.0) + max(sd_lr, 0.0))
    return {"status": "completed", "fitness": float(fit), "logratio": float(mean_lr), "gate": bool(gate)}


def envelope(raw: bytes, seed: int | None) -> bytes:
    """training_cv_envelope + optional balanced resplit (fold bytes only)."""
    if len(raw) < 17 or raw[:4] != b"JTT1" or raw[16] != 2:
        raise ValueError("invalid training packet")
    nt, nq, width = struct.unpack_from("<III", raw, 4)
    if nq or len(raw) != 17 + nt + nt * (width + 1) * 8:
        raise ValueError("expected training-only rows")
    out = bytearray(raw)
    if seed is not None:
        folds = np.frombuffer(raw, np.uint8, nt, 17).copy()
        out[17:17 + nt] = folds[np.random.default_rng(seed).permutation(nt)].tobytes()
    struct.pack_into("<I", out, 8, 10)
    out.extend(bytes(10 * (width + 1) * 8))
    return bytes(out)


@dataclass
class Ledger:
    """Campaign-wide durable ledger shared by all shards (fcntl lock on every read-modify-write). Resume carries used
    resources and the original deadline. cpu_s is native-evaluator child CPU; gpu_s is synchronized device time."""
    path: Path
    max_fits: int
    max_cpu_s: float
    max_gpu_s: float
    deadline_utc: str

    def _locked(self, fn):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path.with_suffix(".lock"), "a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            state = json.loads(self.path.read_text()) if self.path.exists() else dict(
                fits=0, failed=0, timeouts=0, cpu_s=0.0, wall_s=0.0, gpu_s=0.0, steps=0, deadline_utc=self.deadline_utc)
            if state["deadline_utc"] != self.deadline_utc:
                raise SystemExit("deadline changed on resume")
            out = fn(state)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(state, indent=1))
            os.replace(tmp, self.path)
            return out

    @property
    def state(self) -> dict:
        return self._locked(lambda s: dict(s))

    def check(self, fits_needed: int = 0):
        def chk(s):
            if time.time() >= calendar.timegm(time.strptime(self.deadline_utc, "%Y-%m-%dT%H:%M:%SZ")):
                raise SystemExit("deadline reached")
            if s["fits"] + fits_needed > self.max_fits:
                raise SystemExit(f"native-call cap: {s['fits']} + {fits_needed} > {self.max_fits}")
            if s["cpu_s"] > self.max_cpu_s or s["gpu_s"] > self.max_gpu_s:
                raise SystemExit("CPU or GPU second cap reached")
        self._locked(chk)

    def charge(self, res: dict):
        def ch(s):
            s["fits"] += 1; s["cpu_s"] += res["cpu_s"]; s["wall_s"] += res["wall_s"]
            s["failed"] += res["status"] == "failed"; s["timeouts"] += res["status"] == "timeout"
        self._locked(ch)

    def add(self, **kw):
        def ad(s):
            for k, v in kw.items():
                s[k] = s.get(k, 0) + v
        self._locked(ad)


def native_fit(context: str, dsl: str, seed: int | None, scratch: Path, timeout_s: int = 600) -> dict:
    job = hkey("fit", context, hashlib.sha256(dsl.encode()).hexdigest(), seed)[:24]
    d = scratch / job
    d.mkdir(parents=True, exist_ok=True)
    inp, cand, out = d / "input.jtt1", d / "candidate.json", d / "eval_s1.json"
    inp.write_bytes(envelope((INPUT / "tables" / f"{context}.jtt1").read_bytes(), seed))
    cand.write_text(json.dumps({"dsl": dsl, "program_sha256": hashlib.sha256(dsl.encode()).hexdigest(),
                                "birth_seq": 0, "parents": [], "operator": "h75"}, sort_keys=True))
    env = {"PATH": "/usr/bin:/bin", "RAYON_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "TMPDIR": str(d), "HOME": str(d)}
    t0 = time.monotonic()
    proc = subprocess.Popen(["nice", "-n", "15", str(WORKER), "--fit-s1", str(inp), str(cand), str(out)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    while True:
        pid, st, ru = os.wait4(proc.pid, os.WNOHANG)
        if pid:
            rc, cpu = os.waitstatus_to_exitcode(st), ru.ru_utime + ru.ru_stime
            break
        if time.monotonic() - t0 > timeout_s:
            proc.kill(); _, st, ru = os.wait4(proc.pid, 0)
            rc, cpu = "timeout", ru.ru_utime + ru.ru_stime
            break
        time.sleep(0.05)
    proc.returncode = rc if isinstance(rc, int) else -9
    res = dict(rc=rc, wall_s=round(time.monotonic() - t0, 3), cpu_s=round(cpu, 3),
               input_sha256=hashlib.sha256(inp.read_bytes()).hexdigest(), cut_id="original" if seed is None else f"resplit-{seed}")
    if rc == 0 and out.exists():
        res.update(s1_metrics(dsl, json.loads(out.read_text()).get("inner_cv") or []))
    else:
        res.update(status="timeout" if rc == "timeout" else "failed", fitness=None, logratio=None, gate=False)
    return res


def fit_many(jobs: list[tuple], scratch: Path, ledger: Ledger, workers: int) -> list[dict]:
    """jobs: (context, dsl, seed). Ledger-checked before launch, charged after each call; load-gated at 85%."""
    from concurrent.futures import ThreadPoolExecutor
    ledger.check(len(jobs))

    def run(j):
        while 100.0 * os.getloadavg()[0] / os.cpu_count() > 85.0:
            time.sleep(15)
        return native_fit(j[0], j[1], j[2], scratch)

    with ThreadPoolExecutor(max(1, workers)) as ex:
        results = list(ex.map(run, jobs))
    for r in results:
        ledger.charge(r)
    return results


# ----------------------------------------------------------------------------- data + model (heavy imports lazily)

class World:
    """Admitted inputs: DL-Train pools, H58 cache rows indexed by DL-Train cid only, fold manifest, r882, proposer."""

    def __init__(self, device: str = "cpu", threads: int = 4):
        import torch
        self.torch = torch
        torch.set_num_threads(threads)
        for key, path in (("fold_manifest", FOLDS), ("selected", INPUT / "selected.json"),
                          ("source_manifest", INPUT / "source_manifest.json"), ("r882", R882),
                          ("fused_source", INPUT / "cpu_exact_fused.py"), ("worker", WORKER),
                          ("micro_rust_evaluator", MICRO / "python/rust_evaluator.py"),
                          ("champions_featurized", CACHE / "champions_featurized.json"),
                          ("context_features", CACHE / "context_features.npz"), ("pool_priors", CACHE / "pool_priors.npz")):
            if sha256_file(path) != PINS[key]:
                raise SystemExit(f"pin mismatch: {key}")
        sys.path.insert(0, str(MICRO / "python"))
        self.fm = json.loads(FOLDS.read_text())
        sel = json.loads((INPUT / "selected.json").read_text())
        self.train = {r["context_sha256"]: r for r in sel["train"]}
        if len(self.train) != 1219 or len({r["family_id"] for r in self.train.values()}) != 205:
            raise SystemExit("DL-Train roster mismatch")
        cf = np.load(CACHE / "context_features.npz", allow_pickle=False)
        cids = [c.decode() if isinstance(c, bytes) else str(c) for c in cf["cids"]]
        pos = {c: i for i, c in enumerate(cids)}
        tok_all, mask_all, lm_all = cf["tokens_10x32"], cf["mask_10"], cf["lm48_z"]  # mixed-role arrays: DL-Train rows only indexed
        self.ctx = {}
        for c, r in self.train.items():
            p = np.load(INPUT / "pools" / f"{c}.npz", allow_pickle=False)
            if sha256_file(INPUT / "pools" / f"{c}.npz") != r["pool_sha256"]:
                raise SystemExit(f"pool sha mismatch {c}")
            i = pos[c]
            self.ctx[c] = dict(family=r["family_id"], program=p["program"].astype(np.float32),
                               fitness=p["fitness"].astype(np.float64), gate=p["gate"].astype(bool),
                               sha=[x.decode() for x in p["program_sha256"]],
                               tokens=torch.as_tensor(tok_all[i], dtype=torch.float32),
                               mask=torch.as_tensor(mask_all[i]).bool(), lm=lm_all[i].astype(np.float64))
        from rust_evaluator import featurize_dsl_412, get_w1_sign_matrix, propose_flywheel_candidates
        self.featurize, self.propose = featurize_dsl_412, propose_flywheel_candidates
        self.w1 = get_w1_sign_matrix().astype(np.float64)
        self.bank_all = [r for r in json.loads((CACHE / "champions_featurized.json").read_text())
                         if r["dl_split"] == "DL-Train" and r["context_sha256"] in self.train]
        self.device = torch.device(device)
        self._load_net()

    def _load_net(self):
        import importlib.util
        torch = self.torch
        def imp(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
            return mod
        self.guard = imp("h75_freeze_guard", REPO / "python/src/jope/north_star/dope_freeze_guard.py")
        fused = imp("h75_fused", INPUT / "cpu_exact_fused.py")
        net = fused.FusedJopeNetwork(device=self.device, adapter_variant="h40b")
        for p in net.parameters():
            p.requires_grad_(False)
        self.guard.lock_dope_weights_and_eval_mode(net)
        self.r882_state = torch.load(R882, map_location="cpu", weights_only=True)
        if self.r882_state.get("round") != 882 or self.r882_state.get("dope_sha256") != PINS["dope"]:
            raise SystemExit("r882 metadata mismatch")
        net.load_state_dict(self.r882_state["state_dict"], strict=True)
        net.to(self.device).eval()
        self.net = net
        self.check_dope()

    def check_dope(self):
        if self.guard.snapshot_dope_weights(self.net).composite_sha256 != PINS["dope"]:
            raise SystemExit("DOPE changed: INVALID")

    def reset_r882(self):
        self.net.load_state_dict(self.r882_state["state_dict"], strict=True)
        self.net.eval()

    def trainable(self):
        """Adapter tensors reached by the loss; anchor and unused parent_delta parameters stay frozen."""
        params = []
        for name, p in self.net.adapter_10tok.named_parameters():
            on = not name.startswith("anchor.") and not name.startswith("parent_delta_net.")
            p.requires_grad_(on)
            if on:
                params.append(p)
        return params

    def state_hash(self) -> str:
        h = hashlib.sha256()
        for k, v in sorted(self.net.adapter_10tok.state_dict().items()):
            h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    # rows: dict(program=(412,), proj16=(16,), fitness, gate, sha)
    def proj(self, feat: np.ndarray) -> np.ndarray:
        return (feat.astype(np.float64) @ self.w1 / math.sqrt(412)).astype(np.float32)

    def predict(self, cids: list[str], slates: list[dict], grad: bool = False):
        """Legacy contract (run_fused_closed_loop.py:444-457): ctx/prior/SoS zero, scale .4208, first four slots
        revealed with their observed fitness/gate. slates[i] = dict(program (m,412), fitness (m,), gate (m,))."""
        torch = self.torch
        b, m = len(cids), len(slates[0]["fitness"])
        dev = self.device
        prog = torch.as_tensor(np.stack([s["program"] for s in slates]), dtype=torch.float32, device=dev)
        proj = torch.as_tensor(np.stack([self.proj(s["program"]) for s in slates]), dtype=torch.float32, device=dev)
        hist_f = torch.as_tensor(np.stack([s["fitness"][:4] for s in slates]), dtype=torch.float32, device=dev)
        hist_g = torch.as_tensor(np.stack([s["gate"][:4] for s in slates]).astype(np.float32), device=dev)
        ctxm = torch.enable_grad() if grad else torch.no_grad()
        with ctxm:
            return self.net.adapter_10tok.predict_costs(
                ctx_feat=torch.zeros((b, 584), device=dev), prog_412=prog, proj_16=proj,
                base_prior_cost=torch.zeros((b, m), device=dev), scale=torch.full((b,), SCALE, device=dev),
                sos_prior=torch.zeros((b, m, 2), device=dev),
                mm_tokens_10x32=torch.stack([self.ctx[c]["tokens"] for c in cids]).to(dev),
                mm_mask_10=torch.stack([self.ctx[c]["mask"] for c in cids]).to(dev),
                revealed_indices=torch.arange(4, device=dev).reshape(1, -1).expand(b, -1),
                revealed_raw_f=hist_f, revealed_gate=hist_g)["total_cost"].float()


# ----------------------------------------------------------------------------- slates, loss, training

def anchors(ctx: dict) -> dict:
    return dict(program=ctx["program"][:4], fitness=ctx["fitness"][:4], gate=ctx["gate"][:4], sha=ctx["sha"][:4])


def training_slate(ctx: dict, fresh: list[dict], reserve: int, rng: np.random.Generator, size: int = 192) -> tuple[dict, dict]:
    """Four fixed hash-ordered anchors + 188 unrevealed. D: up to `reserve` hash-ordered completed fresh rows in the
    reserved slots; C passes fresh=[] and fills them uniformly from its static pool. Remaining slots outcome-blind
    uniform from the static pool (anchors excluded). Never pads with duplicated fresh labels."""
    n = len(ctx["fitness"])
    fresh = sorted([r for r in fresh if r["status"] == "completed"], key=lambda r: hkey("slot", r["record_id"]))[:reserve]
    n_static = size - 4 - len(fresh)
    idx = rng.choice(np.arange(4, n), n_static, replace=n - 4 < n_static)
    prog = np.concatenate([ctx["program"][:4], ctx["program"][idx]] + ([np.stack([r["feat412"] for r in fresh])] if fresh else []))
    fit = np.concatenate([ctx["fitness"][:4], ctx["fitness"][idx], np.array([r["fitness"] for r in fresh], dtype=np.float64)])
    gate = np.concatenate([ctx["gate"][:4], ctx["gate"][idx], np.array([r["gate"] for r in fresh], dtype=bool)])
    return dict(program=prog.astype(np.float32), fitness=fit, gate=gate), dict(fresh=len(fresh), static=int(n_static))


def h75_loss(torch, pred, fitness: np.ndarray, gate: np.ndarray):
    """SmoothL1 + 0.1 KL(T=0.2) over UNREVEALED slots only; target = fitness if gate passes else +2."""
    F = torch.nn.functional
    target = torch.as_tensor(np.where(gate, fitness, 2.0), dtype=torch.float32, device=pred.device)
    p, t = pred[:, 4:], target[:, 4:]
    return F.smooth_l1_loss(p, t) + 0.1 * F.kl_div(F.log_softmax(-p / 0.2, dim=-1), F.softmax(-t / 0.2, dim=-1),
                                                  reduction="batchmean")


def train_steps(world: World, opt, fit_cids: list[str], acq: list[str], fresh: dict, reserve: int, seq_seed: int,
                steps: int, ledger: Ledger) -> dict:
    """Persistent C/D/replay training. Each batch: 2 acquisition-roster contexts + 6 uniform other fit contexts; the
    context sequence depends only on seq_seed, so C and D see identical sequences."""
    torch = world.torch
    params = [p for g in opt.param_groups for p in g["params"]]
    before = world.state_hash()
    world.net.train(); world.guard.lock_dope_weights_and_eval_mode(world.net)
    others = [c for c in fit_cids if c not in acq]
    exposure, losses, gnorms = dict(fresh=0, static=0, fresh_winner=0, fresh_negative=0), [], []
    t0 = time.monotonic()
    for step in range(steps):
        rng = np.random.default_rng(hseed("batch", seq_seed, step))
        batch = list(rng.choice(acq, 2, replace=len(acq) < 2)) + list(rng.choice(others, 6, replace=len(others) < 6))
        slates = []
        for j, c in enumerate(batch):
            srng = np.random.default_rng(hseed("slate", seq_seed, step, j))
            s, ex = training_slate(world.ctx[c], fresh.get(c, []), reserve, srng)
            slates.append(s)
            exposure["fresh"] += ex["fresh"]; exposure["static"] += ex["static"]
            for r in sorted([r for r in fresh.get(c, []) if r["status"] == "completed"], key=lambda r: hkey("slot", r["record_id"]))[:reserve]:
                exposure["fresh_winner" if r.get("apparent_hwm") else "fresh_negative"] += 1
        pred = world.predict(batch, slates, grad=True)
        loss = h75_loss(torch, pred, np.stack([s["fitness"] for s in slates]), np.stack([s["gate"] for s in slates]))
        if not torch.isfinite(loss):
            raise SystemExit("nonfinite loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorms.append(float(torch.nn.utils.clip_grad_norm_(params, 1.0)))
        opt.step()
        losses.append(float(loss.detach()))
    gpu = 0.0
    if world.device.type == "cuda":
        torch.cuda.synchronize(); gpu = time.monotonic() - t0
    ledger.add(steps=steps, gpu_s=gpu)
    world.net.eval()
    world.check_dope()
    after = world.state_hash()
    if steps and after == before:
        raise SystemExit("parameters did not change: zero effective updates")
    return dict(steps=steps, state_before=before, state_after=after, losses=losses, grad_norms=gnorms,
                loss_first=losses[0] if losses else None,
                loss_last=losses[-1] if losses else None, grad_norm_mean=float(np.mean(gnorms)) if gnorms else None,
                exposure=exposure)


def new_optimizer(world: World, protocol: dict):
    o = protocol["optimizer"]
    return world.torch.optim.AdamW(world.trainable(), lr=o["lr"], weight_decay=o["weight_decay"],
                                   betas=tuple(o["betas"]), eps=o["eps"])


# ----------------------------------------------------------------------------- search cycles

def propose(world: World, cid: str, parents: list[dict], bank: list[dict], known: set[str], seed: int, n: int = 8) -> list[dict]:
    out, seen, batch = [], set(known), 0
    while len(out) < n and batch < 50:
        for p in world.propose(parents, world.ctx[cid]["lm"], bank, world.ctx[cid]["family"], n_proposals=16,
                               seed=hseed("proposal", seed, batch) % (2 ** 31)):
            if p["program_sha256"] not in seen and len(out) < n:
                seen.add(p["program_sha256"])
                feat, _ = world.featurize(p["dsl"])
                out.append(dict(dsl=p["dsl"], program_sha256=p["program_sha256"], operator=p["operator"],
                                parents=p.get("parents", []), feat412=np.asarray(feat, dtype=np.float32)))
        batch += 1
    return out


def history_slate(hist: list[dict], proposals: list[dict]) -> dict:
    return dict(program=np.stack([h["feat412"] for h in hist] + [p["feat412"] for p in proposals]).astype(np.float32),
                fitness=np.array([h["fitness"] for h in hist] + [0.0] * len(proposals), dtype=np.float64),
                gate=np.array([h["gate"] for h in hist] + [False] * len(proposals), dtype=bool))


# ----------------------------------------------------------------------------- fit cache (resume never re-charges)

class FitCache:
    def __init__(self, path: Path):
        self.path = path
        self.rows = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line); self.rows[r["key"]] = r

    @staticmethod
    def key(context: str, program_sha: str, seed) -> str:
        return hkey("fitkey", context, program_sha, seed)

    def run(self, jobs: list[tuple], scratch: Path, ledger: Ledger, workers: int) -> list[dict]:
        """jobs: (context, dsl, seed). Cached results are reused without a new charge; misses are charged."""
        keyed = [(self.key(c, hashlib.sha256(d.encode()).hexdigest(), s), (c, d, s)) for c, d, s in jobs]
        miss, seen = [], set()
        for k, j in keyed:
            if k not in self.rows and k not in seen:
                miss.append((k, j)); seen.add(k)
        if miss:
            res = fit_many([j for _, j in miss], scratch, ledger, workers)
            with self.path.open("a") as f:
                for (k, j), r in zip(miss, res):
                    r = dict(r, key=k, context=j[0], program_sha256=hashlib.sha256(j[1].encode()).hexdigest(), seed=j[2])
                    self.rows[k] = r; f.write(json.dumps(r, sort_keys=True) + "\n")
        return [self.rows[k] for k, _ in keyed]


def anchor_rows(ctx: dict) -> list[dict]:
    rows = []
    for i in range(4):
        r = dict(feat412=ctx["program"][i], fitness=float(ctx["fitness"][i]), gate=bool(ctx["gate"][i]),
                 status="completed", program_sha256=ctx["sha"][i], record_id=f"anchor:{ctx['sha'][i]}",
                 utility=utility("completed", bool(ctx["gate"][i]), float(ctx["fitness"][i])))
        dsl = ctx.get("anchor_dsl", {}).get(ctx["sha"][i])
        if dsl is not None:
            r["dsl"] = dsl
        rows.append(r)
    return rows


def load_anchor_dsl(path: Path, world_ctx: dict) -> dict:
    """Admitted training-only export {context_sha256: [{program_sha256, dsl}, ...]} for the four hash-ordered anchors.
    Every DSL must hash to its program SHA, and every program SHA must be one of that context's anchors 0..3."""
    data = json.loads(path.read_text())
    n = 0
    for cid, rows in data.items():
        if cid not in world_ctx:
            raise SystemExit(f"anchor DSL export names a non-DL-Train context {cid}")
        allowed = set(world_ctx[cid]["sha"][:4])
        m = {}
        for r in rows:
            if hashlib.sha256(r["dsl"].encode()).hexdigest() != r["program_sha256"] or r["program_sha256"] not in allowed:
                raise SystemExit(f"anchor DSL export row fails identity check for {cid}")
            m[r["program_sha256"]] = r["dsl"]; n += 1
        world_ctx[cid]["anchor_dsl"] = m
    return dict(path=str(path), sha256=sha256_file(path), rows=n)


def static_best_fitness(ctx: dict) -> float:
    g = ctx["gate"]
    return float(np.min(ctx["fitness"][g])) if g.any() else math.inf


# ----------------------------------------------------------------------------- shard orchestration

class Shard:
    def __init__(self, world: World, protocol: dict, fold: int, seed: int, out: Path, ledger: Ledger, workers: int,
                 meta: dict):
        self.w, self.p, self.fold, self.seed, self.out, self.ledger, self.workers = world, protocol, fold, seed, out, ledger, workers
        self.meta = meta
        out.mkdir(parents=True, exist_ok=True)
        self.cache = FitCache(out / "fits.jsonl")
        self.scratch = out / "scratch"
        f = world.fm["folds"][fold]
        held = set(world.fm_held[fold])
        self.held = sorted(held)
        self.fit = sorted(c for c in world.train if c not in held)
        held_fams = {world.train[c]["family_id"] for c in held}
        fams_fit, fams_held = {}, {}
        for c in self.fit:
            fams_fit.setdefault(world.train[c]["family_id"], []).append(c)
        for c in self.held:
            fams_held.setdefault(world.train[c]["family_id"], []).append(c)
        if set(fams_fit) & held_fams:
            raise SystemExit("fit/held family overlap")
        self.acq = acquisition_roster(fold, fams_fit, "fit")
        self.prosp = acquisition_roster(fold, fams_held, "eval")
        self.bank = [r for r in world.bank_all if r["family_id"] not in held_fams]
        self.champ = {}
        for r in world.bank_all:
            self.champ.setdefault(r["context_sha256"], []).append({"dsl": r["dsl"], "program_sha256": r["program_sha256"]})
        if f["held_context_count"] != len(self.held):
            raise SystemExit("held count mismatch vs fold manifest")

    # -- records
    def outcome_record(self, cell, cycle, cid, prop, sel_policy, prob, res, apparent, phase="fit_acquisition"):
        ctx = self.w.ctx[cid]
        return dict(schema="jope.h75-closed-loop.outcome/v1", **self.meta, cell=cell, fold=self.fold, seed=self.seed,
                    cycle=cycle, phase=phase, role="DL-Train", context_sha256=cid, family_id=ctx["family"],
                    table_sha256=self.w.train[cid].get("train_only_jtt1_sha256"),
                    original_rows_sha256=self.w.table_sha[cid], cut_id=res["cut_id"], evaluator_sha256=PINS["worker"],
                    vocabulary_sha256=self.w.vocab_sha, evaluation_protocol="s1main --fit-s1 two-fold training CV; training_cv_envelope (10 structural zero query rows)",
                    evaluation_scope="full_train_cv", nq=0, dope_sha256=PINS["dope"], program_sha256=prop["program_sha256"],
                    dsl=prop["dsl"], parent_program_sha256s=list(prop.get("parents", [])), operator=prop.get("operator"),
                    proposal_seed=prop.get("proposal_seed"), selection_policy=sel_policy, selection_probability=prob,
                    status=res["status"], gate=res["gate"] if res["status"] == "completed" else None,
                    fitness=res["fitness"], logratio=res["logratio"], cpu_s=res["cpu_s"], wall_s=res["wall_s"],
                    input_sha256=res["input_sha256"], apparent_hwm=bool(apparent),
                    source_object=None,  # filled at publication with the S3 key/version/sha of this jsonl
                    record_id=hkey("outcome", self.fold, self.seed, cell, cycle, cid, prop["program_sha256"], res["cut_id"]))

    def write_jsonl(self, name: str, rows: list[dict]):
        with (self.out / name).open("w") as f:
            for r in rows:
                f.write(json.dumps(r, sort_keys=True, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")

    # -- model state per cell
    def load_state(self, state):
        self.w.net.load_state_dict(state, strict=True); self.w.net.eval()

    def snapshot(self):
        return {k: v.detach().clone() for k, v in self.w.net.state_dict().items()}

    # -- one search step for a set of (policy, context) with a given model/history
    def plan_search(self, policy, cid, hist, parents, known, cycle, scorer):
        stream_seed = hseed("stream", self.fold, self.seed, cycle, cid)
        props = propose(self.w, cid, parents, self.bank, known, stream_seed)
        for p in props:
            p["proposal_seed"] = stream_seed
        rng = np.random.default_rng(hseed("select", self.fold, self.seed, cycle, cid))
        if len(props) < 4:
            return props, [], [], dict(shortfall=len(props))
        if policy == "R":
            idx, prob = select_uniform(len(props), rng)
        else:
            idx, prob = select_proposals(scorer(cid, hist, props), rng)
        return props, idx, prob, {}

    def dl_scorer(self, cid, hist, props):
        s = history_slate(hist, props)
        return self.w.predict([cid], [s])[0].cpu().numpy()[4:]

    # -- acquisition (cells A/B/C/D, two cycles) + training + replay
    def acquisition(self):
        done = self.out / "acquisition.done.json"
        if done.exists():
            return json.loads(done.read_text())
        w, proto = self.w, self.p
        r882 = {k: v.clone() for k, v in w.r882_state["state_dict"].items()}
        states = {c: r882 for c in CELLS}
        opts = {}
        fresh = {c: {cid: [] for cid in self.acq} for c in CELLS}  # B/D admitted memory; A/C shadow log
        records, train_receipts = [], []
        for cycle in (1, 2):
            plans = []
            for cell, (upd, fr) in CELLS.items():
                self.load_state(states[cell])
                for cid in self.acq:
                    base = anchor_rows(w.ctx[cid])
                    hist = best_history(base + fresh[cell][cid]) if (fr and cycle > 1) else base
                    parents = ([{"dsl": h["dsl"], "program_sha256": h["program_sha256"]} for h in hist if "dsl" in h] or self.champ.get(cid, [])) if (fr and cycle > 1) else self.champ.get(cid, [])
                    known = set(w.ctx[cid]["sha"]) | {r["program_sha256"] for r in fresh[cell][cid]}
                    props, idx, prob, short = self.plan_search(cell, cid, hist, parents, known, cycle, self.dl_scorer)
                    plans.append((cell, cid, props, idx, prob, short))
            jobs = [(cid, props[i]["dsl"], None) for cell, cid, props, idx, prob, _ in plans for i in idx]
            results = iter(self.cache.run(jobs, self.scratch, self.ledger, self.workers))
            for cell, cid, props, idx, prob, short in plans:
                ctx = w.ctx[cid]
                pre_best = min([static_best_fitness(ctx)] + [r["fitness"] for r in fresh[cell][cid] if r["status"] == "completed" and r["gate"]])
                for i, pr in zip(idx, prob):
                    res = next(results)
                    u = utility(res["status"], res["gate"], res["fitness"] if res["fitness"] is not None else 100.0)
                    apparent = res["status"] == "completed" and res["gate"] and res["fitness"] < pre_best - MARGIN
                    rec = self.outcome_record(cell, cycle, cid, props[i], "dl_top2_plus_uniform2", pr, res, apparent)
                    records.append(rec)
                    if res["status"] == "completed":
                        fresh[cell][cid].append(dict(feat412=props[i]["feat412"], fitness=res["fitness"], gate=res["gate"],
                                                     status="completed", utility=u, program_sha256=props[i]["program_sha256"],
                                                     dsl=props[i]["dsl"], record_id=rec["record_id"], apparent_hwm=bool(apparent)))
                if short:
                    records.append(dict(schema="jope.h75-closed-loop.shortfall/v1", cell=cell, cycle=cycle, context_sha256=cid, **short))
            # persistent training for C and D: identical context sequences (seed independent of cell)
            for cell in ("C", "D"):
                self.load_state(states[cell])
                if cell not in opts:
                    opts[cell] = new_optimizer(w, proto)
                else:
                    o = new_optimizer(w, proto); o.load_state_dict(opts[cell]); opts[cell] = o
                rec = train_steps(w, opts[cell], self.fit, self.acq, fresh[cell] if cell == "D" else {}, 48,
                                  hseed("train", self.fold, self.seed, cycle), proto["updates_per_cycle"], self.ledger)
                train_receipts.append(dict(cell=cell, cycle=cycle, **rec))
                states[cell] = self.snapshot(); opts[cell] = opts[cell].state_dict()
        # common-learner replay from B and D logs
        rb = [r for cid in self.acq for r in fresh["B"][cid]]
        rd = [r for cid in self.acq for r in fresh["D"][cid]]
        mb, md, match = stratified_match(rb, rd)
        replay = {}
        for name, rows in (("replay_B", mb), ("replay_D", md)):
            if not rows:
                replay[name] = dict(status="UNTESTED", reason="no common support"); continue
            w.reset_r882()
            o = new_optimizer(w, proto)
            mem = {}
            for r in rows:
                cid = next(c for c in self.acq if any(x["record_id"] == r["record_id"] for x in fresh["B" if name == "replay_B" else "D"][c]))
                mem.setdefault(cid, []).append(r)
            rec = train_steps(w, o, self.fit, self.acq, mem, 48, hseed("replay", self.fold, self.seed), proto["common_replay_updates"], self.ledger)
            states[name] = self.snapshot(); replay[name] = dict(status="trained", **rec)
        frozen = {}
        for name, st in states.items():
            self.load_state(st)
            frozen[name] = w.state_hash()
            w.torch.save(st, self.out / f"state_{name}.pt") if name in ("C", "D", "replay_B", "replay_D") else None
        if frozen["A"] != frozen["B"]:
            raise SystemExit("A/B must stay bitwise frozen")
        self.write_jsonl("outcomes_acquisition.jsonl", records)
        summary = dict(acq_contexts=self.acq, prospective_contexts=self.prosp, train=train_receipts, replay=replay,
                       replay_match=match, frozen_state_hashes=frozen,
                       fresh_counts={c: {cid: len(v) for cid, v in fresh[c].items()} for c in CELLS},
                       apparent_hwm={c: sum(r.get("apparent_hwm", False) for cid in self.acq for r in fresh[c][cid]) for c in CELLS})
        done.write_text(json.dumps(summary, indent=1, default=str))
        return summary

    def frozen_states(self):
        r882 = self.w.r882_state["state_dict"]
        st = {"A": r882, "B": r882}
        for name in ("C", "D", "replay_B", "replay_D"):
            p = self.out / f"state_{name}.pt"
            if p.exists():
                st[name] = self.w.torch.load(p, map_location="cpu", weights_only=True)
        return st

    # -- terminal reads (only after acquisition.done exists: every learner frozen)
    def legacy_panel(self):
        done = self.out / "legacy.done.json"
        if done.exists():
            return
        if not (self.out / "acquisition.done.json").exists():
            raise SystemExit("terminal read before learners were frozen")
        rows = []
        for name, st in self.frozen_states().items():
            self.load_state(st)
            sh = self.w.state_hash()
            for cid in self.held:
                ctx = self.w.ctx[cid]
                pred = self.w.predict([cid], [dict(program=ctx["program"], fitness=ctx["fitness"], gate=ctx["gate"])])[0].cpu().numpy()
                i = int(np.argmin(pred))
                rows.append(dict(schema="jope.h75-closed-loop.score/v1", **self.meta, panel="legacy", policy_id=name,
                                 fold=self.fold, seed=self.seed, context_sha256=cid, family_id=ctx["family"], role="DL-Train",
                                 selected_program_sha256=ctx["sha"][i], evaluation_identity_sha256=hkey("legacy-eval", PINS["selected"], "full-pool-argmin", "anchors0-3"),
                                 model_state_sha256=sh, status="completed",
                                 utility=utility("completed", bool(ctx["gate"][i]), float(ctx["fitness"][i]))))
        a = {r["context_sha256"]: r["utility"] for r in rows if r["policy_id"] == "A"}
        b = {r["context_sha256"]: r["utility"] for r in rows if r["policy_id"] == "B"}
        if a != b:
            raise SystemExit("A and B disagree on the legacy panel")
        self.write_jsonl("scores_legacy.jsonl", rows)
        done.write_text(json.dumps(dict(rows=len(rows)), indent=1))

    def tree_scorer(self):
        from sklearn.ensemble import HistGradientBoostingRegressor
        X, y = [], []
        for cid in self.fit:
            ctx = self.w.ctx[cid]
            u = np.where(ctx["gate"], -ctx["fitness"], -2.0)
            idx = sorted(range(len(u)), key=lambda i: hkey("G96", cid, ctx["sha"][i]))[:96]
            X.append(ctx["program"][idx]); y.append(u[idx] - u.mean())
        m = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=15, learning_rate=0.05, l2_regularization=1.0,
                                          random_state=self.seed).fit(np.concatenate(X), np.concatenate(y))
        return lambda cid, hist, props: -m.predict(np.stack([p["feat412"] for p in props]))  # cost = -predicted utility

    def prospective_panel(self):
        done = self.out / "prospective.done.json"
        if done.exists():
            return
        if not (self.out / "acquisition.done.json").exists():
            raise SystemExit("terminal read before learners were frozen")
        states = self.frozen_states()
        gscore = self.tree_scorer()
        mem = {p: {cid: [] for cid in self.prosp} for p in SEARCH_POLICIES}
        records, plans_log = [], []
        for cycle in (1, 2):
            plans = []
            for pol in SEARCH_POLICIES:
                if pol in states:
                    self.load_state(states[pol])
                for cid in self.prosp:
                    base = anchor_rows(self.w.ctx[cid])
                    adaptive = pol in ("B", "D", "G", "R")
                    hist = best_history(base + mem[pol][cid]) if (adaptive and cycle > 1) else base
                    parents = ([{"dsl": h["dsl"], "program_sha256": h["program_sha256"]} for h in hist if "dsl" in h] or self.champ.get(cid, [])) if (adaptive and cycle > 1) else self.champ.get(cid, [])
                    known = set(self.w.ctx[cid]["sha"]) | {r["program_sha256"] for r in mem[pol][cid]}
                    scorer = gscore if pol == "G" else self.dl_scorer
                    props, idx, prob, short = self.plan_search(pol, cid, hist, parents, known, cycle, scorer)
                    plans.append((pol, cid, props, idx, prob, short))
            jobs = [(cid, props[i]["dsl"], None) for pol, cid, props, idx, prob, _ in plans for i in idx]
            results = iter(self.cache.run(jobs, self.scratch, self.ledger, self.workers))
            for pol, cid, props, idx, prob, short in plans:
                for i, pr in zip(idx, prob):
                    res = next(results)
                    u = utility(res["status"], res["gate"], res["fitness"] if res["fitness"] is not None else 100.0)
                    rec = self.outcome_record(pol, cycle, cid, props[i], "uniform4" if pol == "R" else ("tree_top2_plus_uniform2" if pol == "G" else "dl_top2_plus_uniform2"), pr, res, False, phase="prospective_search")
                    records.append(rec)
                    if res["status"] == "completed":
                        mem[pol][cid].append(dict(feat412=props[i]["feat412"], fitness=res["fitness"], gate=res["gate"], status="completed",
                                                  utility=u, program_sha256=props[i]["program_sha256"], dsl=props[i]["dsl"], record_id=rec["record_id"], cycle=cycle))
                if short:
                    records.append(dict(schema="jope.h75-closed-loop.shortfall/v1", policy=pol, cycle=cycle, context_sha256=cid, **short))
        # final retained-best accumulator (anchors + all completed new evaluations), U4 after cycle 1, U8 after cycle 2
        selections = []
        for pol in SEARCH_POLICIES:
            for cid in self.prosp:
                base = anchor_rows(self.w.ctx[cid])
                c1 = best_history(base + [r for r in mem[pol][cid] if r["cycle"] == 1], 1)[0]
                c2 = best_history(base + mem[pol][cid], 1)[0]
                selections.append(dict(policy=pol, context=cid, u4_cycle1=c1["utility"], u8=c2["utility"], selected=c2,
                                       anchor_best=best_history(base, 1)[0]))
        # common confirmation: each selected program and the initial best-known anchor on 801-803 (deduplicated)
        cjobs = {}
        for s in selections:
            for prog in (s["selected"], s["anchor_best"]):
                dsl = prog.get("dsl")
                if dsl is None:  # anchor without an admitted DSL export: confirmation unavailable -> INCOMPLETE
                    continue
                for cs in CONFIRM_SEEDS:
                    cjobs[(s["context"], prog["program_sha256"], cs)] = (s["context"], dsl, cs)
        cres = dict(zip(cjobs, self.cache.run(list(cjobs.values()), self.scratch, self.ledger, self.workers)))
        scores = []
        for s in selections:
            sel = s["selected"]
            cu = [cres.get((s["context"], sel["program_sha256"], cs)) for cs in CONFIRM_SEEDS]
            row = dict(schema="jope.h75-closed-loop.score/v1", **self.meta, panel="confirmed_U8", policy_id=s["policy"],
                       fold=self.fold, seed=self.seed, context_sha256=s["context"], family_id=self.w.ctx[s["context"]]["family"],
                       role="DL-Train", selected_program_sha256=sel["program_sha256"],
                       evaluation_identity_sha256=hkey("confirm", *CONFIRM_SEEDS), u8_original_cut=s["u8"], u4_cycle1=s["u4_cycle1"],
                       cut_ids=[f"resplit-{cs}" for cs in CONFIRM_SEEDS])
            if all(c is not None for c in cu):
                row.update(status="completed", utility=float(np.mean([utility(c["status"], c["gate"], c["fitness"] if c["fitness"] is not None else 100.0) for c in cu])))
            else:
                row.update(status="unavailable", utility=None,
                           reason="selected program is a pool anchor without DSL in the admitted pool payload")
            scores.append(row)
        self.write_jsonl("outcomes_prospective.jsonl", records)
        self.write_jsonl("scores_confirmed_U8.jsonl", scores)
        done.write_text(json.dumps(dict(selections=len(selections), confirm_jobs=len(cjobs)), indent=1))


# ----------------------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["plan", "shard"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--fold", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--phases", default="acquisition,legacy,prospective")
    ap.add_argument("--anchor-dsl", type=Path, help="admitted anchor DSL export (required for a COMPLETE confirmation panel)")
    a = ap.parse_args()
    proto = load_protocol()
    lock = None
    if a.device.startswith("cuda"):  # hold the shared GPU lock from before any CUDA allocation until teardown
        lock = open(QUAL_LOCK, "a")
        fcntl.flock(lock, fcntl.LOCK_EX)
    world = World(a.device, a.threads)
    world.fm_held = [f["held_contexts"] for f in world.fm["folds"]]
    world.table_sha = {}
    world.vocab_sha = sha256_file(MICRO / "assets/projections/program-feature-names.json")
    anchor_export = load_anchor_dsl(a.anchor_dsl, world.ctx) if a.anchor_dsl else None
    meta = dict(run_id=a.run_id, source_git_sha=os.environ.get("H75_SOURCE_GIT_SHA"),
                anchor_dsl_export=anchor_export,
                prereg_sha256=sha256_file(WAVE_DIR / "PREREG.md"),
                execution_manifest_sha256=os.environ.get("H75_EXECUTION_MANIFEST_SHA256"))
    if a.stage == "plan":
        rost = []
        for k in range(5):
            world.table_sha = world.table_sha
            sh = Shard.__new__(Shard)
            sh.w, sh.fold = world, k
            held = set(world.fm_held[k]); fit = [c for c in world.train if c not in held]
            fams_fit, fams_held = {}, {}
            for c in fit:
                fams_fit.setdefault(world.train[c]["family_id"], []).append(c)
            for c in held:
                fams_held.setdefault(world.train[c]["family_id"], []).append(c)
            rost.append(dict(fold=k, held=len(held), fit=len(fit), acquisition=acquisition_roster(k, fams_fit, "fit"),
                             prospective=acquisition_roster(k, fams_held, "eval")))
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "plan.json").write_text(json.dumps(dict(meta=meta, pins=PINS, rosters=rost, protocol_sha256=sha256_file(WAVE_DIR / "protocol.json")), indent=1))
        print(json.dumps(rost, indent=1)[:2000])
        return
    for cid in list(world.train):
        world.table_sha[cid] = sha256_file(INPUT / "tables" / f"{cid}.jtt1")
    led = Ledger(a.out / "campaign_ledger.json", max_fits=9600, max_cpu_s=4 * 3600, max_gpu_s=4 * 3600,
                 deadline_utc="2026-10-31T14:00:00Z")
    shard = Shard(world, proto, a.fold, a.seed, a.out / f"fold{a.fold}-seed{a.seed}", led, a.workers, meta)
    phases = a.phases.split(",")
    if "acquisition" in phases:
        print(json.dumps(shard.acquisition(), default=str)[:3000], flush=True)
    if "legacy" in phases:
        shard.legacy_panel()
    if "prospective" in phases:
        shard.prospective_panel()
    world.check_dope()
    print("ledger", json.dumps(led.state))


if __name__ == "__main__":
    main()
