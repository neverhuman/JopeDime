"""H81 — W4 prospective stratum: prior-top4 versus two-plus-two reveal on NEW programs (native fits, CPU only).

H79 (cached pools) and the H75 W4 shard screened existing pool candidates only and state that the prospective
new-program stratum and matched elapsed time remain required. This runner supplies that stratum on the frozen H67
roster (50 fit-only DL-Train contexts, 44 families) x seeds 701-703:

  1. models   W1-faithful context-aware GBM per W1 outer fold (prog412 + lm48_z + dope629; 192 unrevealed slots per
              context sampled by default_rng(700) over selected.json train order; target u - context mean;
              HistGBM(300, lr .08, leaves 63, l2 1, random_state 800+k)). Replication is checked against W1's stored
              out-of-fold predictions `oof_u_prog412_ctx` before any proposal is scored.
  2. run      For each context and seed: 16 NEW proposals (never in the pool) from the pinned live proposer; the
              calibrated prior mu = b_k * u_hat - a_k (W1 per-fold calibration, so mu = -oof cost exactly as in H79);
              arms random4 / gbm_top4 / gbm_2plus2 / gbm_2plus2_swap (H79's rule: RBF GP on proj16, lengthscale =
              median pairwise distance over the context's full pool, noise ratio 1), each revealing exactly four
              distinct new programs by native original-cut training CV. r882 is refused: it needs four revealed
              anchors and this endpoint has zero history.
  3. analyze  Retained-best utility per arm; seeds averaged within context, contexts within family; paired
              family-cluster bootstrap (10,000 draws, seed 20261031, one shared index matrix); common gate on the
              primary gbm_2plus2 - gbm_top4; family Elo (canonical elo_tracker, gbm_top4 = 1500); matched elapsed
              time per arm (its four fits' wall and child CPU, plus selection CPU).
No DL-Val, outer-test, sealed or holdout row is read. DOPE is not involved. Zero GPU.
"""
from __future__ import annotations

import argparse
import calendar
import fcntl
import hashlib
import json
import math
import os
import pickle
import struct
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
INPUT = Path.home() / "jope-data/h50/input"
W1 = Path.home() / "jope-data/w1"
MICRO = Path.home() / "JopeDimeMicro"
WORKER = MICRO / "bin/bootstrap.s1main"
ROSTER = HERE.parent / "h67_w4_active_reveal_2plus2/family_fold_manifest.json"
ELO_SOURCE = REPO / "python/src/jope/north_star/elo_tracker.py"

PINS = {
    "selected": "981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1",
    "source_manifest": "99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4",
    "w1_fold_manifest": "dd211134bba127960b1612687478064969dca422e35dfa889790760f316961c1",
    "w1_priors": "5213faa8d0bcc906e1d3c3226d6104e3846cf11c9da9c222298e9f0dc2fb46d9",
    "w1_context_features": "d9e2275f425cdac1107b1f010c75b69d64dd3f7d29b41f3e1d5fd91bf128fca6",
    "worker": "e3be938644722d14a1c1a0e82168518838c4d174cfe031076a339745c48870a8",
    "micro_rust_evaluator": "53422343699172b61ed69ed707015558f7cf1a344e5449f30e1eb9d708aef900",
    "elo_tracker": "9ece98326c32453179b0c765eb45944b68ca9de944d2828a1d42dee1401d1835",
}
SEEDS = (701, 702, 703)
ARMS = ("random4", "gbm_top4", "gbm_2plus2", "gbm_2plus2_swap")
ANCHOR = "gbm_top4"
PRIMARY = ("gbm_2plus2", "gbm_top4")
SECONDARY = (("gbm_top4", "random4"), ("gbm_2plus2", "gbm_2plus2_swap"), ("gbm_2plus2", "random4"))
N_PROPOSALS = 16
NOISE_RATIO = 1.0
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 10_000, 20261031
TIE, HARM, GATE_LCB = 1e-4, 0.01, 0.001
FIT_CAP = 1800
DEADLINE = "2026-10-31T14:00:00Z"
MARGIN = 1e-6


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


def utility(status: str, gate, fitness) -> float | None:
    """Measured utility; None for an execution error (never mapped to a scientific loss)."""
    if status != "completed":
        return None
    return -float(fitness) if gate else -2.0


# ----------------------------------------------------------------------------- selection rules (pure; H79 rule)

def ranked(score: np.ndarray, ids: list[str], exclude=frozenset()) -> list[int]:
    return sorted((i for i in range(len(ids)) if i not in exclude), key=lambda i: (-float(score[i]), ids[i]))


def lengthscale(x: np.ndarray) -> float:
    sq = np.sum(x * x, axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (x @ x.T), 0.0)
    value = float(np.median(np.sqrt(d2[np.triu_indices(len(x), 1)])))
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("degenerate program embedding")
    return value


def posterior_mean(mu, x, shown, residual, scale):
    def kernel(a, b):
        d2 = np.sum(a * a, axis=1)[:, None] + np.sum(b * b, axis=1)[None, :] - 2.0 * (a @ b.T)
        return np.exp(-np.maximum(d2, 0.0) / (2.0 * scale * scale))
    xs = x[shown]
    alpha = np.linalg.solve(kernel(xs, xs) + NOISE_RATIO * np.eye(len(shown)), np.asarray(residual, dtype=np.float64))
    return mu + kernel(x, xs) @ alpha


def first_stage(arm: str, mu: np.ndarray, ids: list[str], rng_seed: int) -> list[int]:
    if arm == "random4":
        return [int(i) for i in np.random.default_rng(rng_seed).choice(len(ids), 4, replace=False)]
    if arm == "gbm_top4":
        return ranked(mu, ids)[:4]
    return ranked(mu, ids)[:2]  # two-plus-two arms reveal the prior's top two first


def second_stage(arm: str, mu, x, ids, first: list[int], first_u: list[float], scale: float) -> list[int]:
    residual = [u - float(mu[i]) for i, u in zip(first, first_u)]
    if arm.endswith("swap"):
        residual.reverse()
    return ranked(posterior_mean(mu, x, first, residual, scale), ids, set(first))[:2]


# ----------------------------------------------------------------------------- native evaluation, ledger, fit store

def envelope(raw: bytes) -> bytes:
    if len(raw) < 17 or raw[:4] != b"JTT1" or raw[16] != 2:
        raise ValueError("invalid training packet")
    nt, nq, width = struct.unpack_from("<III", raw, 4)
    if nq or len(raw) != 17 + nt + nt * (width + 1) * 8:
        raise ValueError("expected training-only rows")
    out = bytearray(raw)
    struct.pack_into("<I", out, 8, 10)
    out.extend(bytes(10 * (width + 1) * 8))
    return bytes(out)


def s1_metrics(dsl: str, inner_cv: list[dict]) -> dict:
    """Verbatim logic of rust_evaluator.compute_s1_metrics_from_inner_cv."""
    if len(inner_cv) != 2:
        return {"status": "failed", "fitness": None, "logratio": None, "gate": None}
    ratios, wins = [], 0
    for rec in inner_cv:
        m = (rec.get("metrics") or {}).get("rmse")
        b = (rec.get("baseline_metrics") or {}).get("rmse")
        if m is None or b is None or not math.isfinite(m) or not math.isfinite(b) or b <= 1e-12:
            return {"status": "failed", "fitness": None, "logratio": None, "gate": None}
        r = max(float(m) / float(b), 1e-12)
        ratios.append(r)
        wins += r <= 1.0
    lr0, lr1 = math.log(ratios[0]), math.log(ratios[1])
    mean_lr, tail_lr, sd_lr = 0.5 * (lr0 + lr1), max(lr0, lr1), 0.5 * abs(lr0 - lr1)
    gate = wins == 2 and math.exp(mean_lr) <= 0.90
    fit = (0.8 * mean_lr + 0.1 * tail_lr + 0.1 * sd_lr + 0.0005 * dsl.count("(")) if gate else (
        100.0 + max(mean_lr, 0.0) + max(tail_lr, 0.0) + max(sd_lr, 0.0))
    return {"status": "completed", "fitness": float(fit), "logratio": float(mean_lr), "gate": bool(gate)}


def native_fit(context: str, dsl: str, scratch: Path, timeout_s: int = 600) -> dict:
    d = scratch / hkey("h81fit", context, dsl)[:24]
    d.mkdir(parents=True, exist_ok=True)
    inp, cand, out = d / "input.jtt1", d / "candidate.json", d / "eval_s1.json"
    inp.write_bytes(envelope((INPUT / "tables" / f"{context}.jtt1").read_bytes()))
    out.unlink(missing_ok=True)
    cand.write_text(json.dumps({"dsl": dsl, "program_sha256": hashlib.sha256(dsl.encode()).hexdigest(),
                                "birth_seq": 0, "parents": [], "operator": "h81"}, sort_keys=True))
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
               input_sha256=hashlib.sha256(inp.read_bytes()).hexdigest())
    if rc == 0 and out.exists():
        res.update(s1_metrics(dsl, json.loads(out.read_text()).get("inner_cv") or []))
    else:
        res.update(status="timeout" if rc == "timeout" else "failed", fitness=None, logratio=None, gate=None)
    return res


class Ledger:
    """Durable fit ledger: calls are reserved (charged) before launch under a lock; completions settle CPU/wall."""

    def __init__(self, path: Path, cap: int = FIT_CAP, deadline: str = DEADLINE):
        self.path, self.cap, self.deadline = path, cap, deadline

    def _locked(self, fn):
        with open(self.path.with_suffix(".lock"), "a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            s = json.loads(self.path.read_text()) if self.path.exists() else dict(
                fits=0, settled=0, failed=0, timeouts=0, cpu_s=0.0, wall_s=0.0, cap=self.cap, deadline_utc=self.deadline)
            if s["cap"] != self.cap or s["deadline_utc"] != self.deadline:
                raise SystemExit("ledger cap/deadline changed on resume")
            out = fn(s)
            tmp = self.path.with_suffix(f".{os.getpid()}.tmp"); tmp.write_text(json.dumps(s, indent=1)); os.replace(tmp, self.path)
            return out

    def reserve(self, n: int):
        def rs(s):
            if time.time() >= calendar.timegm(time.strptime(self.deadline, "%Y-%m-%dT%H:%M:%SZ")):
                raise SystemExit("deadline reached")
            if s["fits"] + n > self.cap:
                raise SystemExit(f"fit cap: {s['fits']} + {n} > {self.cap}")
            s["fits"] += n
        self._locked(rs)

    def settle(self, r: dict):
        def st(s):
            s["settled"] += 1; s["cpu_s"] += r["cpu_s"]; s["wall_s"] += r["wall_s"]
            s["failed"] += r["status"] == "failed"; s["timeouts"] += r["status"] == "timeout"
        self._locked(st)

    @property
    def state(self):
        return self._locked(lambda s: dict(s))


class FitStore:
    """(context, program) -> native result; each completion appended and fsynced; reuse never re-charges."""

    def __init__(self, path: Path, run_id: str):
        self.path, self.run_id, self.rows = path, run_id, {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    if r["run_id"] != run_id:
                        raise SystemExit("fit store belongs to another run; refusing reuse")
                    self.rows[(r["context"], r["program_sha256"])] = r

    def run(self, jobs: list[tuple], scratch: Path, ledger: Ledger, workers: int) -> dict:
        miss = sorted({(c, d) for c, d in jobs if (c, hashlib.sha256(d.encode()).hexdigest()) not in self.rows})
        if miss:
            ledger.reserve(len(miss))

            def run1(job):
                while 100.0 * os.getloadavg()[0] / os.cpu_count() > 85.0:
                    time.sleep(15)
                return native_fit(job[0], job[1], scratch)

            with ThreadPoolExecutor(max(1, workers)) as ex, self.path.open("a") as f:
                futs = {ex.submit(run1, j): j for j in miss}
                for fut in as_completed(futs):
                    c, d = futs[fut]
                    r = dict(fut.result(), run_id=self.run_id, context=c, program_sha256=hashlib.sha256(d.encode()).hexdigest())
                    f.write(json.dumps(r, sort_keys=True) + "\n"); f.flush(); os.fsync(f.fileno())
                    ledger.settle(r)
                    self.rows[(c, r["program_sha256"])] = r
        return {(c, hashlib.sha256(d.encode()).hexdigest()): self.rows[(c, hashlib.sha256(d.encode()).hexdigest())] for c, d in jobs}


# ----------------------------------------------------------------------------- inputs and models

def verify_pins():
    paths = {"selected": INPUT / "selected.json", "source_manifest": INPUT / "source_manifest.json",
             "w1_fold_manifest": W1 / "fold_manifest.json", "w1_priors": W1 / "cache/w1_train_only_priors.npz",
             "w1_context_features": W1 / "cache/train_only_context_features.npz", "worker": WORKER,
             "micro_rust_evaluator": MICRO / "python/rust_evaluator.py", "elo_tracker": ELO_SOURCE}
    for k, p in paths.items():
        if sha256_file(p) != PINS[k]:
            raise SystemExit(f"pin mismatch: {k}")


def load_inputs():
    sel = json.loads((INPUT / "selected.json").read_text())
    train = sel["train"]
    if len(train) != 1219 or any(r["dl_split"] != "DL-Train" for r in train):
        raise SystemExit("DL-Train roster mismatch")
    fm = json.loads((W1 / "fold_manifest.json").read_text())
    fold_by_cid = {c: k for k, f in enumerate(fm["folds"]) for c in f["held_contexts"]}
    cf = np.load(W1 / "cache/train_only_context_features.npz", allow_pickle=False)
    cids = [c.decode() if isinstance(c, bytes) else str(c) for c in cf["cids"]]
    if cids != [r["context_sha256"] for r in train]:
        raise SystemExit("train-only context features are not in selected.json train order")
    ctxf = np.concatenate([cf["lm48_z"], cf["dope629"]], axis=1).astype(np.float32)
    roster = json.loads(ROSTER.read_text())
    if roster["selected_train_file_sha256"] != PINS["selected"]:
        raise SystemExit("H67 roster is bound to another packet")
    acq = [r["context_sha256"] for r in roster["contexts"]]
    return train, fold_by_cid, ctxf, acq


def pool(cid: str) -> dict:
    p = np.load(INPUT / "pools" / f"{cid}.npz", allow_pickle=False)
    fit, gate = p["fitness"].astype(np.float64), p["gate"].astype(bool)
    return dict(program=p["program"].astype(np.float32), u=np.where(gate, -fit, -2.0), fitness=fit, gate=gate,
                sha=[s.decode() for s in p["program_sha256"]])


def fit_models(out: Path, threads: int) -> dict:
    """W1 recipe, context-aware variant only (the W1/H79 comparator)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from threadpoolctl import threadpool_limits
    train, fold_by_cid, ctxf, _ = load_inputs()
    rng = np.random.default_rng(700)
    xp, cf, y, fk = [], [], [], []
    for i, r in enumerate(train):
        c = r["context_sha256"]; p = pool(c)
        take = min(192, len(p["u"]) - 4)
        samp = 4 + rng.choice(len(p["u"]) - 4, size=take, replace=False)
        xp.append(p["program"][samp]); cf.append(np.repeat(ctxf[i][None], take, 0)); y.append((p["u"][samp] - p["u"].mean()).astype(np.float32))
        fk.append(np.full(take, fold_by_cid[c]))
    X = np.concatenate([np.concatenate(xp), np.concatenate(cf)], axis=1); Y = np.concatenate(y); FK = np.concatenate(fk)
    models = {}
    with threadpool_limits(limits=threads):
        for k in range(5):
            m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_leaf_nodes=63, l2_regularization=1.0,
                                              random_state=800 + k).fit(X[FK != k], Y[FK != k])
            models[k] = m
            print("fold model", k, "rows", int((FK != k).sum()), flush=True)
    with open(out / "models.pkl", "wb") as f:
        pickle.dump(models, f, protocol=pickle.HIGHEST_PROTOCOL)
    return models


def comparator_check(models: dict, tolerance: float = 0.02) -> dict:
    """Comparator strength, fixed before any proposal is scored. W1's fitted models were not retained (its prep reused
    cached shards from an earlier run), so exact replication of `oof_u_prog412_ctx` is impossible. Instead the refit
    must be at least as strong a prior as W1's stored out-of-fold predictions on the frozen H67 roster's admitted
    pools: cached gbm_top4 family-mean retained-best >= W1's minus `tolerance`, plus held-context rank correlation
    with true utility reported for both. Pool labels of these fit-only contexts are already admitted (H79)."""
    train, fold_by_cid, ctxf, acq = load_inputs()
    z = np.load(W1 / "cache/w1_train_only_priors.npz", allow_pickle=False)
    pos = {(c.decode() if isinstance(c, bytes) else str(c)): i for i, c in enumerate(z["cids"])}
    off, oof = z["offsets"], z["oof_u_prog412_ctx"]
    idx = {r["context_sha256"]: i for i, r in enumerate(train)}
    fam = {r["context_sha256"]: r["family_id"] for r in train}
    rank = lambda v: np.argsort(np.argsort(v))
    per = defaultdict(lambda: defaultdict(list))
    rho_mine, rho_w1, corr_w1 = [], [], []
    for c in acq:
        p = pool(c); k = fold_by_cid[c]
        X = np.concatenate([p["program"], np.repeat(ctxf[idx[c]][None], len(p["u"]), 0)], axis=1)
        mine = models[k].predict(X); ref = oof[off[pos[c]]:off[pos[c] + 1]].astype(np.float64)
        un = np.arange(4, len(p["u"]))
        for name, score in (("refit", mine), ("w1_stored", ref)):
            picks = sorted(un, key=lambda i: (-float(score[i]), p["sha"][i]))[:4]
            per[fam[c]][name].append(float(np.max(p["u"][picks])))
        rho_mine.append(float(np.corrcoef(rank(mine[4:]), rank(p["u"][4:]))[0, 1]))
        rho_w1.append(float(np.corrcoef(rank(ref[4:]), rank(p["u"][4:]))[0, 1]))
        corr_w1.append(float(np.corrcoef(mine, ref)[0, 1]))
    fm = {n: mean(mean(v[n]) for v in per.values()) for n in ("refit", "w1_stored")}
    passed = fm["refit"] >= fm["w1_stored"] - tolerance
    return dict(contexts=len(acq), families=len(per), cached_gbm_top4_family_mean=fm, tolerance=tolerance,
                held_rank_corr_with_true_u=dict(refit=float(np.mean(rho_mine)), w1_stored=float(np.mean(rho_w1))),
                corr_refit_vs_w1_stored=dict(mean=float(np.mean(corr_w1)), min=float(np.min(corr_w1))),
                w1_models_retained=False, passed=bool(passed))


def replication_check(models: dict, n_ctx: int = 40) -> dict:
    """Compare refitted predictions to W1's stored out-of-fold u_hat on pool slots of held contexts."""
    train, fold_by_cid, ctxf, _ = load_inputs()
    z = np.load(W1 / "cache/w1_train_only_priors.npz", allow_pickle=False)
    pos = {(c.decode() if isinstance(c, bytes) else str(c)): i for i, c in enumerate(z["cids"])}
    off, oof = z["offsets"], z["oof_u_prog412_ctx"]
    idx = {r["context_sha256"]: i for i, r in enumerate(train)}
    pick = sorted(idx, key=lambda c: hkey("h81-repl", c))[:n_ctx]
    diffs, corrs = [], []
    for c in pick:
        p = pool(c); k = fold_by_cid[c]
        X = np.concatenate([p["program"], np.repeat(ctxf[idx[c]][None], len(p["u"]), 0)], axis=1)
        mine = models[k].predict(X)
        ref = oof[off[pos[c]]:off[pos[c] + 1]].astype(np.float64)
        diffs.append(float(np.max(np.abs(mine - ref)))); corrs.append(float(np.corrcoef(mine, ref)[0, 1]))
    return dict(contexts=n_ctx, max_abs_diff=max(diffs), min_corr=min(corrs), median_abs_diff=float(np.median(diffs)))


# ----------------------------------------------------------------------------- run

def calibration() -> dict:
    cal = json.loads((W1 / "cache/w1_prior_manifest.json").read_text())["calibration_by_fold"]
    return {k: (cal[f"fold_{k}"]["prog412_ctx_intercept_a"], cal[f"fold_{k}"]["prog412_ctx_slope_b"]) for k in range(5)}


def run(out: Path, run_id: str, workers: int, threads: int):
    sys.path.insert(0, str(MICRO / "python"))
    from rust_evaluator import featurize_dsl_412, get_w1_sign_matrix, propose_flywheel_candidates
    train, fold_by_cid, ctxf, acq = load_inputs()
    idx = {r["context_sha256"]: i for i, r in enumerate(train)}
    fam = {r["context_sha256"]: r["family_id"] for r in train}
    with open(out / "models.pkl", "rb") as f:
        models = pickle.load(f)
    cal = calibration()
    W = get_w1_sign_matrix().astype(np.float64)
    champs = json.loads((INPUT / "champions.json").read_text())
    champ = defaultdict(list)
    for r in champs:
        champ[r["context_sha256"]].append({"dsl": r["dsl"], "program_sha256": r["program_sha256"]})
    bank = [r for r in json.loads((Path.home() / "jope-data/h58/cache/champions_featurized.json").read_text())
            if r["dl_split"] == "DL-Train" and r["context_sha256"] in idx]
    cfz = np.load(W1 / "cache/train_only_context_features.npz", allow_pickle=False)
    lm = cfz["lm48_z"].astype(np.float64)
    ledger, store = Ledger(out / "ledger.json"), FitStore(out / "fits.jsonl", run_id)
    scratch = out / "scratch"
    units = []
    for c in acq:
        p = pool(c); k = fold_by_cid[c]; a_k, b_k = cal[k]
        scale = lengthscale((p["program"].astype(np.float64) @ W) / math.sqrt(412))
        known = set(p["sha"])
        for seed in SEEDS:
            props, seen, batch = [], set(known), 0
            while len(props) < N_PROPOSALS and batch < 50:
                for q in propose_flywheel_candidates(champ[c], lm[idx[c]], bank, fam[c], n_proposals=16,
                                                     seed=hseed("h81-stream", c, seed, batch) % (2 ** 31)):
                    if q["program_sha256"] not in seen and len(props) < N_PROPOSALS:
                        seen.add(q["program_sha256"]); props.append(q)
                batch += 1
            if len(props) < 4:
                units.append(dict(context=c, seed=seed, shortfall=len(props))); continue
            t0 = time.process_time()
            feats = np.stack([np.asarray(featurize_dsl_412(q["dsl"])[0], dtype=np.float32) for q in props])
            X = np.concatenate([feats, np.repeat(ctxf[idx[c]][None], len(props), 0)], axis=1)
            mu = b_k * models[k].predict(X) - a_k  # calibrated utility scale, = -oof cost (H79)
            x = (feats.astype(np.float64) @ W) / math.sqrt(412)
            sel_cpu = time.process_time() - t0
            units.append(dict(context=c, seed=seed, props=props, mu=mu, x=x, scale=scale, ids=[q["program_sha256"] for q in props],
                              pool_best_fitness=float(np.min(p["fitness"][p["gate"]])) if p["gate"].any() else math.inf,
                              score_cpu_s=sel_cpu))
    live = [u for u in units if "props" in u]
    for u in live:
        u["first"] = {arm: first_stage(arm, u["mu"], u["ids"], hseed("h81-random", u["context"], u["seed"])) for arm in ARMS}
    res1 = store.run([(u["context"], u["props"][i]["dsl"]) for u in live for arm in ARMS for i in u["first"][arm]],
                     scratch, ledger, workers)
    for u in live:
        u["second"] = {}
        for arm in ("gbm_2plus2", "gbm_2plus2_swap"):
            fu = [utility(**{k2: res1[(u["context"], u["ids"][i])][k2] for k2 in ("status", "gate", "fitness")}) for i in u["first"][arm]]
            if any(v is None for v in fu):
                u["second"][arm] = None  # an execution error in the first pair blocks the update: arm INCOMPLETE
                continue
            t0 = time.process_time()
            u["second"][arm] = second_stage(arm, u["mu"], u["x"], u["ids"], u["first"][arm], fu, u["scale"])
            u.setdefault("update_cpu_s", {})[arm] = time.process_time() - t0
    res2 = store.run([(u["context"], u["props"][i]["dsl"]) for u in live for arm, s in u["second"].items() if s for i in s],
                     scratch, ledger, workers)
    allres = dict(res1, **res2)
    rows, hwm = [], []
    for u in live:
        for arm in ARMS:
            picks = u["first"][arm] + (u["second"].get(arm) or [] if arm.startswith("gbm_2plus2") else [])
            fits = [allres[(u["context"], u["ids"][i])] for i in picks]
            vals = [utility(f["status"], f["gate"], f["fitness"]) for f in fits]
            complete = len(picks) == 4 and all(v is not None for v in vals)
            rows.append(dict(context=u["context"], family=fam[u["context"]], seed=u["seed"], arm=arm,
                             status="completed" if complete else "incomplete",
                             retained_best=max(vals) if complete else None,
                             revealed=[u["ids"][i] for i in picks], revealed_utility=vals,
                             native_wall_s=sum(f["wall_s"] for f in fits), native_cpu_s=sum(f["cpu_s"] for f in fits),
                             selection_cpu_s=u["score_cpu_s"] + (u.get("update_cpu_s", {}).get(arm, 0.0))))
            for i, f in zip(picks, fits):
                if f["status"] == "completed" and f["gate"] and f["fitness"] < u["pool_best_fitness"] - MARGIN:
                    hwm.append(dict(context_sha256=u["context"], family_id=fam[u["context"]], program_sha256=u["ids"][i],
                                    dsl=u["props"][i]["dsl"], fitness_original_cut=f["fitness"],
                                    pool_best_fitness=u["pool_best_fitness"], found_by=arm, seed=u["seed"]))
    with open(out / "rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    uniq = {}
    for h in hwm:
        uniq.setdefault((h["context_sha256"], h["program_sha256"]), h)
    (out / "apparent_hwm.json").write_text(json.dumps(list(uniq.values()), indent=1))
    (out / "shortfalls.json").write_text(json.dumps([u for u in units if "props" not in u], indent=1))
    print("rows", len(rows), "apparent HWM", len(uniq), "ledger", json.dumps(ledger.state))


# ----------------------------------------------------------------------------- analysis

def load_elo():
    import importlib.util
    if sha256_file(ELO_SOURCE) != PINS["elo_tracker"]:
        raise SystemExit("elo_tracker hash mismatch")
    spec = importlib.util.spec_from_file_location("h81_elo", ELO_SOURCE)
    m = importlib.util.module_from_spec(spec); sys.modules["h81_elo"] = m; spec.loader.exec_module(m)
    return m


def contrast(values: np.ndarray, draws: np.ndarray) -> dict:
    boot = values[draws].mean(axis=1)
    lcb = float(np.percentile(boot, 5.0))
    med = float(np.median(values)); harm = float(np.mean(values < -HARM))
    return dict(families=len(values), family_mean=float(values.mean()), family_median=med,
                families_losing_more_than_0_01_fraction=harm,
                family_win_tie_loss=[int(np.sum(values > TIE)), int(np.sum(np.abs(values) <= TIE)), int(np.sum(values < -TIE))],
                bootstrap_p2_5=float(np.percentile(boot, 2.5)), bootstrap_p5_one_sided_lcb=lcb,
                bootstrap_p97_5=float(np.percentile(boot, 97.5)),
                common_gate_passed=bool(lcb > GATE_LCB and med >= 0.0 and harm <= 0.10))


def analyze(rows: list[dict], elo=None) -> dict:
    by = defaultdict(dict)
    incomplete = [r for r in rows if r["status"] != "completed"]
    ctx_seed = defaultdict(lambda: defaultdict(list))
    fam = {}
    for r in rows:
        fam[r["context"]] = r["family"]
        if r["status"] == "completed":
            ctx_seed[r["context"]][r["arm"]].append(r["retained_best"])
    # a context enters only if every arm has all three seeds completed (complete paired panel)
    ctx_means = {c: {a: mean(v[a]) for a in ARMS} for c, v in ctx_seed.items() if all(len(v.get(a, [])) == len(SEEDS) for a in ARMS)}
    families = sorted({fam[c] for c in ctx_means})
    fam_vals = {f: {a: mean(ctx_means[c][a] for c in ctx_means if fam[c] == f) for a in ARMS} for f in families}
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(families), size=(BOOTSTRAP_DRAWS, len(families)))
    d = lambda a, b: np.array([fam_vals[f][a] - fam_vals[f][b] for f in families])
    out = dict(contexts=len(ctx_means), families=len(families), incomplete_rows=len(incomplete),
               family_mean_retained_best={a: mean(fam_vals[f][a] for f in families) for a in ARMS},
               context_mean_retained_best={a: mean(v[a] for v in ctx_means.values()) for a in ARMS},
               primary=dict(contrast=f"{PRIMARY[0]} - {PRIMARY[1]}", **contrast(d(*PRIMARY), draws)),
               secondary={f"{a} - {b}": contrast(d(a, b), draws) for a, b in SECONDARY},
               elapsed_per_arm={a: dict(native_wall_s=sum(r["native_wall_s"] for r in rows if r["arm"] == a),
                                        native_cpu_s=sum(r["native_cpu_s"] for r in rows if r["arm"] == a),
                                        selection_cpu_s=sum(r["selection_cpu_s"] for r in rows if r["arm"] == a),
                                        fits=sum(len(r["revealed"]) for r in rows if r["arm"] == a)) for a in ARMS})
    if elo is not None:
        recs = [elo.DatasetEvalRecord(c, fam[c], "regression", v) for c, v in sorted(ctx_means.items())]
        out["family_elo"] = elo.compute_family_grouped_bradley_terry_elo(recs, anchor_contender=ANCHOR, anchor_elo=1500.0,
                                                                         tie_margin=TIE, prior=0.01, n_iters=400)
    out["verdict"] = ("INCOMPLETE" if incomplete or len(ctx_means) < 50 else
                      ("ADVANCE" if out["primary"]["common_gate_passed"] else "NO_ADVANCE (primary gate not met)"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["models", "check", "run", "analyze"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    verify_pins()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.stage in ("models", "check"):
        if a.stage == "models":
            models = fit_models(a.out, a.threads)
        else:  # re-evaluate the guard on already fitted models without refitting
            with open(a.out / "models.pkl", "rb") as f:
                models = pickle.load(f)
        rep = dict(replication_vs_w1_stored=replication_check(models), comparator=comparator_check(models),
                   models_pkl_sha256=sha256_file(a.out / "models.pkl"))
        (a.out / "comparator_check.json").write_text(json.dumps(rep, indent=1))
        print("comparator check", json.dumps(rep, indent=1))
        if not rep["comparator"]["passed"]:
            raise SystemExit("refit comparator is weaker than W1's stored prior; refusing to score proposals")
    elif a.stage == "run":
        chk = a.out / "comparator_check.json"
        if not chk.exists() or not json.loads(chk.read_text())["comparator"]["passed"]:
            raise SystemExit("comparator check missing or failed; run the models stage first")
        if json.loads(chk.read_text())["models_pkl_sha256"] != sha256_file(a.out / "models.pkl"):
            raise SystemExit("models.pkl changed after the comparator check")
        run(a.out, a.run_id, a.workers, a.threads)
    else:
        rows = [json.loads(l) for l in (a.out / "rows.jsonl").read_text().splitlines() if l.strip()]
        res = analyze(rows, load_elo())
        (a.out / "metrics.json").write_text(json.dumps(dict(res, run_id=a.run_id, ledger=Ledger(a.out / "ledger.json").state), indent=1))
        print(json.dumps({k: res[k] for k in ("contexts", "families", "family_mean_retained_best", "primary", "verdict")}, indent=1))


if __name__ == "__main__":
    main()
