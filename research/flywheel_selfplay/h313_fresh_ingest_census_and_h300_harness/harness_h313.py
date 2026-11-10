"""H313 part B: the H300 confirmation harness for the FROZEN H303 NESTED ENS_VOTE2 chooser (H310 code at 819a45aa5e).

Phases:
  lock     seeds 30004-30006 fitted with H310's lock code on DL-Train only (dev91 guard, TRAIN-only loader); seeds
           30001-30003 reused from H310's archived locks (S3 H310 tar, exact VersionId, sha-verified against H310's
           lock_receipt.json). Writes lock_manifest.json (6 fit_lock sha256s), fsynced.
  dryrun   the full score + decide pipeline on DL-Train P0 outer fold 0 (46 families / 335 contexts), out-of-fold for
           H303's archived fold-0 locks of seeds 30001-30003; reference = the pool's slot 0 (r882); harness regression:
           seed 30001 picks sha == 18fdb2538cea...
Library (for H314): FreshGate (fresh-path opens raise until all 6 locks verify), score_population, summarize_seeds,
sign_flip_p, decide (FRESH_CONFIRMED rule). No fresh population is scored in H313."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FS = Path(os.environ.get("H313_FS", str(HERE.parent)))  # run copies from a path without H310 guard needles
H310 = FS / "h310_ensvote2_dev91_vs_r3175"
DATA = Path(os.environ.get("H313_DATA", "/home/ubuntu/jope-data/h313-opus"))
HARNESS = DATA / "harness"
SEEDS = (30001, 30002, 30003, 30004, 30005, 30006)
NEW_SEEDS = (30004, 30005, 30006)
BUCKET = "veox-jopedime-learning-use2"
H310_TAR = ("experiments/rd_breakthrough_20261030/H310/h310_locks_models_outputs.tar.gz", "1JWgsGRcV3fD5r4hvAJXptCQ_crMYoQl",
            "f42107b67cf9540f4192c8edc8d2f04a1af374e221d34329af9b30d9177918e1")  # fmt: skip
H303_TAR = ("experiments/rd_breakthrough_20261030/H303/h303_models_locks_and_outputs.tar.gz", "hshnLPUZuvpHewykQNTh..mWAGu9KGHo",
            "8957cf0e0b55c21231fd3d40d56ce5bdf27aa915dc97786bcbc00dbee782857d")  # fmt: skip
FRESH_NEEDLES = (
    "fresh_pools",
    "fresh_cohort",
    "fresh_top8",
    "fresh-top8",
    "fresh-build",
    "fresh_v7",
    "cohort_v7",
    "/v7/",
)
BOOT_SEED, DRAWS, FLIP_SEED, FLIP_DRAWS, TIE, HARM = 302150, 10000, 313150, 200000, 1e-4, 0.01
ELO_LABEL_FRESH = "reference = r3175 slot0 (fresh v7)"
ELO_LABEL_DRY = "reference = r882 slot0 (DL-Train pseudo-fresh dry run)"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def lock_path(base, seed):
    return Path(base) / "lock" / "fold_4" / f"root_{seed}" / "fit_lock.json"


# ---------------------------------------------------------------- fresh gate


class FreshGate:
    """Audit hook: any open whose path contains a fresh needle raises unless the 6-lock manifest verifies."""

    def __init__(self, base):
        self.base = Path(base)
        self.denied = 0

    def open_ok(self):
        m = self.base / "lock_manifest.json"
        if not m.exists():
            return False
        man = json.loads(m.read_text())
        return set(map(int, man["locks"])) == set(SEEDS) and all(
            lock_path(self.base, int(s)).exists() and sha(lock_path(self.base, int(s))) == v
            for s, v in man["locks"].items()
        )

    def install(self):
        def hook(event, args):
            if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
                s = os.fsdecode(args[0]).lower()
                if any(n in s for n in FRESH_NEEDLES) and "lock_manifest" not in s:
                    if not self.open_ok():
                        self.denied += 1
                        raise RuntimeError("H313 fresh gate: fresh path refused before all 6 locks verify: " + s)

        sys.addaudithook(hook)


# ---------------------------------------------------------------- statistics (S seeds)


def family_d(u, families):
    fams = sorted(set(families))
    f = np.asarray(families)
    return fams, np.stack(
        [np.array([u[s, f == x, 1].mean() - u[s, f == x, 0].mean() for x in fams]) for s in range(u.shape[0])]
    )


def sign_flip_p(d, seed=FLIP_SEED, draws=FLIP_DRAWS):
    """One-sided family-level sign-flip p for mean(d) > 0; exact for <= 20 families, else Monte Carlo."""
    d = np.asarray(d, dtype=np.float64)
    obs = d.mean()
    n = len(d)
    if n <= 20:
        signs = np.array(list(itertools.product((1.0, -1.0), repeat=n)))
        stats = (signs * d).mean(axis=1)
        return float((stats >= obs - 1e-15).mean()), "exact"
    rng = np.random.default_rng(seed)
    hits = 0
    for start in range(0, draws, 20000):
        m = min(20000, draws - start)
        s = rng.choice((1.0, -1.0), size=(m, n))
        hits += int(((s * d).mean(axis=1) >= obs - 1e-15).sum())
    return float((hits + 1) / (draws + 1)), "monte_carlo"


def bt_elo(w, lo):
    return 1500.0 + 400.0 * np.log10((w + 0.01) / (lo + 0.01))


def summarize_seeds(u, inv, families, picks):
    """u, inv: (S, contexts, 2) [reference, arm]. Shared whole-family bootstrap (RNG 302150, 10k) across seeds."""
    fams, D = family_d(u, families)
    S, F = D.shape
    idx = np.random.default_rng(BOOT_SEED).integers(0, F, size=(DRAWS, F))
    B = np.stack([D[s][idx].mean(axis=1) for s in range(S)])
    pooled = np.median(B, axis=0)
    wtl = [[int((d > TIE).sum()), int((np.abs(d) <= TIE).sum()), int((d < -TIE).sum())] for d in D]
    flips = [sign_flip_p(d) for d in D]
    order = [np.argsort(-d, kind="mergesort") for d in D]
    return dict(
        seeds=S,
        families=F,
        contexts=int(u.shape[1]),
        per_seed_family_mean_gain=D.mean(axis=1).tolist(),
        per_seed_ci95=[np.quantile(B[s], [0.025, 0.975]).tolist() for s in range(S)],
        per_seed_lcb95=[float(np.quantile(B[s], 0.05)) for s in range(S)],
        median_gain_across_seeds=float(np.median(D.mean(axis=1))),
        pooled_ci95=np.quantile(pooled, [0.025, 0.975]).tolist(),
        pooled_lcb95=float(np.quantile(pooled, 0.05)),
        per_seed_family_median_gain=np.median(D, axis=1).tolist(),
        wtl=wtl,
        sign_flip_p=[p for p, _ in flips],
        sign_flip_method=flips[0][1],
        harmed_share=[float((d < -HARM).mean()) for d in D],
        invalid=[[int(v[:, 0].sum()), int(v[:, 1].sum())] for v in np.asarray(inv, dtype=bool)],
        switches=[int((np.asarray(p) != 0).sum()) for p in picks],
        elo_by_seed=[float(bt_elo(w, lo)) for w, _, lo in wtl],
        drop_top5_gain=[float(np.delete(d, o[:5]).mean()) for d, o in zip(D, order)],
        drop_top10_gain=[float(np.delete(d, o[:10]).mean()) for d, o in zip(D, order)],
    )


def decide(s, need_flips=4):
    crit = dict(
        c1_median_gain_gt_0_001=bool(s["median_gain_across_seeds"] > 0.001),
        c2_pooled_lcb95_gt_0=bool(s["pooled_lcb95"] > 0),
        c3_sign_flip_p_lt_05_on_enough_seeds=bool(sum(p < 0.05 for p in s["sign_flip_p"]) >= need_flips),
        c4_invalid_le_reference_each=bool(all(a <= r for r, a in s["invalid"])),
        c5_harmed_le_15pct_each=bool(all(h <= 0.15 for h in s["harmed_share"])),
    )
    return crit, ("FRESH_CONFIRMED" if all(crit.values()) else "FRESH_NOT_CONFIRMED")


# ---------------------------------------------------------------- scoring with frozen locks


def score_population(r303, mods, packets, cids, lock_files):
    """Each frozen chooser picks once among a population's top-8 pool; returns (S, n) picks."""
    core, veto, bagged, ens = (mods[k] for k in ("core", "veto", "bagged", "ens"))
    X = np.concatenate([packets[c]["X"] for c in cids])
    picks = []
    for lp in lock_files:
        lk = json.loads(Path(lp).read_text())
        ch = lk["chosen"]
        outs = r303.score_members(Path(lp).parent / "refit", lk["members"], X, len(cids), ens)
        ml = [r303.member_picks(o, ch["tau"], ch["k"], veto, core, bagged) for o in outs]
        picks.append(np.asarray(r303.ens_picks(ml, ch["a"], ens), dtype=np.int32))
    return np.stack(picks)


def panel(packets, cids, picks, ref_u=None, ref_inv=None):
    S, n = picks.shape
    u = np.zeros((S, n, 2))
    inv = np.zeros((S, n, 2), dtype=bool)
    for s in range(S):
        for i, c in enumerate(cids):
            j = int(picks[s, i])
            p = packets[c]
            r_u = float(p["utility_k"][0]) if ref_u is None else ref_u[i]
            r_i = (
                (not bool(p["gate_k"][0])) or float(p["utility_k"][0]) <= -2.0 + 1e-9 if ref_inv is None else ref_inv[i]
            )
            u[s, i] = [r_u, float(p["utility_k"][j])]
            inv[s, i] = [r_i, (not bool(p["gate_k"][j])) or float(p["utility_k"][j]) <= -2.0 + 1e-9]
    return u, inv


# ---------------------------------------------------------------- phases


def s3_get(key, version, digest, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        subprocess.run(["aws", "s3api", "get-object", "--bucket", BUCKET, "--key", key, "--version-id", version, str(dest)],
                       check=True, stdout=subprocess.DEVNULL, env=dict(os.environ, AWS_PROFILE="veox-jepson-prod"))  # fmt: skip
    if sha(dest) != digest:
        raise RuntimeError("S3 object sha mismatch: " + key)
    return dest


def h310_module():
    sys.path.insert(0, str(H310))
    import run_h310 as R

    pins = json.loads((HERE / "manifest.json").read_text())["h310_sources"]
    for name, digest in pins.items():
        if sha(H310 / name) != digest:
            raise RuntimeError("H310 source pin mismatch: " + name)
    R.DATA = HARNESS
    return R


def phase_lock(workers):
    R = h310_module()
    R.check_pins()
    dev, _ = R.dev_cids()
    R.install_dev_guard(dev)
    r303, counters = R.load_train_side()
    # Member seeds for root seeds beyond H302's table: the H300 rule, verified first on 30001-30003.
    G = r303.G
    table = G["a302"].EXPECTED_MEMBER_SEEDS
    rule = lambda s: tuple(G["core"].stream_seed(s, "H300_ENS_VOTE2_MEMBER", m) for m in range(3))  # noqa: E731
    assert all(rule(s) == tuple(table[s]) for s in (30001, 30002, 30003))
    for s in NEW_SEEDS:
        table[s] = rule(s)
    counters["member_seeds_new"] = {str(s): list(table[s]) for s in NEW_SEEDS}
    R.LOCK_CTX["r303"] = r303
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(R.lock_task, NEW_SEEDS))
    print(json.dumps(dict(new_locks=res)), flush=True)
    # seeds 30001-30003: H310's archived locks, exact version, sha-verified against H310's receipt
    tar = s3_get(*H310_TAR, DATA / "h310_ref" / "h310.tar.gz")
    subprocess.run(["tar", "-xzf", str(tar), "-C", str(DATA / "h310_ref"),
                    *[f"lock/fold_4/root_{s}" for s in (30001, 30002, 30003)]], check=True)  # fmt: skip
    h310_receipt = json.loads((H310 / "lock_receipt.json").read_text())
    locks = {}
    for s in SEEDS:
        src = lock_path(DATA / "h310_ref", s) if s < 30004 else lock_path(HARNESS, s)
        if s < 30004:
            if sha(src) != h310_receipt["locks"][str(s)]["fit_lock_sha256"]:
                raise RuntimeError(f"H310 lock {s} sha mismatch")
            dst = lock_path(HARNESS, s).parent
            if not dst.exists():
                subprocess.run(["cp", "-a", str(src.parent), str(dst)], check=True)
        locks[str(s)] = sha(lock_path(HARNESS, s))
    man = dict(wave="H313", locks=locks, chosen={str(s): json.loads(lock_path(HARNESS, s).read_text())["chosen"] for s in SEEDS},
               counters=counters, dev91_rows_read_before_lock=int(counters["dlval_feature_rows_read"]),
               h310_reused=[30001, 30002, 30003], fitted_here=list(NEW_SEEDS))  # fmt: skip
    tmp = HARNESS / "lock_manifest.json.new"
    with tmp.open("w") as f:
        json.dump(man, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, HARNESS / "lock_manifest.json")
    print(json.dumps(dict(lock_manifest_sha256=sha(HARNESS / "lock_manifest.json"), **man)), flush=True)


def phase_dryrun():
    """Pseudo-fresh = DL-Train P0 outer fold 0, scored with H303's out-of-fold fold-0 locks (3 seeds)."""
    sys.path.insert(0, str(FS / "h302_ensvote2_trainseed_nested_cv"))
    sys.path.insert(1, str(FS / "h303_ensvote2_fully_nested_hparams"))
    sys.path.insert(2, str(FS / "h308_ensvote2_trainonly_repartition"))
    import run_h303 as r303
    import trainonly_loader as tl

    sel = json.loads(Path("/home/ubuntu/jope-data/h50/input/selected.json").read_text())
    r303.np = tl.TrainOnlyNumpy({r["context_sha256"] for r in sel["train"]})
    r303.SCRATCH = DATA / "dry_setup"
    r303.setup()
    r303.np = np
    G = r303.G
    tar = s3_get(*H303_TAR, DATA / "h303_ref" / "h303.tar.gz")
    subprocess.run(["tar", "-xzf", str(tar), "-C", str(DATA / "h303_ref"),
                    *[f"scratch/fold_0/root_{s}" for s in (30001, 30002, 30003)]], check=True)  # fmt: skip
    rows = [r for r in G["train_rows"] if G["a302"].outer_fold_of_family(r["family_id"]) == 0]
    cids = [r["context_sha256"] for r in rows]
    fams = [r["family_id"] for r in rows]
    locks = [DATA / "h303_ref" / "scratch" / "fold_0" / f"root_{s}" / "fit_lock.json" for s in (30001, 30002, 30003)]
    mods = {k: G[k] for k in ("core", "veto", "bagged", "ens")}
    picks = score_population(r303, mods, G["P8"], cids, locks)
    h = [hashlib.sha256(p.astype(np.int32).tobytes()).hexdigest() for p in picks]
    regression_ok = h[0] == "18fdb2538ceabd9a9e5d07ac4a80cf9ff0eaf7597d9dc5fd1645acf0566bd3aa"
    u, inv = panel(G["P8"], cids, picks)
    s = summarize_seeds(u, inv, fams, picks)
    crit, verdict = decide(s, need_flips=2)  # 3-seed dry run: >= 2/3 in place of >= 4/6
    out = dict(wave="H313", phase="dryrun", pseudo_fresh="DL-Train P0 outer fold 0", elo_label=ELO_LABEL_DRY,
               harness_regression_seed30001_picks_sha_ok=regression_ok, picks_sha256=h, summary=s, criteria=crit,
               verdict_dry=verdict, note="pipeline exercise on already-analysed DL-Train OOF; not new evidence",
               loader_counters=dict(tl.COUNTERS))  # fmt: skip
    (DATA / "dryrun.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(dict(regression_ok=regression_ok, verdict_dry=verdict, median=s["median_gain_across_seeds"],
                          pooled=s["pooled_ci95"], flips=s["sign_flip_p"])), flush=True)  # fmt: skip
    if not regression_ok:
        raise SystemExit("REGRESSION_FAIL: harness dry-run picks differ from H303 fold-0 picks")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("lock", "dryrun"))
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    if os.uname().nodename.split(".")[0] != "xbabe2" or os.getpriority(os.PRIO_PROCESS, 0) != 19:
        raise RuntimeError("H313 requires CPU xbabe2 at nice 19")
    FreshGate(HARNESS).install()
    phase_lock(a.workers) if a.phase == "lock" else phase_dryrun()
