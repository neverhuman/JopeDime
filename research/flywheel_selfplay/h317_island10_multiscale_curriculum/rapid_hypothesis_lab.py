#!/usr/bin/env python3
"""H317 Gate H0 neural block (H0.1-H0.3): family-grouped 5-fold OOF harness, decision helper, GP screener API, timing.

OOF protocol (PREREG §2.2): for an arm and a training seed, train on 4 family folds and predict all pool candidates of
every held-out context from its view-0 (canonical) DOPE features. Per context: top-1 index and its native utility, the
top-32 indices, the predicted heads, the R882 reference (the pinned per-context TRAIN r882 pick from H189 attempt 2,
`neural_data.load_r882_picks`; a reference only, never an input; its anchors family 1.385355 / context 1.423073 are
asserted before any true-target training) and the pool oracle. All utilities are float64. Per-context utilities are
averaged over training seeds first.

Arms
  H0.1 (Scale S, opt1)  A = 525-context family-stratified TRAIN subset x A4 views (trained only on the subset's
                        training-fold members, evaluated on the full OOF), B = 1,219 x B16, C = 1,219 x C32 + cross-view
                        latent mixup Beta(0.4, 0.4). Baseline A.
  H0.2 (H0.1 winner, opt1)  S / M / L. Baseline S (the H0.1 winner run itself is reused as H0.2-S).
  H0.3 (H0.2 winner)  opt1 / opt1p / opt2 / opt3 / opt4. Primary baseline opt1 (PREREG; reused from H0.2); the
                      isolating contrasts opt2-opt1p (SAM), opt3-opt1p (Muon), opt4-opt1p (Lion+SAM) are reported too.
Decision (PREREG §2.2 contract gate, `decide`): per challenger `common_h317.robustness_summary` vs the baseline (seed
317150); Holm step-down over the experiment's m challengers on the one-sided sign-flip p (Holm-adjusted p < 0.025); the
Holm-adjusted percentile LCB (`paired_family_bootstrap_alpha`, alpha = 0.05 / (m - rank)) > +0.001; family median >= 0;
<= 10% of families losing > 0.01; leave-one-family-out minimum gain > 0. The eligible challenger with the highest
family-equal mean wins; exact ties (and no eligible challenger) keep the baseline. `cmd_decide` first validates every
arm directory (all 1,219 contexts / 205 families, 5 folds per seed, all seeds, identical budgets, lr_mult 1.0).
Compute: every arm of every experiment trains for the same number of optimizer steps (`Budget.steps`), not epochs.
Folds are pinned to GPUs (FOLD_HOST): folds 0-2 -> xbabe1 RTX 4090, folds 3-4 -> xbabe3 RTX 3090, for every arm and
seed, so paired arm differences never mix GPU architectures. "Save the gene, not the model": fold models are written
only with keep_models=True (the H0.1 arms, whose winner is the GP screener; sha256 recorded); otherwise only config,
compact per-context predictions and receipts are kept.

Screener: `load_screener(arm_dir, fold).score(view_feats[B, 997], feat412[M, 412], proj16[M, 16], sem24[M, 24])`
returns [B, M] predicted utility (higher = better), averaged over that fold's seed models. The B views are passed as one
context's views (G = B); for Scale L they interact on the view axis unless `joint_views=False`. View features come from
`crop_shuffle_synthetic_views.context_view_feats(cid, kind)`, kind "full" (whole TRAIN table) or "search" (SEARCH rows
only, the GP pilot contexts); candidates from any DSL via `rust_evaluator.featurize_dsl_412` + Micro sem24.

CLI
  time      timing / stability only, targets randomly permuted across candidates of every context (no outcome signal,
            nothing saved except timings, loss finiteness and drift); one fold.
  oof       a real arm run on TRUE targets; requires --confirm-true-targets (only after the preregistration is pushed);
            runs this host's pinned folds; --keep-models for the H0.1 arms only.
  lock-fit  the locked arm retrained on all TRAIN contexts (models kept, sha256 in lock_models.json); TRUE targets.
  decide    the decision rule over finished arm directories (--isolating adds the H0.3 component contrasts).
"""

from __future__ import annotations

import os

# Numeric thread cap: <= 4 numeric threads per host in total (CUDA runtime threads exempt): 2 OpenMP/MKL/OpenBLAS
# threads and 2 torch intra-op threads; no DataLoader workers (all data is GPU-resident).
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common_h317 as C  # noqa: E402

NEURAL_ROOT = C.DATA_ROOT / "neural"
FOLD_HOST = {0: "xbabe1", 1: "xbabe1", 2: "xbabe1", 3: "xbabe3", 4: "xbabe3"}


def default_run_root() -> Path:
    """$H317_RUN_ROOT, else /dev/shm/h317-cl/runs on xbabe3 (its root disk is near a kill guard; outputs are pulled to
    xbabe2), else $H317_DATA/neural/runs."""
    if os.environ.get("H317_RUN_ROOT"):
        return Path(os.environ["H317_RUN_ROOT"])
    if os.uname().nodename == "xbabe3":
        return Path("/dev/shm/h317-cl/runs")
    return NEURAL_ROOT / "runs"


RUN_ROOT = default_run_root()
ARM_A_N = 525
TOP_K_OUT = 32


# ----------------------------------------------------------------------------------------------------------------------
# Arms
# ----------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ArmConfig:
    arm_id: str  # e.g. "H0.1-A"
    scale: str
    recipe: str
    views: str  # A4 | B16 | C32
    subset: str  # all | A525
    mixup: bool = False


@dataclass(frozen=True)
class Budget:
    steps: int  # optimizer steps per training run, identical for every arm (compute-matched)
    slate: int = 128
    top_k: int = 48
    batch_contexts: int = 16
    n_seeds: int = 2

    def seeds(self) -> Sequence[int]:
        return C.TRAIN_SEEDS[: self.n_seeds]


def h01_arms() -> Dict[str, ArmConfig]:
    return {
        "A": ArmConfig("H0.1-A", "S", "opt1", "A4", "A525"),
        "B": ArmConfig("H0.1-B", "S", "opt1", "B16", "all"),
        "C": ArmConfig("H0.1-C", "S", "opt1", "C32", "all", mixup=True),
    }


def h02_arms(h01_winner: ArmConfig) -> Dict[str, ArmConfig]:
    return {
        s: replace(
            h01_winner, arm_id=(h01_winner.arm_id if s == "S" else f"H0.2-{s}"), scale=s
        )
        for s in "SML"
    }


def h03_arms(h02_winner: ArmConfig) -> Dict[str, ArmConfig]:
    return {
        o: replace(
            h02_winner,
            arm_id=(h02_winner.arm_id if o == "opt1" else f"H0.3-{o}"),
            recipe=o,
        )
        for o in ("opt1", "opt1p", "opt2", "opt3", "opt4")
    }


BASELINES = {"H0.1": "A", "H0.2": "S", "H0.3": "opt1"}
# Planned contrasts per experiment (PREREG §2.2): decide() refuses any other arm set, so Holm's m is fixed in advance.
PLANNED = {
    "H0.1": {"baseline": "A", "challengers": ("B", "C"), "m": 2},
    "H0.2": {"baseline": "S", "challengers": ("M", "L"), "m": 2},
    "H0.3": {
        "baseline": "opt1",
        "challengers": ("opt1p", "opt2", "opt3", "opt4"),
        "m": 4,
    },
}
H03_ISOLATING = {
    "SAM": ("opt2", "opt1p"),
    "Muon": ("opt3", "opt1p"),
    "Lion+SAM": ("opt4", "opt1p"),
    "scaffold(WSD+SWA+curriculum)": ("opt1p", "opt1"),
}


def arm_a_subset(rows: Sequence[dict], n: int = ARM_A_N) -> List[str]:
    """Seeded family-stratified subset: largest-remainder allocation of n over families by size, then seeded picks."""
    seed = C.stream_seed("H317", "armA525", C.FOLD_SEED)
    fam: Dict[str, List[str]] = {}
    for r in rows:
        fam.setdefault(r["family_id"], []).append(r["context_sha256"])
    total = sum(len(v) for v in fam.values())
    quota = {f: n * len(v) / total for f, v in fam.items()}
    alloc = {f: int(np.floor(q)) for f, q in quota.items()}
    rest = n - sum(alloc.values())
    for f in sorted(
        fam, key=lambda f: (-(quota[f] - alloc[f]), C.stream_seed(seed, "rem", f))
    )[:rest]:
        alloc[f] += 1
    out: List[str] = []
    for f, ctx in fam.items():
        out += sorted(ctx, key=lambda c: C.stream_seed(seed, "ctx", c))[: alloc[f]]
    if len(out) != n:
        raise RuntimeError("arm A subset size")
    return sorted(out)


def fold_split(meta: dict, fold: int, subset: Optional[set] = None):
    folds = meta["fold"]
    cids = meta["cids"]
    train = [
        i
        for i in range(len(cids))
        if folds[i] != fold and (subset is None or cids[i] in subset)
    ]
    test = [i for i in range(len(cids)) if folds[i] == fold]
    fam = meta["family_id"]
    if {fam[i] for i in train} & {fam[i] for i in test}:
        raise RuntimeError("family leak across folds")
    return np.array(train, np.int64), np.array(test, np.int64)


# ----------------------------------------------------------------------------------------------------------------------
# Resource checks
# ----------------------------------------------------------------------------------------------------------------------
def mem_available_gib() -> float:
    with open("/proc/meminfo") as f:
        for ln in f:
            if ln.startswith("MemAvailable"):
                return int(ln.split()[1]) / 2**20
    return 0.0


def gpu_compute_apps() -> List[str]:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except Exception as e:  # pragma: no cover
        return [f"nvidia-smi failed: {e}"]
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def require_resources(allow_busy_gpu: bool = False) -> dict:
    apps = gpu_compute_apps()
    if apps and not allow_busy_gpu:
        raise SystemExit(f"GPU not free: {apps}")
    mem = mem_available_gib()
    if mem < 8.0:
        raise SystemExit(f"MemAvailable {mem:.1f} GiB < 8 GiB")
    return {"gpu_apps_at_start": apps, "mem_available_gib": round(mem, 1)}


def dope_hashes() -> dict:
    """Frozen DOPE: checkpoint file sha (fast) and the composite parameter/buffer sha of the frozen network."""
    import crop_shuffle_synthetic_views as V

    enc = V.DopeViewEncoder(threads=2)
    return {
        "composite_sha256": enc.dope_sha(),
        "ckpt_sha256": C.sha256_file(
            C.MICRO / "assets/models/dope_mlc_v3_multitask_best.pt"
        ),
    }


# ----------------------------------------------------------------------------------------------------------------------
# OOF harness
# ----------------------------------------------------------------------------------------------------------------------
def spec_for(cfg: ArmConfig, budget: Budget):
    import crop_shuffle_synthetic_views as V
    import curriculum_trainer as T

    return T.TrainSpec(
        scale=cfg.scale,
        recipe=cfg.recipe,
        view_subset=list(V.VIEW_SUBSETS[cfg.views]),
        mixup=cfg.mixup,
        steps=budget.steps,
        batch_contexts=budget.batch_contexts,
        slate=budget.slate,
        top_k=budget.top_k,
    )


def _list_sha(items) -> str:
    return hashlib.sha256("\n".join(sorted(items)).encode()).hexdigest()


def save_fold_model(
    path: Path,
    model,
    cfg: ArmConfig,
    spec,
    budget: Budget,
    seed: int,
    fold: int,
    stats: dict,
    train_cids: Sequence[str],
    train_families: Sequence[str],
    keep_model: bool = True,
) -> Optional[str]:
    """Writes config.json always (with the sorted training cid / family lists and their sha256); model.pt (carrying
    the same lists, so a screener refuses any target in its training set) and its sha256 only when keep_model."""
    import torch

    path.mkdir(parents=True, exist_ok=True)
    train = {
        "train_cids": sorted(set(train_cids)),
        "train_families": sorted(set(train_families)),
    }
    train["train_cids_sha256"] = _list_sha(train["train_cids"])
    train["train_families_sha256"] = _list_sha(train["train_families"])
    sha = None
    if keep_model:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "scale": cfg.scale,
                "arm": asdict(cfg),
                "seed": seed,
                "fold": fold,
                **train,
            },
            path / "model.pt",
        )
        sha = C.sha256_file(path / "model.pt")
    C.write_json(
        path / "config.json",
        {
            "arm": asdict(cfg),
            "spec": spec.to_dict(),
            "budget": asdict(budget),
            "seed": seed,
            "fold": fold,
            "stats": stats,
            "model_kept": keep_model,
            "model_sha256": sha,
            **train,
        },
    )
    return sha


def folds_for_host(host: Optional[str] = None) -> List[int]:
    host = host or os.uname().nodename
    return [j for j, h in FOLD_HOST.items() if h == host]


def run_arm(
    cfg: ArmConfig,
    budget: Budget,
    tt,
    meta: dict,
    device,
    out_root: Path = RUN_ROOT,
    folds: Optional[Sequence[int]] = None,
    keep_models: bool = False,
    allow_any_host: bool = False,
) -> Path:
    """Train/predict every (seed, fold) of one arm on TRUE targets; writes predictions, receipts and (keep_models only)
    the fold models. Folds default to this host's pinned folds (FOLD_HOST); other folds are refused."""
    import torch

    import curriculum_trainer as T

    host = os.uname().nodename
    folds = list(folds_for_host(host) if folds is None else folds)
    if not allow_any_host and any(FOLD_HOST[j] != host for j in folds):
        raise SystemExit(
            f"fold->GPU pinning: {host} may only run folds {folds_for_host(host)}"
        )
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    arm_dir = out_root / cfg.arm_id
    subset = set(arm_a_subset(C.train_rows())) if cfg.subset == "A525" else None
    spec = spec_for(cfg, budget)
    u_all = meta["u64"]  # float64 native utilities
    if "r882_pick" not in meta:
        raise RuntimeError("run_arm needs the pinned R882 picks (true-target meta)")
    for seed in budget.seeds():
        for j in folds:
            d = arm_dir / f"seed{seed}" / f"fold{j}"
            if (d / "preds.npz").exists():
                continue
            tr, te = fold_split(meta, j, subset)
            run_seed = C.stream_seed(
                "H317-neural", seed, j
            )  # (seed, fold) only: arms are paired on init/batch draws
            model, stats = T.train_one(tt, tr, spec, run_seed % (2**31), device)
            stats = {
                **stats,
                "n_train": len(tr),
                "n_test": len(te),
                "host": host,
                "gpu": gpu,
            }
            sha = save_fold_model(
                d,
                model,
                cfg,
                spec,
                budget,
                seed,
                j,
                stats,
                [meta["cids"][i] for i in tr],
                [meta["family_id"][i] for i in tr],
                keep_model=keep_models,
            )
            preds = T.predict_contexts(model, tt, te, device)
            rows = {
                "cids": [],
                "top1": [],
                "top1_u": [],
                "top32": [],
                "r882_idx": [],
                "r882_u": [],
                "oracle_u": [],
            }
            heads = np.zeros((len(te), u_all.shape[1], 4), np.float16)
            for k, (i, p) in enumerate(zip(te, preds)):
                n = len(p)
                s = p[:, 0]
                top = np.argsort(-s, kind="stable")
                r882 = int(meta["r882_pick"][i])
                rows["cids"].append(meta["cids"][i])
                rows["top1"].append(int(top[0]))
                rows["top1_u"].append(float(u_all[i, top[0]]))
                rows["top32"].append(top[:TOP_K_OUT])
                rows["r882_idx"].append(r882)
                rows["r882_u"].append(float(u_all[i, r882]))
                rows["oracle_u"].append(float(u_all[i, :n].max()))
                heads[k, :n] = p.astype(np.float16)
            np.savez_compressed(
                d / "preds.npz",
                cids=np.array(rows["cids"]),
                top1=np.array(rows["top1"]),
                top1_u=np.array(rows["top1_u"]),
                top32=np.stack(rows["top32"]),
                r882_idx=np.array(rows["r882_idx"]),
                r882_u=np.array(rows["r882_u"]),
                oracle_u=np.array(rows["oracle_u"]),
                heads=heads,
            )
            C.write_json(
                d / "receipt.json",
                {
                    "arm": cfg.arm_id,
                    "seed": seed,
                    "fold": j,
                    "stats": stats,
                    "model_sha256": sha,
                    "guard": C.guard_report(),
                },
            )
            del model
    return arm_dir


def fit_full(
    cfg: ArmConfig, budget: Budget, tt, meta: dict, device, out_dir: Path, seed: int
) -> str:
    """LOCK model: the locked arm retrained on all 1,219 TRAIN contexts (A525 arms: the whole subset); kept, sha256."""
    import curriculum_trainer as T

    subset = set(arm_a_subset(C.train_rows())) if cfg.subset == "A525" else None
    tr = np.array(
        [i for i, c in enumerate(meta["cids"]) if subset is None or c in subset],
        np.int64,
    )
    spec = spec_for(cfg, budget)
    model, stats = T.train_one(
        tt, tr, spec, C.stream_seed("H317-neural-lock", seed) % (2**31), device
    )
    sha = save_fold_model(
        out_dir / f"seed{seed}",
        model,
        cfg,
        spec,
        budget,
        seed,
        -1,
        {**stats, "n_train": len(tr)},
        [meta["cids"][i] for i in tr],
        [meta["family_id"][i] for i in tr],
        keep_model=True,
    )
    return sha


def arm_context_utilities(arm_dir: Path, key: str = "top1_u") -> Dict[str, float]:
    """Per-context utility averaged over training seeds (every seed must cover every context)."""
    per: Dict[str, List[float]] = {}
    seeds = sorted(p for p in Path(arm_dir).glob("seed*") if p.is_dir())
    for sd in seeds:
        for f in sorted(sd.glob("fold*/preds.npz")):
            with np.load(f, allow_pickle=False) as z:
                for c, v in zip(z["cids"].astype(str), z[key]):
                    per.setdefault(c, []).append(float(v))
    if any(len(v) != len(seeds) for v in per.values()):
        raise RuntimeError(f"{arm_dir}: incomplete seed coverage")
    return {c: float(np.mean(v)) for c, v in per.items()}


def isolating_contrasts(
    per_arm: Dict[str, Dict[str, float]],
    fam_of: Dict[str, str],
    seed: int = C.BOOT_SEED_GATE_H0,
) -> dict:
    """H0.3 secondary report: component contrasts vs Opt1+ (and the scaffold vs Opt1); descriptive, not the rule."""
    out = {}
    for name, (a, b) in H03_ISOLATING.items():
        if a in per_arm and b in per_arm:
            out[name] = {
                "arm": a,
                "vs": b,
                **C.paired_family_bootstrap(per_arm[a], per_arm[b], fam_of, seed),
            }
    return out


CONTRACT_MARGIN = 0.001
HOLM_P = 0.025  # one-sided sign-flip threshold (two-sided 0.05)


def holm_adjust(pvals: Dict[str, float]) -> Dict[str, dict]:
    """Holm step-down: rank r (0-based, ascending p) has adjusted p = max_{j<=r} min(1, (m - j) * p_(j))."""
    m = len(pvals)
    order = sorted(pvals, key=lambda a: (pvals[a], a))
    out, running = {}, 0.0
    for r, a in enumerate(order):
        running = max(running, min(1.0, (m - r) * pvals[a]))
        out[a] = {
            "rank": r,
            "m": m,
            "p_raw": pvals[a],
            "p_holm": running,
            "lcb_alpha": C.holm_adjusted_lcb_alpha(m, r),
        }
    return out


def decide(
    per_arm: Dict[str, Dict[str, float]],
    experiment: str,
    fam_of: Dict[str, str],
    seed: int = C.BOOT_SEED_GATE_H0,
) -> dict:
    """PREREG §2.2 contract gate with Holm over the experiment's m challengers (see the module docstring). The arm
    names must be exactly PLANNED[experiment] (baseline + its m challengers); anything else is refused."""
    if experiment not in PLANNED:
        raise ValueError(
            f"unknown experiment {experiment!r}; planned: {sorted(PLANNED)}"
        )
    plan = PLANNED[experiment]
    baseline = plan["baseline"]
    challengers = sorted(a for a in per_arm if a != baseline)
    if (
        baseline not in per_arm
        or len(challengers) != plan["m"]
        or set(challengers) != set(plan["challengers"])
    ):
        raise ValueError(
            f"{experiment}: arms {sorted(per_arm)} != planned baseline {baseline} + m={plan['m']} challengers "
            f"{list(plan['challengers'])}"
        )
    base = per_arm[baseline]
    out = {
        "experiment": experiment,
        "baseline": baseline,
        "baseline_family_mean": C.family_equal_mean(base, fam_of),
        "m": len(challengers),
        "seed": seed,
        "margin": CONTRACT_MARGIN,
        "challengers": {},
    }
    summ = {
        a: C.robustness_summary(per_arm[a], base, fam_of, seed) for a in challengers
    }
    holm = (
        holm_adjust({a: summ[a]["signflip_p_one_sided"] for a in challengers})
        if challengers
        else {}
    )
    eligible = []
    for a in challengers:
        h = holm[a]
        adj = C.paired_family_bootstrap_alpha(
            per_arm[a], base, fam_of, seed, alpha=h["lcb_alpha"]
        )
        gs = {
            **summ[a],
            "holm_lcb": adj["lcb"],
            "signflip_p_raw": summ[a]["signflip_p_one_sided"],
            "signflip_p_one_sided": h["p_holm"],
        }
        gate = C.contract_advance_gate(gs, lcb_key="holm_lcb", margin=CONTRACT_MARGIN)
        fm = C.family_equal_mean(per_arm[a], fam_of)
        out["challengers"][a] = {
            "family_mean": fm,
            "summary": summ[a],
            "holm": h,
            "holm_lcb": adj["lcb"],
            "holm_ci": [adj["lcb"], adj["ucb"]],
            "gate": gate,
            "eligible": gate["pass"],
        }
        if gate["pass"]:
            eligible.append((fm, a))
    if eligible:
        best = max(fm for fm, _ in eligible)
        top = sorted(a for fm, a in eligible if fm == best)
        out["winner"] = (
            top[0] if len(top) == 1 else baseline
        )  # exact ties never advance
    else:
        out["winner"] = baseline
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Screener API for the GP track
# ----------------------------------------------------------------------------------------------------------------------
class ScreenerLeak(RuntimeError):
    pass


def family_of(cid: str) -> Optional[str]:
    sel = C.load_selected()
    for role in ("train", "dev"):
        for r in sel[role]:
            if r.get("context_sha256") == cid:
                return r.get("family_id")
    return None


class Screener:
    """Seed-averaged OOF models of one fold plus the union of their training cid / family sets. Every score() is for
    one target context, which must lie outside those sets (checked; `target` bound at load time or given per call)."""

    def __init__(
        self, models, device, train_cids, train_families, target: Optional[tuple] = None
    ):
        self.models = models
        self.device = device
        self.train_cids = frozenset(train_cids)
        self.train_families = frozenset(train_families)
        self.target = None
        if target is not None:
            self.target = self.assert_target(*target)

    def assert_target(self, cid: str, family: Optional[str] = None) -> tuple:
        fam = family or family_of(cid)
        if fam is None:
            raise ScreenerLeak(f"unknown family for {cid[:12]}: pass target_family")
        if cid in self.train_cids or fam in self.train_families:
            raise ScreenerLeak(
                f"target {cid[:12]} / family {fam[:20]} is in this screener's training set"
            )
        return cid, fam

    def score(
        self,
        view_feats: np.ndarray,
        feat412: np.ndarray,
        proj16: np.ndarray,
        sem24: np.ndarray,
        joint_views: bool = True,
        target_cid: Optional[str] = None,
        target_family: Optional[str] = None,
    ) -> np.ndarray:
        """view_feats [B, 997] (`context_view_feats(cid, kind)`); candidates [M, ...] -> [B, M] predicted utility."""
        import torch

        import multiscale_axial_bridge as MB

        if target_cid is not None:
            self.assert_target(target_cid, target_family)
        elif self.target is None:
            raise ScreenerLeak(
                "score() needs target_cid (this screener was loaded without a bound target)"
            )
        v = np.asarray(view_feats, np.float32).reshape(-1, MB.CTX_DIM)
        # training stores candidate vectors as float16: round identically
        cand = (
            MB.candidate_vector(feat412, proj16, sem24)
            .astype(np.float16)
            .astype(np.float32)
        )
        B, M = v.shape[0], cand.shape[0]
        ct = torch.as_tensor(cand, device=self.device).unsqueeze(0)
        vt = torch.as_tensor(v, device=self.device)
        acc = np.zeros((B, M), np.float64)
        with torch.no_grad():
            for m in self.models:
                if joint_views:
                    s = m(vt.unsqueeze(0), ct)[0, :, :, 0]
                else:
                    s = m(vt.unsqueeze(1), ct.expand(B, M, MB.CAND_DIM))[:, 0, :, 0]
                acc += s.float().cpu().numpy()
        return (acc / len(self.models)).astype(np.float32)


def _load_fold_models(
    run_dir: Path | str, fold: int, seeds: Optional[Sequence[int]], device: str
):
    import torch

    import multiscale_axial_bridge as MB

    run_dir = Path(run_dir)
    paths = sorted(run_dir.glob(f"seed*/fold{fold}/model.pt"))
    if seeds is not None:
        paths = [p for p in paths if int(p.parent.parent.name[4:]) in set(seeds)]
    if not paths:
        raise FileNotFoundError(f"no fold-{fold} models under {run_dir}")
    dev = torch.device(device)
    models, cids, fams = [], set(), set()
    for p in paths:
        ck = torch.load(p, map_location=dev, weights_only=False)
        if (
            "train_cids" not in ck
            or _list_sha(ck["train_cids"]) != ck["train_cids_sha256"]
            or _list_sha(ck["train_families"]) != ck["train_families_sha256"]
        ):
            raise ScreenerLeak(f"{p}: training manifest missing or corrupt")
        cids |= set(ck["train_cids"])
        fams |= set(ck["train_families"])
        m = MB.build_model(ck["scale"]).to(dev)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        models.append(m)
    return models, dev, cids, fams


def load_screener(
    run_dir: Path | str,
    fold: int,
    target_cid: str,
    target_family: Optional[str] = None,
    seeds: Optional[Sequence[int]] = None,
    device: str = "cpu",
) -> Screener:
    """The OOF model(s) of `fold` (one per seed; scores averaged), bound to `target_cid`. Raises ScreenerLeak unless
    target_cid AND its family (from the pinned selected.json, or `target_family`) are absent from every loaded model's
    training set (sorted cid / family lists + sha256 stored in each model.pt)."""
    models, dev, cids, fams = _load_fold_models(run_dir, fold, seeds, device)
    return Screener(models, dev, cids, fams, target=(target_cid, target_family))


def load_fold_screener(
    run_dir: Path | str,
    fold: int,
    seeds: Optional[Sequence[int]] = None,
    device: str = "cpu",
) -> Screener:
    """Unbound variant for screening many contexts of one fold: every score() call must pass target_cid (checked)."""
    models, dev, cids, fams = _load_fold_models(run_dir, fold, seeds, device)
    return Screener(models, dev, cids, fams)


# ----------------------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------------------
def _device(threads: int):
    import torch

    if not 1 <= threads <= 4:
        raise SystemExit("numeric thread cap: 1 <= threads <= 4")
    torch.set_num_threads(threads)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def cmd_time(args) -> None:
    """Timing / stability only: TARGETS RANDOMLY PERMUTED across the candidates of every context (no outcome signal);
    nothing but timings, loss finiteness and parameter drift is saved. Case syntax for --cases (comma separated):
    scale:recipe:views:mixup01:subset[:lr_mult]."""
    import torch

    import crop_shuffle_synthetic_views as V
    import curriculum_trainer as T
    import neural_data as ND

    C.install_guard()
    res = require_resources(args.allow_busy_gpu)
    dev = _device(args.threads)
    dope_before = dope_hashes()
    t0 = time.time()
    tt, meta = ND.build_train_tensors(dev, permute_seed=args.permute_seed)
    load_s = time.time() - t0
    rows = C.train_rows()
    subset = set(arm_a_subset(rows))
    out = {
        "schema": "jope.h317.neural-timing/v2",
        "host": os.uname().nodename,
        "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
        "targets": f"PERMUTED across candidates per context (seed stream H317-timing-permute/{args.permute_seed})",
        "fold": args.fold,
        "steps": args.steps,
        "threads": args.threads,
        "env_threads": {
            k: os.environ.get(k)
            for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "load_seconds": round(load_s, 1),
        "resources": res,
        "dope_before": dope_before,
        "cases": [],
    }
    for case_s in args.cases.split(","):
        parts = case_s.split(":")
        sc, rc, vw, mx, sub = parts[:5]
        lr_mult = float(parts[5]) if len(parts) > 5 else 1.0
        tr, te = fold_split(meta, args.fold, subset if sub == "A525" else None)
        spec = T.TrainSpec(
            scale=sc,
            recipe=rc,
            view_subset=list(V.VIEW_SUBSETS[vw]),
            mixup=mx == "1",
            steps=args.steps,
            batch_contexts=args.batch_contexts,
            slate=args.slate,
            top_k=args.top_k,
            lr_mult=lr_mult,
        )
        T.train_one(
            tt, tr, replace(spec, steps=5), 12345, dev
        )  # warm-up (kernels, allocator)
        model, st = T.train_one(tt, tr, spec, 12346, dev)
        torch.cuda.synchronize() if dev.type == "cuda" else None
        tp = time.time()
        T.predict_contexts(model, tt, te, dev)
        torch.cuda.synchronize() if dev.type == "cuda" else None
        pred_s = time.time() - tp
        case = {
            "case": case_s,
            "scale": sc,
            "recipe": rc,
            "views": vw,
            "mixup": mx == "1",
            "subset": sub,
            "lr_mult": lr_mult,
            "n_train": len(tr),
            "n_test": len(te),
            "slate": args.slate,
            "batch_contexts": args.batch_contexts,
            "steps": st["steps"],
            "epoch_equivalents": st["epoch_equivalents"],
            "views_per_step": st["views_per_step"],
            "params": st["params"],
            "train_seconds": st["train_seconds"],
            "sec_per_step": st["sec_per_step"],
            "predict_fold_seconds": pred_s,
            "fold_run_seconds_est": st["train_seconds"] + pred_s,
            "peak_mem_mb": st["peak_mem_mb"],
            "finite": st["finite"],
            "loss_first": st["loss_first"],
            "loss_last": st["loss_last"],
            "drift_raw": st["drift_raw"],
            "drift_eval": st["drift_eval"],
        }
        out["cases"].append(case)
        print(json.dumps(case), flush=True)
        del model
    out["dope_after"] = dope_hashes()
    out["dope_unchanged"] = (
        out["dope_after"] == dope_before
        and dope_before["composite_sha256"] == C.DOPE_SHA256
    )
    out["gpu_apps_at_end"] = gpu_compute_apps()
    out["guard"] = C.guard_report()
    C.write_json(
        NEURAL_ROOT / "receipts" / f"timing_{os.uname().nodename}_{args.tag}.json", out
    )
    print(json.dumps({k: v for k, v in out.items() if k != "cases"}))


def _true_target_setup(args):
    if not args.confirm_true_targets:
        raise SystemExit(
            "this trains on TRUE targets: pass --confirm-true-targets only after the preregistration is pushed"
        )
    import neural_data as ND

    C.install_guard()
    res = require_resources(args.allow_busy_gpu)
    dev = _device(args.threads)
    dope_before = dope_hashes()
    tt, meta = ND.build_train_tensors(dev)
    res["r882_regression_gate"] = ND.r882_anchor_check(
        meta
    )  # raises before any training on a mismatch
    return res, dev, dope_before, tt, meta


def cmd_oof(args) -> None:
    res, dev, dope_before, tt, meta = _true_target_setup(args)
    arms = json.loads(
        Path(args.arms_json).read_text()
    )  # [{"arm_id":..., "scale":..., ...}, ...]
    budget = Budget(**json.loads(args.budget_json))
    folds = [int(f) for f in args.folds.split(",")] if args.folds else None
    for a in arms:
        run_arm(
            ArmConfig(**a),
            budget,
            tt,
            meta,
            dev,
            Path(args.out_root),
            folds=folds,
            keep_models=args.keep_models,
        )
    after = dope_hashes()
    C.write_json(
        Path(args.out_root)
        / f"oof_receipt_{os.uname().nodename}_{int(time.time())}.json",
        {
            "arms": arms,
            "budget": asdict(budget),
            "folds": folds or folds_for_host(),
            "keep_models": args.keep_models,
            "resources": res,
            "dope_before": dope_before,
            "dope_after": after,
            "dope_unchanged": after == dope_before
            and after["composite_sha256"] == C.DOPE_SHA256,
            "guard": C.guard_report(),
        },
    )


def cmd_lock_fit(args) -> None:
    res, dev, dope_before, tt, meta = _true_target_setup(args)
    cfg = ArmConfig(**json.loads(args.arm_json))
    budget = Budget(**json.loads(args.budget_json))
    shas = {
        s: fit_full(cfg, budget, tt, meta, dev, Path(args.out_dir), s)
        for s in budget.seeds()
    }
    after = dope_hashes()
    C.write_json(
        Path(args.out_dir) / "lock_models.json",
        {
            "arm": asdict(cfg),
            "budget": asdict(budget),
            "model_sha256": shas,
            "resources": res,
            "dope_before": dope_before,
            "dope_after": after,
            "dope_unchanged": after == dope_before,
            "guard": C.guard_report(),
        },
    )


def validate_arm_dirs(arm_dirs: Dict[str, Path]) -> dict:
    """HIGH-3 / MEDIUM-4: every arm has every expected seed, 5 folds per seed, all 1,219 TRAIN contexts / 205 families
    per seed, one identical budget across arms, spec.steps == budget.steps and lr_mult == 1.0 everywhere."""
    rows = C.train_rows()
    all_cids = {r["context_sha256"] for r in rows}
    fam_of = {r["context_sha256"]: r["family_id"] for r in rows}
    budgets, report = set(), {}
    for name, d in arm_dirs.items():
        d = Path(d)
        confs = sorted(d.glob("seed*/fold*/config.json"))
        if not confs:
            raise RuntimeError(f"{name}: no fold configs under {d}")
        n_seeds = {json.loads(c.read_text())["budget"]["n_seeds"] for c in confs}
        if len(n_seeds) != 1:
            raise RuntimeError(f"{name}: mixed n_seeds {n_seeds}")
        seeds = list(C.TRAIN_SEEDS[: n_seeds.pop()])
        per = {}
        for s_ in seeds:
            sd = d / f"seed{s_}"
            folds = sorted(
                int(p.name[4:]) for p in sd.glob("fold*") if (p / "preds.npz").exists()
            )
            if folds != list(range(C.N_FOLDS)):
                raise RuntimeError(
                    f"{name}/seed{s_}: folds with predictions {folds} != 0..4"
                )
            cids = []
            for j in folds:
                conf = json.loads((sd / f"fold{j}" / "config.json").read_text())
                b = conf["budget"]
                key = (
                    b["steps"],
                    b["slate"],
                    b["top_k"],
                    b["batch_contexts"],
                    b["n_seeds"],
                )
                budgets.add(key)
                if (
                    conf["spec"]["steps"] != b["steps"]
                    or float(conf["spec"].get("lr_mult", 1.0)) != 1.0
                ):
                    raise RuntimeError(
                        f"{name}/seed{s_}/fold{j}: spec.steps/lr_mult differ from the budget"
                    )
                with np.load(sd / f"fold{j}" / "preds.npz", allow_pickle=False) as z:
                    cids += list(z["cids"].astype(str))
            if len(cids) != len(set(cids)) or set(cids) != all_cids:
                raise RuntimeError(
                    f"{name}/seed{s_}: predictions cover {len(set(cids))} contexts (dups {len(cids) - len(set(cids))})"
                )
            per[s_] = {
                "folds": len(folds),
                "contexts": len(set(cids)),
                "families": len({fam_of[c] for c in cids}),
            }
            if per[s_]["families"] != 205:
                raise RuntimeError(f"{name}/seed{s_}: {per[s_]['families']} families")
        report[name] = {"seeds": seeds, "per_seed": per}
    if len(budgets) != 1:
        raise RuntimeError(f"arms do not share one budget: {sorted(budgets)}")
    b = budgets.pop()
    return {
        "arms": report,
        "budget": dict(
            zip(("steps", "slate", "top_k", "batch_contexts", "n_seeds"), b)
        ),
    }


def cmd_decide(args) -> None:
    C.install_guard()
    fam_of = {r["context_sha256"]: r["family_id"] for r in C.train_rows()}
    arm_dirs = dict(kv.split("=", 1) for kv in args.arms)
    validation = validate_arm_dirs({k: Path(v) for k, v in arm_dirs.items()})
    per = {k: arm_context_utilities(Path(v)) for k, v in arm_dirs.items()}
    rep = decide(per, args.experiment, fam_of)
    rep["validation"] = validation
    if args.isolating:
        rep["isolating_contrasts"] = isolating_contrasts(per, fam_of)
    rep["guard"] = C.guard_report()
    print(json.dumps(rep, indent=1))
    if args.out:
        C.write_json(Path(args.out), rep)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("time")
    t.add_argument(
        "--cases", required=True, help="scale:recipe:views:mixup01:subset[:lr_mult],..."
    )
    t.add_argument("--fold", type=int, default=0)
    t.add_argument("--steps", type=int, default=300)
    t.add_argument("--slate", type=int, default=128)
    t.add_argument("--top-k", type=int, default=48)
    t.add_argument("--batch-contexts", type=int, default=16)
    t.add_argument("--permute-seed", type=int, default=317199)
    t.add_argument("--threads", type=int, default=2)
    t.add_argument("--tag", default="t")
    t.add_argument("--allow-busy-gpu", action="store_true")
    for name in ("oof", "lock-fit"):
        o = sub.add_parser(name)
        o.add_argument("--budget-json", required=True)
        o.add_argument("--threads", type=int, default=2)
        o.add_argument("--confirm-true-targets", action="store_true")
        o.add_argument("--allow-busy-gpu", action="store_true")
        if name == "oof":
            o.add_argument("--arms-json", required=True)
            o.add_argument("--out-root", default=str(RUN_ROOT))
            o.add_argument(
                "--folds",
                default="",
                help="default: this host's pinned folds (FOLD_HOST)",
            )
            o.add_argument("--keep-models", action="store_true")
        else:
            o.add_argument("--arm-json", required=True)
            o.add_argument("--out-dir", required=True)
    d = sub.add_parser("decide")
    d.add_argument("--experiment", required=True, choices=sorted(PLANNED))
    d.add_argument("--arms", nargs="+", required=True, help="name=arm_dir ...")
    d.add_argument("--isolating", action="store_true")
    d.add_argument("--out", default="")
    args = ap.parse_args()
    {"time": cmd_time, "oof": cmd_oof, "lock-fit": cmd_lock_fit, "decide": cmd_decide}[
        args.cmd
    ](args)


if __name__ == "__main__":
    main()
