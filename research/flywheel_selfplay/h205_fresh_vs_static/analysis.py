"""H205 statistical analysis, equal-compute 50-round schedule, H178 per-round pooled selector, and common gate."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

AUTHORITY_STRING = (
    '"Jepson via Grok Bot, card answer 8:22 PM MT" (2026-11-01, t11s3): '
    '"Start H205 and H204 now with those unblocks; H206 after xbabe3 disk is cleared."'
)
WAVE = "H205"
SEEDS = (205101, 205102, 205103)
FOLD_SEED = 205140
BOOT_SEED = 205141
ROUNDS = 50
MICROBATCHES_PER_ROUND = 16
FAMILIES_PER_MICROBATCH = 8
ARMS = ("STATIC-DATA", "STATIC-TEACHER", "FRESH")
ALL_ARMS = ("R882",) + ARMS
R882_SHA256 = "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64"
DOPE_SHA256 = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
R882_NATIVE_DLVAL_MEAN = 1.1377941898164525
DLVAL_CONTEXT_COUNT = 91
DLTRAIN_CONTEXT_COUNT = 1219
DLTRAIN_FAMILY_COUNT = 205
MAX_HARM_FRACTION = 0.10
HARM_EPS = 0.01
MIN_GAIN_MARGIN = 0.001
DRAWS = 10000
EPS = 1e-12


def _load_sibling_module(mod_name, rel_path):
    target = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(mod_name, target)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


h177_analysis = _load_sibling_module(
    "h177_pooled_selector_analysis_for_h205",
    "research/flywheel_selfplay/h177_pooled_selector/analysis.py",
)
h178_calibration = _load_sibling_module(
    "h178_pooled_calibration_for_h205",
    "research/flywheel_selfplay/h178_pooled_calibration/calibration.py",
)


def stream_seed(*parts):
    return (
        int.from_bytes(
            hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "little"
        )
        & 0x7FFFFFFF
    )


def folds(rows, fold_seed=FOLD_SEED):
    fams = sorted({r["family_id"] for r in rows})
    if len(fams) != DLTRAIN_FAMILY_COUNT:
        raise ValueError("205 TRAIN families required")
    order = np.random.default_rng(fold_seed).permutation(DLTRAIN_FAMILY_COUNT)
    return [sorted(fams[i] for i in group) for group in np.array_split(order, 5)]


def partition(rows, outer, fold_seed=FOLD_SEED):
    groups = folds(rows, fold_seed=fold_seed)
    a, b = set(groups[outer]), set(groups[(outer + 1) % 5])
    return (
        [r for r in rows if r["family_id"] not in a | b],
        [r for r in rows if r["family_id"] in b],
        [r for r in rows if r["family_id"] in a],
    )


def verify_pooled_b_coverage(rows, fold_seed=FOLD_SEED):
    """Prove that B_k never overlaps FIT_k or A_k and pooling B_0..B_4 covers all 205 families exactly once."""
    all_fams = sorted({r["family_id"] for r in rows})
    all_cids = sorted({r["context_sha256"] for r in rows})
    if len(all_fams) != DLTRAIN_FAMILY_COUNT or len(all_cids) != len(rows):
        raise ValueError("expected 205 distinct families and unique context IDs")
    b_fam_seen, a_fam_seen = [], []
    b_cid_seen, a_cid_seen = [], []
    for outer in range(5):
        fit, select, assess = partition(rows, outer, fold_seed=fold_seed)
        fit_fams = {r["family_id"] for r in fit}
        b_fams = {r["family_id"] for r in select}
        a_fams = {r["family_id"] for r in assess}
        if len(fit_fams) != 123 or len(b_fams) != 41 or len(a_fams) != 41:
            raise ValueError("invalid 123/41/41 family fold sizes")
        if (fit_fams & b_fams) or (fit_fams & a_fams) or (b_fams & a_fams):
            raise ValueError("family overlap between FIT, B, or A in fold")
        b_fam_seen.extend(sorted(b_fams))
        a_fam_seen.extend(sorted(a_fams))
        b_cid_seen.extend(sorted(r["context_sha256"] for r in select))
        a_cid_seen.extend(sorted(r["context_sha256"] for r in assess))
    if (
        sorted(b_fam_seen) != all_fams
        or sorted(a_fam_seen) != all_fams
        or sorted(b_cid_seen) != all_cids
        or sorted(a_cid_seen) != all_cids
    ):
        raise ValueError("pooled B or A failed exact-once coverage")
    return dict(
        total_families=DLTRAIN_FAMILY_COUNT,
        total_contexts=len(rows),
        b_exact_once=True,
        a_exact_once=True,
        zero_fit_b_overlap=True,
        zero_b_a_same_fold_overlap=True,
    )


def build_equal_compute_round_plan(
    fit_rows, seed, outer, rounds=ROUNDS, excluded_cids=()
):
    """Build 50 rounds x 16 family-stratified microbatches (8 distinct fit families per microbatch) for each arm."""
    if rounds != ROUNDS:
        raise ValueError(f"H205 requires exactly {ROUNDS} rounds per arm, got {rounds}")
    excl = set(excluded_cids)
    fam_map = {}
    for r in fit_rows:
        cid = r["context_sha256"]
        if cid in excl:
            raise ValueError("excluded context present in fit_rows")
        fam_map.setdefault(r["family_id"], []).append(cid)
    for f in fam_map:
        fam_map[f] = sorted(set(fam_map[f]))
    fams = sorted(fam_map)
    if len(fams) < FAMILIES_PER_MICROBATCH:
        raise ValueError("at least 8 distinct fit families required")
    plan_by_arm = {}
    for arm in ARMS:
        rounds_plan = []
        for r_idx in range(1, rounds + 1):
            stream = stream_seed(seed, outer, arm, "round", r_idx)
            rng = np.random.default_rng(stream)
            need = MICROBATCHES_PER_ROUND * FAMILIES_PER_MICROBATCH
            fam_seq = []
            while len(fam_seq) < need:
                perm = [fams[i] for i in rng.permutation(len(fams))]
                if fam_seq:
                    rem = len(fam_seq) % FAMILIES_PER_MICROBATCH
                    tail = set(fam_seq[-rem:]) if rem else set()
                    head_ok = [f for f in perm if f not in tail]
                    head_rest = [f for f in perm if f in tail]
                    perm = head_ok + head_rest
                fam_seq.extend(perm)
            fam_seq = fam_seq[:need]
            microbatches = []
            for m in range(MICROBATCHES_PER_ROUND):
                mb_fams = fam_seq[
                    m * FAMILIES_PER_MICROBATCH : (m + 1) * FAMILIES_PER_MICROBATCH
                ]
                if len(set(mb_fams)) != FAMILIES_PER_MICROBATCH:
                    raise RuntimeError("microbatch family collision")
                cids = [
                    fam_map[f][int(rng.integers(0, len(fam_map[f])))] for f in mb_fams
                ]
                microbatches.append(dict(microbatch=m, families=mb_fams, cids=cids))
            rounds_plan.append(
                dict(
                    round=r_idx,
                    arm=arm,
                    stream=stream,
                    microbatches=microbatches,
                )
            )
        plan_by_arm[arm] = rounds_plan
    return plan_by_arm


def verify_r882_native_parity(actual_values, reference_values):
    """Enforce native r882 parity 91/91 exact (1.1377941898164525) before any candidate forward."""
    actual = np.asarray(actual_values, dtype=np.float64)
    expected = np.asarray(reference_values, dtype=np.float64)
    if (
        actual.shape != (DLVAL_CONTEXT_COUNT,)
        or expected.shape != (DLVAL_CONTEXT_COUNT,)
        or actual.tobytes() != expected.tobytes()
        or float(actual.mean()) != R882_NATIVE_DLVAL_MEAN
    ):
        raise RuntimeError("91-context exact native r882 baseline parity gate failed")
    return dict(
        status="PASS",
        dl_val_contexts=DLVAL_CONTEXT_COUNT,
        mean=float(actual.mean()),
        bit_exact=True,
    )


def verify_fresh_admission_gate(receipt_path):
    """Gate FRESH arm execution on luna_v1_tel's JRB-01 ADMISSION PASS receipt."""
    p = Path(receipt_path)
    if not p.is_file():
        raise RuntimeError(f"JRB-01 admission receipt missing: {p}")
    payload = json.loads(p.read_text())
    if payload.get("status") != "PASS":
        raise RuntimeError("JRB-01 admission receipt status is not PASS")
    if int(payload.get("admitted_families", 0)) <= 0:
        raise RuntimeError("JRB-01 admission receipt has zero admitted families")
    if int(payload.get("d4_overlap_count", -1)) != 0:
        raise RuntimeError("JRB-01 admission receipt has non-zero D4 overlap")
    if int(payload.get("sealed_overlap_count", 0)) != 0:
        raise RuntimeError("JRB-01 admission receipt has non-zero sealed overlap")
    manifest_sha = str(payload.get("manifest_sha256", "")).strip()
    if len(manifest_sha) != 64:
        raise RuntimeError("JRB-01 admission receipt missing valid manifest_sha256")
    return dict(
        status="PASS",
        admitted_families=int(payload["admitted_families"]),
        admitted_datasets=int(payload.get("admitted_datasets", 0)),
        manifest_sha256=manifest_sha,
    )


def h178_round_select(rows, incumbent_values, candidate_panels_by_arm, seed=BOOT_SEED):
    """Per-round H178 calibrated 205-family pooled cross-fit selector.

    Reuses H177/H178 family-matrix & 10,000-draw paired family bootstrap after averaging the 3 seeds
    within each of the 205 families, and additionally enforces family harm <= 10% (<= 20/205 families
    degraded by > 0.01).
    """
    if not isinstance(candidate_panels_by_arm, dict) or set(
        candidate_panels_by_arm
    ) != set(ARMS):
        raise ValueError(f"exact 3-arm H205 map required ({list(ARMS)})")

    # Verify compatibility with the imported H177/H178 4-arm rule on the same 205-family panel
    h177_compat_map = {
        "CTRL": candidate_panels_by_arm["STATIC-DATA"],
        "ACC4": candidate_panels_by_arm["STATIC-TEACHER"],
        "ACC16": candidate_panels_by_arm["FRESH"],
        "OC15": candidate_panels_by_arm["FRESH"],
    }
    h177_check = h177_analysis.pooled_b_select(
        rows, incumbent_values, h177_compat_map, seed=seed
    )

    families = [r["family_id"] for r in rows]
    fams_sorted = sorted(set(families))
    n_fams = len(fams_sorted)
    if n_fams != DLTRAIN_FAMILY_COUNT:
        raise ValueError(f"pooled selector requires 205 families, got {n_fams}")

    base = np.asarray(incumbent_values, dtype=np.float64)
    if base.ndim == 1:
        if base.shape != (len(rows),):
            raise ValueError("1D incumbent_values must match len(rows)")
        base_3 = np.broadcast_to(base[None, :], (3, len(rows)))
    elif base.ndim == 2 and base.shape == (3, len(rows)):
        base_3 = base
    else:
        raise ValueError("incumbent_values must have shape (N_ctx,) or (3, N_ctx)")
    if not np.isfinite(base_3).all():
        raise ValueError("nonfinite incumbent_values")

    fm_base = np.stack(
        [h177_analysis.family_matrix(v, families)[1] for v in base_3]
    )  # (3, 205)
    ix = np.random.default_rng(seed).integers(0, n_fams, (DRAWS, n_fams))

    raw_decisions = {}
    pvals = {}
    for arm in ARMS:
        arr = np.asarray(candidate_panels_by_arm[arm], dtype=np.float64)
        if arr.ndim != 2 or arr.shape != (3, len(rows)):
            raise ValueError(f"arm {arm} must provide shape (3, {len(rows)})")
        if not np.isfinite(arr).all():
            raise ValueError(f"nonfinite candidate values in {arm}")
        fm_cand = np.stack([h177_analysis.family_matrix(v, families)[1] for v in arr])
        diff_3 = fm_cand - fm_base
        fam_diff = diff_3.mean(axis=0)  # (205,)
        gain = float(fam_diff.mean())
        boot_means = fam_diff[ix].mean(axis=1)
        lcb95, ucb95 = (
            float(np.quantile(boot_means, 0.025)),
            float(np.quantile(boot_means, 0.975)),
        )
        p_raw = float((np.sum(boot_means <= MIN_GAIN_MARGIN) + 1) / (DRAWS + 1))
        harm_count = int(np.sum(fam_diff < -HARM_EPS))
        harm_rate = float(harm_count / n_fams)
        pvals[arm] = p_raw
        raw_decisions[arm] = dict(
            arm=arm,
            pooled_b_families=n_fams,
            gain=gain,
            ci95=[lcb95, ucb95],
            lcb95=lcb95,
            ucb95=ucb95,
            p_raw=p_raw,
            harm_count=harm_count,
            harm_rate=harm_rate,
            improved_families=int(np.sum(fam_diff > EPS)),
            degraded_families=int(np.sum(fam_diff < -EPS)),
        )

    holm_p = h177_analysis.holm(pvals)
    decisions = {}
    for arm in ARMS:
        d = dict(raw_decisions[arm])
        d["p_holm"] = float(holm_p[arm])
        admit = bool(
            d["gain"] > MIN_GAIN_MARGIN + EPS
            and d["lcb95"] > 0.0
            and d["harm_rate"] <= MAX_HARM_FRACTION + EPS
            and d["p_holm"] < 0.05
        )
        d["admit"] = admit
        d["decision"] = "ADMIT" if admit else "REJECT_RETAIN_INCUMBENT"
        decisions[arm] = d

    return dict(
        wave=WAVE,
        selector="H178_calibrated_pooled_205_family",
        h177_rule_verified=bool(
            len(h177_check["CTRL"]["seed_averaged_family_means"]) == 205
        ),
        pooled_b_families=n_fams,
        decisions=decisions,
    )


def evaluate_h205_common_gate(
    r882_val_values,
    ref_val_values,
    rows,
    r882_a_values,
    arm_a_panels,
    seed=BOOT_SEED,
):
    """Evaluate the H205 common gate:
    1. Native r882 91/91 exact parity on DL-Val (mean 1.1377941898164525).
    2. 205-family pooled cross-fit LCB95 > 0 and harm <= 10% of families.
    3. Primary Holm m = 2 contrasts: FRESH vs STATIC-DATA and FRESH vs STATIC-TEACHER.
    """
    parity = verify_r882_native_parity(r882_val_values, ref_val_values)
    vs_r882 = h178_round_select(rows, r882_a_values, arm_a_panels, seed=seed)

    families = [r["family_id"] for r in rows]
    n_fams = len(set(families))
    ix = np.random.default_rng(seed + 17).integers(0, n_fams, (DRAWS, n_fams))

    fm_by_arm = {}
    for arm in ARMS:
        arr = np.asarray(arm_a_panels[arm], dtype=np.float64)
        fm_by_arm[arm] = np.stack(
            [h177_analysis.family_matrix(v, families)[1] for v in arr]
        ).mean(axis=0)

    contrasts = {}
    pvals = {}
    for ctrl in ("STATIC-DATA", "STATIC-TEACHER"):
        key = f"FRESH_minus_{ctrl}"
        diff = fm_by_arm["FRESH"] - fm_by_arm[ctrl]
        gain = float(diff.mean())
        boots = diff[ix].mean(axis=1)
        lcb95, ucb95 = (
            float(np.quantile(boots, 0.025)),
            float(np.quantile(boots, 0.975)),
        )
        p_raw = float((np.sum(boots <= MIN_GAIN_MARGIN) + 1) / (DRAWS + 1))
        harm_count = int(np.sum(diff < -HARM_EPS))
        harm_rate = float(harm_count / n_fams)
        pvals[key] = p_raw
        contrasts[key] = dict(
            contrast=key,
            gain=gain,
            ci95=[lcb95, ucb95],
            lcb95=lcb95,
            ucb95=ucb95,
            p_raw=p_raw,
            harm_count=harm_count,
            harm_rate=harm_rate,
        )

    holm_p = h177_analysis.holm(pvals)
    for key in contrasts:
        c = contrasts[key]
        c["p_holm"] = float(holm_p[key])
        c["pass"] = bool(
            c["gain"] > MIN_GAIN_MARGIN + EPS
            and c["lcb95"] > 0.0
            and c["harm_rate"] <= MAX_HARM_FRACTION + EPS
            and c["p_holm"] < 0.05
        )

    fresh_vs_r882 = vs_r882["decisions"]["FRESH"]
    candidate_advance = bool(
        parity["bit_exact"]
        and fresh_vs_r882["admit"]
        and all(contrasts[k]["pass"] for k in contrasts)
    )
    return dict(
        wave=WAVE,
        r882_parity=parity,
        vs_r882=vs_r882["decisions"],
        primary_contrasts=contrasts,
        sealed_confirmation_eligible=candidate_advance,
        verdict="CANDIDATE" if candidate_advance else "NO_ADVANCE",
    )
