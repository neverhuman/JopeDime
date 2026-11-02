"""H205 statistical analysis, equal-compute 50-round schedule, H178 per-round pooled selector, and common gate."""

import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
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
SLATE_CANDIDATES = 24
PROGRAM_DIM = 412
TOKEN_STEPS = 10
TOKEN_DIM = 32
NATIVE_RUST_EVALUATOR_ID = "bootstrap.s1main --fit-s1"

# Cumulative campaign budget ledger (preserved across attempts; never reset)
CAMPAIGN_WALL_BUDGET_CAP_SECONDS = 25200.0
PRIOR_ATTEMPTS_ELAPSED_SECONDS = 1970.9372255061753
REMAINING_WALL_BUDGET_SECONDS = (
    CAMPAIGN_WALL_BUDGET_CAP_SECONDS - PRIOR_ATTEMPTS_ELAPSED_SECONDS
)


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


def _is_hex64(value):
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _row_family_aliases(row):
    aliases = set()
    fid = row.get("family_id") or row.get("family")
    if isinstance(fid, str) and fid.strip():
        aliases.add(fid.strip())
    fk = row.get("family_key")
    if isinstance(fk, str) and fk.strip():
        aliases.add(fk.strip())
    raw_aliases = row.get("family_aliases") or row.get("union_evidence") or ()
    if isinstance(raw_aliases, (list, tuple, set)):
        for item in raw_aliases:
            if isinstance(item, str) and item.strip():
                aliases.add(item.strip())
    return aliases


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


def partition_fresh_rows(fresh_rows, outer, fold_seed=FOLD_SEED):
    """Partition admitted fresh D4 families into 5 deterministic folds and return FIT_outer rows (3/5 of fresh families)."""
    fams = sorted({r["family_id"] for r in fresh_rows})
    if len(fams) < FAMILIES_PER_MICROBATCH * 2:
        raise ValueError(
            f"fresh pool requires at least {FAMILIES_PER_MICROBATCH * 2} distinct families, got {len(fams)}"
        )
    order = np.random.default_rng(fold_seed + 1000).permutation(len(fams))
    groups = [sorted(fams[i] for i in group) for group in np.array_split(order, 5)]
    hold_a, hold_b = set(groups[outer]), set(groups[(outer + 1) % 5])
    fit_rows = [r for r in fresh_rows if r["family_id"] not in (hold_a | hold_b)]
    if len({r["family_id"] for r in fit_rows}) < FAMILIES_PER_MICROBATCH:
        raise ValueError("fresh FIT fold has fewer than 8 distinct families")
    return fit_rows


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


def _build_single_schedule(
    fit_rows, stream_tag, seed, outer, rounds=ROUNDS, excluded_cids=()
):
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

    rounds_plan = []
    for r_idx in range(1, rounds + 1):
        stream = stream_seed(seed, outer, stream_tag, "round", r_idx)
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
            cids = [fam_map[f][int(rng.integers(0, len(fam_map[f])))] for f in mb_fams]
            microbatches.append(dict(microbatch=m, families=mb_fams, cids=cids))
        rounds_plan.append(
            dict(
                round=r_idx,
                arm=stream_tag,
                stream=stream,
                microbatches=microbatches,
            )
        )
    return rounds_plan


def build_equal_compute_round_plan(
    fit_rows, seed, outer, rounds=ROUNDS, excluded_cids=(), fresh_fit_rows=None
):
    """Build 50 rounds x 16 family-stratified microbatches (8 distinct fit families per microbatch) for each arm.

    When `fresh_fit_rows` is provided (prospective P1-repaired mode), `STATIC-DATA` uses `fit_rows`
    while `STATIC-TEACHER` and `FRESH` share the exact same fresh-family schedule built from
    `fresh_fit_rows` (`stream_tag="FRESH_SHARED"`), isolating fresh covariate support from fresh
    ground-truth native-Rust-CV labels.
    """
    static_plan = _build_single_schedule(
        fit_rows,
        "STATIC-DATA",
        seed,
        outer,
        rounds=rounds,
        excluded_cids=excluded_cids,
    )
    if fresh_fit_rows is not None:
        static_fams = {r["family_id"] for r in fit_rows}
        fresh_fams = {r["family_id"] for r in fresh_fit_rows}
        if static_fams & fresh_fams:
            raise ValueError(
                "fresh_fit_rows must be strictly disjoint from static fit_rows families"
            )
        shared_fresh = _build_single_schedule(
            fresh_fit_rows,
            "FRESH_SHARED",
            seed,
            outer,
            rounds=rounds,
            excluded_cids=excluded_cids,
        )
        teacher_plan = [
            dict(
                round=r["round"],
                arm="STATIC-TEACHER",
                stream=r["stream"],
                microbatches=[dict(mb) for mb in r["microbatches"]],
            )
            for r in shared_fresh
        ]
        fresh_plan = [
            dict(
                round=r["round"],
                arm="FRESH",
                stream=r["stream"],
                microbatches=[dict(mb) for mb in r["microbatches"]],
            )
            for r in shared_fresh
        ]
        return {
            "STATIC-DATA": static_plan,
            "STATIC-TEACHER": teacher_plan,
            "FRESH": fresh_plan,
        }

    return {
        arm: _build_single_schedule(
            fit_rows, arm, seed, outer, rounds=rounds, excluded_cids=excluded_cids
        )
        for arm in ARMS
    }


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
    if not _is_hex64(manifest_sha):
        raise RuntimeError("JRB-01 admission receipt missing valid manifest_sha256")
    return dict(
        status="PASS",
        admitted_families=int(payload["admitted_families"]),
        admitted_datasets=int(payload.get("admitted_datasets", 0)),
        manifest_sha256=manifest_sha,
        fresh_pool_manifest=payload.get("fresh_pool_manifest"),
    )


def extract_and_verify_fresh_row_payload(row):
    """Extract and hash-verify a fresh row's program, fitness, gate, tokens_10x32, and mask_10 arrays.

    If `pool_npz_path` is present on the row, loads the NPZ, verifies `pool_sha256`, and enforces
    that the NPZ's `fitness`, `gate`, `program`, `tokens_10x32`, and `mask_10` are byte-identical
    to both the row arrays (if inline arrays are also supplied) and the receipt digests
    (`labels_sha256`, `programs_sha256`, `features_sha256`).
    """
    ds_id = str(row.get("dataset_id") or row.get("dataset") or "").strip()
    cid = str(row.get("context_sha256", "")).strip()
    receipt = row.get("native_evaluator_receipt")
    if not isinstance(receipt, dict):
        raise RuntimeError(f"fresh row ({ds_id}) missing native_evaluator_receipt")

    npz_data = None
    pool_npz_path = row.get("pool_npz_path")
    if pool_npz_path is not None:
        p = Path(pool_npz_path)
        if not p.is_file():
            raise RuntimeError(f"fresh row ({ds_id}) pool_npz_path missing: {p}")
        expected_pool_sha = str(row.get("pool_sha256", "")).strip()
        if not _is_hex64(expected_pool_sha) or _file_sha256(p) != expected_pool_sha:
            raise RuntimeError(f"fresh row ({ds_id}) pool_sha256 mismatch for {p}")
        with np.load(p, allow_pickle=False) as z:
            npz_data = {k: z[k].copy() for k in z.files}

    def _resolve_array(field_name, npz_key, dtype):
        inline_val = row.get(field_name)
        npz_val = npz_data.get(npz_key) if npz_data is not None else None
        if inline_val is None and npz_val is None:
            raise RuntimeError(
                f"fresh row ({ds_id}) missing required payload array {field_name!r}"
            )
        if inline_val is not None and npz_val is not None:
            a_in = np.asarray(inline_val, dtype=dtype)
            a_npz = np.asarray(npz_val, dtype=dtype)
            if a_in.shape != a_npz.shape or a_in.tobytes() != a_npz.tobytes():
                raise RuntimeError(
                    f"fresh row ({ds_id}) mismatch between receipt/row {field_name!r} and loaded NPZ {npz_key!r}"
                )
            return a_npz
        return np.asarray(
            inline_val if inline_val is not None else npz_val, dtype=dtype
        )

    fitness = _resolve_array("fitness", "fitness", np.float64)
    gate_arr = _resolve_array("gate", "gate", bool)
    program = _resolve_array("program", "program", np.float32)
    tokens_10x32 = _resolve_array("tokens_10x32", "tokens_10x32", np.float32)
    mask_10 = _resolve_array("mask_10", "mask_10", bool)

    n_cand = fitness.shape[0] if fitness.ndim == 1 else -1
    if (
        n_cand < SLATE_CANDIDATES
        or gate_arr.shape != (n_cand,)
        or not np.isfinite(fitness).all()
        or not bool(gate_arr.any())
    ):
        raise RuntimeError(f"fresh row ({ds_id}) invalid native fitness/gate arrays")
    if (
        program.ndim != 2
        or program.shape != (n_cand, PROGRAM_DIM)
        or not np.isfinite(program).all()
    ):
        raise RuntimeError(
            f"fresh row ({ds_id}) invalid program array shape {program.shape}; expected ({n_cand}, {PROGRAM_DIM})"
        )
    if (
        tokens_10x32.shape != (TOKEN_STEPS, TOKEN_DIM)
        or not np.isfinite(tokens_10x32).all()
        or mask_10.shape != (TOKEN_STEPS,)
        or not bool(mask_10.any())
    ):
        raise RuntimeError(f"fresh row ({ds_id}) invalid tokens_10x32 / mask_10 arrays")

    expected_label_sha = hashlib.sha256(
        fitness.tobytes() + gate_arr.tobytes()
    ).hexdigest()
    if receipt.get("labels_sha256") != expected_label_sha:
        raise RuntimeError(
            f"fresh row ({ds_id}) native labels_sha256 mismatch against fitness/gate payload"
        )

    expected_prog_sha = hashlib.sha256(program.tobytes()).hexdigest()
    rec_prog_sha = receipt.get("programs_sha256")
    if not _is_hex64(rec_prog_sha) or rec_prog_sha != expected_prog_sha:
        raise RuntimeError(
            f"fresh row ({ds_id}) native programs_sha256 mismatch against candidate program payload"
        )

    expected_feat_sha = hashlib.sha256(
        tokens_10x32.tobytes() + mask_10.tobytes()
    ).hexdigest()
    rec_feat_sha = receipt.get("features_sha256")
    if not _is_hex64(rec_feat_sha) or rec_feat_sha != expected_feat_sha:
        raise RuntimeError(
            f"fresh row ({ds_id}) native features_sha256 mismatch against tokens_10x32/mask_10 payload"
        )

    return dict(
        context_sha256=cid,
        dataset_id=ds_id,
        program=program,
        fitness=fitness,
        gate=gate_arr,
        tokens_10x32=tokens_10x32,
        mask_10=mask_10,
        labels_sha256=expected_label_sha,
        programs_sha256=expected_prog_sha,
        features_sha256=expected_feat_sha,
    )


def verify_and_load_fresh_d4_pool(
    receipt_path,
    static_train_rows,
    eval_rows=(),
    sealed_family_keys=(),
    fresh_rows_override=None,
):
    """Validate JRB-01 admission receipt plus admitted fresh D4 regression rows & native-Rust-CV label receipts.

    Enforces:
    1. Valid JRB-01 ADMISSION PASS receipt.
    2. Every fresh row has task=='regression', valid dataset_id, source_split=='train',
       dl_split=='DL-Train', has_meta_test==False, 64-hex context_sha256 and train_sha256,
       and a verified native-Rust-CV label receipt (`bootstrap.s1main --fit-s1`, status=='PASS',
       64-hex evaluator_binary_sha256, labels_sha256, programs_sha256, and features_sha256 matching
       the row/NPZ `fitness`, `gate`, `program`, `tokens_10x32`, and `mask_10` arrays).
    3. Zero family/alias or context_sha256 overlap with the static 205-family roster (`static_train_rows`).
       Fake-fresh rows drawn from the old 205 families are unconditionally rejected.
    4. Zero family/alias or context_sha256 overlap with any evaluation (`B_k`, `A_k`, `DL-Val`) or sealed family.
    """
    gate = verify_fresh_admission_gate(receipt_path)
    if fresh_rows_override is not None:
        fresh_rows = list(fresh_rows_override)
    else:
        manifest_rel = gate.get("fresh_pool_manifest")
        if not manifest_rel:
            raise RuntimeError(
                "JRB-01 admission receipt missing fresh_pool_manifest path for fresh D4 regression rows"
            )
        manifest_path = Path(manifest_rel)
        if not manifest_path.is_file():
            raise RuntimeError(f"fresh D4 pool manifest missing: {manifest_path}")
        if (
            hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            != gate["manifest_sha256"]
        ):
            raise RuntimeError(
                "fresh D4 pool manifest SHA256 does not match receipt manifest_sha256"
            )
        raw_manifest = json.loads(manifest_path.read_text())
        fresh_rows = list(raw_manifest.get("records", ()))

    if not fresh_rows:
        raise RuntimeError("fresh D4 pool contains zero rows")

    static_fams = set()
    static_cids = set()
    for r in static_train_rows:
        static_fams |= _row_family_aliases(r)
        if r.get("context_sha256"):
            static_cids.add(r["context_sha256"])

    forbidden_eval_aliases = {
        str(k).strip() for k in sealed_family_keys if str(k).strip()
    }
    forbidden_eval_cids = set()
    for r in eval_rows:
        forbidden_eval_aliases |= _row_family_aliases(r)
        if r.get("context_sha256"):
            forbidden_eval_cids.add(r["context_sha256"])

    validated_rows = []
    seen_cids = set()
    for idx, row in enumerate(fresh_rows):
        if not isinstance(row, dict):
            raise RuntimeError(f"fresh row {idx} is not a dict")
        if row.get("task") != "regression":
            raise RuntimeError(
                f"fresh row {idx} invalid task {row.get('task')!r}; regression required"
            )
        ds_id = str(row.get("dataset_id") or row.get("dataset") or "").strip()
        if not ds_id:
            raise RuntimeError(f"fresh row {idx} missing dataset_id")
        if (
            row.get("source_split") != "train"
            or row.get("dl_split") != "DL-Train"
            or bool(row.get("has_meta_test", False))
        ):
            raise RuntimeError(
                f"fresh row {idx} ({ds_id}) invalid cut/split or has_meta_test=True"
            )
        cid = str(row.get("context_sha256", "")).strip()
        train_sha = str(row.get("train_sha256", "")).strip()
        if not _is_hex64(cid) or not _is_hex64(train_sha):
            raise RuntimeError(
                f"fresh row {idx} ({ds_id}) missing 64-hex context_sha256 or train_sha256"
            )
        if cid in seen_cids:
            raise RuntimeError(f"duplicate fresh context_sha256 {cid}")
        seen_cids.add(cid)

        fam_id = str(row.get("family_id") or row.get("family") or "").strip()
        if not fam_id:
            raise RuntimeError(f"fresh row {idx} ({ds_id}) missing family_id")
        aliases = _row_family_aliases(row)

        if cid in static_cids or (aliases & static_fams):
            raise RuntimeError(
                f"fake-fresh violation: row {ds_id} ({fam_id}) overlaps static 205-family roster"
            )
        if cid in forbidden_eval_cids or (aliases & forbidden_eval_aliases):
            raise RuntimeError(
                f"leakage violation: evaluation or sealed family/alias {aliases & forbidden_eval_aliases} cannot enter training"
            )

        receipt = row.get("native_evaluator_receipt")
        if not isinstance(receipt, dict):
            raise RuntimeError(
                f"fresh row {idx} ({ds_id}) missing native_evaluator_receipt"
            )
        if receipt.get("status") != "PASS":
            raise RuntimeError(
                f"fresh row {idx} ({ds_id}) native_evaluator_receipt status is not PASS"
            )
        if receipt.get("evaluator") != NATIVE_RUST_EVALUATOR_ID:
            raise RuntimeError(
                f"fresh row {idx} ({ds_id}) invalid native evaluator {receipt.get('evaluator')!r}"
            )
        if not _is_hex64(receipt.get("evaluator_binary_sha256", "")) or not _is_hex64(
            receipt.get("labels_sha256", "")
        ):
            raise RuntimeError(
                f"fresh row {idx} ({ds_id}) native_evaluator_receipt missing 64-hex binary/labels sha256"
            )
        if receipt.get("context_sha256") != cid or receipt.get("dataset_id") != ds_id:
            raise RuntimeError(
                f"fresh row {idx} ({ds_id}) native_evaluator_receipt dataset/context mismatch"
            )

        verified_payload = extract_and_verify_fresh_row_payload(row)
        enriched = dict(row)
        enriched["_verified_payload"] = verified_payload
        validated_rows.append(enriched)

    distinct_fresh_fams = {r["family_id"] for r in validated_rows}
    if len(distinct_fresh_fams) < FAMILIES_PER_MICROBATCH * 2:
        raise RuntimeError(
            f"fresh D4 pool requires at least {FAMILIES_PER_MICROBATCH * 2} distinct families, got {len(distinct_fresh_fams)}"
        )

    return dict(
        status="PASS",
        admitted_rows=len(validated_rows),
        admitted_families=len(distinct_fresh_fams),
        manifest_sha256=gate["manifest_sha256"],
        records=validated_rows,
    )


def verify_frozen_teacher_identity(
    teacher_checkpoint_sha256,
    dope_sha256,
    initial_param_sha256,
    current_param_sha256,
    teacher_training_mode=False,
    any_requires_grad=False,
):
    """Verify that the frozen r882 teacher actor/parameters never mutate or enter training mode."""
    if teacher_checkpoint_sha256 != R882_SHA256:
        raise RuntimeError("frozen-teacher identity violation: r882 SHA256 mismatch")
    if dope_sha256 != DOPE_SHA256:
        raise RuntimeError("frozen-teacher identity violation: DOPE SHA256 mismatch")
    if not _is_hex64(initial_param_sha256) or not _is_hex64(current_param_sha256):
        raise RuntimeError(
            "frozen-teacher identity violation: invalid parameter digest"
        )
    if initial_param_sha256 != current_param_sha256:
        raise RuntimeError(
            "frozen-teacher identity violation: teacher parameter digest mutated"
        )
    if bool(teacher_training_mode):
        raise RuntimeError(
            "frozen-teacher identity violation: teacher model is in training mode"
        )
    if bool(any_requires_grad):
        raise RuntimeError(
            "frozen-teacher identity violation: teacher parameter has requires_grad=True"
        )
    return dict(
        status="PASS",
        teacher_checkpoint_sha256=teacher_checkpoint_sha256,
        dope_sha256=dope_sha256,
        param_sha256=current_param_sha256,
        immutable=True,
    )


def admit_round_checkpoint_pooled(
    rows, incumbent_values, candidate_panel, seed=BOOT_SEED
):
    """Per-round single-arm pooled 205-family paired bootstrap checkpoint selector.

    Evaluates a candidate checkpoint across all 3 seeds x 205 pooled B families (10,000 draws)
    BEFORE checkpoint retention. A positive mean gain with 95% CI crossing 0 (`lcb95 <= 0`)
    or excess family harm (`harm_rate > 0.10`) is strictly rejected.
    """
    families = [r["family_id"] for r in rows]
    fams_sorted = sorted(set(families))
    n_fams = len(fams_sorted)
    if n_fams != DLTRAIN_FAMILY_COUNT:
        raise ValueError(
            f"pooled checkpoint selector requires 205 families, got {n_fams}"
        )

    base = np.asarray(incumbent_values, dtype=np.float64)
    if base.ndim == 1:
        if base.shape != (len(rows),):
            raise ValueError("1D incumbent_values must match len(rows)")
        base_3 = np.broadcast_to(base[None, :], (3, len(rows)))
    elif base.ndim == 2 and base.shape == (3, len(rows)):
        base_3 = base
    else:
        raise ValueError("incumbent_values must have shape (N_ctx,) or (3, N_ctx)")

    cand = np.asarray(candidate_panel, dtype=np.float64)
    if cand.ndim != 2 or cand.shape != (3, len(rows)):
        raise ValueError(f"candidate_panel must have shape (3, {len(rows)})")
    if not np.isfinite(base_3).all() or not np.isfinite(cand).all():
        raise ValueError("nonfinite values in pooled checkpoint selector")

    fm_base = np.stack([h177_analysis.family_matrix(v, families)[1] for v in base_3])
    fm_cand = np.stack([h177_analysis.family_matrix(v, families)[1] for v in cand])
    diff_3 = fm_cand - fm_base
    fam_diff = diff_3.mean(axis=0)  # (205,)
    gain = float(fam_diff.mean())
    ix = np.random.default_rng(seed).integers(0, n_fams, (DRAWS, n_fams))
    boot_means = fam_diff[ix].mean(axis=1)
    lcb95 = float(np.quantile(boot_means, 0.025))
    ucb95 = float(np.quantile(boot_means, 0.975))
    p_raw = float((np.sum(boot_means <= MIN_GAIN_MARGIN) + 1) / (DRAWS + 1))
    harm_count = int(np.sum(fam_diff < -HARM_EPS))
    harm_rate = float(harm_count / n_fams)

    admit = bool(
        gain > MIN_GAIN_MARGIN + EPS
        and lcb95 > 0.0
        and harm_rate <= MAX_HARM_FRACTION + EPS
        and p_raw < 0.05
    )
    return dict(
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
        admit=admit,
        decision="ADMIT" if admit else "REJECT_RETAIN_INCUMBENT",
    )


def h178_round_select(rows, incumbent_values, candidate_panels_by_arm, seed=BOOT_SEED):
    """Per-round H178 calibrated 205-family pooled cross-fit selector across all 3 arms.

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

    raw_decisions = {}
    pvals = {}
    for arm in ARMS:
        single = admit_round_checkpoint_pooled(
            rows, incumbent_values, candidate_panels_by_arm[arm], seed=seed
        )
        pvals[arm] = single["p_raw"]
        raw_decisions[arm] = dict(arm=arm, **single)

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
        pooled_b_families=DLTRAIN_FAMILY_COUNT,
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


# ---------------------------------------------------------------------------
# Disjoint Regression-Only Data Construction Seam (H205 Owner Scope)
# ---------------------------------------------------------------------------

S3_DATASET_CATALOG_PATH = Path(
    "/home/ubuntu/jope_plan/master/source/S3_DATASET_CATALOG.json"
)
S3_DATASET_CATALOG_SHA256 = (
    "e5f732fd784bf67bfc16f4173c585b0861e6e8a6026808df857097a721bbf0ba"
)
FAMILY_KEY_POOL_MANIFEST_PATH = Path(
    "/home/ubuntu/jope_plan/evidence/JRB-01/tel-t1510/family-key-pool-manifest.json"
)
FAMILY_KEY_POOL_MANIFEST_SHA256 = (
    "e42c31d3fda53545900c0064f2685f78506149fdad49251de6994d7f030124f4"
)
JRB01_SEALED_EXCLUSION_PATH = Path(
    "/home/ubuntu/jope_plan/master/approved/JRB01_SEALED_FAMILY_EXCLUSION.json"
)
JRB01_SEALED_EXCLUSION_SHA256 = (
    "ca0eff866a87548803feb03e50f8e7e99b74993b2e1d0b8ce56e3ec1b184a651"
)
H150_SEALED_FAMILY_MANIFEST_PATH = (
    ROOT
    / "research/flywheel_selfplay/h150_w5_provenance_unblock/sealed_family_manifest.json"
)
H150_SEALED_FAMILY_MANIFEST_SHA256 = (
    "0314599fe06515d3f3e976c0d435cbe4e9fbe26738014bd89470f525702f983b"
)
D4_ROLE_CANDIDATE_MANIFEST_PATH = Path(
    "/home/ubuntu/jope_plan/evidence/JRB-01/tel-t1510/d4-role-candidate-manifest.json"
)
D4_ROLE_CANDIDATE_MANIFEST_SHA256 = (
    "4a96c3952cb3df19934d342bfe0fdf546bb19de1c56f7acabc253fd68bff4ac0"
)
SELECTED_JSON_PATH = Path("/home/ubuntu/jope-data/h50/input/selected.json")
SELECTED_JSON_SHA256 = (
    "981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1"
)
HISTORICAL_COHORT_JSON_PATH = Path(
    "/home/ubuntu/jope-evidence/dope-lift-20261021/cohort.json"
)
HISTORICAL_COHORT_JSON_SHA256 = (
    "329c8cd92c13a67185dc3eacc6b02c59d92be06189035b7f9400d5d35a6e2d94"
)
HISTORICAL_RANKING_IDENTITIES_PATH = Path(
    "/home/ubuntu/jope-evidence/dope-lift-20261021/corrected-study/ranking-identities.jsonl.gz"
)
HISTORICAL_RANKING_IDENTITIES_SHA256 = (
    "d4a63f698d00706439bf95327a86909c6b0c3f9938936bc362e54c3e31d21eab"
)
EVALUATOR_BINARY_PATH = Path(
    "/home/ubuntu/scratch/jopedime-dl-training-v1/receipts/trainonly-v2/s1-prod/binaries/bootstrap.s1main"
)
EVALUATOR_BINARY_SHA256 = (
    "e3be938644722d14a1c1a0e82168518838c4d174cfe031076a339745c48870a8"
)
PROGRAM_FEATURE_NAMES_PATH = Path(
    "/home/ubuntu/jope-evidence/dope-lift-20261021/corrected-study/program-feature-names.json"
)
PROGRAM_FEATURE_NAMES_SHA256 = (
    "c4e4373d1bc7e99bc38e44eb4acda22d0f9ecc9e332fa8cb9c6a0b463fe37962"
)

# Pinned target dataset & context for the single-context regression seam milestone
TARGET_REGRESSION_DATASET_ID = "1db3750d4ceccab9"
TARGET_REGRESSION_FAMILY_ID = (
    "9a9933b47cabdfdbef3c7d567074e553aaf3dfd92c9a71e3365c777d025452c0"
)
TARGET_REGRESSION_FAMILY_KEY = "e70bef3cf802dadb1f22458c1e1c6bb553a5d9f3"
TARGET_REGRESSION_TRAIN_SHA256 = (
    "6ee10fbb69512880590388c7f608961eaa84532343b45372b7f5b171164ca39d"
)
TARGET_REGRESSION_CONTEXT_SHA256 = (
    "fa480de30f3fba6fda1d7e9a1df65019acab3de752ded46b96448d0848f86d45"
)
TARGET_REGRESSION_VIEW_ID = "1db3750d4ceccab9__r0__f0__rowselraw1_84a908fcde0e"
TARGET_REGRESSION_S1_INPUT_SHA256 = (
    "eafffab9dd483f2f132ad665bee0549a9218ea8e96481db2cc88ac3dac2617bf"
)
TARGET_REGRESSION_TRAINING_BLOCK_SHA256 = (
    "2ad53cd8323023aef82cdab74d62e69582747fe58f45ff9a32f1ad7450979cc8"
)
TARGET_REGRESSION_TEACHER_GZ_SHA256 = (
    "b0bc4a216499b0ae876600f265aeccbb6de02deb10fce5f543f91eb4416808f3"
)
TARGET_REGRESSION_TEACHER_RAW_SHA256 = (
    "fb4709fd6ed8feee0f7d28e98dd9914e36aee7104ea0aab1a07875b90009f231"
)
TARGET_REGRESSION_SUMMARY_TAR_GZ_SHA256 = (
    "8eceac827857a53de1f30041495756767708b1e2e31a5c6667790c6e93b0db7c"
)


def _normalize_nonempty_key(val, label="family_key"):
    if val is None or not isinstance(val, str) or not val.strip():
        raise RuntimeError(f"missing or null {label} refused")
    return val.strip()


def build_allowed_regression_train_roster(
    catalog_path=S3_DATASET_CATALOG_PATH,
    family_pool_path=FAMILY_KEY_POOL_MANIFEST_PATH,
    jrb01_sealed_path=JRB01_SEALED_EXCLUSION_PATH,
    h150_sealed_path=H150_SEALED_FAMILY_MANIFEST_PATH,
    d4_roles_path=D4_ROLE_CANDIDATE_MANIFEST_PATH,
    selected_path=SELECTED_JSON_PATH,
    cohort_path=HISTORICAL_COHORT_JSON_PATH,
    ranking_identities_path=HISTORICAL_RANKING_IDENTITIES_PATH,
    *,
    verify_hashes=True,
    caller_excluded_families=None,
    caller_excluded_datasets=None,
    catalog_override=None,
    family_by_dataset_override=None,
):
    """Build the deterministic source-bound allowed regression TRAIN roster for H205.

    Fail-closed contract:
    1. If a caller supplies `caller_excluded_families` or `caller_excluded_datasets` and it is empty,
       raises `RuntimeError("empty exclusion set refused")`.
    2. Verifies pinned SHA256 digests of the catalog, family-key pool, JRB-01 sealed exclusion,
       H150 sealed family manifest, D4 role manifest, and historical cohorts (`selected.json`,
       `cohort.json`, `ranking-identities.jsonl.gz`) when `verify_hashes=True`.
    3. Excludes the union of all sealed/protected families & datasets, all D4 role-manifest datasets
       and their families (`panel`, `regression_val`, `binary_train`, `regression_train`), and all
       historical contexts, datasets, and families (`selected.json`, `cohort.json`,
       `ranking-identities.jsonl.gz`).
    4. Refuses any candidate with `None`/empty `family_key` or `family_id`, non-regression task,
       or missing `train_sha256`.
    """
    if caller_excluded_families is not None and len(caller_excluded_families) == 0:
        raise RuntimeError(
            "empty exclusion set refused: caller_excluded_families is empty"
        )
    if caller_excluded_datasets is not None and len(caller_excluded_datasets) == 0:
        raise RuntimeError(
            "empty exclusion set refused: caller_excluded_datasets is empty"
        )

    if verify_hashes and catalog_override is None:
        pinned_checks = [
            (catalog_path, S3_DATASET_CATALOG_SHA256, "S3_DATASET_CATALOG.json"),
            (
                family_pool_path,
                FAMILY_KEY_POOL_MANIFEST_SHA256,
                "family-key-pool-manifest.json",
            ),
            (
                jrb01_sealed_path,
                JRB01_SEALED_EXCLUSION_SHA256,
                "JRB01_SEALED_FAMILY_EXCLUSION.json",
            ),
            (
                h150_sealed_path,
                H150_SEALED_FAMILY_MANIFEST_SHA256,
                "sealed_family_manifest.json",
            ),
            (
                d4_roles_path,
                D4_ROLE_CANDIDATE_MANIFEST_SHA256,
                "d4-role-candidate-manifest.json",
            ),
            (selected_path, SELECTED_JSON_SHA256, "selected.json"),
            (cohort_path, HISTORICAL_COHORT_JSON_SHA256, "cohort.json"),
            (
                ranking_identities_path,
                HISTORICAL_RANKING_IDENTITIES_SHA256,
                "ranking-identities.jsonl.gz",
            ),
        ]
        for p, expected_sha, label in pinned_checks:
            if not Path(p).is_file():
                raise RuntimeError(f"required roster input missing: {label} ({p})")
            actual_sha = _file_sha256(p)
            if actual_sha != expected_sha:
                raise RuntimeError(
                    f"pinned SHA256 mismatch for {label}: {actual_sha} != {expected_sha}"
                )

    if catalog_override is not None:
        cat_datasets = list(catalog_override)
    else:
        cat_obj = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
        cat_datasets = list(cat_obj.get("datasets", ()))

    if family_by_dataset_override is not None:
        family_by_dataset = dict(family_by_dataset_override)
    else:
        fpool_obj = json.loads(Path(family_pool_path).read_text(encoding="utf-8"))
        family_by_dataset = dict(fpool_obj.get("family_by_dataset", {}))

    forbidden_families = set()
    forbidden_datasets = set()
    forbidden_cids = set()

    if caller_excluded_families is not None:
        for fk in caller_excluded_families:
            forbidden_families.add(
                _normalize_nonempty_key(fk, "caller_excluded_family")
            )
    if caller_excluded_datasets is not None:
        for ds in caller_excluded_datasets:
            forbidden_datasets.add(
                _normalize_nonempty_key(ds, "caller_excluded_dataset")
            )

    if catalog_override is None:
        jrb_sealed = json.loads(Path(jrb01_sealed_path).read_text(encoding="utf-8"))
        for fk in jrb_sealed.get("excluded_family_keys", ()):
            forbidden_families.add(
                _normalize_nonempty_key(fk, "jrb01_excluded_family_key")
            )
        for ds in jrb_sealed.get("excluded_dataset_ids", ()):
            ds_norm = _normalize_nonempty_key(ds, "jrb01_excluded_dataset_id")
            forbidden_datasets.add(ds_norm)
            if ds_norm in family_by_dataset and family_by_dataset[ds_norm]:
                forbidden_families.add(str(family_by_dataset[ds_norm]).strip())

        h150_sealed = json.loads(Path(h150_sealed_path).read_text(encoding="utf-8"))
        for fk in h150_sealed.get("protected_family_ids", ()):
            forbidden_families.add(
                _normalize_nonempty_key(fk, "h150_protected_family_id")
            )
        for key_list in ("sealed_dataset_ids", "audit_holdout_dataset_ids"):
            for ds in h150_sealed.get(key_list, ()):
                ds_norm = _normalize_nonempty_key(ds, key_list)
                forbidden_datasets.add(ds_norm)
                if ds_norm in family_by_dataset and family_by_dataset[ds_norm]:
                    forbidden_families.add(str(family_by_dataset[ds_norm]).strip())

        d4_roles = json.loads(Path(d4_roles_path).read_text(encoding="utf-8"))
        for role_name, entries in d4_roles.get("roles", {}).items():
            for entry in entries:
                ds_id = str(entry.get("dataset_id", "")).strip()
                fk = str(entry.get("family_key", "")).strip()
                if ds_id:
                    forbidden_datasets.add(ds_id)
                    if ds_id in family_by_dataset and family_by_dataset[ds_id]:
                        forbidden_families.add(str(family_by_dataset[ds_id]).strip())
                if fk:
                    forbidden_families.add(fk)

        selected_obj = json.loads(Path(selected_path).read_text(encoding="utf-8"))
        for split_key in ("train", "dev"):
            for row in selected_obj.get(split_key, ()):
                cid = str(row.get("context_sha256", "")).strip()
                if cid:
                    forbidden_cids.add(cid)
                ds_h = str(
                    row.get("dataset_hash") or row.get("dataset_id") or ""
                ).strip()
                if ds_h:
                    forbidden_datasets.add(ds_h)
                    if ds_h in family_by_dataset and family_by_dataset[ds_h]:
                        forbidden_families.add(str(family_by_dataset[ds_h]).strip())
                forbidden_families |= _row_family_aliases(row)

        cohort_rows = json.loads(Path(cohort_path).read_text(encoding="utf-8"))
        for row in cohort_rows:
            cid = str(row.get("context_sha256", "")).strip()
            if cid:
                forbidden_cids.add(cid)
            ds_h = str(row.get("dataset_hash") or row.get("dataset_id") or "").strip()
            if ds_h:
                forbidden_datasets.add(ds_h)
                if ds_h in family_by_dataset and family_by_dataset[ds_h]:
                    forbidden_families.add(str(family_by_dataset[ds_h]).strip())
            forbidden_families |= _row_family_aliases(row)

        with gzip.open(ranking_identities_path, "rt", encoding="utf-8") as gz_in:
            for line in gz_in:
                if not line.strip():
                    continue
                r = json.loads(line)
                cid = str(r.get("context_sha256", "")).strip()
                if cid:
                    forbidden_cids.add(cid)

    if not forbidden_families or not forbidden_datasets:
        raise RuntimeError(
            "empty exclusion set refused: computed forbidden closure is empty"
        )

    # Propagate catalog-level family aliases for any forbidden dataset
    for d in cat_datasets:
        ds_id = str(d.get("id", "")).strip()
        fk = d.get("family_key")
        fpool_fk = family_by_dataset.get(ds_id)
        if ds_id in forbidden_datasets:
            if isinstance(fk, str) and fk.strip():
                forbidden_families.add(fk.strip())
            if isinstance(fpool_fk, str) and fpool_fk.strip():
                forbidden_families.add(fpool_fk.strip())

    allowed_records = []
    skipped_missing_family = 0
    for d in cat_datasets:
        if d.get("task") != "regression":
            continue
        ds_id = str(d.get("id", "")).strip()
        if not ds_id:
            raise RuntimeError("catalog regression entry missing dataset id")
        raw_fk = d.get("family_key")
        pool_fk = family_by_dataset.get(ds_id)
        if raw_fk is None or not isinstance(raw_fk, str) or not raw_fk.strip():
            skipped_missing_family += 1
            continue
        if pool_fk is None or not isinstance(pool_fk, str) or not pool_fk.strip():
            skipped_missing_family += 1
            continue
        raw_fk = raw_fk.strip()
        pool_fk = pool_fk.strip()
        aliases = {raw_fk, pool_fk, f"dedupe:{raw_fk}", f"dedupe:{pool_fk}"}
        if ds_id in forbidden_datasets or (aliases & forbidden_families):
            continue
        train_sha = str(d.get("train_sha256", "")).strip()
        train_rows = int(d.get("train_rows", 0))
        features = int(d.get("features", 0))
        if not _is_hex64(train_sha) or train_rows <= 0 or features <= 0:
            continue
        allowed_records.append(
            dict(
                dataset_id=ds_id,
                name=str(d.get("name", "")),
                task="regression",
                role="regression_train",
                source_split="train",
                dl_split="DL-Train",
                has_meta_test=False,
                family_id=pool_fk,
                family_key=raw_fk,
                family_aliases=sorted(aliases),
                train_rows=train_rows,
                features=features,
                categorical_features=int(d.get("categorical_features", 0)),
                train_sha256=train_sha,
                s3_bucket=str(d.get("s3_bucket", "veox-jopedime-datasets-use2")),
            )
        )

    allowed_records.sort(key=lambda r: (r["family_id"], r["dataset_id"]))
    roster_bytes = json.dumps(allowed_records, sort_keys=True).encode("utf-8")
    roster_sha256 = hashlib.sha256(roster_bytes).hexdigest()
    distinct_fams = {r["family_id"] for r in allowed_records}

    return dict(
        status="PASS",
        wave=WAVE,
        allowed_datasets_count=len(allowed_records),
        allowed_families_count=len(distinct_fams),
        skipped_missing_family_count=skipped_missing_family,
        forbidden_datasets_count=len(forbidden_datasets),
        forbidden_family_aliases_count=len(forbidden_families),
        forbidden_cids_count=len(forbidden_cids),
        roster_sha256=roster_sha256,
        forbidden_datasets=forbidden_datasets,
        forbidden_families=forbidden_families,
        forbidden_cids=forbidden_cids,
        records=allowed_records,
    )


def verify_single_regression_train_candidate(candidate_row, roster_bundle):
    """Fail-closed check that a single candidate dataset/context row is admitted for H205 regression TRAIN."""
    if not isinstance(candidate_row, dict):
        raise RuntimeError("candidate_row must be a dict")
    if not isinstance(roster_bundle, dict) or roster_bundle.get("status") != "PASS":
        raise RuntimeError("roster_bundle must be a PASS allowed-roster dict")

    forbidden_fams = roster_bundle.get("forbidden_families")
    forbidden_ds = roster_bundle.get("forbidden_datasets")
    forbidden_cids = roster_bundle.get("forbidden_cids")
    if not forbidden_fams or not forbidden_ds:
        raise RuntimeError(
            "empty exclusion set refused in verify_single_regression_train_candidate"
        )

    fam_id = _normalize_nonempty_key(
        candidate_row.get("family_id") or candidate_row.get("family"), "family_id"
    )
    fam_key = _normalize_nonempty_key(
        candidate_row.get("family_key") or fam_id, "family_key"
    )
    ds_id = _normalize_nonempty_key(
        candidate_row.get("dataset_id") or candidate_row.get("dataset"), "dataset_id"
    )
    if candidate_row.get("task") != "regression":
        raise RuntimeError(
            f"non-regression task refused for {ds_id}: {candidate_row.get('task')!r}"
        )
    role = str(candidate_row.get("role", "regression_train")).strip()
    if (
        role != "regression_train"
        or candidate_row.get("source_split") != "train"
        or candidate_row.get("dl_split") != "DL-Train"
        or bool(candidate_row.get("has_meta_test", False))
    ):
        raise RuntimeError(
            f"invalid role or split cut refused for {ds_id}: role={role!r}"
        )

    cid = str(candidate_row.get("context_sha256", "")).strip()
    if cid and not _is_hex64(cid):
        raise RuntimeError(f"invalid context_sha256 for {ds_id}: {cid!r}")
    if cid and forbidden_cids and cid in forbidden_cids:
        raise RuntimeError(
            f"forbidden context_sha256 overlap refused: {cid} is in historical cohort"
        )

    aliases = _row_family_aliases(candidate_row) | {
        fam_id,
        fam_key,
        f"dedupe:{fam_id}",
        f"dedupe:{fam_key}",
    }
    if ds_id in forbidden_ds or (aliases & set(forbidden_fams)):
        raise RuntimeError(
            f"forbidden family/dataset overlap refused for {ds_id} ({fam_id})"
        )

    allowed_by_ds = {r["dataset_id"]: r for r in roster_bundle.get("records", ())}
    if ds_id not in allowed_by_ds:
        raise RuntimeError(
            f"dataset {ds_id} is not present in the allowed regression TRAIN roster"
        )
    return dict(
        status="PASS",
        dataset_id=ds_id,
        family_id=fam_id,
        family_key=fam_key,
        context_sha256=cid,
        roster_sha256=roster_bundle["roster_sha256"],
    )


def encode_train_only_jtt1_from_csv(
    csv_path, out_jtt1_path=None, expected_csv_sha256=None
):
    """Encode a verified regression TRAIN CSV into a pure train-only JTT1 (`nq = 0`, `k = 2`).

    Never reads or invents any test/query rows (`nq == 0`, `outer_test_rows_decoded == 0`)
    and never re-cuts `train.csv` into an arbitrary pseudo-test split.
    """
    p = Path(csv_path)
    if not p.is_file():
        raise RuntimeError(f"train CSV missing: {p}")
    actual_sha = _file_sha256(p)
    if expected_csv_sha256 is not None and actual_sha != expected_csv_sha256:
        raise RuntimeError(
            f"train CSV SHA256 mismatch: {actual_sha} != {expected_csv_sha256}"
        )

    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header or len(header) < 2:
            raise RuntimeError(f"invalid train CSV header in {p}")
        rows = []
        for line_idx, row in enumerate(reader):
            if not row:
                continue
            if len(row) != len(header):
                raise RuntimeError(
                    f"row {line_idx} column count mismatch in {p}: {len(row)} != {len(header)}"
                )
            rows.append([float(v) if v != "" else float("nan") for v in row])

    mat = np.asarray(rows, dtype="<f8")
    if mat.ndim != 2 or mat.shape[0] < 4 or mat.shape[1] < 2:
        raise RuntimeError(f"invalid train CSV matrix shape {mat.shape} in {p}")
    nt = int(mat.shape[0])
    width = int(mat.shape[1] - 1)
    y = mat[:, -1]
    if not np.isfinite(y).all():
        raise RuntimeError(f"non-finite target values in train CSV {p}")

    # Deterministic 2-fold inner CV assignment sorted by target (alternating parity)
    order = np.argsort(y, kind="mergesort")
    folds = np.zeros(nt, dtype=np.uint8)
    folds[order[1::2]] = 1

    header_bytes = struct.pack("<4sIIIB", b"JTT1", nt, 0, width, 2)
    payload_bytes = (
        header_bytes
        + folds.tobytes()
        + np.ascontiguousarray(mat, dtype="<f8").tobytes()
    )
    jtt1_sha = hashlib.sha256(payload_bytes).hexdigest()

    if out_jtt1_path is not None:
        out_p = Path(out_jtt1_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_bytes(payload_bytes)

    return dict(
        status="PASS",
        csv_sha256=actual_sha,
        jtt1_sha256=jtt1_sha,
        jtt1_bytes_len=len(payload_bytes),
        nt=nt,
        nq=0,
        p=width,
        k=2,
        fold_sizes=[int(np.sum(folds == 0)), int(np.sum(folds == 1))],
        outer_test_rows_decoded=0,
        X_train=mat[:, :-1].astype(np.float32),
        y_train=y.astype(np.float32),
    )


def probe_s1main_nq0_protocol_blocker(
    jtt1_path,
    worker_bin=EVALUATOR_BINARY_PATH,
    expected_worker_sha256=EVALUATOR_BINARY_SHA256,
):
    """Probe `bootstrap.s1main --fit-s1` on an `nq == 0` train-only JTT1 table and record the exact binary response.

    Pins the exact protocol mismatch where `bootstrap.s1main --fit-s1` (`e3be9386...`) checks `nq >= 10`
    in the 17-byte JTT1 header (`deploy/lambda/training/teacher-v2/rust/lambda-worker/src/main.rs:567`)
    even though `parse_s1_data` at `main.rs:587-606` only loops `0..nt` and never reads a single byte
    of the `nq` query block.
    """
    jtt1_p = Path(jtt1_path)
    bin_p = Path(worker_bin)
    if not jtt1_p.is_file():
        raise RuntimeError(f"JTT1 file missing for nq=0 probe: {jtt1_p}")
    if not bin_p.is_file():
        raise RuntimeError(f"worker binary missing: {bin_p}")
    bin_sha = _file_sha256(bin_p)
    if expected_worker_sha256 is not None and bin_sha != expected_worker_sha256:
        raise RuntimeError(
            f"worker binary SHA256 mismatch: {bin_sha} != {expected_worker_sha256}"
        )

    raw_head = jtt1_p.read_bytes()[:17]
    if len(raw_head) < 17 or raw_head[:4] != b"JTT1":
        raise RuntimeError(f"invalid JTT1 header in {jtt1_p}")
    nt, nq, p_dim, k_folds = struct.unpack("<IIIB", raw_head[4:17])

    dsl = "(pipe X (standardize) (ridge_gcv))"
    cand_obj = dict(
        dsl=dsl,
        program_sha256=hashlib.sha256(dsl.encode("utf-8")).hexdigest(),
        birth_seq=0,
        parents=[],
        operator="probe_nq0",
    )
    with tempfile.TemporaryDirectory(prefix="h205_nq0_probe_") as td:
        td_path = Path(td)
        cand_path = td_path / "candidate.json"
        out_path = td_path / "out.json"
        cand_path.write_text(json.dumps(cand_obj, sort_keys=True), encoding="utf-8")
        env = {
            "PATH": "/usr/bin:/bin",
            "RAYON_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "TMPDIR": str(td_path),
            "HOME": str(td_path),
        }
        proc = subprocess.run(
            [str(bin_p), "--fit-s1", str(jtt1_p), str(cand_path), str(out_path)],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
            check=False,
        )

    return dict(
        evaluator=NATIVE_RUST_EVALUATOR_ID,
        evaluator_binary_sha256=bin_sha,
        jtt1_sha256=_file_sha256(jtt1_p),
        header=dict(nt=int(nt), nq=int(nq), p=int(p_dim), k=int(k_folds)),
        returncode=int(proc.returncode),
        stderr=proc.stderr.strip(),
        stdout=proc.stdout.strip(),
        nq0_refused_by_header_guard=bool(
            nq == 0
            and proc.returncode != 0
            and "S1 requires exactly two inner folds and an admitted shape"
            in proc.stderr
        ),
        rust_source_pin="deploy/lambda/training/teacher-v2/rust/lambda-worker/src/main.rs:567 (checks `nq < 10` in header while lines 587-606 `for _ in 0..nt` read only the `nt` training rows)",
    )


def verify_single_context_regression_seam_bundle(
    bundle_dir,
    *,
    forbidden_cids=None,
    forbidden_families=None,
    forbidden_datasets=None,
):
    """Fail-closed verifier for a single-context regression seam bundle and its engine consumption receipt.

    Rejects:
    - Empty caller-supplied exclusion sets (`forbidden_cids`, `forbidden_families`, `forbidden_datasets`).
    - Missing/null family keys, wrong role/task/split/cut, `outer_test_rows_decoded != 0`.
    - Overlap with any forbidden CID, dataset, or family alias (including old 1,310/205 fallback).
    - Missing or hash-mismatched `pool.npz`, `features.npz`, `native_evaluator_receipt.json`,
      `s3_custody_receipt.json`, or `engine_consumption_receipt.json`.
    - Wrong evaluator binary SHA256, wrong program hash, wrong DOPE/r882 encoder hash, or
      zero/unchanged adapter parameter delta on the 1-step bounded AdamW check.
    """
    for name, coll in (
        ("forbidden_cids", forbidden_cids),
        ("forbidden_families", forbidden_families),
        ("forbidden_datasets", forbidden_datasets),
    ):
        if coll is not None and len(coll) == 0:
            raise RuntimeError(f"empty exclusion set refused: {name} is empty")

    bdir = Path(bundle_dir)
    if not bdir.is_dir():
        raise RuntimeError(f"regression seam bundle directory missing: {bdir}")

    manifest_path = bdir / "context_manifest.json"
    eval_rec_path = bdir / "native_evaluator_receipt.json"
    pool_path = bdir / "pool.npz"
    feat_path = bdir / "features.npz"
    s3_rec_path = bdir / "s3_custody_receipt.json"
    eng_rec_path = bdir / "engine_consumption_receipt.json"

    for req in (
        manifest_path,
        eval_rec_path,
        pool_path,
        feat_path,
        s3_rec_path,
        eng_rec_path,
    ):
        if not req.is_file():
            raise RuntimeError(f"regression seam bundle missing required file: {req}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    eval_rec = json.loads(eval_rec_path.read_text(encoding="utf-8"))
    s3_rec = json.loads(s3_rec_path.read_text(encoding="utf-8"))
    eng_rec = json.loads(eng_rec_path.read_text(encoding="utf-8"))

    ds_id = _normalize_nonempty_key(manifest.get("dataset_id"), "dataset_id")
    fam_id = _normalize_nonempty_key(manifest.get("family_id"), "family_id")
    fam_key = _normalize_nonempty_key(manifest.get("family_key"), "family_key")
    cid = _normalize_nonempty_key(manifest.get("context_sha256"), "context_sha256")
    if not _is_hex64(cid) or not _is_hex64(manifest.get("train_sha256", "")):
        raise RuntimeError(
            "bundle manifest missing valid 64-hex context_sha256 or train_sha256"
        )

    if manifest.get("task") != "regression":
        raise RuntimeError(f"bundle manifest invalid task: {manifest.get('task')!r}")
    if (
        manifest.get("role") != "regression_train"
        or manifest.get("source_split") != "train"
        or manifest.get("dl_split") != "DL-Train"
        or bool(manifest.get("has_meta_test", False))
        or int(manifest.get("outer_test_rows_decoded", -1)) != 0
    ):
        raise RuntimeError(
            "bundle manifest invalid role/split/cut or non-zero outer_test_rows_decoded"
        )

    aliases = _row_family_aliases(manifest) | {
        fam_id,
        fam_key,
        f"dedupe:{fam_id}",
        f"dedupe:{fam_key}",
    }
    if forbidden_cids is not None and cid in set(forbidden_cids):
        raise RuntimeError(
            f"bundle context_sha256 {cid} overlaps forbidden/historical CIDs"
        )
    if forbidden_datasets is not None and ds_id in set(forbidden_datasets):
        raise RuntimeError(f"bundle dataset_id {ds_id} overlaps forbidden datasets")
    if forbidden_families is not None and (aliases & set(forbidden_families)):
        raise RuntimeError(f"bundle family {fam_id} overlaps forbidden families")

    if (
        eval_rec.get("status") != "PASS"
        or eval_rec.get("evaluator") != NATIVE_RUST_EVALUATOR_ID
        or eval_rec.get("evaluator_binary_sha256") != EVALUATOR_BINARY_SHA256
        or eval_rec.get("context_sha256") != cid
        or eval_rec.get("dataset_id") != ds_id
        or eval_rec.get("dope_sha256") != DOPE_SHA256
        or eval_rec.get("r882_sha256") != R882_SHA256
        or int(eval_rec.get("outer_test_rows_decoded", -1)) != 0
    ):
        raise RuntimeError(
            "native_evaluator_receipt failed evaluator/encoder/identity check"
        )

    if _file_sha256(pool_path) != manifest.get("pool_sha256"):
        raise RuntimeError("pool.npz file SHA256 mismatch against manifest")
    if _file_sha256(feat_path) != manifest.get("features_npz_sha256"):
        raise RuntimeError("features.npz file SHA256 mismatch against manifest")

    with np.load(pool_path, allow_pickle=False) as zp:
        for req_k in (
            "program",
            "proj16",
            "fitness",
            "gate",
            "tokens_10x32",
            "mask_10",
        ):
            if req_k not in zp.files:
                raise RuntimeError(f"pool.npz missing required tensor {req_k!r}")
        prog = zp["program"].astype(np.float32)
        proj16 = zp["proj16"].astype(np.float32)
        tok = zp["tokens_10x32"].astype(np.float32)
        msk = zp["mask_10"].astype(bool)

    with np.load(feat_path, allow_pickle=False) as zf:
        for req_k in ("tokens_10x32", "mask_10", "dope629", "lm48_z", "ctx_feat_584"):
            if req_k not in zf.files:
                raise RuntimeError(f"features.npz missing required tensor {req_k!r}")
        if (
            zf["tokens_10x32"].astype(np.float32).tobytes() != tok.tobytes()
            or zf["mask_10"].astype(bool).tobytes() != msk.tobytes()
            or zf["dope629"].shape != (629,)
            or zf["lm48_z"].shape != (48,)
            or zf["ctx_feat_584"].shape != (584,)
            or not np.isfinite(zf["dope629"]).all()
            or not np.isfinite(zf["lm48_z"]).all()
            or not np.isfinite(zf["ctx_feat_584"]).all()
        ):
            raise RuntimeError(
                "features.npz tensor shape/finite/consistency check failed"
            )

    if (
        proj16.ndim != 2
        or proj16.shape != (prog.shape[0], 16)
        or not np.isfinite(proj16).all()
    ):
        raise RuntimeError(f"invalid proj16 tensor shape {proj16.shape}")

    row_for_check = dict(
        manifest,
        pool_npz_path=str(pool_path),
        native_evaluator_receipt=eval_rec,
    )
    verified_payload = extract_and_verify_fresh_row_payload(row_for_check)

    # Verify live 24-candidate s1main re-evaluation parity recorded in native_evaluator_receipt
    live_check = eval_rec.get("live_s1main_slate24_verification")
    if (
        not isinstance(live_check, dict)
        or live_check.get("status") != "PASS"
        or int(live_check.get("evaluated_candidates", 0)) < SLATE_CANDIDATES
        or not bool(live_check.get("gate_exact_match", False))
        or float(live_check.get("max_abs_fitness_diff", 1.0)) > 1e-9
    ):
        raise RuntimeError(
            "native_evaluator_receipt missing valid live_s1main_slate24_verification"
        )

    # Verify S3 custody receipt (source dataset object + uploaded bundle objects)
    if s3_rec.get("status") != "PASS":
        raise RuntimeError("s3_custody_receipt status is not PASS")
    src_s3 = s3_rec.get("source_train_object")
    if (
        not isinstance(src_s3, dict)
        or not str(src_s3.get("version_id", "")).strip()
        or src_s3.get("sha256") != manifest.get("train_sha256")
    ):
        raise RuntimeError(
            "s3_custody_receipt invalid source_train_object VersionId/sha256"
        )
    uploaded = s3_rec.get("uploaded_bundle_objects", ())
    if not uploaded:
        raise RuntimeError("s3_custody_receipt has zero uploaded_bundle_objects")
    for obj in uploaded:
        if (
            not str(obj.get("version_id", "")).strip()
            or not _is_hex64(obj.get("local_sha256", ""))
            or obj.get("local_sha256") != obj.get("downloaded_version_sha256")
        ):
            raise RuntimeError(
                f"s3_custody_receipt readback mismatch for {obj.get('name')!r}"
            )

    # Verify inherited-engine consumption & 1-step bounded AdamW parameter delta
    if (
        eng_rec.get("status") != "PASS"
        or eng_rec.get("ingested_context_sha256") != cid
        or eng_rec.get("dataset_id") != ds_id
        or bool(eng_rec.get("used_old_1310_fallback", True))
        or int(eng_rec.get("optimizer_steps", 0)) != 1
        or int(eng_rec.get("trainable_parameters", 0)) != 29211
        or not _is_hex64(eng_rec.get("adapter_sha256_before", ""))
        or not _is_hex64(eng_rec.get("adapter_sha256_after", ""))
        or eng_rec.get("adapter_sha256_before") == eng_rec.get("adapter_sha256_after")
        or eng_rec.get("dope_sha256_before") != DOPE_SHA256
        or eng_rec.get("dope_sha256_after") != DOPE_SHA256
        or not bool(np.isfinite(float(eng_rec.get("loss_value", float("nan")))))
    ):
        raise RuntimeError(
            "engine_consumption_receipt failed consumption or parameter-delta check"
        )

    return dict(
        status="PASS",
        dataset_id=ds_id,
        family_id=fam_id,
        context_sha256=cid,
        candidates_count=int(prog.shape[0]),
        labels_sha256=verified_payload["labels_sha256"],
        programs_sha256=verified_payload["programs_sha256"],
        features_sha256=verified_payload["features_sha256"],
        adapter_sha256_before=eng_rec["adapter_sha256_before"],
        adapter_sha256_after=eng_rec["adapter_sha256_after"],
    )
