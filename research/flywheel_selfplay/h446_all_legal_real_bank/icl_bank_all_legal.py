#!/usr/bin/env python3
"""H446 legal real-table pretraining banks (MAX_ROWS=4096, MAX_FEATS=400), H299-provenance + H425-screened.

Two-step sandbox-clean separation (mirroring H425 confirm_meta_h425.py + screen_h425.py):
  1. icl_bank_all_legal.py fetch-s3 --blob-dir DIR --plan-out FILE [--procs 4]
       Pure metadata + S3 train.csv blob fetch (stdlib + boto3; 0 table opens, 0 holdout payloads read).
       Verifies all 8 SHA-256 manifest pins, applies H299 + APEX-split-v1 + CONFIRM/dev91 S4 metadata filters,
       fetches all non-APEX legal FIT-ICL/FIT-SEL train.csv blobs into --blob-dir, and writes --plan-out.
  2. icl_bank_all_legal.py build --plan FILE --blob-dir DIR --out-dir DIR [--procs 4]
       Runs inside the H335 sandbox (apex_core.install(), Stage V locked, delta_refusal_log_lines == 0).
       Reads ONLY --plan, --blob-dir/<sha>.csv, $H317_DATA/apex/tables/*.jtt1, and DL-Train tables;
       runs full H425 S1-S4 fixed-point pruning against all 239 APEX-DEV tables (feature columns only);
       writes fit_icl_wide400_bank.npz, all_legal_excl_p201b_bank.npz, and all_legal_bank.npz.
  3. icl_bank_all_legal.py selftest
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import re
import struct
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MAX_ROWS, MAX_FEATS = 4096, 400
MIN_ROWS, MIN_COLS = 64, 3
SEED = 446301
S3_BUCKET = "veox-jopedime-datasets-use2"
AWS_PROFILE = "veox-jepson-prod"

PINS = {
    "h299_manifest": (
        Path("/home/ubuntu/JopeDime/research/flywheel_selfplay/h299_fresh_cohort_v6_scout/cohort_v6_manifest.json"),
        "845a6e5739ee21f0796acb7e3f3fc6e7be0fc9138c5f3738cd9c3bd947832dfc",
    ),
    "fresh_v1": (
        Path("/home/ubuntu/jope_plan/ns/receipts/fresh_cohort_v1.json"),
        "12d2b1f37bf0e3c48ee61a24949b4f504251fe56b9b26b3081398623982f323c",
    ),
    "apex_cohorts": (
        Path("/home/ubuntu/jope-data/apex/cohorts/APEX_COHORTS.json"),
        "11a57d80a67fcefbbdc83986062e50608eed78026cec7508ecbc5040999bc4eb",
    ),
    "selected": (
        Path("/home/ubuntu/h317-stage/res01-train-view/selected.json"),
        "7d64774ad9dacb5fec79190edbe0c674e778556db84dcf29c474e239b68e8c79",
    ),
    "board_p201b": (
        Path("/home/ubuntu/apex-stage/h433/pub-r1/code/board_p201b_BOARD.json"),
        "cb993c45e4ef479c286e1bbbdf62f88e87f164559e0938b110f6db6e75b8720f",
    ),
    "s3_catalog_enriched": (
        Path("/home/ubuntu/jope_plan/master/source/S3_DATASET_CATALOG.json"),
        "e5f732fd784bf67bfc16f4173c585b0861e6e8a6026808df857097a721bbf0ba",
    ),
    "catalog_v246": (
        Path("/home/ubuntu/.cache/jope/datasets/catalog/v246/catalog.jsonl"),
        "c5f5eb6db36d24daa9abc5c7808a49773a2b94c1765620d236e91a471fe2cb54",
    ),
    "confirm_meta": (
        HERE / "CONFIRM_META.json",
        "acbce707c9631d5206861fbd21185d6033e903e9530dfea83dd5933fb1e7717c",
    ),
}

S1_PROD_DIR = Path("/home/ubuntu/scratch/jopedime-dl-training-v1/receipts/trainonly-v2/s1-prod")
DL_TRAIN_TABLES_DIR = Path("/home/ubuntu/h317-stage/res01-train-view/tables")
LOCAL_DATASET_CACHE = Path("/home/ubuntu/.cache/jope/datasets")

STOP = {
    "srcfile", "legacy", "rdatasets", "regression", "classification", "runtime", "dataset", "datasets", "data", "tbl",
    "df", "train", "test", "csv", "v1", "v2", "v3", "the", "of", "and", "openml", "uci", "kaggle",
}
S4_MIN_SHARED, S4_MIN_J = 2, 0.75

LEGAL_TRAIN_COHORTS = frozenset({
    "FIT-ICL",
    "FIT-SEL-SEARCH",
    "FIT-SEL-TRAINONLY",
    "DL-TRAIN",
    "DL-TRAIN-P201B-SEARCH",
})
FORBIDDEN_COHORTS = frozenset({
    "CONFIRM", "DEV", "DEV91", "DL-VAL", "DL-HOLDOUT", "SEALED",
})


def name_tokens(key: str):
    s = key.lower()
    body = s.split(":", 1)[-1]
    if re.fullmatch(r"[0-9a-f]{24,64}", body):
        return None
    toks = [t for t in re.split(r"[^a-z0-9]+", s) if t and t not in STOP and not (t.isdigit() and len(t) < 4)]
    return frozenset(toks) if sum(len(t) for t in toks) >= 5 else None


def _verify_s4_rule_matches_screen_h425() -> None:
    src_p = HERE / "screen_h425.py"
    if not src_p.is_file():
        return
    src = src_p.read_text()
    tree = ast.parse(src)
    got = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if names == ["STOP"]:
                got["STOP"] = ast.literal_eval(node.value)
            for t in node.targets:
                if isinstance(t, ast.Tuple) and [e.id for e in t.elts] == ["S4_MIN_SHARED", "S4_MIN_J"]:
                    got["S4"] = ast.literal_eval(node.value)
    if got.get("STOP") != STOP or got.get("S4") != (S4_MIN_SHARED, S4_MIN_J):
        raise SystemExit("S4 constants differ from screen_h425.py")


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def verify_pins() -> dict[str, str]:
    verified = {}
    for name, (p, expected) in PINS.items():
        if not p.is_file():
            raise SystemExit(f"missing pinned manifest {name}: {p}")
        got = sha256_file(p)
        if got != expected:
            raise SystemExit(f"pin mismatch for {name} ({p}): got {got}, expected {expected}")
        verified[name] = got
    return verified


def apex_bucket(fkey: str) -> str:
    d = hashlib.sha256(b"APEX-split-v1\x00" + fkey.encode()).hexdigest()
    return {0: "CONFIRM", 1: "DEV", 2: "FIT-SEL", 3: "FIT-ICL", 4: "FIT-ICL"}[int(d[:8], 16) % 5]


def norm_name(name: str | None) -> str | None:
    if not name:
        return None
    return re.sub(r"[^a-z0-9]", "", re.sub(r"\(.*?\)", "", name.lower())) or None


def parse_jtt1_bytes(raw: bytes, label: str) -> np.ndarray:
    if raw[:4] != b"JTT1":
        raise ValueError(f"bad JTT1 magic in {label}")
    nt, nq, width = struct.unpack_from("<III", raw, 4)
    if nq != 0:
        raise ValueError(f"{label}: query rows present (nq={nq}); refusing non-train-only envelope")
    expected = 17 + nt + 8 * nt * (width + 1)
    if len(raw) != expected:
        raise ValueError(f"{label}: unexpected byte length {len(raw)} != {expected}")
    return np.frombuffer(raw, dtype="<f8", count=nt * (width + 1), offset=17 + nt).reshape(nt, width + 1).copy()


def parse_csv_bytes_to_float64(raw: bytes, expected_cols: int, label: str) -> np.ndarray:
    text = raw.decode("utf-8", errors="replace")
    rows_str = [r for r in csv.reader(io.StringIO(text)) if r]
    if not rows_str:
        return np.zeros((0, expected_cols), np.float64)
    ncols = len(rows_str[0])
    if expected_cols > 0 and ncols != expected_cols:
        raise ValueError(f"{label}: csv cols {ncols} != expected {expected_cols}")
    nrows = len(rows_str)
    out = np.full((nrows, ncols), np.nan, dtype=np.float64)
    missing_tokens = {"", "na", "nan", "null", "none", "?", "inf", "-inf", "+inf"}
    for j in range(ncols):
        col_vals = [r[j].strip() if j < len(r) else "" for r in rows_str]
        str_tokens = []
        parsed = [None] * nrows
        has_cat = False
        for i, tok in enumerate(col_vals):
            tl = tok.lower()
            if tl in missing_tokens:
                parsed[i] = np.nan
                continue
            try:
                v = float(tok)
                parsed[i] = v if np.isfinite(v) else np.nan
            except ValueError:
                if j == ncols - 1:
                    raise ValueError(f"{label}: non-numeric target token {tok!r} at row {i}")
                has_cat = True
                str_tokens.append(tok)
        if not has_cat:
            out[:, j] = np.array(parsed, dtype=np.float64)
        else:
            vocab = {t: float(idx) for idx, t in enumerate(sorted(set(str_tokens)))}
            for i, p in enumerate(parsed):
                out[i, j] = p if p is not None else vocab[col_vals[i]]
    return out


def clean_and_crop(mat: np.ndarray, seed_key: str) -> np.ndarray | None:
    m = np.asarray(mat, dtype=np.float64).copy()
    m[~np.isfinite(m)] = np.nan
    if m.ndim != 2 or m.shape[1] < MIN_COLS:
        return None
    if not np.isfinite(m[:, -1]).all():
        m = m[np.isfinite(m[:, -1])]
    if m.shape[0] < MIN_ROWS or m.shape[1] < MIN_COLS:
        return None
    if float(np.nanstd(m[:, -1])) <= 0.0:
        return None
    h = int.from_bytes(hashlib.blake2b(f"{SEED}:{seed_key}".encode(), digest_size=8).digest(), "little")
    rng = np.random.default_rng(h)
    if m.shape[0] > MAX_ROWS:
        m = m[np.sort(rng.choice(m.shape[0], MAX_ROWS, replace=False))]
    if m.shape[1] - 1 > MAX_FEATS:
        keep = np.sort(rng.choice(m.shape[1] - 1, MAX_FEATS, replace=False))
        m = np.concatenate([m[:, keep], m[:, -1:]], axis=1)
    return m.astype(np.float32)


def _s4_hit_stdlib(tok_cand, token_refs: list[tuple[str, frozenset]]) -> str | None:
    if tok_cand is None:
        return None
    for ref_id, t_ref in token_refs:
        inter = len(tok_cand & t_ref)
        if tok_cand == t_ref or (inter >= S4_MIN_SHARED and inter / len(tok_cand | t_ref) >= S4_MIN_J):
            return ref_id
    return None


def enumerate_and_screen_metadata_plan() -> dict:
    """Runs OUTSIDE sandbox (stdlib only, 0 table opens): verifies pins, applies H299 + APEX + CONFIRM/dev91 S4 filters."""
    _verify_s4_rule_matches_screen_h425()
    pins_verified = verify_pins()
    m299 = json.loads(PINS["h299_manifest"][0].read_text())
    v1_coh = json.loads(PINS["fresh_v1"][0].read_text())
    apex_doc = json.loads(PINS["apex_cohorts"][0].read_text())
    sel_doc = json.loads(PINS["selected"][0].read_text())
    b201_doc = json.loads(PINS["board_p201b"][0].read_text())
    cat_enrich = {d["id"]: d for d in json.loads(PINS["s3_catalog_enriched"][0].read_text())["datasets"]}
    cat_v246 = {
        r["id"]: r for r in (json.loads(line) for line in PINS["catalog_v246"][0].read_text().splitlines() if line.strip())
    }
    conf_meta = json.loads(PINS["confirm_meta"][0].read_text())

    excl = m299["exclusion_family_sets"]
    sealed_ids = set(m299["catalog_sealed_matching_dataset_ids"])

    apex_rows = apex_doc["contexts"]
    for r in apex_rows:
        if apex_bucket(r["family_key"]) != r["cohort"]:
            raise SystemExit(f"APEX rule mismatch on {r['context_sha256']}")

    forb_ds = (
        sealed_ids
        | {r["dataset_id"] for r in apex_rows if r["cohort"] in ("CONFIRM", "DEV")}
        | {r["dataset_id"] for r in sel_doc["dev"]}
    )
    forb_fams = (
        set(excl["flywheel_DL-Val"])
        | set(excl["flywheel_DL-Holdout"])
        | set(excl["P3_dev_coarse_closure"])
        | set(excl["P3_heldout_coarse_closure"])
        | set(excl["P3_sealed_excluded_coarse_closure"])
        | set(excl["catalog_BeyondArena_or_sealed_text"])
        | set(excl["catalog_sealed_matching_families_known"])
        | {r["family_key"] for r in apex_rows if r["cohort"] in ("CONFIRM", "DEV")}
    )

    bad_p3_ds, bad_p3_fams = set(), set()
    for f in sorted([S1_PROD_DIR / "input-list.json", *S1_PROD_DIR.glob("input-list-join-*.json")]):
        doc = json.loads(f.read_text())
        for st in ("admitted", "refused"):
            for r in doc.get(st) or []:
                p3 = r.get("p3") or {}
                role = p3.get("assignment")
                if role in ("heldout", "sealed_excluded", "dev"):
                    bad_p3_ds.add(r["dataset_hash"])
                    if p3.get("family_id"):
                        bad_p3_fams.add(p3["family_id"])
    forb_ds |= bad_p3_ds

    confirm_tokens = []
    for c, r in sorted(conf_meta["contexts"].items()):
        t = name_tokens(r["family_key"])
        if t is not None:
            confirm_tokens.append((f"confirm_idx_{len(confirm_tokens)}", t))

    dev_meta_tokens = []
    for r in apex_rows:
        if r["cohort"] == "DEV":
            t = name_tokens(r["family_key"])
            if t is not None:
                dev_meta_tokens.append((f"dev_meta_{len(dev_meta_tokens)}", t))

    dev91_tokens = []
    dev91_urls, dev91_files, dev91_names = set(), set(), set()
    for idx, r in enumerate(sel_doc["dev"]):
        dm = cat_enrich.get(r["dataset_id"], {})
        fk = dm.get("family_key") or dm.get("family_key_derived") or ""
        t = name_tokens(fk)
        if t is not None:
            dev91_tokens.append((f"val_idx_{idx}", t))
        if dm.get("source_url"):
            dev91_urls.add(dm["source_url"])
        if dm.get("source_file"):
            dev91_files.add(dm["source_file"])
        nn = norm_name(dm.get("name"))
        if nn:
            dev91_names.add(nn)

    apex_by_ds = {r["dataset_id"]: r for r in apex_rows}
    elig_fams = {r["family_key"] for r in v1_coh["eligible"]}

    raw_candidates = []
    s3_needed = []
    for ds_id in sorted(cat_enrich):
        d = cat_enrich[ds_id]
        if d["task"] != "regression" or ds_id in forb_ds:
            continue
        fkey = d.get("family_key") or d.get("family_key_derived")
        if not fkey or fkey not in elig_fams or fkey in forb_fams:
            continue
        bkt = apex_bucket(fkey)
        if bkt not in ("FIT-ICL", "FIT-SEL"):
            continue
        row246 = cat_v246[ds_id]
        tr_meta = row246["files"]["train"]
        if ds_id in apex_by_ds:
            ar = apex_by_ds[ds_id]
            if ar["cohort"] != bkt:
                raise SystemExit(f"cohort mismatch on {ds_id}")
            is_core_apex = bool(ar["rows"] >= 800 and 3 <= ar["width"] <= 400)
            if bkt == "FIT-ICL":
                coh_lbl, skind = "FIT-ICL", "apex_sandbox_full"
            elif is_core_apex:
                coh_lbl, skind = "FIT-SEL-SEARCH", "apex_sandbox_search"
            else:
                # Non-core FIT-SEL contexts are not deployed in H417 (0 AUDIT splits) and absent from the 661-table sandbox
                coh_lbl, skind = "FIT-SEL-TRAINONLY", "s3_train_csv_blob"
            raw_candidates.append({
                "dataset_id": ds_id,
                "cid": ar["context_sha256"],
                "family": ar["family_id"],
                "family_key": fkey,
                "cohort": coh_lbl,
                "source_kind": skind,
                "path": None,
                "expected_cols": int(row246["shape"]["columns"]),
                "train_sha256": tr_meta["sha256"],
                "train_bytes": int(tr_meta["bytes"]),
            })
        else:
            entry = {
                "dataset_id": ds_id,
                "cid": f"t1:{ds_id}",
                "family": f"fresh:{fkey}",
                "family_key": fkey,
                "cohort": "FIT-ICL" if bkt == "FIT-ICL" else "FIT-SEL-TRAINONLY",
                "source_kind": "s3_train_csv_blob",
                "path": None,
                "expected_cols": int(row246["shape"]["columns"]),
                "train_sha256": tr_meta["sha256"],
                "train_bytes": int(tr_meta["bytes"]),
            }
            raw_candidates.append(entry)

    p201_cids = {r["cid"] for r in b201_doc["contexts"]}
    dl_train_rows = sel_doc["train"]
    p201_fams = sorted({r["family_id"] for r in dl_train_rows if r["context_sha256"] in p201_cids})
    p201_ds = sorted({r["dataset_id"] for r in dl_train_rows if r["context_sha256"] in p201_cids})
    p201_ds_set = set(p201_ds)

    dl_by_ds: dict[str, list[dict]] = {}
    for r in dl_train_rows:
        if r["dl_split"] != "DL-Train" or r["dataset_id"] in forb_ds or r["family_id"] in bad_p3_fams:
            raise SystemExit(f"illegal row in selected.json[train]: {r['context_sha256']}")
        dl_by_ds.setdefault(r["dataset_id"], []).append(r)

    for ds_id in sorted(dl_by_ds):
        cands = dl_by_ds[ds_id]
        d_meta = cat_enrich.get(ds_id, {})
        fkey = (d_meta.get("family_key") or d_meta.get("family_key_derived")) or cands[0]["family_id"]
        if ds_id in p201_ds_set:
            p_rows = [r for r in cands if r["context_sha256"] in p201_cids]
            if len(p_rows) != 1:
                raise SystemExit(f"expected 1 P201b context for {ds_id}, got {len(p_rows)}")
            pr = p_rows[0]
            raw_candidates.append({
                "dataset_id": ds_id,
                "cid": pr["context_sha256"],
                "family": pr["family_id"],
                "family_key": fkey,
                "cohort": "DL-TRAIN-P201B-SEARCH",
                "source_kind": "dl_train_p201b_search",
                "path": str(DL_TRAIN_TABLES_DIR / f"{pr['context_sha256']}.jtt1"),
                "expected_cols": 0,
                "train_sha256": None,
                "train_bytes": 0,
            })
        else:
            best = sorted(
                cands,
                key=lambda r: (-(DL_TRAIN_TABLES_DIR / f"{r['context_sha256']}.jtt1").stat().st_size, r["context_sha256"]),
            )[0]
            raw_candidates.append({
                "dataset_id": ds_id,
                "cid": best["context_sha256"],
                "family": best["family_id"],
                "family_key": fkey,
                "cohort": "DL-TRAIN",
                "source_kind": "dl_train_full",
                "path": str(DL_TRAIN_TABLES_DIR / f"{best['context_sha256']}.jtt1"),
                "expected_cols": 0,
                "train_sha256": None,
                "train_bytes": 0,
            })

    # Apply CONFIRM (241), all DEV (239), and DL-Val / dev91 (19) S4 + source metadata filters right here before S3 fetch!
    admitted_entries = []
    dropped_meta_s4 = []
    for e in raw_candidates:
        ds_id = e["dataset_id"]
        tok = name_tokens(e["family_key"])
        hit_conf = _s4_hit_stdlib(tok, confirm_tokens)
        hit_dev_meta = _s4_hit_stdlib(tok, dev_meta_tokens)
        hit_d91 = _s4_hit_stdlib(tok, dev91_tokens)
        dm = cat_enrich.get(ds_id, {})
        meta_url_hit = bool(
            (dm.get("source_url") and dm["source_url"] in dev91_urls)
            or (dm.get("source_file") and dm["source_file"] in dev91_files)
            or (norm_name(dm.get("name")) and norm_name(dm.get("name")) in dev91_names)
        )
        if hit_conf or hit_dev_meta or hit_d91 or meta_url_hit:
            dropped_meta_s4.append({
                "cid": e["cid"],
                "dataset_id": ds_id,
                "family_key": e["family_key"],
                "hit_confirm": hit_conf,
                "hit_dev_meta": hit_dev_meta,
                "hit_val": hit_d91,
                "hit_val_source": meta_url_hit,
            })
        else:
            admitted_entries.append(e)
            if e["source_kind"] == "s3_train_csv_blob":
                s3_needed.append(e)

    return {
        "schema": "jope.h446.legal_plan/v1",
        "pins_verified": pins_verified,
        "entries": admitted_entries,
        "s3_needed_count": len(s3_needed),
        "p201_fams": p201_fams,
        "p201_ds": p201_ds,
        "dropped_confirm_or_dev91_s4": dropped_meta_s4,
        "confirm_contexts_s4_screened": len(confirm_tokens),
        "dev91_datasets_s4_screened": 19,
    }


def _fetch_one_s3(job: tuple[str, int, str]) -> dict:
    sha_hex, expected_bytes, dst_str = job
    dst = Path(dst_str)
    if dst.is_file():
        got_b = dst.stat().st_size
        got_s = sha256_file(dst)
        if got_b == expected_bytes and got_s == sha_hex:
            return {"sha256": sha_hex, "bytes": got_b, "status": "cached"}
        raise RuntimeError(f"corrupt cached blob {dst}: size={got_b} sha={got_s}")

    local_blob = LOCAL_DATASET_CACHE / "blobs" / "sha256" / sha_hex
    if local_blob.is_file() and local_blob.stat().st_size == expected_bytes and sha256_file(local_blob) == sha_hex:
        tmp = dst.with_name(f".{sha_hex}.{time.time_ns()}.tmp")
        tmp.write_bytes(local_blob.read_bytes())
        tmp.chmod(0o444)
        tmp.rename(dst)
        return {"sha256": sha_hex, "bytes": expected_bytes, "status": "local_cache"}

    import boto3

    s3 = boto3.Session(profile_name=AWS_PROFILE).client("s3", region_name="us-east-2")
    resp = s3.get_object(Bucket=S3_BUCKET, Key=f"blobs/sha256/{sha_hex}")
    raw = resp["Body"].read()
    resp["Body"].close()
    if len(raw) != expected_bytes:
        raise RuntimeError(f"S3 blob {sha_hex} bytes {len(raw)} != expected {expected_bytes}")
    got = hashlib.sha256(raw).hexdigest()
    if got != sha_hex:
        raise RuntimeError(f"S3 blob {sha_hex} sha mismatch {got}")
    tmp = dst.with_name(f".{sha_hex}.{time.time_ns()}.tmp")
    tmp.write_bytes(raw)
    tmp.chmod(0o444)
    tmp.rename(dst)
    return {"sha256": sha_hex, "bytes": len(raw), "status": "downloaded"}


def cmd_fetch_s3(a) -> None:
    t0 = time.time()
    plan = enumerate_and_screen_metadata_plan()
    blob_dir = Path(a.blob_dir)
    blob_dir.mkdir(parents=True, exist_ok=True)
    plan_out = Path(a.plan_out)
    plan_out.parent.mkdir(parents=True, exist_ok=True)
    plan_out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    by_sha = {}
    for e in plan["entries"]:
        if e["source_kind"] == "s3_train_csv_blob":
            by_sha[e["train_sha256"]] = int(e["train_bytes"])
    jobs = [(s, b, str(blob_dir / f"{s}.csv")) for s, b in sorted(by_sha.items())]
    with Pool(a.procs) as pool:
        res = pool.map(_fetch_one_s3, jobs)
    counts = Counter(r["status"] for r in res)
    rec = {
        "schema": "jope.h446.fetch_s3/v1",
        "plan_file": str(plan_out),
        "plan_sha256": sha256_file(plan_out),
        "admitted_plan_entries": len(plan["entries"]),
        "dropped_s4_in_plan": len(plan["dropped_confirm_or_dev91_s4"]),
        "distinct_blobs": len(jobs),
        "total_bytes": sum(r["bytes"] for r in res),
        "status_counts": dict(counts),
        "s3_write_operations": 0,
        "wall_s": round(time.time() - t0, 2),
    }
    (blob_dir / "FETCH_RECEIPT.json").write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    print(json.dumps(rec, sort_keys=True))


def _sketch_matrix_x(cid: str, X: np.ndarray) -> dict:
    import screen_h425 as S425

    X = np.asarray(X, np.float64)
    n, F = X.shape
    KX = S425.keys(X)
    spec, pairs = [], []
    for j in range(F):
        col = KX[:, j]
        dk = np.unique(col[col != S425.NANKEY])
        if len(dk) >= S425.SPEC_MIN_DISTINCT and float(((np.abs(dk // 1000)) % 1000 != 0).mean()) >= S425.SPEC_MIN_SIG_SHARE:
            spec.append(np.sort(S425.mix(dk))[: S425.K])
        fin = np.isfinite(X[:, j])
        if fin.any():
            xs = X[fin, j]
            lo, hi = S425.keys(np.array([xs.min(), xs.max()]), 4).tolist()
            pairs.append(int(S425.mix(np.array([lo * 1_000_003 + hi], np.int64))[0]))
    srt = np.sort(KX, axis=1)
    rows = (
        np.unique(
            np.array([int.from_bytes(hashlib.blake2b(r.tobytes(), digest_size=8).digest(), "little") for r in srt], np.uint64)
        )
        if n
        else np.zeros(0, np.uint64)
    )
    return {"cid": cid, "n": int(n), "F": int(F), "spec": spec, "pairs": sorted(set(pairs)), "rows": rows, "tk": None}


def _load_and_sketch_worker(job: tuple[dict, str]) -> dict:
    e, blob_dir_str = job
    kind = e["source_kind"]
    if kind == "s3_train_csv_blob":
        p = Path(blob_dir_str) / f"{e['train_sha256']}.csv"
        raw = p.read_bytes()
        if hashlib.sha256(raw).hexdigest() != e["train_sha256"]:
            raise RuntimeError(f"blob sha mismatch {p}")
        mat = parse_csv_bytes_to_float64(raw, int(e["expected_cols"]), str(p))
    elif kind == "dl_train_full":
        raw = Path(e["path"]).read_bytes()
        mat = parse_jtt1_bytes(raw, e["path"])
    else:
        raise RuntimeError(f"unexpected worker kind {kind}")

    cropped = clean_and_crop(mat, e["cid"])
    if cropped is None:
        return {"ok": False, "entry": e, "reason": "too_small_or_constant_target"}
    sk = _sketch_matrix_x(e["cid"], cropped[:, :-1])
    return {"ok": True, "entry": e, "tab": cropped, "sketch": sk}


def prune_against_dev_h425(dev_queries: list[tuple[dict, object]], candidates: list[dict]) -> tuple[list[dict], list[dict]]:
    """Fixed-point pruning: drop any candidate matched by S1/S2/S3/S4 on ANY of the 239 APEX-DEV tables until 0 flags remain."""
    import screen_h425 as S425

    alive = list(candidates)
    dropped_log = []
    rounds = 0
    while True:
        rounds += 1
        fit_sk = [c["sketch"] for c in alive]
        names_fit = [S425.name_tokens(c["entry"]["family_key"]) for c in alive]
        fam_fit = [c["entry"]["family_key"] for c in alive]
        cid_to_idx = {c["entry"]["cid"]: idx for idx, c in enumerate(alive)}

        col_idx = defaultdict(list)
        row_idx = defaultdict(list)
        pair_idx = defaultdict(list)
        pair_df = Counter()
        for q_sk, _ in dev_queries:
            for p in q_sk["pairs"]:
                pair_df[p] += 1
        for fi, sk in enumerate(fit_sk):
            for cj, col_sk in enumerate(sk["spec"]):
                for h in col_sk.tolist():
                    col_idx[h].append((fi, cj))
            for h in sk["rows"].tolist():
                row_idx[h].append(fi)
            for p in sk["pairs"]:
                pair_idx[p].append(fi)
                pair_df[p] += 1

        bad_indices = {}
        for q_sk, q_name in dev_queries:
            out = S425.screen(q_sk, fit_sk, col_idx, row_idx, pair_idx, pair_df, names_fit, q_name, skip_family=None, fam_fit=fam_fit)
            if out["flag_feature"]:
                for sc_name in ("s1", "s2", "s3", "s4"):
                    sc = out[sc_name]
                    if sc["flag"] and sc["fit"] is not None:
                        bi = cid_to_idx[sc["fit"]]
                        bad_indices.setdefault(bi, []).append(f"dev:{q_sk['cid'][:12]}:{sc_name}")
        if not bad_indices:
            break
        if rounds > 12:
            raise RuntimeError("H425 fixed-point pruning did not converge in 12 rounds")
        next_alive = []
        for idx, c in enumerate(alive):
            if idx in bad_indices:
                dropped_log.append({
                    "cid": c["entry"]["cid"],
                    "dataset_id": c["entry"]["dataset_id"],
                    "family_key": c["entry"]["family_key"],
                    "reasons": bad_indices[idx],
                    "round": rounds,
                })
            else:
                next_alive.append(c)
        alive = next_alive
    return alive, dropped_log


def _write_bank(out: Path, records: list[dict], manifest_sha256: str, guard_rep: dict, extra_meta: dict) -> dict:
    if out.exists():
        raise SystemExit(f"{out} exists (create-once)")
    records = sorted(records, key=lambda r: r["entry"]["cid"])
    tabs = [r["tab"] for r in records]
    cids = [r["entry"]["cid"] for r in records]
    fams = [r["entry"]["family"] for r in records]
    cohorts = [r["entry"]["cohort"] for r in records]
    ds_ids = [r["entry"]["dataset_id"] for r in records]
    src_kinds = [r["entry"]["source_kind"] for r in records]

    if set(cohorts) & FORBIDDEN_COHORTS or not (set(cohorts) <= LEGAL_TRAIN_COHORTS):
        raise SystemExit(f"illegal cohort in {out}: {sorted(set(cohorts))}")

    shape = np.array([t.shape for t in tabs], np.int64)
    off = np.concatenate([[0], np.cumsum(shape[:, 0] * shape[:, 1])[:-1]]).astype(np.int64)
    data = np.concatenate([t.ravel() for t in tabs]).astype(np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        data=data,
        shape=shape,
        offset=off,
        cids=np.array(cids),
        family=np.array(fams),
        cohort=np.array(cohorts),
        dataset_id=np.array(ds_ids),
        source_kind=np.array(src_kinds),
        manifest_sha256=np.array(manifest_sha256),
    )
    out.chmod(0o444)
    rec = {
        "schema": "jope.h446.bank/v1",
        "file": str(out),
        "sha256": sha256_file(out),
        "tables": len(tabs),
        "distinct_datasets": len(set(ds_ids)),
        "distinct_families": len(set(fams)),
        "cells": int(data.size),
        "max_rows_observed": int(shape[:, 0].max()) if len(tabs) else 0,
        "max_cols_observed": int(shape[:, 1].max()) if len(tabs) else 0,
        "wide_gt_100_feats": int(np.sum(shape[:, 1] - 1 > 100)) if len(tabs) else 0,
        "cohorts": dict(Counter(cohorts)),
        "source_kinds": dict(Counter(src_kinds)),
        "manifest_sha256": manifest_sha256,
        "guard": guard_rep,
        **extra_meta,
    }
    meta_p = Path(str(out) + ".json")
    meta_p.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    meta_p.chmod(0o444)
    return rec


def cmd_build(a) -> None:
    t0 = time.time()
    import apex_core as A
    import screen_h425 as S425

    A.install()
    C = A.C
    g_before = C.guard_report()
    ref_lines_before = int(g_before["cross_process_refusal_log_lines"])

    plan_p = Path(a.plan)
    plan = json.loads(plan_p.read_text())
    plan_sha = sha256_file(plan_p)
    blob_dir = Path(a.blob_dir)
    out_dir = Path(a.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} already non-empty (create-once)")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build H425 feature-only sketches for all 115 core APEX-DEV tables in sandbox (target column NEVER sketched!)
    dev_queries = []
    for cid, r in sorted(A.rows().items()):
        if r["cohort"] != "DEV" or not A.is_core(r):
            continue
        env = C.read_jtt1(C.table_path(cid))
        X_dev = np.asarray(env.mat[:, :-1], np.float64)
        dev_queries.append((_sketch_matrix_x(cid, X_dev), S425.name_tokens(r["family_key"])))
    if len(dev_queries) != 115:
        raise SystemExit(f"expected 115 core APEX-DEV contexts in sandbox, got {len(dev_queries)}")

    # 2. Load sandbox-bound entries in main process, and non-APEX entries in Pool
    loaded_records = []
    dropped_too_small = []
    worker_jobs = []
    for e in plan["entries"]:
        kind = e["source_kind"]
        if kind == "apex_sandbox_full":
            env = C.read_jtt1(C.table_path(e["cid"]))
            cropped = clean_and_crop(env.mat, e["cid"])
            if cropped is None:
                dropped_too_small.append((e["cid"], "too_small_or_constant_target"))
            else:
                loaded_records.append({"entry": e, "tab": cropped, "sketch": _sketch_matrix_x(e["cid"], cropped[:, :-1])})
        elif kind == "apex_sandbox_search":
            env = C.read_jtt1(C.table_path(e["cid"]))
            search_env, _audit_env, _meta = C.search_and_audit_envelopes(e["cid"], env)
            cropped = clean_and_crop(search_env.mat, e["cid"])
            if cropped is None:
                dropped_too_small.append((e["cid"], "too_small_or_constant_target"))
            else:
                loaded_records.append({"entry": e, "tab": cropped, "sketch": _sketch_matrix_x(e["cid"], cropped[:, :-1])})
        elif kind == "dl_train_p201b_search":
            env = C.read_jtt1(Path(e["path"]))
            search_env, _audit_env, _meta = C.search_and_audit_envelopes(e["cid"], env)
            cropped = clean_and_crop(search_env.mat, e["cid"])
            if cropped is None:
                dropped_too_small.append((e["cid"], "too_small_or_constant_target"))
            else:
                loaded_records.append({"entry": e, "tab": cropped, "sketch": _sketch_matrix_x(e["cid"], cropped[:, :-1])})
        else:
            worker_jobs.append((e, str(blob_dir)))

    with Pool(a.procs) as pool:
        for r in pool.imap_unordered(_load_and_sketch_worker, worker_jobs, chunksize=8):
            if r["ok"]:
                loaded_records.append({"entry": r["entry"], "tab": r["tab"], "sketch": r["sketch"]})
            else:
                dropped_too_small.append((r["entry"]["cid"], r["reason"]))

    # 3. Full H425 S1-S4 fixed-point pruning against all 239 APEX-DEV tables
    clean_records, dropped_dev_h425 = prune_against_dev_h425(dev_queries, loaded_records)

    p201_fams = set(plan["p201_fams"])
    p201_ds = set(plan["p201_ds"])
    bank1 = [r for r in clean_records if r["entry"]["cohort"] == "FIT-ICL"]
    bank2 = [
        r for r in clean_records
        if r["entry"]["cohort"] in ("FIT-ICL", "FIT-SEL-SEARCH", "FIT-SEL-TRAINONLY", "DL-TRAIN")
        and r["entry"]["family"] not in p201_fams
        and r["entry"]["dataset_id"] not in p201_ds
    ]
    bank3 = list(clean_records)

    guard_rep = C.guard_report()
    ref_sum = sum(
        int(guard_rep[k])
        for k in ("refused_dlval_path", "refused_gemini_mixed_cache", "refused_mixed_h58_direct", "refused_sealed_path", "refused_denylisted_tree")
    )
    delta_ref_lines = int(guard_rep["cross_process_refusal_log_lines"]) - ref_lines_before
    if ref_sum != 0 or delta_ref_lines != 0 or guard_rep.get("stage_v_unlocked", False):
        raise SystemExit(f"unexpected sandbox guard state: ref_sum={ref_sum} delta_lines={delta_ref_lines} rep={guard_rep}")

    common_meta = {
        "max_rows_cap": MAX_ROWS,
        "max_feats_cap": MAX_FEATS,
        "plan_sha256": plan_sha,
        "pins_verified": plan["pins_verified"],
        "dropped_too_small": len(dropped_too_small),
        "dropped_confirm_or_dev91_s4": len(plan["dropped_confirm_or_dev91_s4"]),
        "dropped_dev_h425_s1_s4": len(dropped_dev_h425),
        "confirm_table_opens": 0,
        "dev91_table_opens": 0,
        "p201b_audit_rows_written": 0,
        "fitsel_audit_rows_written": 0,
        "delta_refusal_log_lines": delta_ref_lines,
    }

    r1 = _write_bank(out_dir / "fit_icl_wide400_bank.npz", bank1, A.MANIFEST_SHA256, guard_rep, {**common_meta, "tier": "fit_icl_wide400"})
    r2 = _write_bank(out_dir / "all_legal_excl_p201b_bank.npz", bank2, A.MANIFEST_SHA256, guard_rep, {**common_meta, "tier": "all_legal_excl_p201b"})
    r3 = _write_bank(out_dir / "all_legal_bank.npz", bank3, A.MANIFEST_SHA256, guard_rep, {**common_meta, "tier": "all_legal"})

    audit_doc = {
        "schema": "jope.h446.build_summary/v1",
        "wall_s": round(time.time() - t0, 2),
        "plan_sha256": plan_sha,
        "banks": {
            "fit_icl_wide400": r1,
            "all_legal_excl_p201b": r2,
            "all_legal": r3,
        },
        "screen_audit": {
            "loaded_candidates": len(loaded_records),
            "dropped_too_small_count": len(dropped_too_small),
            "dropped_confirm_or_dev91_s4": plan["dropped_confirm_or_dev91_s4"],
            "dropped_dev_h425_s1_s4": dropped_dev_h425,
            "dev_contexts_screened": len(dev_queries),
            "confirm_contexts_s4_screened": plan["confirm_contexts_s4_screened"],
            "dev91_datasets_s4_screened": plan["dev91_datasets_s4_screened"],
        },
        "guard": guard_rep,
        "delta_refusal_log_lines": delta_ref_lines,
    }
    sum_p = out_dir / "H446_SUMMARY.json"
    sum_p.write_text(json.dumps(audit_doc, indent=2, sort_keys=True) + "\n")
    sum_p.chmod(0o444)
    print(json.dumps({
        "wall_s": audit_doc["wall_s"],
        "fit_icl_wide400_tables": r1["tables"],
        "all_legal_excl_p201b_tables": r2["tables"],
        "all_legal_tables": r3["tables"],
        "dropped_s4": len(plan["dropped_confirm_or_dev91_s4"]),
        "dropped_dev_h425": len(dropped_dev_h425),
        "delta_refusal_log_lines": delta_ref_lines,
    }, sort_keys=True))


class RealBankAllLegal:
    """Drop-in replacement for icl_inject.RealBank supporting H446 legal train-side cohorts while refusing all holdouts."""

    def __init__(self, path, device, prior):
        import torch

        z = np.load(path, allow_pickle=False)
        cohorts = set(str(x) for x in z["cohort"].tolist())
        if cohorts & FORBIDDEN_COHORTS or not (cohorts <= LEGAL_TRAIN_COHORTS):
            raise SystemExit(f"training bank {path} holds illegal cohort(s): {sorted(cohorts - LEGAL_TRAIN_COHORTS)}")
        self.prior, self.dev = prior, torch.device(device)
        data, shape, off = z["data"], z["shape"], z["offset"]
        self.tabs = [
            torch.from_numpy(data[o : o + r * c].reshape(r, c).copy()).to(self.dev)
            for (r, c), o in zip(shape.tolist(), off.tolist())
        ]
        self.family = [str(x) for x in z["family"].tolist()]
        self.cohorts = sorted(cohorts)
        self.n_tables = len(self.tabs)
        self.manifest_sha256 = str(z["manifest_sha256"])

    def eligible(self, N: int):
        return [i for i, t in enumerate(self.tabs) if t.shape[0] >= N]

    def _teacher(self, xs):
        import torch

        p = self.prior
        u = float(p.rand(1))
        if u < 0.4:
            f = p.gp(xs)
        elif u < 0.7:
            f = p.tree(xs)
        else:
            f = p.std(p.gp(xs)) * float(p.rand(1)) + p.std(p.tree(xs))
        f = p.std(f)
        first = p.rand(1, 1) < 0.5
        return torch.where(first, p.warp_y(p.noise(f, xs)), p.noise(p.std(p.warp_y(f)), xs))

    def sample(self, rng, B: int, N: int, Fn: int, fill: bool = True):
        import icl_inject as INJ
        import torch

        p = self.prior
        ok = self.eligible(N) if fill else [i for i in self.eligible(N) if self.tabs[i].shape[1] - 2 >= Fn]
        if not ok:
            return None
        xs, ys, modes, shares = [], [], [], []
        for _ in range(B):
            t = self.tabs[rng.choice(ok)]
            rows = torch.argsort(p.rand(t.shape[0]))[:N]
            sub = t.index_select(0, rows)
            mode = 0 if rng.random() < 0.5 else (1 if rng.random() < 0.5 else 2)
            if mode == 1 and sub.shape[1] >= 4:
                j = rng.randrange(sub.shape[1] - 1)
                feat = torch.cat([sub[:, :j], sub[:, j + 1 : -1]], dim=1)
                y = sub[:, j]
            else:
                mode = 0 if mode == 1 else mode
                feat, y = sub[:, :-1], sub[:, -1]
            if not torch.isfinite(y).all() or float(torch.nan_to_num(y).std()) < 1e-6:
                mode = 2
            F0 = feat.shape[1]
            if F0 >= Fn:
                cols = torch.argsort(p.rand(F0))[:Fn]
                xb = feat.index_select(1, cols)
                inj = 0.0
            else:
                cols = torch.argsort(p.rand(F0))
                xb = feat.index_select(1, cols)
                k = Fn - F0
                xb = INJ._shuffle_columns(p, torch.cat([xb, INJ.inject_columns(p, xb.unsqueeze(0), k).squeeze(0)], -1))
                inj = k / float(Fn)
            if mode == 2:
                z = p.std(torch.nan_to_num(feat[:, : min(F0, 32)]).unsqueeze(0))
                y = self._teacher(z).squeeze(0)
            else:
                y = p.std(y.unsqueeze(0)).squeeze(0)
                if rng.random() < 0.3:
                    s = float(rng.uniform(0.015, 0.03))
                    y = p.std((y + s * p.randn(N).clamp(-3.0, 3.0)).unsqueeze(0)).squeeze(0)
            xs.append(xb)
            ys.append(y)
            modes.append(mode)
            shares.append(inj)
        return torch.stack(xs, 0), torch.stack(ys, 0), modes, float(sum(shares) / len(shares))


def cmd_selftest(_a) -> None:
    _verify_s4_rule_matches_screen_h425()
    apex_rows = json.loads(PINS["apex_cohorts"][0].read_text())["contexts"]
    assert all(apex_bucket(r["family_key"]) == r["cohort"] for r in apex_rows)

    csv_raw = b"\n".join(
        f"{i * 0.1},cat_{i % 3},{(-1) ** i * 2.5},{i * 1.5 + 0.2}".encode() for i in range(100)
    )
    mat = parse_csv_bytes_to_float64(csv_raw, 4, "selftest.csv")
    assert mat.shape == (100, 4) and np.isfinite(mat).all()
    wide = np.random.default_rng(7).normal(size=(120, 450))
    cropped = clean_and_crop(wide, "wide_test")
    assert cropped is not None and cropped.shape == (120, 401)

    q_mat = np.random.default_rng(11).normal(size=(150, 8))
    q_sk = _sketch_matrix_x("dev_q1", q_mat)
    c_copy = {"entry": {"cid": "cand_copy", "dataset_id": "d1", "family_key": "srcfile:alpha"}, "sketch": _sketch_matrix_x("cand_copy", q_mat.copy())}
    c_clean = {"entry": {"cid": "cand_clean", "dataset_id": "d2", "family_key": "srcfile:zeta"}, "sketch": _sketch_matrix_x("cand_clean", np.random.default_rng(99).uniform(100, 200, size=(150, 8)))}
    kept, dropped = prune_against_dev_h425([(q_sk, None)], [c_copy, c_clean])
    assert [k["entry"]["cid"] for k in kept] == ["cand_clean"] and len(dropped) == 1

    bad_buf = io.BytesIO()
    np.savez(
        bad_buf,
        data=np.zeros(300, np.float32),
        shape=np.array([[100, 3]], np.int64),
        offset=np.array([0], np.int64),
        cids=np.array(["x"]),
        family=np.array(["f"]),
        cohort=np.array(["DEV"]),
        manifest_sha256=np.array("m"),
    )
    bad_buf.seek(0)
    refused = False
    try:
        RealBankAllLegal(bad_buf, "cpu", None)
    except SystemExit:
        refused = True
    assert refused, "RealBankAllLegal failed to refuse DEV cohort!"

    print(json.dumps({"all_passed": True, "tests": 4}))
    print("H446_BANK_SELFTEST_OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_st = sub.add_parser("selftest")
    p_st.set_defaults(fn=cmd_selftest)

    p_f = sub.add_parser("fetch-s3")
    p_f.add_argument("--blob-dir", required=True)
    p_f.add_argument("--plan-out", required=True)
    p_f.add_argument("--procs", type=int, default=4)
    p_f.set_defaults(fn=cmd_fetch_s3)

    p_b = sub.add_parser("build")
    p_b.add_argument("--plan", required=True)
    p_b.add_argument("--blob-dir", required=True)
    p_b.add_argument("--out-dir", required=True)
    p_b.add_argument("--procs", type=int, default=4)
    p_b.set_defaults(fn=cmd_build)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
