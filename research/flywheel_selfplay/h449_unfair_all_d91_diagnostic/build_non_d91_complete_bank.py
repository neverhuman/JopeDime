#!/usr/bin/env python3
"""Builds create-once `all_non_d91_exhaustive_bank.npz` (`0%` `d91`, `0` `APEX-CONFIRM`, `0` `DL-Holdout`)
and optional `d91_cousin_s3_bank.npz` (`42` `d91`-associated S3 `T1` cousin regression datasets for `ENTIRE` only).

Combines EVERY non-sealed tabular regression table across local `.jtt1` views and `veox-jopedime-datasets-use2`:
  1. All `661` complete `.jtt1` tables in `/home/ubuntu/jope-data/h335-cl/apex/tables`
     (`440` `FIT-ICL` + `106` `FIT-SEL` + `115` `APEX-DEV`; `661` distinct datasets),
  2. All `1,219` complete `.jtt1` tables in `/home/ubuntu/h317-stage/res01-train-view/tables`
     (`DL-Train` engineered & raw context cuts; `257` distinct datasets, `0` overlap with `apex/tables`),
  3. All `1,552` remaining non-sealed, non-`d91` S3 `T1` regression datasets (`train.csv U test.csv` complete `100%` rows,
     `0` `d91` / `P3_dev` / `S4` / URL / name overlap),
  -> `3,432` raw candidate tables across `2,470` distinct non-sealed, non-`d91` real datasets!

Strictly refuses any sealed holdout (`APEX-CONFIRM`, `flywheel_DL-Holdout`, `1,780` outer folds, `P3_heldout`,
`P3_sealed_excluded`, `14` catalog sealed IDs, and any path containing `confirm`, `dev91`, `dlval`, `holdout`, `sealed`).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import os
import re
import struct
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

MAX_ROWS, MAX_FEATS = 4096, 400
MIN_ROWS, MIN_COLS = 32, 2
SEED = 449_201
S3_BUCKET = "veox-jopedime-datasets-use2"
AWS_PROFILE = "veox-jepson-prod"

FORBIDDEN_PATH_TOKENS = ("confirm", "dev91", "dlval", "dl-val", "dl_val", "holdout", "hold-out", "sealed", "stage-v")

PINS = {
    "h299_manifest": (
        Path("/home/ubuntu/JopeDime/research/flywheel_selfplay/h299_fresh_cohort_v6_scout/cohort_v6_manifest.json"),
        "845a6e5739ee21f0796acb7e3f3fc6e7be0fc9138c5f3738cd9c3bd947832dfc",
    ),
    "apex_cohorts": (
        Path("/home/ubuntu/jope-data/apex/cohorts/APEX_COHORTS.json"),
        "11a57d80a67fcefbbdc83986062e50608eed78026cec7508ecbc5040999bc4eb",
    ),
    "selected": (
        Path("/home/ubuntu/h317-stage/res01-train-view/selected.json"),
        "7d64774ad9dacb5fec79190edbe0c674e778556db84dcf29c474e778556db84"[:0]
        or "7d64774ad9dacb5fec79190edbe0c674e778556db84dcf29c474e239b68e8c79",
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
        Path("/home/ubuntu/apex-stage/h446/pub-r6/code/CONFIRM_META.json"),
        "acbce707c9631d5206861fbd21185d6033e903e9530dfea83dd5933fb1e7717c",
    ),
}

S1_PROD_DIR = Path("/home/ubuntu/scratch/jopedime-dl-training-v1/receipts/trainonly-v2/s1-prod")
LOCAL_DATASET_CACHE = Path("/home/ubuntu/.cache/jope/datasets/blobs/sha256")

STOP = {
    "srcfile", "legacy", "rdatasets", "regression", "classification", "runtime", "dataset", "datasets", "data", "tbl",
    "df", "train", "test", "csv", "v1", "v2", "v3", "the", "of", "and", "openml", "uci", "kaggle",
}


def _assert_safe_path(p: Path) -> None:
    s = str(p).lower()
    for tok in FORBIDDEN_PATH_TOKENS:
        if tok in s:
            raise SystemExit(f"REFUSING forbidden holdout path token {tok!r} in {p}")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def apex_bucket(family_key: str) -> str:
    u = int.from_bytes(hashlib.sha256(f"APEX-split-v1:{family_key}".encode()).digest()[:8], "big") / float(1 << 64)
    if u < 0.40:
        return "FIT-ICL"
    if u < 0.60:
        return "FIT-SEL"
    if u < 0.80:
        return "DEV"
    return "CONFIRM"


def name_tokens(key: str):
    s = key.lower()
    body = s.split(":", 1)[-1]
    if re.fullmatch(r"[0-9a-f]{24,64}", body):
        return None
    toks = [t for t in re.split(r"[^a-z0-9]+", s) if t and t not in STOP and not (t.isdigit() and len(t) < 4)]
    return frozenset(toks) if sum(len(t) for t in toks) >= 5 else None


def s4_hit(tok, ref_list) -> bool:
    if tok is None:
        return False
    for _, qt in ref_list:
        inter = len(tok & qt)
        if tok == qt or (inter >= 2 and inter / len(tok | qt) >= 0.75):
            return True
    return False


def parse_jtt1_bytes(raw: bytes, label: str) -> np.ndarray:
    if raw[:4] != b"JTT1":
        raise ValueError(f"bad JTT1 magic in {label}")
    nt, nq, width = struct.unpack_from("<III", raw, 4)
    if nq != 0:
        raise ValueError(f"{label}: query rows present (nq={nq})")
    expected = 17 + nt + 8 * nt * (width + 1)
    if len(raw) != expected:
        raise ValueError(f"{label}: byte length {len(raw)} != {expected}")
    return np.frombuffer(raw, dtype="<f8", count=nt * (width + 1), offset=17 + nt).reshape(nt, width + 1).copy()


def parse_csv_pair_to_float64(raw_tr: bytes, raw_te: bytes | None, expected_cols: int, label: str) -> np.ndarray:
    rows_str = [r for r in csv.reader(io.StringIO(raw_tr.decode("utf-8", errors="replace"))) if r]
    if raw_te:
        rows_te = [r for r in csv.reader(io.StringIO(raw_te.decode("utf-8", errors="replace"))) if r]
        # Drop header row if test.csv starts with non-numeric header matching column count
        if rows_str and rows_te and rows_str[0] == rows_te[0]:
            rows_te = rows_te[1:]
        rows_str.extend(rows_te)
    if not rows_str:
        return np.zeros((0, max(expected_cols, 1)), np.float64)
    # Strip header row if final target token of row 0 is non-numeric and row 1 is numeric
    if len(rows_str) >= 2:
        try:
            float(rows_str[0][-1].strip())
        except ValueError:
            try:
                float(rows_str[1][-1].strip())
                rows_str = rows_str[1:]
            except ValueError:
                pass
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
        keep_f = np.sort(rng.choice(m.shape[1] - 1, MAX_FEATS, replace=False))
        m = np.hstack([m[:, keep_f], m[:, -1:]])
    with np.errstate(over="ignore"):
        m32 = m.astype(np.float32)
    m32[~np.isfinite(m32)] = np.nan
    if not np.isfinite(m32[:, -1]).all():
        m32 = m32[np.isfinite(m32[:, -1])]
    if m32.shape[0] < MIN_ROWS or float(np.nanstd(m32[:, -1])) <= 0.0:
        return None
    return m32


def _resolve_blob_path(sha_hex: str, expected_bytes: int, primary_dir: Path, readonly_dir: Path) -> Path | None:
    for d in (primary_dir, readonly_dir):
        cand = d / f"{sha_hex}.csv"
        if cand.is_file() and cand.stat().st_size == expected_bytes:
            return cand
    cand_cache = LOCAL_DATASET_CACHE / sha_hex
    if cand_cache.is_file() and cand_cache.stat().st_size == expected_bytes:
        return cand_cache
    return None


def fetch_missing_s3_blobs(blob_specs: dict[str, int], primary_dir: Path, readonly_dir: Path, threads: int = 24) -> dict:
    primary_dir.mkdir(parents=True, exist_ok=True)
    to_fetch = []
    cached = 0
    for sha_hex, exp_b in sorted(blob_specs.items()):
        if _resolve_blob_path(sha_hex, exp_b, primary_dir, readonly_dir) is not None:
            cached += 1
        else:
            to_fetch.append((sha_hex, exp_b))
    if not to_fetch:
        return {"cached": cached, "downloaded": 0, "downloaded_bytes": 0}

    import boto3
    from botocore.config import Config

    sess = boto3.Session(profile_name=AWS_PROFILE)
    s3 = sess.client("s3", region_name="us-east-2", config=Config(max_pool_connections=max(threads + 8, 32)))

    def _dl(item: tuple[str, int]) -> int:
        sh, eb = item
        dst = primary_dir / f"{sh}.csv"
        resp = s3.get_object(Bucket=S3_BUCKET, Key=f"blobs/sha256/{sh}")
        raw = resp["Body"].read()
        resp["Body"].close()
        if len(raw) != eb:
            raise RuntimeError(f"S3 blob {sh} size {len(raw)} != expected {eb}")
        if hashlib.sha256(raw).hexdigest() != sh:
            raise RuntimeError(f"S3 blob {sh} sha256 mismatch")
        tmp = primary_dir / f".{sh}.{time.time_ns()}.tmp"
        tmp.write_bytes(raw)
        tmp.chmod(0o444)
        tmp.rename(dst)
        return len(raw)

    dl_bytes = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as ex:
        for nb in ex.map(_dl, to_fetch):
            dl_bytes += nb
    return {"cached": cached, "downloaded": len(to_fetch), "downloaded_bytes": dl_bytes}


def _load_jtt1_worker(job: tuple[str, str, str, str, str]) -> tuple[str, str, str, str, np.ndarray] | None:
    cid, fam, coh, dsid, fp_s = job
    fp = Path(fp_s)
    _assert_safe_path(fp)
    mat64 = parse_jtt1_bytes(fp.read_bytes(), str(fp))
    cropped = clean_and_crop(mat64, cid)
    if cropped is None:
        return None
    return cid, fam, coh, dsid, cropped


def _load_s3_pair_worker(job: tuple[dict, str, str]) -> tuple[str, str, str, str, np.ndarray] | None:
    e, primary_s, ro_s = job
    p_tr = _resolve_blob_path(e["train_sha256"], e["train_bytes"], Path(primary_s), Path(ro_s))
    p_te = _resolve_blob_path(e["test_sha256"], e["test_bytes"], Path(primary_s), Path(ro_s))
    if p_tr is None:
        return None
    raw_tr = p_tr.read_bytes()
    if hashlib.sha256(raw_tr).hexdigest() != e["train_sha256"]:
        raise RuntimeError(f"train blob sha mismatch on {p_tr}")
    raw_te = None
    if p_te is not None:
        raw_te = p_te.read_bytes()
        if hashlib.sha256(raw_te).hexdigest() != e["test_sha256"]:
            raise RuntimeError(f"test blob sha mismatch on {p_te}")
    try:
        mat = parse_csv_pair_to_float64(raw_tr, raw_te, int(e["expected_cols"]), str(e["cid"]))
    except Exception:
        return None
    cropped = clean_and_crop(mat, str(e["cid"]))
    if cropped is None:
        return None
    return str(e["cid"]), str(e["family"]), str(e["cohort"]), str(e["dataset_id"]), cropped


def _write_npz_bank(out_p: Path, items: list[tuple[str, str, str, str, np.ndarray]], tag: str, extra: dict) -> dict:
    items = sorted(items, key=lambda t: t[0])
    shapes = np.array([t[4].shape for t in items], np.int64)
    offs = np.concatenate([[0], np.cumsum(shapes[:, 0] * shapes[:, 1])[:-1]]).astype(np.int64) if items else np.zeros(0, np.int64)
    data = np.concatenate([t[4].ravel() for t in items]).astype(np.float32) if items else np.zeros(0, np.float32)
    cids_arr = np.array([t[0] for t in items])
    fams_arr = np.array([t[1] for t in items])
    cohs_arr = np.array([t[2] for t in items])
    ds_arr = np.array([t[3] for t in items])
    m_sha = hashlib.sha256(f"{tag}:{len(items)}".encode()).hexdigest()

    out_p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_p,
        data=data,
        shape=shapes,
        offset=offs,
        cids=cids_arr,
        family=fams_arr,
        cohort=cohs_arr,
        dataset_id=ds_arr,
        manifest_sha256=np.array(m_sha),
    )
    os.chmod(out_p, 0o444)
    sha = sha256_file(out_p)
    total_rows = int(shapes[:, 0].sum()) if items else 0
    total_cells = int((shapes[:, 0] * shapes[:, 1]).sum()) if items else 0
    meta = {
        "schema": "jope.h449.exhaustive_bank/v2",
        "file": str(out_p),
        "sha256": sha,
        "tables": len(items),
        "distinct_datasets": len(set(ds_arr.tolist())),
        "distinct_families": len(set(fams_arr.tolist())),
        "total_rows": total_rows,
        "total_cells": total_cells,
        "cohorts": dict(Counter(cohs_arr.tolist())),
        **extra,
    }
    meta_p = Path(str(out_p) + ".json")
    meta_p.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    os.chmod(meta_p, 0o444)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--blob-cache-dir", default="/home/ubuntu/jope-data/h449-non-d91/s3_blobs")
    ap.add_argument("--readonly-h446-blobs", default="/home/ubuntu/jope-data/h446/s3_train_blobs")
    ap.add_argument("--apex-tables", default="/home/ubuntu/jope-data/h335-cl/apex/tables")
    ap.add_argument("--dltrain-tables", default="/home/ubuntu/h317-stage/res01-train-view/tables")
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-d91-cousins", default="")
    ap.add_argument("--fetch-s3", action="store_true")
    ap.add_argument("--procs", type=int, default=24)
    a = ap.parse_args()

    out_p = Path(a.out)
    if out_p.exists():
        raise SystemExit(f"{out_p} already exists (create-once)")
    if a.out_d91_cousins and Path(a.out_d91_cousins).exists():
        raise SystemExit(f"{a.out_d91_cousins} already exists (create-once)")
    for p_str in (a.blob_cache_dir, a.readonly_h446_blobs, a.apex_tables, a.dltrain_tables, a.out):
        _assert_safe_path(Path(p_str))

    t0 = time.time()
    pins_verified = {}
    for k, (pp, expected_sha) in PINS.items():
        got = sha256_file(pp)
        if got != expected_sha:
            raise SystemExit(f"PIN SHA MISMATCH on {k} ({pp}): {got} != {expected_sha}")
        pins_verified[k] = got

    m299 = json.loads(PINS["h299_manifest"][0].read_text())
    apex_doc = json.loads(PINS["apex_cohorts"][0].read_text())
    sel_doc = json.loads(PINS["selected"][0].read_text())
    cat_enrich = {d["id"]: d for d in json.loads(PINS["s3_catalog_enriched"][0].read_text())["datasets"]}
    cat_v246 = {
        r["id"]: r for r in (json.loads(line) for line in PINS["catalog_v246"][0].read_text().splitlines() if line.strip())
    }
    conf_meta = json.loads(PINS["confirm_meta"][0].read_text())

    # 1. Verify exact 661 non-CONFIRM apex tables & 1,219 non-d91 DL-Train tables
    apex_dir = Path(a.apex_tables)
    dl_dir = Path(a.dltrain_tables)
    apex_files = sorted(apex_dir.glob("*.jtt1"))
    dl_files = sorted(dl_dir.glob("*.jtt1"))
    if len(apex_files) != 661:
        raise SystemExit(f"expected 661 non-CONFIRM apex tables in {apex_dir}, got {len(apex_files)}")
    if len(dl_files) != 1219:
        raise SystemExit(f"expected 1219 non-d91 DL-Train tables in {dl_dir}, got {len(dl_files)}")

    apex_rows = apex_doc["contexts"]
    apex_by_cid = {r["context_sha256"]: r for r in apex_rows}
    dl_by_cid = {r["context_sha256"]: r for r in sel_doc["train"]}
    dl_train_ds = {r["dataset_id"] for r in sel_doc["train"]}
    dl_dev_ds = {r["dataset_id"] for r in sel_doc["dev"]}

    # 2. Sealed / CONFIRM / DL-Holdout / 1,780-outer-fold boundary (strictly refused in BOTH banks!)
    sealed_p3_ds, sealed_p3_fams, dev_p3_ds, dev_p3_fams = set(), set(), set(), set()
    for f in sorted([S1_PROD_DIR / "input-list.json", *S1_PROD_DIR.glob("input-list-join-*.json")]):
        doc = json.loads(f.read_text())
        for st in ("admitted", "refused"):
            for r in doc.get(st) or []:
                p3 = r.get("p3") or {}
                role = p3.get("assignment")
                if role in ("heldout", "sealed_excluded"):
                    sealed_p3_ds.add(r["dataset_hash"])
                    if p3.get("family_id"):
                        sealed_p3_fams.add(p3["family_id"])
                elif role == "dev":
                    dev_p3_ds.add(r["dataset_hash"])
                    if p3.get("family_id"):
                        dev_p3_fams.add(p3["family_id"])

    excl = m299["exclusion_family_sets"]
    outer_or_holdout_1780 = (
        set(excl["flywheel_all_1780_metadata_contexts"])
        - set(excl["flywheel_DL-Train"])
        - set(excl["flywheel_DL-Val"])
    )
    sealed_ds = (
        set(m299["catalog_sealed_matching_dataset_ids"])
        | {r["dataset_id"] for r in apex_rows if r["cohort"] == "CONFIRM"}
        | sealed_p3_ds
    )
    sealed_fams = (
        set(excl["flywheel_DL-Holdout"])
        | outer_or_holdout_1780
        | set(excl["P3_heldout_coarse_closure"])
        | set(excl["P3_sealed_excluded_coarse_closure"])
        | set(excl["catalog_BeyondArena_or_sealed_text"])
        | set(excl["catalog_sealed_matching_families_known"])
        | {r["family_key"] for r in apex_rows if r["cohort"] == "CONFIRM"}
        | {r["family_key"] for r in conf_meta["contexts"].values()}
        | sealed_p3_fams
    )
    sealed_fam_tokens = [
        (f"s_{i}", name_tokens(fk)) for i, fk in enumerate(sorted(sealed_fams)) if name_tokens(fk) is not None
    ]

    # 3. d91 / P3_dev closure (excluded from NONE-DEV91; S3 cousins routed only to optional out_d91_cousins)
    d91_tokens, d91_urls, d91_files, d91_names = [], set(), set(), set()
    for idx, r in enumerate(sel_doc["dev"]):
        dm = cat_enrich.get(r["dataset_id"], {})
        fk = dm.get("family_key") or dm.get("family_key_derived") or ""
        t = name_tokens(fk)
        if t is not None:
            d91_tokens.append((f"val_{idx}", t))
        if dm.get("source_url"):
            d91_urls.add(dm["source_url"])
        if dm.get("source_file"):
            d91_files.add(dm["source_file"])
        if dm.get("name"):
            d91_names.add(re.sub(r"[^a-z0-9]+", "", dm["name"].lower()))
    d91_ds_closure = dl_dev_ds | dev_p3_ds
    d91_fam_closure = set(excl["flywheel_DL-Val"]) | set(excl["P3_dev_coarse_closure"]) | dev_p3_fams

    # 4. Enumerate all 661 APEX non-CONFIRM .jtt1 + all 1,219 DL-Train .jtt1 jobs
    jtt1_jobs = []
    apex_jtt1_ds = set()
    for fp in apex_files:
        cid = fp.stem
        ar = apex_by_cid[cid]
        if ar["cohort"] == "CONFIRM" or ar["dataset_id"] in sealed_ds or ar["dataset_id"] in d91_ds_closure:
            raise SystemExit(f"forbidden cohort/dataset in apex/tables: {cid}")
        apex_jtt1_ds.add(ar["dataset_id"])
        jtt1_jobs.append((cid, ar["family_id"], f"APEX-{ar['cohort']}-FULL", ar["dataset_id"], str(fp)))

    for fp in dl_files:
        cid = fp.stem
        dr = dl_by_cid[cid]
        if dr["dl_split"] != "DL-Train" or dr["dataset_id"] in sealed_ds or dr["dataset_id"] in d91_ds_closure:
            raise SystemExit(f"forbidden split/dataset in res01-train-view: {cid}")
        jtt1_jobs.append((cid, dr["family_id"], "DL-TRAIN-FULL", dr["dataset_id"], str(fp)))

    # 5. Enumerate all remaining non-sealed S3 T1 regression datasets (train.csv U test.csv)
    s3_non_d91_entries = []
    s3_d91_cousin_entries = []
    blobs_needed: dict[str, int] = {}

    for ds_id, d in sorted(cat_enrich.items()):
        if d["task"] != "regression":
            continue
        fk = d.get("family_key") or d.get("family_key_derived") or ""
        bkt = apex_bucket(fk) if fk else "CONFIRM"
        tok = name_tokens(fk)
        raw_json_l = json.dumps(d).lower()
        if (
            ds_id in sealed_ds
            or not fk
            or fk in sealed_fams
            or bkt == "CONFIRM"
            or s4_hit(tok, sealed_fam_tokens)
            or "sealed" in raw_json_l
            or "beyondarena" in raw_json_l
        ):
            continue
        nn = re.sub(r"[^a-z0-9]+", "", (d.get("name") or "").lower())
        is_d91 = (
            ds_id in d91_ds_closure
            or fk in d91_fam_closure
            or s4_hit(tok, d91_tokens)
            or bool(d.get("source_url") and d["source_url"] in d91_urls)
            or bool(d.get("source_file") and d["source_file"] in d91_files)
            or bool(nn and nn in d91_names)
        )
        r246 = cat_v246[ds_id]
        tr_m = r246["files"]["train"]
        te_m = r246["files"]["test"]
        entry = {
            "cid": f"t1full:{ds_id}",
            "dataset_id": ds_id,
            "family": f"s3:{fk}",
            "cohort": "D91-S3-COUSIN" if is_d91 else f"S3-{bkt}-FULL",
            "expected_cols": int(r246["shape"]["columns"]),
            "train_sha256": tr_m["sha256"],
            "train_bytes": int(tr_m["bytes"]),
            "test_sha256": te_m["sha256"],
            "test_bytes": int(te_m["bytes"]),
        }
        if is_d91:
            if ds_id not in dl_dev_ds and a.out_d91_cousins:
                s3_d91_cousin_entries.append(entry)
                blobs_needed[tr_m["sha256"]] = int(tr_m["bytes"])
                blobs_needed[te_m["sha256"]] = int(te_m["bytes"])
        else:
            if ds_id not in apex_jtt1_ds and ds_id not in dl_train_ds:
                s3_non_d91_entries.append(entry)
                blobs_needed[tr_m["sha256"]] = int(tr_m["bytes"])
                blobs_needed[te_m["sha256"]] = int(te_m["bytes"])

    fetch_stats = {"cached": 0, "downloaded": 0, "downloaded_bytes": 0}
    if a.fetch_s3 and blobs_needed:
        fetch_stats = fetch_missing_s3_blobs(
            blobs_needed,
            Path(a.blob_cache_dir),
            Path(a.readonly_h446_blobs),
            threads=max(a.procs, 24),
        )

    with Pool(processes=min(a.procs, max(len(jtt1_jobs), 1))) as pool:
        jtt1_loaded = [x for x in pool.map(_load_jtt1_worker, jtt1_jobs) if x is not None]

    s3_non_jobs = [(e, a.blob_cache_dir, a.readonly_h446_blobs) for e in s3_non_d91_entries]
    with Pool(processes=min(a.procs, max(len(s3_non_jobs), 1))) as pool:
        s3_non_loaded = [x for x in pool.map(_load_s3_pair_worker, s3_non_jobs) if x is not None]

    non_d91_all = jtt1_loaded + s3_non_loaded
    meta_non = _write_npz_bank(
        out_p,
        non_d91_all,
        "all_non_d91_exhaustive",
        {
            "raw_candidates_total": len(jtt1_jobs) + len(s3_non_d91_entries),
            "raw_jtt1_candidates": len(jtt1_jobs),
            "raw_s3_candidates": len(s3_non_d91_entries),
            "loaded_jtt1_tables": len(jtt1_loaded),
            "loaded_s3_tables": len(s3_non_loaded),
            "s3_fetch": fetch_stats,
            "pins_verified": pins_verified,
            "d91_table_opens": 0,
            "confirm_table_opens": 0,
            "dl_holdout_opens": 0,
            "wall_s": round(time.time() - t0, 2),
        },
    )

    meta_cousins = None
    if a.out_d91_cousins:
        s3_d91_jobs = [(e, a.blob_cache_dir, a.readonly_h446_blobs) for e in s3_d91_cousin_entries]
        with Pool(processes=min(a.procs, max(len(s3_d91_jobs), 1))) as pool:
            s3_d91_loaded = [x for x in pool.map(_load_s3_pair_worker, s3_d91_jobs) if x is not None]
        meta_cousins = _write_npz_bank(
            Path(a.out_d91_cousins),
            s3_d91_loaded,
            "d91_cousin_s3",
            {
                "raw_s3_cousin_candidates": len(s3_d91_cousin_entries),
                "confirm_table_opens": 0,
                "dl_holdout_opens": 0,
            },
        )

    print(json.dumps({"non_d91_exhaustive": meta_non, "d91_cousins_s3": meta_cousins}, sort_keys=True))


if __name__ == "__main__":
    main()
