"""L7a v2 train-only build, local and unpublished (owner decisions L6(a), L7a): original T1 source rows, content-matched.

For every included epoch-0 context (scale_check.resolve_context lineage), grouped by dataset:
  held (rights gate, no licence)  -> context.json values_mode "held_no_license" (excluded, counted; never fetched or read)
  fetched and content-matched     -> L7a files, same layout as the L7b build (tools/v2_build_local.py):
        encoder.jtt1.gz  train-only min-max JTT1 of the L7a values (t1raw.prepare)
        teacher.jtt1.gz  same rows, columns and inner folds, refitted (train-median imputation, train-only category
                         codes, train-only target-copy selection) but NOT min-max scaled
        a0.f32           A0 meta-features of encoder.jtt1 (metafeatures.from_raw)
        context.json     identities, file sha256s, t1raw report (imputation counts), match statistics, raw source
                         custody, row_selection_leak, code sha256s, values_mode "L7a", published=false
  not fetched, not parseable, or not matched (or an L7a split that cannot be prepared)
                                  -> context.json values_mode "L7b-fallback": reason + pointer to the L7b context
Every context gets .done last; a context with .done is skipped (resumable, idempotent). Per dataset a match record goes
to OUT/shard-<i>/_datasets/<dataset_hash>.json. Per shard: shard-<i>-of-<n>.receipt.json (expected, built, already done,
by values_mode, failed). No S3, network or GPU; nothing is published. Sharding is the L7b rule (dataset_hash[:8] % n),
so L7a shard i points into L7b shard i.

Usage (inside the lane slice):
  v2_build_l7a.py --out DIR --shard I --of N
  v2_build_l7a.py --out DIR --summarize --of N --tests-log LOG --mutants JSON [--receipt PATH]
"""
import argparse
import collections
import glob
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from dltrain import cpuguard, metafeatures, scale_check, t1raw, trainonly

RUN = Path("/home/ubuntu/scratch/jopedime-dl-training-v1")
META_DIR = RUN / "inputs" / "t1-meta"
RAW_DIR = RUN / "inputs" / "t1-raw"
DATASETS_INV = RUN / "tmp" / "trainonly-v1" / "epoch0-datasets.json"
RECEIPTS = RUN / "receipts" / "trainonly-v2"
PLAN = RECEIPTS / "l7a-fetch-plan-licensed-only.jsonl"
LEAK = RECEIPTS / "row-selection-leak-20261005T185553Z.jsonl"
L7B = RUN / "build-v2" / "L7b"
SCHEMA = "jope.dl-v2-l7a-context/v1"
CODE_FILES = ("tools/v2_build_l7a.py", "dltrain/t1raw.py", "dltrain/t1prep.py", "dltrain/trainonly.py", "dltrain/metafeatures.py",
              "dltrain/scale_check.py", "dltrain/jtt1.py")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic(path: Path, data: bytes):
    tmp = path.with_name("." + path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def jdump(o) -> bytes:
    return (json.dumps(o, sort_keys=True) + "\n").encode()


def code_shas():
    root = Path(__file__).resolve().parents[1]
    return {f: sha((root / f).read_bytes()) for f in CODE_FILES}


def held_set():
    paths = sorted(glob.glob(str(RECEIPTS / "rights-gate-no-license-*.jsonl")))
    if len(paths) != 1:
        raise SystemExit(f"expected one rights-gate receipt, found {paths}")
    b = Path(paths[0]).read_bytes()
    rows = [json.loads(l) for l in b.decode().splitlines() if l.strip()]
    return {r["dataset_hash"]: r for r in rows}, {"path": paths[0], "sha256": sha(b), "datasets": len(rows),
                                                  "contexts": sum(int(r["contexts"]) for r in rows)}


def leak_map():
    b = LEAK.read_bytes()
    out = {}
    for l in b.decode().splitlines():
        if l.strip():
            r = json.loads(l)
            out[r["view_id"]] = r["row_selection_leak"]
    return out, {"path": str(LEAK), "sha256": sha(b), "rows": len(out)}


def fetch_errors():
    """dataset_hash -> last fetch-run result that was not a success (from the fetch-run receipts)."""
    errs = {}
    for p in sorted(glob.glob(str(RAW_DIR / "fetch-run-*.json"))):
        d = json.loads(Path(p).read_bytes())
        for r in d.get("results", []):
            if not str(r.get("status", "")).startswith(("fetched", "ok", "skipped")):
                errs[r["dataset_hash"]] = dict(r, receipt=p)
    return errs


def identity(row):
    return {"view_id": row["view_id"], "dataset_hash": row["dataset_hash"], "fold": int(row["fold"]),
            "context_sha256": row["context_sha256"], "cohort": row.get("cohort")}


def l7b_pointer(row, shard):
    p = L7B / f"shard-{shard}" / row["view_id"]
    ok = (p / ".done").exists() and (p / "context.json").exists()
    return {"path": str(p / "context.json"), "sha256": sha_file(p / "context.json") if ok else None, "done": ok}


def write_record(d: Path, rec: dict):
    d.mkdir(parents=True, exist_ok=True)
    atomic(d / "context.json", jdump(rec))
    atomic(d / ".done", b"ok\n")


def build_l7a(c, values, specs, d: Path):
    """One L7a context: refit on outer-train rows, train-only min-max, inner folds, JTT1; files as the L7b build."""
    R, T = [int(i) for i in c["R"]], [int(i) for i in c["T"]]
    split = t1raw.prepare(values, specs, R, T)
    fitted = split["fitted"]
    teacher = trainonly.jtt1_bytes(fitted[R], fitted[T].reshape(-1, fitted.shape[1]), split["inner_labels"])
    enc_gz = gzip.compress(split["jtt1"], compresslevel=3, mtime=0)
    tea_gz = gzip.compress(teacher, compresslevel=3, mtime=0)
    a0 = metafeatures.from_raw(split["jtt1"])[0]
    d.mkdir(parents=True, exist_ok=True)
    atomic(d / "encoder.jtt1.gz", enc_gz)
    atomic(d / "teacher.jtt1.gz", tea_gz)
    atomic(d / "a0.f32", a0)
    return {"outer_train_indices_sha256": sha(json.dumps(R).encode()), "outer_test_indices_sha256": sha(json.dumps(T).encode()),
            "n_train": len(R), "n_test": len(T),
            "files": {"encoder.jtt1.gz": {"sha256": sha(enc_gz), "raw_sha256": sha(split["jtt1"]), "bytes": len(enc_gz)},
                      "teacher.jtt1.gz": {"sha256": sha(tea_gz), "raw_sha256": sha(teacher), "bytes": len(tea_gz)},
                      "a0.f32": {"sha256": sha(a0), "bytes": len(a0)}},
            "scaler_sha256": sha(trainonly.scaler_bytes(split)),
            "inner_labels_sha256": sha(np.asarray(split["inner_labels"], np.uint8).tobytes()),
            "inner_loads": split["inner_cv"]["loads"], "t1raw": split["t1raw"]}


def match_summary(st):
    if not st:
        return None
    keys = ("rows", "matched_rows", "unmatched_rows", "ambiguous_rows", "oversubscribed_rows", "raw_rows", "raw_value_classes",
            "duplicate_classes_claimed", "matched_missing_cells", "print_class", "residual", "tol_rel")
    return {k: st.get(k) for k in keys}


def run_dataset(h, ctx_rows, a, out, codes, src, cache, held, held_ref, leak, ferr, inv, counts, failed):
    shard_dir = out / f"shard-{a.shard}"
    base = {"schema": SCHEMA, "published": False, "code_sha256": codes}
    if h in held:
        for r in ctx_rows:
            rec = dict(base, **identity(r), values_mode="held_no_license", excluded=True,
                       reason="rights gate: no licence recorded; source never fetched or read",
                       rights_gate_receipt=held_ref, row_selection_leak=leak.get(r["view_id"]), utc_us=int(time.time() * 1e6))
            write_record(shard_dir / r["view_id"], rec)
            counts["held_no_license"] += 1
        return
    k = inv["meta_key"][h].rsplit("/", 1)[1]
    mb = (META_DIR / f"{k}.json").read_bytes()
    if sha(mb) != k:
        raise RuntimeError(f"{h}: meta sha mismatch")
    meta = json.loads(mb)
    fdir = RAW_DIR / h
    status, reason, detail, ds_rec, values, specs = "unmatched", None, "", None, None, None
    raw_ref = None
    if not (fdir / ".done").exists():
        e = ferr.get(h, {})
        reason, detail = "fetch_missing", f"{e.get('status')}: {e.get('error')} ({Path(e.get('receipt', '')).name})"
    else:
        fj = json.loads((fdir / "fetch.json").read_bytes())
        raw_path = Path(fj["file"])
        got = sha_file(raw_path)
        raw_ref = {"file": str(raw_path), "sha256": got, "bytes": raw_path.stat().st_size, "fetch_status": fj.get("status"),
                   "url": fj.get("url"), "byte_identical_to_original": got.startswith(h)}
        if got != fj.get("sha256"):
            reason, detail = "raw_custody_mismatch", f"{got} != fetch.json {fj.get('sha256')}"
        else:
            c0 = scale_check.resolve_context(ctx_rows[0], src, cache)
            data_sha = c0["refs"]["data_sha256"]
            t0 = time.time()
            m = t1raw.match_dataset(raw_path, meta, c0["A"])
            status, reason, detail = m["status"], m["reason"], m["detail"]
            if status == "matched":
                values, specs = m["values"], m["specs"]
            ds_rec = {"schema": "jope.dl-v2-l7a-dataset/v1", "dataset_hash": h, "status": status, "reason": reason, "detail": detail,
                      "data_sha256": data_sha, "t1_meta_sha256": k, "tool": (meta.get("acquisition") or {}).get("tool"),
                      "raw": raw_ref, "read": m.get("read"), "match": m.get("stats"),
                      "values_sha256": sha(np.ascontiguousarray(values, "<f8").tobytes()) if values is not None else None,
                      "elapsed_s": round(time.time() - t0, 2), "code_sha256": codes, "utc_us": int(time.time() * 1e6)}
    if ds_rec is None:
        ds_rec = {"schema": "jope.dl-v2-l7a-dataset/v1", "dataset_hash": h, "status": "unmatched", "reason": reason, "detail": detail,
                  "t1_meta_sha256": k, "tool": (meta.get("acquisition") or {}).get("tool"), "raw": raw_ref, "code_sha256": codes,
                  "utc_us": int(time.time() * 1e6)}
    (shard_dir / "_datasets").mkdir(parents=True, exist_ok=True)
    atomic(shard_dir / "_datasets" / f"{h}.json", jdump(ds_rec))
    for n, r in enumerate(ctx_rows):
        d = shard_dir / r["view_id"]
        if (d / ".done").exists():
            counts["already_done"] += 1
            continue
        if n % 25 == 0:
            guard()
        try:
            rec = dict(base, **identity(r), row_selection_leak=leak.get(r["view_id"]), t1_meta_sha256=k,
                       match=match_summary(ds_rec.get("match")))
            if status == "matched":
                c = scale_check.resolve_context(r, src, cache)
                if c["refs"]["data_sha256"] != ds_rec["data_sha256"]:     # matched rows belong to another data object
                    rec.update(values_mode="L7b-fallback", reason="data_object_differs_from_matched",
                               detail=c["refs"]["data_sha256"], l7b=l7b_pointer(r, a.shard), raw_source=raw_ref,
                               utc_us=int(time.time() * 1e6))
                    write_record(d, rec)
                    counts["L7b-fallback"] += 1
                    continue
                try:
                    built = build_l7a(c, values, specs, d)
                except (ValueError, t1raw.T1RawError) as e:      # IneligibleSplit, refusals: this context stays on L7b
                    rec.update(values_mode="L7b-fallback", reason="l7a_prepare_failed", detail=f"{type(e).__name__}: {e}"[:300],
                               l7b=l7b_pointer(r, a.shard), raw_source=raw_ref, utc_us=int(time.time() * 1e6))
                    write_record(d, rec)
                    counts["L7b-fallback"] += 1
                    continue
                rec.update(built, values_mode="L7a", raw_source=raw_ref, data_sha256=ds_rec["data_sha256"],
                           values_sha256=ds_rec["values_sha256"], utc_us=int(time.time() * 1e6))
                write_record(d, rec)
                counts["L7a"] += 1
            else:
                rec.update(values_mode="L7b-fallback", reason=reason, detail=detail, l7b=l7b_pointer(r, a.shard), raw_source=raw_ref,
                           utc_us=int(time.time() * 1e6))
                write_record(d, rec)
                counts["L7b-fallback"] += 1
        except Exception as e:  # recorded per context; the shard continues
            failed.append({"view_id": r["view_id"], "error": f"{type(e).__name__}: {e}"[:300]})


def guard():
    ok, reasons, _ = cpuguard.check()
    while not ok:
        print(json.dumps({"guard": "PAUSE", "reasons": reasons}), flush=True)
        time.sleep(15)
        ok, reasons, _ = cpuguard.check()


def build(a):
    cpuguard.set_threads(2)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    codes = code_shas()
    src = scale_check.Sources(load_index=False)
    cache = scale_check.DataCache(max_items=2)
    inv = json.loads(DATASETS_INV.read_text())
    held, held_ref = held_set()
    leak, _ = leak_map()
    ferr = fetch_errors()
    rows = sorted(src.included(), key=lambda r: (r["dataset_hash"], int(r["fold"])))
    mine = [r for r in rows if int(r["dataset_hash"][:8], 16) % a.of == a.shard]
    if a.datasets:                                    # smoke runs only; a full build leaves this unset
        keep = set(a.datasets.split(","))
        mine = [r for r in mine if r["dataset_hash"] in keep]
    groups = collections.OrderedDict()
    for r in mine:
        groups.setdefault(r["dataset_hash"], []).append(r)
    t0 = time.time()
    counts = collections.Counter()
    failed = []
    for n, (h, ctx_rows) in enumerate(groups.items()):
        if all((out / f"shard-{a.shard}" / r["view_id"] / ".done").exists() for r in ctx_rows):
            counts["already_done"] += len(ctx_rows)
            continue
        guard()
        try:
            run_dataset(h, ctx_rows, a, out, codes, src, cache, held, held_ref, leak, ferr, inv, counts, failed)
        except Exception as e:  # a dataset-level failure: every context not yet done is recorded as failed
            for r in ctx_rows:
                if not (out / f"shard-{a.shard}" / r["view_id"] / ".done").exists():
                    failed.append({"view_id": r["view_id"], "error": f"{type(e).__name__}: {e}"[:300]})
        if n % 20 == 0:
            print(json.dumps({"shard": a.shard, "datasets": n + 1, "of": len(groups), "counts": counts, "failed": len(failed),
                              "elapsed_s": round(time.time() - t0)}), flush=True)
    modes = collections.Counter()
    for r in mine:
        p = out / f"shard-{a.shard}" / r["view_id"] / "context.json"
        if (p.parent / ".done").exists():
            modes[json.loads(p.read_bytes())["values_mode"]] += 1
    rec = {"schema": "jope.dl-v2-l7a-shard/v1", "shard": a.shard, "of": a.of, "expected": len(mine), "datasets": len(groups),
           "built_now": {k: counts[k] for k in ("L7a", "L7b-fallback", "held_no_license")}, "already_done": counts["already_done"],
           "done_by_values_mode": dict(modes), "failed": failed, "complete": sum(modes.values()) == len(mine) and not failed,
           "elapsed_s": round(time.time() - t0, 1), "published": False, "code_sha256": codes, "utc_us": int(time.time() * 1e6)}
    atomic(out / f"shard-{a.shard}-of-{a.of}.receipt.json", (json.dumps(rec, indent=1, sort_keys=True) + "\n").encode())
    print(json.dumps({k: rec[k] for k in ("shard", "expected", "built_now", "already_done", "done_by_values_mode", "complete", "elapsed_s")}
                     | {"failed": len(failed)}))
    return 0 if rec["complete"] else 1


def summarize(a):
    """The build summary receipt: contexts by values_mode (must sum to the epoch's included contexts), datasets by match
    status with a reason histogram, the missing-fetch reconciliation, code sha256s, test and mutant results."""
    out = Path(a.out)
    src = scale_check.Sources(load_index=False)
    rows = src.included()
    by_view = {r["view_id"]: r for r in rows}
    held, held_ref = held_set()
    _, leak_ref = leak_map()
    modes, ds_modes, missing, reasons_ctx = collections.Counter(), collections.defaultdict(set), [], collections.Counter()
    for r in rows:
        shard = int(r["dataset_hash"][:8], 16) % a.of
        p = out / f"shard-{shard}" / r["view_id"]
        if not (p / ".done").exists():
            missing.append(r["view_id"]); continue
        c = json.loads((p / "context.json").read_bytes())
        if c["view_id"] != r["view_id"] or c["context_sha256"] != r["context_sha256"]:
            raise SystemExit(f"{r['view_id']}: context record identity mismatch")
        modes[c["values_mode"]] += 1
        ds_modes[r["dataset_hash"]].add(c["values_mode"])
        if c["values_mode"] == "L7b-fallback":
            reasons_ctx[c["reason"]] += 1
            if not (c.get("l7b") or {}).get("done"):
                raise SystemExit(f"{r['view_id']}: fallback without a done L7b context")
    ds_recs = {}
    for f in glob.glob(str(out / "shard-*" / "_datasets" / "*.json")):
        d = json.loads(Path(f).read_bytes())
        ds_recs[d["dataset_hash"]] = d
    plan = [json.loads(l) for l in PLAN.read_bytes().decode().splitlines() if l.strip()]
    plan_h = {p["dataset_hash"] for p in plan}
    epoch_h = {r["dataset_hash"] for r in rows}
    done = {h for h in plan_h if (RAW_DIR / h / ".done").exists()}
    ferr = fetch_errors()
    fetch_status = collections.Counter(json.loads((RAW_DIR / h / "fetch.json").read_bytes()).get("status") for h in done)
    not_done = sorted(plan_h - done)
    reconciliation = {"plan": {"path": str(PLAN), "sha256": sha_file(PLAN), "rows": len(plan)}, "fetched_done": len(done),
                      "fetch_json_status": dict(fetch_status), "missing": [], "held": held_ref,
                      "plan_and_held_disjoint": not (plan_h & set(held)), "plan_plus_held_equals_epoch_datasets": plan_h | set(held) == epoch_h,
                      "fetch_run_receipts": [{"path": p, "sha256": sha_file(Path(p))} for p in sorted(glob.glob(str(RAW_DIR / "fetch-run-*.json")))],
                      "licensed_run_log": {"path": str(RAW_DIR / "licensed-run.log"), "sha256": sha_file(RAW_DIR / "licensed-run.log"),
                                           "text": (RAW_DIR / "licensed-run.log").read_text().strip()[:500]}}
    for h in not_done:
        e = ferr.get(h, {})
        pr = [p for p in plan if p["dataset_hash"] == h][0]
        d = RAW_DIR / h
        reconciliation["missing"].append({
            "dataset_hash": h, "url": pr.get("url"), "host": pr.get("host"), "declared_bytes": pr.get("declared_bytes"),
            "status": e.get("status"), "error": e.get("error"), "receipt": e.get("receipt"),
            "attempts": "2 (l7a_fetch.py retries a transport error once; HTTP status or cap errors are not retried)",
            "dir_contents": sorted(os.listdir(d)) if d.exists() else None,
            "contexts": sum(1 for r in rows if r["dataset_hash"] == h),
            "disposition": "values_mode L7b-fallback, reason fetch_missing; not re-fetched (no network in this step)"})
    ds_status = collections.Counter()
    reason_hist = collections.Counter()
    tool_hist = collections.Counter()
    unmatched = []
    for h in sorted(plan_h):
        d = ds_recs.get(h)
        st = (d or {}).get("status", "missing_record")
        ds_status[st] += 1
        tool_hist[((d or {}).get("tool") or "?", st)] += 1
        if st != "matched":
            reason_hist[(d or {}).get("reason")] += 1
            unmatched.append({"dataset_hash": h, "reason": (d or {}).get("reason"), "detail": ((d or {}).get("detail") or "")[:160],
                              "tool": (d or {}).get("tool"), "byte_identical_raw": ((d or {}).get("raw") or {}).get("byte_identical_to_original"),
                              "unmatched_rows": ((d or {}).get("match") or {}).get("unmatched_rows"),
                              "ambiguous_rows": ((d or {}).get("match") or {}).get("ambiguous_rows"),
                              "oversubscribed_rows": ((d or {}).get("match") or {}).get("oversubscribed_rows"),
                              "blame": dict(list((((d or {}).get("match") or {}).get("blame_columns_first_unmatched") or {}).items())[:4]),
                              "nonnumeric_examples": {k: v[:2] for k, v in list((((d or {}).get("read") or {}).get("nonnumeric_examples") or {}).items())[:3]}})
    matched = [d for d in ds_recs.values() if d["status"] == "matched"]
    worst = max((d["match"]["residual"].get("max_dev_over_tol", 0.0) for d in matched), default=None)
    shard_receipts = []
    for p in sorted(glob.glob(str(out / f"shard-*-of-{a.of}.receipt.json"))):
        r = json.loads(Path(p).read_bytes())
        shard_receipts.append({"path": p, "sha256": sha_file(Path(p)), "expected": r["expected"], "complete": r["complete"],
                               "failed": len(r["failed"]), "done_by_values_mode": r["done_by_values_mode"]})
    tests = {"log": {"path": a.tests_log, "sha256": sha_file(Path(a.tests_log))}, "tail": Path(a.tests_log).read_text().strip().splitlines()[-3:]}
    mut = json.loads(Path(a.mutants).read_bytes())
    mutants = {"path": a.mutants, "sha256": sha_file(Path(a.mutants)), "mutants": mut["mutants"], "killed": mut["killed"],
               "invalid": mut["invalid"], "survived": mut["survived"], "suite": mut["suite"]}
    root = Path(__file__).resolve().parents[1]
    codes = code_shas() | {f: sha_file(root / f) for f in ("tests/test_t1raw.py", "tests/test_t1prep.py", "tests/test_trainonly.py")}
    total = sum(modes.values())
    doc = {"schema": "jope.dl-v2-l7a-build-summary/v1", "utc_us": int(time.time() * 1e6), "out": str(out), "of": a.of,
           "epoch": src.epoch_ref, "included_contexts": len(rows), "contexts_by_values_mode": dict(modes), "contexts_total": total,
           "sums_to_included": total == len(rows) and not missing, "contexts_missing": missing[:20],
           "fallback_contexts_by_reason": dict(reasons_ctx),
           "datasets": {"epoch": len(epoch_h), "licensed_plan": len(plan_h), "held_no_license": len(set(held)),
                        "by_match_status": dict(ds_status), "matched_tool_status": {f"{t}|{s}": n for (t, s), n in sorted(tool_hist.items())},
                        "unmatched_reason_histogram": dict(reason_hist.most_common()),
                        "mixed_mode_datasets": sorted(h for h, m in ds_modes.items() if len(m) > 1)},
           "match_rule": {"tol": "float32 '%.9g' bound t1prep.F32_PRINT_REL * span (x max(1, |scaled|)); decimal9 columns (T1 printed '%.9g' "
                                 "from float64, read from the T1 values) use 1e-8 * |raw - lo|; both + 8 * 2^-52 * magnitude",
                          "unique": "every A row matches exactly one raw value class; claims <= raw multiplicity",
                          "worst_matched_dev_over_tol": worst},
           "fetch_reconciliation": reconciliation, "row_selection_leak": leak_ref, "shard_receipts": shard_receipts,
           "tests": tests, "mutants": mutants, "code_sha256": codes, "unmatched_datasets": unmatched,
           "policy": {"s3_calls": 0, "network": False, "gpu": False, "published": False, "held_sources_read": False}}
    rp = Path(a.receipt) if a.receipt else RECEIPTS / f"l7a-build-summary-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    data = (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode()
    with open(rp, "xb") as f:
        f.write(data)
    print(json.dumps({"receipt": str(rp), "sha256": sha(data), "contexts_by_values_mode": dict(modes), "total": total,
                      "sums_to_included": doc["sums_to_included"], "datasets": doc["datasets"]["by_match_status"]}))
    return 0 if doc["sums_to_included"] else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int)
    ap.add_argument("--of", type=int, required=True)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--tests-log")
    ap.add_argument("--mutants")
    ap.add_argument("--receipt")
    ap.add_argument("--datasets", help="comma-separated dataset hashes (smoke runs only)")
    a = ap.parse_args(argv)
    if a.summarize:
        return summarize(a)
    if a.shard is None:
        ap.error("--shard is required for a build")
    return build(a)


if __name__ == "__main__":
    sys.exit(main())
