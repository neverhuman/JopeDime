"""REMAINING-COVERAGE bucket of the raw-universe rebuild: the 275 frozen "clean" contexts (55 datasets) whose T1 meta
shows an acquisition-time cap (acquisition-selection: cohort frozen, class clean, reduced=true). Owner rulings:
REMAINING-COVERAGE (owner-decisions-20261005T233942Z.json, delegated) and RIGHTS-RESEARCH (owner-decisions-
20261005T223750Z.json); the rebuild itself is ROWSEL-TRAINONLY + ROWSEL-SPLIT.

Same tool, rules and gates as tools/v2_rowsel_raw.py: this module only supplies the inputs. tools/v2_rowsel_raw.run_dataset
does the universe, pre-cap, split, per-fold selection, L7a values, both counterfactual gates and the dataset-atomic .done.
What differs: these datasets have no L7a match record, so the content match is run here for the first time
(t1raw.read_source + t1raw.match, refused unless every T1 row matches one raw value class one-to-one) and written as this
build's match receipt (_match/<h>.json), which the universe identity then pins.

Per dataset disposition (plan):
  rights_held     the rights gate holds it and RIGHTS-RESEARCH does not lift it: never fetched, never read (reason rights_held)
  research-only   RIGHTS-RESEARCH lifts the hold for research-only lift measurement: built; every context.json and the
                  dataset record carry rights="research-only" and rights_decision=<owner file>#sha256=<sha>. The raw file
                  is read from RUN/inputs/t1-raw-held-no-license (fetched earlier, recorded) or RUN/inputs/t1-raw-research-only
  licensed        built from RUN/inputs/t1-raw or RUN/inputs/t1-raw-licensed-refetch
A zip source with several table members is resolved by content: each member is tried and the dataset is accepted only
when exactly one member matches (recorded). Output: OUT/shard-0 (licensed), OUT/shard-1 (research-only).

Usage (inside the lane slice):
  v2_rowsel_raw_coverage.py plan      [--out DIR]
  v2_rowsel_raw_coverage.py build     [--out DIR] [--datasets h1,h2]
  v2_rowsel_raw_coverage.py summarize [--out DIR] --main-summary JSON --tests-log LOG --mutants JSON [--receipt PATH]
No S3 or GPU; nothing is published. Fetching is a separate, logged step (tools/l7a_fetch.py with the coverage fetch plans).
"""
import argparse
import collections
import copy
import glob
import importlib.util
import json
import time
import zipfile
from pathlib import Path

import numpy as np

from dltrain import cpuguard, rowsel_raw, scale_check, t1raw

_spec = importlib.util.spec_from_file_location("v2_rowsel_raw", Path(__file__).resolve().parent / "v2_rowsel_raw.py")
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)

RUN = tool.RUN
RECEIPTS = tool.RECEIPTS
OUT = RUN / "build-v2" / "ROWSEL-RAW-v1-coverage"
OUT_V2 = RUN / "build-v2" / "ROWSEL-RAW-v2-coverage"
RAW_DIRS = {"licensed": [RUN / "inputs" / "t1-raw", RUN / "inputs" / "t1-raw-licensed-refetch"],
            "research-only": [RUN / "inputs" / "t1-raw-held-no-license", RUN / "inputs" / "t1-raw-research-only"]}
RR_LIST = RUN / "tmp" / "rights-research-datasets.json"
RR_DECISION = Path("/home/ubuntu/scratch/jopedime-dltrain-audit-v1/claude/receipts/owner-decisions-20261005T223750Z.json")
RR_DECISION_SHA256 = "75217a2a50010bde04f27c4977ee9c3d462986d4ea0d9b6f340043895e983924"
COVERAGE_DECISION = Path("/home/ubuntu/scratch/jopedime-dltrain-audit-v1/claude/receipts/owner-decisions-20261005T233942Z.json")
CLASS = "acquisition_capped_frozen"
LABEL = ["clean"]
SHARD_OF_RIGHTS = {"licensed": 0, "research-only": 1}
MATCH_SCHEMA = "jope.dl-v2-rowsel-raw-match/v1"
PLAN_SCHEMA = "jope.dl-v2-rowsel-raw-coverage-plan/v1"
SUMMARY_SCHEMA = "jope.dl-v2-rowsel-raw-coverage-summary/v1"
CODE_FILES = tool.CODE_FILES + ("tools/v2_rowsel_raw_coverage.py",)


def code_shas():
    root = Path(__file__).resolve().parents[1]
    return {f: tool.sha((root / f).read_bytes()) for f in CODE_FILES}


# --------------------------------------------------------------------------------------------------- plan

def plan(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if tool.sha_file(RR_DECISION) != RR_DECISION_SHA256:
        raise SystemExit("RIGHTS-RESEARCH owner record drift")
    acq = json.loads(tool.ACQ.read_bytes())
    leak = {}
    for l in tool.LEAK.read_bytes().decode().splitlines():
        if l.strip():
            r = json.loads(l)
            leak[r["view_id"]] = r["row_selection_leak"]
    held, held_ref = tool.held_set()
    rr = set(json.loads(RR_LIST.read_bytes()))
    src = scale_check.Sources(load_index=False)
    by = collections.defaultdict(list)
    for r in src.included():
        by[r["dataset_hash"]].append(r)
    entries = []
    for h in sorted(acq):
        x = acq[h]
        if not (x["cohort"] == "frozen" and x["class"] == "clean" and x["reduced"]):
            continue
        ctx = sorted(by[h], key=lambda r: int(r["fold"]))
        labels = {json.dumps(leak.get(r["view_id"])) for r in ctx}
        if labels != {json.dumps(LABEL)}:
            raise SystemExit(f"{h}: epoch-0 labels {labels}")
        e = {"dataset_hash": h, "class": CLASS, "cohort": "frozen", "epoch0_labels": LABEL,
             "selection_markers": x["selection_markers"], "acq_pre_selection_rows": x["pre_selection_rows"],
             "epoch0": [{"view_id": c["view_id"], "context_sha256": c["context_sha256"], "fold": int(c["fold"])} for c in ctx]}
        if h in held and h not in rr:
            e.update(rights="held", disposition="not_built", reason="rights_held")
        else:
            rights = "research-only" if h in rr else "licensed"
            e.update(rights=rights, shard=SHARD_OF_RIGHTS[rights], disposition="build")
            if rights == "research-only":
                e["context_extra"] = {"rights": "research-only", "rights_decision": f"{RR_DECISION}#sha256={RR_DECISION_SHA256}"}
            dirs = [d for d in RAW_DIRS[rights] if (d / h / ".done").exists()]
            e["raw_dir"] = str(dirs[0]) if dirs else None
            e["raw_from_held_dir"] = bool(dirs) and dirs[0].name == "t1-raw-held-no-license"
            if not dirs:
                e.update(disposition="not_built", reason="raw_not_fetched")
        e["match_record"] = None
        entries.append(e)
    doc = {"schema": PLAN_SCHEMA, "class": CLASS, "datasets": entries,
           "counts": {"datasets": len(entries), "contexts": sum(len(e["epoch0"]) for e in entries),
                      "by_rights": dict(collections.Counter(e["rights"] for e in entries)),
                      "by_rights_contexts": {k: sum(len(e["epoch0"]) for e in entries if e["rights"] == k)
                                             for k in sorted({e["rights"] for e in entries})},
                      "by_disposition": dict(collections.Counter(f"{e['disposition']}:{e.get('reason')}" for e in entries))},
           "inputs": {"acquisition_selection": tool.ref(tool.ACQ), "row_selection_leak": tool.ref(tool.LEAK), "rights_gate": held_ref,
                      "rights_research_list": tool.ref(RR_LIST), "rights_research_decision": tool.ref(RR_DECISION),
                      "coverage_decision": tool.ref(COVERAGE_DECISION) if COVERAGE_DECISION.exists() else None},
           "epoch": src.epoch_ref}
    tool.stamp_split_version(doc, a.split_version)
    p = out / "plan.json"
    body = (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode()
    if p.exists() and p.read_bytes() != body:
        raise SystemExit(f"{p} exists with different content; refusing to replace a pinned plan")
    if not p.exists():
        tool.atomic(p, body)
    print(json.dumps({"plan": str(p), "sha256": tool.sha(body), "counts": doc["counts"]}))
    return 0


# --------------------------------------------------------------------------------------------------- inputs + first match

def _zip_tables(path: Path):
    with open(path, "rb") as f:
        if f.read(4) != b"PK\x03\x04":
            return None
    with zipfile.ZipFile(path) as zf:
        names = [i.filename for i in zf.infolist() if not i.is_dir()]
    return [n for n in names if n.lower().endswith((".csv", ".tsv", ".txt"))]


def _classify(st, read):
    """t1raw.match_dataset's reason codes for an unmatched result."""
    if st["unmatched_rows"]:
        nonnum = set(read["nonnumeric_cells"]) & set(st["blame_columns_first_unmatched"])
        return "rows_unmatched_nonnumeric_raw" if nonnum else (
            "rows_unmatched_target" if set(st["blame_columns_first_unmatched"]) == {"target"} else "rows_unmatched")
    return "rows_ambiguous" if st["ambiguous_rows"] else "rows_oversubscribed"


def _read_match(raw_path, meta, A):
    specs = t1raw.columns(meta, A.shape[1])
    got = t1raw.read_source(raw_path, meta, specs)
    m = t1raw.match(A, specs, got["values"])
    return specs, got, m


def first_match(raw_path: Path, meta, A):
    """read_source + match; a zip with several table members is tried member by member and accepted only when exactly
    one member matches. Returns (specs, read, match, member_trial) or raises RowselRawError with the t1raw reason."""
    try:
        return (*_read_match(raw_path, meta, A), None)
    except t1raw.T1RawError as err:
        if err.reason != "zip_member_ambiguous":
            raise rowsel_raw.RowselRawError(err.reason, err.detail)
    trial, hits, best_any = {}, [], []
    for member in _zip_tables(raw_path) or []:
        m2 = copy.deepcopy(meta)
        m2.setdefault("acquisition", {}).setdefault("download", {})["container"] = f"zip:{member}"
        try:
            specs, got, m = _read_match(raw_path, m2, A)
            trial[member] = {"matched": bool(m["matched"]), "matched_rows": m["stats"]["matched_rows"], "rows": m["stats"]["rows"]}
            (hits if m["matched"] else best_any).append((member, specs, got, m))
        except t1raw.T1RawError as err:
            trial[member] = {"matched": False, "reason": err.reason}
    if len(hits) > 1:
        raise rowsel_raw.RowselRawError("zip_member_not_unique_by_content", json.dumps(trial))
    rule = "every table member tried; accepted only when exactly one content-matches"
    if hits:
        member, specs, got, m = hits[0]
        return specs, got, m, {"rule": rule, "selected": member, "members": trial}
    best = max(best_any, key=lambda t: t[3]["stats"]["matched_rows"]) if best_any else None
    if best is None:
        raise rowsel_raw.RowselRawError("zip_no_readable_member", json.dumps(trial))
    member, specs, got, m = best                        # none matched: report the closest member's unmatched result
    return specs, got, m, {"rule": rule, "selected": None, "closest_unmatched": member, "members": trial}


def make_loader(out: Path, codes):
    def load(e, src, cache, inv, held):
        h = e["dataset_hash"]
        if e.get("rights") not in ("licensed", "research-only") or e.get("disposition") != "build":
            raise rowsel_raw.RowselRawError("rights_held")                       # before any read of its source
        k = inv["meta_key"][h].rsplit("/", 1)[1]
        meta_b = (tool.META_DIR / f"{k}.json").read_bytes()
        if tool.sha(meta_b) != k:
            raise rowsel_raw.RowselRawError("meta_sha_mismatch", k)
        meta = json.loads(meta_b)
        fdir = Path(e["raw_dir"]) / h
        if not (fdir / ".done").exists():
            raise rowsel_raw.RowselRawError("raw_not_fetched")
        fj = json.loads((fdir / "fetch.json").read_bytes())
        recorded = Path(fj["file"])
        raw_path = fdir / recorded.name           # held-dir sources were moved after the fetch; fetch.json keeps the old path
        if not raw_path.is_file() or raw_path.is_symlink() or "/" in recorded.name:
            raise rowsel_raw.RowselRawError("raw_not_in_dataset_dir", str(recorded))
        got_sha = tool.sha_file(raw_path)
        if got_sha != fj.get("sha256"):
            raise rowsel_raw.RowselRawError("raw_custody_mismatch", f"{got_sha} != fetch.json {fj.get('sha256')}")
        rows = sorted((r for r in src.included() if r["dataset_hash"] == h), key=lambda r: int(r["fold"]))
        if [(r["view_id"], r["context_sha256"]) for r in rows] != [(c["view_id"], c["context_sha256"]) for c in e["epoch0"]]:
            raise rowsel_raw.RowselRawError("epoch0_contexts_differ_from_plan")
        ctxs = [scale_check.resolve_context(r, src, cache) for r in rows]
        data = {c["refs"]["data_sha256"] for c in ctxs}
        if len(data) != 1:
            raise rowsel_raw.RowselRawError("epoch0_contexts_on_different_data_objects", json.dumps(sorted(data)))
        A = ctxs[0]["A"]
        raw_ref = {"file": str(raw_path), "sha256": got_sha, "bytes": raw_path.stat().st_size, "fetch_status": fj.get("status"),
                   "url": fj.get("url"), "byte_identical_to_original": got_sha.startswith(h), "raw_dir": str(Path(e["raw_dir"])),
                   "read_from_held_dir": Path(e["raw_dir"]).name == "t1-raw-held-no-license",
                   "fetch_json_file": str(recorded), "relocated": recorded.parent.resolve() != fdir.resolve()}
        specs, read, m, member_trial = first_match(raw_path, meta, A)
        st = m["stats"]
        values_sha = tool.sha(np.ascontiguousarray(t1raw.matched_values(m), "<f8").tobytes()) if m["matched"] else None
        mrec = {"schema": MATCH_SCHEMA, "dataset_hash": h, "status": "matched" if m["matched"] else "unmatched",
                "reason": None if m["matched"] else _classify(st, read["report"]), "data_sha256": data.pop(),
                "t1_meta_sha256": k, "raw": raw_ref, "read": read["report"], "match": st, "values_sha256": values_sha,
                "zip_member_trial": member_trial, "rights": e["rights"], "code_sha256": codes, "utc_us": int(time.time() * 1e6)}
        (out / "_match").mkdir(parents=True, exist_ok=True)
        mp = out / "_match" / f"{h}.json"
        body = tool.jdump(mrec)
        tool.atomic(mp, body)
        e["match_record"] = {"path": str(mp), "sha256": tool.sha(body)}
        if not m["matched"]:
            raise rowsel_raw.RowselRawError(mrec["reason"], json.dumps({x: st[x] for x in ("rows", "unmatched_rows", "ambiguous_rows",
                                                                                         "oversubscribed_rows", "raw_rows")}))
        return {"meta": meta, "meta_sha256": k, "raw_path": raw_path, "raw_ref": raw_ref, "record": mrec, "A": A,
                "rows": rows, "sealed": sum(1 for r in rows if r["context_sha256"] in src.sealed_ctx),
                "_values": read["values"], "_specs": specs}
    return load


def matcher(inp):
    """The match already ran in the loader (first match); hand its full raw values on, once."""
    return inp.pop("_values"), inp.pop("_specs"), {"values_sha256": inp["record"]["values_sha256"], "first_match": True,
                                                   "zip_member_trial": inp["record"]["zip_member_trial"]}


# --------------------------------------------------------------------------------------------------- build

def build(a):
    cpuguard.set_threads(2)
    out = Path(a.out)
    pl = json.loads((out / "plan.json").read_bytes())
    plan_sha = tool.sha_file(out / "plan.json")
    tool.plan_split_version(pl, a.split_version)
    codes = code_shas()
    src = scale_check.Sources(load_index=False)
    cache = scale_check.DataCache(max_items=2)
    inv = json.loads(tool.DATASETS_INV.read_text())
    guard = cpuguard.Guard(journal_path=out / "coverage.guard.jsonl")
    load = make_loader(out, codes)
    pending_datasets = [e for e in pl["datasets"] if e["disposition"] == "build"]
    if a.datasets:
        keep = set(a.datasets.split(","))
        pending_datasets = [e for e in pending_datasets if e["dataset_hash"] in keep]
    t0, c0 = time.time(), tool.cpu_s()
    results = []
    for e in pending_datasets:
        sd = out / f"shard-{e['shard']}"
        sd.mkdir(parents=True, exist_ok=True)
        if tool.dataset_done(sd, e["dataset_hash"], codes):
            results.append({"dataset_hash": e["dataset_hash"], "status": "already_done"})
            continue
        guard.wait()
        rec = tool.run_dataset(dict(e), sd, codes, src, cache, inv, set(), guard, loader=load, matcher=matcher, split_version=a.split_version)
        results.append({"dataset_hash": e["dataset_hash"], "rights": e["rights"], "status": rec["status"], "reason": rec.get("reason"),
                        "detail": (rec.get("detail") or "")[:500], "cpu_s": rec["cpu_s"]})
        print(json.dumps(results[-1]), flush=True)
    rec = {"schema": "jope.dl-v2-rowsel-raw-coverage-build/v1", "plan_sha256": plan_sha, "datasets_filter": a.datasets,
           "results": results, "counts": dict(collections.Counter(r["status"] for r in results)),
           "cpu_s": round(tool.cpu_s() - c0, 1), "wall_s": round(time.time() - t0, 1), "maxrss_mb": tool.maxrss_mb(),
           "guard": guard.summary(), "code_sha256": codes, "published": False, "utc_us": int(time.time() * 1e6)}
    name = "build.receipt.json" if not a.datasets else f"build.partial-{int(time.time())}.receipt.json"
    tool.atomic(out / name, (json.dumps(rec, indent=1, sort_keys=True) + "\n").encode())
    print(json.dumps({k: rec[k] for k in ("counts", "cpu_s", "wall_s", "maxrss_mb")}))
    return 0


# --------------------------------------------------------------------------------------------------- summary (all buckets)

def summarize(a):
    out = Path(a.out)
    pl = json.loads((out / "plan.json").read_bytes())
    codes = code_shas()
    main = json.loads(Path(a.main_summary).read_bytes())
    main_plan = json.loads((Path(a.main_out) / "plan.json").read_bytes())
    version = tool.plan_split_version(pl, a.split_version)
    if tool.plan_split_version(main_plan, version) != version or main.get("split_version", tool.SPLIT_V1) != version:
        raise SystemExit("main tree and coverage tree use different split versions")
    per_ds = []
    counts = collections.Counter()
    for e in pl["datasets"]:
        h, n0 = e["dataset_hash"], len(e["epoch0"])
        row = {"dataset_hash": h, "rights": e["rights"], "epoch0_contexts": n0, "epoch0_view_ids": [c["view_id"] for c in e["epoch0"]]}
        if e["disposition"] != "build":
            row.update(status="not_built", reason=e["reason"])
        else:
            sd = out / f"shard-{e['shard']}"
            p = sd / "_datasets" / f"{h}.json"
            if not p.exists():
                row.update(status="not_built", reason="no_dataset_record")
            else:
                r = json.loads(p.read_bytes())
                row.update(status=r["status"], reason=r.get("reason"), detail=(r.get("detail") or "")[:600],
                           record={"path": str(p), "sha256": tool.sha_file(p)}, raw=(r.get("rematch") or {}),
                           read_from_held_dir=e.get("raw_from_held_dir"), cpu_s=r.get("cpu_s"))
                mp = out / "_match" / f"{h}.json"
                if mp.exists():
                    mr = json.loads(mp.read_bytes())
                    row["match_record"] = {"path": str(mp), "sha256": tool.sha_file(mp), "status": mr["status"], "reason": mr["reason"],
                                           "rows": mr["match"]["rows"], "unmatched_rows": mr["match"]["unmatched_rows"],
                                           "raw_rows": mr["match"]["raw_rows"], "raw_sha256": mr["raw"]["sha256"],
                                           "raw_file": mr["raw"]["file"], "fetch_url": mr["raw"]["url"]}
                if r["status"] == "done":
                    if r["split"]["schema"] != version or r.get("split_version", tool.SPLIT_V1) != version:
                        raise SystemExit(f"{h}: split {r['split']['schema']} is not the plan's {version}")
                    if (not tool.code_known({k: v for k, v in r["code_sha256"].items() if k in tool.CODE_FILES}, tool.code_shas())
                            or r["code_sha256"].get("tools/v2_rowsel_raw_coverage.py") != codes["tools/v2_rowsel_raw_coverage.py"]):
                        raise SystemExit(f"{h}: built with unknown code")
                    cs = []
                    for c in r["contexts"]:
                        d = sd / c["view_id"]
                        cj = json.loads((d / "context.json").read_bytes())
                        if not (d / ".done").exists() or cj["rowsel"]["identity"] != c["identity"]:
                            raise SystemExit(f"{h}: context {c['view_id']} not done or identity differs")
                        if e["rights"] == "research-only" and (cj.get("rights"), cj.get("rights_decision")) != (
                                "research-only", e["context_extra"]["rights_decision"]):
                            raise SystemExit(f"{h}: research-only context without its rights fields")
                        if cj["row_selection_leak"] != LABEL:
                            raise SystemExit(f"{h}: label {cj['row_selection_leak']}")
                        for name, meta in cj["files"].items():
                            if tool.sha_file(d / name) != meta["sha256"]:
                                raise SystemExit(f"{h}: {name} hash mismatch")
                        cs.append(c["view_id"])
                    row["new_view_ids"] = cs
        counts[f"{row['rights']}|{row['status']}|{row.get('reason')}"] += 1
        counts[f"contexts|{row['rights']}|{row['status']}"] += n0
        per_ds.append(row)
    bucket = {"datasets": len(per_ds), "epoch0_contexts": sum(r["epoch0_contexts"] for r in per_ds),
              "rebuilt_datasets": sum(r["status"] == "done" for r in per_ds),
              "rebuilt_new_contexts": sum(len(r.get("new_view_ids", [])) for r in per_ds),
              "epoch0_contexts_replaced": sum(r["epoch0_contexts"] for r in per_ds if r["status"] == "done"),
              "not_rebuilt_epoch0_contexts_kept": sum(r["epoch0_contexts"] for r in per_ds if r["status"] != "done"),
              "by_rights_status_reason": dict(sorted(counts.items()))}
    acct = main_plan["accounting"]
    def left(cls):
        return {k[len("not_matched:"):]: v for k, v in acct[cls].items() if k.startswith("not_matched:")}
    buckets = {
        "frozen_all_rows_coreset": {"datasets": acct["all_rows_coreset"]["class_datasets"],
                                    "epoch0_contexts": acct["all_rows_coreset"]["class_contexts"],
                                    "rebuilt": main["counts_per_class"]["all_rows_coreset"],
                                    "unrebuildable_by_l7a_match_reason": left("all_rows_coreset"),
                                    "unrebuildable_disposition": "keeps its epoch-0 context, label all_rows_coreset"},
        "incoming_acquisition_capped_kmeans": {"datasets": acct["acquisition_capped"]["class_datasets"],
                                               "epoch0_contexts": acct["acquisition_capped"]["class_contexts"],
                                               "built_from_raw": main["counts_per_class"]["acquisition_capped"],
                                               "left_to_t1_fallback_by_l7a_match_reason": left("acquisition_capped"),
                                               "t1_fallback": "T1-universe rebuild (other lane, ROWSEL-T1-kmeans-v1) with label acquisition_cap"},
        "frozen_clean_acquisition_capped": bucket}
    for cls, key in (("all_rows_coreset", "frozen_all_rows_coreset"), ("acquisition_capped", "incoming_acquisition_capped_kmeans")):
        ds = []
        for e in main_plan["datasets"]:
            if e["class"] == cls:
                ds.append({"dataset_hash": e["dataset_hash"], "status": "done", "epoch0_contexts": len(e["epoch0"])})
        buckets[key]["datasets_rebuilt"] = ds
    recs = tool.l7a_records()
    acq = json.loads(tool.ACQ.read_bytes())
    for cls, key in (("all_rows_coreset", "frozen_all_rows_coreset"), ("acquisition_capped", "incoming_acquisition_capped_kmeans")):
        nb = []
        built = {d["dataset_hash"] for d in buckets[key]["datasets_rebuilt"]}
        for h, x in sorted(acq.items()):
            isc = (x["class"] == "all_rows_coreset") if cls == "all_rows_coreset" else (
                x["class"] == "kmeans" and x["cohort"] == "incoming" and x["reduced"])
            if isc and h not in built:
                r = recs.get(h, (None,))[0] or {}
                nb.append({"dataset_hash": h, "epoch0_contexts": x["contexts"], "reason": r.get("reason") or r.get("status") or "no_l7a_record",
                           "detail": (r.get("detail") or "")[:200]})
        buckets[key]["datasets_not_rebuilt"] = nb
    mut = json.loads(Path(a.mutants).read_bytes())
    doc = {"schema": SUMMARY_SCHEMA, "utc_us": int(time.time() * 1e6), "out": str(out), "split_version": version,
           "plan": {"path": str(out / "plan.json"), "sha256": tool.sha_file(out / "plan.json"), "counts": pl["counts"], "inputs": pl["inputs"]},
           "main_build_summary": tool.ref(a.main_summary), "buckets": buckets, "coverage_datasets": per_ds,
           "build_receipt": tool.ref(out / "build.receipt.json") if (out / "build.receipt.json").exists() else None,
           "fetch_runs": [tool.ref(p) for p in sorted(glob.glob(str(RUN / "inputs" / "t1-raw-research-only" / "fetch-run-*.json"))
                                                  + glob.glob(str(RUN / "inputs" / "t1-raw-licensed-refetch" / "fetch-run-*.json")))],
           "tests": {"log": tool.ref(a.tests_log), "tail": Path(a.tests_log).read_text().strip().splitlines()[-3:]},
           "mutants": {"path": a.mutants, "sha256": tool.sha_file(a.mutants), "totals": mut.get("totals")},
           "code_sha256": codes,
           "policy": {"s3_calls": 0, "gpu": False, "git": False, "published": False, "network": "read-only public GETs by tools/l7a_fetch.py only",
                      "rights_held_sources_read": False, "credential_files_read": False}}
    stem = "rowsel-raw-coverage-summary" if version == tool.SPLIT_V1 else "rowsel-raw-v2-coverage-summary"
    rp = Path(a.receipt) if a.receipt else RECEIPTS / "build" / f"{stem}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    data = (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode()
    with open(rp, "xb") as f:
        f.write(data)
    if tool.should_mark_complete(not any(r.get("reason") == "no_dataset_record" for r in per_ds), version):
        tool.write_complete(out, rp, {k: v for k, v in bucket.items() if k != "by_rights_status_reason"}, version)
    print(json.dumps({"receipt": str(rp), "sha256": tool.sha(data), "bucket": {k: v for k, v in bucket.items() if k != "by_rights_status_reason"},
                      "by": bucket["by_rights_status_reason"]}))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("plan", "build", "summarize"))
    ap.add_argument("--out")
    ap.add_argument("--split-version", choices=tuple(tool.SPLIT_VERSIONS), default=tool.SPLIT_V2,
                    help=f"default {tool.SPLIT_V2} (target-free); {tool.SPLIT_V1} only when passed explicitly")
    ap.add_argument("--main-out", help="the main tree whose plan/summary this coverage summary is read with")
    ap.add_argument("--datasets")
    ap.add_argument("--main-summary")
    ap.add_argument("--tests-log")
    ap.add_argument("--mutants")
    ap.add_argument("--receipt")
    a = ap.parse_args(argv)
    if a.out is None:
        a.out = str(OUT if a.split_version == tool.SPLIT_V1 else OUT_V2)
    if a.main_out is None:
        a.main_out = str(tool.default_out("l7a", a.split_version))
    return {"plan": plan, "build": build, "summarize": summarize}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
