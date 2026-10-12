#!/usr/bin/env python3
"""S1 post-selection outer-test scorer (ruling OUTER-TEST-SCORER, owner-decisions-20261006T105750Z.json, option (i)).

UNREVIEWED ADAPTER. No number produced here may be quoted as a headline or written as an HWM until this adapter lands in
a reviewed PR (the RAW-GROUP follow-up PR, neverhumanbot review). Every output record says so.

What it does, per context (only contexts whose S1 selection is final):
  (a) final.json (s1-prod/contexts/<ctx>/final.json) must exist with no write permission bits AND its sha256 must equal
      the `final` row of the production/research ledger; it is re-hashed after scoring and must be byte-identical.
      Inputs are exactly: the ORIGINAL teacher JTT1 (gz sha256, bytes and raw sha256 checked against the input list)
      and the frozen selected program final.json.selection.program (DSL sha256 == program_sha256 == selection sha).
  The pinned worker's legacy 7-fold `--fit` computes official_test = fit min/max scaling + model on ALL outer-train rows,
  predict the outer-test rows. It refuses k != 7, so the scorer feeds it a copy of the original JTT1 whose header k byte
  is 7 and whose n_train fold bytes are i % 7; every row byte is unchanged. The 7 relabelled inner-CV fits it also runs
  are DISCARDED: the worker writes its whole result to ONE rolling scratch file (outer-test-scratch/worker-output.json,
  replaced by the worker's own atomic rename on every call and finally overwritten by a synthetic fixture call); only
  allow-listed official_test scalars and hashes are copied into the stored record, which is asserted to contain no
  inner_cv product and no prediction/target arrays.
  (b) Outputs: create-only s1-prod/outer-test/scores/<ctx>.json (0444) + append-only s1-prod/outer-test/scores.jsonl.
      Nothing is written to the S1 ledgers, contexts/, the execution hosts or any reader's input path.
Runs on xbabe2 inside jopedime-dltrain.slice at low priority. No S3, Git, network or /tmp.

PAYLOAD-STRIP-PUBLISH (owner ruling 2026-10-06 06:52 MT, owner-decisions-20261006T125305Z.json): outer-test labels are
read ONLY from the scorer-only JTS1 artifact. This version (proposed replacement for s1-prod/code/outer_test_scorer.py,
sha256 efede484...) never opens the original teacher file: the train block comes from the stripped teacher payload
(n_test = 0) and the outer-test block from the scorer artifact, both located through create-only strip summaries
(tools/v2_strip_outer_test.py, schema jope.dl-v2-strip-outer-test-summary/v1, PASS) by (view_id, original teacher
raw_sha256 from the input list). dltrain.scorer_labels.reconstruct rebuilds the original JTT1 bytes exactly (it refuses
unless they hash to the artifact's original_raw_sha256, here pinned to the input list), so the worker input, and with it
every official_test number, is byte-identical to the original adapter's. This module is the only scorer-side reader of
the artifact (tests/test_payload_strip.py, static isolation). Everything else is unchanged.

usage: s1_outer_test_scorer.py [--out DIR] --strip-summary PATH#SHA256 [...] score [--jobs N] [--follow]
       s1_outer_test_scorer.py [--out DIR] --strip-summary PATH#SHA256 [...] mustfail CTX [CTX ...]
       --out defaults to s1-prod/outer-test (scratch: s1-prod/outer-test-scratch); with --out, scratch is DIR/scratch.
"""
import gzip
import hashlib
import json
import os
import stat
import struct
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dltrain import jtt1, scorer_labels  # noqa: E402 (PAYLOAD-STRIP-PUBLISH: the scorer-only artifact reader)

P = Path("/home/ubuntu/scratch/jopedime-dl-training-v1/receipts/trainonly-v2/s1-prod")
OUT = P / "outer-test"
SCRATCH = P / "outer-test-scratch"
WORKER = P / "binaries/bootstrap.s1main"
WORKER_SHA = "e3be938644722d14a1c1a0e82168518838c4d174cfe031076a339745c48870a8"
RULING = Path("/home/ubuntu/scratch/jopedime-dltrain-audit-v1/claude/receipts/owner-decisions-20261006T105750Z.json")
TAG = "unreviewed_adapter"
NOTICE = ("UNREVIEWED ADAPTER: not a headline, not an HWM, not a training/selection input; becomes citable only after the "
          "adapter lands in a reviewed PR (RAW-GROUP follow-up PR, neverhumanbot review)")
RELABEL = "header byte 16 := 7; fold byte i := i % 7 for i < n_train; every row byte unchanged"
KEEP_METRICS = ("rmse", "mae", "r2", "max_abs", "max_abs_error", "n", "rows", "mse", "nrmse")
TIMEOUT_S = 600


def sha(b):
    return hashlib.sha256(b).hexdigest()


def canonical(v):
    return json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def create(path, data, mode=0o444):
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def replace(path, data):
    tmp = path.with_name(path.name + ".pending")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def ledger_finals():
    out = {}
    for name in ("production", "research"):
        for line in (P / f"ledger-{name}.jsonl").read_bytes().splitlines():
            r = json.loads(line)
            if r.get("event") == "final":
                prev = out.get(r["context_sha256"])
                if prev and prev["final_sha256"] != r["final_sha256"]:
                    raise SystemExit(f"conflicting final rows for {r['context_sha256']}")
                out[r["context_sha256"]] = r
    return out


def pinned_final(ctx, finals):
    row = finals.get(ctx)
    if row is None:
        raise ValueError("no final row in the ledger")
    path = P / "contexts" / ctx / "final.json"
    st = path.stat()
    if st.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError("final.json is writable (not immutable)")
    b = path.read_bytes()
    if sha(b) != row["final_sha256"]:
        raise ValueError("final.json differs from its ledger final row")
    return b, json.loads(b), row


def relabel(raw):
    if raw[:4] != b"JTT1" or raw[16] != 2:
        raise ValueError("expected an S1 two-fold JTT1")
    nt = struct.unpack_from("<I", raw, 4)[0]
    out = bytearray(raw)
    out[16] = 7
    for i in range(nt):
        out[17 + i] = i % 7
    out = bytes(out)
    assert out[17 + nt:] == raw[17 + nt:] and out[:16] == raw[:16]
    return out


def flip_targets(raw):
    """Outer-test target cell i := (-1)**i * 1e6, written without decoding; all other bytes equal (must-fail input)."""
    nt, nq, p = struct.unpack_from("<III", raw, 4)
    row = (p + 1) * 8
    end = 17 + nt + nt * row
    out = bytearray(raw)
    for i in range(nq):
        struct.pack_into("<d", out, end + i * row + p * 8, (-1.0) ** i * 1e6)
    out = bytes(out)
    assert out[:end] == raw[:end]
    return out


STRIP_SUMMARY_SCHEMA = "jope.dl-v2-strip-outer-test-summary/v1"
TEACHER = "teacher.jtt1.gz"


def strip_index(specs):
    """(view_id, original teacher raw_sha256) -> stripped teacher payload and scorer artifact, from pinned PASS summaries."""
    idx = {}
    for spec in specs:
        path, want = spec.split("#", 1)
        b = Path(path).read_bytes()
        if sha(b) != want:
            raise SystemExit(f"{path}: strip summary sha256 differs from the pin")
        d = json.loads(b)
        if d.get("schema") != STRIP_SUMMARY_SCHEMA or d.get("verdict") != "PASS":
            raise SystemExit(f"{path}: not a PASS strip summary")
        for row in d["contexts"]:
            rec = (row.get("payloads") or {}).get(TEACHER)
            if rec is None:
                continue
            key = (row["view_id"], rec["original"]["raw_sha256"])
            ent = {"stripped_path": str(Path(d["dst"]) / row["shard"] / row["view_id"] / TEACHER), "stripped": rec["stripped"],
                   "scorer_path": str(Path(d["scorer_dst"]) / row["shard"] / row["view_id"] / rec["scorer"]["name"]),
                   "scorer_sha256": rec["scorer"]["sha256"], "summary": {"path": path, "sha256": want}}
            if idx.setdefault(key, ent)["scorer_sha256"] != ent["scorer_sha256"]:
                raise SystemExit(f"{key}: two strip summaries disagree")
    return idx


def teacher_from_strip(ctx, il_rows, index):
    """The original teacher JTT1 bytes, rebuilt from the stripped payload (train block) and the scorer-only artifact
    (outer-test block); the original teacher file is never opened."""
    r = il_rows[ctx]
    view = Path(r["context_dir"]).name
    ent = index.get((view, r["teacher"]["raw_sha256"]))
    if ent is None:
        raise ValueError("no stripped teacher / scorer artifact for this context")
    sgz = Path(ent["stripped_path"]).read_bytes()
    if sha(sgz) != ent["stripped"]["sha256"] or len(sgz) != ent["stripped"]["bytes"]:
        raise ValueError("stripped teacher identity")
    stripped = gzip.decompress(sgz)
    if sha(stripped) != ent["stripped"]["raw_sha256"] or jtt1.parse_header(stripped)["n_test"] != 0:
        raise ValueError("stripped teacher raw identity")
    agz = Path(ent["scorer_path"]).read_bytes()
    if sha(agz) != ent["scorer_sha256"]:
        raise ValueError("scorer artifact identity")
    art = gzip.decompress(agz)
    h, _ = scorer_labels.decode(art)
    if (h["payload"], h["view_id"], h["original_raw_sha256"], h["stripped_raw_sha256"]) != (TEACHER, view, r["teacher"]["raw_sha256"], sha(stripped)):
        raise ValueError("scorer artifact binding")
    try:
        raw = scorer_labels.reconstruct(stripped, art)
    except scorer_labels.ScorerError as e:
        raise ValueError(f"reconstruction: {e}") from None
    if sha(raw) != r["teacher"]["raw_sha256"] or len(raw) != r["teacher"]["raw_bytes"]:
        raise ValueError("teacher raw identity")
    src = {"stripped_teacher": {"path": ent["stripped_path"], "gz_sha256": sha(sgz), "raw_sha256": sha(stripped)},
           "scorer_artifact": {"path": ent["scorer_path"], "sha256": ent["scorer_sha256"]}, "strip_summary": ent["summary"],
           "rule": "original teacher JTT1 = scorer_labels.reconstruct(stripped teacher, scorer-only artifact); original file not opened"}
    return r, raw, src


def run_worker(scorer_input, program, slot=0):
    SCRATCH.mkdir(mode=0o700, exist_ok=True)
    sd = SCRATCH / f"slot-{slot}"
    sd.mkdir(mode=0o700, exist_ok=True)
    inp, cand, out, err = (sd / "scorer-input.jtt1", sd / "candidate.json", sd / "worker-output.json",
                           sd / "worker-stderr.log")
    replace(inp, scorer_input)
    replace(cand, canonical(program))
    before = out.stat().st_mtime_ns if out.exists() else None
    env = {"PATH": "/usr/bin:/bin", "RAYON_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "TMPDIR": str(sd),
           "HOME": str(sd)}
    t0 = time.time()
    try:
        with err.open("ab") as e:
            rc = subprocess.run([str(WORKER), "--fit", str(inp), str(cand), str(out)], stdout=subprocess.DEVNULL,
                                stderr=e, timeout=TIMEOUT_S, env=env).returncode
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}, round(time.time() - t0, 3)
    wall = round(time.time() - t0, 3)
    if rc != 0 or not out.exists() or out.stat().st_mtime_ns == before:
        return {"status": "failed", "worker_exit": rc, "detail": "worker failed before official_test (e.g. a relabelled CV fit errored)"}, wall
    doc = json.loads(out.read_bytes())
    if doc.get("schema") != "jope.s1-evaluation/v1" or doc.get("program", {}).get("program_sha256") != program["program_sha256"]:
        return {"status": "failed", "detail": "unexpected worker output"}, wall
    return doc.get("official_test") or {"status": "missing"}, wall


def summarize(ot):
    """Allow-listed official_test fields only; arrays become hashes."""
    keep = {"status": ot.get("status")}
    if ot.get("status") == "completed":
        for k in ("metrics", "baseline_metrics", "training_metrics"):
            keep[k] = {m: v for m, v in (ot.get(k) or {}).items() if not isinstance(v, (list, dict))}
        for k in ("train_rows", "score_rows", "model_sha256", "scaler_sha256", "input_scaling", "model_bytes",
                  "serialized_prediction_replay_exact", "kind"):
            keep[k] = ot.get(k)
        keep["predictions_sha256"] = sha(canonical(ot.get("predictions")))
        keep["targets_sha256"] = sha(canonical(ot.get("targets")))
    else:
        keep["error"] = str(ot.get("error") or ot.get("detail") or "")[:500]
    text = json.dumps(keep)
    assert "inner_cv" not in text and '"predictions":' not in text and '"targets":' not in text
    return keep


def flush_scratch(slots=1):
    """Overwrite every slot's rolling worker output with a synthetic fixture so no real relabelled CV fit stays on disk."""
    nt, nq, p = 70, 14, 2
    b = bytearray(b"JTT1" + struct.pack("<III", nt, nq, p) + bytes([7]) + bytes(i % 7 for i in range(nt)))
    for i in range(nt + nq):
        for v in (float(i), float(i % 5), 2.0 * i + 1.0):
            b += struct.pack("<d", v)
    dsl = "(pipe X (ridge))"
    prog = {"dsl": dsl, "program_sha256": sha(dsl.encode()), "birth_seq": 0, "parents": [], "operator": "grammar_random"}
    out = {}
    for k in range(slots):
        ot, _ = run_worker(bytes(b), prog, k)
        out[f"slot-{k}"] = {"status": ot.get("status"),
                            "worker_output_sha256_after": sha((SCRATCH / f"slot-{k}" / "worker-output.json").read_bytes())}
    return {"synthetic_fixture": "70x2 linear fixture, (pipe X (ridge))", "slots": out}


def score_one(ctx, finals, il_rows, index, scorer_input_fn=None, label="score", slot=0):
    fb, final, frow = pinned_final(ctx, finals)
    if final.get("outcome") != "selection_reached":
        return None
    sel = final["selection"]
    prog = sel["program"]
    if not (prog and sha(prog["dsl"].encode()) == prog["program_sha256"] == sel["program_sha256"]):
        raise ValueError("selected program identity")
    r, raw, src = teacher_from_strip(ctx, il_rows, index)
    base = relabel(raw)
    scorer_input = scorer_input_fn(base) if scorer_input_fn else base
    ot, wall = run_worker(scorer_input, prog, slot)
    fb2 = (P / "contexts" / ctx / "final.json").read_bytes()
    if fb2 != fb:
        raise ValueError("final.json changed during scoring")
    return {
        "schema": "jope.dl-s1-outer-test-score/v1", "tag": TAG, "notice": NOTICE,
        "ruling": {"key": "OUTER-TEST-SCORER", "file": str(RULING), "sha256": sha(RULING.read_bytes())},
        "context_sha256": ctx, "view_id": final["view_id"], "set": final["set"], "use": final["use"], "cut_id": final["cut_id"],
        "final_json": {"path": str(P / "contexts" / ctx / "final.json"), "sha256": sha(fb), "ledger_final_row_matches": True,
                       "byte_identical_after_scoring": True},
        "selected_program_sha256": sel["program_sha256"],
        "teacher": {"path": r["teacher"]["path"], "gz_sha256": r["teacher"]["gz_sha256"], "raw_sha256": r["teacher"]["raw_sha256"],
                    "read_via": src},
        "scorer_input": {"sha256": sha(scorer_input), "rule": RELABEL, "variant": label},
        "worker": {"path": str(WORKER), "sha256": WORKER_SHA, "mode": "--fit (legacy 7-fold; official_test only kept)"},
        "official_test": summarize(ot),
        "discarded": {"relabelled_inner_cv_fits": 7, "persisted": False,
                      "how": "worker output goes to one rolling scratch file replaced on every call and overwritten by a synthetic fixture after each batch; only allow-listed official_test scalars/hashes are stored"},
        "wall_s": wall, "code_sha256": sha(Path(__file__).read_bytes()), "utc": utc(),
    }


def configure(out=None):
    global OUT, SCRATCH
    if out is not None:
        OUT, SCRATCH = Path(out), Path(out) / "scratch"


def main():
    if sha(WORKER.read_bytes()) != WORKER_SHA:
        raise SystemExit("worker identity")
    argv, specs = list(sys.argv[1:]), []
    while argv and argv[0] in ("--out", "--strip-summary"):
        opt, val = argv.pop(0), argv.pop(0)
        if opt == "--out":
            configure(val)
        else:
            specs.append(val)
    if not specs:
        raise SystemExit("--strip-summary PATH#SHA256 is required (outer-test labels come only from the scorer artifact)")
    index = strip_index(specs)
    sys.argv = [sys.argv[0], *argv]
    cmd = sys.argv[1]
    finals = ledger_finals()
    il_rows = {r["context_sha256"]: r for r in json.loads((P / "input-list.json").read_bytes())["admitted"]}
    (OUT / "scores").mkdir(parents=True, exist_ok=True)
    (OUT / "checks").mkdir(parents=True, exist_ok=True)
    if cmd == "score":
        # usage: score [--jobs N] [--follow]  (follow: re-read the ledger every 60 s until every assigned context is final
        # and every final selection is scored)
        import concurrent.futures
        import threading
        args = sys.argv[2:]
        jobs = int(args[args.index("--jobs") + 1]) if "--jobs" in args else 1
        follow = "--follow" in args
        lock = threading.Lock()
        superseded = set()
        for f in P.glob("superseded-pending-*.json"):
            superseded |= {c["context_sha256"] for c in json.loads(f.read_bytes())["contexts"]}
        free = list(range(jobs))
        total = 0

        def work(ctx, finals):
            with lock:
                slot = free.pop()
            try:
                try:
                    rec = score_one(ctx, finals, il_rows, index, slot=slot)
                except (ValueError, OSError) as e:
                    rec = {"schema": "jope.dl-s1-outer-test-score/v1", "tag": TAG, "notice": NOTICE, "context_sha256": ctx,
                           "official_test": {"status": "refused", "error": str(e)[:300]}, "utc": utc()}
                if rec is None:
                    return 0
                if ctx in superseded:
                    rec["superseded_pending"] = True
                body = canonical(rec)
                create(OUT / "scores" / f"{ctx}.json", body)
                with lock:
                    with (OUT / "scores.jsonl").open("a") as f:
                        f.write(json.dumps({"context_sha256": ctx, "record_sha256": sha(body), "tag": TAG, "utc": rec["utc"],
                                            "status": rec["official_test"]["status"]}, sort_keys=True) + "\n")
                        f.flush()
                        os.fsync(f.fileno())
                return 1
            finally:
                with lock:
                    free.append(slot)
        while True:
            finals = ledger_finals()
            todo = [c for c in sorted(finals) if finals[c]["outcome"] == "selection_reached"
                    and not (OUT / "scores" / f"{c}.json").exists()]
            if todo:
                with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
                    total += sum(pool.map(lambda c: work(c, finals), todo))
            assigned = set()
            for name in ("production", "research"):
                for line in (P / f"ledger-{name}.jsonl").read_bytes().splitlines():
                    r = json.loads(line)
                    if r.get("event") == "assign":
                        assigned.add(r["context_sha256"])
            if not follow or (assigned <= set(finals) and not todo):
                break
            time.sleep(60)
        fl = flush_scratch(jobs)
        print(json.dumps({"scored": total, "flush": fl}))
    elif cmd == "mustfail":
        results = []
        for ctx in sys.argv[2:]:
            a = score_one(ctx, finals, il_rows, index, None, "orig")
            b = score_one(ctx, finals, il_rows, index, flip_targets, "outer_test_targets_flipped")
            oa, ob = a["official_test"], b["official_test"]
            res = {"context_sha256": ctx,
                   "final_json_sha256": [a["final_json"]["sha256"], b["final_json"]["sha256"]],
                   "final_json_byte_identical": a["final_json"]["sha256"] == b["final_json"]["sha256"] == finals[ctx]["final_sha256"],
                   "scorer_inputs_differ": a["scorer_input"]["sha256"] != b["scorer_input"]["sha256"],
                   "official_test_rmse": [oa.get("metrics", {}).get("rmse"), ob.get("metrics", {}).get("rmse")],
                   "official_test_changed": oa.get("metrics") != ob.get("metrics") or oa.get("targets_sha256") != ob.get("targets_sha256"),
                   "fit_side_identical": {k: oa.get(k) == ob.get(k) for k in ("model_sha256", "scaler_sha256", "training_metrics",
                                                                              "predictions_sha256")},
                   "no_inner_cv_in_records": all("inner_cv" not in json.dumps(x["official_test"]) for x in (a, b))}
            res["pass"] = (res["final_json_byte_identical"] and res["scorer_inputs_differ"] and res["official_test_changed"]
                           and all(res["fit_side_identical"].values()) and res["no_inner_cv_in_records"])
            for x, tag in ((a, "orig"), (b, "flip")):
                create(OUT / "checks" / f"{ctx}.{tag}.json", canonical(x))
            results.append(res)
        fl = flush_scratch()
        doc = {"schema": "jope.dl-s1-outer-test-mustfail/v1", "tag": TAG, "notice": NOTICE, "results": results, "flush": fl,
               "verdict": "PASS" if results and all(r["pass"] for r in results) else "FAIL",
               "code_sha256": sha(Path(__file__).read_bytes()), "utc": utc()}
        body = canonical(doc)
        create(OUT / "checks" / f"mustfail-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json", body)
        print(json.dumps(doc, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
