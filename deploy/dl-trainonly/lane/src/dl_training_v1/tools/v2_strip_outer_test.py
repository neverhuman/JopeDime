"""PAYLOAD-STRIP-PUBLISH: create-only publish copies of v2 build trees whose encoder/teacher JTT1 payloads hold no outer-test rows.

Owner ruling (ENG:watcher delegated, 2026-10-06 06:52 MT; owner-decisions-20261006T125305Z.json): flip-test-only is not
accepted for publication; nothing that carries outer-test targets goes to S3. Each published payload is the JTT1 with the
outer-test block DROPPED:

    stripped = b"JTT1" + <IIIB n_train, 0, p, inner_folds> + the same inner labels + the same train block

It is a valid JTT1 (jtt1.parse_header admits n_test == 0); jtt1.train_view and jtt1.train_rows_sha256 read exactly the same
bytes, so every encoding built through train_view is unchanged. The dropped rows (features and target, byte for byte) move to
a scorer-only JTS1 artifact (dltrain/scorer_labels.py), written to a SEPARATE scorer tree that is local only and never
published. Per context the copy's context.json gets `files` entries for the stripped payloads and a `publication_strip`
record (original and stripped raw/gz sha256, train_rows_sha256 equality, scorer artifact sha256); context_sha256 (the build
identity) and code_sha256 (the object key stem) are kept.

Modes (lane slice, offline; the source tree is only read):
  strip  --src TREE --dst NEW_TREE --scorer-dst NEW_SCORER_TREE [--receipt PATH] [--views V1,V2]
         copies every shard-*/<view_id>/ context, strips its row-level payloads, re-verifies every written byte, scans the
         copy, and writes a create-only summary receipt (default RUN/receipts/trainonly-v2/build/strip-outer-test-<dst>-summary-<utc>.json)
  scan   --tree TREE [--scorer SCORER_TREE] [--views ...] [--receipt PATH]
         fails if any encoder/teacher payload has n_test > 0, any payload row beyond the recorded n_train carries a target
         whose exact float64 bytes are a target of the scorer artifact, or any scorer artifact sits inside the tree
  pilot  --out NEW_DIR --tree FAMILY=PATH[:schema][@GPU_TREE] ... [--per-tree 3] [--estimate PATH ...]
         strips a few contexts per family into NEW_DIR and proves train_rows equality, unchanged encodings, the scorer
         round-trip and the publisher gate (stripped admitted, original refused); estimates the full strip from the pilot
"""
import argparse
import collections
import glob
import gzip
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from dltrain import jtt1, quarantine, scorer_labels

RUN = Path("/home/ubuntu/scratch/jopedime-dl-training-v1")
ROW_LEVEL = ("encoder.jtt1.gz", "teacher.jtt1.gz")
CONTROL = ("context.json", ".done")
RECORD_SCHEMA = "jope.dl-v2-publication-strip/v1"
SUMMARY_SCHEMA = "jope.dl-v2-strip-outer-test-summary/v1"
SCAN_SCHEMA = "jope.dl-v2-strip-outer-test-scan-summary/v1"
PILOT_SCHEMA = "jope.dl-v2-strip-outer-test-pilot-summary/v1"
RULING = {"name": "PAYLOAD-STRIP-PUBLISH (ENG:watcher delegated, 2026-10-06 06:52 MT)",
          "file": "/home/ubuntu/scratch/jopedime-dltrain-audit-v1/claude/receipts/owner-decisions-20261006T125305Z.json",
          "sha256": "240e0af5521c03f95157c447916d61f3e49dffa2e564a72bc097f811df957baf"}
RULE = ("published encoder/teacher payloads are JTT1 with the outer-test block dropped (n_test=0; same labels and train block); "
        "the outer-test rows live only in a local scorer-only JTS1 artifact read by dltrain/scorer_labels.py")
ROOT = Path(__file__).resolve().parents[1]


class StripRefused(Exception):
    def __init__(self, reason, detail=""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason, self.detail = reason, detail


def sha(b) -> str:
    return hashlib.sha256(b).hexdigest()


def code_identity(*extra) -> dict:
    return {f: sha((ROOT / f).read_bytes()) for f in ("tools/v2_strip_outer_test.py", "dltrain/scorer_labels.py", "dltrain/jtt1.py",
                                                      "dltrain/quarantine.py", *extra)}


def cut_of(ctx):
    """The context's HWM cut identity (tools/v2_rowsel_raw_manifest.cut_of): the re-split cut block, or for raw v2 contexts
    "trainonly-split-v2:" + rowsel.split.sha256; None otherwise."""
    c = ctx.get("cut") or {}
    if str(c.get("cut_id", "")).startswith(("trainonly-split-v2:", "trainonly-split-v2g:")):
        return c["cut_id"]
    sp = (ctx.get("rowsel") or {}).get("split") or {}
    if sp.get("schema") == "rowsel-raw-split-v2" and len(str(sp.get("sha256", ""))) == 64:
        return "trainonly-split-v2:" + sp["sha256"]
    return None


def recorded_train_rows(ctx, name):
    """The builder's own record of jtt1.train_rows_sha256 for this payload, where it kept one (raw rowsel and clean builds)."""
    stem = name.split(".", 1)[0]
    for path in (("rowsel", "counterfactual", "values", "train_side"), ("counterfactual", "train_side")):
        d = ctx
        for k in path:
            d = d.get(k) if isinstance(d, dict) else None
        if isinstance(d, dict) and isinstance(d.get(f"{stem}_train_rows"), str):
            return d[f"{stem}_train_rows"], "/".join(path) + f"/{stem}_train_rows"
    return None, None


def strip_payload(raw: bytes, expect_train_rows=None) -> dict:
    """Drop the outer-test block of one JTT1. Refuses unless the stripped train bytes, train_rows_sha256 and train_view are
    identical to the original's (and to `expect_train_rows`, the builder's record, when given)."""
    h = jtt1.parse_header(raw)
    _, stop = jtt1.train_range(h)
    stripped = jtt1.MAGIC + jtt1.HEADER.pack(h["n_train"], 0, h["p"], h["inner_folds"]) + bytes(raw[jtt1.HEADER_BYTES:stop])
    block = bytes(raw[stop:])
    hs = jtt1.parse_header(stripped)
    if hs != dict(h, n_test=0) or len(block) != 8 * (h["p"] + 1) * h["n_test"]:
        raise StripRefused("shape", f"{hs} vs {h}")
    trs, trs_s = jtt1.train_rows_sha256(raw), jtt1.train_rows_sha256(stripped)
    if trs != trs_s or (expect_train_rows is not None and trs != expect_train_rows):
        raise StripRefused("train_rows_mismatch", f"original {trs[:12]} stripped {trs_s[:12]} recorded {str(expect_train_rows)[:12]}")
    a, b = jtt1.train_view(raw), jtt1.train_view(stripped)
    if a[0] != dict(b[0], n_test=h["n_test"]) or any(x.dtype != y.dtype or x.tobytes() != y.tobytes() for x, y in zip(a[1:], b[1:])):
        raise StripRefused("train_view_differs")
    return {"header": h, "stripped": stripped, "test_block": block, "train_rows_sha256": trs}


def _create(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as f:          # create-only
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.chmod(path, 0o444)


def _inside(a: Path, b: Path) -> bool:
    a, b = a.resolve(), b.resolve()
    return a == b or b in a.parents


def plan_context(d: Path, tree_label: str = "") -> dict:
    """Read and verify one source context, strip its payloads in memory, and return every output (nothing is written)."""
    if d.is_symlink() or not d.is_dir():
        raise StripRefused("not_a_directory", str(d))
    if not (d / ".done").is_file() or (d / ".done").is_symlink():
        raise StripRefused("not_done")
    cj = (d / "context.json").read_bytes()
    ctx = json.loads(cj)
    if ctx.get("view_id") != d.name:
        raise StripRefused("view_id_mismatch", f"{ctx.get('view_id')!r} != {d.name!r}")
    if "publication_strip" in ctx:
        raise StripRefused("already_stripped")
    files = ctx.get("files")
    if not isinstance(files, dict) or not files:
        raise StripRefused("context_field", "files")
    entries = sorted(p.name for p in d.iterdir())
    for name in entries:
        p = d / name
        if p.is_symlink() or not p.is_file():
            raise StripRefused("not_a_regular_file", name)
        if name not in files and name not in CONTROL:
            raise StripRefused("unlisted_file", name)
    out, new_files, record = {}, json.loads(json.dumps(files)), {}
    for name in sorted(files):
        body = (d / name).read_bytes()
        if sha(body) != files[name].get("sha256") or len(body) != files[name].get("bytes"):
            raise StripRefused("hash_mismatch", name)
        if name not in ROW_LEVEL:
            out[name] = body
            continue
        raw = gzip.decompress(body)
        if sha(raw) != files[name].get("raw_sha256"):
            raise StripRefused("raw_sha256_mismatch", name)
        ref, ref_path = recorded_train_rows(ctx, name)
        r = strip_payload(raw, ref)
        h = r["header"]
        stripped_gz = scorer_labels.gz(r["stripped"])
        if gzip.decompress(stripped_gz) != r["stripped"]:
            raise StripRefused("gzip_roundtrip", name)
        art = scorer_labels.encode({"payload": name, "view_id": d.name, "cut_id": cut_of(ctx), "dataset_hash": ctx.get("dataset_hash"),
                                    "fold": ctx.get("fold"), "n_train": h["n_train"], "n_test": h["n_test"], "p": h["p"],
                                    "inner_folds": h["inner_folds"], "original_raw_sha256": sha(raw), "original_gz_sha256": sha(body),
                                    "stripped_raw_sha256": sha(r["stripped"]), "stripped_gz_sha256": sha(stripped_gz),
                                    "train_rows_sha256": r["train_rows_sha256"]}, r["test_block"])
        if scorer_labels.reconstruct(r["stripped"], art) != raw:
            raise StripRefused("reconstruction_differs", name)
        art_gz = scorer_labels.gz(art)
        aname = scorer_labels.artifact_name(name)
        out[name] = stripped_gz
        out["<scorer>/" + aname] = art_gz
        new_files[name] = {"bytes": len(stripped_gz), "raw_sha256": sha(r["stripped"]), "sha256": sha(stripped_gz)}
        record[name] = {"original": {"sha256": sha(body), "bytes": len(body), "raw_sha256": sha(raw)},
                        "stripped": dict(new_files[name]), "n_train": h["n_train"], "n_test_original": h["n_test"],
                        "n_test_payload": 0, "p": h["p"], "train_rows_sha256": r["train_rows_sha256"],
                        "train_rows_sha256_equal": True, "train_view_identical": True, "reconstruction_exact": True,
                        "train_rows_recorded_at": ref_path,
                        "scorer": {"name": aname, "sha256": sha(art_gz), "bytes": len(art_gz), "raw_sha256": sha(art),
                                   "local_only": True}}
    new_ctx = dict(ctx, files=new_files)
    if record:
        new_ctx["publication_strip"] = {"schema": RECORD_SCHEMA, "ruling": RULING["name"], "ruling_sha256": RULING["sha256"],
                                        "rule": RULE, "cut_id": cut_of(ctx), "source_tree": tree_label,
                                        "key_scheme": quarantine.key_scheme(),
                                        "source_context_json_sha256": sha(cj), "files": record,
                                        "code_sha256": code_identity()}
    nl = "\n" if cj.endswith(b"\n") else ""
    out["context.json"] = (json.dumps(new_ctx, sort_keys=True) + nl).encode() if record else cj
    out[".done"] = (d / ".done").read_bytes()
    return {"view_id": d.name, "shard": d.parent.name, "outputs": out, "record": record, "source_context_json_sha256": sha(cj),
            "dataset_hash": ctx.get("dataset_hash"), "schema": ctx.get("schema")}


def contexts(tree: Path, views=None):
    ds = sorted(Path(p).parent for p in glob.glob(f"{tree}/shard-*/*/context.json"))
    if views is not None:
        ds = [d for d in ds if d.name in views]
    return ds


def strip_tree(src: Path, dst: Path, scorer_dst: Path, views=None) -> dict:
    src, dst, scorer_dst = Path(src), Path(dst), Path(scorer_dst)
    for a, b in ((dst, src), (src, dst), (scorer_dst, src), (src, scorer_dst), (scorer_dst, dst), (dst, scorer_dst)):
        if _inside(a, b):
            raise StripRefused("tree_overlap", f"{a} is inside {b}")
    if dst.exists() or scorer_dst.exists():
        raise StripRefused("exists", "create-only: --dst and --scorer-dst must not exist")
    t0, c0 = time.time(), time.process_time()
    ds = contexts(src, views)
    non_context = sorted(str(p.relative_to(src)) for s in sorted(src.glob("shard-*")) if s.is_dir() for p in s.iterdir() if not (p / "context.json").is_file())
    if views is not None and len(ds) != len(set(views)):
        raise StripRefused("views_missing", ",".join(sorted(set(views) - {d.name for d in ds}))[:300])
    plans, refused = [], []
    for d in ds:
        try:
            plans.append(plan_context(d, str(src)))
        except (StripRefused, jtt1.JTTError, scorer_labels.ScorerError, OSError, ValueError) as err:
            refused.append({"view_id": d.name, "reason": getattr(err, "reason", type(err).__name__), "detail": str(err)[:300]})
    doc = {"src": str(src), "dst": str(dst), "scorer_dst": str(scorer_dst), "contexts_in_src": len(ds), "non_context_entries": non_context,
           "refused": refused}
    if refused or not plans:
        doc["verdict"] = "FAIL"
        doc["stopped"] = "refused contexts; nothing was written" if refused else "no contexts"
        return doc
    expected = {}
    for p in plans:
        for name, data in p["outputs"].items():
            if name.startswith("<scorer>/"):
                path = scorer_dst / p["shard"] / p["view_id"] / name.split("/", 1)[1]
            else:
                path = dst / p["shard"] / p["view_id"] / name
            _create(path, data)
            expected[path] = sha(data)
    actual = {Path(r) / f for root in (dst, scorer_dst) for r, _, fs in os.walk(root) for f in fs}   # re-verify every byte
    bad = sorted(str(p) for p in actual ^ set(expected)) + sorted(str(p) for p, h in expected.items() if sha(p.read_bytes()) != h)
    rows = [{"view_id": p["view_id"], "shard": p["shard"], "schema": p["schema"], "source_context_json_sha256": p["source_context_json_sha256"],
             "context_json_sha256": sha(p["outputs"]["context.json"]), "payloads": p["record"]} for p in plans]
    rl = [f for r in rows for f in r["payloads"].values()]
    doc.update({"contexts_written": len(plans), "contexts_with_row_level": sum(1 for r in rows if r["payloads"]),
                "payloads_stripped": len(rl), "reverify": {"files": len(expected), "mismatched_or_unexpected": bad},
                "bytes": {"row_level_original_gz": sum(f["original"]["bytes"] for f in rl),
                          "row_level_stripped_gz": sum(f["stripped"]["bytes"] for f in rl),
                          "scorer_gz": sum(f["scorer"]["bytes"] for f in rl),
                          "dst_total": sum(p.stat().st_size for p in expected if _inside(p, dst)),
                          "scorer_total": sum(p.stat().st_size for p in expected if _inside(p, scorer_dst))},
                "contexts": rows})
    doc["scan"] = scan_tree(dst, scorer_dst)
    doc["cpu_s"], doc["wall_s"] = round(time.process_time() - c0, 3), round(time.time() - t0, 3)
    doc["verdict"] = "PASS" if not bad and doc["scan"]["verdict"] == "PASS" else "FAIL"
    return doc


def scan_tree(tree: Path, scorer=None, views=None) -> dict:
    """Must-fail scan of a publish tree. Findings: payload_unreadable, n_test_positive, n_train_differs_from_record,
    outer_test_target_in_payload (exact float64 bytes of the target column of rows beyond the recorded n_train against the
    scorer artifact's target column), scorer_shape_differs, scorer_artifact_in_tree."""
    tree = Path(tree)
    findings, checked, hits_total = [], 0, 0
    for p in sorted(tree.rglob("*")):
        if p.is_file() and (p.name.endswith(scorer_labels.SUFFIX) or ".jts1" in p.name):
            findings.append({"path": str(p.relative_to(tree)), "finding": "scorer_artifact_in_tree"})
    for d in contexts(tree, views):
        ctx = json.loads((d / "context.json").read_bytes())
        rec = (ctx.get("publication_strip") or {}).get("files") or {}
        for p in sorted(d.iterdir()):
            if p.name.endswith(".gz") and p.name not in ROW_LEVEL:
                try:
                    if gzip.decompress(p.read_bytes())[:4] == scorer_labels.MAGIC:
                        findings.append({"view_id": d.name, "file": p.name, "finding": "scorer_artifact_in_tree"})
                except (OSError, EOFError):
                    pass
        for name in ROW_LEVEL:
            if not (d / name).exists() and name not in (ctx.get("files") or {}):
                continue
            checked += 1
            try:
                raw = gzip.decompress((d / name).read_bytes())
                h = jtt1.parse_header(raw)
            except (OSError, EOFError, jtt1.JTTError) as err:
                findings.append({"view_id": d.name, "file": name, "finding": "payload_unreadable", "detail": str(err)[:200]})
                continue
            if h["n_test"] != 0:
                findings.append({"view_id": d.name, "file": name, "finding": "n_test_positive", "n_test": h["n_test"]})
            n_bind = (rec.get(name) or {}).get("n_train", h["n_train"])
            if n_bind != h["n_train"]:
                findings.append({"view_id": d.name, "file": name, "finding": "n_train_differs_from_record",
                                 "header": h["n_train"], "record": n_bind})
            art = Path(scorer) / d.parent.name / d.name / scorer_labels.artifact_name(name) if scorer is not None else None
            if art is not None and art.is_file():
                sraw = gzip.decompress(art.read_bytes())
                sh, _ = scorer_labels.decode(sraw)
                if sh["p"] != h["p"] or sh["view_id"] != d.name:
                    findings.append({"view_id": d.name, "file": name, "finding": "scorer_shape_differs"})
                    continue
                targets = scorer_labels.target_bytes(sraw)
                w, base = 8 * (h["p"] + 1), jtt1.HEADER_BYTES + h["n_train"]
                hits = sum(1 for i in range(min(n_bind, h["n_train"] + h["n_test"]), h["n_train"] + h["n_test"])
                           if bytes(raw[base + i * w + w - 8:base + i * w + w]) in targets)
                if hits:
                    hits_total += hits
                    findings.append({"view_id": d.name, "file": name, "finding": "outer_test_target_in_payload", "rows": hits})
    by = dict(collections.Counter(f["finding"] for f in findings))
    return {"tree": str(tree), "scorer": str(scorer) if scorer is not None else None, "payloads_checked": checked,
            "findings_by_kind": by, "findings": findings[:200], "target_hits": hits_total,
            "verdict": "PASS" if not findings and checked else "FAIL"}


def _write_receipt(path: Path, doc: dict) -> dict:
    body = (json.dumps(doc, sort_keys=True, indent=1) + "\n").encode()
    _create(Path(path), body)
    return {"path": str(path), "sha256": sha(body)}


def _stamp():
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


# ------------------------------------------------------------------------------------------------------- pilot
def _publisher():
    spec = importlib.util.spec_from_file_location("v2_publish", ROOT / "tools" / "v2_publish.py")
    vp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vp)
    return vp


def _gpu_twin(gpu_tree, view_id):
    if not gpu_tree:
        return None
    hit = sorted(glob.glob(f"{gpu_tree}/shard-*/{view_id}/context.json"))
    return json.loads(Path(hit[0]).read_bytes()) if hit else None


def _verify_pilot_context(vp, src_dir: Path, dst_dir: Path, scorer_dir: Path, gpu_tree, ledger_keys) -> dict:
    from dltrain import metafeatures
    sctx = json.loads((src_dir / "context.json").read_bytes())
    dctx = json.loads((dst_dir / "context.json").read_bytes())
    out = {"view_id": src_dir.name, "schema": sctx.get("schema"), "payloads": {}}
    ok = True
    for name in ROW_LEVEL:
        if name not in sctx["files"]:
            continue
        o_gz, s_gz = (src_dir / name).read_bytes(), (dst_dir / name).read_bytes()
        o, s = gzip.decompress(o_gz), gzip.decompress(s_gz)
        ho, hs = jtt1.parse_header(o), jtt1.parse_header(s)
        tvo, tvs = jtt1.train_view(o), jtt1.train_view(s)
        aname = scorer_labels.artifact_name(name)
        rec = dctx["publication_strip"]["files"][name]
        sh, Xt, yt = scorer_labels.load(scorer_dir / aname, expect={"artifact_sha256": rec["scorer"]["sha256"], "view_id": src_dir.name,
                                                                    "cut_id": cut_of(sctx), "stripped_raw_sha256": sha(s),
                                                                    "original_raw_sha256": sha(o)})
        _, stop = jtt1.train_range(ho)
        test = np.frombuffer(o[stop:], dtype="<f8").reshape(ho["n_test"], ho["p"] + 1)
        r = {"n_train": ho["n_train"], "n_test_original": ho["n_test"], "n_test_stripped": hs["n_test"], "p": ho["p"],
             "ctx_n_train_matches_header": sctx.get("n_train") == ho["n_train"],
             "train_rows_sha256_equal": jtt1.train_rows_sha256(o) == jtt1.train_rows_sha256(s) == rec["train_rows_sha256"],
             "train_view_bytes_identical": all(a.tobytes() == b.tobytes() for a, b in zip(tvo[1:], tvs[1:])),
             "train_bytes_prefix_identical": s[jtt1.HEADER_BYTES:] == o[jtt1.HEADER_BYTES:stop],
             "scorer_roundtrip": Xt.tobytes() == np.ascontiguousarray(test[:, :-1]).tobytes() and yt.tobytes() == np.ascontiguousarray(test[:, -1]).tobytes()
                                 and sh["n_test"] == ho["n_test"],
             "scorer_reconstructs_original": sha(scorer_labels.reconstruct(s, gzip.decompress((scorer_dir / aname).read_bytes()))) == sha(o),
             "scorer_refused_by_jtt1": _refused(lambda: jtt1.parse_header(gzip.decompress((scorer_dir / aname).read_bytes()))),
             "gate_admits_stripped": _gate(vp, name, s_gz) is None, "gate_refuses_original": _gate(vp, name, o_gz) == "outer_test_rows",
             "scorer_sha256": rec["scorer"]["sha256"], "stripped_gz_sha256": sha(s_gz), "original_gz_sha256": sha(o_gz)}
        if name == "encoder.jtt1.gz":
            if "a0.f32" in sctx["files"]:
                a0 = (src_dir / "a0.f32").read_bytes()
                r["a0_from_stripped_equals_built"] = metafeatures.from_raw(s)[0] == a0 == (dst_dir / "a0.f32").read_bytes()
            twin = _gpu_twin(gpu_tree, src_dir.name)
            if twin is not None:
                inp = (twin.get("gpu") or {}).get("input") or {}
                X, y = tvs[2], tvs[3]
                r["gpu_binding_train_block_equal"] = inp.get("binding", {}).get("train_block_sha256") == \
                    sha(np.ascontiguousarray(np.column_stack([X, y])).astype("<f8").tobytes())
                r["gpu_input_is_original_encoder"] = inp.get("encoder_jtt1_sha256") == sha(o_gz)
        ok = ok and all(v for k, v in r.items() if isinstance(v, bool))
        out["payloads"][name] = r
    unchanged = [f for f in sctx["files"] if f not in ROW_LEVEL]
    out["other_files_byte_identical"] = all((src_dir / f).read_bytes() == (dst_dir / f).read_bytes() for f in unchanged)
    out["context_identity_kept"] = all(dctx.get(k) == sctx.get(k) for k in ("context_sha256", "code_sha256", "view_id", "n_train", "n_test"))
    try:
        vp.scan_context(dst_dir)
        out["publisher_scan_stripped"] = "admitted"
    except vp.Refused as err:
        out["publisher_scan_stripped"] = err.reason
    try:
        vp.scan_context(src_dir)
        out["publisher_scan_original"] = "admitted"
    except vp.Refused as err:
        out["publisher_scan_original"] = err.reason
    research = sctx.get("rights") == "research-only"
    out["publisher_expected"] = ("rights_row_data", "rights_row_data") if research else ("admitted", "outer_test_rows")
    out["publisher_ok"] = (out["publisher_scan_stripped"], out["publisher_scan_original"]) == out["publisher_expected"]
    if not research:
        plan = vp.scan_context(dst_dir)
        out["keys_already_in_publish_ledger"] = sum(1 for o in plan["objects"] if o["key"] in ledger_keys or o["receipt_key"] in ledger_keys)
    out["ok"] = ok and out["other_files_byte_identical"] and out["context_identity_kept"] and out["publisher_ok"] \
        and not out.get("keys_already_in_publish_ledger")
    return out


def _refused(fn) -> bool:
    try:
        fn()
    except (jtt1.JTTError, ValueError):
        return True
    return False


def _gate(vp, name, body):
    try:
        vp.assert_no_outer_test(name, body)
    except vp.Refused as err:
        return err.reason
    return None


def pilot(out: Path, trees: list, per_tree: int, estimate: list) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    vp = _publisher()
    ledger_keys = set()
    if vp.DEFAULT_PUBLISH_LEDGER.exists():
        for line in vp.DEFAULT_PUBLISH_LEDGER.read_bytes().splitlines():
            r = json.loads(line)
            if r.get("event") == "verified":
                ledger_keys.add(r["key"])
    fams, per_family = [], {}
    for spec in trees:
        fam, rest = spec.split("=", 1)
        rest, gpu = (rest.split("@", 1) + [None])[:2]
        path, schema = (rest.split(":", 1) + [None])[:2]
        ds = contexts(Path(path))
        if schema:
            ds = [d for d in ds if json.loads((d / "context.json").read_bytes()).get("schema") == schema]
        ds = [d for d in ds if any((d / n).exists() for n in ROW_LEVEL)]
        if not ds:
            raise StripRefused("pilot_empty", spec)
        pick = sorted({ds[0], ds[len(ds) // 2], ds[-1]} if per_tree == 3 else set(ds[:per_tree]))
        views = {d.name for d in pick}
        c0 = time.process_time()
        st = strip_tree(Path(path), out / fam / "publish", out / fam / "scorer", views=views)
        cpu = time.process_time() - c0
        checks = [_verify_pilot_context(vp, d, out / fam / "publish" / d.parent.name / d.name, out / fam / "scorer" / d.parent.name / d.name,
                                        gpu, ledger_keys) for d in pick]
        control = scan_tree(Path(path), out / fam / "scorer", views=views)    # must-fail: the unstripped originals
        per_family[fam] = {"tree": path, "schema_filter": schema, "gpu_tree": gpu, "contexts_in_tree": len(ds), "views": sorted(views),
                           "strip_verdict": st["verdict"], "strip_refused": st["refused"], "strip_scan": st.get("scan"),
                           "bytes": st.get("bytes"), "cpu_s": round(cpu, 3), "checks": checks,
                           "unstripped_control_scan": {k: control[k] for k in ("verdict", "findings_by_kind", "target_hits", "payloads_checked")},
                           "ok": st["verdict"] == "PASS" and all(c["ok"] for c in checks) and control["verdict"] == "FAIL"
                           and control["findings_by_kind"].get("n_test_positive") == control["payloads_checked"]
                           and control["target_hits"] > 0}
        fams.append(fam)
    # estimate the full strip from the pilot: cpu per original gz byte and output/input ratios, over every listed tree
    o = sum(f["bytes"]["row_level_original_gz"] for f in per_family.values() if f.get("bytes"))
    ratio_s = sum(f["bytes"]["row_level_stripped_gz"] for f in per_family.values() if f.get("bytes")) / o
    ratio_c = sum(f["bytes"]["scorer_gz"] for f in per_family.values() if f.get("bytes")) / o
    cpu_per_byte = sum(f["cpu_s"] for f in per_family.values()) / o
    est = []
    for t in estimate:
        ds = contexts(Path(t))
        rl = sum((d / n).stat().st_size for d in ds for n in ROW_LEVEL if (d / n).exists())
        other = sum(p.stat().st_size for d in ds for p in d.iterdir() if p.name not in ROW_LEVEL)
        est.append({"tree": t, "contexts": len(ds), "row_level_gz_bytes": rl, "other_bytes_copied": other,
                    "est_stripped_gz_bytes": int(rl * ratio_s), "est_scorer_bytes": int(rl * ratio_c),
                    "est_new_bytes": int(other + rl * (ratio_s + ratio_c)), "est_cpu_s": round(rl * cpu_per_byte, 1)})
    tot = {k: sum(e[k] for e in est) for k in ("contexts", "row_level_gz_bytes", "other_bytes_copied", "est_new_bytes", "est_cpu_s")}
    return {"families": per_family, "family_order": fams, "estimate": {"basis": {"cpu_s_per_original_gz_byte": cpu_per_byte,
                                                                                    "stripped_over_original": ratio_s, "scorer_over_original": ratio_c},
                                                                          "trees": est, "total": tot},
            "verdict": "PASS" if all(f["ok"] for f in per_family.values()) else "FAIL"}


# -------------------------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="v2_strip_outer_test.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("strip")
    s.add_argument("--src", required=True)
    s.add_argument("--dst", required=True)
    s.add_argument("--scorer-dst", required=True)
    s.add_argument("--views", default=None)
    s.add_argument("--receipt", default=None)
    c = sub.add_parser("scan")
    c.add_argument("--tree", required=True)
    c.add_argument("--scorer", default=None)
    c.add_argument("--views", default=None)
    c.add_argument("--receipt", default=None)
    p = sub.add_parser("pilot")
    p.add_argument("--out", required=True)
    p.add_argument("--tree", action="append", required=True, help="FAMILY=PATH[:schema][@GPU_TREE]")
    p.add_argument("--per-tree", type=int, default=3)
    p.add_argument("--estimate", action="append", default=[])
    a = ap.parse_args(argv)
    views = set(a.views.split(",")) if getattr(a, "views", None) else None
    t0 = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    argv_rec = ["v2_strip_outer_test.py", *(sys.argv[1:] if argv is None else argv)]
    if a.mode == "strip":
        receipt = Path(a.receipt) if a.receipt else RUN / "receipts/trainonly-v2/build" / f"strip-outer-test-{Path(a.dst).name}-summary-{_stamp()}.json"
        if receipt.exists():
            ap.error(f"{receipt} exists (create-only)")
        try:
            doc = strip_tree(Path(a.src), Path(a.dst), Path(a.scorer_dst), views)
        except StripRefused as err:
            doc = {"verdict": "FAIL", "stopped": str(err)}
        doc |= {"schema": SUMMARY_SCHEMA, "ruling": RULING, "rule": RULE, "argv": argv_rec, "utc_start": t0, "code_sha256": code_identity()}
    elif a.mode == "scan":
        doc = scan_tree(Path(a.tree), Path(a.scorer) if a.scorer else None, views) | {"schema": SCAN_SCHEMA, "argv": argv_rec,
                                                                                     "utc_start": t0, "code_sha256": code_identity()}
        receipt = Path(a.receipt) if a.receipt else None
    else:
        doc = pilot(Path(a.out), a.tree, a.per_tree, a.estimate) | {"schema": PILOT_SCHEMA, "ruling": RULING, "rule": RULE, "argv": argv_rec,
                                                                    "utc_start": t0, "code_sha256": code_identity("tools/v2_publish.py", "dltrain/metafeatures.py")}
        receipt = Path(a.out) / "pilot-summary.json"
    doc["utc_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    res = _write_receipt(receipt, doc) if receipt is not None else {}
    print(json.dumps({"verdict": doc["verdict"], **res, **({"stopped": doc["stopped"]} if "stopped" in doc else {})}, sort_keys=True))
    return 0 if doc["verdict"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
