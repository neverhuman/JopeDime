"""Offline proofs for dltrain.families (P2 build, verify and publish; spec algorithm steps 2-5).

A synthetic World writes every input in the layout and naming of dltrain.family_pin (alias-sources.json, inputs/,
registry snapshot, metas, sealed references) plus the external pinned files. No test touches SQLite, S3 or the network;
publish runs against an in-memory fake bucket that applies the real lane guard (s3io.Guard.check_params). Every write
goes under TMPDIR (RUN/tmp).

Covered (spec 'Unit tests'): each identity mapped once; transitive aliases; deterministic output under shuffled input
order; collections, ingest, origin, aliases and column_order never union; sealed members never printed (canary scan of
every output, the receipt and captured stdout); null-safe keys; stop columns and the circuit breaker; id stability and
the post-freeze merge STOP; frozen_fold0; decision application rules; rights union; exposure Unknown/Unseen rules; no
forbidden imports; read-only SQLite. Plus: sealed bases and cross-check STOPs, provisional seals, curated aliases need
value confirmation, seed asserts, re-pin and policy drift STOPs, review rows and recommendations, the CLI and publish.
"""
import ast
import contextlib
import csv
import gzip
import hashlib
import io
import json
import os
import random
import sqlite3
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.exceptions import ClientError
from botocore.response import StreamingBody

from dltrain import common, s3io
from dltrain import families as fam
from dltrain import family_keys as fk
from dltrain import family_pin as fp

TMP_ROOT = os.environ.get("TMPDIR") or ""
if not TMP_ROOT or not os.path.isdir(TMP_ROOT) or os.path.realpath(TMP_ROOT).startswith("/tmp"):
    raise RuntimeError("TMPDIR must name RUN/tmp (never /tmp); run the tests inside the lane slice")
FIX = "jope.dl-p2-core-fixture/v1"
ROWS = 80

CANARIES = {
    "sealed_display": "CANARY-SEALED-DISPLAY-NAME",
    "sealed_url": "https://canary-sealed.example.net/files/CANARY-SEALED-URL.csv",
    "alias_display": "CANARY-ALIAS-DISPLAY-NAME",
    "mate_display": "CANARY-MATE-DISPLAY-NAME",
    "nearseal_display": "CANARY-NEARSEAL-DISPLAY",
    "auditmate_display": "CANARY-AUDITMATE-DISPLAY",
    "lb_family_t": "CANARY-LB-FAMILY-META-TEST",
    "lb_family_v": "CANARY-LB-FAMILY-META-VAL",
    "lb_x_key": "rdatasets:CanaryXPkg/CanaryXItem",
    "lb_x_record": "CANARY-X-RECORD-NAME",
    "lb_x_display": "CANARY-X-CATALOG-DISPLAY",
    "audit_family": "CANARY-AUDIT-FAMILY",
    "related_family": "CANARY-RELATED-LB-FAMILY",
    "withdrawn_family": "rdatasets-canarywithdrawn-canaryitem",
    "legacy_sealed": "canary-legacy-family",
    "outmate_display": "CANARY-OUTMATE-DISPLAY",
    "legacy_out": "canary-out-legacy-family",
}


def H(tag: str) -> str:
    return hashlib.sha256((FIX + ":" + tag).encode()).hexdigest()


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def jb(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, indent=1).encode()


def put(p: Path, data: bytes) -> bytes:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return data


def mat(seed: int, n: int = ROWS, cols: int = 3) -> np.ndarray:
    """Features plus a last target column; 6-decimal values in [0, 1), so tables never collide by chance."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 10 ** 6, size=(n, cols + 1)) / 10 ** 6


def csv_gz(m: np.ndarray) -> bytes:
    body = "\n".join(",".join(f"{v:.9g}" for v in row) for row in m) + "\n"
    return gzip.compress(body.encode(), mtime=0)


def parquet_bytes(cols: dict) -> bytes:
    buf = io.BytesIO()
    pq.write_table(pa.table(cols), buf)
    return buf.getvalue()


class World:
    """A miniature P2 world. Datasets are declared with frozen()/incoming(); write() materializes the inputs and an
    alias-sources.json the way family_pin writes it, and returns the config path."""

    def __init__(self, root):
        self.root = Path(root)
        self.out = self.root / "families"
        self.ds = {}
        self.ba = {}
        self.classic = []
        self.seed_sources = []
        self.fit_train, self.fit_val, self.catalog_only = [], [], {}
        self.target_records = []
        self.split_excluded = []
        self.native_extra = []
        self.lb_extra = {}
        self.audit_mates = []
        self.withdrawn = [CANARIES["withdrawn_family"]]
        self.shuffle = None
        self.v1_meta_test_override = None
        self.partition_change = False
        self.seed_tamper = None
        # the sealed reference XOUT outside the epoch (review-fix tests vary these; defaults = the original world)
        self.x_source_key = CANARIES["lb_x_key"]
        self.x_evidence = ["https://vincentarelbundock.github.io/Rdatasets/doc/CanaryXPkg/CanaryXItem.html"]
        self.x_record_source_key = CANARIES["lb_x_key"]
        self.x_catalog_prov = {"platform": "rdatasets", "dataset_id": "CanaryXPkg/CanaryXItem"}
        self.x_names = ["xa_col", "xb_col", "xc_col"]
        self.x_target = "xt"
        self.content_unavailable = None          # None: [VOUT]
        self.spec_path, self.spec_text = None, None   # a pinned spec file (e.g. the P2 survey) under the receipts root

    # ------------------------------------------------------------------ declarations
    def frozen(self, name, **kw):
        self.ds[name] = dict(name=name, cohort="frozen", **kw)

    def incoming(self, name, **kw):
        self.ds[name] = dict(name=name, cohort="incoming", **kw)

    def ident(self, name):
        can = H("canonical:" + name)
        h, s = can[:16], H("sample:" + name)
        return h, can, s, fp.identity_key(h, can, s)

    def item(self, name):
        return "item:" + self.ident(name)[3]

    def h(self, name):
        return self.ident(name)[0]

    # ------------------------------------------------------------------ materialization
    def write(self) -> Path:
        R, out = self.root, self.out
        inputs = []

        def add(name, role, path, **extra):
            raw = Path(path).read_bytes()
            inputs.append({"name": name, "role": role, "path": str(path), "sha256": sha(raw), "bytes": len(raw), **extra})

        def add_local(name, role, rel, data, **extra):
            put(out / rel, data)
            e = {"name": name, "role": role, "local_path": rel, "local_sha256": sha(data), "local_bytes": len(data)}
            e.setdefault("sha256", sha(data))
            e.setdefault("bytes", len(data))
            e.update(extra)
            inputs.append(e)

        rows, readiness, catalog, items, snap_ctx, legacy = [], [], [], [], [], []
        metas_by_h = {}
        for name in sorted(self.ds):
            d = self.ds[name]
            h, can, s, iid = self.ident(name)
            data = d.get("data")
            data = mat(int(H("seed:" + name)[:8], 16)) if data is None else data
            dbytes = csv_gz(data)
            dpath = R / "objects" / f"{sha(dbytes)}.csv.gz"
            put(dpath, dbytes)
            ncols = data.shape[1] - 1
            fnames = d.get("names") or [f"{name.lower()}_col{j}_{'abcdefgh'[j % 8]}" for j in range(ncols)]
            meta = {"feature_columns": fnames, "target_column": d.get("target", f"{name.lower()}_target"),
                    "original_headers": fnames + ["t"], "output_headers": fnames + [d.get("target", f"{name.lower()}_target")],
                    "dropped_feature_columns": [], "feature_scaling": {"min": [0.0] * ncols, "max": [1.0] * ncols}}
            meta.update(d.get("meta_extra") or {})
            meta_blob = H("metablob:" + name)
            mb = common.canonical(meta)
            add_local(f"meta:{meta_blob}", "meta", f"inputs/meta/{meta_blob}.json", mb, sha256=meta_blob, bytes=len(mb) + 7,
                      datasets=[h])
            metas_by_h[h] = meta_blob
            prov = dict(d.get("prov") or {"platform": "examplehub", "dataset_id": f"{name.lower()}-id",
                                          "url": f"https://data.{name.lower()}-src.org/{name.lower()}.csv"})
            prov.setdefault("license", {"status": "unknown"})
            row = {"dataset_hash": h, "id": h, "canonical_sha256": can, "display_name": d.get("display", f"{name} display"),
                   "provenance": prov, "files": {"meta": {"sha256": meta_blob, "bytes": len(mb) + 7}},
                   "collections": ["registry-real-v1"], "aliases": [], "origin": "real", "schema": "jope.dataset-catalog/v1",
                   "ingest": {"source": "quant", "tool_version": "x", "relpath": f"{name}.csv"},
                   "target": {"index": -1, "name": "y", "type": "numeric", "column_order_sha256": "31770f25" + "0" * 56},
                   "shape": {"rows": ROWS}, "split": {"seed": 1}, "task": "regression"}
            row.update(d.get("row_extra") or {})
            d["_row"] = row
            folds = d.get("folds", (0, 1, 2))
            status = d.get("status", "included")
            for fold in folds:
                ctx = H(f"ctx:{name}:{fold}")
                prep = jb({"view_id": f"{h}__r0__f{fold}", "source_manifest": {"key": f"research/x/datasets/{h}/{s}/manifest.json",
                                                                                "sha256": H("sm:" + name)},
                           "context": {"context_sha256": ctx, "dataset_hash": h, "input_raw_sha256": H(f"raw:{name}:{fold}"),
                                       "evaluator_sha256": H("evaluator"), "candidate_timeout_ms": 90000, "train_rows": 64,
                                       "test_rows": 16, "encoded_features": ncols, "task": "regression", "schema": "x"}})
                pp = R / "prepared" / name / str(fold) / "prepared.json"
                put(pp, prep)
                frozen0 = d["cohort"] == "frozen" and fold == 0
                r = {"cohort": d["cohort"], "view_id": f"{h}__r0__f{fold}", "dataset_hash": h, "sample_identity_sha256": s,
                     "fold": fold, "context_sha256": ctx, "status": status,
                     "exclusion_reason": None if status == "included" else "withdrawn",
                     "eligibility_source": ("frozen_fold0_by_construction(x)" if frozen0 else
                                            "frozen_fold1_4" if d["cohort"] == "frozen" else "incoming"),
                     "prepared_ref": {"path": str(pp), "sha256": sha(prep), "bytes": len(prep)},
                     "input_ref": {"raw_sha256": H(f"raw:{name}:{fold}")}}
                if d.get("fold0_mismatch") and fold == 0:
                    r["eligibility_source"] = "frozen_fold1_4"
                if d["cohort"] == "incoming":
                    r["item_id"] = iid
                rows.append(r)
            if d["cohort"] == "frozen":
                items.append({"id": iid, "dataset": h, "canonical": can, "sample": s, "state": "frozen", "reason": None,
                              "admitted_ns": "1", "retired_ns": None, "epoch_role": "frozen", "payload": {}})
                readiness.append({"cohort": "frozen", "dataset_hash": h, "canonical_sha256": can, "identity": f"frozen:{h}",
                                  "display_name": row["display_name"], "provenance": prov, "origin": "real",
                                  "catalog_ref": {"canonical_sha256": can, "provenance": prov}})
                catalog.append(row)
                add(f"frozen_data:{h}", "frozen_data", dpath, dataset_hash=h)
            else:
                items.append({"id": iid, "dataset": h, "canonical": can, "sample": s, "state": "complete", "reason": None,
                              "admitted_ns": "2", "retired_ns": None, "epoch_role": "incoming",
                              "payload": {"catalog_row": row, "catalog_row_sha256": sha(fp.intake_canonical(row))}})
                cr = R / "items" / iid / "catalog-row.json"
                put(cr, fp.intake_canonical(row) + b"\n")
                add(f"incoming_catalog_row:{iid}", "incoming_catalog_row", cr, dataset_hash=h, item_id=iid)
                add(f"incoming_data:{iid}", "incoming_data", dpath, dataset_hash=h, item_id=iid)
            snap_ctx.append([H(f"regonly:{name}"), iid])
            for k in range(d.get("siblings", 0)):
                ss = H(f"sample:{name}:sib{k}")
                sid = fp.identity_key(h, can, ss)
                items.append({"id": sid, "dataset": h, "canonical": can, "sample": ss, "state": "retired",
                              "reason": "missing_from_latest_corpus", "admitted_ns": "1", "retired_ns": "2",
                              "epoch_role": "sibling", "payload": {"catalog_row": row}})
                snap_ctx.append([H(f"sibctx:{name}:{k}"), sid])
            if d.get("legacy"):
                legacy.append((h, d["legacy"]))
        # catalog-only rows (fit sets, sealed references outside the epoch)
        for h2, crow in sorted(self.catalog_only.items()):
            catalog.append(crow)
        # ---------------- epoch and prepared set
        if self.shuffle is not None:
            random.Random(self.shuffle).shuffle(rows)
            random.Random(self.shuffle + 1).shuffle(items)
            random.Random(self.shuffle + 2).shuffle(catalog)
            random.Random(self.shuffle + 3).shuffle(legacy)
        epoch = jb({"schema": "jope.dl-corpus-epoch/v1", "experiment_id": fp.EXPERIMENT_ID, "epoch": 0, "contexts": rows})
        put(R / "epoch.json", epoch)
        add("epoch", "epoch", R / "epoch.json")
        digest = [[r["context_sha256"], r["prepared_ref"]["sha256"], r["prepared_ref"]["bytes"]] for r in rows]
        inputs.append({"name": "epoch_prepared_set", "role": "prepared_set", "sha256": fp._set_sha256(digest),
                       "bytes": sum(x[2] for x in digest), "files": len(digest)})
        snap = {"schema": fp.SNAPSHOT_SCHEMA, "experiment_id": fp.EXPERIMENT_ID, "poll_seq": 7, "items": items,
                "contexts": sorted(snap_ctx)}
        sb = common.canonical(snap)
        add_local("registry_snapshot", "registry_snapshot", "inputs/registry-snapshot.json", sb, poll_seq=7)
        put(R / "readiness.json", jb({"records": readiness + [{"cohort": "incoming", "dataset_hash": "f" * 16}]}))
        add("readiness", "frozen_provenance", R / "readiness.json")
        put(R / "catalog.jsonl", b"".join(json.dumps(c, sort_keys=True).encode() + b"\n" for c in catalog))
        add("catalog_v1", "catalog_v1", R / "catalog.jsonl")
        legacy += [(H("canonical:XOUT")[:16], "canary-out-legacy-family")]          # a sealed dataset outside the epoch
        tsv = "dataset_hash\tsource_file\tsource_family\n" + "".join(f"{h2}\tx.csv\t{v}\n" for h2, v in legacy)
        put(R / "legacy.tsv", tsv.encode())
        add("legacy_inventory_tsv", "legacy_source_family", R / "legacy.tsv")
        put(R / "expansion.json", jb({"datasets": []}))
        add("expansion_inventory_v7", "expansion_family", R / "expansion.json")
        # ---------------- lineage-B v1..v5, freeze, policy, cross-check sources
        hS = self.h("SEALED") if "SEALED" in self.ds else H("sealed-missing")[:16]
        hX = self.hX = H("canonical:XOUT")[:16]
        fam_t = {"family": CANARIES["lb_family_t"], "partition": "meta_test", "source_keys": [CANARIES["lb_x_key"]],
                 "datasets": [{"dataset": hS, "source_key": "/abs/path/canary_sealed_file.csv", "audit_family": None,
                               "metadata_sha256": H("lbmeta:SEALED"), "evidence": []},
                              {"dataset": hX, "source_key": self.x_source_key, "audit_family": CANARIES["audit_family"],
                               "metadata_sha256": H("lbmeta:X"),
                               "evidence": [{"source_url": u} for u in self.x_evidence]}]}
        fam_v = {"family": CANARIES["lb_family_v"], "partition": "meta_validation", "source_keys": [],
                 "datasets": [{"dataset": H("canonical:VOUT")[:16], "source_key": "rdatasets:CanaryPkg/CanaryItem",
                               "audit_family": None, "metadata_sha256": None,
                               "evidence": [{"source_url": "https://vincentarelbundock.github.io/Rdatasets/doc/CanaryPkg/CanaryItem.html"}]}]}
        train_ds = [{"dataset": self.h(n), "source_key": f"rdatasets:trainpkg/{n.lower()}", "audit_family": None, "evidence": []}
                    for n in sorted(self.ds) if self.ds[n].get("lb_train")]
        train_ds += [{"dataset": self.h(n), "source_key": "/abs/train.csv", "audit_family": CANARIES["audit_family"],
                      "evidence": []} for n in self.audit_mates]
        fam_tr = {"family": "train-fam", "partition": "meta_train", "source_keys": [], "datasets": train_ds}
        prev, lb_inputs = None, []
        for v in range(1, 6):
            fams = {CANARIES["lb_family_t"]: fam_t, "train-fam": fam_tr}
            if v >= 2:
                fams[CANARIES["lb_family_v"]] = fam_v
            if self.partition_change and v == 5:
                fams = json.loads(json.dumps(fams))
                fams["train-fam"]["partition"] = "meta_validation"
            for key, f in self.lb_extra.items():
                fams[key] = f
            doc = jb({"schema": "parallel.family_registry.v1", "parent": None if prev is None else {"sha256": prev},
                      "families": fams})
            p = R / "lb" / f"{sha(doc)}-qualification-v{v}.json"
            put(p, doc)
            add(f"lineage_b_qualification_v{v}", "sealed_registry", p)
            prev = sha(doc)
        put(R / "lb" / "freeze.json", jb({"withdrawn_families": self.withdrawn,
                                          "related_meta_test_never_clean_unseen_family_heldout": {
                                              "uniformity-trial": [CANARIES["related_family"]]}}))
        add("lineage_b_partition_freeze", "sealed_policy", R / "lb" / "freeze.json")
        put(R / "lb" / "policy.json", jb({"sealed_partitions": ["meta_validation", "meta_test"]}))
        add("lineage_b_family_exclusion_policy", "sealed_policy", R / "lb" / "policy.json")
        put(R / "index.json", jb({"encodings": [{"dataset": hS, "partition": "meta_test", "family": "f"}]
                                  + [{"dataset": x, "partition": "meta_test", "family": "f"} for x in self.native_extra]}))
        add("native_index", "sealed_crosscheck", R / "index.json")
        v1mt = self.v1_meta_test_override if self.v1_meta_test_override is not None else [hS, hX]
        put(R / "excl.jsonl", b"".join(json.dumps({"dataset_hash": x, "reasons": ["historical_sealed_meta_test"]}).encode() + b"\n"
                                       for x in v1mt) + json.dumps({"dataset_hash": "e" * 16, "reasons": ["zero_feature_dataset"]}).encode() + b"\n")
        add("canonical_exclusions", "sealed_crosscheck", R / "excl.jsonl")
        put(R / "audit.jsonl.gz", gzip.compress(json.dumps({"dataset_hash": hX, "classification": {
            "exclusion_reasons": ["historical_sealed_meta_test"]}}).encode() + b"\n", mtime=0))
        add("corpus_audit_v2", "sealed_crosscheck", R / "audit.jsonl.gz")
        split_rows = [{"dataset_hash": x, "classification": {"status": "excluded"}} for x in ["d" * 16] + self.split_excluded]
        put(R / "split.jsonl.gz", gzip.compress(b"".join(json.dumps(r).encode() + b"\n" for r in split_rows), mtime=0))
        add("split_registry_native", "sealed_crosscheck", R / "split.jsonl.gz")
        # sealed reference XOUT (content outside the epoch) and its record
        xdata = self.x_data if hasattr(self, "x_data") else mat(4242)
        tr, te = xdata[:60], xdata[60:]
        for part, m in (("train", tr), ("test", te)):
            body = ("\n".join(",".join(f"{v:.9g}" for v in r) for r in m) + "\n").encode()
            add_local(f"sealed_ref_{part}:{hX}", f"sealed_ref_{part}", f"inputs/sealed-ref/{sha(body)}", body, dataset_hash=hX,
                      has_header=False)
        xmeta = common.canonical({"feature_columns": list(self.x_names), "target_column": self.x_target,
                                  "source_url": {"value_sha256": H("x")}})
        add_local(f"sealed_ref_meta:{hX}", "sealed_ref_meta", f"inputs/sealed-ref/meta/{sha(xmeta)}.json", xmeta, dataset_hash=hX)
        xman = common.canonical({"dataset_hash": hX, "display_name": {"value_sha256": H("xd")}})
        add_local(f"sealed_ref_manifest:{hX}", "sealed_ref_manifest", f"inputs/sealed-ref/manifest/{sha(xman)}.json", xman,
                  dataset_hash=hX)
        put(R / "records" / f"{hX}.json", jb({"dataset_hash": hX, "name": CANARIES["lb_x_record"],
                                              "source_key": self.x_record_source_key}))
        add(f"dataset_record:{hX}", "sealed_dataset_record", R / "records" / f"{hX}.json", dataset_hash=hX)
        catalog_x = {"dataset_hash": hX, "id": hX, "canonical_sha256": H("canonical:XOUT"),
                     "display_name": CANARIES["lb_x_display"], "provenance": dict(self.x_catalog_prov)}
        put(R / "catalog.jsonl", (R / "catalog.jsonl").read_bytes() + json.dumps(catalog_x, sort_keys=True).encode() + b"\n")
        inputs[:] = [e for e in inputs if e["name"] != "catalog_v1"]
        add("catalog_v1", "catalog_v1", R / "catalog.jsonl")
        if self.spec_path is not None:
            put(Path(self.spec_path), self.spec_text.encode())
            add("p2_survey", "spec", Path(self.spec_path))
        # ---------------- seeds and BeyondArena
        seed_rows = []
        for src in self.seed_sources:
            for k in range(2):
                dsl = f"(seed {src} {k})"
                seed_rows.append({"program": {"program_sha256": sha(dsl.encode()), "dsl": dsl},
                                  "source_evidence": [{"source_view": f"{src}__r0__f0", "source_rank": k}]})
        broad = [seed_rows[i] for i in range(0, len(seed_rows), 2)]
        recovery = [seed_rows[i] for i in range(1, len(seed_rows), 2)]
        pool = broad + recovery
        if self.seed_tamper == "dup":
            pool[-1] = pool[0]
        elif self.seed_tamper == "dsl":
            pool[0] = json.loads(json.dumps(pool[0]))
            pool[0]["program"]["dsl"] += " "
        put(R / "seeds" / "recovery.jsonl", b"".join(json.dumps(r, sort_keys=True).encode() + b"\n" for r in pool))
        put(R / "seeds" / "broad.jsonl", b"".join(json.dumps(r, sort_keys=True).encode() + b"\n" for r in broad))
        add("seed_pool_recovery192", "seed_pool", R / "seeds" / "recovery.jsonl")
        add("seed_pool_broad96", "seed_pool", R / "seeds" / "broad.jsonl")
        for name, b in sorted(self.ba.items()):
            pbytes = parquet_bytes(b["cols"])
            pp = R / "ba" / name / "dataset.parquet"
            put(pp, pbytes)
            ptr = {"name": name, "source_family_id": "beyondarena:" + name, "original_source": b.get("url"),
                   "license": b.get("license", "CC BY 4.0"), "target": b["target"], "task": "regression",
                   "files": {"dataset.parquet": {"sha256": sha(pbytes), "bytes": len(pbytes)}}}
            put(R / "ba" / f"{name}.json", jb(ptr))
            add(f"ba_pointer:{name}", "ba_pointer", R / "ba" / f"{name}.json")
            if name in self.seed_sources:
                add(f"ba_parquet:{name}", "seed_source_parquet", pp, target=b["target"], seed_source=True)
        classic = {"schema": fp.CLASSIC_ALIASES_SCHEMA, "entries": self.classic}
        add_local("classic_aliases", "classic_aliases", "inputs/classic-aliases.json", common.canonical(classic))
        # ---------------- exposure fit sets
        put(R / "split_manifest.json", jb({"datasets": {x: {"split": "train"} for x in self.fit_train}
                                           | {x: {"split": "validation"} for x in self.fit_val}}))
        add("lineage_a_split_manifest", "exposure_fit_set", R / "split_manifest.json")
        put(R / "assembly.json", jb({"meta_training_families": {"train-fam": 1}}))
        add("lineage_b_assembly", "exposure_fit_set", R / "assembly.json")
        recs = [fam.target_fit_id(task, nm) for task, nm in self.target_records] + ["0" * 32]
        put(R / "target_meta.json", jb({"artifact_sha256": "a" * 64, "fit_records": recs}))
        add("target_metadata", "exposure_fit_set", R / "target_meta.json")
        # ---------------- alias-sources.json
        if self.shuffle is not None:
            random.Random(self.shuffle + 4).shuffle(inputs)
        S = sorted({hS, hX, H("canonical:VOUT")[:16]} | set(self.native_extra and [])
                   | {ds["dataset"] for f in self.lb_extra.values() if f.get("partition") in fp.SEALED_PARTITIONS
                      for ds in f.get("datasets") or []})
        # the S3 GET journal a live pin would have written (metas and sealed-reference files), for the local re-pin
        s3_reads = []
        for e in sorted(inputs, key=lambda x: x["name"]):
            if e["role"] == "meta" or e["role"].startswith("sealed_ref_"):
                s3_reads.append({"op": "GetObject", "role": e["role"], "bucket": "fixture", "key": f"blobs/sha256/{e['sha256']}",
                                 "sha256": e["sha256"], "bytes": e["bytes"], "status": "ok", "seq": len(s3_reads) + 1})
        cfg_extra = {"s3_reads": s3_reads}
        cfg = {"schema": fp.ALIAS_SOURCES_SCHEMA, "experiment_id": fp.EXPERIMENT_ID, "created_utc_us": 1, **cfg_extra,
               "policy": fp.jsonable(fam.POLICY),
               "epoch": {"path": str(R / "epoch.json"), "sha256": sha(epoch), "bytes": len(epoch), "s3": None, "binding": None},
               "registry_snapshot": {"path": "inputs/registry-snapshot.json", "sha256": sha(sb), "bytes": len(sb), "poll_seq": 7},
               "sealed": {"S": S, "content_unavailable": ([H("canonical:VOUT")[:16]] if self.content_unavailable is None
                                                          else list(self.content_unavailable))},
               "seeds": {"seed_sources": sorted(self.seed_sources)},
               "beyondarena": {"v6_names": []}, "inputs": inputs}
        self.config = out / "alias-sources.json"
        body = put(self.config, common.canonical(cfg))
        journal = R / "pin-receipts" / "pin.ledger.jsonl"
        put(journal, b"".join(json.dumps(e, sort_keys=True).encode() + b"\n" for e in s3_reads))
        self.pin_receipt = R / "pin-receipts" / "pin.receipt.json"
        put(self.pin_receipt, common.canonical({
            "schema": fp.RECEIPT_SCHEMA, "experiment_id": fp.EXPERIMENT_ID, "verdict": "PASS",
            "alias_sources": {"path": str(self.config), "sha256": sha(body), "bytes": len(body)},
            "s3_totals": {"journal": str(journal), "gets": len(s3_reads), "bytes": sum(e["bytes"] for e in s3_reads)},
            "counts": {"s3_gets": len(s3_reads)}}))
        return self.config

    # ------------------------------------------------------------------ helpers
    def build(self, decisions=None, **kw):
        return fam.run_build(self.config, decisions, guard=fp.NoGuard(), expect=None, **kw)


def standard_world(root, *, breaker=False) -> World:
    w = World(root)
    w.seed_sources = ["srcA", "srcB"]
    rng = np.random.default_rng(7)
    raw = np.column_stack([rng.integers(0, 10 ** 6, size=ROWS) for _ in range(3)]).astype(float)
    scaled = (raw - raw.min(axis=0)) / (raw.max(axis=0) - raw.min(axis=0))
    tgt = rng.integers(0, 10 ** 6, size=ROWS) / 10 ** 6
    w.ba["srcA"] = {"cols": {"ra": raw[:, 0], "rb": raw[:, 1], "rc": raw[:, 2], "price": tgt * 100},
                    "target": "price", "url": "https://kaggle.com/datasets/someone/srca"}
    w.ba["srcB"] = {"cols": {"p": mat(77)[:, 0] * 1000, "q": mat(78)[:, 0] * 1000, "r": mat(79)[:, 0] * 1000,
                             "y": mat(80)[:, 0]}, "target": "y", "url": "https://kaggle.com/datasets/someone/srcb"}
    w.frozen("PLAIN1", siblings=1)
    w.frozen("PLAIN2")
    w.frozen("URLA", prov={"platform": "socrata", "dataset_id": "aaaa-1111",
                           "url": "http://data.city-x.gov/api/views/abcd-1234/rows.csv#top"})
    w.frozen("URLB", prov={"platform": "socrata", "dataset_id": "bbbb-2222",
                           "url": "https://data.city-x.gov/api/views/abcd-1234/rows.csv"})
    w.frozen("URLC", prov={"platform": "Socrata ", "dataset_id": "bbbb-2222", "url": "https://other.city-y.gov/x.csv"})
    w.frozen("SEALED", display=CANARIES["sealed_display"], legacy=CANARIES["legacy_sealed"],
             prov={"platform": "examplehub", "dataset_id": "canary-sealed-id", "url": CANARIES["sealed_url"]})
    w.frozen("MATE", display=CANARIES["mate_display"], legacy=CANARIES["legacy_sealed"])
    w.frozen("ALIAS", display=CANARIES["alias_display"],
             prov={"platform": None, "url": "https://vincentarelbundock.github.io/Rdatasets/csv/CanaryPkg/CanaryItem.csv"})
    seed_data = np.column_stack([scaled, rng.integers(0, 10 ** 6, size=ROWS) / 10 ** 6])
    w.frozen("SEED", data=seed_data)
    base = mat(101)
    near2 = mat(102)
    near2[:76] = base[:76]
    w.frozen("NEAR1", data=base)
    w.frozen("NEAR2", data=near2)
    weak1 = mat(201)
    weak2 = weak1.copy()
    for j in range(weak2.shape[1]):
        weak2[:, j] = np.random.default_rng(300 + j).permutation(weak2[:, j])
    w.frozen("WEAK1", data=weak1)
    w.frozen("WEAK2", data=weak2)
    w.frozen("PMLB", prov={"platform": "pmlb", "dataset_id": "537_houses",
                           "url": "https://github.com/EpistasisLab/pmlb/tree/master/datasets/537_houses",
                           "license": {"status": "recorded", "spdx": "MIT"}})
    w.frozen("TRAIN1", lb_train=True)
    w.frozen("AUDITMATE", display=CANARIES["auditmate_display"])
    w.frozen("OUTMATE", display=CANARIES["outmate_display"], legacy="canary-out-legacy-family")
    w.audit_mates = ["AUDITMATE"]
    forb = {"display_name": "Shared Display", "collections": ["shared-collection"], "origin": "shared",
            "aliases": ["shared-alias"], "ingest": {"source": "shared", "tool_version": "shared", "relpath": "shared.csv"},
            "task": "regression", "schema": "shared", "shape": {"rows": 1}, "split": {"seed": 9},
            "target": {"name": "same", "column_order_sha256": "9" * 64}}
    w.frozen("FORB1", row_extra=forb)
    w.frozen("FORB2", row_extra=forb)
    w.frozen("NULL1", prov={"platform": None, "dataset_id": "unknown", "url": None, "dedupe_fingerprint": "n/a"})
    w.frozen("NULL2", prov={"platform": None, "dataset_id": "unknown", "url": None, "dedupe_fingerprint": "n/a"})
    w.frozen("PKG1", prov={"platform": "rdatasets", "dataset_id": "pkgx/alpha",
                           "url": "https://vincentarelbundock.github.io/Rdatasets/csv/pkgx/alpha.csv"})
    w.frozen("PKG2", prov={"platform": "rdatasets", "dataset_id": "pkgx/beta",
                           "url": "https://vincentarelbundock.github.io/Rdatasets/csv/pkgx/beta.csv"})
    xdata = mat(4242)
    nearseal = mat(4343)
    nearseal[:76] = xdata[:76]
    w.x_data = xdata
    w.frozen("NEARSEAL", display=CANARIES["nearseal_display"], data=nearseal)
    w.incoming("INC1", folds=(0, 1), prov={"platform": "nyc open data / socrata", "dataset_id": "inc1-id",
                                           "url": "https://data.inc1-host.org/inc1.csv",
                                           "license": {"status": "unknown", "note": "reported by discovery agent: CC BY-NC 4.0"}})
    w.incoming("INCW", folds=(0,), status="excluded")
    w.classic = [{"name": "srcA", "action": "union", "members": [w.h("SEED")]},
                 {"name": "review-near", "action": "review", "members": [w.h("PLAIN2")], "related_seed_sources": ["srcB"]}]
    w.fit_train = [w.h("TRAIN1"), "c" * 16]
    w.fit_val = [w.h("PLAIN2")]
    w.catalog_only = {"c" * 16: {"dataset_hash": "c" * 16, "id": "c" * 16, "canonical_sha256": "c" * 64,
                                 "provenance": {"platform": "examplehub", "dataset_id": "plain1-id"}}}
    w.target_records = [("regression", w.h("PLAIN2"))]
    if breaker:
        for k in range(12):
            w.frozen(f"GH{k:02d}", prov={"platform": "github", "dataset_id": f"gh{k}",
                                         "url": f"https://raw.githubusercontent.com/bigowner/bigrepo/main/file{k}.csv"})
    return w


class Case(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(dir=TMP_ROOT, prefix="p2core-")
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def world(self, **kw) -> World:
        w = standard_world(self.tmp / "w", **kw)
        w.write()
        return w


def fam_of(res, ident):
    model = res.final or res.pre
    return model.families[model.fid_of[ident]]


def csv_rows(res) -> list:
    return list(csv.DictReader(io.StringIO(res.outputs[fam.OUT_CSV].decode("utf-8"))))


# ======================================================================================================== tests
class TestIdentitiesAndUnions(Case):
    def test_each_identity_mapped_exactly_once_and_every_row_maps(self):
        w = self.world()
        res = w.build()
        reg = json.loads(res.outputs[fam.OUT_PRE])
        owner = {}
        for fid, f in reg["families"].items():
            for m in f["members"]:
                self.assertNotIn(m["identity"], owner)
                owner[m["identity"]] = fid
        self.assertEqual(owner, reg["identities"])
        self.assertEqual(set(owner), set(res.base.idents))
        roles = {I.role for I in res.base.idents.values()}
        self.assertEqual(roles, {"frozen", "incoming", "sibling", "beyondarena"})
        epoch = json.loads((w.root / "epoch.json").read_bytes())["contexts"]
        inc = sorted(r["context_sha256"] for r in epoch if r["status"] == "included")
        exc = sorted(r["context_sha256"] for r in epoch if r["status"] != "included")
        self.assertEqual(sorted(c for f in reg["families"].values() for c in f["contexts"]), inc)
        self.assertEqual(sorted(c for f in reg["families"].values() for c in f["excluded_contexts"]), exc)
        self.assertEqual(reg["counts"]["identities"], len(owner))

    def test_transitive_aliases_union_into_one_family(self):
        res = self.world().build()
        w = res.base
        fa, fb, fc = (fam_of(res, standard_world_item(w, n))["family_id"] for n in ("URLA", "URLB", "URLC"))
        self.assertEqual(fa, fb)
        self.assertEqual(fb, fc)
        kinds = {e.kind for e in res.pre.by_fid[fa]}
        self.assertIn("normalized_url", kinds)          # http/fragment variants normalize to one URL
        self.assertIn("platform_dataset_id", kinds)     # 'Socrata ' normalizes to 'socrata'

    def test_sibling_joins_its_dataset_and_registry_only_contexts(self):
        w = self.world()
        res = w.build()
        F = fam_of(res, w.item("PLAIN1"))
        roles = sorted(res.base.idents[m].role for m in F["_members"])
        self.assertEqual(roles, ["frozen", "sibling"])
        self.assertEqual(F["registry_only_contexts"], sorted([H("regonly:PLAIN1"), H("sibctx:PLAIN1:0")]))
        self.assertEqual(F["independence_qualified"], "auto")       # canonical-equal sibling allowed

    def test_forbidden_fields_never_union(self):
        w = self.world()
        res = w.build()
        self.assertNotEqual(fam_of(res, w.item("FORB1"))["family_id"], fam_of(res, w.item("FORB2"))["family_id"])
        for e in res.pre.edges:
            fk.check_edge(e.kind, e.value)
            self.assertFalse(any(part in fk.FORBIDDEN_FIELDS for part in e.kind.lower().replace(":", ".").split(".")))
        with self.assertRaises(fk.ForbiddenKey):
            fk.derive(w.ds["FORB1"]["_row"])          # a raw catalog row must be projected first

    def test_null_values_never_link(self):
        w = self.world()
        res = w.build()
        self.assertNotEqual(fam_of(res, w.item("NULL1"))["family_id"], fam_of(res, w.item("NULL2"))["family_id"])
        for kind, value in res.base.idents[w.item("NULL1")].U:
            self.assertTrue(fk.known(value))

    def test_aggregator_repos_are_keyed_per_file(self):
        w = self.world()
        res = w.build()
        kinds = {k for k, _ in res.base.idents[w.item("PMLB")].U}
        self.assertIn("pmlb_name", kinds)
        self.assertNotIn("github_repo", kinds)                 # never github:epistasislab/pmlb
        for n in ("PKG1", "PKG2"):
            self.assertIn("rdatasets_item", {k for k, _ in res.base.idents[w.item(n)].U})
        self.assertNotEqual(fam_of(res, w.item("PKG1"))["family_id"], fam_of(res, w.item("PKG2"))["family_id"])

    def test_deterministic_under_shuffled_input_order(self):
        outs = []
        for shuffle in (None, 11, 23):
            w = standard_world(self.tmp / "w")
            w.shuffle = shuffle
            w.write()
            res = w.build()
            reg = json.loads(res.outputs[fam.OUT_PRE])
            for k in ("alias_sources_sha256", "epoch_ref", "registry_snapshot"):
                reg.pop(k)
            outs.append((common.canonical(reg), res.outputs[fam.OUT_CSV],
                         {k: v for k, v in res.outputs.items() if k.startswith("screen/")}))
            import shutil
            shutil.rmtree(self.tmp / "w")
        self.assertEqual(outs[0], outs[1])
        self.assertEqual(outs[0], outs[2])

    def test_family_id_rule(self):
        res = self.world().build()
        for fid, F in res.pre.families.items():
            self.assertEqual(fid, "fam-" + hashlib.sha256(b"jope.dl-family-id/v1\0" + F["anchor_identity"].encode()).hexdigest())
            self.assertEqual(F["anchor_identity"], min(F["_members"]))


def standard_world_item(base, name):
    """identity of a named dataset in a built Base (frozen or incoming role)."""
    h, can = H("canonical:" + name)[:16], H("canonical:" + name)
    return "item:" + fp.identity_key(h, can, H("sample:" + name))


class TestScreenAndCurated(Case):
    def test_strong_value_link_unions_and_weak_link_is_a_review_row(self):
        w = self.world()
        res = w.build()
        self.assertEqual(fam_of(res, w.item("NEAR1"))["family_id"], fam_of(res, w.item("NEAR2"))["family_id"])
        f1, f2 = fam_of(res, w.item("WEAK1"))["family_id"], fam_of(res, w.item("WEAK2"))["family_id"]
        self.assertNotEqual(f1, f2)
        weak = [r for r in csv_rows(res) if r["kind"] == "nds_value_weak"]
        self.assertTrue(any(set(r["family_ids"].split(";")) == {f1, f2} for r in weak))
        self.assertEqual(fam_of(res, w.item("WEAK1"))["independence_qualified"], "unresolved")

    def test_curated_alias_unions_seed_and_requires_value_confirmation(self):
        w = self.world()
        res = w.build()
        F = fam_of(res, "beyondarena:srcA")
        self.assertIn(w.item("SEED"), F["_members"])
        self.assertTrue(F["seed_selection_exposed"])
        self.assertEqual(F["seeds_from_sources"], 2)
        self.assertIn("curated_alias", {e.kind for e in res.pre.by_fid[F["family_id"]]})
        lone = fam_of(res, "beyondarena:srcB")
        self.assertEqual(lone["contexts"], [])
        self.assertEqual(lone["independence_qualified"], "unresolved")
        # a union entry whose member carries no value link is a STOP
        w2 = standard_world(self.tmp / "w2")
        w2.classic = [{"name": "srcA", "action": "union", "members": [w2.h("SEED"), w2.h("PLAIN2")]}]
        w2.write()
        with self.assertRaisesRegex(fam.FamilyStop, "no strong or content screen link"):
            w2.build()

    def test_circuit_breaker_holds_edges_and_makes_a_review_row(self):
        w = standard_world(self.tmp / "w", breaker=True)
        w.write()
        res = w.build()
        self.assertEqual(res.summary["verdict"], "STOP")
        self.assertEqual(len(res.base.trips), 1)
        self.assertEqual(res.base.trips[0]["datasets"], 12)
        fids = {fam_of(res, w.item(f"GH{k:02d}"))["family_id"] for k in range(12)}
        self.assertEqual(len(fids), 12)                     # held back: no union through the tripped edges
        rows = [r for r in csv_rows(res) if r["kind"] == "circuit_breaker"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(set(rows[0]["family_ids"].split(";")), fids)
        # deciding the breaker row 'union' re-admits the held edges
        fam.write_outputs(w.out, res.outputs)
        dec = decisions_doc(res, group={rows[0]["review_group_id"]: {"decision": "union"}})
        put(w.out / "dec.json", common.canonical(dec))
        fin = w.build(w.out / "dec.json")
        self.assertEqual(fin.summary["verdict"], "PASS")
        self.assertEqual(len({fam_of(fin, w.item(f"GH{k:02d}"))["family_id"] for k in range(12)}), 1)

    def test_stop_columns_reported(self):
        res = self.world().build()
        files = {k: json.loads(v) for k, v in res.outputs.items() if k.endswith(".json")}
        self.assertEqual(files["screen/stop-columns.json"]["T"], 5)
        self.assertIn("10", files["screen/sensitivity.json"]["by_T"])


class TestSealed(Case):
    def test_sealed_bases_and_provisional_seals(self):
        w = self.world()
        res = w.build()
        S = fam_of(res, w.item("SEALED"))
        self.assertTrue(S["sealed"])
        self.assertEqual(S["sealed_basis"], "lineage_b_member")
        A = fam_of(res, w.item("ALIAS"))
        self.assertTrue(A["sealed"])
        self.assertEqual(A["sealed_basis"], "lineage_b_rdatasets_alias")
        N = fam_of(res, w.item("NEARSEAL"))
        self.assertTrue(N["sealed"])
        self.assertEqual(N["sealed_basis"], "value_near_dup_of_sealed_reference")
        M = fam_of(res, w.item("MATE"))
        self.assertTrue(M["sealed"])
        self.assertEqual(M["sealed_basis"], "provisional_pending_review")
        AM = fam_of(res, w.item("AUDITMATE"))
        self.assertEqual(AM["sealed_basis"], "provisional_pending_review")
        OM = fam_of(res, w.item("OUTMATE"))             # legacy mate of a sealed dataset outside the epoch
        self.assertEqual(OM["sealed_basis"], "provisional_pending_review")
        for n in ("PKG1", "PKG2", "PLAIN1", "TRAIN1"):          # a package trigger alone never seals
            self.assertFalse(fam_of(res, w.item(n))["sealed"])
        sa = [r for r in csv_rows(res) if r["kind"] == "sealed_adjacent"]
        self.assertEqual({r["family_ids"].split(";")[0] for r in sa} & {M["family_id"], AM["family_id"]},
                         {M["family_id"], AM["family_id"]})
        for r in sa:
            self.assertTrue(r["trigger_value"].startswith("sha256:"))
            self.assertIn(r["recommendation"], ("keep_sealed", "unseal"))
        for F in (S, A, N, M):
            self.assertEqual(F["independence_qualified"], "unresolved")

    def test_independent_sealed_recompute_and_verify_pass(self):
        w = self.world()
        res = w.build()
        fam.write_outputs(w.out, res.outputs)
        rep = fam.verify(w.config, w.out, guard=fp.NoGuard(), expect=None)
        self.assertEqual(rep["verdict"], "PASS", rep["failed"])
        self.assertTrue(rep["checks"]["base_sealed_recomputed_equal"])

    def test_verify_fails_on_a_canary_leak(self):
        w = self.world()
        res = w.build()
        fam.write_outputs(w.out, res.outputs)
        rep = fam.verify(w.config, w.out, guard=fp.NoGuard(), expect=None,
                         extra_texts={"captured_stdout": "counts ... " + CANARIES["sealed_display"]})
        self.assertEqual(rep["verdict"], "FAIL")
        self.assertIn("canary_scan_clean", rep["failed"])
        self.assertEqual(rep["canary"]["hits"]["captured_stdout"], 1)
        self.assertNotIn(CANARIES["sealed_display"], json.dumps(rep))
        hits = fam.canary_scan({"a": "x " + CANARIES["lb_family_t"].lower().replace("-", "_") + " y"}, res.cans)
        self.assertEqual(hits["a"], 1)                       # the normalized form of a long canary is a leak too

    def test_crosscheck_stops(self):
        cases = [
            ("split", lambda w: w.split_excluded.append(w.h("PLAIN1")), "split-registry"),
            ("native", lambda w: w.native_extra.append("ab" * 8), "native index"),
            ("v1", lambda w: setattr(w, "v1_meta_test_override", [w.h("SEALED")]), "v1 meta_test"),
            ("partition", lambda w: setattr(w, "partition_change", True), "partition change"),
        ]
        for tag, mutate, msg in cases:
            with self.subTest(tag):
                w = standard_world(self.tmp / tag)
                mutate(w)
                w.write()
                with self.assertRaisesRegex(fam.FamilyStop, msg):
                    w.build()

    def test_canary_never_written_or_printed(self):
        w = self.world()
        buf_out, buf_err = io.StringIO(), io.StringIO()
        roots = (w.out, self.tmp / "receipts")
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            code = fam._main(["build", "--config", str(w.config), "--out", str(w.out), "--receipt",
                              str(self.tmp / "receipts" / "build.json")], roots=roots, guard=fp.NoGuard(), expect=None)
            vcode = fam._main(["verify", "--config", str(w.config), "--out", str(w.out), "--receipt",
                               str(self.tmp / "receipts" / "verify.json")], roots=roots, guard=fp.NoGuard(), expect=None)
        self.assertEqual(code, 0, buf_err.getvalue())
        self.assertEqual(vcode, 0, buf_out.getvalue()[-2000:])
        texts = [buf_out.getvalue(), buf_err.getvalue()]
        for p in sorted(w.out.rglob("*")):
            if p.is_file() and p.name not in ("alias-sources.json",) and "inputs" not in p.parts:
                texts.append(p.read_text(encoding="utf-8"))
        for p in (self.tmp / "receipts").glob("*.json"):
            texts.append(p.read_text(encoding="utf-8"))
        blob = "\n".join(texts)
        for key, canary in CANARIES.items():
            self.assertNotIn(canary, blob, key)
            self.assertNotIn(canary.lower(), blob.lower(), key)
        self.assertGreater(len(blob), 10_000)


class TestIdsAndStability(Case):
    def test_previous_registry_keeps_ids_and_post_freeze_merge_stops(self):
        w = self.world()
        res = w.build()
        prev = json.loads(res.outputs[fam.OUT_PRE])
        again = w.build(previous=prev)
        self.assertEqual(set(again.pre.families), set(res.pre.families))
        # a new identity smaller than the old anchor joins an existing family: the persisted id and anchor are kept
        w2 = standard_world(self.tmp / "w2")
        w2.frozen("AAA-JOINS-PLAIN2", prov=dict(w2.ds.get("PLAIN2", {}).get("prov") or {
            "platform": "examplehub", "dataset_id": "plain2-id", "url": "https://data.plain2-src.org/plain2.csv"}))
        w2.write()
        joined = w2.build(previous=prev)
        F = fam_of(joined, w2.item("PLAIN2"))
        self.assertIn(w2.item("AAA-JOINS-PLAIN2"), F["_members"])
        self.assertEqual(F["family_id"], fam_of(res, w.item("PLAIN2"))["family_id"])
        self.assertEqual(F["anchor_identity"], fam_of(res, w.item("PLAIN2"))["anchor_identity"])
        # a post-freeze merge of two existing families is a STOP
        w3 = standard_world(self.tmp / "w3")
        w3.ds["PLAIN2"]["prov"] = {"platform": "examplehub", "dataset_id": "plain1-id",
                                   "url": "https://data.plain1-src.org/plain1.csv"}
        w3.write()
        with self.assertRaisesRegex(fam.FamilyStop, "post-freeze merge"):
            w3.build(previous=prev)

    def test_final_merge_retires_the_larger_anchor_id(self):
        w = self.world()
        res = w.build()
        fam.write_outputs(w.out, res.outputs)
        row = next(r for r in res.pre.rows if r["kind"] == "rdatasets_pkg")
        dec = decisions_doc(res, group={row["review_group_id"]: {"decision": "union"}})
        put(w.out / "dec.json", common.canonical(dec))
        fin = w.build(w.out / "dec.json")
        anchors = sorted((res.pre.families[f]["anchor_identity"], f) for f in row["family_ids"])
        kept, gone = anchors[0][1], anchors[1][1]
        self.assertIn(kept, fin.final.families)
        self.assertNotIn(gone, fin.final.families)
        self.assertEqual(fin.final.retired, {gone: kept})
        reg = json.loads(fin.outputs[fam.OUT_FINAL])
        self.assertEqual(reg["retired_family_ids"], {gone: kept})


class TestContexts(Case):
    def test_frozen_fold0_and_mismatch_stop(self):
        w = self.world()
        res = w.build()
        F = fam_of(res, w.item("PLAIN2"))
        self.assertEqual(sorted(F["frozen_fold0"].values()), [False, False, True])
        self.assertEqual(F["frozen_fold0"][H("ctx:PLAIN2:0")], True)
        I = fam_of(res, w.item("INC1"))
        self.assertEqual(set(I["frozen_fold0"].values()), {False})
        W = fam_of(res, w.item("INCW"))
        self.assertEqual(W["contexts"], [])
        self.assertEqual(W["excluded_contexts"], {H("ctx:INCW:0"): "withdrawn"})
        self.assertEqual(W["independence_qualified"], "unresolved")
        w2 = standard_world(self.tmp / "w2")
        w2.ds["PLAIN2"]["fold0_mismatch"] = True
        w2.write()
        with self.assertRaisesRegex(fam.FamilyStop, "frozen_fold0 disagrees"):
            w2.build()


class TestRightsExposureSeeds(Case):
    def test_rights_union_rules(self):
        w = self.world()
        res = w.build()
        self.assertEqual(fam_of(res, w.item("PMLB"))["rights_status"], "recorded:MIT")
        self.assertEqual(fam_of(res, w.item("PLAIN2"))["rights_status"], "unknown")
        self.assertEqual(fam_of(res, "beyondarena:srcB")["rights_status"], "unknown")    # external members excluded
        rm = fam.family_rights([w.item("PMLB"), w.item("PLAIN2")], res.base.idents, False)
        self.assertEqual(rm[0], "unknown")                                                 # mixed -> unknown
        self.assertEqual(fam.family_rights([w.item("PMLB")], res.base.idents, True)[0], "prohibited")
        # one spdx but a member that is not 'recorded' -> unknown; every member recorded with one spdx -> recorded
        mk = lambda i, lic: fam.Ident(identity=i, kind="registry_item", role="frozen", license=lic)  # noqa: E731
        idents = {"a": mk("a", {"status": "recorded", "spdx": "MIT"}), "b": mk("b", {"status": "unknown", "spdx": "MIT"}),
                  "c": mk("c", {"status": "recorded", "spdx": "MIT"}), "d": mk("d", {"status": "recorded", "spdx": "CC BY 4.0"}),
                  "x": fam.Ident(identity="x", kind="beyondarena", role="beyondarena", license={"status": "external"})}
        self.assertEqual(fam.family_rights(["a", "b"], idents, False)[0], "unknown")
        self.assertEqual(fam.family_rights(["a", "c", "x"], idents, False)[0], "recorded:MIT")
        self.assertEqual(fam.family_rights(["d"], idents, False)[0], "unknown")          # not an SPDX identifier
        self.assertEqual(fam.family_rights(["x"], idents, False)[0], "unknown")
        inc = fam_of(res, w.item("INC1"))
        self.assertTrue(inc["rights_members"][w.item("INC1")]["restrictive_note"])
        rows = [r for r in csv_rows(res) if r["kind"] == "rights_note"]
        self.assertEqual([r["family_ids"] for r in rows], [inc["family_id"]])
        self.assertEqual(rows[0]["recommendation"], "unknown")

    def test_exposure_states(self):
        w = self.world()
        res = w.build()
        A, B, T = ("lineageA_normalizers@3df46159+6c07305f", "lineageB_transforms@1b779c6f", "target_fused_v1@be96adc6")
        tr = fam_of(res, w.item("TRAIN1"))["encoder_weight_exposure"]
        self.assertEqual(tr[A]["state"], "Exposed")
        self.assertEqual(tr[B]["state"], "Exposed")
        plain1 = fam_of(res, w.item("PLAIN1"))["encoder_weight_exposure"]
        self.assertEqual(plain1[A]["state"], "Exposed")      # shares platform_dataset_id with a catalog-only fit row
        p2 = fam_of(res, w.item("PLAIN2"))["encoder_weight_exposure"]
        self.assertEqual(p2[A]["state"], "Unseen")
        self.assertEqual(p2[T]["state"], "Exposed")           # resolved target fit record
        self.assertEqual(p2["tabpfn35@ece4d67e"]["state"], "Unknown")
        inc = fam_of(res, w.item("INC1"))["encoder_weight_exposure"]
        self.assertEqual(inc[A]["state"], "Unknown")         # fresh: never Unseen
        seed = fam_of(res, "beyondarena:srcB")["encoder_weight_exposure"]
        self.assertEqual(seed[B]["state"], "Unknown")        # external member

    def test_seed_asserts_stop(self):
        for tamper, msg in (("dup", "not distinct"), ("dsl", r"sha256\(dsl\)")):
            with self.subTest(tamper):
                w = standard_world(self.tmp / tamper)
                w.seed_tamper = tamper
                w.write()
                with self.assertRaisesRegex(fam.FamilyStop, msg):
                    w.build()


class TestReviewRows(Case):
    def test_rows_kinds_order_ids_and_recommendations(self):
        w = self.world()
        res = w.build()
        rows = csv_rows(res)
        self.assertEqual(tuple(rows[0].keys()), fam.REVIEW_COLUMNS)
        kinds = [r["kind"] for r in rows]
        mand = [k for k in kinds if k in fam.MANDATORY_KINDS]
        self.assertEqual(kinds[:len(mand)], mand)              # mandatory rows first
        self.assertEqual(kinds.count("seed_source_mapping"), 2)
        for r in rows:
            fids = r["family_ids"].split(";")
            self.assertEqual(r["review_group_id"], fam._group_id(r["kind"], r["trigger_value"], fids))
            self.assertEqual(r["decision_options"].split(";"), list(fam.DECISION_OPTIONS[r["kind"]]))
            self.assertTrue(r["recommendation_basis"].startswith("agent-generated"))
        pkg = [r for r in rows if r["kind"] == "rdatasets_pkg"]
        self.assertEqual(len(pkg), 1)
        self.assertEqual(pkg[0]["trigger_value"], "rdatasets:pkgx")
        self.assertEqual(pkg[0]["recommendation"], "independent")
        multi = [r for r in rows if r["kind"] == "multi_member_family"]
        self.assertTrue(multi and all(r["recommendation"] == "confirm_union" for r in multi))
        seed = [r for r in rows if r["kind"] == "seed_source_mapping"]
        self.assertTrue(all(r["recommendation"] == "confirm_mapping" and r["default_if_undecided"] == "P3 STOP" for r in seed))
        # a family with any review row never auto-qualifies; plain singletons do
        self.assertEqual(fam_of(res, w.item("PKG1"))["independence_qualified"], "unresolved")
        self.assertEqual(fam_of(res, w.item("PLAIN2"))["independence_qualified"], "unresolved")   # classic review row
        self.assertEqual(fam_of(res, w.item("PLAIN1"))["independence_qualified"], "auto")

    def test_pairwise_legacy_evidence_recommends_union_subset(self):
        w = standard_world(self.tmp / "w")
        w.ds["PKG1"]["legacy"] = "shared-legacy-slug"
        w.ds["PKG2"]["legacy"] = "shared-legacy-slug"
        w.frozen("PKG3", prov={"platform": "rdatasets", "dataset_id": "pkgx/gamma",
                               "url": "https://vincentarelbundock.github.io/Rdatasets/csv/pkgx/gamma.csv"})
        w.write()
        res = w.build()
        pkg = next(r for r in csv_rows(res) if r["kind"] == "rdatasets_pkg")
        self.assertEqual(pkg["recommendation"], "union_subset")
        pair = sorted(res.pre.fid_of[w.item(n)] for n in ("PKG1", "PKG2"))
        self.assertIn(json.dumps([pair]).replace(" ", ""), pkg["recommendation_basis"])
        leg = next(r for r in csv_rows(res) if r["kind"] == "legacy_source_family" and r["trigger_value"] == "shared-legacy-slug")
        # review fix CL-1: a legacy row's own shared legacy family is pairwise legacy evidence (spec: union_subset)
        self.assertEqual(leg["recommendation"], "union_subset")
        self.assertIn(json.dumps([pair]).replace(" ", ""), leg["recommendation_basis"])


def decisions_doc(res, *, group=None, classes=None, seed="confirm_mapping", provenance=True, rulings=None, binds=None):
    rows = res.pre.rows
    dec = [{"review_group_id": r["review_group_id"], "decision": seed} for r in rows if r["kind"] == "seed_source_mapping"]
    for g, d in (group or {}).items():
        dec = [x for x in dec if x["review_group_id"] != g] + [dict(review_group_id=g, **d)]
    doc = {"schema": fam.DECISIONS_SCHEMA, "reviewer": "owner",
           "binds": binds or {"review_csv_sha256": sha(res.outputs[fam.OUT_CSV]),
                              "registry_pre_review_sha256": sha(res.outputs[fam.OUT_PRE]),
                              "alias_sources_sha256": res.base.inp.sha256},
           "rulings": rulings or {"rdatasets_granularity": "item", "provisional_seals": "keep",
                                  "related_class_policy": "unresolved", "split_registry_excluded_seals_nothing": True},
           "class_decisions": classes or [], "decisions": dec}
    if provenance:
        doc["provenance"] = {"kind": "owner_verbatim_message", "session": "fixture session (test only)",
                             "message_utc": "2026-10-05T08:00:00Z", "message_sha256": H("owner-message")}
    return doc


class TestDecisions(Case):
    def setUp(self):
        super().setUp()
        self.w = self.world()
        self.res = self.w.build()
        fam.write_outputs(self.w.out, self.res.outputs)
        self.rows = {r["kind"]: r for r in self.res.pre.rows}

    def final(self, doc):
        p = self.w.out / f"dec-{sha(common.canonical(doc))[:8]}.json"
        put(p, common.canonical(doc))
        return self.w.build(p)

    def test_class_and_group_decisions_review_and_union(self):
        pkg = self.rows["rdatasets_pkg"]
        classes = [{"kind": k, "when": "recommendation == independent", "decision": "independent"}
                   for k in fam.TRIGGER_ROW_KINDS + fam.CANDIDATE_ROW_KINDS]
        classes += [{"kind": "multi_member_family", "when": "recommendation==confirm_union", "decision": "confirm_union"},
                    {"kind": "rights_note", "when": "recommendation==unknown", "decision": "unknown"},
                    {"kind": "sealed_adjacent", "when": "recommendation==keep_sealed", "decision": "keep_sealed"},
                    {"kind": "sealed_adjacent", "when": "recommendation==unseal", "decision": "unseal"}]
        rulings = {"rdatasets_granularity": "item", "provisional_seals": "override",
                   "related_class_policy": "unresolved", "split_registry_excluded_seals_nothing": True}
        fin = self.final(decisions_doc(self.res, classes=classes, rulings=rulings,
                                       group={pkg["review_group_id"]: {"decision": "union"}}))
        reg = json.loads(fin.outputs[fam.OUT_FINAL])
        self.assertEqual(reg["phase"], "final")
        self.assertEqual(reg["review_decisions_sha256"], fin.final.decisions_sha256)
        merged = fam_of(fin, self.w.item("PKG1"))
        self.assertIn(self.w.item("PKG2"), merged["_members"])
        self.assertEqual(merged["independence_qualified"], "reviewed")
        self.assertEqual(fam_of(fin, self.w.item("PLAIN1"))["independence_qualified"], "auto")
        self.assertIn("owner_decision", {e.kind for e in fin.final.by_fid[merged["family_id"]]})
        outmate = fam_of(fin, self.w.item("OUTMATE"))           # legacy adjacency only, no recommended union -> unseal
        self.assertFalse(outmate["sealed"])
        mate = fam_of(fin, self.w.item("MATE"))      # CL-1: a legacy row recommends uniting it with SEALED -> keep_sealed
        self.assertTrue(mate["sealed"])
        self.assertEqual(mate["sealed_basis"], "owner_decision")
        self.assertEqual(reg["decision_deltas"]["rulings"]["applied"]["provisional_seals"], "override")
        self.assertEqual(reg["decision_deltas"]["rulings"]["applied"]["unseal_decisions"], 1)
        # CL-3: provisional_seals 'keep' refuses the same unseal decision
        with self.assertRaisesRegex(fam.FamilyStop, "provisional_seals 'keep' contradicts 1 unseal"):
            self.final(decisions_doc(self.res, classes=classes, group={pkg["review_group_id"]: {"decision": "union"}}))
        # P3's registry adapter accepts both registries
        from dltrain import holdout
        self.assertEqual(holdout.validate_registry(json.loads(self.res.outputs[fam.OUT_PRE]), require_final=False), [])
        self.assertEqual(holdout.validate_registry(reg, require_final=True), [])

    def test_seed_rows_reach_p3_gate(self):
        fin = self.final(decisions_doc(self.res))
        from dltrain import holdout
        reg = holdout.Registry(path="x", sha256=sha(fin.outputs[fam.OUT_FINAL]), bytes=0,
                               doc=json.loads(fin.outputs[fam.OUT_FINAL]), families={})
        dec_path = sorted(self.w.out.glob("dec-*.json"))[0]
        with mock.patch.object(holdout, "EXPECTED_SEED_ROWS", 2):
            gate = holdout.review_gate(reg, dec_path.read_bytes(), self.res.outputs[fam.OUT_CSV])
        self.assertEqual(gate["seed_rows"], 2)

    def test_illegal_decisions_stop(self):
        pkg, multi, sa = self.rows["rdatasets_pkg"], self.rows["multi_member_family"], self.rows["sealed_adjacent"]
        rights = self.rows["rights_note"]
        bad = [
            ({"group": {"rg-000000000000000000000000": {"decision": "independent"}}}, "unknown review group"),
            ({"group": {pkg["review_group_id"]: {"decision": "confirm_mapping"}}}, "not allowed"),
            ({"group": {pkg["review_group_id"]: {"decision": "union_subset", "sets": [[pkg["family_ids"][0]]]}}}, "union_subset"),
            ({"provenance": False}, "provenance"),
            ({"binds": {"review_csv_sha256": "0" * 64}}, "different review.csv"),
            ({"rulings": {"rdatasets_granularity": "kingdom"}}, "rulings"),
            ({"classes": [{"kind": "rdatasets_pkg", "when": "recommendation==independent", "decision": "independent"},
                          {"kind": "rdatasets_pkg", "when": "recommendation == independent", "decision": "union"}]}, "ambiguous"),
        ]
        for kw, msg in bad:
            with self.subTest(msg):
                with self.assertRaisesRegex(fam.FamilyStop, msg):
                    self.final(decisions_doc(self.res, **kw))
        # unseal is allowed only for provisional seals; split only through github/nds/curated edges
        sealed_row = next(r for r in self.res.pre.rows if r["kind"] == "sealed_adjacent")
        self.assertTrue(sealed_row)
        exact_multi = next(r for r in self.res.pre.rows if r["kind"] == "multi_member_family"
                           and not ({e.kind for e in self.res.pre.by_fid[r["family_ids"][0]]} & set(fam.SPLITTABLE_KINDS)))
        with self.assertRaisesRegex(fam.FamilyStop, "split"):
            self.final(decisions_doc(self.res, group={exact_multi["review_group_id"]: {"decision": "split"}}))
        self.assertEqual(rights["kind"], "rights_note")

    def test_split_prohibited_and_add_members(self):
        near_row = next(r for r in self.res.pre.rows if r["kind"] == "multi_member_family"
                        and self.w.item("NEAR1") in self.res.pre.families[r["family_ids"][0]]["_members"])
        rights = self.rows["rights_note"]
        seed_b = next(r for r in self.res.pre.rows if r["kind"] == "seed_source_mapping"
                      and r["trigger_value"] == "beyondarena:srcB")
        plain2 = self.res.pre.fid_of[self.w.item("PLAIN2")]
        fin = self.final(decisions_doc(self.res, group={       # CL-6: value_strong edges split only when listed
            near_row["review_group_id"]: {"decision": "split", "edge_kinds": ["nds:value_strong"]},
            rights["review_group_id"]: {"decision": "prohibited"},
            seed_b["review_group_id"]: {"decision": "add_members", "family_ids": [plain2]}}))
        self.assertNotEqual(fam_of(fin, self.w.item("NEAR1"))["family_id"], fam_of(fin, self.w.item("NEAR2"))["family_id"])
        self.assertEqual(fam_of(fin, self.w.item("INC1"))["rights_status"], "prohibited")
        F = fam_of(fin, self.w.item("PLAIN2"))
        self.assertIn("beyondarena:srcB", F["_members"])
        self.assertTrue(F["seed_selection_exposed"])
        reg = json.loads(fin.outputs[fam.OUT_FINAL])
        self.assertEqual(reg["decision_deltas"]["seed_exposed_families"], 0)

    def test_undecided_rows_leave_families_unresolved(self):
        fin = self.final(decisions_doc(self.res))
        self.assertEqual(fam_of(fin, self.w.item("PKG1"))["independence_qualified"], "unresolved")
        self.assertTrue(fam_of(fin, self.w.item("MATE"))["sealed"])            # undecided provisional seal stays
        self.assertEqual(fam_of(fin, self.w.item("MATE"))["sealed_basis"], "provisional_pending_review")


class TestInputsAndHygiene(Case):
    def test_repin_and_policy_drift_stop(self):
        w = self.world()
        p = w.root / "legacy.tsv"
        p.write_bytes(p.read_bytes() + b"x\ty\tz\n")
        with self.assertRaisesRegex(fam.FamilyStop, "re-pin"):
            w.build()
        w2 = standard_world(self.tmp / "w2")
        w2.write()
        cfg = json.loads(w2.config.read_bytes())
        cfg["policy"]["STOP_T"] = 6
        w2.config.write_bytes(common.canonical(cfg))
        with self.assertRaisesRegex(fam.FamilyStop, "policy drift"):
            w2.build()

    def test_outputs_are_create_only(self):
        w = self.world()
        res = w.build()
        fam.write_outputs(w.out, res.outputs)
        fam.write_outputs(w.out, res.outputs)                       # equal bytes: success
        changed = dict(res.outputs, **{fam.OUT_CSV: res.outputs[fam.OUT_CSV] + b"\n"})
        with self.assertRaisesRegex(fam.FamilyStop, "create-only"):
            fam.write_outputs(w.out, changed)

    def test_no_forbidden_imports_and_no_sqlite(self):
        tree = ast.parse(Path(fam.__file__).read_text())
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split(".")[0])
                names |= {a.name for a in node.names}
        for bad in fp.FORBIDDEN_IMPORTS + ("sqlite3", "boto3", "botocore"):
            self.assertNotIn(bad, names)

    def test_sqlite_read_only_connection_refuses_writes(self):
        db = self.tmp / "ro.sqlite3"
        con = sqlite3.connect(db)
        con.execute("create table t(x)")
        con.commit()
        con.close()
        ro = fp.connect_ro(db)
        with self.assertRaises(sqlite3.OperationalError):
            ro.execute("insert into t values (1)")
        ro.close()

    def test_policy_round_trips_through_pin_serialization(self):
        pol = fp.jsonable(fam.POLICY)
        self.assertEqual(json.loads(common.canonical(pol)), pol)
        self.assertEqual(pol["SEALED_PARTITIONS"], ["meta_test", "meta_validation"])
        self.assertEqual(pol["FID_DOMAIN"], "jope.dl-family-id/v1\x00")
        self.assertEqual(fs_policy_subset(pol)["STOP_T"], 5)

    def test_cli_pin_delegates_to_family_pin_with_policy(self):
        out = self.tmp / "families"
        with mock.patch.object(fp, "pin", return_value={"verdict": "PREFLIGHT_PASS", "counts": {"epoch_rows": 1}}) as m:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = fam._main(["pin", "--out", str(out), "--preflight"], roots=(out, self.tmp / "receipts"),
                                 guard=fp.NoGuard())
        self.assertEqual(code, 0)
        self.assertIs(m.call_args.kwargs["policy"], fam.POLICY)
        self.assertTrue(m.call_args.kwargs["preflight"])
        self.assertIsNone(m.call_args.kwargs["ledger_path"])
        self.assertEqual(json.loads(buf.getvalue())["counts"], {"epoch_rows": 1})

    def test_decisions_sha_pin(self):
        w = self.world()
        res = w.build()
        fam.write_outputs(w.out, res.outputs)
        put(w.out / "dec.json", common.canonical(decisions_doc(res)))
        with self.assertRaisesRegex(fam.FamilyStop, "differs from its pin"):
            w.build(w.out / "dec.json", decisions_sha256="0" * 64)
        ok = w.build(w.out / "dec.json", decisions_sha256=sha((w.out / "dec.json").read_bytes()))
        self.assertEqual(ok.final.phase, "final")

    def test_cli_rejects_outputs_outside_roots(self):
        w = self.world()
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = fam._main(["build", "--config", str(w.config), "--out", str(self.tmp / "elsewhere")],
                             roots=(w.out, self.tmp / "receipts"), guard=fp.NoGuard(), expect=None)
        self.assertEqual(code, fam.EXIT["STOP"])


def fs_policy_subset(pol):
    from dltrain import family_screen as fs
    return fs.screen_policy(pol)


# ------------------------------------------------------------------------------------------------ publish (fake S3)
def _client_error(status, code, op):
    return ClientError({"Error": {"Code": code, "Message": code}, "ResponseMetadata": {"HTTPStatusCode": status}}, op)


class FakeS3:
    """Versioned in-memory bucket; every call first passes the real lane guard."""

    def __init__(self, writes: bool, store: dict):
        self.store, self.guard, self.calls = store, s3io.Guard(writes=writes), []

    def _check(self, op, params):
        self.guard.check_params(params, types.SimpleNamespace(name=op))
        self.calls.append((op, params.get("Key")))

    def put_object(self, **params):
        self._check("PutObject", params)
        versions = self.store.setdefault(params["Key"], [])
        if params.get("IfNoneMatch") == "*" and versions:
            raise _client_error(412, "PreconditionFailed", "PutObject")
        vid = f"v{len(versions) + 1}-{sha(bytes(params['Body']))[:6]}"
        versions.append((vid, bytes(params["Body"])))
        return {"VersionId": vid, "ETag": '"' + "0" * 32 + '"'}

    def get_object(self, **params):
        self._check("GetObject", params)
        versions = self.store.get(params["Key"]) or []
        found = [v for v in versions if v[0] == params.get("VersionId")] if "VersionId" in params else versions[-1:]
        if not found:
            raise _client_error(404, "NoSuchKey", "GetObject")
        vid, body = found[0]
        return {"Body": StreamingBody(io.BytesIO(body), len(body)), "VersionId": vid, "ETag": '"' + "0" * 32 + '"',
                "ContentLength": len(body)}


class TestPublish(Case):
    def test_publish_after_verify_is_create_only_and_reruns_cleanly(self):
        w = self.world()
        res = w.build()
        fam.write_outputs(w.out, res.outputs)
        rec = self.tmp / "receipts"
        vr = fam.verify(w.config, w.out, guard=fp.NoGuard(), expect=None, receipt_path=rec / "verify.json")
        self.assertEqual(vr["verdict"], "PASS", vr["failed"])
        store = {}
        clients = {"writer": FakeS3(True, store), "reader": FakeS3(False, store)}
        pub = fam.publish(w.out, receipt_path=rec / "receipt.json", verify_receipt_path=rec / "verify.json", clients=clients)
        keys = sorted(store)
        self.assertTrue(all(k.startswith(s3io.PREFIX_KEY) for k in keys))
        self.assertEqual(len(keys), 6)          # alias-sources, pre-review registry, review.csv, screen bundle, manifest, receipt
        self.assertIn(f"{s3io.PREFIX_KEY}objects/family-registry/sha256/{sha(res.outputs[fam.OUT_PRE])}.json", keys)
        self.assertIn(f"{s3io.PREFIX_KEY}objects/family-review/sha256/{sha(res.outputs[fam.OUT_CSV])}.csv", keys)
        receipt = json.loads((rec / "receipt.json").read_bytes())
        self.assertEqual(receipt["schema"], fam.PACKET_RECEIPT_SCHEMA)
        self.assertEqual(len(receipt["s3_ledger"]), 5)
        self.assertEqual(pub["refs"]["receipt"]["sha256"], sha((rec / "receipt.json").read_bytes()))
        # a second attempt re-puts equal bytes (create-only 412 -> equal) under a new receipt path
        pub2 = fam.publish(w.out, receipt_path=rec / "receipt2.json", verify_receipt_path=rec / "verify.json", clients=clients)
        self.assertEqual(pub2["refs"]["registry_pre_review"]["outcome"], "already_present_equal_bytes")
        with self.assertRaisesRegex(fam.FamilyStop, "exists"):
            fam.publish(w.out, receipt_path=rec / "receipt.json", verify_receipt_path=rec / "verify.json", clients=clients)

    def test_publish_refuses_without_verify_pass(self):
        w = self.world()
        res = w.build()
        fam.write_outputs(w.out, res.outputs)
        rec = self.tmp / "receipts"
        put(rec / "verify.json", common.receipt_bytes({"schema": fam.VERIFY_RECEIPT_SCHEMA, "verdict": "FAIL"}))
        with self.assertRaisesRegex(fam.FamilyStop, "verify PASS"):
            fam.publish(w.out, receipt_path=rec / "r.json", verify_receipt_path=rec / "verify.json",
                        clients={"writer": None, "reader": None})


if __name__ == "__main__":
    unittest.main()
