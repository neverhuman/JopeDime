"""Offline proofs for dltrain.family_pin (P2 step 1). No network: S3 goes through real dltrain.s3io guarded clients
built from a fake-credential session, answered by botocore Stubber, with a before-send tripwire.

Covered: identity-rule asserts, payload and meta allowlists, ledger accounting, the S3 budget STOP before any GET,
STOP on every mismatch class, read-only SQLite (a write raises), redaction of sealed members (canary scan), the
create-only/once-per-OUT contract, rehash_inputs, policy serialization, the CPU guard and the import ban.
"""
import ast
import contextlib
import copy
import hashlib
import io
import json
import re
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import boto3
import botocore
from botocore.response import StreamingBody
from botocore.stub import Stubber

from dltrain import common, s3io
from dltrain import family_pin as fp

DS = fp.DATASETS_BUCKET


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def icanon(value) -> bytes:
    """Independent copy of intake_contract.canonical (ensure_ascii=True); fixtures never use the module under test."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def iid(h: str, can: str, s: str) -> str:
    return sha(icanon([h, can, s]))


def jbytes(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, indent=1).encode()


def put(path: Path, data: bytes) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def ident(name: str) -> tuple[str, str]:
    can = sha(f"canonical:{name}".encode())
    return can[:16], can


def sample(name: str) -> str:
    return sha(f"sample:{name}".encode())


def _session():
    return boto3.session.Session(aws_access_key_id="testing", aws_secret_access_key="testing", region_name="us-east-2")


def _client(mode="lane"):
    c = s3io.make_client(mode=mode, session=_session())

    def tripwire(**_):
        raise AssertionError("a request reached before-send in an offline test")
    c.meta.events.register("before-send.*.*", tripwire)
    return c


def _resp(body: bytes, vid: str) -> dict:
    return {"Body": StreamingBody(io.BytesIO(body), len(body)), "VersionId": vid, "ETag": '"' + "0" * 32 + '"', "ContentLength": len(body)}


CANARIES = ("CANARY-FAMILY-T", "CANARY-FAMILY-V", "CanaryPkg", "CanaryItem", "CANARYX2", "CANARY-F2-PATH", "CANARY-F2-URL",
            "CANARY-F2-DISPLAY", "CANARY-F2-META-URL", "CANARY-F2-FAMILY", "CANARY-F3-DISPLAY", "CANARY-X1-DISPLAY", "CANARY-X1-URL",
            "CANARY-Y1-DISPLAY", "CANARY-Y1-URL", "CANARY-X1-META", "CANARY-Y1-META", "PERSONAL-ID-CANARY")


class World:
    """A miniature P2 input world: 3 frozen datasets (F1 plain, F2 sealed by id, F3 sealed by Rdatasets alias), 2 incoming
    items (I1 included, I2 withdrawn), 2 retired siblings, 1 unrelated item, lineage-B v1..v5 with sealed X1/Y1 (records)
    and X2 (no record), 2 seed sources, BeyondArena pointers/parquets, and every fixed input the pin step requires."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.build()

    # ------------------------------------------------------------------ builders
    def meta_body(self, name: str, *, url=None, dsid=None, family=None, nan=False, hunt=False) -> bytes:
        doc = {"feature_columns": [f"{name.lower()}_a", "carat", "depth"], "original_headers": ["A", "carat", "depth", "price"],
               "output_headers": ["a", "carat", "depth", "price"], "target_column": "price", "dropped_feature_columns": [],
               "feature_scaling": [{"min": 0.0, "max": float("nan") if nan else 2.0}], "target_scaling": {"min": 1, "max": 9},
               "source_family": family, "source_dataset_id": dsid or f"{name}-id", "source_platform": "examplehub",
               "source_url": url or f"https://example.org/{name}.csv", "dedupe_fingerprint": sha(name.encode())[:40],
               "display_name": f"{name} display", "acquisition": {"agent_id": "PERSONAL-ID-CANARY", "notes": "x"},
               "source_path": f"/home/x/{name}.csv"}
        if hunt:
            doc["quality_checks"] = {"gates": {"deduplication": {"normalized_url": doc["source_url"], "content_sha256": sha(b"c"),
                                                                 "source_keys": [f"k:{name}"]}, "other_gate": {"agent": "PERSONAL-ID-CANARY"}}}
        return json.dumps(doc, sort_keys=True).encode()

    def build(self):
        R = self.root
        self.run = R / "RUN"
        self.out = R / "out" / "families"
        self.exact_calls, self.cas_calls = [], []
        fixed = {}

        def fix(name, path, data, role="test"):
            put(path, data)
            fixed[name] = fp.FixedInput(path, sha(data), role)
            return data

        # ---------------------------------------------------------- datasets and identities
        self.ids = {n: ident(n) for n in ("F1", "F2", "F3", "I1", "I2", "X1", "X2", "Y1", "U1")}
        H = {n: v[0] for n, v in self.ids.items()}
        self.H = H
        prov = {"F1": {"platform": "pmlb", "dataset_id": "1_x", "url": "https://github.com/EpistasisLab/pmlb/1_x.tsv.gz",
                       "dedupe_fingerprint": "a" * 40, "license": {"status": "recorded", "spdx": "MIT"}},
                "F2": {"platform": "examplehub", "dataset_id": "f2", "url": "https://example.org/CANARY-F2-URL.csv",
                       "dedupe_fingerprint": "b" * 40, "license": {"status": "unknown"}},
                "F3": {"platform": "rdatasets", "dataset_id": "CanaryPkg/CanaryItem",
                       "url": "https://vincentarelbundock.github.io/Rdatasets/csv/CanaryPkg/CanaryItem.csv",
                       "dedupe_fingerprint": "c" * 40, "license": {"status": "unknown"}}}
        # metas (bodies served by the stub)
        self.metas = {
            "F1": self.meta_body("F1", nan=True),
            "F2": self.meta_body("F2", url="https://example.org/CANARY-F2-META-URL.csv", family="CANARY-F2-FAMILY"),
            "F3": self.meta_body("F3", url=prov["F3"]["url"], dsid="CanaryPkg/CanaryItem"),
            "I1": self.meta_body("I1", hunt=True), "I2": self.meta_body("I2", hunt=True), "S2": self.meta_body("S2", hunt=True),
            "X1": self.meta_body("X1", url="https://example.org/CANARY-X1-META.csv"),
            "Y1": self.meta_body("Y1", url="https://example.org/CANARY-Y1-META.csv"),
        }
        self.meta_sha = {k: sha(v) for k, v in self.metas.items()}
        rows, self.contexts = [], {}
        research = R / "research"
        records = R / "inventory" / "dataset-records"
        readiness, catalog = [], []
        t1 = {}
        # ---------------------------------------------------------- frozen
        for n, folds in (("F1", (0, 1)), ("F2", (0,)), ("F3", (0,))):
            h, can = self.ids[n]
            s = sample(n)
            data = f"0.1,0.2,{n}\n0.3,0.4,1\n".encode()
            dsha = sha(data)
            put(research / f"objects/data/sha256/{dsha}.csv.gz", data)
            man = {"dataset_hash": h, "canonical_sha256": can, "sample_identity_sha256": s, "schema": "jope.regression-research-dataset-manifest/v1",
                   "objects": {"data": {"bucket": DS, "key": f"research/regression-2k/v1/objects/data/sha256/{dsha}.csv.gz", "sha256": dsha,
                                        "bytes": len(data), "version_id": f"dv-{n}"}}}
            mbytes = put(research / f"datasets/{h}/{s}/manifest.json", jbytes(man))
            sm = {"bucket": DS, "key": f"research/regression-2k/v1/datasets/{h}/{s}/manifest.json", "version_id": f"smv-{n}",
                  "sha256": sha(mbytes), "bytes": len(mbytes)}
            for f in folds:
                rows.append(self._row(n, "frozen", f, sm, R / "frozen_ctx"))
            t1_doc = {"dataset_hash": h, "canonical_sha256": can, "display_name": f"CANARY-{n}-DISPLAY" if n != "F1" else "F1 display",
                      "provenance": prov[n], "files": {"meta": {"sha256": self.meta_sha[n], "bytes": len(self.metas[n])}}}
            t1[n] = jbytes(t1_doc)
            cref = {"bucket": DS, "key": f"datasets/{h}/manifest.json", "version_id": f"t1v-{n}", "sha256": sha(t1[n]), "bytes": len(t1[n]),
                    "canonical_sha256": can, "provenance": prov[n]}
            readiness.append({"cohort": "frozen", "identity": f"frozen:{h}", "dataset_hash": h, "canonical_sha256": can,
                              "display_name": t1_doc["display_name"], "provenance": prov[n], "catalog_ref": cref})
            catalog.append({"dataset_hash": h, "id": h, "canonical_sha256": can, "provenance": prov[n], "collections": ["registry-real-v1"],
                            "files": {"meta": {"sha256": self.meta_sha[n], "bytes": len(self.metas[n])}}})
            put(records / f"{h}.json", jbytes({"dataset_hash": h, "name": f"CANARY-{n}-DISPLAY",
                                               "meta": {"key": f"blobs/sha256/{self.meta_sha[n]}", "sha256": self.meta_sha[n],
                                                        "bytes": len(self.metas[n]), "version": f"metav-{n}"},
                                               "manifest": {"key": cref["key"], "sha256": cref["sha256"], "bytes": cref["bytes"], "version": cref["version_id"]},
                                               "key": f"blobs/sha256/{'1' * 64}", "sha256": "1" * 64, "bytes": 10, "version": "trv"}))
        self.t1 = t1
        # ---------------------------------------------------------- incoming
        intake = R / "intake"
        vo = intake / "verified-objects"
        self.items, payloads = {}, {}
        for n, status in (("I1", "included"), ("I2", "excluded")):
            h, can = self.ids[n]
            s = sample(n)
            item_id = iid(h, can, s)
            self.items[n] = item_id
            data = f"0.5,0.6,{n}\n".encode()
            dsha = sha(data)
            put(vo / dsha, data)
            man = jbytes({"dataset_hash": h, "canonical_sha256": can, "sample_identity_sha256": s})
            msha = sha(man)
            put(vo / msha, man)
            pm = {"bucket": DS, "key": f"research/regression-2k/incoming/v1/datasets/{h}/{s}/manifest.json", "version_id": f"pmv-{n}",
                  "sha256": msha, "bytes": len(man)}
            po = {"bucket": DS, "key": f"research/regression-2k/incoming/v1/objects/data/sha256/{dsha}.csv.gz", "sha256": dsha,
                  "bytes": len(data), "version_id": f"pdv-{n}"}
            row = self._catalog_row(n, h, can, self.meta_sha[n], len(self.metas[n]))
            d = intake / "items" / item_id
            put(d / "catalog-row.json", icanon(row) + b"\n")
            put(d / "materialization.json", icanon({
                "schema": "jope.regression-research-materialization-receipt/v1", "dataset_hash": h, "sample_identity_sha256": s,
                "manifest": str(vo / msha), "published_manifest": pm, "local_objects": {"data": {"path": str(vo / dsha), **po}},
                "published_objects": {"data": po}}) + b"\n")
            payloads[n] = {"catalog_row": row, "catalog_row_sha256": sha(icanon(row)),
                           "catalog_ref": {"bucket": DS, "key": "hunt/catalog.jsonl", "version_id": "hv", "sha256": "2" * 64, "bytes": 1},
                           "manifest_ref": pm, "objects": {"data": po}, "identity_key": item_id, "identity": [h, can, s],
                           "folds": [{"fold": 0}], "eligibility": {"leakage_quarantine": False}, "receipt": str(d / "materialization.json"),
                           "campaign_id": "c", "corpus_hash": "3" * 64}
            for f in ((0, 1) if n == "I1" else (0,)):
                rows.append(self._row(n, "incoming", f, pm, d / "run/gp/contexts", item_id=item_id, status=status))
        self.payloads = payloads
        # ---------------------------------------------------------- registry (SQLite)
        db = intake / "intake.sqlite3"
        self.db = db
        con = sqlite3.connect(db)
        con.executescript("""
            CREATE TABLE items(id TEXT PRIMARY KEY,dataset TEXT NOT NULL,canonical TEXT NOT NULL,sample TEXT NOT NULL,
                state TEXT NOT NULL,payload TEXT NOT NULL,admitted_ns TEXT NOT NULL,retired_ns TEXT,reason TEXT);
            CREATE TABLE events(seq INTEGER PRIMARY KEY,id TEXT,event TEXT,utc_ns TEXT,payload TEXT);
            CREATE TABLE polls(seq INTEGER PRIMARY KEY,utc_ns TEXT,status TEXT,payload TEXT);
            CREATE TABLE tasks(id TEXT,fold INTEGER,kind TEXT,state TEXT,payload TEXT,PRIMARY KEY(id,fold,kind));
            CREATE TABLE contexts(context TEXT PRIMARY KEY,item_id TEXT NOT NULL);""")
        for seq in range(1, 8):
            con.execute("insert into polls values (?,?,?,?)", (seq, str(seq), "ok", "{}"))
        ins = "insert into items values (?,?,?,?,?,?,?,?,?)"
        for n in ("F1", "F2", "F3"):
            h, can = self.ids[n]
            con.execute(ins, (iid(h, can, sample(n)), h, can, sample(n), "frozen", "{}", "1", None, None))
        con.execute(ins, (self.items["I1"], *self.ids["I1"], sample("I1"), "complete", json.dumps(payloads["I1"]), "2", None, '{"exit_code": 0}'))
        con.execute(ins, (self.items["I2"], *self.ids["I2"], sample("I2"), "retired", json.dumps(payloads["I2"]), "2", "3",
                          "withdrawn_from_hunt_catalog"))
        # siblings: S1 shares I1's meta, S2 (I2's older sample) has its own meta; U1 is outside the epoch
        s1 = iid(*self.ids["I1"], sample("S1"))
        s2 = iid(*self.ids["I2"], sample("S2"))
        p_s2 = copy.deepcopy(payloads["I2"])
        p_s2["catalog_row"]["files"]["meta"] = {"sha256": self.meta_sha["S2"], "bytes": len(self.metas["S2"])}
        u1 = iid(*self.ids["U1"], sample("U1"))
        con.execute(ins, (s1, *self.ids["I1"], sample("S1"), "retired", json.dumps(payloads["I1"]), "1", "2", "missing_from_latest_corpus"))
        con.execute(ins, (s2, *self.ids["I2"], sample("S2"), "retired", json.dumps(p_s2), "1", "2", "missing_from_latest_corpus"))
        con.execute(ins, (u1, *self.ids["U1"], sample("U1"), "complete", json.dumps({"catalog_row": {"display_name": "U1"}}), "1", None, None))
        self.siblings = sorted([s1, s2])
        self.unrelated = u1
        for ctx, owner in ((self.contexts[("I1", 0)], self.items["I1"]), (self.contexts[("I1", 1)], self.items["I1"]),
                         (sha(b"extra-ctx"), self.items["I1"]), (self.contexts[("I2", 0)], self.items["I2"]),
                         (sha(b"s1-ctx"), s1), (sha(b"u1-ctx"), u1)):
            con.execute("insert into contexts values (?,?)", (ctx, owner))
        con.commit()
        con.close()
        # ---------------------------------------------------------- epoch, gate
        epoch = jbytes({"schema": "jope.dl-corpus-epoch/v1", "experiment_id": fp.EXPERIMENT_ID, "epoch": 0, "contexts": rows})
        fix("epoch", self.run / "epochs/epoch-000.json", epoch, "epoch")
        esha = sha(epoch)
        fix("p1_receipt", self.run / "receipts/P1/receipt.json", jbytes({"verdict": "PASS", "outputs": {"epoch": {"sha256": esha}}}), "gate")
        fix("p1_publish", self.run / "receipts/P1/publish.json", jbytes({"refs": {
            "epoch": {"bucket": s3io.BUCKET, "key": f"{s3io.PREFIX_KEY}objects/epoch/sha256/{esha}.json", "version_id": "ev", "sha256": esha,
                      "bytes": len(epoch)},
            "binding": {"bucket": s3io.BUCKET, "key": f"{s3io.PREFIX_KEY}bindings/x/epoch-000.json", "version_id": "bv", "sha256": "4" * 64,
                        "bytes": 9}}}), "gate")
        fix("owner_decisions", R / "gate/owner-decisions.json", jbytes({"decisions": {"DL-D5": "x", "DL-D12": "y"}}), "gate")
        # ---------------------------------------------------------- frozen provenance
        fix("readiness", R / "readiness.json", jbytes({"records": readiness + [{"cohort": "incoming", "dataset_hash": H["I1"]}]}), "frozen_provenance")
        # ---------------------------------------------------------- sealed (lineage-B v1..v5) and sealed references
        sealed_rows = {}
        for n in ("X1", "Y1"):
            h, can = self.ids[n]
            train, test = f"0.1,0.9\n0.2,0.8\n{n}\n".encode(), f"0.3,0.7\n{n}\n".encode()
            row = {"dataset_hash": h, "id": h, "canonical_sha256": can, "display_name": f"CANARY-{n}-DISPLAY",
                   "provenance": {"platform": "examplehub", "dataset_id": f"CANARY-{n}-DISPLAY", "url": f"https://example.org/CANARY-{n}-URL"},
                   "ingest": {"relpath": f"canary/CANARY-{n}-DISPLAY.csv"}, "aliases": [], "collections": ["registry-real-v1"],
                   "target": {"index": -1, "name": "price", "type": "numeric"}, "shape": {"rows": 3},
                   "files": {"train": {"sha256": sha(train), "bytes": len(train), "has_header": False},
                             "test": {"sha256": sha(test), "bytes": len(test), "has_header": False},
                             "meta": {"sha256": self.meta_sha[n], "bytes": len(self.metas[n])}}}
            mb = jbytes(row)
            sealed_rows[n] = (row, mb, train, test)
            put(records / f"{h}.json", jbytes({"dataset_hash": h, "name": f"CANARY-{n}-DISPLAY", "source_key": f"CANARY-{n}-URL",
                                               "meta": {"key": f"blobs/sha256/{self.meta_sha[n]}", "sha256": self.meta_sha[n],
                                                        "bytes": len(self.metas[n]), "version": f"metav-{n}"},
                                               "manifest": {"key": f"datasets/{h}/manifest.json", "sha256": sha(mb), "bytes": len(mb),
                                                            "version": f"t1v-{n}"},
                                               "key": f"blobs/sha256/{sha(train)}", "sha256": sha(train), "bytes": len(train), "version": f"trv-{n}"}))
        catalog.append({"dataset_hash": H["X1"], "id": H["X1"], "canonical_sha256": self.ids["X1"][1], "files": sealed_rows["X1"][0]["files"]})
        catalog.append({"dataset_hash": H["U1"], "id": H["U1"], "canonical_sha256": self.ids["U1"][1]})
        fix("catalog_v1", R / "inventory/catalog.jsonl", b"".join(json.dumps(c, sort_keys=True).encode() + b"\n" for c in catalog), "catalog_v1")
        self.sealed_rows = sealed_rows
        fam_t = {"family": "CANARY-FAMILY-T", "partition": "meta_test", "source_keys": ["rdatasets:OtherPkg/CANARYX2"], "datasets": [
            {"dataset": H["F2"], "source_key": "/abs/CANARY-F2-PATH.csv", "metadata_sha256": self.meta_sha["F2"], "evidence": []},
            {"dataset": H["X2"], "source_key": "rdatasets:OtherPkg/CANARYX2", "evidence": []}]}
        fam_tr = {"family": "train-fam", "partition": "meta_train", "datasets": [{"dataset": H["F1"], "source_key": "rdatasets:x/y"}]}
        fam_v = {"family": "CANARY-FAMILY-V", "partition": "meta_validation", "datasets": [
            {"dataset": H["Y1"], "source_key": "rdatasets:CanaryPkg/CanaryItem", "metadata_sha256": self.meta_sha["Y1"],
             "evidence": [{"source_url": "https://vincentarelbundock.github.io/Rdatasets/doc/CanaryPkg/CanaryItem.html", "finding": "f"}]},
            {"dataset": H["X1"], "source_key": "/abs/CANARY-X1-URL", "metadata_sha256": self.meta_sha["X1"], "evidence": []}]}
        lb = R / "lb"
        prev = None
        self.lb_names = []
        for v in range(1, 6):
            fams = {"CANARY-FAMILY-T": fam_t, "train-fam": fam_tr} | ({"CANARY-FAMILY-V": fam_v} if v >= 2 else {})
            doc = jbytes({"schema": "parallel.family_registry.v1", "parent": None if prev is None else {"path": "p", "sha256": prev},
                          "families": fams})
            name = f"lineage_b_qualification_v{v}"
            fix(name, lb / f"{sha(doc)}-qualification-v{v}.json", doc, "sealed_registry")
            self.lb_names.append(name)
            prev = sha(doc)
        pol = jbytes({"sealed_partitions": ["meta_validation", "meta_test"], "withdrawn_families": []})
        fix("lineage_b_family_exclusion_policy", lb / f"{sha(pol)}-family-exclusion-policy.json", pol, "sealed_policy")
        for name in ("native_index", "canonical_exclusions", "corpus_audit_v2", "split_registry_native", "legacy_inventory_tsv"):
            fix(name, R / "misc" / name, f"{name} bytes\n".encode(), "crosscheck")
        # ---------------------------------------------------------- seeds, ten-case views, BeyondArena
        seed_rows = [{"program": {"program_sha256": sha(f"p{i}".encode()), "dsl": f"p{i}"},
                      "source_evidence": [{"source_view": f"{src}__r0__f0", "source_rank": i}]} for i, src in enumerate(["srcA", "srcA", "srcB", "srcB"])]
        fix("seed_pool_recovery192", R / "seeds/recovery-seed-pool.jsonl", b"".join(json.dumps(r).encode() + b"\n" for r in seed_rows), "seed_pool")
        self.seed_view_files = {}
        for src in ("srcA", "srcB"):
            c = put(R / f"v3/contexts/{src}__r0__f0/bootstrap-candidates.jsonl", f"{src} candidates\n".encode())
            p = put(R / f"v3/contexts/{src}__r0__f0/prepared.json", jbytes({"view_id": f"{src}__r0__f0"}))
            self.seed_view_files[src] = (sha(c), sha(p))
        ba = R / "ba"
        idx = fix("beyondarena_index", ba / "pointers/index.json", jbytes({"datasets": 5}), "beyondarena")
        objects = [{"key": "benchmarks/BeyondArena/rev1/index.json", "sha256": sha(idx), "bytes": len(idx), "version_id": "iv", "verified": True}]
        for name, parquet, matches in (("srcA", b"PAR1 srcA", True), ("srcB", b"PAR1 srcB", True), ("v6x", b"PAR1 v6x", True),
                                       ("stale7", b"PAR1 stale7", False), ("ptronly", None, True)):
            pq_sha = sha(parquet) if (parquet and matches) else "5" * 64
            if parquet is not None:
                put(ba / f"source-cache/{name}/dataset.parquet", parquet)
            ptr = jbytes({"name": name, "revision": "rev1", "bucket": DS, "target": "y",
                          "files": {"dataset.parquet": {"key": f"raw/BeyondArena/rev1/{name}/u/dataset.parquet", "sha256": pq_sha,
                                                        "bytes": len(parquet or b""), "version_id": f"pq-{name}"}}})
            put(ba / f"pointers/sources/{name}.json", ptr)
            objects.append({"key": f"benchmarks/BeyondArena/rev1/sources/{name}.json", "sha256": sha(ptr), "bytes": len(ptr),
                            "version_id": f"pv-{name}", "verified": True})
        put(ba / "source-cache/nopointer/dataset.parquet", b"PAR1 orphan")
        fix("beyondarena_pointer_receipt", ba / "pointer-receipt.json", jbytes({"revision": "rev1", "objects": objects, "datasets": 5}), "beyondarena")
        # ---------------------------------------------------------- exposure and encoder identities
        self.target_artifact = sha(b"target weights")
        self.tabpfn_ckpt = sha(b"tabpfn weights")
        fix("target_metadata", R / "exposure/target_metadata", jbytes({"artifact_sha256": self.target_artifact, "fit_records": ["a", "b"]}), "exposure")
        fix("tabpfn35_encoder_spec", R / "tabpfn/encoder-spec.json",
            jbytes({"checkpoint_sha256": self.tabpfn_ckpt, "base_weights_modified": False}), "tabpfn35")
        put(R / "weights/target.bin", b"target weights")
        identities = {"encoder:target_fused_v1": fp.Identity(R / "weights/target.bin", self.target_artifact, "encoder_weights"),
                      "encoder:tabpfn35": fp.Identity(None, self.tabpfn_ckpt, "encoder_weights")}
        self.classic = {"schema": fp.CLASSIC_ALIASES_SCHEMA, "entries": [
            {"name": "srcA", "action": "union", "members": [H["F1"]]},
            {"name": "review1", "action": "review", "members": [H["I1"], "f" * 16]},
            {"name": "keyword:abc", "action": "review", "keyword": "abc", "members": []}]}
        self.fixed = fixed
        self.cfg = fp.PinConfig(fixed=fixed, identities=identities, registry_db=db, incoming_items=intake / "items",
                                frozen_research_root=research, dataset_records=records, ba_pointers=ba / "pointers/sources",
                                ba_source_cache=ba / "source-cache", tencase_v3_contexts=R / "v3/contexts",
                                seed_view_files=self.seed_view_files, classic_aliases=self.classic, cpu_admission=R / "no-admission.json",
                                expected_seed_sources=2, readback_sample=2)
        self.policy = {"SEALED_PARTITIONS": frozenset({"meta_validation", "meta_test"}), "FID_DOMAIN": b"jope.dl-family-id/v1\0",
                       "LSH": (16, 4), "RESTRICTIVE_NOTE": re.compile(r"(?i)nc"), "FRESH_COHORTS": {"incoming"}}
        self._plan_calls()

    def _catalog_row(self, n, h, can, meta_sha, meta_bytes):
        return {"schema": "jope.dataset-catalog/v1", "id": h, "dataset_hash": h, "canonical_sha256": can,
                "display_name": f"Dätaset {n} ✓ (non-ASCII)", "aliases": [], "collections": ["jailgun-regression-hunt"],
                "ingest": {"source": "link", "tool_version": "jailgun-regression-hunt/0.1.0", "relpath": f"{n}.csv"},
                "provenance": {"platform": "nyc open data / socrata", "dataset_id": f"{n.lower()}-abcd",
                               "url": f"https://data.example.gov/api/views/{n.lower()}-abcd/rows.csv", "dedupe_fingerprint": "d" * 40,
                               "license": {"status": "unknown", "note": "reported by discovery agent: CC BY 4.0"}},
                "files": {"meta": {"sha256": meta_sha, "bytes": meta_bytes}}, "target": {"index": -1, "name": "y", "type": "numeric"}}

    def _row(self, n, cohort, fold, sm, ctx_root, item_id=None, status="included"):
        h, can = self.ids[n]
        ctx = sha(f"ctx:{n}:{fold}".encode())
        self.contexts[(n, fold)] = ctx
        view = f"{h}__r0__f{fold}"
        raw = sha(f"raw:{n}:{fold}".encode())
        prep = jbytes({"view_id": view, "source_manifest": sm,
                       "context": {"context_sha256": ctx, "dataset_hash": h, "input_raw_sha256": raw, "evaluator_sha256": "6" * 64,
                                   "candidate_timeout_ms": 90000, "train_rows": 8, "test_rows": 2, "encoded_features": 3}})
        p = ctx_root / view / "prepared.json"
        put(p, prep)
        row = {"cohort": cohort, "view_id": view, "dataset_hash": h, "sample_identity_sha256": sample(n), "fold": fold, "context_sha256": ctx,
               "status": status, "exclusion_reason": None if status == "included" else "withdrawn",
               "prepared_ref": {"path": str(p), "sha256": sha(prep), "bytes": len(prep)}, "input_ref": {"raw_sha256": raw}}
        if item_id:
            row["item_id"] = item_id
        return row

    def _plan_calls(self):
        """Expected request order per client (computed here independently of family_pin's planner)."""
        H = self.H
        for n in sorted(("F1", "F2", "F3"), key=lambda k: H[k]):
            self.exact_calls.append((f"blobs/sha256/{self.meta_sha[n]}", f"metav-{n}", self.metas[n]))
        distinct = sorted({(H["I1"], self.meta_sha["I1"], "I1"), (H["I2"], self.meta_sha["I2"], "I2"), (H["I2"], self.meta_sha["S2"], "S2")})
        for _, msha, n in distinct:
            self.cas_calls.append((f"blobs/sha256/{msha}", self.metas[n]))
        for n in sorted(("X1", "Y1"), key=lambda k: H[k]):
            row, mb, train, test = self.sealed_rows[n]
            self.exact_calls.append((f"datasets/{H[n]}/manifest.json", f"t1v-{n}", mb))
            self.exact_calls.append((f"blobs/sha256/{sha(train)}", f"trv-{n}", train))
            self.exact_calls.append((f"blobs/sha256/{self.meta_sha[n]}", f"metav-{n}", self.metas[n]))
            self.cas_calls.append((f"blobs/sha256/{sha(test)}", test))
        order = sorted(("F1", "F2", "F3"), key=lambda k: (sha(fp.READBACK_DOMAIN + H[k].encode()), H[k]))[:2]
        for n in order:
            self.exact_calls.append((f"datasets/{H[n]}/manifest.json", f"t1v-{n}", self.t1[n]))

    def clients(self, exact_calls=None, cas_calls=None):
        ex, cas = _client(), _client(mode="cas-read")
        se, sc = Stubber(ex), Stubber(cas)
        for key, vid, body in (self.exact_calls if exact_calls is None else exact_calls):
            se.add_response("get_object", _resp(body, vid), {"Bucket": DS, "Key": key, "VersionId": vid})
        for i, (key, body) in enumerate(self.cas_calls if cas_calls is None else cas_calls):
            sc.add_response("get_object", _resp(body, f"casv-{i}"), {"Bucket": DS, "Key": key})
        se.activate()
        sc.activate()
        return {"exact": ex, "cas": cas}, (se, sc)

    def pin(self, **kw):
        clients, stubs = kw.pop("clients_stubs", None) or self.clients()
        res = fp.pin(self.out, policy=self.policy, config=kw.pop("config", self.cfg), clients=clients, guard=fp.NoGuard(), **kw)
        return res, stubs


def tree_text(root: Path) -> str:
    parts = []
    for p in sorted(Path(root).rglob("*")):
        if p.is_file():
            parts.append(p.read_bytes().decode("utf-8", "replace"))
    return "\n".join(parts)


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="p2pin-")
        self.addCleanup(self._tmp.cleanup)
        self.w = World(Path(self._tmp.name))


# ------------------------------------------------------------------------------------------------------ end to end
class PinEndToEnd(Base):
    def test_pass_writes_inputs_and_alias_sources(self):
        ledger = self.w.root / "receipts" / "pin-ledger.jsonl"
        receipt = self.w.root / "receipts" / "pin-receipt.json"
        res, (se, sc) = self.w.pin(ledger_path=ledger, receipt_path=receipt)
        se.assert_no_pending_responses()
        sc.assert_no_pending_responses()
        self.assertEqual(res["verdict"], "PASS")
        c = res["counts"]
        self.assertEqual((c["epoch_datasets"], c["epoch_items"], c["frozen_datasets"], c["incoming_datasets"]), (5, 2, 3, 2))
        self.assertEqual((c["snapshot_items"], c["siblings"], c["poll_seq"]), (7, 2, 7))   # U1 filtered out
        self.assertEqual(c["snapshot_contexts"], 5)
        self.assertEqual((c["metas"], c["metas_redacted"], c["metas_nonfinite_to_null"]), (6, 2, 1))
        self.assertEqual((c["sealed_S"], c["sealed_in_epoch"], c["sealed_refs"], c["sealed_content_unavailable"]), (4, 1, 2, 1))
        self.assertEqual((c["base_sealed_epoch_datasets"], c["readback_ok"], c["seed_sources"]), (2, 2, 2))
        self.assertEqual((c["v6_names"], c["v6_mismatch"], c["ba_pointers"]), (3, 2, 5))
        self.assertEqual((c["s3_gets"], res["s3_totals"]["gets"]), (16, 16))
        doc = json.loads((self.w.out / "alias-sources.json").read_bytes())
        self.assertEqual(doc["schema"], fp.ALIAS_SOURCES_SCHEMA)
        self.assertEqual(doc["experiment_id"], fp.EXPERIMENT_ID)
        self.assertEqual(sorted(doc["sealed"]["base_sealed_epoch_datasets"]), sorted([self.w.H["F2"], self.w.H["F3"]]))
        self.assertIn("lineage_b_rdatasets_alias", doc["sealed"]["base_sealed_epoch_datasets"][self.w.H["F3"]])
        self.assertEqual(doc["sealed"]["content_unavailable"], [self.w.H["X2"]])
        self.assertEqual(set(doc["sealed"]["refs"]), {self.w.H["X1"], self.w.H["Y1"]})
        self.assertEqual(doc["beyondarena"]["v6_names"], ["srcA", "srcB", "v6x"])
        self.assertEqual(doc["seeds"]["seed_sources"], ["srcA", "srcB"])
        # policy serialized deterministically (sets sorted, bytes as text, tuples as lists, regex as pattern)
        self.assertEqual(doc["policy"]["SEALED_PARTITIONS"], ["meta_test", "meta_validation"])
        self.assertEqual(doc["policy"]["FID_DOMAIN"], "jope.dl-family-id/v1\0")
        self.assertEqual(doc["policy"]["LSH"], [16, 4])
        # inputs: unique names, every local file present and hashed
        names = [e["name"] for e in doc["inputs"]]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(fp.rehash_inputs(doc, self.w.out)["ok"])
        # the S3 ledger: in alias-sources, in the journal, and in the receipt, with consistent totals
        lines = [json.loads(x) for x in ledger.read_text().splitlines()]
        self.assertEqual(lines, doc["s3_reads"])
        self.assertEqual(len(lines), 16)
        self.assertTrue(all(e["status"] == "ok" for e in lines))
        self.assertEqual(sum(e["bytes"] for e in lines), doc["s3_totals"]["bytes"])
        self.assertEqual([e["seq"] for e in lines], list(range(1, 17)))
        for e in lines:
            self.assertTrue(set(e) >= {"op", "bucket", "key", "version_id", "bytes", "sha256", "status", "utc_us"})
            self.assertEqual(e["sha256"], e["expected_sha256"])
            self.assertTrue(e["version_id"])
        rec = json.loads(receipt.read_bytes())
        self.assertEqual((rec["verdict"], rec["counts"]["s3_gets"]), ("PASS", 16))
        # snapshot: payload allowlist, sibling and frozen roles, no unrelated item
        snap = json.loads((self.w.out / "inputs/registry-snapshot.json").read_bytes())
        self.assertEqual(snap["poll_seq"], 7)
        self.assertNotIn(self.w.unrelated, {it["id"] for it in snap["items"]})
        for it in snap["items"]:
            self.assertLessEqual(set(it["payload"]), set(fp.PAYLOAD_KEEP))
        roles = sorted(it["epoch_role"] for it in snap["items"])
        self.assertEqual(roles, ["frozen"] * 3 + ["incoming"] * 2 + ["sibling"] * 2)
        i1 = next(it for it in snap["items"] if it["id"] == self.w.items["I1"])
        self.assertEqual(set(i1["payload"]), set(fp.PAYLOAD_KEEP))
        self.assertEqual(sha(icanon(i1["payload"]["catalog_row"])), i1["payload"]["catalog_row_sha256"])
        # metas: allowlist only; personal identifiers stripped; NaN became null
        f1 = json.loads((self.w.out / f"inputs/meta/{self.w.meta_sha['F1']}.json").read_bytes())
        self.assertLessEqual(set(f1), set(fp.META_KEEP) | {"quality_checks"})
        self.assertNotIn("acquisition", f1)
        self.assertIsNone(f1["feature_scaling"][0]["max"])
        i1m = json.loads((self.w.out / f"inputs/meta/{self.w.meta_sha['I1']}.json").read_bytes())
        self.assertEqual(set(i1m["quality_checks"]), {"gates"})
        self.assertEqual(set(i1m["quality_checks"]["gates"]), {"deduplication"})
        # sealed ref raw blobs are content-addressed; extracts are redacted
        x1 = self.w.sealed_rows["X1"]
        self.assertEqual((self.w.out / f"inputs/sealed-ref/{sha(x1[2])}").read_bytes(), x1[2])
        self.assertEqual((self.w.out / f"inputs/sealed-ref/{sha(x1[3])}").read_bytes(), x1[3])
        man = json.loads((self.w.out / f"inputs/sealed-ref/manifest/{sha(x1[1])}.json").read_bytes())
        self.assertEqual(set(man["display_name"]), {"value_sha256"})
        self.assertEqual(man["files"]["test"]["sha256"], sha(x1[3]))

    def test_sealed_members_never_written_or_returned_in_clear(self):
        ledger = self.w.root / "receipts" / "pin-ledger.jsonl"
        receipt = self.w.root / "receipts" / "pin-receipt.json"
        res, _ = self.w.pin(ledger_path=ledger, receipt_path=receipt)
        text = (tree_text(self.w.out) + tree_text(self.w.root / "receipts") + json.dumps(res)).lower()
        for canary in CANARIES:
            self.assertNotIn(canary.lower(), text, canary)
        # the redacted metas still carry the column names the name screen needs
        f2 = json.loads((self.w.out / f"inputs/meta/{self.w.meta_sha['F2']}.json").read_bytes())
        self.assertEqual(f2["feature_columns"], ["f2_a", "carat", "depth"])
        self.assertEqual(f2["source_url"], fp.value_sha256("https://example.org/CANARY-F2-META-URL.csv"))
        # the hash form is checkable without plaintext: an epoch value maps to the same value_sha256
        self.assertEqual(f2["source_family"], fp.value_sha256("CANARY-F2-FAMILY"))

    def test_pin_once_per_out_and_no_get_on_second_attempt(self):
        self.w.pin()
        empty, stubs = self.w.clients(exact_calls=[], cas_calls=[])
        with self.assertRaisesRegex(fp.PinStop, "already pinned"):
            self.w.pin(clients_stubs=(empty, stubs))
        stubs[0].assert_no_pending_responses()

    def test_partial_inputs_without_alias_sources_stop(self):
        put(self.w.out / "inputs/registry-snapshot.json", b"{}")
        empty, stubs = self.w.clients(exact_calls=[], cas_calls=[])
        with self.assertRaisesRegex(fp.PinStop, "earlier attempt"):
            self.w.pin(clients_stubs=(empty, stubs))

    def test_budget_stop_before_any_get_and_out_untouched(self):
        cfg = fp.PinConfig(**{**self.w.cfg.__dict__, "max_gets": 15})
        empty, stubs = self.w.clients(exact_calls=[], cas_calls=[])
        receipt = self.w.root / "receipts" / "stop.json"
        with self.assertRaisesRegex(fp.PinStop, "S3 plan: 16 GETs > 15"):
            self.w.pin(config=cfg, clients_stubs=(empty, stubs), receipt_path=receipt)
        self.assertFalse(self.w.out.exists())
        rec = json.loads(receipt.read_bytes())
        self.assertEqual((rec["verdict"], rec["stage"], rec["s3_totals"]["gets"]), ("STOP", "local", 0))
        cfg = fp.PinConfig(**{**self.w.cfg.__dict__, "max_bytes": 100})
        with self.assertRaisesRegex(fp.PinStop, "S3 plan: .* B > 100 B"):
            self.w.pin(config=cfg, clients_stubs=(empty, stubs))

    def test_get_mismatch_stops_with_ledger_and_out_untouched(self):
        calls = list(self.w.exact_calls)
        key, vid, body = calls[1]
        calls[1] = (key, vid, body + b" ")                       # a frozen meta with the wrong bytes
        clients, stubs = self.w.clients(exact_calls=calls)
        ledger = self.w.root / "receipts" / "ledger.jsonl"
        receipt = self.w.root / "receipts" / "stop.json"
        with self.assertRaisesRegex(fp.PinStop, "sha256, bytes mismatch"):
            self.w.pin(clients_stubs=(clients, stubs), ledger_path=ledger, receipt_path=receipt)
        lines = [json.loads(x) for x in ledger.read_text().splitlines()]
        self.assertEqual([e["status"] for e in lines], ["ok", "mismatch:sha256,bytes"])
        self.assertFalse(self.w.out.exists())
        rec = json.loads(receipt.read_bytes())
        self.assertEqual((rec["stage"], rec["s3_totals"]["gets"], rec["s3_totals"]["failed"]), ("s3", 2, 1))

    def test_policy_must_be_data_and_match_dl_d12(self):
        with self.assertRaisesRegex(fp.PinStop, "not JSON-representable"):
            fp.pin(self.w.out, policy={"known": lambda v: v}, config=self.w.cfg, clients={}, guard=fp.NoGuard())
        with self.assertRaisesRegex(fp.PinStop, "DL-D12"):
            fp.pin(self.w.out, policy={"sealed_partitions": ["meta_test"]}, config=self.w.cfg, clients={}, guard=fp.NoGuard())
        self.assertFalse(self.w.out.exists())

    def test_sealed_test_blob_uses_an_exact_get_when_the_manifest_pins_a_version(self):
        H = self.w.H
        first = min(("X1", "Y1"), key=lambda k: H[k])
        row, mb, train, test = self.w.sealed_rows[first]
        row = copy.deepcopy(row)
        row["files"]["test"]["version_id"] = "tv-exact"
        mb2 = jbytes(row)
        rp = Path(self.w.cfg.dataset_records) / f"{H[first]}.json"
        rec = json.loads(rp.read_bytes())
        rec["manifest"].update(sha256=sha(mb2), bytes=len(mb2))
        rp.write_bytes(jbytes(rec))
        exact = [(k, v, mb2 if k == f"datasets/{H[first]}/manifest.json" else b) for k, v, b in self.w.exact_calls]
        i = next(n for n, c in enumerate(exact) if c[0] == f"blobs/sha256/{self.w.meta_sha[first]}")
        exact.insert(i + 1, (f"blobs/sha256/{sha(test)}", "tv-exact", test))
        cas = [c for c in self.w.cas_calls if c[0] != f"blobs/sha256/{sha(test)}"]
        res, (se, sc) = self.w.pin(clients_stubs=self.w.clients(exact_calls=exact, cas_calls=cas))
        se.assert_no_pending_responses()
        sc.assert_no_pending_responses()
        doc = json.loads((self.w.out / "alias-sources.json").read_bytes())
        e = fp.input_index(doc)[f"sealed_ref_test:{H[first]}"]
        self.assertEqual((e["read_mode"], e["s3"]["version_id"]), ("exact", "tv-exact"))
        self.assertEqual(res["counts"]["s3_gets"], 16)

    def test_cli_refuses_outputs_outside_the_run_tree(self):
        run = self.w.root / "RUN"                       # never the live RUN tree, even for a refused call
        with mock.patch.object(fp, "RUN", run), contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(fp._main(["--out", str(self.w.out)]), 3)
            self.assertEqual(fp._main(["--out", str(run / "families"), "--receipt", str(self.w.root / "r.json")]), 3)
            self.assertEqual(fp._main(["--out", str(run / "families"), "--ledger", str(run / "receipts/P1/x.jsonl")]), 3)
        self.assertEqual(err.getvalue().count('"verdict": "STOP"'), 3)
        self.assertFalse(self.w.out.exists() or (run / "families").exists())

    def test_preflight_makes_no_request_and_writes_nothing(self):
        empty, stubs = self.w.clients(exact_calls=[], cas_calls=[])
        receipt = self.w.root / "receipts" / "pre.json"
        res, _ = self.w.pin(clients_stubs=(empty, stubs), preflight=True, receipt_path=receipt)
        self.assertEqual(res["verdict"], "PREFLIGHT_PASS")
        self.assertEqual((res["s3_plan"]["planned_gets"], res["counts"]["base_sealed_epoch_datasets"], res["counts"]["metas_to_redact"]), (16, 2, 2))
        self.assertFalse(self.w.out.exists() or receipt.exists())
        res, _ = self.w.pin()                                # the real pin can follow on the same OUT
        self.assertEqual(res["verdict"], "PASS")

    def test_rehash_inputs_detects_a_changed_source(self):
        self.w.pin()
        doc = fp.load_alias_sources(self.w.out / "alias-sources.json")
        tsv = Path(self.w.fixed["legacy_inventory_tsv"].path)
        tsv.write_bytes(b"changed\n")
        res = fp.rehash_inputs(doc, self.w.out)
        self.assertFalse(res["ok"])
        self.assertEqual([m["name"] for m in res["mismatches"]], ["legacy_inventory_tsv"])
        self.assertTrue(fp.rehash_inputs(doc, self.w.out, skip_roles=("crosscheck",))["ok"])


# ------------------------------------------------------------------------------------------------------ local STOPs
class LocalStops(Base):
    def assert_stop(self, regex, **kw):
        empty, stubs = self.w.clients(exact_calls=[], cas_calls=[])
        with self.assertRaisesRegex(fp.PinStop, regex):
            self.w.pin(clients_stubs=(empty, stubs), **kw)
        self.assertFalse(self.w.out.exists())

    def test_fixed_input_changed(self):
        Path(self.w.fixed["readiness"].path).write_bytes(b"{}")
        self.assert_stop("readiness: sha256")

    def test_epoch_sha_mismatch(self):
        with self.assertRaisesRegex(fp.PinStop, "epoch sha256"):
            fp.load_epoch(self.w.fixed["epoch"].path, "0" * 64)

    def test_prepared_json_changed(self):
        p = Path(next(iter(self.w.root.glob("frozen_ctx/*/prepared.json"))))
        p.write_bytes(p.read_bytes() + b" ")
        self.assert_stop("prepared.json of .*sha256/bytes differ")

    def test_catalog_row_bytes_must_be_intake_canonical(self):
        d = self.w.root / "intake/items" / self.w.items["I1"]
        row = self.w.payloads["I1"]["catalog_row"]
        self.assertNotEqual(common.canonical(row), icanon(row))                    # the row carries non-ASCII text
        (d / "catalog-row.json").write_bytes(common.canonical(row) + b"\n")         # UTF-8 form: not the intake bytes
        self.assert_stop("catalog-row.json bytes != canonical")

    def test_catalog_row_sha256_mismatch(self):
        con = sqlite3.connect(self.w.db)
        p = dict(self.w.payloads["I1"], catalog_row_sha256="0" * 64)
        con.execute("update items set payload=? where id=?", (json.dumps(p), self.w.items["I1"]))
        con.commit()
        con.close()
        self.assert_stop("catalog_row_sha256 mismatch")

    def test_incoming_data_object_changed(self):
        m = json.loads((self.w.root / "intake/items" / self.w.items["I2"] / "materialization.json").read_bytes())
        Path(m["local_objects"]["data"]["path"]).write_bytes(b"tampered")
        self.assert_stop("incoming_data:.*sha256")

    def test_identity_rule_violation(self):
        con = sqlite3.connect(self.w.db)
        h, can = self.w.ids["I1"]
        con.execute("insert into items values (?,?,?,?,?,?,?,?,?)", ("9" * 64, h, can, sample("Z"), "retired", "{}", "1", None, None))
        con.commit()
        con.close()
        self.assert_stop("id != sha256")

    def test_lineage_b_parent_chain(self):
        v3 = self.w.fixed["lineage_b_qualification_v3"]
        doc = json.loads(Path(v3.path).read_bytes())
        doc["parent"]["sha256"] = "0" * 64
        data = jbytes(doc)
        newp = Path(v3.path).with_name(f"{sha(data)}-qualification-v3.json")
        newp.write_bytes(data)
        fixed = dict(self.w.cfg.fixed, lineage_b_qualification_v3=fp.FixedInput(newp, sha(data), "sealed_registry"))
        cfg = fp.PinConfig(**{**self.w.cfg.__dict__, "fixed": fixed})
        self.assert_stop("lineage_b_qualification_v3: parent sha256", config=cfg)

    def test_exclusion_policy_partitions_must_match(self):
        pol = jbytes({"sealed_partitions": ["meta_test"]})
        p = Path(self.w.root / "lb" / f"{sha(pol)}-family-exclusion-policy.json")
        p.write_bytes(pol)
        fixed = dict(self.w.cfg.fixed, lineage_b_family_exclusion_policy=fp.FixedInput(p, sha(pol), "sealed_policy"))
        self.assert_stop("sealed_partitions differ", config=fp.PinConfig(**{**self.w.cfg.__dict__, "fixed": fixed}))

    def test_classic_alias_union_member_must_be_in_epoch(self):
        bad = copy.deepcopy(self.w.classic)
        bad["entries"][0]["members"] = ["e" * 16]
        self.assert_stop("not an epoch dataset", config=fp.PinConfig(**{**self.w.cfg.__dict__, "classic_aliases": bad}))
        bad = copy.deepcopy(self.w.classic)
        bad["entries"][0]["name"] = "not-a-seed"
        self.assert_stop("must name a seed source", config=fp.PinConfig(**{**self.w.cfg.__dict__, "classic_aliases": bad}))

    def test_seed_parquet_must_equal_pointer(self):
        (self.w.root / "ba/source-cache/srcB/dataset.parquet").write_bytes(b"PAR1 other")
        self.assert_stop("seed source srcB: parquet differs")

    def test_frozen_meta_pin_must_match_catalog(self):
        rp = Path(self.w.cfg.dataset_records) / f"{self.w.H['F1']}.json"
        rec = json.loads(rp.read_bytes())
        rec["meta"]["bytes"] += 1
        rp.write_bytes(jbytes(rec))
        self.assert_stop("catalog v1 files.meta != record.meta")


# ------------------------------------------------------------------------------------------------------ units
def _snap(items):
    return {"items": items, "contexts": [], "poll_seq": 1, "utc_us": 1, "db_path": "x", "items_total_in_db": len(items)}


def _item(name, state="frozen", sample_name=None, item_id=None):
    h, can = ident(name)
    s = sample(sample_name or name)
    return {"id": item_id or iid(h, can, s), "dataset": h, "canonical": can, "sample": s, "state": state, "reason": None,
            "admitted_ns": "1", "retired_ns": None, "payload_full": {}}


class RegistryRules(unittest.TestCase):
    def ep_prep(self, names=("A",)):
        frozen = {ident(n)[0] for n in names}
        sm = {ident(n)[0]: {"key": f"research/regression-2k/v1/datasets/{ident(n)[0]}/{sample(n)}/manifest.json", "sha256": "0" * 64}
              for n in names}
        return {"frozen": frozen, "rows": [], "ITEMS": set()}, {"frozen_source_manifest": sm}

    def test_identity_rule_and_dataset_prefix(self):
        ep, prep = self.ep_prep()
        fp.check_registry(_snap([_item("A")]), ep, prep)
        with self.assertRaisesRegex(fp.PinStop, "id != sha256"):
            fp.check_registry(_snap([_item("A", item_id="1" * 64)]), ep, prep)
        it = _item("A")
        it["dataset"] = "0" * 16
        it["id"] = iid(it["dataset"], it["canonical"], it["sample"])
        with self.assertRaisesRegex(fp.PinStop, "canonical\\[:16\\]"):
            fp.check_registry(_snap([it]), ep, prep)

    def test_exactly_one_frozen_item_with_the_manifest_sample(self):
        ep, prep = self.ep_prep()
        with self.assertRaisesRegex(fp.PinStop, "2 registry items in state 'frozen'"):
            fp.check_registry(_snap([_item("A"), _item("A", sample_name="A2")]), ep, prep)
        with self.assertRaisesRegex(fp.PinStop, "0 registry items"):
            fp.check_registry(_snap([_item("A", state="retired")]), ep, prep)
        with self.assertRaisesRegex(fp.PinStop, "registry sample != source_manifest sample"):
            fp.check_registry(_snap([_item("A", sample_name="other")]), ep, prep)

    def test_epoch_items_must_exist_with_same_dataset_and_sample(self):
        ep, prep = self.ep_prep(())
        inc = _item("B", state="complete")
        ep["ITEMS"] = {inc["id"]}
        ep["rows"] = [{"item_id": inc["id"], "dataset_hash": inc["dataset"], "sample_identity_sha256": inc["sample"]}]
        fp.check_registry(_snap([inc]), ep, prep)
        ep["rows"][0]["sample_identity_sha256"] = "f" * 64
        with self.assertRaisesRegex(fp.PinStop, "dataset/sample differ"):
            fp.check_registry(_snap([inc]), ep, prep)
        with self.assertRaisesRegex(fp.PinStop, "not in the registry snapshot"):
            fp.check_registry(_snap([]), ep, prep)


class ReadOnlySqlite(unittest.TestCase):
    def test_writes_raise_and_the_file_is_unchanged(self):
        with tempfile.TemporaryDirectory(prefix="p2pin-db-") as d:
            db = Path(d) / "r.sqlite3"
            con = sqlite3.connect(db)
            con.executescript("create table items(id text primary key, dataset text, canonical text, sample text, state text, reason text,"
                              " admitted_ns text, retired_ns text, payload text); create table contexts(context text, item_id text);"
                              " create table polls(seq integer primary key);")
            con.execute("insert into polls values (5)")
            h, can = ident("A")
            con.execute("insert into items values (?,?,?,?,?,?,?,?,?)", (iid(h, can, sample("A")), h, can, sample("A"), "frozen",
                                                                         None, "1", None, "{}"))
            con.execute("insert into items values (?,?,?,?,?,?,?,?,?)", ("x", "ffff", "c", "s", "complete", None, "1", None, "{}"))
            con.execute("insert into contexts values ('c1', 'x')")
            con.commit()
            con.close()
            before = sha(db.read_bytes())
            ro = fp.connect_ro(db)
            for stmt in ("insert into polls values (6)", "delete from items", "create table t(x)", "update items set state='x'"):
                with self.assertRaises(sqlite3.OperationalError):
                    ro.execute(stmt)
            ro.close()
            snap = fp.registry_snapshot(db, {h})
            self.assertEqual((snap["poll_seq"], len(snap["items"]), snap["contexts"]), (5, 1, []))
            self.assertEqual(sha(db.read_bytes()), before)
            self.assertEqual(sorted(p.name for p in Path(d).iterdir()), ["r.sqlite3"])
            with self.assertRaisesRegex(fp.PinStop, "missing"):
                fp.connect_ro(Path(d) / "absent.sqlite3")


class MetaAllowlist(unittest.TestCase):
    def body(self):
        return json.dumps({"feature_columns": ["a"], "target_column": "y", "feature_scaling": [float("inf")], "source_url": "https://e/x",
                           "source_family": None, "display_name": "secret display", "acquisition": {"agent_id": "PERSONAL-ID-CANARY"},
                           "quality_checks": {"gates": {"deduplication": {"normalized_url": "https://e/x", "source_keys": ["k"]},
                                                        "agent": "PERSONAL-ID-CANARY"}, "other": 1}}).encode()

    def test_only_allowlisted_fields_survive(self):
        ext, info = fp.meta_extract(self.body())
        self.assertEqual(set(ext), {"feature_columns", "target_column", "feature_scaling", "source_url", "source_family", "quality_checks"})
        self.assertEqual(ext["quality_checks"], {"gates": {"deduplication": {"normalized_url": "https://e/x", "source_keys": ["k"]}}})
        self.assertEqual(ext["feature_scaling"], [None])
        self.assertEqual(info["nonfinite_to_null"], 1)
        self.assertNotIn("PERSONAL-ID-CANARY", json.dumps(ext))
        common.canonical(ext)   # serializable

    def test_redaction_hashes_identifying_values_only(self):
        ext, info = fp.meta_extract(self.body(), redact=True)
        self.assertTrue(info["redacted"])
        self.assertEqual(ext["feature_columns"], ["a"])
        self.assertEqual(ext["source_url"], fp.value_sha256("https://e/x"))
        self.assertIsNone(ext["source_family"])
        dd = ext["quality_checks"]["gates"]["deduplication"]
        self.assertEqual(dd["source_keys"], [fp.value_sha256("k")])
        self.assertNotIn("https://e/x", json.dumps(ext))

    def test_non_json_and_non_object_stop(self):
        for bad in (b"not json", b"[1, 2]"):
            with self.assertRaises(fp.PinStop):
                fp.meta_extract(bad)


class SnapshotAllowlist(unittest.TestCase):
    def test_payload_keeps_only_the_allowlist_and_redacts_sealed_rows(self):
        it = _item("B", state="complete")
        it["payload_full"] = {"catalog_row": {"display_name": "Name", "provenance": {"url": "https://x"}, "target": {"name": "y"}},
                              "catalog_row_sha256": "0" * 64, "objects": {}, "identity_key": it["id"], "folds": [1], "eligibility": {},
                              "receipt": "/p", "identity": [1], "campaign_id": "c", "corpus_hash": "h", "manifest_path": "/m"}
        ep = {"ITEMS": {it["id"]}, "E": {it["dataset"]}}
        reg = {"frozen_item": {}, "siblings": [], "states": {"complete": 1}}
        doc = fp.snapshot_doc(_snap([it]), ep, reg, set())
        self.assertEqual(set(doc["items"][0]["payload"]), {"catalog_row", "catalog_row_sha256", "objects", "identity_key"})
        doc = fp.snapshot_doc(_snap([it]), ep, reg, {it["dataset"]})
        row = doc["items"][0]["payload"]["catalog_row"]
        self.assertTrue(doc["items"][0]["redacted"])
        self.assertEqual(row["display_name"], fp.value_sha256("Name"))
        self.assertEqual(row["provenance"]["url"], fp.value_sha256("https://x"))
        self.assertEqual(row["target"], {"name": "y"})


class LedgerAndGets(unittest.TestCase):
    KEY = "blobs/sha256/" + sha(b"meta")

    def item(self, mode="exact", body=b"meta", key=None, vid="v1"):
        return {"role": "meta_frozen", "mode": mode, "bucket": DS, "key": key or self.KEY, "version_id": vid if mode == "exact" else None,
                "sha256": sha(body), "bytes": len(body)}

    def test_exact_and_content_addressed_gets_are_accounted(self):
        ex, cas = _client(), _client(mode="cas-read")
        other = b"other meta"
        okey = "blobs/sha256/" + sha(other)
        with tempfile.TemporaryDirectory(prefix="p2pin-led-") as d, Stubber(ex) as se, Stubber(cas) as sc:
            se.add_response("get_object", _resp(b"meta", "v1"), {"Bucket": DS, "Key": self.KEY, "VersionId": "v1"})
            sc.add_response("get_object", _resp(other, "v9"), {"Bucket": DS, "Key": okey})
            led = fp.Ledger(Path(d) / "j.jsonl")
            self.assertEqual(fp.s3_get(self.item(), {"exact": ex, "cas": cas}, led), b"meta")
            self.assertEqual(fp.s3_get(self.item("content_addressed", other, okey), {"exact": ex, "cas": cas}, led), other)
            t = led.totals()
            self.assertEqual((t["gets"], t["bytes"], t["ok"], t["failed"]), (2, len(b"meta") + len(other), 2, 0))
            self.assertEqual(t["by_role"]["meta_frozen"]["gets"], 2)
            self.assertEqual([e["version_id"] for e in led.entries], ["v1", "v9"])
            self.assertEqual([e["mode"] for e in led.entries], ["exact", "content_addressed"])
            self.assertEqual([json.loads(x) for x in (Path(d) / "j.jsonl").read_text().splitlines()], led.entries)
            with self.assertRaises(FileExistsError):
                fp.Ledger(Path(d) / "j.jsonl")

    def test_mismatch_classes_stop_and_are_recorded(self):
        ex, cas = _client(), _client(mode="cas-read")
        led = fp.Ledger()
        with Stubber(ex) as se, Stubber(cas) as sc:
            se.add_response("get_object", _resp(b"meta", "WRONG"), {"Bucket": DS, "Key": self.KEY, "VersionId": "v1"})
            se.add_response("get_object", _resp(b"metaX", "v1"), {"Bucket": DS, "Key": self.KEY, "VersionId": "v1"})
            se.add_client_error("get_object", service_error_code="NoSuchVersion", http_status_code=404)
            sc.add_response("get_object", _resp(b"tampered", "v2"), {"Bucket": DS, "Key": self.KEY})
            sc.add_response("get_object", {"Body": StreamingBody(io.BytesIO(b"meta"), 4)}, {"Bucket": DS, "Key": self.KEY})
            cl = {"exact": ex, "cas": cas}
            with self.assertRaisesRegex(fp.PinStop, "version_id mismatch"):
                fp.s3_get(self.item(), cl, led)
            with self.assertRaisesRegex(fp.PinStop, "sha256, bytes mismatch"):
                fp.s3_get(self.item(), cl, led)
            with self.assertRaisesRegex(fp.PinStop, "S3 error NoSuchVersion"):
                fp.s3_get(self.item(), cl, led)
            with self.assertRaisesRegex(fp.PinStop, "content-addressed key"):
                fp.s3_get(self.item("content_addressed"), cl, led)
            with self.assertRaisesRegex(fp.PinStop, "no_version_id"):
                fp.s3_get(self.item("content_addressed"), cl, led)
        self.assertEqual([e["status"] for e in led.entries],
                         ["mismatch:version_id", "mismatch:sha256,bytes", "error:NoSuchVersion:404", "sha_mismatch", "mismatch:no_version_id"])
        self.assertEqual(led.gets, 5)

    def test_transport_error_is_recorded_and_stops(self):
        ex = _client()
        led = fp.Ledger()

        def boom(**_):
            raise botocore.exceptions.ReadTimeoutError(endpoint_url="https://s3.us-east-2.amazonaws.com")
        ex.get_object = boom
        with self.assertRaisesRegex(fp.PinStop, "transport error ReadTimeoutError"):
            fp.s3_get(self.item(), {"exact": ex}, led)
        self.assertEqual([(e["status"], e["bytes"]) for e in led.entries], [("error:ReadTimeoutError", None)])

    def test_guard_refusal_is_recorded_and_stops(self):
        led = fp.Ledger()
        with self.assertRaisesRegex(fp.PinStop, "guard refused"):
            fp.s3_get(self.item(vid="null"), {"exact": _client()}, led)
        with self.assertRaisesRegex(fp.PinStop, "guard refused"):                       # a lane client cannot do CAS reads
            fp.s3_get(self.item("content_addressed"), {"cas": _client()}, led)
        self.assertEqual([e["status"] for e in led.entries], ["refused", "refused"])

    def test_budget_reserve_refuses_before_the_request(self):
        led = fp.Ledger(max_gets=1, max_bytes=10)
        ex = _client()
        with Stubber(ex) as se:
            se.add_response("get_object", _resp(b"meta", "v1"), {"Bucket": DS, "Key": self.KEY, "VersionId": "v1"})
            fp.s3_get(self.item(), {"exact": ex}, led)
            with self.assertRaisesRegex(fp.PinStop, "GET 2 would exceed 1"):
                fp.s3_get(self.item(), {"exact": ex}, led)
        led = fp.Ledger(max_gets=5, max_bytes=3)
        with self.assertRaisesRegex(fp.PinStop, "would exceed 3 B"):
            fp.s3_get(self.item(), {"exact": _client()}, led)
        self.assertEqual(led.gets, 0)


class CpuGuardRules(unittest.TestCase):
    def guard(self, docs, **kw):
        d = tempfile.TemporaryDirectory(prefix="p2pin-cpu-")
        self.addCleanup(d.cleanup)
        p = Path(d.name) / "cpu-admission.json"
        state = {"t": 1000.0, "i": 0, "sleeps": 0}

        def clock():
            return state["t"]

        def sleep(s):
            state["t"] += s
            state["sleeps"] += 1
            state["i"] = min(state["i"] + 1, len(docs) - 1)
            write()

        def write():
            doc = docs[state["i"]]
            if doc is None:
                if p.exists():
                    p.unlink()
            else:
                p.write_text(json.dumps(doc(state["t"]) if callable(doc) else doc))
        write()
        return fp.CpuGuard(p, clock=clock, sleep=sleep, poll_s=10, **kw), state

    @staticmethod
    def ok(cpu=10.0, admit=True, age=1.0):
        return lambda t: {"admit": admit, "sampled_unix_seconds": t - age, "nodes": {"xbabe2": {"cpu_rolling_percent": cpu}}}

    def test_admitted_passes_without_sleeping(self):
        g, st = self.guard([self.ok()])
        g.wait("s")
        self.assertEqual((st["sleeps"], g.summary()["pauses"]), (0, 0))

    def test_pauses_on_stale_denied_busy_or_missing_then_resumes(self):
        g, st = self.guard([self.ok(age=60), self.ok(admit=False), self.ok(cpu=75.0), None, self.ok()])
        g.wait("s3")
        self.assertEqual(st["sleeps"], 4)
        s = g.summary()
        self.assertEqual((s["pauses"], s["paused_s"]), (1, 40.0))
        self.assertIn("old", s["pause_log"][0]["reason"])

    def test_stops_after_max_wait(self):
        g, _ = self.guard([self.ok(cpu=99.0)], max_wait_s=30)
        with self.assertRaisesRegex(fp.PinStop, "CPU guard"):
            g.wait("s3")

    def test_rate_limited_unless_forced(self):
        g, st = self.guard([self.ok()])
        g.wait("a")
        g.wait("b")
        self.assertEqual(g.checks, 1)
        g.wait("c", force=True)
        self.assertEqual(g.checks, 2)


class Serialization(unittest.TestCase):
    def test_jsonable_is_deterministic(self):
        a = fp.jsonable({"s": {"b", "a"}, "t": (1, 2), "b": b"x\0", "r": re.compile("q+"), "f": frozenset({3, 1})})
        self.assertEqual(a, {"s": ["a", "b"], "t": [1, 2], "b": "x\0", "r": "q+", "f": [1, 3]})
        self.assertEqual(common.canonical(a), common.canonical(fp.jsonable({"f": frozenset({1, 3}), "r": re.compile("q+"), "b": b"x\0",
                                                                            "t": (1, 2), "s": {"a", "b"}})))
        for bad in ({"x": float("nan")}, {1: "a"}, {"f": print}, {"b": b"\xff"}):
            with self.assertRaises(fp.PinStop):
                fp.jsonable(bad)

    def test_intake_canonical_is_ascii_escaped(self):
        row = {"b": "ü", "a": 1}
        self.assertEqual(fp.intake_canonical(row), b'{"a":1,"b":"\\u00fc"}')
        self.assertEqual(fp.identity_key("a", "b", "c"), sha(b'["a","b","c"]'))

    def test_rdatasets_keys_both_sides(self):
        prov = {"platform": " Rdatasets ", "dataset_id": "Pkg/Item", "url": "https://vincentarelbundock.github.io/Rdatasets/csv/Pkg/Item.csv"}
        self.assertEqual(fp.rdatasets_keys(prov), {"pkg/item"})
        lb = {"source_key": "rdatasets:Pkg/Item2", "evidence": [{"source_url": "https://vincentarelbundock.github.io/Rdatasets/doc/P/N.html"}]}
        self.assertEqual(fp.lineage_b_rdatasets_keys(lb), {"pkg/item2", "p/n"})
        self.assertEqual(fp.rdatasets_keys({"platform": None, "url": None}), set())


class ImportBan(unittest.TestCase):
    def test_no_forbidden_imports(self):
        tree = ast.parse(Path(fp.__file__).read_text())
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                mods.add((node.module or "").split(".")[0])
                mods |= {a.name for a in node.names}
        self.assertFalse(mods & set(fp.FORBIDDEN_IMPORTS), mods & set(fp.FORBIDDEN_IMPORTS))
        self.assertNotIn("__import__", Path(fp.__file__).read_text())

    def test_default_config_never_takes_inputs_from_argv(self):
        cfg = fp.default_config()
        self.assertEqual(cfg.fixed["epoch"].sha256, fp.EPOCH_SHA256)
        self.assertEqual(len(cfg.seed_view_files), 10)
        self.assertTrue(str(cfg.fixed["tabpfn35_unseen_audit"].path).endswith("s3-audit-final.json"))
        for name in fp.LINEAGE_B_VERSIONS:
            self.assertTrue(Path(cfg.fixed[name].path).name.startswith(cfg.fixed[name].sha256))
        src = Path(fp.__file__).read_text()
        self.assertNotIn("--tabpfn", src)
        self.assertNotIn("--config", src)


if __name__ == "__main__":
    unittest.main()
