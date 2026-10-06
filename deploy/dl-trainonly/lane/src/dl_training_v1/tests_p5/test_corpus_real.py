"""Offline proof of the P5 build on two real epoch-0 contexts (spec tests_p5/test_corpus_real.py).

It reads only the pinned local sources (native, outcome and transport prefixes, history-ref.json or the live incoming
cursor entry, seed-pool.jsonl, bootstrap-results.jsonl, prepared.json, the exposed-side qualification files and the
sealed inventory projection) of:
- the first included frozen context of epoch-000.json (6ebcb49c..., fold 0), as train_fit;
- the first included incoming context in sha order, as dev.
The JTT1 inputs are synthetic, shaped by each prepared.json's train_rows and encoded_features, in a temp cache, and
the epoch is re-pinned to them in a temp copy: no S3 client is ever created (s3io.make_client is patched to fail) and
nothing is skipped. Every write goes under TMPDIR. Run only inside the lane slice:
  python -m unittest tests_p5.test_corpus_real -v
"""
from __future__ import annotations

import gzip
import hashlib
import json
import struct
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pyarrow.parquet as pq

from dltrain import cpuguard
from dltrain import corpus_build as B
from dltrain import corpus_schema as S
from dltrain import dsl_tokenizer as T
from dltrain import programs as P
from tests_p5 import fixtures_p5 as fx

FROZEN = "6ebcb49c66f39192af38ffa7393829365b8d7c5e62fd12fd431476dc37785299"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ok_guard() -> cpuguard.Guard:
    return cpuguard.Guard(checker=lambda: (True, [], {}), poll_s=0.01)


def synthetic_jtt1(n_train: int, p: int, n_test: int, seed: int) -> bytes:
    """A JTT1 input with the real context's shape: features ~ N(0, 1), a smooth non-affine target plus noise, so no
    column is an exact or affine copy of y (the train-only leak audit passes)."""
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    X = rng.standard_normal((n, p))
    y = np.sin(X[:, 0]) + 0.3 * X[:, min(1, p - 1)] ** 2 + 0.1 * rng.standard_normal(n)
    mat = np.concatenate([X, y[:, None]], axis=1).astype("<f8")
    return b"JTT1" + struct.pack("<IIIB", n_train, n_test, p, 2) + bytes(i % 2 for i in range(n_train)) + mat.tobytes()


class Real(unittest.TestCase):

    def setUp(self):
        self._tmp = fx.tmpdir()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        for patcher in (mock.patch.object(cpuguard, "require_slice_limits", return_value={}),
                        mock.patch("dltrain.s3io.make_client", side_effect=AssertionError("no S3 in an offline test"))):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_two_real_contexts_offline(self):
        raw = B.DEFAULT_EPOCH.read_bytes()
        self.assertEqual(sha(raw), B.DEFAULT_EPOCH_SHA256)
        doc = json.loads(raw)
        included = [e for e in doc["contexts"] if e.get("status") == "included"]
        frozen = next(e for e in included if e["cohort"] == "frozen")
        self.assertEqual((frozen["context_sha256"], frozen["fold"]), (FROZEN, 0))
        incoming = min((e for e in included if e["cohort"] == "incoming"), key=lambda e: e["context_sha256"])
        # synthetic JTT1 inputs shaped by prepared.json (sha-verified), in a temp cache; the epoch copy is re-pinned
        cache = self.tmp / "jtt1"
        cache.mkdir()
        entries = []
        for e in (frozen, incoming):
            cid = e["context_sha256"]
            prep = S.load_prepared(e["prepared_ref"]["path"], e["prepared_ref"], cid)
            body = synthetic_jtt1(prep["train_rows"], prep["encoded_features"], 11, int(cid[:8], 16))
            gz = gzip.compress(body, compresslevel=3, mtime=0)
            ref = {"bucket": "synthetic", "bytes": len(gz), "compression": "gzip-3-mtime0",
                   "key": f"synthetic/{sha(body)}.jtt1.gz", "raw_bytes": len(body), "raw_sha256": sha(body),
                   "sha256": sha(gz), "version_id": "synthetic"}
            (cache / f"{ref['raw_sha256']}.jtt1.gz").write_bytes(gz)
            entries.append({**e, "input_ref": ref})
        epoch = self.tmp / "epoch.json"
        epoch.write_text(json.dumps({**{k: v for k, v in doc.items() if k != "contexts"}, "contexts": entries}))
        pmap = self.tmp / "partition-map.json"
        pmap.write_text(json.dumps({frozen["context_sha256"]: {"family_id": "p5-real:" + frozen["dataset_hash"],
                                                               "split_role": "train_fit"},
                                    incoming["context_sha256"]: {"family_id": "p5-real:" + incoming["dataset_hash"],
                                                                 "split_role": "dev"}}))
        pool = B.DEFAULT_SEED_POOL.read_bytes()
        self.assertEqual(sha(pool), B.DEFAULT_SEED_POOL_SHA256)
        seeds = self.tmp / "seeds.json"
        seeds.write_text(json.dumps(sorted({json.loads(x)["program"]["program_sha256"] for x in pool.splitlines()
                                            if x.strip()})))
        # P4a and a sample-only tokenizer over these two contexts (train_fit programs only)
        dict_dir, tok_dir = self.tmp / "program-dict", self.tmp / "tokenizer"
        P.build(epoch, pmap, seeds, dict_dir)
        T.build_artifacts(dict_dir, epoch, tok_dir, automaton=False)
        out = self.tmp / "corpus"
        res = B.run(epoch=epoch, epoch_sha256=sha(epoch.read_bytes()), partition_map=pmap, seeds=seeds,
                    tokenizer=tok_dir, out=out, jtt1_cache=cache, budget_bytes=0, offline=True,
                    program_dict=dict_dir, inventory=doc["inventory"]["path"], guard=ok_guard())
        self.assertEqual((res["status"], res["exit"]), ("COMPLETE", B.EXIT_OK), res)
        rep = B.verify(out, schema_allowlist=True, counts=True, rehash_sources=True, guard=ok_guard())
        self.assertEqual(rep["verdict"], "PASS", rep["failures"])
        self.assertGreater(rep["checked"]["sources_rehashed"], 6)
        cur = json.loads((out / "cursor.json").read_bytes())
        recs = {r["context_sha256"]: r for p in ("train_fit", "dev") for r in B.read_ctx_records(out, cur, p)}
        for e in (frozen, incoming):
            r = recs[e["context_sha256"]]
            c = r["counts"]
            with self.subTest(cohort=e["cohort"]):
                self.assertIsNone(r["excluded"])
                self.assertFalse(r["leak"]["target_leak"])
                self.assertEqual(c["native_completed"], S.EXPECT_COMPLETED_PER_FOLD)       # R2: 1280 per fold
                self.assertEqual(r["records"] + r["refused"], r["summary"]["unique_programs"])   # R5/R6
                self.assertEqual(r["checks"], {f"R{i}": True for i in range(1, 11)})       # incl. native/outcome cv
                self.assertEqual((r["dict_crosscheck"]["source"], r["dict_crosscheck"]["equal"]), ("program_dict", True))
                self.assertLessEqual(set(c["by_backend"]), set(S.BACKENDS))
                self.assertEqual(sum(c["by_outcome"].values()), r["records"])
                self.assertEqual(r["prepared"], {k: True for k in ("train_rows", "encoded_features", "inner_folds",
                                                                  "candidate_timeout_ms", "evaluator_sha256",
                                                                  "inner_cv_schema", "context_sha256", "dataset_hash",
                                                                  "outer_fold")})
                self.assertIn(r["label"]["status"], S.LABEL_STATUSES)
        self.assertEqual(recs[frozen["context_sha256"]]["label"]["status"], "OK")
        f = recs[frozen["context_sha256"]]["counts"]
        self.assertGreater(f["detailed_completed"], 0)
        self.assertGreater(f["detailless_completed"], 0)
        self.assertGreater(f["by_backend"].get("aws_lambda", 0), 0)
        self.assertEqual(recs[frozen["context_sha256"]]["refused"], 0)                     # train_fit never refuses
        # the training schema has no exposed name, and the sealed exposed table is non-empty
        self.assertFalse([n for n in S.TRAIN_RECORD_SCHEMA.names if S.name_is_exposed(n)])
        files = B.open_exposed(out, "train_fit", role="exposed_report")
        self.assertTrue(files)
        self.assertEqual(sum(pq.ParquetFile(x).metadata.num_rows for x in files),
                         recs[frozen["context_sha256"]]["summary"]["unique_programs"])
        rows = B.load_split(out, "train_fit")[1]
        self.assertEqual(len(rows), recs[frozen["context_sha256"]]["records"])
        self.assertTrue(all(list(r) == S.TRAIN_RECORD_SCHEMA.names for r in rows))


if __name__ == "__main__":
    unittest.main()
