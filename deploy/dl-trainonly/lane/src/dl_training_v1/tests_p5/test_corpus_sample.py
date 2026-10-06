"""Offline proofs for the bounded real-sample driver (spec tests_p5/test_corpus_sample.py; measurement plan S0-S2, S5).

The selection and the fake partition map are checked on the real epoch-000.json (read-only, pinned by sha256) and
on synthetic epochs; the 64-attempt GET guard runs the fetch phase of corpus_build.run against a fake read client
that spends up to three HTTP attempts per call (the s3io read client's retry ceiling). Nothing touches S3, and every
write goes under TMPDIR. Run only inside the lane slice: python -m unittest tests_p5.test_corpus_sample -v
"""
from __future__ import annotations

import gzip
import hashlib
import inspect
import json
import os
import random
import shutil
import struct
import unittest
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from dltrain import cpuguard
from dltrain import corpus_build as B
from dltrain import corpus_schema as S
from dltrain import programs as P
from tests_p5 import fixtures_p5 as fx

SEED = 20261005
ROLE_DOMAIN = b"jope.dl-p5-sample-role/v1\0"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ok_guard() -> cpuguard.Guard:
    return cpuguard.Guard(checker=lambda: (True, [], {}), poll_s=0.01)


class FakeEvents:
    def __init__(self):
        self.handlers = []

    def register(self, name, handler, unique_id=None):
        self.handlers.append((name, handler))


class RetryingClient:
    """A read-client stand-in whose every GET spends `attempts` HTTP attempts (before-send fires once per attempt)."""

    def __init__(self, attempts: int):
        self.attempts = attempts
        self.meta = SimpleNamespace(events=FakeEvents())


def tiny_jtt1(k: int) -> bytes:
    """A distinct, valid JTT1 input: n_train=3, n_test=1, p=1."""
    vals = [float(k), 1.0, float(k) + 0.5, 2.0, float(k) + 1.5, 4.0, 9.0, 9.0]
    return b"JTT1" + struct.pack("<IIIB", 3, 1, 1, 2) + bytes([0, 1, 0]) + struct.pack("<8d", *vals)


class Sample(unittest.TestCase):

    def setUp(self):
        self._tmp = fx.tmpdir()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_selection_and_fake_map(self):
        # ---- S1 on the real epoch: deterministic, 4 per (cohort, fold) = 40, re-derived independently
        raw = B.DEFAULT_EPOCH.read_bytes()
        self.assertEqual(sha(raw), B.DEFAULT_EPOCH_SHA256)
        doc = json.loads(raw)
        picks = B.select_sample(doc, per_cell=4, seed=SEED)
        self.assertEqual(picks, B.select_sample(B.DEFAULT_EPOCH, 4, SEED))
        self.assertEqual(len(picks), 40)
        self.assertEqual(len({e["context_sha256"] for e in picks}), 40)
        self.assertEqual(Counter((e["cohort"], e["fold"]) for e in picks),
                         {(c, f): 4 for c in P.COHORTS for f in range(5)})
        self.assertTrue(all(e["status"] == "included" for e in picks))
        cells = defaultdict(list)
        for e in doc["contexts"]:
            if e.get("status") == "included":
                cells[(e["cohort"], e["fold"])].append(e)
        self.assertEqual({k: len(v) for k, v in cells.items() if k[0] == "frozen"},
                         {("frozen", 0): 570, ("frozen", 1): 570, ("frozen", 2): 570, ("frozen", 3): 568,
                          ("frozen", 4): 556})
        rng, want = random.Random(SEED), []
        for cohort in ("frozen", "incoming"):
            for fold in range(5):
                want += rng.sample(sorted(cells[(cohort, fold)], key=lambda e: e["context_sha256"]), 4)
        self.assertEqual([e["context_sha256"] for e in picks], [e["context_sha256"] for e in want])
        self.assertNotEqual([e["context_sha256"] for e in B.select_sample(doc, 4, SEED + 1)],
                            [e["context_sha256"] for e in picks])
        # a cell with fewer than per_cell contexts is a STOP
        thin = {"contexts": [e for e in doc["contexts"]
                             if not (e.get("cohort") == "incoming" and e.get("fold") == 4)]
                + [e for e in doc["contexts"] if e.get("cohort") == "incoming" and e.get("fold") == 4
                   and e.get("status") == "included"][:3]}
        with self.assertRaises(S.CorpusStop) as cm:
            B.select_sample(thin, 4, SEED)
        self.assertEqual(cm.exception.kind, "sample")
        # ---- S2 fake map on the real selection: one family per dataset, roles per dataset, quotas, diag rule
        pmap = B.sample_partition_map(picks)
        self.assertEqual(set(pmap), {e["context_sha256"] for e in picks})
        role_of_ds: dict[str, set] = defaultdict(set)
        for e in picks:
            m = pmap[e["context_sha256"]]
            self.assertEqual(m["family_id"], "p5-sample-fam:" + e["dataset_hash"])
            role_of_ds[e["dataset_hash"]].add("train" if m["split_role"] in ("train_fit", "diag") else m["split_role"])
            if m["split_role"] == "diag":
                self.assertEqual(e["fold"], 3)                       # never frozen fold 0
            if m["split_role"] == "train_fit":
                self.assertNotEqual(e["fold"], 3)
        self.assertTrue(all(len(v) == 1 for v in role_of_ds.values()))
        # independent re-derivation: per cohort, new datasets in sha256(domain + dataset) order fill sealed 1,
        # heldout 3, dev 3, then train; a dataset already assigned in the other cohort keeps its role
        want_role: dict[str, str] = {}
        for cohort in P.COHORTS:
            ds = sorted({e["dataset_hash"] for e in picks if e["cohort"] == cohort},
                        key=lambda d: (sha(ROLE_DOMAIN + d.encode()), d))
            filled = Counter()
            for d in ds:
                if d in want_role:
                    continue
                role = next((r for r, q in (("sealed", 1), ("heldout", 3), ("dev", 3)) if filled[r] < q), "train")
                filled[role] += 1
                want_role[d] = role
            with self.subTest(cohort=cohort):
                self.assertEqual((filled["sealed"], filled["heldout"], filled["dev"]), (1, 3, 3))
        self.assertEqual({d: next(iter(v)) for d, v in role_of_ds.items()}, want_role)
        roles = Counter(m["split_role"] for m in pmap.values())
        self.assertEqual(set(roles), {"train_fit", "dev", "diag", "heldout", "sealed"})
        self.assertEqual(sum(1 for m in pmap.values() if m["split_role"] in P.PARTITIONS), 40 - roles["sealed"])
        # the fake map is a valid P4a map and passes P5's training-closure checks
        mp = self.tmp / "map.json"
        mp.write_text(json.dumps(pmap, sort_keys=True))
        sample_epoch = self.tmp / "sample-epoch.json"
        sample_epoch.write_text(json.dumps({**{k: v for k, v in doc.items() if k != "contexts"}, "contexts": picks}))
        plan = P._plan(sample_epoch, mp)
        self.assertEqual(len(plan.contexts), 40 - roles["sealed"])
        members = {p: [c for c in plan.contexts if c.split_role == p] for p in P.PARTITIONS}
        B.closure_checks({e["context_sha256"]: e for e in picks}, B._load_map(mp), members)
        # ---- S5 guard: a call is admitted only while calls + 1 <= 64 and http_attempts + 3 <= 64
        self.assertEqual(inspect.signature(B.sample).parameters["max_gets"].default, 64)
        n = 24
        entries, store = [], {}
        for k in range(n):
            body_raw = tiny_jtt1(k)
            gz = gzip.compress(body_raw, compresslevel=3, mtime=0)
            ref = {"bucket": "synthetic", "key": f"synthetic/{sha(body_raw)}.jtt1.gz", "version_id": f"v{k}",
                   "sha256": sha(gz), "bytes": len(gz), "raw_sha256": sha(body_raw), "raw_bytes": len(body_raw)}
            store[ref["key"]] = gz
            cid = sha(f"guard-ctx:{k}".encode())
            entries.append({"cohort": "frozen", "status": "included", "context_sha256": cid, "fold": k % 5,
                            "view_id": f"ds{k:04d}__r0__f{k % 5}", "dataset_hash": f"ds{k:04d}",
                            "native_source": {"path": f"/nonexistent/ds{k:04d}__r0__f{k % 5}/evaluations-{cid[:16]}.jsonl",
                                              "bytes": 1, "sha256": "0" * 64},
                            "history_ref": {"sha256": "1" * 64}, "input_ref": ref})
        epoch = self.tmp / "guard-epoch.json"
        epoch.write_text(json.dumps({"experiment_id": "p5-guard", "epoch": 0, "contexts": entries}))
        gmap = self.tmp / "guard-map.json"
        gmap.write_text(json.dumps({e["context_sha256"]: {"family_id": "fam-" + e["dataset_hash"], "split_role": "train_fit"}
                                    for e in entries}))
        tok = self.tmp / "tok"
        fx.write_tokenizer(tok, ["(pipe X (ridge_gcv))"])
        def fetch(out: str, attempts: int) -> tuple[dict, dict, Counter]:
            calls = Counter()

            def fake_get(client, key, version_id, bucket=None):
                calls[key] += 1
                for _ in range(client.attempts):
                    for name, handler in client.meta.events.handlers:
                        if name == "before-send.s3.GetObject":
                            handler()
                return store[key], {"VersionId": version_id}

            with mock.patch.object(cpuguard, "require_slice_limits", return_value={}), \
                    mock.patch("dltrain.s3io.get_exact", fake_get):
                res = B.run(epoch=epoch, partition_map=gmap, seeds=[], tokenizer=tok, out=self.tmp / out,
                            jtt1_cache=self.tmp / f"cache-{out}", budget_bytes=10**9, max_gets=64, offline=False,
                            phase="fetch", guard=ok_guard(), client_factory=lambda: RetryingClient(attempts))
            return res, json.loads((self.tmp / out / "cursor.json").read_bytes())["ledger"], calls

        res, led, calls = fetch("worst", 3)
        self.assertEqual((res["status"], res["exit"]), ("PAUSED_BUDGET", B.EXIT_PAUSED), res)
        self.assertEqual((led["calls"], led["http_attempts"]), (21, 63))     # a 22nd call could reach 66 > 64
        self.assertLessEqual(led["http_attempts"], 64)
        self.assertEqual((len(calls), set(calls.values())), (21, {1}))
        self.assertEqual(len(list((self.tmp / "cache-worst").iterdir())), 21)
        res, led, calls = fetch("worst", 3)                                  # the ledger is cumulative across resumes
        self.assertEqual(res["status"], "PAUSED_BUDGET")
        self.assertEqual((led["calls"], led["http_attempts"], len(calls)), (21, 63, 0))
        res, led, calls = fetch("single", 1)
        self.assertEqual(res["status"], "FETCHED", res)
        self.assertEqual((led["calls"], led["http_attempts"]), (n, n))
        self.assertEqual((len(calls), set(calls.values())), (n, {1}))
        # ---- S0: sample() refuses outputs outside RUN/tmp/p5-sample-<label> and sets tempfile.tempdir = OUT/tmp
        for bad in ("../x", "a/b", "", ".", "..", "x y"):
            with self.subTest(label=bad), self.assertRaises(S.CorpusStop):
                B.sample_out(bad)
        with mock.patch.object(cpuguard, "require_slice_limits", return_value={}):
            for bad_out in (self.tmp / "p5-sample-ok", B.RUN / "p5-sample-ok", B.SAMPLE_ROOT / "p5-sample-ok" / "nested"):
                with self.subTest(out=str(bad_out)), self.assertRaises(S.CorpusStop) as cm:
                    B.sample(label="ok", out=bad_out, jtt1_cache=self.tmp / "cache", guard=ok_guard())
                self.assertEqual(cm.exception.kind, "gate")
                self.assertFalse(bad_out.exists())
        label = "unittest-sample-" + sha(str(self.tmp).encode())[:12]
        out = B.sample_out(label)
        self.assertEqual(out, Path(os.path.realpath(B.SAMPLE_ROOT / f"p5-sample-{label}")))
        self.addCleanup(shutil.rmtree, out, True)
        self.addCleanup(setattr, B.tempfile, "tempdir", B.tempfile.tempdir)
        with mock.patch.object(cpuguard, "require_slice_limits", return_value={}), self.assertRaises(S.CorpusStop) as cm:
            B.sample(label=label, jtt1_cache=self.tmp / "cache", epoch=B.DEFAULT_EPOCH, epoch_sha256="0" * 64,
                     guard=ok_guard())
        self.assertEqual(cm.exception.kind, "input")
        self.assertEqual(B.tempfile.tempdir, str(out / "tmp"))
        self.assertEqual(sorted(x.name for x in out.iterdir()), ["tmp"])     # nothing written before the epoch pin
        with mock.patch.object(cpuguard, "require_slice_limits", return_value={}), self.assertRaises(S.CorpusStop) as cm:
            (out / "tmp" / "leftover").write_bytes(b"x")
            B.sample(label=label, jtt1_cache=self.tmp / "cache", guard=ok_guard())     # non-empty OUT needs --resume
        self.assertEqual(cm.exception.kind, "gate")


if __name__ == "__main__":
    unittest.main()
