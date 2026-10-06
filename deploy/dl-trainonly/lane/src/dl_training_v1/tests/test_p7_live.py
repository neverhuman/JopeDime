"""P7 test 8 (live writer: bridge replay, trainer-mode documents, dlMetrics) plus the live, trainer and evaluator unit
checks (PLAN §5 P7 test 8, §6.7, §6.6, §6.8, §5 P10).

Every file these tests write lives in a ``tempfile.TemporaryDirectory`` under ``$TMPDIR`` and is removed in tearDown.
``claude/live.json`` and ``claude/live.schema.json`` are only read; Codex's consumer files are only read and run as a
subprocess (python3 for the bridge program, node for the dlMetrics vm slices), never imported. Nothing is skipped: a
missing node binary fails the dlMetrics check as NOT_EXECUTED.
"""
from __future__ import annotations

import ast
import copy
import dataclasses
import hashlib
import json
import math
import os
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from collections import defaultdict
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from dltrain import ckpt, common, evaluate, live, model, objective, sampler, train
from tests import fixtures_p7

TMP_ROOT = Path(os.environ["TMPDIR"])
CLAUDE_LIVE = live.CLAUDE_DIR / "live.json"
UNIFIED_OBSERVER = live.CONSUMER_DIR / "unified_observer.py"
# unified_observer.py pin as recorded in claude/live.schema.json and the audit document's contract.consumer_pins
UNIFIED_OBSERVER_SHA256 = "4eef0f26ec0f57b05f574ed600d6011d6fc02cd5ffbb4f0cdc05d6fc29d90e58"
OBSERVATION_SCHEMA = "jope.dl-training-observation/v1"
BRIDGE_TIMEOUT_S = 13
BRIDGE_ATTEMPTS = 3
BRIDGE_RETRYABLE = ("audit review/progress pins differ", "audit artifact changed while reading")
MAX_OBSERVER_STDOUT = 10 * 1024 ** 2
STALE_RECEIPT_US = 46_000_000


def _sha(data: bytes | str) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


# A synthetic s3ref shape for admitted-document tests; it names no real grant and no real object.
FAKE_ADMISSION_REF = {"bucket": "fixture-bucket", "key": "fixture/admission.json", "version_id": "fixture-v1",
                      "sha256": _sha("fixture admission receipt"), "bytes": 25}


def setUpModule():
    model.configure_determinism(4)
    live.assert_outside_claude(TMP_ROOT, "TMPDIR")


def _refuse_constant(name):
    raise ValueError(f"non-finite JSON constant {name}")


def _make_plan(**overrides) -> dict:
    kw = fixtures_p7.plan_kwargs()
    kw.update(overrides)
    return train.make_run_plan(**kw)


def _events(work: Path) -> list[dict]:
    return [json.loads(line) for line in (work / "events.jsonl").read_bytes().splitlines()]


def _rng_snapshot():
    return torch.get_rng_state().clone(), random.getstate(), np.random.get_state()


def _assert_rng_equal(case: unittest.TestCase, a, b) -> None:
    case.assertTrue(torch.equal(a[0], b[0]), "torch RNG state changed")
    case.assertEqual(a[1], b[1], "python RNG state changed")
    case.assertEqual(a[2][0], b[2][0])
    case.assertTrue(np.array_equal(a[2][1], b[2][1]), "numpy RNG state changed")
    case.assertEqual(tuple(a[2][2:]), tuple(b[2][2:]))


def _write_guards():
    """Patch the writers live.py uses so a broken refusal can never write under claude/."""
    boom = AssertionError("a write was attempted after a claude/ path should have been refused")
    return (mock.patch.object(common, "atomic_write", side_effect=boom),
            mock.patch.object(common, "immutable_write", side_effect=boom))


class _TmpCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(dir=str(TMP_ROOT), prefix="p7-live-")
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    # ---------------------------------------------------------------- bridge replay (observe() checks)
    def run_bridge(self, source_path: Path):
        """Run the pinned bridge program with python3 via stdin, retrying only on the two audit-race messages."""
        program, pins = live.bridge_program(source_path)
        python = shutil.which("python3") or sys.executable
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        for attempt in range(1, BRIDGE_ATTEMPTS + 1):
            proc = subprocess.run([python, "-"], input=program, text=True, capture_output=True,
                                  timeout=BRIDGE_TIMEOUT_S, cwd=str(self.tmp), env=env)
            if proc.returncode != 0 and attempt < BRIDGE_ATTEMPTS and any(m in proc.stderr for m in BRIDGE_RETRYABLE):
                continue
            return proc, program, pins

    def check_observation(self, proc, source_path: Path, raw: bytes) -> dict:
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertLessEqual(len(proc.stdout), MAX_OBSERVER_STDOUT)
        data = json.loads(proc.stdout, parse_constant=_refuse_constant)
        self.assertEqual(data["schema"], OBSERVATION_SCHEMA)
        self.assertEqual(data["source_pin"], {"path": str(source_path), "sha256": _sha(raw), "bytes": len(raw)})
        self.assertEqual(data["source"], json.loads(raw))
        self.assertEqual((data["remote_controls"], data["aws_lambda_calls"]), (0, 0))
        self.assertIsInstance(data["review_artifacts"], list)
        for ref in data["review_artifacts"]:                     # as observe(): text bytes must match sha/bytes
            body = ref["text"].encode()
            self.assertEqual(len(body), ref["bytes"], ref["name"])
            self.assertEqual(_sha(body), ref["sha256"], ref["name"])
        return data                                               # process_alive_exact is never asserted


class P7Test8aBridgeReplay(_TmpCase):
    def test_audit_document_replays_through_the_pinned_bridge(self):
        raw = CLAUDE_LIVE.read_bytes()                            # read-only; the replay reads a TMPDIR copy
        fixture = self.tmp / "audit-live.json"
        common.atomic_write(fixture, raw)
        proc, program, pins = self.run_bridge(fixture)
        self.assertEqual(pins["dl_bridge.py"], live.CONSUMER_PINS["dl_bridge.py"])
        self.assertEqual(pins["dl_feed.py"], live.CONSUMER_PINS["dl_feed.py"])
        self.assertEqual(pins["program_sha256"], live.BRIDGE_PROGRAM_SHA256)
        self.assertEqual(pins["substituted_program_sha256"], _sha(program))
        self.assertNotIn(live.SOURCE_LINE, program)
        self.assertEqual(program.count(f"p=pathlib.Path({str(fixture)!r})"), 1)
        data = self.check_observation(proc, fixture, raw)
        self.assertEqual(data["source"]["schema"], live.LIVE_SCHEMA_ID)

    def test_bridge_program_refuses_claude_paths_relative_paths_and_pin_drift(self):
        for bad in (CLAUDE_LIVE, live.CLAUDE_DIR / "x" / ".." / "live.json"):
            with self.assertRaises(live.LiveRefused):
                live.bridge_program(bad)
        link = self.tmp / "audit-link"
        link.symlink_to(live.CLAUDE_DIR, target_is_directory=True)
        try:
            with self.assertRaises(live.LiveRefused):
                live.bridge_program(link / "live.json")
        finally:
            link.unlink()                                   # removes the link itself, never its target
        with self.assertRaises(live.LiveRefused):
            live.bridge_program(Path("relative/live.json"))
        for name in ("dl_bridge.py", "dl_feed.py"):
            with self.subTest(pin=name), mock.patch.dict(live.CONSUMER_PINS, {name: "0" * 64}):
                with self.assertRaisesRegex(live.LiveRefused, "re-pin required"):
                    live.bridge_program(self.tmp / "live.json")
        with mock.patch.object(live, "BRIDGE_PROGRAM_SHA256", "0" * 64):
            with self.assertRaisesRegex(live.LiveRefused, "re-pin required"):
                live.bridge_program(self.tmp / "live.json")


class P7Test8bTrainerDocuments(_TmpCase):
    def test_trainer_mode_documents_validate_and_replay(self):
        audit_copy = self.tmp / "audit-copy.json"
        common.atomic_write(audit_copy, CLAUDE_LIVE.read_bytes())
        audit = json.loads(audit_copy.read_bytes())
        base = live.trainer_base_from_audit(audit)
        schema, _ = live.load_schema(live.LIVE_SCHEMA_PATH)
        out = self.tmp / "run" / "live.json"
        update_dir = self.tmp / "updates"
        writer = live.LiveWriter(out, schema=schema, base=base, interval_s=0.05)
        plan = _make_plan(max_updates=4, train=fixtures_p7.train_config(receipt_interval_s=0.0))
        result = train.run(plan, fixtures_p7.train_data(), self.tmp / "work", live_writer=writer, update_dir=update_dir)
        self.assertEqual((result.status, result.reason, result.step), ("COMPLETED", None, 4))
        self.assertTrue(result.optimizer_parameter_change_verified)
        self.assertIsNone(writer.error)
        self.assertGreaterEqual(writer.writes, 2)
        self.assertFalse(writer.running)

        raw = out.read_bytes()
        self.assertLessEqual(len(raw), live.MAX_DOC_BYTES)
        doc = json.loads(raw, parse_constant=_refuse_constant)
        self.assertEqual(raw, json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                         allow_nan=False).encode())
        self.assertEqual(live.validate(doc, schema), [])
        self.assertEqual(doc["source_kind"], "trainer_telemetry")
        self.assertEqual((doc["contract"]["producer_mode"], doc["contract"]["trainer_phase"]), ("TRAINER", "completed"))
        self.assertEqual(doc["contract"]["consumer_pins"], live.CONSUMER_PINS)
        self.assertEqual((doc["status"], doc["training_admitted"]), ("AUDITING_PLAN_UNADMITTED", False))
        self.assertTrue(doc["optimizer_parameter_change_verified"])
        proof = doc["training_proof"]
        self.assertEqual(proof["initial_parameter_sha256"], result.initial_parameter_sha256)
        self.assertEqual(proof["current_parameter_sha256"], result.parameter_sha256)
        self.assertNotEqual(proof["initial_parameter_sha256"], proof["current_parameter_sha256"])
        self.assertEqual(proof["optimizer_updates_total"], 4)
        self.assertEqual(doc["process_identity"], audit["process_identity"])
        native = doc["native_process_identity"]
        self.assertEqual(native, live.native_process_identity())
        self.assertNotEqual(native["pid"], audit["process_identity"]["pid"])
        self.assertEqual(doc["not_applicable"], [])
        self.assertTrue(all(doc[e["field"]] is None for e in doc["unknown"] if e["field"] in doc))

        receipt, ref = doc["optimizer_update_receipt"], doc["optimizer_update_ref"]
        self.assertEqual((receipt["optimizer_step"], doc["optimizer_step"]), (4, 4))
        self.assertEqual((receipt["run_id"], doc["run_id"]), (plan["run_id"], plan["run_id"]))
        self.assertEqual(receipt["process_identity"], native)
        self.assertEqual(receipt["after_parameter_sha256"], result.parameter_sha256)
        body = Path(ref["path"]).read_bytes()
        self.assertEqual((len(body), _sha(body)), (ref["bytes"], ref["sha256"]))
        self.assertLessEqual(len(body), live.MAX_UPDATE_RECEIPT_BYTES)
        self.assertEqual(json.loads(body), receipt)
        names = sorted(os.listdir(update_dir / plan["run_id"]))
        self.assertEqual(names, [f"update-{k:010d}.json" for k in range(1, 5)])
        chain = [json.loads((update_dir / plan["run_id"] / n).read_bytes()) for n in names]
        self.assertEqual(chain[0]["before_parameter_sha256"], result.initial_parameter_sha256)
        for prev, nxt in zip(chain, chain[1:]):             # every step reported: after(k) == before(k+1)
            self.assertEqual(prev["after_parameter_sha256"], nxt["before_parameter_sha256"])

        proc, _, _ = self.run_bridge(out)                   # the audit identity is kept: accepted
        observed = self.check_observation(proc, out, raw)
        self.assertEqual(observed["source"]["run_id"], plan["run_id"])

        bad = copy.deepcopy(doc)                            # trainer identity in process_identity: rejected
        bad["process_identity"] = copy.deepcopy(doc["native_process_identity"])
        bad_path = self.tmp / "trainer-identity-live.json"
        common.atomic_write(bad_path, json.dumps(bad, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        proc, _, _ = self.run_bridge(bad_path)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("process identity not owner-qualified", proc.stderr)
        with self.assertRaises(live.LiveRefused):         # and the builder never produces such a document
            live.build_document({**base, "process_identity": native}, live.LiveState(phase="preflight"), schema=schema)


class P7Test8cDlMetrics(_TmpCase):
    NODE_SCRIPT = r"""
'use strict';
const fs=require('fs'),vm=require('vm'),crypto=require('crypto');
const input=JSON.parse(fs.readFileSync(0,'utf8'));
const sha=s=>crypto.createHash('sha256').update(s).digest('hex');
const health=fs.readFileSync(input.health_path,'utf8'),unifiedRaw=fs.readFileSync(input.unified_path),unified=unifiedRaw.toString('utf8');
const freshSlice=health.slice(health.indexOf('function fresh('),health.indexOf('function liveState('));
const dlSlice=unified.slice(unified.indexOf('function dlMetrics('),unified.indexOf('function renderDLOperations('));
const pins={'health.js#fresh':sha(Buffer.from(freshSlice,'utf8')),'unified.js#dlMetrics':sha(Buffer.from(dlSlice,'utf8')),'unified.js':sha(unifiedRaw)};
for(const k of Object.keys(pins)){if(pins[k]!==input.pins[k]){process.stdout.write(JSON.stringify({error:'re-pin required',pins}));process.exit(3);}}
const now=input.now_ms,context={Date:{now:()=>now},Number,Set,Array,Error};
vm.createContext(context);
vm.runInContext(freshSlice,context);
vm.runInContext(dlSlice,context);
const results={};
for(const c of input.cases)results[c.name]=context.dlMetrics(c.o,c.v,now);
results.bound_default_now=context.dlMetrics(input.cases[0].o,input.cases[0].v);
process.stdout.write(JSON.stringify({pins,node:process.version,results}));
"""

    def test_dlmetrics_reports_training_only_for_a_bound_receipt(self):
        node = shutil.which("node") or ("/usr/bin/nodejs" if os.path.exists("/usr/bin/nodejs") else None)
        if node is None:
            self.fail("NOT_EXECUTED: node absent")
        health = (live.CONSUMER_DIR / "health.js").read_text(encoding="utf-8")
        unified_raw = (live.CONSUMER_DIR / "unified.js").read_bytes()
        unified = unified_raw.decode("utf-8")
        local_pins = {
            "health.js#fresh": _sha(health[health.index("function fresh("):health.index("function liveState(")]),
            "unified.js#dlMetrics": _sha(unified[unified.index("function dlMetrics("):
                                                 unified.index("function renderDLOperations(")]),
            "unified.js": _sha(unified_raw),
        }
        for key, value in local_pins.items():
            self.assertEqual(value, live.CONSUMER_PINS[key], f"re-pin required: {key}")

        schema, _ = live.load_schema()
        base = live.trainer_base_from_audit(json.loads(CLAUDE_LIVE.read_bytes()))
        native = live.native_process_identity()
        run_id, update_dir = "p7-live-8c", self.tmp / "updates"
        before, after = _sha("parameters before step 3"), _sha("parameters after step 3")
        receipt, ref = live.write_update_receipt(update_dir, run_id=run_id, optimizer_step=3, process_identity=native,
                                                 before_sha256=before, after_sha256=after)
        state = live.LiveState(phase="stepping", training_admitted=True, admission_ref=FAKE_ADMISSION_REF,
                               run_id=run_id, native_process_identity=native, update_receipt=receipt, update_ref=ref,
                               optimizer_parameter_change_verified=True,
                               initial_parameter_sha256=_sha("initial parameters"), current_parameter_sha256=after,
                               optimizer_updates_total=3, verified_utc_us=receipt["utc_us"], train_loss=1.25)
        doc = live.build_document(base, state, schema=schema)
        self.assertEqual(doc["status"], "TRAINING")

        stale_receipt, stale_ref = live.write_update_receipt(
            update_dir, run_id=run_id, optimizer_step=2, process_identity=native, before_sha256=_sha("b2"),
            after_sha256=before, utc_us=doc["utc_us"] - STALE_RECEIPT_US)
        stale_doc = live.build_document(base, dataclasses.replace(state, update_receipt=stale_receipt,
                                                                  update_ref=stale_ref), schema=schema)
        unadmitted_doc = live.build_document(base, dataclasses.replace(state, training_admitted=False,
                                                                       admission_ref=None), schema=schema)
        self.assertEqual(unadmitted_doc["status"], "AUDITING_PLAN_UNADMITTED")
        wrong_step = copy.deepcopy(doc)
        wrong_step["optimizer_step"] = 4
        wrong_run = copy.deepcopy(doc)
        wrong_run["optimizer_update_receipt"]["run_id"] = "another-run"
        same_hash = copy.deepcopy(doc)                     # hand-built: write_update_receipt refuses before == after
        same_hash["optimizer_update_receipt"] = {"schema": live.UPDATE_RECEIPT_SCHEMA, "utc_us": receipt["utc_us"],
                                                 "run_id": run_id, "optimizer_step": 3, "process_identity": native,
                                                 "before_parameter_sha256": before, "after_parameter_sha256": before}

        def observation(source: dict) -> dict:
            raw = live.encode_document(source)
            return {"schema": OBSERVATION_SCHEMA, "source": source,
                    "source_pin": {"path": str(self.tmp / "trainer-live.json"), "sha256": _sha(raw), "bytes": len(raw)}}

        observer_utc_us = common.utc_us()

        def unified_observation(o: dict) -> dict:
            return {"utc_us": observer_utc_us, "native_training_process_verified": True,
                    "native_update_receipt_verified": True, "native_training_run_id": run_id,
                    "dl_source_pin": {"sha256": o["source_pin"]["sha256"], "bytes": o["source_pin"]["bytes"]}}

        bound = observation(doc)
        mismatched = unified_observation(bound)
        mismatched["dl_source_pin"]["sha256"] = _sha("another producer body")
        cases = [("bound", bound, unified_observation(bound))]
        for name, source in (("wrong_step", wrong_step), ("wrong_run_id", wrong_run),
                             ("before_equals_after", same_hash), ("stale_receipt", stale_doc),
                             ("unadmitted", unadmitted_doc)):
            o = observation(source)
            cases.append((name, o, unified_observation(o)))
        cases.append(("mismatched_pin", bound, mismatched))
        payload = {"health_path": str(live.CONSUMER_DIR / "health.js"),
                   "unified_path": str(live.CONSUMER_DIR / "unified.js"), "pins": local_pins, "now_ms": observer_utc_us // 1000 + 1000,
                   "cases": [{"name": n, "o": o, "v": v} for n, o, v in cases]}
        proc = subprocess.run([node, "-e", self.NODE_SCRIPT], input=json.dumps(payload, allow_nan=False), text=True,
                              capture_output=True, timeout=60, cwd=str(self.tmp))
        self.assertEqual(proc.returncode, 0, proc.stdout[-2000:] + proc.stderr[-2000:])
        out = json.loads(proc.stdout)
        self.assertEqual(out["pins"], local_pins)
        res = out["results"]
        expected = {  # name: (training, known, validUpdate)
            "bound": (True, True, True),
            "bound_default_now": (True, True, True),
            "wrong_step": (False, True, False),
            "wrong_run_id": (False, True, False),
            "before_equals_after": (False, True, False),
            "stale_receipt": (False, True, False),
            "unadmitted": (False, True, True),
            "mismatched_pin": (False, False, True),
        }
        for name, (training, known, valid) in expected.items():
            with self.subTest(case=name):
                got = res[name]
                self.assertIs(got["training"], training)
                self.assertIs(got["known"], known)
                self.assertIs(got["validUpdate"], valid)
                self.assertIs(got["sourceFresh"], True)


class LiveUnitChecks(_TmpCase):
    def setUp(self):
        super().setUp()
        self.schema, _ = live.load_schema()
        self.base = live.trainer_base_from_audit(json.loads(CLAUDE_LIVE.read_bytes()))
        self.native = live.native_process_identity()
        self.receipt, self.ref = live.write_update_receipt(self.tmp / "unit-updates", run_id="unit-run",
                                                           optimizer_step=5, process_identity=self.native,
                                                           before_sha256=_sha("b5"), after_sha256=_sha("a5"))

    def verified_state(self, **kw) -> live.LiveState:
        fields = dict(phase="stepping", run_id="unit-run", native_process_identity=self.native,
                      update_receipt=self.receipt, update_ref=self.ref, optimizer_parameter_change_verified=True,
                      initial_parameter_sha256=_sha("init"), current_parameter_sha256=_sha("a5"),
                      optimizer_updates_total=5, verified_utc_us=self.receipt["utc_us"], train_loss=1.0, epoch=1)
        fields.update(kw)
        return live.LiveState(**fields)

    def test_writer_refuses_claude_paths_and_intervals_above_3_s(self):
        link = self.tmp / "audit-link"
        link.symlink_to(live.CLAUDE_DIR, target_is_directory=True)
        guard_atomic, guard_immutable = _write_guards()
        try:
            with guard_atomic, guard_immutable:
                for bad in (CLAUDE_LIVE, live.CLAUDE_DIR / "sub" / "x.json",
                            live.CLAUDE_DIR / ".." / "claude" / "y.json", link / "live.json"):
                    with self.subTest(path=str(bad)), self.assertRaises(live.LiveRefused):
                        live.LiveWriter(bad, schema=self.schema, base=self.base, interval_s=1.0)
                    with self.subTest(receipt=str(bad)), self.assertRaises(live.LiveRefused):
                        live.write_update_receipt(bad, run_id="r", optimizer_step=1, process_identity=self.native,
                                                  before_sha256=_sha("b"), after_sha256=_sha("a"))
        finally:
            link.unlink()                                   # removes the link itself, never its target
        for interval in (3.5, 3.0001, 0, -1, float("nan")):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                live.LiveWriter(self.tmp / "live.json", schema=self.schema, base=self.base, interval_s=interval)
        writer = live.LiveWriter(self.tmp / "live.json", schema=self.schema, base=self.base, interval_s=3.0)
        self.assertEqual(writer.interval_s, 3.0)

    def test_trainer_refuses_claude_workdir_and_update_dir(self):
        data = fixtures_p7.train_data()
        plan = _make_plan(max_updates=1)
        boom = AssertionError("trainer created a directory after a claude/ path should have been refused")
        guard_atomic, guard_immutable = _write_guards()
        with mock.patch.object(Path, "mkdir", side_effect=boom), guard_atomic, guard_immutable:
            with self.assertRaises(live.LiveRefused):
                train.run(plan, data, live.CLAUDE_DIR / "dltrain-work")
            with self.assertRaises(live.LiveRefused):
                train.run(plan, data, self.tmp / "work", update_dir=live.CLAUDE_DIR / "updates")
        self.assertFalse((self.tmp / "work").exists())

    def test_update_receipts_are_create_only_bounded_and_require_a_change(self):
        udir = self.tmp / "updates"
        receipt, ref = live.write_update_receipt(udir, run_id="r1", optimizer_step=1, process_identity=self.native,
                                                 before_sha256=_sha("b"), after_sha256=_sha("a"))
        path = Path(ref["path"])
        self.assertEqual(path, udir / "r1" / "update-0000000001.json")
        body = path.read_bytes()
        self.assertEqual((ref["bytes"], ref["sha256"]), (len(body), _sha(body)))
        self.assertEqual(json.loads(body), receipt)
        self.assertEqual(receipt["schema"], live.UPDATE_RECEIPT_SCHEMA)
        self.assertEqual(os.stat(path).st_mode & 0o222, 0)
        with self.assertRaises(common.ImmutableConflict):  # create-only: never overwritten
            live.write_update_receipt(udir, run_id="r1", optimizer_step=1, process_identity=self.native,
                                      before_sha256=_sha("b"), after_sha256=_sha("c"))
        self.assertEqual(path.read_bytes(), body)
        refusals = {
            "before_equals_after": dict(optimizer_step=2, before_sha256=_sha("b"), after_sha256=_sha("b")),
            "oversize": dict(optimizer_step=3, process_identity=dict(self.native, padding="x" * 70_000)),
            "audit_identity": dict(optimizer_step=4, process_identity=self.base["process_identity"]),
            "step_zero": dict(optimizer_step=0),
            "not_hex": dict(optimizer_step=5, after_sha256="A" * 64),
            "unsafe_run_id": dict(optimizer_step=6, run_id="../escape"),
        }
        for name, kw in refusals.items():
            args = dict(run_id="r1", process_identity=self.native, before_sha256=_sha("b"), after_sha256=_sha("a"))
            args.update(kw)
            with self.subTest(refusal=name), self.assertRaises(live.LiveRefused):
                live.write_update_receipt(udir, **args)
        self.assertEqual(sorted(os.listdir(udir / "r1")), ["update-0000000001.json"])
        self.assertEqual(sorted(os.listdir(udir)), ["r1"])

    def test_unverified_documents_keep_history_empty_and_losses_epoch_null(self):
        state = live.LiveState(phase="stepping", training_admitted=True, admission_ref=FAKE_ADMISSION_REF,
                               run_id="unit-run", native_process_identity=self.native, train_loss=1.0,
                               validation_loss=2.0, validation_split="dev-family-v1", epoch=3,
                               history=({"epoch": 1, "train_loss": 1.0, "validation_loss": 2.0},),
                               metrics={"token_validity": 0.5})
        doc = live.build_document(self.base, state, schema=self.schema)
        self.assertEqual(live.validate(doc, self.schema), [])
        self.assertEqual(doc["history"], [])
        for key in ("epoch", "train_loss", "validation_loss", "validation_split", "training_proof", "optimizer_step"):
            self.assertIsNone(doc[key], key)
        self.assertIsNone(doc["metrics"]["token_validity"])
        self.assertFalse(doc["optimizer_parameter_change_verified"])
        self.assertEqual(doc["status"], "ADMITTED_PREFLIGHT")       # admitted and stepping, but no receipt yet
        unknown = {e["field"]: e["reason"] for e in doc["unknown"]}
        self.assertIn("optimizer_parameter_change_verified", unknown["train_loss"])
        with self.assertRaises(live.LiveRefused):          # a receipt cannot ride on an unverified document
            live.build_document(self.base, dataclasses.replace(
                self.verified_state(), optimizer_parameter_change_verified=False), schema=self.schema)
        with self.assertRaises(live.LiveRefused):          # verified needs initial != current
            live.build_document(self.base, self.verified_state(current_parameter_sha256=_sha("init")),
                                schema=self.schema)

    def test_validation_loss_requires_validation_split_dev_family_v1(self):
        for split in (None, "heldout", "train_fit", "dev"):
            with self.subTest(split=split), self.assertRaises(live.LiveRefused):
                live.build_document(self.base, self.verified_state(validation_loss=1.5, validation_split=split),
                                    schema=self.schema)
        doc = live.build_document(self.base, self.verified_state(validation_loss=1.5, validation_split="dev-family-v1"),
                                  schema=self.schema)
        self.assertEqual((doc["validation_loss"], doc["validation_split"]), (1.5, "dev-family-v1"))
        self.assertEqual(live.validate(doc, self.schema), [])
        stripped = copy.deepcopy(doc)
        stripped["validation_split"] = None
        self.assertNotEqual(live.validate(stripped, self.schema), [])
        with self.assertRaises(live.LiveRefused):          # heldout values never come from the trainer
            live.build_document(self.base, self.verified_state(metrics={"heldout_contexts_evaluated": 3}),
                                schema=self.schema)

    def test_nan_and_infinity_are_refused(self):
        bad_states = {
            "train_loss": dict(train_loss=float("nan")),
            "gradient_norm": dict(gradient_norm=float("inf")),
            "rate": dict(rates={"examples_per_second": float("nan")}),
            "history": dict(validation_split="dev-family-v1",
                            history=({"epoch": 1, "train_loss": float("nan"), "validation_loss": 1.0},)),
            "metric": dict(metrics={"token_validity": float("-inf")}),
        }
        for name, kw in bad_states.items():
            with self.subTest(field=name), self.assertRaises(live.LiveRefused):
                live.build_document(self.base, self.verified_state(**kw), schema=self.schema)
        with self.assertRaises(live.LiveRefused):
            live.encode_document({"x": float("nan")})
        with self.assertRaises(live.LiveRefused):
            live.encode_document({"x": "y" * (live.MAX_DOC_BYTES + 1)})

    def test_history_decimation_keeps_first_and_last_rows(self):
        rows = [{"epoch": e, "optimizer_step": 11 * e, "train_loss": 1.0 / e, "validation_loss": 2.0 / e}
                for e in range(1, 451)]
        kept = live.decimate_history(rows)
        self.assertEqual(len(kept), live.MAX_HISTORY)
        self.assertEqual((kept[0], kept[-1]), (rows[0], rows[-1]))
        self.assertTrue(all(a["epoch"] < b["epoch"] for a, b in zip(kept, kept[1:])))
        self.assertEqual(live.decimate_history(rows[:5]), rows[:5])
        doc = live.build_document(self.base, self.verified_state(history=tuple(rows), validation_loss=0.5,
                                                                 validation_split="dev-family-v1"), schema=self.schema)
        self.assertEqual(len(doc["history"]), live.MAX_HISTORY)
        self.assertEqual((doc["history"][0], doc["history"][-1]), (rows[0], rows[-1]))
        self.assertEqual(live.validate(doc, self.schema), [])
        with self.assertRaises(live.LiveRefused):
            live.decimate_history([rows[1], rows[0]])

    def test_status_mapping(self):
        for phase in live.PHASES:
            self.assertEqual(live.status_for(phase, False, True), "AUDITING_PLAN_UNADMITTED")
        self.assertEqual(live.status_for("stepping", True, True), "TRAINING")
        self.assertEqual(live.status_for("stepping", True, False), "ADMITTED_PREFLIGHT")
        self.assertEqual(live.status_for("checkpointing", True, True), "CHECKPOINTING")
        self.assertEqual(live.status_for("yielded", True, True), "YIELDED_TO_INTAKE")
        with self.assertRaises(ValueError):
            live.status_for("training", True, True)

    def test_native_identity_matches_unified_observer_proc_computation(self):
        source = UNIFIED_OBSERVER.read_bytes()
        self.assertEqual(_sha(source), UNIFIED_OBSERVER_SHA256, "re-pin required: unified_observer.py")
        remote = next(node.value.value for node in ast.parse(source).body if isinstance(node, ast.Assign)
                      and any(getattr(t, "id", None) == "GPU_REMOTE" for t in node.targets))
        lines = [line.strip() for line in remote.splitlines()]
        start = next(i for i, line in enumerate(lines) if line.startswith("pid=IDENTITY['pid'];"))
        self.assertTrue(lines[start + 1].startswith("actual={"))
        program = "\n".join(["import hashlib,json,pathlib", f"IDENTITY={{'pid':{os.getpid()}}}", lines[start],
                             lines[start + 1], "print(json.dumps(actual,sort_keys=True))", ""])
        proc = subprocess.run([sys.executable, "-"], input=program, text=True, capture_output=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        actual = json.loads(proc.stdout)
        mine = live.native_process_identity()
        for key in ("pid", "starttick", "cwd", "executable_sha256", "argv_sha256"):
            self.assertEqual(str(actual[key]), str(mine[key]), key)
        self.assertEqual(mine["host"], socket.gethostname())
        self.assertIsInstance(mine["pid"], int)
        self.assertIsInstance(mine["starttick"], str)

    def test_writer_thread_publishes_on_schedule_and_records_errors(self):
        out = self.tmp / "live.json"
        writer = live.LiveWriter(out, schema=self.schema, base=self.base, interval_s=0.05)
        writer.update(self.verified_state())
        writer.start()
        deadline = time.monotonic() + 10
        while writer.writes < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertGreaterEqual(writer.writes, 3)
        self.assertIsNotNone(writer.stop())
        self.assertIsNone(writer.error)
        self.assertEqual(live.validate(json.loads(out.read_bytes()), self.schema), [])
        broken = live.LiveWriter(self.tmp / "broken.json", schema=self.schema, base=self.base, interval_s=0.05)
        broken.update(self.verified_state(train_loss=float("nan")))
        broken.start()
        deadline = time.monotonic() + 10
        while broken.running and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertIsInstance(broken.error, live.LiveRefused)
        self.assertIsNone(broken.stop())
        self.assertFalse((self.tmp / "broken.json").exists())


class TrainEvaluateChecks(_TmpCase):
    @classmethod
    def setUpClass(cls):
        cls.data = fixtures_p7.train_data()

    def test_dev_metrics_refuses_heldout_before_any_data_access(self):
        class Untouchable:
            def __getattr__(self, name):
                raise AssertionError(f"data or model accessed ({name}) before the split refusal")

        for split in ("heldout", "held", "sealed_excluded", "train_fit", "HELDOUT", "dev ", None):
            with self.subTest(split=split), self.assertRaises(evaluate.EvaluateRefused):
                evaluate.dev_metrics(Untouchable(), Untouchable(), split=split)

    def test_dev_metrics_keeps_rng_and_mode_and_reports_family_macro_ce(self):
        torch.manual_seed(1234)
        net = model.PolicyLM(fixtures_p7.model_config())
        net.train()
        random.seed(99)
        np.random.seed(7)
        before = _rng_snapshot()
        metrics = evaluate.dev_metrics(net, self.data, split="dev")
        _assert_rng_equal(self, before, _rng_snapshot())
        self.assertTrue(net.training)
        corpus = self.data.corpus
        dev = sorted(c for c, row in corpus.contexts.items()
                     if row["split_role"] == "dev" and corpus.assignment[row["family_id"]] == "dev")
        self.assertEqual((metrics["schema"], metrics["split"], metrics["role"], metrics["validation_split"]),
                         ("jope.dl-eval-dev/v1", "dev", "training_validation", "dev-family-v1"))
        self.assertEqual(metrics["contexts"], len(dev))
        self.assertEqual(metrics["families"], len({corpus.contexts[c]["family_id"] for c in dev}))

        per_family, all_terms = defaultdict(list), []      # independent float64 recomputation, one sequence at a time
        net.eval()
        with torch.no_grad():
            for ctx in dev:
                plan = sampler.context_plan(corpus, ctx, self.data.config)
                if plan.stream != "group":
                    continue
                z = torch.from_numpy(np.array(corpus.encodings[ctx], dtype=np.float32)).reshape(1, -1)
                terms = []
                for prog in plan.teachers:
                    toks = list(corpus.tokens(ctx, prog))
                    mask = fixtures_p7.legal_mask(toks)
                    out = net(torch.tensor([toks]), torch.tensor([len(toks)]), z, torch.tensor([0]))
                    logits = out.logits[0].double()
                    for i in range(len(toks) - 1):
                        allowed = torch.from_numpy(np.flatnonzero(mask[i]))
                        terms.append(float(torch.logsumexp(logits[i, allowed], 0) - logits[i, toks[i + 1]]))
                per_family[corpus.contexts[ctx]["family_id"]].append(math.fsum(terms) / len(terms))
                all_terms.extend(terms)
        net.train()
        self.assertTrue(per_family, "fixture dev split has no group-stream context")
        macro = math.fsum(math.fsum(v) / len(v) for v in per_family.values()) / len(per_family)
        self.assertAlmostEqual(metrics["ce_per_token_family_macro"], macro, delta=1e-4)
        self.assertAlmostEqual(metrics["ce_per_token_token_micro"], math.fsum(all_terms) / len(all_terms), delta=1e-4)
        self.assertEqual(metrics["tokens"], len(all_terms))
        self.assertGreater(metrics["pairs"], 0)
        self.assertTrue(0.0 <= metrics["token_validity"] <= 1.0)
        self.assertTrue(0.0 <= metrics["pairwise_accuracy_family_macro"] <= 1.0)
        diag = evaluate.dev_metrics(net, self.data, split="diag")
        self.assertEqual((diag["role"], diag["validation_split"]), ("diagnostic_only", None))
        self.assertGreater(diag["contexts"], 0)

    def test_stop_file_before_step_1_yields_at_step_0_with_a_commit(self):
        plan = _make_plan(max_updates=6)
        stop, work = self.tmp / "STOP", self.tmp / "work"
        common.atomic_write(stop, b"")
        store = ckpt.FileVersionedStore(self.tmp / "store")
        res = train.run(plan, self.data, work, store=store, store_prefix="p7/", stop_file=stop)
        self.assertEqual((res.status, res.reason, res.step), ("YIELDED_TO_INTAKE", "stop_file", 0))
        self.assertFalse(res.optimizer_parameter_change_verified)
        self.assertEqual(res.parameter_sha256, res.initial_parameter_sha256)
        commit_dir = work / "commits" / f"{0:010d}"
        self.assertEqual(res.last_commit, str(commit_dir))
        closure = ckpt.read_commit(commit_dir)
        self.assertEqual((closure.step, closure.resumable, closure.journal_bytes), (0, True, b""))
        self.assertEqual(closure.sampler_state["cursor"], 0)
        self.assertEqual(res.journal_sha256, _sha(b""))
        self.assertEqual(res.published["pointer"]["seq"], 1)
        self.assertEqual(ckpt.load_expected(work).to_json(), res.published["pointer"])
        restored, pointer = ckpt.restore_closure(store, prefix="p7/", run_id=plan["run_id"], dest_dir=self.tmp / "rst")
        self.assertEqual(restored.doc["parameter_sha256"], res.parameter_sha256)
        self.assertEqual(pointer.to_json(), res.published["pointer"])
        names = [e["event"] for e in _events(work)]
        self.assertEqual(names[0], "run_started")
        self.assertEqual(names[-1], "finished")
        self.assertIn("commit", names)
        self.assertIn("published", names)
        journal_lines = [json.loads(line) for line in (work / "journal.jsonl").read_bytes().splitlines()]
        self.assertEqual(journal_lines, [])

        resumed = train.resume_from_commit(plan, self.data, commit_dir, self.tmp / "resumed")
        straight = train.run(plan, self.data, self.tmp / "straight")
        for r in (resumed, straight):
            self.assertEqual((r.status, r.step), ("COMPLETED", 6))
            self.assertTrue(r.optimizer_parameter_change_verified)
        self.assertEqual(resumed.parameter_sha256, straight.parameter_sha256)
        self.assertEqual(resumed.journal_sha256, straight.journal_sha256)
        self.assertEqual(_events(self.tmp / "resumed")[0]["event"], "resumed")

    def test_deadline_in_the_past_gives_stopped_budget(self):
        plan = _make_plan(max_updates=6, deadline_utc_us=common.utc_us() - 1_000_000)
        work = self.tmp / "work"
        res = train.run(plan, self.data, work)
        self.assertEqual((res.status, res.reason, res.step), ("STOPPED_BUDGET", "deadline", 0))
        self.assertEqual(res.last_commit, str(work / "commits" / f"{0:010d}"))
        self.assertIsNone(res.published)
        self.assertEqual(ckpt.read_commit(Path(res.last_commit)).step, 0)

    def test_nonfinite_objective_fails_without_a_new_commit(self):
        plan = _make_plan(max_updates=6, train=fixtures_p7.train_config(commit_every_updates=2))
        real = objective.successor_v1
        calls = []

        def nan_on_third(out, batch):
            result = real(out, batch)
            calls.append(batch.step)
            if len(calls) == 3:
                result.loss = result.loss * float("nan")
            return result

        work = self.tmp / "work"
        with mock.patch.object(objective, "successor_v1", nan_on_third):
            res = train.run(plan, self.data, work)
        self.assertEqual((res.status, res.reason, res.step), ("FAILED", "nonfinite_objective", 2))
        self.assertEqual(sorted(os.listdir(work / "commits")), [f"{2:010d}"])
        self.assertEqual(res.last_commit, str(work / "commits" / f"{2:010d}"))
        self.assertEqual(len((work / "journal.jsonl").read_bytes().splitlines()), 2)
        commit_doc = json.loads((work / "commits" / f"{2:010d}" / "commit.json").read_bytes())
        self.assertEqual(commit_doc["parameter_sha256"], res.parameter_sha256)   # the bad step was never applied

        def raising(out, batch):
            raise objective.ObjectiveNonFinite("forced non-finite term")

        work2 = self.tmp / "work2"
        with mock.patch.object(objective, "successor_v1", raising):
            res2 = train.run(plan, self.data, work2)
        self.assertEqual((res2.status, res2.reason, res2.step, res2.last_commit),
                         ("FAILED", "nonfinite_objective", 0, None))
        self.assertFalse((work2 / "commits").exists())
        self.assertEqual(res2.parameter_sha256, res2.initial_parameter_sha256)

    def test_fresh_run_refuses_an_existing_journal(self):
        work = self.tmp / "work"
        work.mkdir()
        common.atomic_write(work / "journal.jsonl", b"")
        before = sorted(os.listdir(work))
        with self.assertRaises(FileExistsError):
            train.run(_make_plan(max_updates=2), self.data, work)
        self.assertEqual(sorted(os.listdir(work)), before)

    def test_run_plan_is_create_only_and_hash_stable(self):
        plan = _make_plan()
        self.assertEqual(plan["schema"], train.RUN_PLAN_SCHEMA)
        self.assertEqual(plan["objective_sha256"], objective.objective_sha256("successor-v1"))
        self.assertEqual(plan["determinism"], model.configure_determinism(4))
        self.assertEqual(plan["schedule"]["id"], objective.SCHEDULE_ID)
        self.assertEqual(plan["run_seed_hex"], fixtures_p7.RUN_SEED.hex())
        path = self.tmp / "run-plan.json"
        written = train.write_run_plan(path, plan)
        self.assertEqual(written["sha256"], train.plan_sha256(plan))
        self.assertEqual(train.load_run_plan(path), plan)
        self.assertEqual(train.plan_sha256(_make_plan()), train.plan_sha256(plan))
        with self.assertRaises(common.ImmutableConflict):
            train.write_run_plan(path, _make_plan(max_updates=81))
        cfg = fixtures_p7.train_config()
        self.assertEqual(train.TrainConfig.from_json(json.loads(common.canonical(cfg.to_json()))), cfg)
        with self.assertRaises(ValueError):
            train.TrainConfig.from_json({**cfg.to_json(), "unknown": 1})
        with self.assertRaises(ValueError):
            train.TrainConfig(live_interval_s=3.5)
        for bad in (dict(run_seed=b"short"), dict(run_id="../x"), dict(refs={"heldout": "x"}), dict(max_updates=-1)):
            with self.subTest(bad=sorted(bad)), self.assertRaises(ValueError):
                _make_plan(**bad)

    def test_dev_validation_cadence_feeds_history_and_admitted_training_documents(self):
        epoch_len = train.epoch_steps(self.data)
        plan = _make_plan(max_updates=epoch_len + 1,
                          train=fixtures_p7.train_config(eval_every_epochs=1, receipt_interval_s=0.0))
        schema, _ = live.load_schema()
        base = live.trainer_base_from_audit(json.loads(CLAUDE_LIVE.read_bytes()))
        out = self.tmp / "live.json"
        writer = live.LiveWriter(out, schema=schema, base=base, interval_s=0.05)
        seen = {}

        def after_step(step: int) -> None:
            if step == 2:                                   # publish the trainer's current snapshot synchronously
                writer.publish(writer.snapshot())
                seen["doc"] = json.loads(out.read_bytes())

        store = ckpt.FileVersionedStore(self.tmp / "store")
        res = train.run(plan, self.data, self.tmp / "work", store=store, store_prefix="p7/", live_writer=writer,
                        update_dir=self.tmp / "updates", training_admitted=True, admission_ref=FAKE_ADMISSION_REF,
                        hooks=train.TrainHooks(after_step))
        self.assertEqual((res.status, res.step), ("COMPLETED", epoch_len + 1))
        mid = seen["doc"]
        self.assertEqual(live.validate(mid, schema), [])
        self.assertEqual((mid["status"], mid["optimizer_step"], mid["training_admitted"]), ("TRAINING", 2, True))
        self.assertEqual(mid["optimizer_update_receipt"]["process_identity"], mid["native_process_identity"])
        final = json.loads(out.read_bytes())
        self.assertEqual(live.validate(final, schema), [])
        self.assertEqual(final["status"], "COMPLETED")
        self.assertEqual(len(final["history"]), 1)
        row = final["history"][0]
        self.assertEqual((row["epoch"], row["optimizer_step"]), (1, epoch_len))
        self.assertEqual(final["validation_loss"], row["validation_loss"])
        self.assertEqual(final["validation_split"], "dev-family-v1")
        self.assertIsNotNone(final["metrics"]["token_validity"])
        manifest_ref = res.published["manifest_ref"]
        self.assertEqual(final["checkpoint_ref"], {k: v for k, v in manifest_ref.items() if k != "new_bytes"})
        self.assertEqual(final["checkpoint"], {"step": epoch_len + 1, "upload_status": "PUBLISHED_VERIFIED",
                                               "pointer_version_id": res.published["pointer"]["version_id"]})
        self.assertEqual(final["checkpoint_destination_readback"], {"status": "pass"})
        self.assertGreater(final["storage_bytes"], 0)
        self.assertEqual(final["storage_cap_bytes"], plan["ceilings"]["storage_cap_bytes"])
        evals = [e for e in _events(self.tmp / "work") if e["event"] == "dev_validation"]
        self.assertEqual(len(evals), 1)
        self.assertEqual(evals[0]["metrics"]["validation_split"], "dev-family-v1")
        self.assertEqual(evals[0]["metrics"]["ce_per_token_family_macro"], row["validation_loss"])

    def test_gradient_scale_audit_reports_every_term_and_leaves_no_trace(self):
        torch.manual_seed(5)
        net = model.PolicyLM(fixtures_p7.model_config())
        draw = sampler.MacroSampler(self.data.fit, fixtures_p7.RUN_SEED, self.data.config).draw(1)
        batch = sampler.collate(draw, self.data)
        coefficients = dict(objective.COEF_SUCCESSOR)
        before = _rng_snapshot()
        audit = train.gradient_scale_audit(net, batch, objective.SUCCESSOR_V1)
        _assert_rng_equal(self, before, _rng_snapshot())
        self.assertEqual(objective.COEF_SUCCESSOR, coefficients)
        self.assertEqual(set(audit["term_norms"]), set(coefficients))
        self.assertTrue(all(math.isfinite(v) and v >= 0 for v in audit["term_norms"].values()))
        self.assertGreater(audit["ce_norm"], 0.0)
        self.assertEqual(audit["ce_norm"], audit["term_norms"]["ce"])
        self.assertTrue(set(audit["exceeds_10x"]) <= set(coefficients) - {"ce"})
        self.assertTrue(all(p.grad is None for p in net.parameters()))
        self.assertTrue(net.training)


if __name__ == "__main__":
    unittest.main()
