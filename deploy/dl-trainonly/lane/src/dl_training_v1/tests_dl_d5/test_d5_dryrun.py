"""dryrun: the step plan, the refusals that keep it to DRAFT and what-if files under the staging dir, and the
orchestration itself, driven through stand-in entry points. The plan is families build (CLI), families verify (API,
receipts_dir = RUN/receipts/P2, full rehash by default) and families publish (CLI); then holdout commit, freeze,
verify --assignment-poison, closure and publish, all with draft_dry_run=True. The P2 receipt is pinned explicitly.

TestDryrunPipeline replaces families._main, families.verify and holdout._main with fakes. The fakes:
- check the argv the tool built, and that the decision bytes arrive through a memfd as the rehearsal variant, never as
  a file;
- write where the real entry points write;
- PUT through s3io.put_create_only on the tool's own shared FakeBucket (the real lane guard applies).
The tests prove:
- the step order and the pins handed from step to step;
- the PUT counts and the byte-identical alias-sources copy;
- that holdout's draft_dry_run mode is used and the real-client factories are refused;
- the write audit, and the STOP receipt of a failed step;
- D-MAP-6: the rehearsal tree is inventoried into the receipt and then removed, whatever the verdict.
The real fake-S3 run on the real entry points is a separate, receipted step of the run agent on g2."""
import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

from tests_dl_d5.fixtures_dl_d5 import (ALIAS, ALIAS_PIN, CSV_PIN, EPOCH, MASTER, OWNER_DECISIONS, PRE_PIN, REGISTRY_DB,
                                        RUN, D5Case, T, dryrun_args, forget_violation, sha)
from dltrain import common, families

STAGE = Path("/home/ubuntu/scratch/jopedime-dl-training-v1/tmp/dl-d5-g2")
STRAY = "/nonexistent-dl-d5-audit-probe/stray.txt"      # a failing open outside TMPDIR: audited, never written
STEPS = ["families.build", "families.verify", "families.publish", "holdout.commit", "holdout.freeze", "holdout.verify",
         "holdout.closure", "holdout.publish"]


class TestDryrunPlan(unittest.TestCase):
    def test_default_roots_are_the_g2_paths(self):
        r = T.Roots()
        self.assertEqual((r.families, r.receipts, r.stage, r.stage_receipts),
                         (RUN / "families", RUN / "receipts" / "P2", RUN / "tmp" / "dl-d5-g2",
                          RUN / "receipts" / "P2" / "staging"))
        self.assertEqual((T.DRAFT_PREFIX, T.PREFIX["what_if"], T.OUT_NAME, T.MESSAGE_NAME),
                         ("DRAFT-UNCONFIRMED-", "WHATIF-NOT-A-DECISION-", "review-decisions.json",
                          "dl-d5-owner-message-{sha16}.txt"))
        self.assertEqual((T.MESSAGE_NAME, T.OUT_NAME), (families.OWNER_MESSAGE_NAME, families.DECISIONS_NAME))
        self.assertEqual(T.EXIT, {"PASS": 0, "STOP": 2, "ERROR": 3})

    def test_tree_and_dirs_carry_the_prefix(self):
        tree = T.dryrun_tree(STAGE, "draft", "ab" * 32, 1)
        self.assertEqual(tree, STAGE / ("DRAFT-UNCONFIRMED-rehearsal-" + "ab" * 8 + "-a1"))
        self.assertEqual(T.dryrun_tree(STAGE, "what_if", "ab" * 32, 2).name, "WHATIF-NOT-A-DECISION-rehearsal-"
                         + "ab" * 8 + "-a2")
        self.assertEqual({k: str(v.relative_to(tree)) for k, v in T.dryrun_dirs(tree).items()},
                         {"families": "families", "receipts": "receipts/P2", "holdout": "holdout"})

    def test_families_steps(self):
        dirs = T.dryrun_dirs(T.dryrun_tree(STAGE, "draft", "0" * 64, 1))
        steps = T.families_argv(dirs=dirs, alias=ALIAS, decisions="/proc/self/fd/7", decisions_sha256="1" * 64, utc=7)
        self.assertEqual([n for n, _ in steps], ["families.build", "families.publish"])
        build, publish = (a for _, a in steps)
        self.assertEqual((build[build.index("--config") + 1], build[build.index("--decisions") + 1]),
                         (str(ALIAS), "/proc/self/fd/7"))
        self.assertEqual(publish[publish.index("--verify-receipt") + 1], str(dirs["receipts"] / "verify-7.json"))
        for argv in (build, publish):
            for flag in ("--out", "--receipt"):
                if flag in argv:
                    p = Path(argv[argv.index(flag) + 1])
                    self.assertTrue(p.is_relative_to(STAGE) and "DRAFT-UNCONFIRMED-" in str(p.relative_to(STAGE)))

    def test_holdout_steps_order_and_pins(self):
        dirs = T.dryrun_dirs(T.dryrun_tree(STAGE, "draft", "0" * 64, 1))
        b = lambda p, s: T.Bound(Path(p), b"", s)  # noqa: E731
        steps = T.holdout_argv(dirs=dirs, decisions="/proc/self/fd/7", registry_sha256="1" * 64,
                               p2_receipt_sha256="2" * 64, owner=b(*OWNER_DECISIONS), epoch=b(*EPOCH),
                               master=b(*MASTER), registry_db=REGISTRY_DB, alias=b(ALIAS, ALIAS_PIN))
        self.assertEqual([n for n, _ in steps],
                         ["holdout.commit", "holdout.freeze", "holdout.verify", "holdout.closure", "holdout.publish"])
        commit, freeze, verify, closure, publish = (a for _, a in steps)
        self.assertEqual(commit[commit.index("--decisions") + 1], "/proc/self/fd/7")
        self.assertEqual(commit[commit.index("--p2-receipt") + 1], str(dirs["receipts"] / "receipt.json"))
        self.assertEqual(commit[commit.index("--p2-receipt-sha256") + 1], "2" * 64)
        self.assertEqual(commit[commit.index("--owner-decisions-sha256") + 1], OWNER_DECISIONS[1])
        self.assertIn("--assignment-poison", verify)
        for flag, want in (("--epoch-sha256", EPOCH[1]), ("--master-sha256", MASTER[1]),
                           ("--alias-sources-sha256", ALIAS_PIN), ("--registry-db", str(REGISTRY_DB))):
            self.assertEqual(closure[closure.index(flag) + 1], want)
        self.assertEqual(publish[publish.index("--closure") + 1], str(dirs["holdout"] / "closure.json"))
        for argv in (commit, freeze, closure):
            self.assertEqual(argv[argv.index("--out") + 1], str(dirs["holdout"]))
        everything = " ".join(a for _, argv in steps for a in argv)
        self.assertNotIn(str(RUN / "holdout"), everything)
        self.assertNotIn(str(RUN / "families" / "g2" / "registry.json"), everything)
        self.assertNotIn(str(RUN / "receipts" / "P2" / "receipt.json"), everything)


class TestDryrunRefusals(D5Case):
    def test_only_unconfirmed_files_in_the_stage(self):
        draft, dsha = self.draft()
        plain = Path(self.roots.stage) / "review-decisions.json"
        plain.write_bytes(draft.read_bytes())
        self.stop(dryrun_args(plain, dsha), "R10")
        outside = self.tmp / draft.name
        outside.write_bytes(draft.read_bytes())
        self.stop(dryrun_args(outside, dsha), "R10")
        wrong_prefix = Path(self.roots.stage) / ("WHATIF-NOT-A-DECISION-" + draft.name)
        wrong_prefix.write_bytes(draft.read_bytes())
        self.stop(dryrun_args(wrong_prefix, dsha), "R10")
        doc = json.loads(draft.read_bytes())
        doc["provenance"] = dict(doc["provenance"], note="edited")
        fake = Path(self.roots.stage) / "DRAFT-UNCONFIRMED-not-the-draft.json"
        fake.write_bytes(common.canonical(doc))
        self.stop(dryrun_args(fake, sha(fake.read_bytes())), "R9", contains="exact draft provenance")
        doc = json.loads(draft.read_bytes())
        doc["decisions"] = doc["decisions"][1:]
        fake2 = Path(self.roots.stage) / "DRAFT-UNCONFIRMED-trimmed.json"
        fake2.write_bytes(common.canonical(doc))
        self.stop(dryrun_args(fake2, sha(fake2.read_bytes())), "EXPANSION")

    def test_stage_must_sit_under_tmpdir(self):
        draft, dsha = self.draft()
        with mock.patch.dict("os.environ", {"TMPDIR": str(self.tmp / "elsewhere")}):
            self.stop(dryrun_args(draft, dsha), "R10")

    def test_pins_outcome_and_attempts(self):
        draft, dsha = self.draft()
        self.stop(dryrun_args(draft, "f" * 64), "R7")
        for flag in ("--owner-decisions-sha256", "--epoch-sha256", "--master-sha256"):
            with self.subTest(pin=flag):
                argv = dryrun_args(draft, dsha)
                argv[argv.index(flag) + 1] = "e" * 64
                self.stop(argv, "R7")
        argv = dryrun_args(draft, dsha)
        argv[argv.index("--registry-db") + 1] = str(self.tmp / "missing.sqlite3")
        self.stop(argv, "R7")
        self.stop(dryrun_args(draft, dsha, outcome="families=797"), "R11")
        self.stop(dryrun_args(draft, dsha, "--attempt", "0"), "R10")
        T.dryrun_tree(Path(self.roots.stage).resolve(), "draft", dsha, 1).mkdir(parents=True)
        self.stop(dryrun_args(draft, dsha), "R10", contains="already ran")


class FakeEntryPoints:
    """Stand-ins for families._main, families.verify and holdout._main (see the module docstring). fail names one step
    that STOPs; stray names one step that also tries a (failing) write outside TMPDIR."""

    def __init__(self, *, fail=None, stray=None):
        self.fail, self.stray = fail, stray
        self.seen, self.factories_refused, self.decisions, self.verify_kw, self.dry_flags = [], [], [], [], []

    @staticmethod
    def opts(argv) -> dict:
        out, i = {}, 1
        while i < len(argv):
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                out[argv[i]] = argv[i + 1]
                i += 2
            else:
                out[argv[i]] = True
                i += 1
        return out

    @staticmethod
    def s3io():
        return sys.modules["dltrain.s3io"]                  # loaded by tests_p3.fixtures_p3 inside cmd_dryrun

    def puts(self, clients, n: int, tag: str) -> list:
        s3io = self.s3io()
        refs = []
        for i in range(n):
            body = common.canonical({"fake": tag, "i": i})
            key = f"{s3io.PREFIX_KEY}objects/dl-d5-fake/sha256/{sha(body)}.json"
            refs.append(s3io.put_create_only(clients["writer"], key, body, clients["reader"]))
        return refs

    def probe_factories(self):
        from dltrain import holdout
        for name, fn in (("families", families._default_publish_clients), ("holdout", holdout._default_clients)):
            try:
                fn()
            except AssertionError:
                self.factories_refused.append(name)

    def maybe_stray(self, step):
        if self.stray == step:
            try:
                open(STRAY, "w").close()
            except OSError:
                pass

    def read_decisions(self, path, want_sha):
        assert path.startswith("/proc/self/fd/"), path                 # a memfd, never a file
        raw = Path(path).read_bytes()
        assert sha(raw) == want_sha
        self.decisions.append(raw)

    def families_main(self, argv, *, roots=None, clients=None, guard=None):
        cmd, o = argv[0], self.opts(argv)
        step = f"families.{cmd}"
        self.seen.append((step, list(argv)))
        fam_root, rec_root = (Path(r).resolve() for r in roots)
        assert Path(o["--out"]).resolve() == fam_root and Path(o["--receipt"]).resolve().parent == rec_root
        self.maybe_stray(step)
        if self.fail == step:
            print(json.dumps({"verdict": "STOP", "stop": "fake STOP"}), file=sys.stderr)
            return 2
        out = Path(o["--out"])
        if cmd == "build":
            self.read_decisions(o["--decisions"], o["--decisions-sha256"])
            reg = common.canonical({"fake": "registry", "phase": "final", "decisions": o["--decisions-sha256"]})
            common.immutable_write(out / families.OUT_FINAL, reg)
            summary = {"verdict": "PASS", "phase": "final", "counts": {"families": 797, "N_pool": 640},
                       "outputs": {families.OUT_PRE: {"sha256": PRE_PIN}, families.OUT_CSV: {"sha256": CSV_PIN},
                                   families.OUT_FINAL: {"sha256": sha(reg), "bytes": len(reg)}}}
            common.write_receipt(o["--receipt"], dict(summary, schema="dl-d5-fake-build"))
            print(json.dumps(summary, sort_keys=True))
            return 0
        assert cmd == "publish"
        assert (out / "alias-sources.json").read_bytes() == ALIAS.read_bytes()   # the tool's byte-identical copy
        assert Path(o["--verify-receipt"]).is_file()
        self.probe_factories()
        refs = self.puts(clients, 7, "families.publish")
        receipt = {"schema": "jope.dl-packet-receipt/v1", "packet": "P2", "verdict": "PASS", "refs": refs}
        common.immutable_write(o["--receipt"], common.receipt_bytes(receipt))
        print(json.dumps({"verdict": "PASS", "phase": "final"}, sort_keys=True))
        return 0

    def families_verify(self, config, out_dir, decisions_path=None, *, guard=None, expect=None, full_rehash=True,
                        receipt_path=None, decisions_sha256=None, receipts_dir=None, **kw):
        self.seen.append(("families.verify", [str(config), str(out_dir)]))
        self.verify_kw.append({"full_rehash": full_rehash, "receipts_dir": str(receipts_dir), "expect": expect})
        self.read_decisions(decisions_path, decisions_sha256)
        self.maybe_stray("families.verify")
        verdict = "FAIL" if self.fail == "families.verify" else "PASS"
        report = {"schema": families.VERIFY_RECEIPT_SCHEMA, "verdict": verdict, "phase": "final",
                  "failed": ["canary_scan_clean"] if verdict == "FAIL" else [],
                  "canary": {"scope": {"outputs": 7, "receipts": 3}, "hit_files": int(verdict == "FAIL"),
                             "undecodable": []}}
        common.write_receipt(receipt_path, report)
        return report

    def holdout_main(self, argv, clients=None, *, draft_dry_run=False):
        cmd, o = argv[0], self.opts(argv)
        step = f"holdout.{cmd}"
        self.seen.append((step, list(argv)))
        self.dry_flags.append(draft_dry_run)
        self.maybe_stray(step)
        status = "STOP" if self.fail == step else "PASS"
        rdir = Path(o["--out"] if cmd in ("commit", "freeze", "closure") else Path(o["--holdout"]).parent) / "receipts"
        doc = {"schema": "jope.dl-holdout-receipt/v1", "command": cmd, "status": status, "draft_dry_run": draft_dry_run}
        if status == "PASS" and cmd == "commit":
            assert sha(Path(o["--registry"]).read_bytes()) == o["--registry-sha256"]
            assert sha(Path(o["--p2-receipt"]).read_bytes()) == o["--p2-receipt-sha256"]
            assert o["--owner-decisions-sha256"] == OWNER_DECISIONS[1]
            self.read_decisions(o["--decisions"], json.loads(Path(o["--registry"]).read_bytes())["decisions"])
            self.puts(clients, 1, "commitment")
            common.immutable_write(Path(o["--out"]) / "seed.hex", b"0" * 64 + b"\n")
            doc.update(pool_N=640, k_expected=32, s3_outcome="created", seed_sha256=sha(b"fake seed"))
        elif status == "PASS" and cmd == "freeze":
            assert (Path(o["--seed-file"])).is_file()
            w = common.immutable_write(Path(o["--out"]) / "holdout.json", common.canonical({"fake": "holdout"}))
            doc.update(N=640, k=32, dev=31, diag=10, train_fit=600, holdout_sha256=w["sha256"],
                       expect={"complete": False})
        elif status == "PASS" and cmd == "verify":
            ran = "--assignment-poison" in o
            doc.update(N=640, k=32, bytes_equal=True, failed_checks=[],
                       assignment_poison={"ran": ran, "pass": ran, "variants": ["v1", "v2"]})
        elif status == "PASS" and cmd == "closure":
            w = common.immutable_write(Path(o["--out"]) / "closure.json", common.canonical({"fake": "closure"}))
            doc.update(closure=w, counts={"cross_family_links": 0})
        elif status == "PASS":
            assert Path(o["--verify-receipt"]).is_file() and Path(o["--closure"]).is_file()
            m, c, x, b = self.puts(clients, 4, "holdout.publish")
            doc.update(manifest=m, campaign_manifest=c, closure_object=x, binding=dict(b, outcome="created"))
        rec = common.write_receipt(rdir / f"{cmd}-{common.utc_us()}.json", doc)
        print(json.dumps({"command": cmd, "status": status, "receipt": {k: rec[k] for k in ("path", "sha256", "bytes")}}))
        return 0 if status == "PASS" else 2


class TestDryrunPipeline(D5Case):
    def run_fake(self, fake, *extra, made=None):
        from dltrain import holdout
        draft, dsha = made or self.draft()
        with mock.patch.object(families, "_main", fake.families_main), \
                mock.patch.object(families, "verify", fake.families_verify), \
                mock.patch.object(holdout, "_main", fake.holdout_main):
            code, out, err = self.cli(dryrun_args(draft, dsha, *extra))
        return code, json.loads(out) if out.strip() else None, err, (draft, dsha)

    def receipt(self, s) -> dict:
        p = Path(s["receipt"]["path"])
        self.assertEqual(p.parent, Path(self.roots.stage_receipts).resolve())
        self.assertTrue(p.name.startswith("DRAFT-UNCONFIRMED-p3-dryrun-"), p.name)
        self.assertEqual(sha(p.read_bytes()), s["receipt"]["sha256"])
        return json.loads(p.read_text())

    def stage_files(self):
        return [f for f in self.files() if f.startswith("stage/")]

    def test_pipeline_order_pins_puts_receipt_and_removal(self):
        fake = FakeEntryPoints()
        code, s, err, (draft, dsha) = self.run_fake(fake)
        self.assertEqual(code, 0, err)
        self.assertEqual([st for st, _ in fake.seen], STEPS)
        self.assertEqual((s["verdict"], s["stop"], s["fake_puts"], s["write_audit_violations"], s["tree_removed"]),
                         ("PASS", None, 12, 0, True))
        self.assertEqual([x["step"] for x in s["steps"]], STEPS)
        self.assertEqual(fake.factories_refused, ["families", "holdout"])
        self.assertEqual(fake.dry_flags, [True] * 5)                       # holdout's DRAFT dry-run mode, every step
        self.assertEqual(fake.verify_kw, [{"full_rehash": True, "receipts_dir": str(Path(self.roots.receipts).resolve()),
                                           "expect": families.EXPECT}])
        # the decision bytes every step read are the in-memory rehearsal variant, never the DRAFT file itself
        self.assertEqual(len(fake.decisions), 3)
        reh = json.loads(fake.decisions[0])
        self.assertTrue(all(d == fake.decisions[0] for d in fake.decisions))
        self.assertEqual((reh["schema"], reh["rehearsal_of"]["sha256"]), (families.DECISIONS_SCHEMA, dsha))
        self.assertTrue(families.provenance_marks_draft(reh["provenance"]))
        rec = self.receipt(s)
        self.assertEqual((rec["schema"], rec["verdict"], rec["attempt"], rec["mode"]), (T.DRYRUN_SCHEMA, "PASS", 1, "draft"))
        self.assertIn("removed after this inventory", rec["notice"])
        self.assertEqual(rec["rehearsal"], {"sha256": sha(fake.decisions[0]), "bytes": len(fake.decisions[0]),
                                            "via": "memfd (never a file)"})
        self.assertEqual([x["fake_puts"] for x in rec["steps"]], [0, 0, 7, 1, 0, 0, 0, 4])
        self.assertEqual(rec["write_audit"]["violations"], [])
        self.assertEqual(len(rec["fake_s3"]["puts"]), 12)
        self.assertTrue(set(rec["code"]) >= {"families", "family_keys", "family_pin", "family_screen", "tool", "holdout",
                                             "holdout_source_tree", "s3io", "common", "fixtures_p3"})
        # D-MAP-6: the tree is inventoried, then removed; nothing of the rehearsal stays in the stage
        tree = Path(rec["tree"]["path"])
        self.assertTrue(tree.name.startswith("DRAFT-UNCONFIRMED-rehearsal-") and tree.name.endswith("-a1"))
        self.assertEqual((rec["tree"]["removed"], tree.exists()), (True, False))
        inv = {f["path"]: f for f in rec["tree"]["files"]}
        self.assertTrue({"families/registry.json", "families/alias-sources.json", "receipts/P2/receipt.json",
                         "holdout/seed.hex", "holdout/holdout.json", "holdout/closure.json"} <= set(inv))
        self.assertEqual(inv["families/alias-sources.json"]["sha256"], ALIAS_PIN)
        self.assertEqual(self.stage_files(), [f"stage/{draft.name}"])
        self.assertNotIn(sha(fake.decisions[0]), {sha((self.tmp / f).read_bytes()) for f in self.files()})
        verify_rec = next(x for x in rec["steps"] if x["step"] == "holdout.verify")["receipt"]["path"]
        publish_argv = next(a for st, a in fake.seen if st == "holdout.publish")
        self.assertEqual(publish_argv[publish_argv.index("--verify-receipt") + 1], verify_rec)

    def test_quick_rehash_and_a_whatif_file(self):
        fake = FakeEntryPoints()
        made = self.whatif(rights="undecided")
        code, s, err, _ = self.run_fake(fake, "--quick-rehash", made=made)
        self.assertEqual(code, 0, err)
        self.assertEqual(fake.verify_kw[0]["full_rehash"], False)
        p = Path(s["receipt"]["path"])
        self.assertTrue(p.name.startswith("WHATIF-NOT-A-DECISION-p3-dryrun-"))

    def test_a_failed_step_stops_with_a_receipt_removes_the_tree_and_uses_the_attempt(self):
        fake = FakeEntryPoints(fail="holdout.verify")
        code, s, err, made = self.run_fake(fake)
        self.assertEqual(code, 2, err)
        self.assertEqual((s["verdict"], s["tree_removed"]), ("STOP", True))
        self.assertIn("holdout.verify did not PASS", s["stop"])
        self.assertEqual([st for st, _ in fake.seen], STEPS[:6])
        rec = self.receipt(s)
        self.assertEqual((rec["verdict"], rec["steps"][-1]["step"], rec["steps"][-1]["exit"]), ("STOP", "holdout.verify", 2))
        self.assertTrue(rec["tree"]["files"])
        self.assertEqual(self.stage_files(), [f"stage/{made[0].name}"])
        self.stop(dryrun_args(*made), "R10", contains="already ran")       # the attempt's receipt exists
        code, s2, err, _ = self.run_fake(FakeEntryPoints(), "--attempt", "2", made=made)
        self.assertEqual(code, 0, err)
        self.assertTrue(self.receipt(s2)["tree"]["path"].endswith("-a2"))

    def test_a_failed_verify_stops_before_publish_and_holdout(self):
        fake = FakeEntryPoints(fail="families.verify")
        code, s, err, _ = self.run_fake(fake)
        self.assertEqual(code, 2, err)
        self.assertIn("families.verify did not PASS", s["stop"])
        self.assertIn("the real S3 verify would FAIL too", s["stop"])
        self.assertEqual([st for st, _ in fake.seen], STEPS[:2])
        self.assertEqual(self.receipt(s)["fake_s3"]["puts"], [])

    def test_a_write_outside_tmpdir_fails_the_run(self):
        fake = FakeEntryPoints(stray="families.publish")
        code, s, err, _ = self.run_fake(fake)
        forget_violation(STRAY)                       # this test's own audit saw the same (failing) open
        self.assertFalse(Path(STRAY).exists())
        self.assertEqual(code, 2, err)
        self.assertEqual((s["verdict"], s["write_audit_violations"]), ("STOP", 1))
        rec = self.receipt(s)
        self.assertEqual(rec["write_audit"]["violations"], [["open", STRAY]])
        self.assertEqual(rec["stop"], "the write audit saw writes outside TMPDIR")


if __name__ == "__main__":
    unittest.main()
