"""Shared fixtures for the tools/p2_decisions.py tests (version 2, P2 generation 2).

- The tool is loaded by path (tools/ has no __init__.py) and registered as sys.modules['p2_decisions'].
- PINS: the bound g2 inputs (RUN/families/g2), read-only. Pin drift fails loudly.
- R7 code binding: registry.pre-review.json e21690dd records the sha256 of families, family_keys, family_pin and
  family_screen ("code"). Tests that run the real families.run_build are skipped with the drifted module names unless
  the current code still matches (CODE_MATCHES); TestCodeBinding proves without any patch that the tool refuses R7 under
  drift. On the frozen g2 code nothing is skipped.
- A process audit hook records every write outside RUN/tmp/dl-d5-g2 (the Track D g2 dir) while a test is armed; D5Case
  arms it around each test and fails the test on a violation. Each test gets its own TemporaryDirectory there and its
  own injected Roots, so the tool can never reach RUN/families, RUN/receipts or RUN/holdout.
- caching_build wraps families.run_build: the pre-review build (no decisions) is computed once per process and shared by
  every check test; final builds keep their registry.json bytes, keyed by the decisions sha256.
- stub_build stands in for families.run_build: a final build with a given summary (expand --simulate), or the pre-review
  outputs plus a model made from the bound review.csv and families table (check).
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from dltrain import common, families

SRC = Path(__file__).resolve().parents[1]
RUN = Path("/home/ubuntu/scratch/jopedime-dl-training-v1")
TMP_ROOT = os.environ.get("TMPDIR") or ""
if not TMP_ROOT or not os.path.isdir(TMP_ROOT) or os.path.realpath(TMP_ROOT).startswith("/tmp"):
    raise RuntimeError("TMPDIR must name RUN/tmp (never /tmp); run the tests inside the lane slice")
TRACK = Path(os.path.realpath(TMP_ROOT)) / "dl-d5-g2"
TRACK.mkdir(exist_ok=True)
ALLOWED = str(TRACK)


def load_tool():
    name = "p2_decisions"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SRC / "tools" / "p2_decisions.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


T = load_tool()

FAM = RUN / "families" / "g2"
CSV_PIN = "aad99b1a9af6056a4cd812a962871990ae35fce24fb8b699cbd9df116d188fcb"
PRE_PIN = "e21690dd73e3ad11ad48c5ce9151dfe8d2f036de44985306bb90e5f7b55ecb5a"
ALIAS_PIN = "f818efaf847a043a26c5a47c858e37f2b499ccc056f33afc166a6856b07671cd"
SHAPE_PIN = "6790b1020be2a21896ffa275a5ffa811062600654ee180b999adc91d0808bc17"
G1_CSV_PIN = "c7377cf8bc867b4e168edfca3e5d45b823b3ee5b72b3b06568a7ff3938d677d6"       # superseded generation 1
CSV, PRE, ALIAS = FAM / "review.csv", FAM / "registry.pre-review.json", FAM / "alias-sources.json"
OWNER_DECISIONS = (Path("/home/ubuntu/scratch/jopedime-dltrain-audit-v1/claude/receipts/"
                        "owner-decisions-20261005T044116Z.json"),
                   "77a115c78e927000e77f052c86c2fde5cc66c149d4a16814172b91426aa3a602")
EPOCH = (RUN / "epochs" / "epoch-000.json", "bcfb5230bc0f5f32f6b42edbfa3a6bf76b89e0593b593b9042290e8c077a87ba")
MASTER = (Path("/home/ubuntu/scratch/jopedime-regression-research-2k-per-fold-hwm-20261001/gp/input-manifest-master.json"),
          "7ab3eb70a0cab3950151b113536d494e19e8b006cc8a4808351dfe39cfc6bf44")
REGISTRY_DB = Path("/home/ubuntu/scratch/jopedime-regression-research-2k-incoming-intake-v1/intake.sqlite3")
BREAKER_ID = "rg-e6fb3d57ef2782074eeb468d"
A4 = {"Q1": "a", "Q2": "a", "Q3": "a", "Q4": "a"}
# measured on g2 at families 7dcd65ff (in-memory staging final builds, 2026-10-05T14:32Z, tmp/dl-d5-g2/measure_g2.log)
RECOMMENDED = {"families": 797, "N_pool": 652, "k": 33, "pool_contexts": 3862, "auto": 173, "reviewed": 480,
               "unresolved": 144, "sealed": 28, "sealed_hard": 8, "sealed_provisional": 20, "seed_exposed": 10}
LITERAL = dict(RECOMMENDED, N_pool=640, k=32, pool_contexts=3801, reviewed=468, unresolved=156)


def outcome_arg(o: dict) -> str:
    return ",".join(f"{k}={o[k]}" for k in T.OUTCOME_KEYS)


REC_OUTCOME, LITERAL_OUTCOME = outcome_arg(RECOMMENDED), outcome_arg(LITERAL)


def counts_of(o: dict) -> dict:
    return {"families": o["families"], "N_pool": o["N_pool"], "pool_contexts": o["pool_contexts"],
            "independence_qualified": {"auto": o["auto"], "reviewed": o["reviewed"], "unresolved": o["unresolved"]},
            "sealed": {"families": o["sealed"], "hard_families": o["sealed_hard"],
                       "provisional_families": o["sealed_provisional"]},
            "seed_exposed_families": o["seed_exposed"]}


REC_COUNTS, LITERAL_COUNTS = counts_of(RECOMMENDED), counts_of(LITERAL)
REC_DECISIONS = {"confirm_mapping": 10, "confirm_union": 88, "independent": 94, "keep_sealed": 37, "union_subset": 72,
                 "unknown": 14, "unresolved": 98}
LITERAL_DECISIONS = {k: v for k, v in REC_DECISIONS.items() if k != "unknown"}
# the line the g2 question set asks Jepson to type himself (dl-d5-confirm/v1)
CONFIRM_TEXT = ("DL-D5 confirmed for review.csv aad99b1a9af6: Q1=a, Q2=a, Q3=a, Q4=a, Q5=a; "
                "split-registry excluded seals nothing.")
# the relayed paste's answer lines (10:07:27Z entry): lettered (a)..(d), no Qn=x, no split-registry, no rights
RELAY_STYLE_TEXT = ("OWNER DECISION (DL-D5), from Jepson via owner loop. Jepson chose: accept your recommended answer on "
                    "all four questions.\n(a) Accept the 285 link-group recommendations AND confirm the 10 seed-source "
                    "mappings (692 holdout-eligible families, k=35).\n(b) Keep all 23 provisionally sealed "
                    "lineage-B-adjacent families sealed.\n(c) Rdatasets families at ITEM level.\n(d) Leave the leftover "
                    "rows unresolved; do not union the breaker cluster.\n")
SESSION = "screen 442374.JopeDime_DLTrain pid 457442 starttick 199149167 cwd /home/ubuntu/JopeDime"
POSTED_UTC, MESSAGE_UTC = "2026-10-05T15:00:00Z", "2026-10-05T15:30:00Z"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ------------------------------------------------------------------------------------------------ R7 code binding
def _pinned_code() -> dict:
    raw = PRE.read_bytes()
    if sha(raw) != PRE_PIN:
        raise RuntimeError("registry.pre-review.json differs from its pin e21690dd...: the tests bind that exact file")
    return json.loads(raw)["code"]


PINNED_CODE = _pinned_code()
CURRENT_CODE = families._code_shas()
CODE_DRIFT = sorted(k for k in set(PINNED_CODE) | set(CURRENT_CODE) if PINNED_CODE.get(k) != CURRENT_CODE.get(k))
CODE_MATCHES = not CODE_DRIFT
DRIFT_REASON = (f"code drift: dltrain {CODE_DRIFT} differ from the 'code' block of registry.pre-review.json e21690dd "
                "(edited after the g2 pre-review build), so a real families.run_build cannot rebuild the bound bytes; the "
                "tool refuses R7 (proved by TestCodeBinding)")


def pinned_code():
    """Patch the tool's R7 code reading to the pins of the bound set (only for tests that never run a real build)."""
    return mock.patch.object(T, "current_code_shas", lambda: dict(PINNED_CODE))


def bound_args(csv_path=CSV, csv_pin=CSV_PIN, pre_path=PRE, pre_pin=PRE_PIN, alias_path=ALIAS, alias_pin=ALIAS_PIN):
    return ["--review-csv", str(csv_path), "--review-csv-sha256", csv_pin, "--pre-review", str(pre_path),
            "--pre-review-sha256", pre_pin, "--alias-sources", str(alias_path), "--alias-sources-sha256", alias_pin]


def draft_args(*extra, answers="Q1=a,Q2=a,Q3=a,Q4=a", rights="undecided", **kw):
    return ["expand", *bound_args(**kw), "--answers", answers, "--rights", rights, "--expect-shape", "builtin:aad99b1a",
            "--draft", "--relay-sha256", T.RELAY_SHA256, *extra]


def whatif_args(*extra, answers="Q1=a,Q2=a,Q3=a,Q4=a", rights="unknown", **kw):
    return ["expand", *bound_args(**kw), "--answers", answers, "--rights", rights, "--expect-shape", "builtin:aad99b1a",
            "--what-if", *extra]


# ------------------------------------------------------------------------------------------------ write audit
_STATE = {"armed": False, "busy": False, "violations": []}
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
_DIR_FD_ARG = {"os.mkdir": 2, "os.chmod": 2, "os.remove": 1, "os.rmdir": 1, "os.utime": 3, "os.chown": 3}


def _under(path: str) -> bool:
    return path == ALLOWED or path.startswith(ALLOWED + os.sep)


def _resolve(path, dir_fd=-1):
    if isinstance(path, int):
        return None
    path = os.fsdecode(path) if isinstance(path, bytes) else os.fspath(path)
    if not os.path.isabs(path):
        base = os.readlink(f"/proc/self/fd/{dir_fd}") if isinstance(dir_fd, int) and dir_fd >= 0 else os.getcwd()
        path = os.path.join(base, path)
    return os.path.realpath(path)


def _target(event, args):
    try:
        if event == "open":
            path, mode, flags = args
            writes = (isinstance(mode, str) and any(ch in mode for ch in "wax+")) or \
                     (isinstance(flags, int) and flags >= 0 and bool(flags & _WRITE_FLAGS))
            return _resolve(path) if writes else None
        if event in _DIR_FD_ARG:
            i = _DIR_FD_ARG[event]
            return _resolve(args[0], args[i] if len(args) > i else -1)
        if event == "os.rename":
            src, dst = _resolve(args[0], args[2]), _resolve(args[1], args[3])
            return next((p for p in (src, dst) if p is not None and not _under(p)), dst)
        if event in ("os.link", "os.symlink"):
            return _resolve(args[1])
        if event in ("os.truncate", "shutil.copyfile"):
            return _resolve(args[-1] if event == "shutil.copyfile" else args[0])
        if event == "sqlite3.connect":
            db = os.fsdecode(args[0]) if isinstance(args[0], bytes) else str(args[0])
            if db.startswith("file:"):
                body, _, query = db[len("file:"):].partition("?")
                return None if "mode=ro" in query.split("&") else _resolve(body)
            return None if db == ":memory:" else _resolve(db)
    except Exception:  # noqa: BLE001 (an unresolvable target is itself a violation)
        return f"<unresolved {event}>"
    return None


def _hook(event, args):
    if not _STATE["armed"] or _STATE["busy"]:
        return
    _STATE["busy"] = True
    try:
        t = _target(event, args)
        if t is not None and not _under(t):
            _STATE["violations"].append((event, t))
    finally:
        _STATE["busy"] = False


sys.addaudithook(_hook)


def arm():
    _STATE["violations"] = []
    _STATE["armed"] = True


def disarm():
    _STATE["armed"] = False
    return list(_STATE["violations"])


def forget_violation(path: str) -> None:
    """Drop the audit entries for one deliberate, failing open outside the allowed root (a probe that writes nothing)."""
    _STATE["violations"] = [v for v in _STATE["violations"] if v[1] != path]


# ------------------------------------------------------------------------------------------------ builds
SHARED = {"pre": None, "final": {}, "calls": []}


def caching_build(config, decisions_path=None, *, guard=None, decisions_sha256=None, **kw):
    """families.run_build, with the pre-review result shared by every check test (it is the same pure function of the
    same pinned inputs: keyed by the alias-sources bytes, so a bound copy whose inputs link to g2 shares it) and
    registry.json kept from final builds."""
    SHARED["calls"].append(("pre", None) if decisions_path is None else ("final", kw.get("real", False)))
    if decisions_path is None:
        key = sha(Path(config).read_bytes())
        if SHARED["pre"] is None or SHARED["pre"][0] != key:
            SHARED["pre"] = (key, families.run_build(config, None, guard=guard, **kw))
        return SHARED["pre"][1]
    res = families.run_build(config, decisions_path, guard=guard, decisions_sha256=decisions_sha256, **kw)
    SHARED["final"][decisions_sha256] = res.outputs.get(families.OUT_FINAL)
    return res


_MODELS = {}


def pre_model(pre_raw: bytes, csv_raw: bytes) -> SimpleNamespace:
    """A stand-in for run_build(...).pre made from the bound files: the review rows (family_ids as lists) and the
    pre-review families table. families.validate_decisions reads nothing else (add_members is never generated)."""
    key = (sha(pre_raw), sha(csv_raw))
    if key not in _MODELS:
        rows = [{"review_group_id": r["review_group_id"], "kind": r["kind"], "family_ids": T.family_ids(r),
                 "recommendation": r["recommendation"]} for r in T.parse_rows(csv_raw)]
        _MODELS[key] = SimpleNamespace(rows=rows, families=json.loads(pre_raw)["families"], phase="pre_review")
    return _MODELS[key]


class StubResult:
    """A run_build stand-in: summary, outputs and (for check) the pre-review model."""

    def __init__(self, verdict="PASS", phase="final", counts=None, pre=None, csv_b=None, final=b"{}", model=None,
                 deltas=None):
        self.summary = {"verdict": verdict, "phase": phase, "counts": counts or {}, "decision_deltas": deltas}
        self.outputs = {families.OUT_PRE: pre, families.OUT_CSV: csv_b}
        if final is not None:
            self.outputs[families.OUT_FINAL] = final
        self.pre = model


def stub_build(*, pre_review=None, calls=None, seen=None, **kw):
    """With a decisions path (expand --simulate): a final build whose summary is given by kw; the memfd must hold
    exactly the pinned bytes. Without one (check): the pre-review outputs `pre_review` = (pre bytes, csv bytes), by
    default the bound files, and the stand-in model of the bound files. seen collects (decisions bytes, kwargs)."""
    def fn(config, decisions_path=None, *, guard=None, decisions_sha256=None, **extra):
        if guard is not None:
            guard.wait("stub-build", force=True)
        if calls is not None:
            calls.append("pre" if decisions_path is None else "final")
        if decisions_path is None:
            bound = (PRE.read_bytes(), CSV.read_bytes())
            pb, cb = pre_review if pre_review is not None else bound
            return StubResult(verdict="STOP", phase="pre_review", pre=pb, csv_b=cb, final=None, model=pre_model(*bound))
        raw = Path(decisions_path).read_bytes()            # the memfd holds exactly the bytes
        assert decisions_sha256 is None or sha(raw) == decisions_sha256
        if seen is not None:
            seen.append((raw, dict(extra)))
        return StubResult(**kw)
    return fn


def sim_ok(counts=None, **kw):
    """A stub build that reproduces the recommended g2 outcome and both bound pre-review outputs."""
    return stub_build(counts=counts or REC_COUNTS, pre=PRE.read_bytes(), csv_b=CSV.read_bytes(), **kw)


class NoGuard:
    def __init__(self):
        self.checks = 0

    def wait(self, stage, force=False):
        self.checks += 1

    def summary(self):
        return {"guard": "none (offline test)", "checks": self.checks, "pauses": 0}


def synthetic_final_registry(decisions_sha256: str, *, pre_sha=PRE_PIN, csv_sha=CSV_PIN, alias_sha=ALIAS_PIN,
                             provenance=None) -> bytes:
    """A small phase-final registry (tests_p3 fixture families, NOT a families build) that binds a decision file, the
    bound review.csv, pre-review registry and alias-sources: enough for holdout.load_registry and review_gate.
    provenance: the decision_deltas.provenance block a families build would embed (None: none)."""
    from tests_p3 import fixtures_p3 as fx3       # brings boto3/s3io into the test process only
    doc = fx3.registry_doc(fx3.standard_families(), alias_sha=alias_sha, csv_sha=csv_sha, decisions_sha=decisions_sha256)
    doc["registry_pre_review_sha256"] = pre_sha
    if provenance is not None:
        doc["decision_deltas"] = {"provenance": provenance}
    return common.canonical(doc)


# ------------------------------------------------------------------------------------------------ test case
class D5Case(unittest.TestCase):
    """A TemporaryDirectory under the Track D g2 dir, injected Roots there, the write audit armed and (PIN_CODE) the R7
    code reading patched to the bound set's pins. The owner-record locations are moved under the test dir for the tool
    (OWNER_DIRECTIVES_PATH) and for holdout (P2_RECEIPTS_DIR = the injected receipts root, HOLDOUT_DIR,
    OWNER_DIRECTIVES_PATH), and holdout's real client factory raises."""
    PIN_CODE = True

    def setUp(self):
        from dltrain import holdout
        if self.PIN_CODE:
            patch = pinned_code()
            patch.start()
            self.addCleanup(patch.stop)
        self._tmp = tempfile.TemporaryDirectory(dir=TRACK, prefix="t-")
        self.tmp = Path(self._tmp.name)
        self.roots = T.Roots(families=self.tmp / "families", receipts=self.tmp / "receipts" / "P2",
                             stage=self.tmp / "stage", stage_receipts=self.tmp / "receipts" / "P2" / "staging")
        self.directives_path = self.tmp / "plans" / "OWNER_DIRECTIVES.md"
        for target, name, kw in ((T, "OWNER_DIRECTIVES_PATH", {"new": self.directives_path}),
                                 (holdout, "OWNER_DIRECTIVES_PATH", {"new": self.directives_path}),
                                 (holdout, "P2_RECEIPTS_DIR", {"new": Path(self.roots.receipts)}),
                                 (holdout, "HOLDOUT_DIR", {"new": self.tmp / "RUN-holdout"}),
                                 (holdout, "_default_clients",
                                  {"side_effect": AssertionError("a real S3 client was requested in an offline test")})):
            patch = mock.patch.object(target, name, **kw)
            patch.start()
            self.addCleanup(patch.stop)
        arm()

    def tearDown(self):
        violations = disarm()
        self._tmp.cleanup()
        self.assertEqual(violations, [], "the test wrote outside RUN/tmp/dl-d5-g2")

    def cli(self, argv, *, build_fn=None, guard_factory=NoGuard, roots=None):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = T._main([str(a) for a in argv], roots=roots or self.roots, build_fn=build_fn or stub_build(),
                           guard_factory=guard_factory)
        return code, out.getvalue(), err.getvalue()

    def ok(self, argv, **kw) -> dict:
        code, out, err = self.cli(argv, **kw)
        self.assertEqual(code, 0, err)
        return json.loads(out)

    def stop(self, argv, refusal, contains=None, *, writes=False, **kw):
        before = self.files()
        code, out, err = self.cli(argv, **kw)
        self.assertEqual(code, 2, f"expected STOP {refusal}; stdout={out[:300]} stderr={err[:600]}")
        line = json.loads(err.strip().splitlines()[-1])
        self.assertEqual((line["verdict"], line["refusal"]), ("STOP", refusal), line.get("reason"))
        if contains is not None:
            self.assertIn(contains, line["reason"])
        if not writes:
            self.assertEqual(self.files(), before, "a refusal wrote something")
        return line

    def files(self) -> list:
        return sorted(str(p.relative_to(self.tmp)) for p in self.tmp.rglob("*") if p.is_file())

    def bound_copy(self, name="g2", *, inputs=False) -> Path:
        """A bound families dir <roots.families>/<name> holding copies of review.csv, registry.pre-review.json and
        alias-sources.json; inputs=True adds an `inputs` symlink to RUN/families/g2/inputs (read-only) so a real build
        can run from the copy."""
        d = Path(self.roots.families) / name if name else Path(self.roots.families)
        d.mkdir(parents=True, exist_ok=True)
        for src in (CSV, PRE, ALIAS):
            if not (d / src.name).exists():
                (d / src.name).write_bytes(src.read_bytes())
        if inputs and not (d / "inputs").exists():
            (d / "inputs").symlink_to(FAM / "inputs", target_is_directory=True)
        return d

    def message(self, text: str, name="owner-message.txt") -> tuple[Path, str]:
        p = self.tmp / name
        p.write_bytes(text.encode("utf-8"))
        return p, sha(text.encode("utf-8"))

    def directives(self, *lines: str) -> tuple[Path, str]:
        """The owner's OWNER_DIRECTIVES.md at the patched OWNER_DIRECTIVES_PATH (tool and holdout) under the test dir."""
        d = self.directives_path
        d.parent.mkdir(parents=True, exist_ok=True)
        d.write_text("\n".join(lines) + "\n")
        return d, sha(d.read_bytes())

    def draft(self, *extra, build_fn=None, **kw) -> tuple[Path, str]:
        s = self.ok(draft_args(*extra, **kw), build_fn=build_fn)
        return Path(s["out"]["path"]), s["out"]["sha256"]

    def whatif(self, *extra, build_fn=None, **kw) -> tuple[Path, str]:
        s = self.ok(whatif_args(*extra, **kw), build_fn=build_fn)
        return Path(s["out"]["path"]), s["out"]["sha256"]


def real_args(bdir: Path, receipts: Path, msg: Path, msg_sha: str, *extra, answers="Q1=a,Q2=a,Q3=a,Q4=a",
              rights="unknown", utc=1791197000000000, kind="owner_verbatim_message", outcome=REC_OUTCOME,
              posted=POSTED_UTC, message_utc=MESSAGE_UTC):
    return ["expand", *bound_args(csv_path=bdir / "review.csv", pre_path=bdir / "registry.pre-review.json",
                                  alias_path=bdir / "alias-sources.json"),
            "--answers", answers, "--rights", rights, "--expect-shape", "builtin:aad99b1a", "--simulate",
            "--expect-outcome", outcome, "--split-registry-confirmed", "--out", str(bdir / "review-decisions.json"),
            "--receipt", str(receipts / f"decisions-expand-{utc}.json"), "--provenance-kind", kind,
            "--session", SESSION, "--message-utc", message_utc, "--message-file", str(msg), "--message-sha256", msg_sha,
            "--question-posted-utc", posted, *extra]


def drop_flags(argv, *flags) -> list:
    """argv without the given flags and their values."""
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        if a in flags:
            skip = True
            continue
        out.append(a)
    return out


def put_flag(argv, flag, value) -> list:
    a = list(argv)
    a[a.index(flag) + 1] = str(value)
    return a


def directive_args(bdir: Path, receipts: Path, directive: Path, dsha: str, *, version="v7", lines="3", **kw):
    """real_args with owner_directive provenance in place of the message arguments."""
    base = drop_flags(real_args(bdir, receipts, directive, dsha, kind="owner_directive", **kw),
                      "--session", "--message-utc", "--message-file", "--message-sha256", "--question-posted-utc")
    return base + ["--directive-file", str(directive), "--directive-version", version, "--directive-lines", lines,
                   "--directive-sha256", dsha]


def check_args(decisions, dsha, *extra, bdir=None):
    kw = {} if bdir is None else {"csv_path": bdir / "review.csv", "pre_path": bdir / "registry.pre-review.json",
                                  "alias_path": bdir / "alias-sources.json"}
    return ["check", "--decisions", str(decisions), "--decisions-sha256", dsha, *bound_args(**kw), *extra]


def dryrun_args(decisions, dsha, *extra, outcome="families=797,N_pool=640,k=32"):
    return ["dryrun", "--decisions", str(decisions), "--decisions-sha256", dsha, *bound_args(),
            "--owner-decisions", str(OWNER_DECISIONS[0]), "--owner-decisions-sha256", OWNER_DECISIONS[1],
            "--epoch", str(EPOCH[0]), "--epoch-sha256", EPOCH[1], "--master", str(MASTER[0]), "--master-sha256", MASTER[1],
            "--registry-db", str(REGISTRY_DB), "--expect-outcome", outcome, *extra]


# ------------------------------------------------------------------------------------------------ synthetic worlds
def fid(tag: str) -> str:
    return "fam-" + sha(tag.encode())


def row(gid_tag: str, kind: str, fids, rec: str, *, basis=None, options=None) -> dict:
    return {"review_group_id": "rg-" + sha(gid_tag.encode())[:24], "kind": kind, "trigger_kinds": kind,
            "trigger_value": "synthetic", "family_ids": ";".join(sorted(fids)), "members": "", "n_families": len(fids),
            "n_datasets": len(fids), "n_contexts": 1, "n_fold0": 0, "cohorts": "frozen", "evidence": "{}", "flags": "",
            "impact_if_union": "{}", "recommendation": rec,
            "recommendation_basis": basis if basis is not None else "agent-generated (deterministic rule, not a decision): x",
            "default_if_undecided": "unresolved",
            "decision_options": options if options is not None else ";".join(families.DECISION_OPTIONS.get(kind, ()))}


def subset_basis(sets) -> str:
    return ("agent-generated (deterministic rule, not a decision): pairwise nds/name/legacy evidence; sets="
            + common.canonical(sets).decode())


def synthetic_rows() -> list:
    a, b, c, d = (fid(x) for x in "abcd")
    return [row("seed", "seed_source_mapping", [a], "confirm_mapping"),
            row("pkg", "rdatasets_pkg", [a, b, c], "union_subset", basis=subset_basis([sorted([a, b])])),
            row("pub", "publisher", [c, d], "independent"),
            row("mm", "multi_member_family", [d], "confirm_union"),
            row("seal", "sealed_adjacent", [b], "unseal"),
            row("rights", "rights_note", [c], "unknown"),
            row("breaker", "circuit_breaker", [a, d], "unresolved"),
            row("cov", "coverage_values_only", [b], "unresolved")]


def csv_bytes(rows) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=families.REVIEW_COLUMNS, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue().encode()


def write_world(d: Path, rows, *, phase="pre_review", code=None, csv_override=None, alias=b'{"synthetic": true}\n',
                bind_csv=None, bind_alias=None) -> dict:
    """review.csv, registry.pre-review.json (only the fields the tool reads), alias-sources.json and the shape file.
    The pre-review 'code' defaults to what the tool reads now (T.current_code_shas, patched under PIN_CODE)."""
    d.mkdir(parents=True, exist_ok=True)
    cb = csv_override if csv_override is not None else csv_bytes(rows)
    (d / "review.csv").write_bytes(cb)
    (d / "alias-sources.json").write_bytes(alias)
    fams = sorted({f for r in rows for f in r["family_ids"].split(";") if f})
    pre = {"schema": families.REGISTRY_SCHEMA, "phase": phase, "review_csv_sha256": bind_csv or sha(cb),
           "alias_sources_sha256": bind_alias or sha(alias), "code": code if code is not None else T.current_code_shas(),
           "families": {f: {} for f in fams}}
    pb = common.canonical(pre)
    (d / "registry.pre-review.json").write_bytes(pb)
    out = {"csv": (d / "review.csv", sha(cb)), "pre": (d / "registry.pre-review.json", sha(pb)),
           "alias": (d / "alias-sources.json", sha(alias))}
    try:
        shape = common.canonical(T.compute_shape(T.parse_rows(cb), sha(cb)))
        (d / "shape.json").write_bytes(shape)
        out["shape"] = (d / "shape.json", sha(shape))
    except T.Refused:
        out["shape"] = None
    return out


def world_args(w: dict) -> list:
    return bound_args(csv_path=w["csv"][0], csv_pin=w["csv"][1], pre_path=w["pre"][0], pre_pin=w["pre"][1],
                      alias_path=w["alias"][0], alias_pin=w["alias"][1])


def world_whatif_args(w: dict, *extra, answers="Q1=a,Q2=a,Q3=a,Q4=a", rights="unknown", shape=None):
    s = shape or w["shape"]
    return ["expand", *world_args(w), "--answers", answers, "--rights", rights, "--expect-shape", str(s[0]),
            "--expect-shape-sha256", s[1], "--what-if", *extra]
