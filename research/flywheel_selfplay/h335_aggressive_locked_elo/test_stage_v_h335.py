"""H335 Stage V driver tests (pytest). SYNTHETIC fixtures and TRAIN data only: no DL-Val byte is ever opened.

Every test redirects $H317_DATA (common_h317.DATA_ROOT), the LOCK receipt and the Stage V root to its own scratch dir
under /home/ubuntu/jope-data/h335-cl/stagev-test/<UTC stamp>/<id>/ (H335_STAGEV_TEST_ROOT overrides; never /tmp; nothing
is removed) and restores common_h317's Stage V state afterwards, so no production LOCK / marker / receipt is touched.
"Target" contexts are synthetic ids or NON-panel TRAIN contexts (neither P205 nor P201b) relabelled; GP units use the
frozen H317 deterministic fake worker (test_h317_gp.fake_runner); the LLM test runs the pinned frontier binary on 3 small
non-panel TRAIN contexts. Run from the H335 dir:
  ./sandbox_h335.sh env H335_GROUP=BASE python3 -m pytest -q -p no:cacheprovider test_stage_v_h335.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage_v_h335 as SVH  # noqa: E402  (FIRST: h335_core, then the frozen stage_v_h317 with the H335 root)

import argparse  # noqa: E402
import errno  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import test_h317_gp as T  # noqa: E402  (frozen H317 fixtures: fake_runner, HashCoder)

K, C, L, SV, LAB, FA = SVH.K, SVH.C, SVH.L, SVH.SV, SVH.LAB, SVH.FA
D, E, S = K.D, K.E, K.S
SCRATCH = (Path(os.environ.get("H335_STAGEV_TEST_ROOT", "/home/ubuntu/jope-data/h335-cl/stagev-test"))
           / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) / uuid.uuid4().hex[:8])
SEEDS = list(K.SEEDS_V)


# ----------------------------------------------------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------------------------------------------------
@pytest.fixture
def env(request, monkeypatch):
    """Isolated DATA_ROOT / receipt / Stage V root / guard log; Stage V state restored after the test."""
    root = SCRATCH / re.sub(r"[^A-Za-z0-9_.-]", "_", request.node.name)
    (root / "data" / "stage_v").mkdir(parents=True)
    monkeypatch.setattr(C, "DATA_ROOT", root / "data")
    monkeypatch.setattr(C, "LOCK_RECEIPT", root / "receipts" / "H335_LOCK_sha256.txt")
    monkeypatch.setattr(SV, "STAGEV_ROOT", root / "stagev")
    monkeypatch.setitem(C._STATE, "stage_v", False)
    monkeypatch.setenv("H317_GUARD_LOG", str(root / "guard.log"))
    return root


@pytest.fixture(scope="module")
def small_train():
    """Non-panel TRAIN contexts (neither P201b nor any H317 pilot shard), smallest tables first."""
    rows = C.train_rows()
    prot = set(K.panel_p201b(rows))
    folds = C.family_folds(rows)
    for n in (24, 48, 96, 205, C.PILOT_N):
        prot |= set(C.pilot_shard(rows, folds, n=n))
    return sorted((r["context_sha256"] for r in rows if r["context_sha256"] not in prot),
                  key=lambda c: (K.table_size(c), c))


def synth_lock(root: Path, gstar: int = 10, checkpoints=(10,)) -> dict:
    """A minimal H335 LOCK whose only code file is a scratch file (the full LOCK is tested in test_write_lock_*)."""
    f = root / "code" / "synthetic_source.py"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("# synthetic LOCK file\n")
    return {"schema": SVH.SCHEMA_LOCK, "stage_v_runs": True, "decision": {"hard_stop": False},
            "design": {"G_star": gstar, "stack": K.STACK, "seeds": SEEDS, "checkpoints": list(checkpoints),
                       "gens_agg": gstar, "gens_base": gstar + K.OLD_OFFSET, "stacks": list(LAB.STACKS), "max_expanded_width": L.RUN_MAX_EXPANDED_WIDTH,
                       "timeout_s": 60.0, "worker": str(SVH.WORKER), "worker_sha256": K.SUPERSET_SHA256,
                       "grammar_sha256": {"AGG": K.GRAMMAR_PORT_SHA256, "BASE": K.H317_PINS["gp_grammar_h317.py"]},
                       "neural_run_dir": str(SVH.NEURAL_RUN_DIR)},
            "entrants": SVH.entrant_keys(gstar, K.STACK),
            "llm": {"binary": str(SVH.FRONTIER), "binary_sha256": SVH.FRONTIER_SHA256, "entrants": list(FA.ENTRANTS),
                    "cpu_limit_s": 60, "wall_limit_s": 120},
            "prep_sha256": {}, "code_groups": {"code": [str(f)]}, "code_sha256": {str(f): C.sha256_file(f)}}


def write_lock(lock: dict, receipt_sha: str | None = None) -> str:
    sha = SV.create_once(SV.lock_path(), SV.json_bytes(lock))
    C.LOCK_RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    C.LOCK_RECEIPT.write_text(f"{receipt_sha or sha}  LOCK.json\n")
    return sha


def unlocked(lock: dict) -> str:
    sha = write_lock(lock)
    assert SVH.cmd_unlock(None) == 0 and C._STATE["stage_v"] is True
    return sha


def _floats(x) -> list:
    if isinstance(x, float):
        return [x]
    if isinstance(x, dict):
        return [f for v in x.values() for f in _floats(v)]
    if isinstance(x, list):
        return [f for v in x for f in _floats(v)]
    return []


# ----------------------------------------------------------------------------------------------------------------------
# LOCK
# ----------------------------------------------------------------------------------------------------------------------
def _decision(root: Path, **over) -> Path:
    dec = {"G_star": 50, "G_star_after_time_cap": 25, "stack": K.STACK, "hard_stop": False,
           "projection": {"projected_hours_v": 3.2}, "agg_invalid_share_locked": 0.0}
    top = {k: over.pop(k) for k in list(over) if k in ("locked_output", "schema")}
    dec.update(over)
    d = {"schema": SVH.SCHEMA_DECISION, "decision": dec, "locked_output": f"AGG.g{dec['G_star_after_time_cap']}."
                                                                          f"{dec['stack']}"}
    d.update(top)
    p = root / "decisions" / f"decision_{uuid.uuid4().hex[:6]}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d))
    return p


def _phase_t_dirs(root: Path, tamper: str | None = None) -> str:
    """Two synthetic Phase T run dirs whose config.json is h335_lab._config's code_sha256 format."""
    out = []
    for host in ("T_xbabe2", "T_xbabe0"):
        man = K.code_manifest(LAB.PHASE_T_FILES)
        if tamper and host == "T_xbabe0":
            man[str(HERE / tamper)] = "0" * 64
        d = root / "runs" / host
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text(json.dumps({"schema": "jope.h335.run-config/v1", "phase": "T", "host": host,
                                                   "cids": ["c1", "c2"], "code_sha256": man}))
        out.append(str(d))
    return ",".join(out)


def test_write_lock_schema_create_once_unlock_and_enter(env, capsys):
    p = _decision(env)
    args = argparse.Namespace(decision=str(p), frontier_binary=str(SVH.FRONTIER), phase_t_run_dirs=_phase_t_dirs(env))
    SVH.cmd_write_lock(args)
    sha = capsys.readouterr().out.strip().splitlines()[-1]
    lock = json.loads(SV.lock_path().read_text())
    assert sha == C.sha256_file(SV.lock_path()) and oct(SV.lock_path().stat().st_mode & 0o777) == "0o444"
    assert lock["schema"] == "jope.h335.lock/v1" and lock["stage_v_runs"] is True
    d = lock["design"]
    assert (d["G_star"], d["stack"], d["checkpoints"], d["gens_agg"], d["gens_base"]) == (25, K.STACK, [10, 25], 25, 27)
    assert (d["n_prop"], d["n_live"], d["old_offset"], d["seeds"]) == (48, 6, 2, [335101, 335102])
    assert lock["decision"]["sha256"] == C.sha256_file(p) and lock["decision"]["hard_stop"] is False
    assert lock["entrants"] == {
        "H335_AGG": "AGG.g25.CV3@12", "OLD_MATCHED": "OLD.g25.single", "OLD_MATCHED_CV3": "OLD.g25.CV3@12",
        "H317_BASE": "H317_BASE.single", "H257_REPLICA": "H257_REPLICA.CV1", "BANK_NN8_PICK": "BANK_NN8_PICK.single",
        "BANK_NN8_PORTFOLIO": "BANK_NN8_PORTFOLIO.CV5", "BANK_NN1": "BANK_NN1.single",
        "Opus5p5_Extra": "LLM:Opus5p5_Extra", "Gemini4_argon_high": "LLM:Gemini4_argon_high",
        "GPT6p1_Sol_Max": "LLM:GPT6p1_Sol_Max", "GPT6_Astra_Max": "LLM:GPT6_Astra_Max"}
    c = lock["constants"]
    assert (c["clip"], c["alpha"], lock["elo_anchor"], lock["primary"]["P1"][1], lock["primary"]["P2"][1]) == \
        (0.5, 0.025, "H257_REPLICA", "H257_REPLICA", "OLD_MATCHED")
    code, groups = lock["code_sha256"], lock["code_groups"]
    assert set(code) == {k for ks in groups.values() for k in ks} and all(k.startswith("/") for k in code)
    # LOCK freeze: code = exactly the runtime-code list + the 18 H317 pins; docs/tests/helpers are recorded, not verified
    h335 = {Path(k).name for k in groups["code"] if k.startswith(str(HERE) + "/")}
    assert h335 == set(SVH.LOCK_CODE_FILES) and len(groups["code"]) == len(SVH.LOCK_CODE_FILES) + len(K.H317_PINS)
    assert str(K.H317_SRC / "stage_v_h317.py") in groups["code"]
    docs = set(lock["docs_sha256"])
    assert {str(HERE / n) for n in ("PREREG.md", "test_stage_v_h335.py", "run_prep_v.sh", "stage_v_xbabe0.sh",
                                    "h335_replay.py", "h335_encode.py", "stage_v_mounts_h335.md")} <= docs
    assert not docs & set(code) and str(HERE / "stage_v_h335.py") not in docs
    assert groups["micro"] == ["/home/ubuntu/JopeDimeMicro/python/rust_evaluator.py",
                               "/home/ubuntu/JopeDimeMicro/python/rank_prepare.py",
                               "/home/ubuntu/JopeDimeMicro/assets/projections/program-feature-names.json"]
    assert len(lock["phase_t_run_configs"]) == 2 and lock["design"]["max_unit_attempts"] == 2
    assert code[str(SVH.FRONTIER)] == SVH.FRONTIER_SHA256 and code[str(SVH.WORKER)] == K.SUPERSET_SHA256
    assert len(groups["screener"]) == 10 and all(x.endswith("/model.pt") for x in groups["screener"])
    assert code[groups["views"][0]] == SV.VIEWS_MERGED_SHA256
    assert all(code[k] == C.sha256_file(Path(k)) for k in (groups["inputs"] + groups["code"][:3]))
    assert lock["prep_sha256"] and lock["train_inputs"]["n_tables"] == 1219
    with pytest.raises(FileExistsError):  # create-once
        SVH.cmd_write_lock(args)
    capsys.readouterr()
    assert SVH.cmd_verify_lock_files(argparse.Namespace(phase="audit", prep=True)) == 0  # never unlocks
    v = json.loads(capsys.readouterr().out)
    assert v["missing"] == v["changed"] == 0 and v["files"] == len(set(code) | set(lock["prep_sha256"]))
    assert C._STATE["stage_v"] is False and not SV.started_marker().exists()
    # receipt -> pinned one-shot unlock (every LOCK file re-hashed) -> enter re-verifies all of them
    C.LOCK_RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    C.LOCK_RECEIPT.write_text(f"{sha}  LOCK.json\n")
    assert SVH.cmd_unlock(None) == 0 and SV.started_marker().read_text().split()[1] == sha
    _, got, rep = SVH.enter("audit")
    assert got == sha and rep["files_verified"] == len(code) and rep["train_tables_verified"] == 1219
    with pytest.raises(C.GuardViolation, match="not allowed in this phase"):  # prep needs the ORIGINAL selected.json
        SVH.enter("prep")


@pytest.mark.parametrize("bad", [{"hard_stop": True}, {"G_star_after_time_cap": 30}, {"stack": "CV1"},
                                 {"locked_output": "AGG.g10.single"}, {"schema": "jope.h317.lock/v1"}])
def test_write_lock_refusals(env, bad):
    with pytest.raises(SVH.StageVError):
        SVH.build_lock(_decision(env, **bad), _phase_t_dirs(env).split(","))
    assert not SV.lock_path().exists()


@pytest.mark.parametrize("tamper", ["h335_lab.py", "gp_grammar_h317_learners_v1.py", "h335_core.py", "h335_stats.py"])
def test_write_lock_refuses_phase_t_code_mismatch(env, tamper):
    with pytest.raises(SVH.StageVError, match="Phase T code differs"):
        SVH.build_lock(_decision(env), _phase_t_dirs(env, tamper=tamper).split(","))
    with pytest.raises(SVH.StageVError, match="phase-t-run-dirs"):
        SVH.build_lock(_decision(env), [])
    assert not SV.lock_path().exists()


def test_verify_lock_files_refuses_changed_or_missing_file(env, capsys, monkeypatch):
    lock = synth_lock(env)
    write_lock(lock)
    assert SVH.cmd_verify_lock_files(argparse.Namespace(phase="audit", prep=False)) == 0
    Path(lock["code_groups"]["code"][0]).write_text("# tampered\n")
    unstaged = str(env / "code" / "never_staged.py")  # a LOCK file absent on this host
    lock2 = {**lock, "code_groups": {"code": lock["code_groups"]["code"] + [unstaged]},
             "code_sha256": {**lock["code_sha256"], unstaged: "0" * 64}}
    SV.lock_path().with_name("LOCK2.json").write_bytes(SV.json_bytes(lock2))
    monkeypatch.setattr(SV, "lock_path", lambda: C.DATA_ROOT / "stage_v" / "LOCK2.json")
    capsys.readouterr()
    assert SVH.cmd_verify_lock_files(argparse.Namespace(phase="audit", prep=False)) == 6
    out = capsys.readouterr()
    assert json.loads(out.out)["changed"] == 1 and json.loads(out.out)["missing"] == 1 and "MISSING" in out.err
    assert C._STATE["stage_v"] is False and not SV.started_marker().exists()


# ----------------------------------------------------------------------------------------------------------------------
# Unlock / enter
# ----------------------------------------------------------------------------------------------------------------------
def test_unlock_refuses_mismatched_receipt_then_one_shot(env):
    lock = synth_lock(env)
    sha = write_lock(lock, receipt_sha="0" * 64)
    with pytest.raises(C.GuardViolation):
        SVH.cmd_unlock(None)
    assert C._STATE["stage_v"] is False and not SV.started_marker().exists()
    C.LOCK_RECEIPT.write_text(f"{sha}  LOCK.json\n")
    Path(lock["code_groups"]["code"][0]).write_text("# tampered\n")
    with pytest.raises(C.GuardViolation, match="changed after LOCK"):
        SVH.cmd_unlock(None)
    Path(lock["code_groups"]["code"][0]).write_text("# synthetic LOCK file\n")
    assert SVH.cmd_unlock(None) == 0 and SV.started_marker().read_text().split()[1] == sha
    with pytest.raises(SV.RetryRefused):
        SVH.cmd_unlock(None)
    _, got, _ = SVH.enter("stats")
    assert got == sha


def test_unlock_refuses_non_h335_lock(env):
    lock = synth_lock(env)
    lock["schema"] = "jope.h317.lock/v1"
    write_lock(lock)
    with pytest.raises(SVH.StageVError):
        SVH.cmd_unlock(None)
    assert not SV.started_marker().exists()


def test_production_stagev_root_refused_before_unlock(env):
    assert SVH.STAGEV_ROOT == Path("/home/ubuntu/jope-data/h335-stagev") and C._STATE["stage_v"] is False
    with pytest.raises(C.GuardViolation):  # the guard refuses before any OS call
        open("/home/ubuntu/jope-data/h335-stagev/run/metrics_v.json")


# ----------------------------------------------------------------------------------------------------------------------
# Prep: order, regression, LLM envelopes, sealing, resume
# ----------------------------------------------------------------------------------------------------------------------
class _Enc:
    def dope_sha(self):
        return C.DOPE_SHA256


def _prep_setup(env, monkeypatch, cids, corrupt: bool = False):
    rows_by = {r["context_sha256"]: r for r in C.train_rows()}
    dev = [dict(rows_by[c], family_id=f"synthdev-{i}", dl_split="DL-Val", pool_sha256="0" * 64)
           for i, c in enumerate(cids)]
    real = C.load_selected()
    sel = {"train": real["train"], "dev": dev}
    monkeypatch.setattr(C, "load_selected", lambda: sel)
    monkeypatch.setattr(C, "dev_rows_for_stage_v", lambda: sorted(dev, key=lambda r: r["context_sha256"]))
    order: list = []
    monkeypatch.setattr(SV, "verify_inputs", lambda r, a, b: (order.append("inputs"), {"stub": True})[1])
    dd = {"summary": {m: {"flagged": [], "max_jaccard": 0.0} for m in SV.DEDUPE_MODES}, "per_context": {}}
    monkeypatch.setattr(SV, "run_dedupe", lambda r: (order.append("dedupe"), dd)[1])
    monkeypatch.setattr(SV, "encoder_parity", lambda enc: (order.append("parity"), {"all_exact": True})[1])
    man = {r["context_sha256"]: {"family_id": r["family_id"], "nt": 1, "width": 1, "table_sha256": "0" * 64,
                                 "search_views_sha256": "0" * 64} for r in dev}
    monkeypatch.setattr(SV, "encode_targets", lambda enc, r: (order.append("encode"), man)[1])
    rng = np.random.default_rng(335901)
    pools, costs, ref = {}, {}, []
    for c in cids:
        fit = rng.normal(0.5, 0.2, 40)
        gate = rng.random(40) > 0.2
        pools[c] = {"fitness": fit, "gate": gate, "utility": SV.native_utility_h258(fit, gate),
                    "program": np.zeros((40, 412), np.float32)}
        costs[c] = rng.random(40).astype(np.float32)
        ref.append(float(pools[c]["utility"][int(np.argmin(costs[c]))]))
    if corrupt:
        ref[1] += 1e-9
    inp = env / "inputs"
    inp.mkdir(parents=True)
    (inp / "reference.json").write_text(json.dumps({"evaluation_contract": {"contexts": [{"cid": c} for c in cids]},
                                                    "utility": {"per_context": ref}}))
    np.savez(inp / "costs.npz", **costs)
    monkeypatch.setattr(SV, "load_pool", lambda c: (order.append("pool"), pools[c])[1])
    orig = SV.r882_regression
    mean = float(np.mean([pools[r["context_sha256"]]["utility"][int(np.argmin(costs[r["context_sha256"]]))]
                          for r in dev]))
    monkeypatch.setattr(SV, "r882_regression", lambda *a, **k: (order.append("regression"),
                                                                 orig(*a, **k, expected_mean=mean,
                                                                      n_expected=len(cids)))[1])
    unlocked(synth_lock(env))
    args = argparse.Namespace(resume=False, reference=str(inp / "reference.json"),
                              costs=str(inp / "costs.npz"), threads=1, llm_jobs=2)
    return args, order


def test_prep_order_envelopes_llm_counts_only_and_resume(env, monkeypatch, capsys, small_train):
    cids = sorted(small_train[:3])
    args, order = _prep_setup(env, monkeypatch, cids)
    capsys.readouterr()
    out = SVH.run_prep(args, encoder_factory=_Enc)
    printed = capsys.readouterr().out
    rd = SV.run_dir()
    assert order == ["inputs", "dedupe", "parity", "encode"] + ["pool"] * 3 + ["regression"]
    ev = [e["event"] for e in SV.read_jsonl(rd / "events.jsonl")[0]]
    assert ev.index("dedupe_done") < ev.index("regression_pass") < ev.index("prep_done") < ev.index("llm_done")
    assert json.loads((rd / "native.json").read_text())["regression"]["pass"] is True
    assert json.loads((rd / SV.PREP_DONE).read_text())["regression_pass"] is True
    # LLM AUDIT envelopes: fold 0 = SEARCH rows, fold 1 = AUDIT rows (last), nq = 0, exactly the H317 split
    envs = json.loads((rd / "llm_envelopes.json").read_text())["envelopes"]
    for c in cids:
        raw = (SVH.llm_env_dir() / f"{c}.jtt1").read_bytes()
        env0 = C.read_jtt1(C.table_path(c))
        search, _, mask = C.search_and_audit_envelopes(c, env0)
        nt, nq, w = np.frombuffer(raw[4:16], "<u4")
        folds = np.frombuffer(raw, np.uint8, nt, 17)
        mat = np.frombuffer(raw, "<f8", nt * (w + 1), 17 + nt).reshape(nt, w + 1)
        assert raw[:4] == b"JTT1" and nq == 0 and nt == env0.nt and set(folds.tolist()) == {0, 1}
        assert int(folds.sum()) == int(mask.sum()) == envs[c]["n_audit"] and folds[-int(mask.sum()):].all()
        np.testing.assert_array_equal(mat[folds == 1], env0.mat[mask])
        np.testing.assert_array_equal(mat[folds == 0], search.mat)
        assert FA.read_envelope_counts(SVH.llm_env_dir() / f"{c}.jtt1") == (envs[c]["n_search"], envs[c]["n_audit"])
    # the real pinned frontier binary ran every (context, entrant) pair; values stay in the sealed JSONL
    recs = SV.read_jsonl(rd / SVH.LLM_RECORDS)[0]
    assert {(r["cid"], r["entrant"]) for r in recs} == {(c, e) for c in cids for e in FA.ENTRANTS}
    assert all(r["binary_sha256"] == SVH.FRONTIER_SHA256 and r["n_audit"] == envs[r["cid"]]["n_audit"] for r in recs)
    assert out["llm"]["ran"] == 12 and sum(out["llm"]["status_counts"].values()) == 12
    assert not _floats(json.loads(printed.strip().splitlines()[-1])) and not re.search(r"\d\.\d", printed)
    for r in recs:
        assert repr(r["utility"])[:7] not in printed
    # no retry: prep is complete; --resume completes only the (already complete) LLM step, nothing recomputed
    with pytest.raises(SV.RetryRefused, match="--resume"):
        SVH.run_prep(args, encoder_factory=_Enc)
    n = len(order)
    again = SVH.run_prep(argparse.Namespace(**{**vars(args), "resume": True}), encoder_factory=_Enc)
    assert again["mode"] == "resume" and again["llm"] == out["llm"] and len(order) == n
    assert len(SV.read_jsonl(rd / SVH.LLM_RECORDS)[0]) == 12


def fake_llm(fail_after: int | None = None, calls: list | None = None):
    """frontier_audit_h335.run stand-in: append-only, skip-existing; raises an infrastructure error after
    `fail_after` new records."""
    def run(envs_dir, cids, out_jsonl, binary, *, cpu_limit_s, wall_limit_s, jobs, binary_sha256, entrants):
        done = FA._existing_pairs(Path(out_jsonl), binary_sha256)
        log, ran = SV.JsonlLog(Path(out_jsonl)), 0
        if calls is not None:
            calls.append(len(done))
        for c in cids:
            for e in entrants:
                if (c, e) in done:
                    continue
                if fail_after is not None and ran >= fail_after:
                    raise OSError(errno.EIO, "synthetic LLM host crash")
                log.append({"cid": c, "entrant": e, "status": "completed", "utility": 0.25, "seed": FA.SEED,
                            "binary_sha256": binary_sha256, "wrapper_version": FA.WRAPPER_VERSION})
                ran += 1
        return {"ran": ran, "skipped": len(done), "refused": [], "status_counts": {"completed": ran}}
    return run


def test_prep_resumes_any_number_of_times_before_first_outcome(env, monkeypatch, capsys, small_train):
    cids = sorted(small_train[:3])
    args, order = _prep_setup(env, monkeypatch, cids)
    enc = SV.encode_targets
    crash = {"n": 1}

    def encode_once_crashing(e, r):
        if crash["n"]:
            crash["n"] -= 1
            raise MemoryError("synthetic host failure while encoding")
        return enc(e, r)

    monkeypatch.setattr(SV, "encode_targets", encode_once_crashing)
    resume = argparse.Namespace(**{**vars(args), "resume": True})
    rd, calls = SV.run_dir(), []
    with pytest.raises(MemoryError):  # 1: crash before the regression
        SVH.run_prep(args, encoder_factory=_Enc, llm_runner=fake_llm(calls=calls))
    with pytest.raises(SV.RetryRefused, match="--resume"):
        SVH.run_prep(args, encoder_factory=_Enc, llm_runner=fake_llm(calls=calls))
    with pytest.raises(OSError):  # 2: resume; the regression is recorded, then the LLM step crashes mid-way
        SVH.run_prep(resume, encoder_factory=_Enc, llm_runner=fake_llm(fail_after=5, calls=calls))
    assert (rd / SV.PREP_DONE).exists() and not (rd / SVH.LLM_DONE).exists()
    out = SVH.run_prep(resume, encoder_factory=_Enc, llm_runner=fake_llm(calls=calls))  # 3: LLM step completes
    assert out["llm"] == {"ran": 7, "skipped": 5, "status_counts": {"completed": 7}} and calls == [0, 5]
    again = SVH.run_prep(resume, encoder_factory=_Enc, llm_runner=fake_llm(calls=calls))  # 4: nothing left
    assert again["llm"] == out["llm"] and calls == [0, 5] and len(SV.read_jsonl(rd / SVH.LLM_RECORDS)[0]) == 12
    assert order.count("regression") == 1 and order.count("dedupe") == 1 and order.count("inputs") == 1
    ev = [e for e in SV.read_jsonl(rd / "events.jsonl")[0] if e["event"] == "prep_start"]
    assert [e["mode"] for e in ev] == ["first", "resume", "resume", "resume"]
    SV.create_once(rd / SV.FIRST_OUTCOME, b"synthetic\n")
    with pytest.raises(SV.RetryRefused, match="FIRST_OUTCOME_READ"):
        SVH.run_prep(resume, encoder_factory=_Enc, llm_runner=fake_llm(calls=calls))


def test_prep_regression_fail_stops_stage_v(env, monkeypatch, capsys, small_train):
    args, order = _prep_setup(env, monkeypatch, sorted(small_train[:3]), corrupt=True)
    capsys.readouterr()
    with pytest.raises(SV.RegressionFail):
        SVH.run_prep(args, encoder_factory=_Enc, llm_runner=lambda *a, **k: pytest.fail("LLM ran after a FAIL"))
    rd = SV.run_dir()
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["verdict"] == "REGRESSION_FAIL"
    assert (rd / SV.REG_FAIL).exists() and not (rd / SV.PREP_DONE).exists()
    assert not (rd / SVH.LLM_RECORDS).exists() and not (rd / SVH.LLM_DONE).exists()  # envelopes are outcome-free
    with pytest.raises(SV.RetryRefused, match="REGRESSION_FAIL"):
        SVH.run_prep(argparse.Namespace(**{**vars(args), "resume": True}), encoder_factory=_Enc)
    with pytest.raises(SV.RetryRefused):
        SVH.admit_audit(rd)
    m = SVH.run_stats(argparse.Namespace(final_incomplete=False, resume=False))
    assert m["verdict"] == "REGRESSION_FAIL" and m["regression"]["pass"] is False


# ----------------------------------------------------------------------------------------------------------------------
# Audit: sealed shard units, resume, one process per (group, shard), launcher
# ----------------------------------------------------------------------------------------------------------------------
def _audit_setup(env, monkeypatch, cids):
    monkeypatch.setattr(SV, "N_TARGETS", len(cids))
    rows_by = {r["context_sha256"]: r for r in C.train_rows()}
    monkeypatch.setattr(SV, "N_TARGET_FAMILIES", len({rows_by[c]["family_id"] for c in cids}))
    unlocked(synth_lock(env, gstar=10, checkpoints=(10,)))
    rd = SV.run_dir()
    SV.write_atomic(rd / SV.PREP_DONE, SV.json_bytes({"regression_pass": True, "wall_s": 1}))
    man = {}
    for c in cids:
        vp = SV.views_search_dir() / f"{c}.npz"
        vp.parent.mkdir(parents=True, exist_ok=True)
        np.savez(vp, synthetic=np.arange(3))
        env0 = C.read_jtt1(C.table_path(c))
        man[c] = {"family_id": rows_by[c]["family_id"], "nt": int(env0.nt), "width": int(env0.width),
                  "table_sha256": C.sha256_file(C.table_path(c)), "search_views_sha256": C.sha256_file(vp)}
    SV.write_atomic(rd / "targets_manifest.json", SV.json_bytes({"targets": man}))
    return rd


def fake_rt_factory(cores, d):
    rows = C.train_rows()
    bank, _ = D.build_bank(rows)
    eng = E.EvalEngine(list(cores), root=SVH.gp_dir() / f"gproot_{K.GROUP}", runner=T.fake_runner(fail_frac=0.0))
    fb = S.FallbackScreener()
    return L.Runtime(eng, E.AuditLedger(eng), lambda cid: fb, T.HashCoder(), bank,
                     {r["context_sha256"]: r["family_id"] for r in rows},
                     {r["context_sha256"]: r["dataset_id"] for r in rows}, L.Store(SVH.gp_dir()), tuple(d["seeds"]))


def test_audit_shard_counts_only_resume_and_no_run_after_first_outcome(env, monkeypatch, capsys, small_train):
    cids = sorted(small_train[3:5])
    rd = _audit_setup(env, monkeypatch, cids)
    args = argparse.Namespace(group=K.GROUP, shard="0/1", cores="0-3", jobs=2, torch_threads=1, resume=False)
    capsys.readouterr()
    out = SVH.run_audit_shard(args, rt_factory=fake_rt_factory)
    printed = capsys.readouterr().out
    assert out["units"] == out["executed"] == 4 and out["replayed"] == out["capped"] == 0
    assert out["failed_units"] == out["failed_outputs"] == 0
    assert not _floats(json.loads(printed.strip().splitlines()[-1])) and not re.search(r"\d\.\d", printed)
    st = L.Store(SVH.gp_dir())
    recs = [st.result(f"V_{K.GROUP}", c, s) for c in cids for s in SEEDS]
    assert all(r and r["phase"] == "V" and r["group"] == K.GROUP and r["worker_sha256"] == K.SUPERSET_SHA256
               for r in recs)
    assert set(recs[0]["outputs"]) == set(LAB.unit_keys(K.GROUP, [10]))
    assert not any(o.get("failed") for r in recs for o in r["outputs"].values())
    for r in recs:
        for o in r["outputs"].values():
            assert repr(float(o["value"]))[:7] not in printed
    (p1,) = list(rd.glob(f"AUDIT_DONE.{K.GROUP}.s0of1.*.json"))
    done = json.loads(p1.read_text())
    assert done["audit_ledger"]["outputs"] > 0 and done["engine"]["physical_runs"] > 0
    with pytest.raises(SV.RetryRefused, match="--resume"):
        SVH.run_audit_shard(args, rt_factory=fake_rt_factory)
    again = SVH.run_audit_shard(argparse.Namespace(**{**vars(args), "resume": True}), rt_factory=fake_rt_factory)
    p2 = [p for p in rd.glob(f"AUDIT_DONE.{K.GROUP}.s0of1.*.json") if p != p1]
    assert len(p2) == 1 and p1.exists()  # one AUDIT_DONE per attempt, never overwritten
    done2 = json.loads(p2[0].read_text())
    assert again["replayed"] == 4 and again["executed"] == 0 and done2["engine"]["physical_runs"] == 0
    assert all(st.n_results(f"V_{K.GROUP}", c, s) == 1 for c in cids for s in SEEDS)
    SV.create_once(rd / SV.FIRST_OUTCOME, b"synthetic\n")
    with pytest.raises(SV.RetryRefused, match="FIRST_OUTCOME_READ"):
        SVH.run_audit_shard(argparse.Namespace(**{**vars(args), "resume": True}), rt_factory=fake_rt_factory)


def _fake_unit(calls: list, fail: dict):
    """Stand-in for h335_lab.unit_agg / unit_base (store replay, a record per unit) that raises fail[(cid, seed)]."""
    def unit(rt, phase, cid, seed, gens, checkpoints, stacks=LAB.STACKS):
        hit = rt.store.result(f"{phase}_{K.GROUP}", cid, seed)
        if hit:
            return hit
        calls.append((cid, seed))
        if (cid, seed) in fail:
            raise fail[(cid, seed)]
        rec = {"type": "result", "schema": LAB.SCHEMA, "cid": cid, "seed": seed, "phase": phase, "group": K.GROUP,
               "grammar_sha256": K.GRAMMAR_SHA256, "worker_sha256": K.SUPERSET_SHA256, "gens": gens,
               "outputs": {k: {"value": 0.125, "invalid": False, "live_evals": 1}
                           for k in LAB.unit_keys(K.GROUP, checkpoints, stacks)}}
        rt.store.append(f"{phase}_{K.GROUP}", cid, seed, rec)
        return rec
    return unit


def _first_unit(cids):
    return (sorted(cids, key=lambda c: (-K.table_size(c), c))[0], SEEDS[0])


def test_integrity_error_writes_stop_marker_and_cancels_pending_units(env, monkeypatch, small_train):
    cids = sorted(small_train[3:5])
    rd = _audit_setup(env, monkeypatch, cids)
    calls: list = []
    unit = _fake_unit(calls, {_first_unit(cids): C.GuardViolation("synthetic AUDIT leak")})
    monkeypatch.setattr(LAB, "unit_agg", unit)
    monkeypatch.setattr(LAB, "unit_base", unit)
    args = argparse.Namespace(group=K.GROUP, shard="0/1", cores="0-3", jobs=1, torch_threads=1, resume=False)
    with pytest.raises(C.GuardViolation):
        SVH.run_audit_shard(args, rt_factory=fake_rt_factory)
    assert calls == [_first_unit(cids)]  # the 3 pending units were never started (no run_parallel drain)
    assert (rd / SVH.INTEGRITY_STOP).exists() and not list(rd.glob("AUDIT_DONE.*"))
    assert "integrity_stop" in [e["event"] for e in SV.read_jsonl(rd / "events.jsonl")[0]]
    with pytest.raises(SV.RetryRefused, match="STOPPED_INTEGRITY"):
        SVH.run_audit_shard(argparse.Namespace(**{**vars(args), "resume": True}), rt_factory=fake_rt_factory)
    with pytest.raises(SV.RetryRefused, match="STOPPED_INTEGRITY"):
        SVH.admit_prep(rd, True)
    with pytest.raises(SV.RetryRefused, match="STOPPED_INTEGRITY"):
        SVH.run_stats(argparse.Namespace(final_incomplete=True, resume=False))
    assert not (rd / SV.FIRST_OUTCOME).exists()


def test_max_unit_attempts_records_a_repeating_host_failure(env, monkeypatch, small_train):
    cids = sorted(small_train[3:5])
    rd = _audit_setup(env, monkeypatch, cids)
    bad = _first_unit(cids)
    calls: list = []
    unit = _fake_unit(calls, {bad: OSError(errno.ENOSPC, "synthetic disk full")})
    monkeypatch.setattr(LAB, "unit_agg", unit)
    monkeypatch.setattr(LAB, "unit_base", unit)
    args = argparse.Namespace(group=K.GROUP, shard="0/1", cores="0-3", jobs=1, torch_threads=1, resume=False)
    resume = argparse.Namespace(**{**vars(args), "resume": True})
    with pytest.raises(OSError):  # attempt 1: no record, nothing else started
        SVH.run_audit_shard(args, rt_factory=fake_rt_factory)
    with pytest.raises(OSError):  # attempt 2 (resume)
        SVH.run_audit_shard(resume, rt_factory=fake_rt_factory)
    assert calls == [bad, bad] and not (rd / SVH.INTEGRITY_STOP).exists()
    out = SVH.run_audit_shard(resume, rt_factory=fake_rt_factory)  # attempt 3: recorded failed, not executed
    assert calls.count(bad) == 2 and len(calls) == 5 and out["capped"] == 1 and out["failed_units"] == 1
    rec = L.Store(SVH.gp_dir()).result(f"V_{K.GROUP}", *bad)
    assert rec["failed"] and set(rec["outputs"]) == set(LAB.unit_keys(K.GROUP, [10]))
    assert all(o["value"] == 0.0 and o["invalid"] for o in rec["outputs"].values())
    assert "culprit of 2 propagated errors" in rec["error"]
    log = SV.read_jsonl(rd / f"unit_attempts.{K.GROUP}.s0of1.jsonl")[0]
    starts = [r for r in log if r["type"] == "start"]
    assert [r["n"] for r in starts if (r["cid"], r["seed"]) == bad] == [1, 2] and len(starts) == 5
    assert [(r["cid"], r["seed"], r["error"]) for r in log if r["type"] == "culprit"] == [(*bad, "OSError")] * 2
    assert len(list(rd.glob(f"AUDIT_DONE.{K.GROUP}.s0of1.*.json"))) == 1


def test_attempt_cap_counts_culprits_not_bystanders_of_a_process_kill(env, monkeypatch, small_train):
    """Two process-level kills (cgroup OOM / systemctl stop) each leave 2 bystander units in flight (start lines, no
    record, no culprit) while 1 unit raises in-process each time (culprit line). Only the culprit is capped."""
    cids = sorted(small_train[3:5])
    rd = _audit_setup(env, monkeypatch, cids)
    order = [(c, s) for c in sorted(cids, key=lambda c: (-K.table_size(c), c)) for s in SEEDS]
    culprit, by1, by2 = order[0], order[1], order[2]
    calls: list = []
    unit = _fake_unit(calls, {culprit: MemoryError("synthetic shard OOM")})
    monkeypatch.setattr(LAB, "unit_agg", unit)
    monkeypatch.setattr(LAB, "unit_base", unit)
    args = argparse.Namespace(group=K.GROUP, shard="0/1", cores="0-3", jobs=1, torch_threads=1, resume=False)
    killed = SVH.AttemptLog(rd, K.GROUP, 0, 1)
    for kill in range(2):
        killed.start(*by1)  # what the dying process had written for its in-flight bystanders
        killed.start(*by2)
        with pytest.raises(MemoryError):
            SVH.run_audit_shard(argparse.Namespace(**{**vars(args), "resume": kill > 0}), rt_factory=fake_rt_factory)
    log = SVH.AttemptLog(rd, K.GROUP, 0, 1)
    assert log.started(*by1) == log.started(*by2) == 2 and log.culprit_count(*by1) == log.culprit_count(*by2) == 0
    assert log.culprit_count(*culprit) == 2
    out = SVH.run_audit_shard(argparse.Namespace(**{**vars(args), "resume": True}), rt_factory=fake_rt_factory)
    assert out["capped"] == 1 and out["failed_units"] == 1 and out["reruns"] == 2 and out["units"] == 4
    st = L.Store(SVH.gp_dir())
    assert st.result(f"V_{K.GROUP}", *culprit)["failed"]
    for b in (by1, by2):  # the bystanders ran normally: never capped by the kills
        r = st.result(f"V_{K.GROUP}", *b)
        assert not r.get("failed") and all(o["value"] == 0.125 for o in r["outputs"].values())
    assert calls.count(culprit) == 2 and calls.count(by1) == calls.count(by2) == 1 and len(calls) == 5


def test_shard_run_lock_is_per_group_and_shard(env):
    rd = SV.run_dir()
    with SVH._shard_lock(rd, "AGG", 0, 2):
        with pytest.raises(SV.RetryRefused, match="already running"):
            SVH._shard_lock(rd, "AGG", 0, 2).__enter__()
        with SVH._shard_lock(rd, "BASE", 0, 2), SVH._shard_lock(rd, "AGG", 1, 2):
            pass
    assert (rd / "AUDIT_RUNNING.AGG.s0of2").exists() and (rd / "AUDIT_RUNNING.BASE.s0of2").exists()


def test_shard_cids_partition_is_largest_first_round_robin():
    man = {f"{i:02d}" + "c" * 62: {"nt": 100 + 7 * (i % 5), "width": 3 + i % 3} for i in range(23)}
    parts = [SVH.shard_cids(man, k, 4) for k in range(4)]
    assert sorted(c for p in parts for c in p) == sorted(man) and sum(map(len, parts)) == 23
    big = max(man, key=lambda c: (man[c]["nt"] * (man[c]["width"] + 1), c[::-1]))
    assert any(p and man[p[0]]["nt"] * (man[p[0]]["width"] + 1) == man[big]["nt"] * (man[big]["width"] + 1)
               for p in parts)
    assert parts == [SVH.shard_cids(dict(reversed(list(man.items()))), k, 4) for k in range(4)]


class _Proc:
    def __init__(self, cmd, env, **kw):
        _Proc.calls.append((cmd, env))

    def wait(self):
        return 0


def test_audit_launcher_groups_cores_plan_and_resume(env, monkeypatch, capsys, small_train):
    _audit_setup(env, monkeypatch, sorted(small_train[3:5]))
    _Proc.calls = []
    monkeypatch.setattr(SVH.subprocess, "Popen", _Proc)
    a = argparse.Namespace(cpus="0-11", shards_agg=2, shards_base=1, workers_per_shard=4, jobs=3, torch_threads=1,
                           resume=False)
    assert SVH.cmd_audit(a) == 0
    cmds = {(c[c.index("--group") + 1], c[c.index("--shard") + 1]): (c, e) for c, e in _Proc.calls}
    assert set(cmds) == {("AGG", "0/2"), ("AGG", "1/2"), ("BASE", "0/1")}
    assert all(e["H335_GROUP"] == g and "H317_LEARNERS_V1" not in e and "--resume" not in c
               for (g, _), (c, e) in cmds.items())
    cores = [c[c.index("--cores") + 1] for c, _ in cmds.values()]
    assert sorted(cores) == ["0,1,2,3", "4,5,6,7", "8,9,10,11"]
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["units_recorded"] == {"AGG": 0, "BASE": 0}
    with pytest.raises(SV.RetryRefused):
        SVH.cmd_audit(a)
    with pytest.raises(SV.RetryRefused, match="layout"):
        SVH.cmd_audit(argparse.Namespace(**{**vars(a), "resume": True, "shards_base": 2}))
    _Proc.calls = []
    assert SVH.cmd_audit(argparse.Namespace(**{**vars(a), "resume": True})) == 0
    assert all("--resume" in c for c, _ in _Proc.calls) and len(_Proc.calls) == 3


# ----------------------------------------------------------------------------------------------------------------------
# Stats: completeness, FIRST_OUTCOME_READ, verdict / Elo wiring
# ----------------------------------------------------------------------------------------------------------------------
BASE_OFFS = {"H335_AGG": 0.30, "OLD_MATCHED": 0.15, "OLD_MATCHED_CV3": 0.20, "H317_BASE": 0.10, "H257_REPLICA": 0.0,
             "BANK_NN8_PICK": -0.10, "BANK_NN8_PORTFOLIO": -0.05, "BANK_NN1": -0.20, "Opus5p5_Extra": 0.05,
             "Gemini4_argon_high": 0.02, "GPT6p1_Sol_Max": -0.02, "GPT6_Astra_Max": -0.04}


def synth_cids() -> list:
    return [hashlib.sha256(f"h335-synth-{i}".encode()).hexdigest() for i in range(91)]


def synth_stage(offs: dict, skip_units=(), skip_llm=(), flagged_family: str | None = None,
                unit_mutate: dict | None = None, llm_mutate: dict | None = None):
    """91 synthetic targets / 19 families; per-(context, seed, output) values = base + entrant offset + noise."""
    lock = json.loads(SV.lock_path().read_text())
    keys, ckpt = lock["entrants"], lock["design"]["checkpoints"]
    inv = {v: k for k, v in keys.items()}
    cids = synth_cids()
    fam = {c: f"synthfam-{i % 19:02d}" for i, c in enumerate(cids)}
    rd = SV.run_dir()
    man = {c: {"family_id": fam[c], "dataset_id": f"synthds-{i}", "nt": 100 + i, "width": 5, "table_sha256": "0" * 64,
               "search_views_sha256": "0" * 64} for i, c in enumerate(cids)}
    SV.write_atomic(rd / "targets_manifest.json", SV.json_bytes({"targets": man}))
    SV.write_atomic(rd / SV.PREP_DONE, SV.json_bytes({"regression_pass": True, "wall_s": 12}))
    flag = {c for c in cids if fam[c] == flagged_family}
    per = {c: {m: {"max_jaccard": 0.9 if c in flag else 0.01, "max_overlap": 0.02} for m in SV.DEDUPE_MODES}
           for c in cids}
    summ = {m: {"max_jaccard": 0.9 if flag else 0.01, "flagged": [{"cid12": c[:12]} for c in sorted(flag)]}
            for m in SV.DEDUPE_MODES}
    SV.write_atomic(rd / "dedupe.json", SV.json_bytes({"summary": summ, "per_context": per}))
    rng = np.random.default_rng(335902)
    base = {c: float(rng.normal(1.2, 0.4)) for c in cids}
    st = L.Store(SVH.gp_dir())
    gram = lock["design"]["grammar_sha256"]
    for g in SVH.GROUPS:
        for c in cids:
            for s in SEEDS:
                if (g, c, s) in skip_units:
                    continue
                outs = {}
                for k in LAB.unit_keys(g, ckpt):
                    v = base[c] + offs.get(inv.get(k), 0.07) + float(rng.normal(0, 0.01))
                    live = 22 + 6 * 10 if k.startswith("AGG") else 8 + 6 * 12
                    outs[k] = {"value": v, "invalid": False, "live_evals": live, "search_utility": v + 0.1}
                rec = {"type": "result", "schema": LAB.SCHEMA, "cid": c, "seed": s, "phase": "V", "group": g,
                       "grammar_sha256": gram[g], "worker_sha256": K.SUPERSET_SHA256, "outputs": outs, "wall_s": 1.0,
                       "gens": lock["design"]["gens_agg"] + (K.OLD_OFFSET if g == "BASE" else 0)}
                if g == "AGG":
                    rec["seed_F"] = {"evals_used": 22}
                rec.update((unit_mutate or {}).get((g, c, s), {}))
                _put(st.path(f"V_{g}", c, s), rec)
    lines = [json.dumps({"cid": c, "entrant": e, "status": "completed", "binary_sha256": SVH.FRONTIER_SHA256,
                         "seed": FA.SEED, "wrapper_version": FA.WRAPPER_VERSION,
                         "utility": base[c] + offs[e] + float(rng.normal(0, 0.01)),
                         **(llm_mutate or {}).get((c, e), {})}) + "\n"
             for c in cids for e in FA.ENTRANTS if (c, e) not in skip_llm]
    (rd / SVH.LLM_RECORDS).write_text("".join(lines))
    return cids, fam


def _put(p: Path, rec: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, sort_keys=True) + "\n")


STATS = argparse.Namespace(final_incomplete=False, resume=False)


@pytest.mark.parametrize("offs,verdict", [
    (BASE_OFFS, "BETTER_MODEL"),
    ({**BASE_OFFS, "OLD_MATCHED": 0.40}, "BETTER_WITH_COMPUTE"),
    ({**BASE_OFFS, "H335_AGG": -0.30}, "INFERIOR_TO_INCUMBENT"),
    ({**BASE_OFFS, "H335_AGG": 0.0}, "INCONCLUSIVE"),
])
def test_stats_verdict_and_elo_wiring(env, capsys, offs, verdict):
    unlocked(synth_lock(env))
    cids, fam = synth_stage(offs, flagged_family="synthfam-03")
    capsys.readouterr()
    m = SVH.run_stats(STATS)
    assert m["verdict"] == verdict and m["complete"] is True and m["contexts_used"] == 91
    assert json.loads(capsys.readouterr().out)["verdict"] == verdict
    assert m["primary"]["P1"]["families"] == 19 and m["bytes_compact"] < 50_000
    assert (SV.run_dir() / SVH.METRICS).stat().st_size < 50_000
    elo = m["elo_audit"]
    assert elo["anchor"] == "H257_REPLICA" and elo["elo"]["H257_REPLICA"]["elo"] == 1500.0
    assert set(elo["elo"]) == set(BASE_OFFS) and elo["families"] == 19
    pair = m["elo_pairwise_agg_vs_h257"]["elo"]
    assert pair["H257_REPLICA"]["elo"] == 1500.0
    if abs(offs["H335_AGG"]) >= 0.3:
        assert (pair["H335_AGG"]["elo"] > 1500.0) == (offs["H335_AGG"] > 0)
    if verdict == "BETTER_MODEL":
        assert max(elo["elo"], key=lambda a: elo["elo"][a]["elo"]) == "H335_AGG"
        assert m["primary"]["P1"]["signflip_p"] == 2.0 ** -19  # full precision (not rounded)
        assert m["headline"]["families_won"] == 19 and m["contrasts"]["H257_REPLICA"]["adoption_gate"]["gate_pass"]
        assert m["family_means"]["H335_AGG"] == pytest.approx(m["family_means"]["H257_REPLICA"] + 0.3, abs=0.01)
    s = m["dedupe_sensitivity"]
    assert s["excluded_contexts"] == sum(f == "synthfam-03" for f in fam.values()) and s["P1"]["families"] == 18
    assert m["invalid_flag"] == {"H257_REPLICA": True, "OLD_MATCHED": True} and m["live_match"]["max_abs_gap_agg_vs_old"] \
        == 2 and m["live_match"]["violations"] == 0
    assert m["counts"]["Opus5p5_Extra"]["outputs"] == 91 and m["counts"]["H335_AGG"]["outputs"] == 182
    assert m["budget_shortfall_units"] == 0 and m["seed_F_evals_not_22"] == 0
    panel = json.loads((SV.run_dir() / "panel_stage_v.json").read_text())
    assert len(panel) == 91 and len(panel[cids[0]]["gp"]["H335_AGG"]) == 2 and len(panel[cids[0]]["llm"]) == 4


def test_stats_refuses_incomplete_then_final_incomplete_is_inconclusive(env, capsys, monkeypatch):
    monkeypatch.setattr(SVH, "_utcnow", lambda: SVH.STAGE_V_COMPUTE_STOP_UTC - 3600)
    unlocked(synth_lock(env))
    cids = synth_cids()
    synth_stage(BASE_OFFS, skip_units=[("BASE", cids[5], SEEDS[1])], skip_llm=[(cids[7], "GPT6_Astra_Max")])
    rd = SV.run_dir()
    with pytest.raises(SV.RetryRefused, match="1 unit.*1 LLM"):
        SVH.run_stats(STATS)
    assert not (rd / SV.FIRST_OUTCOME).exists() and not (rd / SVH.METRICS).exists()
    _plan(rd, done=("AGG", "BASE"), llm=False)  # every planned shard done, but the LLM step is not
    with pytest.raises(SV.RetryRefused, match="LLM_DONE"):
        SVH.run_stats(argparse.Namespace(final_incomplete=True, resume=False))
    assert not (rd / SV.FIRST_OUTCOME).exists()
    SV.write_atomic(rd / SVH.LLM_DONE, SV.json_bytes({"summary": {"ran": 363}}))
    capsys.readouterr()
    m = SVH.run_stats(argparse.Namespace(final_incomplete=True, resume=False))
    assert m["verdict"] == "INCONCLUSIVE" and m["complete"] is False
    assert m["incomplete"]["missing_units"] == 1 and m["incomplete"]["missing_llm"] == 1
    assert m["incomplete"]["allowed_because"] == "every planned audit shard recorded AUDIT_DONE and LLM_DONE.json exists"
    assert m["contexts_used"] == 89 and m["primary"]["P1"]["pass"] is True  # descriptive subset; verdict stays
    assert json.loads((rd / SV.FIRST_OUTCOME).read_text())["complete"] is False


def _plan(rd: Path, done=(), llm: bool = True) -> None:
    SV.write_atomic(rd / "AUDIT_PLAN.json", SV.json_bytes({"shards": {"AGG": 1, "BASE": 1}}))
    for g in done:
        SV.write_atomic(rd / f"AUDIT_DONE.{g}.s0of1.20261105T000000Z_1.json", SV.json_bytes({"wall_s": 1}))
    if llm:
        SV.write_atomic(rd / SVH.LLM_DONE, SV.json_bytes({"summary": {"ran": 364}}))


def test_final_incomplete_needs_finished_plan_or_the_compute_stop(env, monkeypatch):
    unlocked(synth_lock(env))
    cids = synth_cids()
    synth_stage(BASE_OFFS, skip_units=[("AGG", cids[2], SEEDS[0])])
    rd = SV.run_dir()
    fin = argparse.Namespace(final_incomplete=True, resume=False)
    monkeypatch.setattr(SVH, "_utcnow", lambda: SVH.STAGE_V_COMPUTE_STOP_UTC - 60)
    with pytest.raises(SV.RetryRefused, match="LLM_DONE"):
        SVH.run_stats(fin)
    SV.write_atomic(rd / SVH.LLM_DONE, SV.json_bytes({"summary": {}}))
    with pytest.raises(SV.RetryRefused, match="no AUDIT_PLAN"):
        SVH.run_stats(fin)
    SV.write_atomic(rd / "AUDIT_PLAN.json", SV.json_bytes({"shards": {"AGG": 1, "BASE": 2}}))
    for n in ("AGG.s0of1", "BASE.s0of2"):  # BASE shard 1/2 has no AUDIT_DONE
        SV.write_atomic(rd / f"AUDIT_DONE.{n}.20261105T000000Z_1.json", SV.json_bytes({"wall_s": 1}))
    with pytest.raises(SV.RetryRefused, match="BASE:1/2"):
        SVH.run_stats(fin)
    assert not (rd / SV.FIRST_OUTCOME).exists()
    monkeypatch.setattr(SVH, "_utcnow", lambda: SVH.STAGE_V_COMPUTE_STOP_UTC + 1)  # past 2026-11-05T07:30:00Z
    m = SVH.run_stats(fin)
    assert m["verdict"] == "INCONCLUSIVE" and "compute stop" in m["incomplete"]["allowed_because"]
    assert SVH.STAGE_V_COMPUTE_STOP_UTC == 1793863800  # 2026-11-05T07:30:00Z


def test_first_outcome_read_is_create_once_and_binds_the_record_set(env, capsys):
    unlocked(synth_lock(env))
    cids, _ = synth_stage(BASE_OFFS)
    rd = SV.run_dir()
    m1 = SVH.run_stats(STATS)
    marker = rd / SV.FIRST_OUTCOME
    first = marker.read_bytes()
    assert oct(marker.stat().st_mode & 0o777) == "0o444" and json.loads(first)["complete"] is True
    with pytest.raises(SV.RetryRefused, match="FIRST_OUTCOME_READ exists"):
        SVH.run_stats(STATS)
    m2 = SVH.run_stats(argparse.Namespace(final_incomplete=False, resume=True))
    assert m2["verdict"] == m1["verdict"] and m2["primary"] == m1["primary"] and m2["elo_audit"] == m1["elo_audit"]
    assert m2["disclosure"]["stats_resumed"] is True and marker.read_bytes() == first
    with pytest.raises(FileExistsError):
        SV.create_once(marker, b"again\n")
    with open(rd / SVH.LLM_RECORDS, "a") as f:  # any later byte (here a torn line) changes the bound record set
        f.write('{"cid": "torn')
    with pytest.raises(SV.RetryRefused, match="records changed"):
        SVH.run_stats(argparse.Namespace(final_incomplete=False, resume=True))
    with pytest.raises(SV.RetryRefused, match="FIRST_OUTCOME_READ"):
        SVH.admit_audit(rd)


@pytest.mark.parametrize("mut", [
    {"unit": {"grammar_sha256": K.H317_PINS["gp_grammar_h317.py"]}, "why": "grammar"},
    {"unit": {"grammar_sha256": K.GRAMMAR_PORT_SHA256}, "group": "BASE", "why": "grammar"},
    {"unit": {"worker_sha256": K.PINNED_WORKER_SHA256}, "why": "worker"},
    {"unit": {"gens": 11}, "why": "gens"},
    {"unit": {"schema": "jope.h317.unit/v1"}, "why": "schema"},
    {"unit": {"seed": 335103}, "why": "identity"},
    {"llm": {"binary_sha256": "0" * 64}, "why": "LLM"},
    {"llm": {"seed": 7}, "why": "LLM"},
])
def test_identity_checks_refuse_before_first_outcome_read(env, mut):
    unlocked(synth_lock(env))
    c0 = synth_cids()[0]
    synth_stage(BASE_OFFS, unit_mutate={(mut.get("group", "AGG"), c0, SEEDS[0]): mut["unit"]} if "unit" in mut else None,
                llm_mutate={(c0, FA.ENTRANTS[0]): mut["llm"]} if "llm" in mut else None)
    with pytest.raises(SVH.StageVError, match=f"identity check.*{mut['why']}"):
        SVH.run_stats(argparse.Namespace(final_incomplete=True, resume=False))
    assert not (SV.run_dir() / SV.FIRST_OUTCOME).exists() and not (SV.run_dir() / SVH.METRICS).exists()


# ----------------------------------------------------------------------------------------------------------------------
# Sandboxes
# ----------------------------------------------------------------------------------------------------------------------
def _sbx(script: str, *a, env_extra=None):
    e = dict(os.environ)
    e.update(env_extra or {})
    return subprocess.run(["bash", str(HERE / script), *a], capture_output=True, text=True, env=e, timeout=60)


def _pairs(a):
    return [(a[i], a[i + 1], a[i + 2]) for i in range(len(a) - 2) if a[i] in ("--bind", "--ro-bind")]


def test_stagev_sandbox_layouts_and_refusals(env):
    rc = "/home/ubuntu/jope_plan/ns/receipts/H335_LOCK_sha256.txt"
    res01 = "/home/ubuntu/jope-data/res-01/input"
    r = _sbx("sandbox_h335_stagev.sh", "--print", "unlock", "python3", "stage_v_h335.py", "unlock")
    assert r.returncode == 0, r.stderr
    a = r.stdout.splitlines()
    assert ("--ro-bind", rc, rc) in _pairs(a)
    assert ("--ro-bind", "/home/ubuntu/jope-data/h335-view/res01-train-view", res01) in _pairs(a)
    assert not any("h335-stagev" in x for x in a) and "--unshare-net" in a and a[-2:] == ["stage_v_h335.py", "unlock"]
    sv, data = str(env / "sv"), str(env / "dataroot")
    r = _sbx("sandbox_h335_stagev.sh", "--print", "v", "python3", "x.py",
             env_extra={"H335_STAGEV_ROOT": sv, "H335_DATA": data})
    assert r.returncode == 0, r.stderr
    a = r.stdout.splitlines()
    b = _pairs(a)
    assert ("--ro-bind", f"{sv}/res01-v-view", res01) in b and ("--ro-bind", rc, rc) in b
    assert ("--ro-bind", f"{data}/stage_v", f"{data}/stage_v") in b and ("--bind", sv, sv) in b
    assert ("--ro-bind", f"{sv}/inputs", f"{sv}/inputs") in b and ("--ro-bind", f"{data}/bin", f"{data}/bin") in b
    i = a.index("H335_STAGEV_ROOT")
    assert a[i - 1] == "--setenv" and a[i + 1] == sv and "--unshare-net" in a
    assert a.index(f"{data}/stage_v") > a.index(data) and a.index(f"{sv}/inputs") > a.index(sv)
    for mode in ("unlock", "v"):  # no frozen copy given / a writable one / no receipt (never exists in tests)
        r = _sbx("sandbox_h335_stagev.sh", mode, "true")
        assert r.returncode == 3 and "H335_CODE" in r.stderr
        cp = env / f"codecopy_{mode}"
        cp.mkdir()
        (cp / "stage_v_h335.py").write_text("# copy\n")
        r = _sbx("sandbox_h335_stagev.sh", mode, "true", env_extra={"H335_CODE": str(cp)})
        assert r.returncode == 3 and "writable" in r.stderr
        os.chmod(cp / "stage_v_h335.py", 0o444)
        r = _sbx("sandbox_h335_stagev.sh", mode, "true", env_extra={"H335_CODE": str(cp)})
        assert r.returncode == 3 and "no LOCK receipt" in r.stderr
    assert _sbx("sandbox_h335_stagev.sh", "bogus", "true").returncode == 2


def test_docker_stage_v_mode_syntax_and_refusal():
    assert subprocess.run(["bash", "-n", str(HERE / "sandbox_h335_docker.sh")]).returncode == 0
    r = _sbx("sandbox_h335_docker.sh", "--stage-v", "true")
    assert r.returncode == 3 and "Stage V refused: no LOCK receipt" in r.stderr
    txt = (HERE / "sandbox_h335_docker.sh").read_text()
    for s in ("H335_LOCK_sha256.txt", "$SVROOT:$SVC:rw", "$SVROOT/views-search:$SVC/views-search:ro",
              "$DATA/stage_v:$DATA/stage_v:ro", "$DATA/bin/frontier_audit_h335", "H335_STAGEV_ROOT", "= 1310",
              "not the frozen read-only code copy"):
        assert s in txt


def test_no_code_dir_file_name_trips_the_guard(env):
    """write-lock hashes every file of the code dir (docs_sha256): no name may carry a DL-Val / sealed guard token."""
    toks = tuple(C._DEV_TOKENS) + tuple(C._SEALED_TOKENS)
    assert not [p.name for p in HERE.iterdir() if any(t in p.name.lower() for t in toks)]
    assert len(SVH.docs_manifest()) > 5  # every non-LOCK file is readable under the pre-unlock guard


def test_run_and_staging_scripts(env):
    for f in ("run_prep_v.sh", "run_audit_v.sh", "stage_v_xbabe0.sh", "build_stagev_view.sh"):
        assert subprocess.run(["bash", "-n", str(HERE / f)]).returncode == 0 and os.access(HERE / f, os.X_OK)
    st = (HERE / "stage_v_xbabe0.sh").read_text()
    assert "&& echo" not in st and st.count("|| die") >= 25 and "sha256sum --quiet -c" in st
    assert "verify-lock-files" in st and "events.xbabe0.jsonl" in st and "FIRST_OUTCOME_READ" in st
    assert "S6_FILES=(run_audit_v.sh)" in st and "docs_sha256" in st and "verify_in_docker --stage-v" in st
    bv = (HERE / "build_stagev_view.sh").read_text()
    assert "exit 5" in bv and bv.count("exit") == 1 and bv.count("|| die") >= 15  # only die() exits the script
    for pin in ("981960ad628f5029", "b9f1468a61aa4dc8", "30853cad78d86087", "0e812aa7ea4ce46a", "1310", "STAGE_V_STARTED"):
        assert pin in bv
    prep = (HERE / "run_prep_v.sh").read_text()
    assert "taskset -c 38-53,102-117" in prep and "nice -n 19" in prep and "MemoryMax=32G" in prep
    assert "H335_CODE=/home/ubuntu/h335-stage/code" in prep
    assert "MemoryMax=100G" in (HERE / "run_audit_v.sh").read_text()
