"""H317 Stage V driver tests (pytest). SYNTHETIC fixtures and TRAIN data only: no DL-Val byte is ever opened.

Every test redirects $H317_DATA (common_h317.DATA_ROOT), the LOCK receipt path and the Stage V root to a tmp dir and
restores common_h317's Stage V state afterwards, so no production LOCK / marker / receipt is touched.
Run inside the sandbox: taskset -c 44,45 nice -n 19 ./sandbox_h317.sh env H317_GUARD_LOG=/tmp/guard_test.log \
    python3 -m pytest -q -p no:cacheprovider test_stage_v_h317.py
"""

from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import errno  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common_h317 as C  # noqa: E402

C.install_guard()

import stage_v_h317 as SV  # noqa: E402
# Import-time paths (V.NEURAL_ROOT / VIEW_SEARCH_DIR) must be bound to the REAL $H317_DATA, as in production, before any
# test redirects common_h317.DATA_ROOT: the LOCK hashes the merged view file at exactly the path these modules read.
import crop_shuffle_synthetic_views as _V  # noqa: E402,F401

SEED = 317101


# ----------------------------------------------------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------------------------------------------------
@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolated DATA_ROOT / receipt / Stage V root; Stage V state restored after the test."""
    data = tmp_path / "data"
    (data / "stage_v").mkdir(parents=True)
    monkeypatch.setattr(C, "DATA_ROOT", data)
    monkeypatch.setattr(C, "LOCK_RECEIPT", tmp_path / "receipts" / "H317_LOCK_sha256.txt")
    monkeypatch.setattr(SV, "STAGEV_ROOT", tmp_path / "stagev")
    monkeypatch.setitem(C._STATE, "stage_v", False)
    monkeypatch.setenv("H317_GUARD_LOG", str(tmp_path / "guard.log"))
    return tmp_path


def _synthetic_lock(env: Path, runs: bool = True) -> dict:
    """A minimal LOCK whose code groups point at real source files + synthetic worker / model files in tmp."""
    worker = env / "bin" / "worker.bin"
    worker.parent.mkdir(parents=True, exist_ok=True)
    worker.write_bytes(b"SYNTHETIC WORKER")
    model = C.DATA_ROOT / "neural" / "lock" / "seed317101" / "model.bin"
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"SYNTHETIC MODEL")
    groups = {"source": ["common_h317.py", "stage_v_h317.py"], "worker": [str(worker)], "lock_model": [str(model)]}
    code = {k: C.sha256_file(SV.resolve_key(k)) for ks in groups.values() for k in ks}
    return {"schema": SV.SCHEMA_LOCK, "stage_v_runs": runs, "adoptions": {"H0.4": runs},
            "canonical_roots": SV.canonical_roots(), "code_groups": groups, "code_sha256": code}


def _write_lock(lock: dict, receipt: bool = True, receipt_sha: str | None = None) -> str:
    sha = SV.create_once(SV.lock_path(), SV.json_bytes(lock))
    if receipt:
        C.LOCK_RECEIPT.parent.mkdir(parents=True, exist_ok=True)
        C.LOCK_RECEIPT.write_text((receipt_sha or sha) + "\n")
    return sha


# ----------------------------------------------------------------------------------------------------------------------
# Unlock / enter
# ----------------------------------------------------------------------------------------------------------------------
def test_unlock_refused_without_receipt(env):
    _write_lock(_synthetic_lock(env), receipt=False)
    with pytest.raises((C.GuardViolation, FileNotFoundError, OSError)):
        SV.cmd_unlock(None)
    assert C._STATE["stage_v"] is False and not SV.started_marker().exists()


def test_unlock_refused_on_receipt_mismatch(env):
    _write_lock(_synthetic_lock(env), receipt_sha="0" * 64)
    with pytest.raises(C.GuardViolation):
        SV.cmd_unlock(None)
    assert C._STATE["stage_v"] is False and not SV.started_marker().exists()


def test_unlock_refused_after_code_change(env):
    lock = _synthetic_lock(env)
    _write_lock(lock)
    Path(lock["code_groups"]["worker"][0]).write_bytes(b"TAMPERED")
    with pytest.raises(C.GuardViolation, match="changed after LOCK"):
        SV.cmd_unlock(None)
    assert not SV.started_marker().exists()


def test_unlock_is_one_shot(env):
    sha = _write_lock(_synthetic_lock(env))
    assert SV.cmd_unlock(None) == 0
    assert C._STATE["stage_v"] is True
    m = SV.started_marker().read_text().split()
    assert m[1] == sha and oct(SV.started_marker().stat().st_mode & 0o777) == "0o444"
    with pytest.raises(SV.RetryRefused):
        SV.cmd_unlock(None)
    with pytest.raises(FileExistsError):  # the common_h317 marker itself is O_EXCL
        C.unlock_stage_v()


def test_unlock_skipped_without_adoption(env, capsys):
    _write_lock(_synthetic_lock(env, runs=False))
    assert SV.cmd_unlock(None) == 3
    assert "NO_ADOPTION_STAGE_V_SKIPPED" in capsys.readouterr().out
    assert not SV.started_marker().exists() and C._STATE["stage_v"] is False


def test_enter_requires_marker_bound_to_lock(env):
    lock = _synthetic_lock(env)
    sha = _write_lock(lock)
    with pytest.raises(SV.StageVError, match="not unlocked"):
        SV.enter_stage_v("audit")
    assert C._STATE["stage_v"] is False
    SV.create_once(SV.started_marker(), f"1 {'f' * 64}\n".encode())  # marker of another LOCK
    with pytest.raises(C.GuardViolation, match="does not bind"):
        SV.enter_stage_v("audit")
    assert C._STATE["stage_v"] is False
    os.chmod(SV.started_marker(), 0o644)
    SV.started_marker().write_text(f"1 {sha}\n")
    _, got, rep = SV.enter_stage_v("audit")
    assert got == sha and C._STATE["stage_v"] is True and rep["files_verified"] == 4
    assert rep["api"] == "common_h317.resume_stage_v"
    assert SV.started_marker().read_text().split()[1] == sha  # resume created nothing new


def test_enter_requires_every_lock_file_at_its_lock_path(env):
    """resume_stage_v semantics: every LOCK file must be present (any phase, any host) and unchanged."""
    lock = _synthetic_lock(env)
    never = env / "dope" / "never_created.pt"  # a LOCK entry whose file is not on this host
    lock["code_groups"]["dope"] = [str(never)]
    lock["code_sha256"][str(never)] = "0" * 64
    sha = _write_lock(lock)
    SV.create_once(SV.started_marker(), f"1 {sha}\n".encode())
    for phase in ("prep", "audit", "stats"):
        with pytest.raises(SV.StageVError, match="absent on this host"):
            SV.enter_stage_v(phase)
        assert C._STATE["stage_v"] is False


def test_enter_refuses_changed_lock_file(env):
    lock = _synthetic_lock(env)
    sha = _write_lock(lock)
    SV.create_once(SV.started_marker(), f"1 {sha}\n".encode())
    Path(lock["code_groups"]["lock_model"][0]).write_bytes(b"CHANGED")
    with pytest.raises(C.GuardViolation, match="changed after LOCK"):
        SV.enter_stage_v("audit")
    assert C._STATE["stage_v"] is False


def test_enter_requires_common_resume_api(env, monkeypatch):
    sha = _write_lock(_synthetic_lock(env))
    SV.create_once(SV.started_marker(), f"1 {sha}\n".encode())
    monkeypatch.delattr(C, "resume_stage_v")
    with pytest.raises(SV.StageVError, match="resume_stage_v is required"):
        SV.enter_stage_v("audit")
    assert C._STATE["stage_v"] is False


def test_production_stagev_root_refused_before_unlock(env):
    for p in ("/home/ubuntu/jope-data/h317-stagev/run/native.json", "/home/ubuntu/jope-data/h317-stagev/views-search",
              "/home/ubuntu/jope_plan/evidence/H258/argon-t1405/inputs/r882/best_dev_checkpoint.json"):
        with pytest.raises(C.GuardViolation):
            C.assert_allowed(p)  # path check only: nothing is opened


def test_stage_v_inputs_refused_before_enter(env):
    with pytest.raises(C.GuardViolation):
        SV.load_pool("0" * 64)
    with pytest.raises(C.GuardViolation):
        SV.encode_targets(None, [])


def test_resolve_key_matches_common(env):
    assert SV.resolve_key("common_h317.py") == C.HERE / "common_h317.py"
    assert SV.resolve_key("/abs/path/f") == Path("/abs/path/f")  # no remapping: exact LOCK paths on every host


# ----------------------------------------------------------------------------------------------------------------------
# No-retry / resumability
# ----------------------------------------------------------------------------------------------------------------------
def test_prep_admission_relaunch_once_and_no_retry_after_outcome(env):
    assert SV.admit_prep(False, False) == "first"
    with pytest.raises(SV.RetryRefused, match="relaunch"):
        SV.admit_prep(False, False)
    assert SV.admit_prep(False, True) == "relaunch"
    with pytest.raises(SV.RetryRefused, match="already used"):
        SV.admit_prep(False, True)
    SV.create_once(SV.run_dir() / SV.FIRST_OUTCOME, b"x\n")
    with pytest.raises(SV.RetryRefused, match="no retry"):
        SV.admit_prep(False, True)
    assert SV.admit_prep(True, False) == "resume"
    SV.create_once(SV.run_dir() / SV.REG_FAIL, b"x\n")
    with pytest.raises(SV.RetryRefused, match="REGRESSION_FAIL"):
        SV.admit_prep(True, False)


def test_relaunch_refused_if_any_outcome_record_exists(env):
    SV.admit_prep(False, False)
    SV.JsonlLog(SV.run_dir() / "audit_units.s0of1.jsonl").append({"type": "unit"})
    with pytest.raises(SV.RetryRefused, match="outcome records exist"):
        SV.relaunch_once()
    assert not (SV.run_dir() / SV.RELAUNCH).exists()


def _fake_unit(name, cid, seed, k):
    v = (C.stream_seed("fakeunit", name, cid, seed) % 1000) / 1000.0
    live = 76 if name in ("H317_LOCKED", "EVOA_MATCHED") else (k if name == "SCREEN_TOPK_PICK" else 8)
    return {"value": v, "value_m2": v, "invalid": False, "live_evals": live, "raw": {"status": "completed"},
            "audit_ledger": {"outputs": 1, "audit_requests": 1}, "engine": {"physical_runs": 3, "memo_hits": 1}}


def _prep_done():
    SV.create_once(SV.run_dir() / SV.PREP_DONE, SV.json_bytes({"regression_pass": True}))


def test_audit_resume_never_recomputes_and_refuses_without_flag(env):
    _prep_done()
    cids = [f"{i:02d}" + "a" * 62 for i in range(3)]
    calls = []
    boom = {"armed": True}

    def ex(name, cid, seed, k):
        calls.append((cid, seed, name))
        if boom["armed"] and cid == cids[1] and seed == 317102 and name == "H257_REPLICA":
            boom["armed"] = False
            raise OSError(errno.ENOSPC, "synthetic disk full (host failure: no record, resumable)")
        return _fake_unit(name, cid, seed, k)

    assert SV.admit_audit((0, 1), False) == "first"
    with pytest.raises(SV.StageVError, match="failed"):
        SV.run_audit_loop(cids, C.SEARCH_SEEDS, [ex], (0, 1), "L" * 64)
    n_first = len(calls)
    with pytest.raises(SV.RetryRefused, match="no retry"):
        SV.admit_audit((0, 1), False)
    assert SV.admit_audit((0, 1), True) == "resume"
    res = SV.run_audit_loop(cids, C.SEARCH_SEEDS, [ex], (0, 1), "L" * 64)
    done, started, _ = SV.all_unit_records()
    total = len(cids) * 2 * len(SV.PIPELINES)
    assert len(done) == total and not started
    # every recorded unit ran exactly once; only the interrupted unit ran twice
    assert len(calls) == total + 1 and n_first + res["executed_units"] == len(calls)
    rerun = [k for k, r in done.items() if r["rerun_after_interruption"]]
    assert rerun == [(cids[1], 317102, "H257_REPLICA")]
    for c in cids:
        for s in C.SEARCH_SEEDS:
            assert done[(c, s, "SCREEN_TOPK_PICK")]["k"] == done[(c, s, "H317_LOCKED")]["live_evals"]


def test_duplicate_unit_record_is_refused(env):
    log = SV.JsonlLog(SV.run_dir() / "audit_units.s0of1.jsonl")
    rec = {"type": "unit", "cid": "a" * 64, "seed": 317101, "pipeline": "BANK_NN1", "value": 0.1}
    log.append(rec)
    SV.JsonlLog(SV.run_dir() / "audit_units.s1of2.jsonl").append(rec)
    with pytest.raises(SV.StageVError, match="duplicated"):
        SV.all_unit_records()


def test_torn_jsonl_tail_is_tolerated(env):
    p = SV.run_dir() / "audit_units.s0of1.jsonl"
    p.parent.mkdir(parents=True)
    p.write_text('{"type": "unit_start", "cid": "x", "seed": 1, "pipeline": "BANK_NN1"}\n{"type": "un')
    SV.JsonlLog(p).append({"type": "unit", "cid": "y", "seed": 1, "pipeline": "BANK_NN1", "value": 0.0})
    recs, bad = SV.read_jsonl(p)
    assert bad == 1 and [r["cid"] for r in recs] == ["x", "y"]


def test_live_match_violations_are_values_not_exceptions():
    a = {"live_evals": 76}
    assert SV.matching_violations("EVOA_MATCHED", {"H317_LOCKED": a, "EVOA_MATCHED": {"live_evals": 78}}) == []
    assert SV.matching_violations("EVOA_MATCHED", {"H317_LOCKED": a, "EVOA_MATCHED": {"live_evals": 79}}) == [
        "evoa_live_gap=3>2"]
    assert SV.matching_violations("SCREEN_TOPK_PICK", {"H317_LOCKED": a, "SCREEN_TOPK_PICK": {"live_evals": 75}}) == [
        "screen_k_mismatch=75!=76"]
    failed = SV.failed_unit("x")
    assert SV.matching_violations("EVOA_MATCHED", {"H317_LOCKED": failed, "EVOA_MATCHED": {"live_evals": 1}}) == []


def test_unit_failures_after_outcome_read_are_recorded_substitutions(env):
    """HIGH-1: deterministic in-pipeline exceptions / non-finite values / budget mismatches never strand the audit:
    each becomes a recorded unit (value 0.0 = SEARCH-mean predictor, invalid) or a recorded violation flag."""
    _prep_done()
    cids = [f"{i:02d}" + "a" * 62 for i in range(2)]

    def ex(name, cid, seed, k):
        if cid == cids[0] and seed == 317101 and name == "H317_LOCKED":
            raise AssertionError("assert len(props) == n_prop (synthetic)")
        if cid == cids[0] and seed == 317102 and name == "BANK_NN1":
            return {**_fake_unit(name, cid, seed, k), "value": float("nan")}
        if cid == cids[1] and seed == 317101 and name == "EVOA_MATCHED":
            return {**_fake_unit(name, cid, seed, k), "live_evals": 80}  # |80 - 76| > 2
        if cid == cids[1] and seed == 317102 and name == "SCREEN_TOPK_PICK":
            raise ValueError("fewer than K unique candidates (synthetic)")
        return _fake_unit(name, cid, seed, k)

    SV.admit_audit((0, 1), False)
    res = SV.run_audit_loop(cids, C.SEARCH_SEEDS, [ex], (0, 1), "L" * 64)
    done, started, _ = SV.all_unit_records()
    assert res["executed_units"] == len(done) == 2 * 2 * len(SV.PIPELINES) and not started
    f = done[(cids[0], 317101, "H317_LOCKED")]
    assert f["status"] == "failed" and f["value"] == 0.0 and f["invalid"] and "n_prop" in f["error"]
    k_undef = done[(cids[0], 317101, "SCREEN_TOPK_PICK")]  # K undefined -> recorded failure, executor not called
    assert k_undef["status"] == "failed" and "K undefined" in k_undef["error"] and k_undef["k"] is None
    assert done[(cids[0], 317102, "BANK_NN1")]["status"] == "failed"  # non-finite value
    assert done[(cids[1], 317101, "EVOA_MATCHED")]["live_match_violation"] == ["evoa_live_gap=4>2"]
    assert done[(cids[1], 317101, "EVOA_MATCHED")]["status"] == "completed"
    assert done[(cids[1], 317102, "SCREEN_TOPK_PICK")]["status"] == "failed"
    vals, info = SV.unit_panel(done, cids, C.SEARCH_SEEDS)
    assert info["pipelines"]["H317_LOCKED"]["failed"] == 1 and info["pipelines"]["SCREEN_TOPK_PICK"]["failed"] == 2
    assert info["pipelines"]["EVOA_MATCHED"]["live_match_violations"] == 1
    assert vals["H317_LOCKED"][cids[0]] == pytest.approx(0.5 * (0.0 + done[(cids[0], 317102, "H317_LOCKED")]["value"]))


def test_integrity_and_host_errors_still_stop_without_a_record(env):
    _prep_done()
    cid = "0f" + "a" * 62
    for exc in (C.GuardViolation("synthetic guard refusal"), MemoryError()):
        def ex(name, c, seed, k, exc=exc):
            if name == "H257_REPLICA":
                raise exc
            return _fake_unit(name, c, seed, k)

        with pytest.raises(SV.StageVError, match="failed"):
            SV.run_audit_loop([cid], (317101,), [ex], (0, 1), "L" * 64)
        done, started, _ = SV.all_unit_records()
        assert (cid, 317101, "H257_REPLICA") not in done and (cid, 317101, "H257_REPLICA") in started


def test_shard_run_lock_one_process_per_shard(env):
    with SV.ShardRunLock((0, 2)):
        with pytest.raises(SV.RetryRefused, match="already running"):
            SV.ShardRunLock((0, 2)).__enter__()
        with SV.ShardRunLock((1, 2)):  # another shard is independent
            pass
    with SV.ShardRunLock((0, 2)):  # holder gone: the stale file is taken over (never deleted)
        pass
    ev = [e["event"] for e in SV.read_jsonl(SV.run_dir() / "events.jsonl")[0]]
    assert ev.count("audit_running_stale_taken_over") == 1
    assert (SV.run_dir() / "AUDIT_RUNNING.s0of2").exists()


def test_compact_unit_requires_finite_value():
    with pytest.raises(SV.StageVError):
        SV.compact_unit({"value": float("nan"), "live_evals": 1})
    out = SV.compact_unit({"value": 0.2, "live_evals": 3, "invalid": False, "spec": {"search": {"utility": 0.31}},
                           "raw": {"status": "completed", "members": [1, 2]}})
    assert out["search_utility"] == 0.31 and "members" not in out["raw"]


# ----------------------------------------------------------------------------------------------------------------------
# R882 regression gate and TOP4_PICK (H258 runner logic)
# ----------------------------------------------------------------------------------------------------------------------
def _synthetic_native(n_ctx=5, m=40, seed=7):
    rng = np.random.default_rng(seed)
    cids = [f"{i:02d}" + "b" * 62 for i in range(n_ctx)]
    pools, costs, ref = {}, {}, {}
    for c in cids:
        fit = rng.normal(-1.2, 0.3, m)
        gate = rng.random(m) > 0.2
        pools[c] = {"fitness": fit, "gate": gate, "utility": SV.native_utility_h258(fit, gate)}
        costs[c] = rng.normal(0, 1, m).astype(np.float32)
        ref[c] = float(pools[c]["utility"][int(np.argmin(costs[c]))])
    return cids, pools, costs, ref


def test_regression_gate_pass_and_bit_exact_fail():
    cids, pools, costs, ref = _synthetic_native()
    mean = float(np.mean([ref[c] for c in cids]))
    ok = SV.r882_regression(cids, ref, costs, pools, cids, expected_mean=mean, n_expected=len(cids))
    assert ok["pass"] and ok["bit_exact_matches"] == len(cids) and ok["status"] == "PASS"
    bad = dict(ref)
    bad[cids[2]] = float(np.nextafter(bad[cids[2]], np.inf))  # one ulp
    r = SV.r882_regression(cids, bad, costs, pools, cids, expected_mean=mean, n_expected=len(cids))
    assert not r["pass"] and r["status"] == "REGRESSION_FAIL" and r["bit_exact_matches"] == len(cids) - 1
    r = SV.r882_regression(cids, ref, costs, pools, cids, expected_mean=mean + 2e-12, n_expected=len(cids))
    assert not r["pass"]
    r = SV.r882_regression(cids, ref, costs, pools, cids[:-1] + ["z" * 64], expected_mean=mean,
                           n_expected=len(cids))
    assert not r["pass"] and not r["reference_set_equals_targets"]


def test_regression_uses_float32_argmin_like_h258():
    fit = np.array([-1.0, -1.5, -0.5])
    gate = np.array([True, True, True])
    pools = {"c": {"fitness": fit, "gate": gate, "utility": SV.native_utility_h258(fit, gate)}}
    costs64 = np.array([0.30000001, 0.30000000, 0.9])  # float64 argmin = 1; equal in float32 -> argmin = 0
    r = SV.r882_regression(["c"], {"c": 1.0}, {"c": costs64.astype(np.float32)}, pools, ["c"], expected_mean=1.0,
                           n_expected=1)
    assert r["per_context"]["c"]["pick"] == 0 and r["pass"]


def test_regression_selection_crosscheck_is_reported_not_gating():
    cids, pools, costs, ref = _synthetic_native(n_ctx=2)
    mean = float(np.mean([ref[c] for c in cids]))
    ctxs = [{"cid": cids[0], "selected_choice": int(np.argmin(costs[cids[0]]))},
            {"cid": cids[1], "selected_choice": (int(np.argmin(costs[cids[1]])) + 1) % 40}]
    r = SV.r882_regression(cids, ref, costs, pools, cids, ref_contexts=ctxs, expected_mean=mean, n_expected=2)
    assert r["pass"] and r["selection_mismatches"] == 1


def test_reference_values_alignment():
    ref = {"evaluation_contract": {"contexts": [{"cid": "b"}, {"cid": "a"}]}, "utility": {"per_context": [2.0, 1.0]}}
    order, vals = SV.reference_values(ref)
    assert order == ["b", "a"] and vals == {"a": 1.0, "b": 2.0}
    ref["utility"]["per_context"] = [1.0]
    with pytest.raises(SV.RegressionFail):
        SV.reference_values(ref)


def test_top4_pick_h258_definition():
    rcost = np.array([0.10, 0.11, 0.12, 0.13, 5.0, 6.0], np.float32)  # small top gap -> gated
    fit = np.array([-1.0, -1.3, -1.3, -2.0, -9.0, -9.0])
    gate = np.array([True, True, True, False, True, True])
    u = SV.native_utility_h258(fit, gate)
    r = SV.top4_pick(rcost, fit, gate, u)
    assert r["gated"] and r["reveals"] == 4
    # revealed gated costs: [-1.0, -1.3, -1.3, 2.0] -> argmin, tie broken in rcost order -> index 1
    assert r["pick"] == 1 and r["u"] == u[1] and r["r882_pick"] == 0
    wide = np.array([0.0, 3.0, 3.1, 3.2, 3.3, 3.4], np.float32)  # big top gap -> 0 reveals, R882 pick
    r = SV.top4_pick(wide, fit, gate, u)
    assert not r["gated"] and r["pick"] == 0 and r["reveals"] == 0 and r["pick_ungated"] == 1
    flat = np.zeros(6, np.float32)
    assert SV.r882_top_zgap(flat) == 999.0 and not SV.top4_pick(flat, fit, gate, u)["gated"]


# ----------------------------------------------------------------------------------------------------------------------
# Statistics: intersection-union, readout, invalid rule, Elo
# ----------------------------------------------------------------------------------------------------------------------
def _panel(gain_evoa, gain_screen, n_fam=19, per=5, noise=0.004, seed=3):
    rng = np.random.default_rng(seed)
    fam_of, base = {}, {}
    for f in range(n_fam):
        for j in range(per):
            c = f"{f:02d}{j:02d}" + "c" * 60
            fam_of[c] = f"fam-{f:02d}"
            base[c] = rng.normal(0.3, 0.1)
    evoa = dict(base)
    screen = {c: v + rng.normal(0, 0.002) for c, v in base.items()}
    locked = {c: v + gain_evoa + rng.normal(0, noise) for c, v in base.items()}
    if gain_screen is not None:
        screen = {c: locked[c] - gain_screen + rng.normal(0, noise) for c in base}
    return fam_of, locked, evoa, screen


def test_intersection_union_requires_both_controls():
    fam_of, locked, evoa, screen = _panel(0.05, 0.05)
    prim = {"EVOA_MATCHED": SV.contract(locked, evoa, fam_of), "SCREEN_TOPK_PICK": SV.contract(locked, screen, fam_of)}
    assert all(p["gate"]["pass"] for p in prim.values())
    rd = SV.readout(prim, {"H317_LOCKED": 0, "EVOA_MATCHED": 0, "SCREEN_TOPK_PICK": 0}, 0.01)
    assert rd["verdict"] == "ADVANCE" and rd["primary_iu_pass"]
    fam_of, locked, evoa, screen = _panel(0.05, 0.0)  # beats EVOA only
    prim = {"EVOA_MATCHED": SV.contract(locked, evoa, fam_of), "SCREEN_TOPK_PICK": SV.contract(locked, screen, fam_of)}
    assert prim["EVOA_MATCHED"]["gate"]["pass"] and not prim["SCREEN_TOPK_PICK"]["gate"]["pass"]
    rd = SV.readout(prim, {"H317_LOCKED": 0, "EVOA_MATCHED": 0, "SCREEN_TOPK_PICK": 0}, 0.01)
    assert rd["verdict"] != "ADVANCE" and not rd["primary_iu_pass"]


def _summ(lo, hi, gate):
    return {"ci95": [lo, hi], "gate": {"pass": gate}}


def test_three_way_readout():
    inv0 = {"H317_LOCKED": 0, "EVOA_MATCHED": 0, "SCREEN_TOPK_PICK": 0}
    both = lambda lo, hi, g: {c: _summ(lo, hi, g) for c in SV.CONTROLS}  # noqa: E731
    assert SV.readout(both(0.01, 0.05, True), inv0, 0.002)["verdict"] == "ADVANCE"
    assert SV.readout(both(0.01, 0.05, True), inv0, 0.0)["verdict"] != "ADVANCE"  # ties never advance
    p = both(-0.01, 0.01, False)
    p["SCREEN_TOPK_PICK"] = _summ(-0.06, -0.001, False)
    assert SV.readout(p, inv0, 0.01)["verdict"] == "NO_ADVANCE-INFERIOR"
    assert SV.readout(both(-0.02, 0.025, False), inv0, 0.01)["verdict"] == "NO_ADVANCE-UNDETECTED"
    p = both(-0.02, 0.025, False)
    p["EVOA_MATCHED"] = _summ(-0.01, 0.04, False)  # one CI leaves the band
    assert SV.readout(p, inv0, 0.01)["verdict"] == "INCONCLUSIVE"


def test_invalid_rule():
    prim = {c: _summ(0.01, 0.05, True) for c in SV.CONTROLS}
    ok = SV.readout(prim, {"H317_LOCKED": 3, "EVOA_MATCHED": 2, "SCREEN_TOPK_PICK": 5}, 0.01)
    assert ok["verdict"] == "ADVANCE" and all(ok["invalid_rule"].values())
    bad = SV.readout(prim, {"H317_LOCKED": 4, "EVOA_MATCHED": 2, "SCREEN_TOPK_PICK": 5}, 0.01)
    assert bad["verdict"] != "ADVANCE" and bad["invalid_rule"] == {"EVOA_MATCHED": False, "SCREEN_TOPK_PICK": True}
    assert "invalid_rule" in bad["advance_failed_conditions"]


def test_elo_sanity():
    fam_of = {f"{i:02d}" + "d" * 62: f"fam{i % 4}" for i in range(16)}
    base = {c: 0.1 * i for i, c in enumerate(fam_of)}
    vals = {"REF": base, "BETTER": {c: v + 0.05 for c, v in base.items()},
            "WORSE": {c: v - 0.05 for c, v in base.items()}, "TWIN": dict(base)}
    e = SV.elo_panel(vals, fam_of, "REF", n_boot=200)["elo"]
    assert e["REF"]["elo"] == pytest.approx(1500.0, abs=1e-9)
    assert e["BETTER"]["elo"] > e["REF"]["elo"] > e["WORSE"]["elo"]
    assert e["TWIN"]["elo"] == pytest.approx(e["REF"]["elo"], abs=1e-6)
    # 2 arms, 4 families, A wins 3 / loses 1: the MM fixed point is s_A / s_B = (w + prior) / (l + prior)
    fam2 = {f"{i:02d}" + "d" * 62: f"fam{i}" for i in range(4)}
    a_vals = {c: (1.0 if i < 3 else -1.0) for i, c in enumerate(fam2)}
    e2 = SV.elo_panel({"B": {c: 0.0 for c in fam2}, "A": a_vals}, fam2, "B", n_boot=50)["elo"]
    want = 1500.0 + 400.0 * np.log10((3 + SV.ELO_PRIOR) / (1 + SV.ELO_PRIOR))
    assert e2["A"]["elo"] == pytest.approx(want, abs=1e-9) and e2["B"]["elo"] == pytest.approx(1500.0)
    W = SV.family_wins_tensor(np.array([[1.0, 1.0 + 5e-5], [0.0, 1.0]]), ["a", "a"])
    assert W.shape == (1, 2, 2) and W[0, 0, 1] == pytest.approx(0.25)  # tie (within 1e-4) + loss


def test_compute_stats_full_panel_and_size():
    fam_of, locked, evoa, screen = _panel(0.06, 0.06, n_fam=19, per=5)
    cids = sorted(fam_of)[:91]
    fam_of = {c: fam_of[c] for c in cids}
    rng = np.random.default_rng(11)
    done = {}
    for c in cids:
        for s in C.SEARCH_SEEDS:
            for p in SV.PIPELINES:
                v = {"H317_LOCKED": locked[c], "EVOA_MATCHED": evoa[c], "SCREEN_TOPK_PICK": screen[c]}.get(
                    p, evoa[c] - 0.02)
                done[(c, s, p)] = {"value": v + rng.normal(0, 1e-3), "invalid": False, "live_evals": 76,
                                   "search_utility": 0.4, "engine": {"physical_runs": 50, "memo_hits": 10},
                                   "wall_s": 10.0, "rerun_after_interruption": False}
    native = {"regression": {"status": "PASS", "bit_exact_matches": 91, "contexts": 91,
                             "mean_target_order": 1.1377941898164525, "abs_diff": 0.0, "mean_bit_equal": True,
                             "selection_mismatches": 0, "per_context": {c: {"u": 1.1} for c in cids}},
              "vnative": {c: {"u": 1.1 + rng.normal(0, 0.01), "per_seed": {"317101": {"top1": 3}, "317102":
                                                                             {"top1": 3}}} for c in cids},
              "top4": {c: {"u": 1.2, "u_ungated": 1.25, "gated": True} for c in cids}}
    dd = {m: {"max_jaccard": 0.1, "max_overlap": 0.2, "n_jaccard_gt_0p5": 0, "flagged": []} for m in SV.DEDUPE_MODES}
    m = SV.compute_stats(done, native, fam_of, cids, dedupe_summary=dd)
    assert m["verdict"] == "ADVANCE" and m["final"]
    assert m["bytes_compact"] < SV.METRICS_MAX_BYTES
    assert m["elo_audit"]["elo"]["EVOA_MATCHED"]["elo"] == pytest.approx(1500.0)
    assert m["v_native"]["elo"]["elo"]["R882"]["elo"] == pytest.approx(1500.0)
    assert m["regression"]["mean_target_order"] == 1.1377941898164525  # full precision kept
    del done[(cids[0], 317102, "BANK_NN1")]
    with pytest.raises(SV.StageVError, match="missing"):
        SV.compute_stats(done, native, fam_of, cids)
    part = SV.compute_stats(done, native, fam_of, cids, partial=True)
    assert part["final"] is False


# ----------------------------------------------------------------------------------------------------------------------
# Dedupe fingerprints
# ----------------------------------------------------------------------------------------------------------------------
def test_dedupe_detects_subsets_scaling_and_permutation():
    rng = np.random.default_rng(5)
    t0 = rng.normal(size=(200, 6))
    t1 = rng.normal(size=(150, 4))
    train = [("t0" + "e" * 62, t0), ("t1" + "e" * 62, t1)]
    fam_of = {"t0" + "e" * 62: "fam-t0", "t1" + "e" * 62: "fam-t1"}
    targets = {
        "subset": t0[rng.permutation(200)[:120]],  # row subset in shuffled order
        "scaled": t0 * np.array([2.0, 3, 4, 5, 6, 7]) + 1.5,  # affine per column, all rows
        "permuted": t0[:, ::-1],  # column order reversed
        "fresh": rng.normal(size=(80, 6)),
    }
    rep = SV.dedupe_report(targets, iter(train), fam_of)["per_context"]
    assert rep["subset"]["raw_sorted"]["max_overlap"] == 1.0
    assert rep["subset"]["raw_sorted"]["max_jaccard"] == pytest.approx(120 / 200)
    assert rep["scaled"]["std"]["max_jaccard"] == 1.0 and rep["scaled"]["raw_sorted"]["max_jaccard"] == 0.0
    assert rep["permuted"]["raw_sorted"]["max_jaccard"] == 1.0
    assert rep["fresh"]["std"]["max_overlap"] == 0.0 and rep["fresh"]["raw_sorted"]["max_overlap"] == 0.0
    assert rep["subset"]["raw_sorted"]["overlap_train_family"] == "fam-t0"
    summ = SV.dedupe_report(targets, iter(train), fam_of)["summary"]
    assert summ["raw_sorted"]["n_overlap_gt_0p5"] == 2 and summ["std"]["n_jaccard_gt_0p5"] == 1


def test_row_fingerprints_nan_and_signed_zero():
    a = np.array([[0.0, np.nan, 1.0]])
    b = np.array([[-0.0, np.nan, 1.0]])
    assert SV.row_fingerprints(a, "raw_sorted")[0] == SV.row_fingerprints(b, "raw_sorted")[0]
    assert SV.row_fingerprints(np.zeros((0, 3)), "std").size == 0


# ----------------------------------------------------------------------------------------------------------------------
# LOCK writer -> unlock -> enter -> LOCK screener / V-native (synthetic models trained on nothing; TRAIN manifests)
# ----------------------------------------------------------------------------------------------------------------------
def _synthetic_lock_fit(lock_dir: Path, arm: dict, cids=None) -> None:
    import torch

    import multiscale_axial_bridge as MB
    import rapid_hypothesis_lab as R

    rows = C.train_rows()
    keep = set(cids) if cids is not None else {r["context_sha256"] for r in rows}
    tc = sorted(keep)
    tf = sorted({r["family_id"] for r in rows if r["context_sha256"] in keep})
    shas = {}
    for s in SV.LOCK_TRAIN_SEEDS:
        torch.manual_seed(s)
        m = MB.build_model(arm["scale"])
        d = lock_dir / f"seed{s}"
        d.mkdir(parents=True)
        torch.save({"state_dict": m.state_dict(), "scale": arm["scale"], "arm": arm, "seed": s, "fold": -1,
                    "train_cids": tc, "train_families": tf, "train_cids_sha256": R._list_sha(tc),
                    "train_families_sha256": R._list_sha(tf)}, d / "model.pt")
        shas[str(s)] = C.sha256_file(d / "model.pt")
    (lock_dir / "lock_models.json").write_text(json.dumps(
        {"arm": arm, "budget": SV.LOCK_BUDGET, "model_sha256": shas, "dope_unchanged": True,
         "dope_after": {"composite_sha256": C.DOPE_SHA256}}))


def _decisions(tmp: Path, w1="B", w2="S", w3="opt1", gp=("C", "evoB", "CV3@5")) -> dict:
    import rapid_hypothesis_lab as R

    out = {}
    for e, w in zip(SV.NEURAL_EXPERIMENTS, (w1, w2, w3), strict=True):
        p = tmp / f"decide_{e}.json"
        p.write_text(json.dumps({"experiment": e, "baseline": R.PLANNED[e]["baseline"], "winner": w,
                                 "challengers": {w: {"family_mean": 1.0}}}))
        out[e] = p
    g = tmp / "decisions.json"
    g.write_text(json.dumps({k: {"used": v, "adopted": v, "pinned_utc": 0}
                             for k, v in zip(("H0.4", "H0.5", "H0.6"), gp, strict=True)}))
    out["gp"] = g
    return out


@pytest.fixture
def locked(env):
    dec = _decisions(env)
    arm = SV.locked_arm("B", "S", "opt1")
    lock_dir = C.DATA_ROOT / "neural" / "lock"
    _synthetic_lock_fit(lock_dir, arm)
    lock = SV.build_lock(dec["H0.1"], dec["H0.2"], dec["H0.3"], dec["gp"], lock_dir)
    sha = _write_lock(lock)
    return lock, sha, dec, lock_dir


def test_write_lock_contents_and_create_once(locked):
    lock, sha, dec, _ = locked
    assert lock["stage_v_runs"] and lock["adoptions"]["H0.1"] and lock["adoptions"]["H0.4"]
    assert lock["gp"]["locked_config"] == {"scheme": "C", "arm": "evoB", "generations": 10, "stack": "CV3@5"}
    assert lock["neural"]["locked_arm"]["views"] == "B16" and lock["neural"]["budget"] == SV.LOCK_BUDGET
    src = set(lock["code_groups"]["source"])
    assert {"stage_v_h317.py", "common_h317.py", "gp_lab.py", "PREREG.md", "rapid_hypothesis_lab.py"} <= src
    assert lock["gp"]["workers"][str(C.WORKER_BIN.resolve())] == SV.WORKER_SHA256
    assert len(lock["code_groups"]["lock_model"]) == 3 and lock["stage_v"]["search_seeds"] == [317101, 317102]
    inp = lock["code_groups"]["inputs"]  # MEDIUM-3: hashed at the paths the code reads, verified on every host
    assert inp == [str(C.RES01 / "champions.json"), str(C.RES01 / "winners.json"), str(SV.merged_views_path()),
                   str(SV.train_inputs_manifest_path())]
    assert lock["code_sha256"][str(SV.merged_views_path())] == SV.VIEWS_MERGED_SHA256
    assert "fold_manifest.json" in src and lock["train_inputs"]["n_tables"] == 1219
    man = json.loads(SV.train_inputs_manifest_path().read_text())
    assert len(man["tables"]) == 1219 and man["selected_allowed_by_phase"]["prep"] == ["original"]
    assert SV.C.sha256_file(SV.lock_path()) == sha
    with pytest.raises(FileExistsError):
        SV.create_once(SV.lock_path(), SV.json_bytes(lock))


def test_write_lock_refuses_mismatched_lock_model(env):
    dec = _decisions(env, w1="C")  # composed winner = H0.1-C, but the models were fit for H0.1-B
    lock_dir = C.DATA_ROOT / "neural" / "lock"
    _synthetic_lock_fit(lock_dir, SV.locked_arm("B", "S", "opt1"))
    with pytest.raises(SV.StageVError, match="composed Gate H0 winner"):
        SV.build_lock(dec["H0.1"], dec["H0.2"], dec["H0.3"], dec["gp"], lock_dir)


def test_no_adoption_lock_skips_stage_v(env):
    dec = _decisions(env, w1="A", gp=("A", "evoA", "single"))
    arm = SV.locked_arm("A", "S", "opt1")
    lock_dir = C.DATA_ROOT / "neural" / "lock"
    import rapid_hypothesis_lab as R
    import torch

    import multiscale_axial_bridge as MB

    sub = sorted(R.arm_a_subset(C.train_rows()))
    fams = sorted({r["family_id"] for r in C.train_rows() if r["context_sha256"] in set(sub)})
    shas = {}
    for s in SV.LOCK_TRAIN_SEEDS:
        d = lock_dir / f"seed{s}"
        d.mkdir(parents=True)
        torch.save({"state_dict": MB.build_model("S").state_dict(), "scale": "S", "fold": -1, "train_cids": sub,
                    "train_families": fams, "train_cids_sha256": R._list_sha(sub),
                    "train_families_sha256": R._list_sha(fams)}, d / "model.pt")
        shas[str(s)] = C.sha256_file(d / "model.pt")
    (lock_dir / "lock_models.json").write_text(json.dumps({"arm": arm, "budget": SV.LOCK_BUDGET,
                                                           "model_sha256": shas, "dope_unchanged": True}))
    lock = SV.build_lock(dec["H0.1"], dec["H0.2"], dec["H0.3"], dec["gp"], lock_dir)
    assert lock["stage_v_runs"] is False and lock["verdict_if_skipped"] == "NO_ADOPTION_STAGE_V_SKIPPED"
    _write_lock(lock)
    assert SV.cmd_unlock(None) == 3 and not SV.started_marker().exists()


def test_lock_unlock_enter_and_lock_screener(locked):
    lock, sha, _, _ = locked
    assert SV.cmd_unlock(None) == 0  # real common_h317.unlock_stage_v over the real code hashes
    C._STATE["stage_v"] = False  # a later process enters via the marker
    lk, got, rep = SV.enter_stage_v("audit")
    assert got == sha and C._STATE["stage_v"] is True and rep["files_verified"] == len(lock["code_sha256"])
    assert rep["api"] == "common_h317.resume_stage_v" and rep["train_tables_verified"] == 1219
    assert rep["selected_sha256"] == C.SELECTED_REDACTED_SHA256
    with pytest.raises(C.GuardViolation, match="not allowed in this phase"):
        SV.verify_train_inputs(lk, "prep")  # prep needs the ORIGINAL selected.json in the V view
    import rapid_hypothesis_lab as R

    lm = SV.LockModels(lk)
    assert sorted(lm.per_seed) == list(SV.LOCK_TRAIN_SEEDS)
    train0 = C.train_rows()[0]
    with pytest.raises(R.ScreenerLeak):
        lm.assert_target(train0["context_sha256"], train0["family_id"])
    lm.assert_target("f" * 64, "synthetic-family-not-in-train")  # DL-Val-like target: allowed (trivially)


def test_vnative_matches_direct_model_forward(locked):
    import torch

    import multiscale_axial_bridge as MB

    lock, *_ = locked
    lm = SV.LockModels(lock)
    rng = np.random.default_rng(9)
    m = 60
    prog = np.abs(rng.normal(size=(m, 412))).astype(np.float32)
    view0 = rng.normal(size=MB.CTX_DIM).astype(np.float32)
    util = rng.normal(1.0, 0.2, m)
    out = SV.vnative_context(lm, "f" * 64, "synthetic-family", view0, prog, util)
    f412, p16, s24 = SV.candidate_inputs(prog)
    cand = MB.candidate_vector(f412, p16, s24).astype(np.float16).astype(np.float32)
    for s, model in lm.models.items():
        with torch.no_grad():
            p = model(torch.as_tensor(view0).view(1, 1, MB.CTX_DIM), torch.as_tensor(cand).unsqueeze(0))[0, 0]
        top = int(np.argsort(-p[:, 0].numpy(), kind="stable")[0])
        assert out["per_seed"][str(s)]["top1"] == top and out["per_seed"][str(s)]["u"] == util[top]
    assert out["u"] == pytest.approx(np.mean([v["u"] for v in out["per_seed"].values()]))
    with pytest.raises(Exception, match="training set"):
        SV.vnative_context(lm, C.train_rows()[0]["context_sha256"], C.train_rows()[0]["family_id"], view0, prog, util)


# ----------------------------------------------------------------------------------------------------------------------
# GP adapter -> gp_lab.run_pipeline (fake worker, non-pilot TRAIN context, injected test screener / coder)
# ----------------------------------------------------------------------------------------------------------------------
def _fake_runner():
    def rmse(a, b):
        return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

    def run(ctx, kind, cand, core, slot):
        q = 0.25 + 0.9 * ((C.stream_seed("q", cand.sha) % 10_000) / 10_000.0)
        rng = np.random.default_rng(C.stream_seed("noise", cand.sha, kind))
        if kind == "audit":
            pa = ctx.ya + q * ctx.ba * rng.standard_normal(len(ctx.ya))
            recs = [{"predictions": [0.0], "targets": [0.0], "metrics": {"rmse": 1.0},
                     "baseline_metrics": {"rmse": 1.0}},
                    {"predictions": pa.tolist(), "targets": ctx.ya.tolist(), "metrics": {"rmse": rmse(pa, ctx.ya)},
                     "baseline_metrics": {"rmse": ctx.ba}}]
        else:
            y0, y1 = ctx.y0, ctx.y1
            b0, b1 = rmse(y0, np.full(len(y0), y1.mean())), rmse(y1, np.full(len(y1), y0.mean()))
            p0, p1 = y0 + q * b0 * rng.standard_normal(len(y0)), y1 + q * b1 * rng.standard_normal(len(y1))
            recs = [{"predictions": p0.tolist(), "targets": y0.tolist(), "metrics": {"rmse": rmse(p0, y0)},
                     "baseline_metrics": {"rmse": b0}},
                    {"predictions": p1.tolist(), "targets": y1.tolist(), "metrics": {"rmse": rmse(p1, y1)},
                     "baseline_metrics": {"rmse": b1}}]
        return {"status": "completed", "inner_cv": recs}, 0.001, ""

    return run


class _NoLeakLM:
    def assert_target(self, cid, fam):
        return None


def _nonpilot_cid() -> str:
    rows = C.train_rows()
    pilot = set(C.pilot_shard(rows, C.family_folds(rows)))
    return sorted((r["context_sha256"] for r in rows if r["context_sha256"] not in pilot),
                  key=lambda c: C.stream_seed("h317v-test", c))[0]


@pytest.mark.skipif(os.environ.get("H317_SV_FAST") == "1", reason="fast mode")
def test_gp_adapter_runs_all_pipelines_through_run_pipeline(env):
    import distributed_10gen_island_gp as D
    import screener_h317 as S

    class HashCoder(D.ContextCoder):
        name = "hash_test"

        def _c(self, cid):
            r = np.random.default_rng(C.stream_seed("hc", cid))
            return {"lm12": r.normal(size=12), "dope": r.normal(size=16)}

        def target(self, cid):
            return self._c(cid)

        def bank(self, cid):
            return self._c(cid)

    lock = {"gp": {"max_expanded_width": SV.GP_MAX_EXPANDED_WIDTH, "timeout_s": SV.GP_TIMEOUT_S,
                   "locked_config": {"scheme": "C", "arm": "evoB", "generations": 10, "stack": "CV5"}}}
    ad = SV.GPAdapter(lock, [0, 1], _NoLeakLM(), env / "gproot", views_dir=env / "views", meta_rows=C.train_rows(),
                      runner=_fake_runner(), guard=False, screener_factory=S.FallbackScreener,
                      coder_factory=HashCoder)
    cid = _nonpilot_cid()
    log = SV.JsonlLog(SV.run_dir() / "audit_units.s0of1.jsonl")
    done = {}
    n = SV.run_context_units(cid, (317101,), ad, log, done, {}, "L" * 64)
    assert n == len(SV.PIPELINES)
    recs = {p: done[(cid, 317101, p)] for p in SV.PIPELINES}
    assert recs["SCREEN_TOPK_PICK"]["live_evals"] == recs["H317_LOCKED"]["live_evals"]
    assert abs(recs["H317_LOCKED"]["live_evals"] - recs["EVOA_MATCHED"]["live_evals"]) <= SV.LIVE_MATCH_TOL
    for p, r in recs.items():
        assert np.isfinite(r["value"]) and r["audit_ledger"]["outputs"] >= 1
        assert r["audit_ledger"]["outputs"] == 1 or p not in ("BANK_NN1",)
    assert recs["BANK_NN1"]["live_evals"] == 0 and recs["H257_REPLICA"]["live_evals"] == 8 + 15
    assert 1 <= recs["BANK_NN8_PICK"]["live_evals"] <= 8 and recs["H317_LOCKED"]["live_evals"] >= 60
    assert set(recs["H317_LOCKED"]["engine"]) >= {"physical_runs", "memo_hits"}
    assert all(r["status"] == "completed" for r in recs.values())
    assert all(r["max_expanded_width"] == SV.GP_MAX_EXPANDED_WIDTH for r in recs.values())
    assert not any(r["live_match_violation"] for r in recs.values())


def test_split_cores_and_parse_cpus():
    assert SV.parse_cpus("16-19,40") == [16, 17, 18, 19, 40]
    assert SV.split_cores(list(range(88)), 14)[13] == list(range(78, 84))
    with pytest.raises(ValueError):
        SV.split_cores([1, 2], 3)


def test_lock_rules_cover_prereg_constants():
    assert SV.R882_NATIVE91_MEAN == 1.1377941898164525 and C.BOOT_SEED_STAGE_V == 317160
    assert SV.PIPELINES[1:3] == ("H317_LOCKED", "EVOA_MATCHED")
    assert SV.CONTROLS == ("EVOA_MATCHED", "SCREEN_TOPK_PICK") and SV.PORTFOLIO_REF == "BANK_NN8_PORTFOLIO"
    assert SV.LOCK_BUDGET == {"steps": 3000, "slate": 128, "top_k": 48, "batch_contexts": 16, "n_seeds": 2}


# ----------------------------------------------------------------------------------------------------------------------
# Orchestration: prep (stubbed DL-Val leaves, real regression / TOP4 / markers) -> audit -> stats on 6 SYNTHETIC targets
# ----------------------------------------------------------------------------------------------------------------------
class _Enc:
    def dope_sha(self):
        return C.DOPE_SHA256


def _orchestration(env, monkeypatch, corrupt_reference, vnative_fail: bool = False):
    import argparse

    cids = [f"{i:02d}" + "9" * 62 for i in range(6)]
    rows = [{"context_sha256": c, "family_id": f"fam-syn-{i % 3}", "dataset_id": f"dsyn{i}",
             "train_only_jtt1_sha256": "t", "pool_sha256": "p"} for i, c in enumerate(cids)]
    sel = {"train": C.load_selected()["train"], "dev": list(reversed(rows))}  # target order != sorted order
    monkeypatch.setattr(SV, "N_TARGETS", 6)
    monkeypatch.setattr(SV, "N_TARGET_FAMILIES", 3)
    monkeypatch.setattr(C, "load_selected", lambda: sel)
    monkeypatch.setattr(C, "dev_rows_for_stage_v", lambda: sorted(rows, key=lambda r: r["context_sha256"]))
    lock = {"gp": {}, "neural": {}}

    def enter(phase):
        C._STATE["stage_v"] = True
        return lock, "L" * 64, {"absent_on_host": [], "phase": phase}

    monkeypatch.setattr(SV, "enter_stage_v", enter)
    monkeypatch.setattr(SV, "verify_inputs", lambda r, a, b: {"stub": True})
    order = []
    dd = {m: {"max_jaccard": 0.0, "max_overlap": 0.0, "n_jaccard_gt_0p5": 0, "flagged": []} for m in SV.DEDUPE_MODES}
    monkeypatch.setattr(SV, "run_dedupe", lambda r: (order.append("dedupe"), {"summary": dd})[1])
    monkeypatch.setattr(SV, "encoder_parity", lambda enc: (order.append("parity"), {"all_exact": True})[1])
    tdir, vdir = env / "tables", SV.views_search_dir()
    tdir.mkdir()
    vdir.mkdir(parents=True)
    man = {}
    for i, r in enumerate(rows):
        c = r["context_sha256"]
        (tdir / f"{c}.jtt1").write_bytes(f"table{i}".encode())
        (vdir / f"{c}.npz").write_bytes(f"views{i}".encode())
        man[c] = {"family_id": r["family_id"], "dataset_id": r["dataset_id"], "nt": 100 + i, "width": 5,
                  "table_sha256": C.sha256_file(tdir / f"{c}.jtt1"),
                  "search_views_sha256": C.sha256_file(vdir / f"{c}.npz")}
    monkeypatch.setattr(SV, "encode_targets", lambda enc, r: (order.append("encode"), man)[1])
    monkeypatch.setattr(C, "table_path", lambda c: tdir / f"{c}.jtt1")
    rng = np.random.default_rng(4)
    pools, costs = {}, {}
    for c in cids:
        fit, gate = rng.normal(-1.2, 0.3, 30), rng.random(30) > 0.2
        pools[c] = {"fitness": fit, "gate": gate, "utility": SV.native_utility_h258(fit, gate),
                    "program": np.zeros((30, 412), np.float32)}
        costs[c] = rng.normal(0, 1, 30).astype(np.float32)
    monkeypatch.setattr(SV, "load_pool", lambda c: (order.append("pool"), pools[c])[1])
    ref_u = [float(pools[c]["utility"][int(np.argmin(costs[c]))]) for c in cids]
    mean = float(np.mean([ref_u[cids.index(r["context_sha256"])] for r in sel["dev"]]))
    if corrupt_reference == "ulp" or corrupt_reference is True:
        ref_u[3] = float(np.nextafter(ref_u[3], -np.inf))
    ref_path, cost_path = env / "ref.json", env / "costs.npz"
    ref_obj = {"evaluation_contract": {"contexts": [{"cid": c} for c in cids]}, "utility": {"per_context": ref_u}}
    if corrupt_reference == "malformed":
        ref_obj = {"evaluation_contract": {"contexts": [{"cid": c} for c in cids]}}  # no utility block
    ref_path.write_text(json.dumps(ref_obj))
    np.savez(cost_path, **costs)
    orig = SV.r882_regression
    monkeypatch.setattr(SV, "r882_regression", lambda *a, **k: orig(*a, **k, expected_mean=mean, n_expected=6))
    monkeypatch.setattr(SV, "full_view0", lambda c: np.zeros(997, np.float32))
    def vn(lm, c, f, v, p, u):
        if vnative_fail and c == cids[2]:
            raise ValueError("synthetic V-native crash")
        return {"u": float(u[1]), "per_seed": {str(s): {"top1": 1, "u": float(u[1])} for s in SV.LOCK_TRAIN_SEEDS}}

    monkeypatch.setattr(SV, "vnative_context", vn)
    args = argparse.Namespace(resume=False, relaunch_before_outcome=False, reference=str(ref_path),
                              costs=str(cost_path), threads=1, cpus="0-3", jobs=2, shard="0/1", partial=False)
    return args, order, cids


def test_orchestration_prep_audit_stats(env, monkeypatch):
    args, order, cids = _orchestration(env, monkeypatch, corrupt_reference=False, vnative_fail=True)
    done = SV.run_prep(args, encoder_factory=_Enc, lock_models_factory=lambda lk: _NoLeakLM())
    assert done["regression_pass"] and (SV.run_dir() / SV.PREP_DONE).exists()
    # every pre-outcome step ran before the first pool (outcome) read; the marker precedes it on disk
    assert order[:3] == ["dedupe", "parity", "encode"] and set(order[3:]) == {"pool"}
    ev = [e["event"] for e in SV.read_jsonl(SV.run_dir() / "events.jsonl")[0]]
    assert ev.index("dedupe_done") < ev.index("first_outcome_read") < ev.index("prep_done")
    with pytest.raises(SV.RetryRefused):
        SV.run_prep(args, encoder_factory=_Enc, lock_models_factory=lambda lk: _NoLeakLM())
    SV.run_audit(args, executor_factory=lambda cores: _fake_unit)
    assert len(SV.all_unit_records()[0]) == len(cids) * 2 * len(SV.PIPELINES)
    with pytest.raises(SV.RetryRefused, match="no retry"):
        SV.run_audit(args, executor_factory=lambda cores: _fake_unit)
    args.resume = True
    def never(name, cid, seed, k):
        raise AssertionError("a recorded unit was recomputed")

    SV.run_audit(args, executor_factory=lambda cores: never)  # resume with nothing left: no unit may run
    m = SV.run_stats(args)
    assert m["final"] and m["verdict"] in ("ADVANCE", "NO_ADVANCE-INFERIOR", "NO_ADVANCE-UNDETECTED", "INCONCLUSIVE")
    assert m["regression"]["status"] == "PASS" and m["disclosure"]["audit_resumes"] == 1
    nat = json.loads((SV.run_dir() / "native.json").read_text())
    assert nat["stage"] == "complete" and nat["vnative"][cids[2]]["failed"] and nat["vnative"][cids[2]]["u"] == -2.0
    assert m["v_native"]["failed_contexts"] == 1 and m["disclosure"]["v_native_failed_contexts"] == 1
    assert m["disclosure"]["failed_units"] == 0 and m["failed_units"]["total"] == 0
    assert (SV.run_dir() / "metrics.json").stat().st_size < SV.METRICS_MAX_BYTES
    assert set(json.loads((SV.run_dir() / "panel_stage_v.json").read_text())) == set(cids)


def test_orchestration_regression_fail_stops_stage(env, monkeypatch):
    args, order, cids = _orchestration(env, monkeypatch, corrupt_reference=True)
    with pytest.raises(SV.RegressionFail):
        SV.run_prep(args, encoder_factory=_Enc, lock_models_factory=lambda lk: _NoLeakLM())
    rd = SV.run_dir()
    assert (rd / SV.REG_FAIL).exists() and not (rd / SV.PREP_DONE).exists()
    nat = json.loads((rd / "native.json").read_text())
    assert nat["regression"]["bit_exact_matches"] == 5 and "vnative" not in nat and "top4" not in nat
    with pytest.raises(SV.RetryRefused, match="REGRESSION_FAIL"):
        SV.admit_audit((0, 1), True)
    with pytest.raises(SV.RetryRefused, match="REGRESSION_FAIL"):
        SV.admit_prep(True, False)
    with pytest.raises(SV.RetryRefused):
        SV.relaunch_once()
    assert SV.run_stats(args)["verdict"] == "REGRESSION_FAIL"


# ----------------------------------------------------------------------------------------------------------------------
# sandbox_h317_stagev.sh (argv layout via --print; refusals). The real layout was also probed on a synthetic tree.
# ----------------------------------------------------------------------------------------------------------------------
def _sbx(*args, env_extra=None):
    import subprocess

    e = {**os.environ, **(env_extra or {})}
    return subprocess.run(["bash", str(HERE / "sandbox_h317_stagev.sh"), *args], capture_output=True, text=True,
                          env=e, cwd=str(HERE))


def _pairs(argv):
    return {(argv[i], argv[i + 1], argv[i + 2]) for i in range(len(argv) - 2) if argv[i] in ("--ro-bind", "--bind")}


def test_stagev_sandbox_unlock_layout():
    r = _sbx("--print", "unlock", "python3", "stage_v_h317.py", "unlock")
    assert r.returncode == 0, r.stderr
    a = r.stdout.splitlines()
    rc = "/home/ubuntu/jope_plan/ns/receipts/H317_LOCK_sha256.txt"
    binds = _pairs(a)
    assert ("--ro-bind", rc, rc) in binds
    view, res01 = "/home/ubuntu/jope-data/h317-cl/res01-train-view", "/home/ubuntu/jope-data/res-01/input"
    assert ("--ro-bind", view, res01) in binds
    assert not any("h317-stagev" in x for x in a)  # no Stage V root before the unlock
    assert "--unshare-net" in a and a[-2:] == ["stage_v_h317.py", "unlock"]


def test_stagev_sandbox_v_layout(tmp_path):
    sv, data = str(tmp_path / "sv"), str(tmp_path / "data")
    r = _sbx("--print", "v", "python3", "x.py", env_extra={"H317_STAGEV_ROOT": sv, "H317_DATA": data})
    assert r.returncode == 0, r.stderr
    a = r.stdout.splitlines()
    binds = _pairs(a)
    assert ("--ro-bind", f"{sv}/res01-v-view", "/home/ubuntu/jope-data/res-01/input") in binds
    assert ("--ro-bind", f"{data}/stage_v", f"{data}/stage_v") in binds and ("--bind", sv, sv) in binds
    assert ("--ro-bind", f"{sv}/inputs", f"{sv}/inputs") in binds
    i = a.index("H317_STAGEV_ROOT")
    assert a[i - 1] == "--setenv" and a[i + 1] == sv and "--unshare-net" in a
    # the ro re-binds come after the rw binds they narrow (bwrap applies mounts in order)
    assert a.index(f"{data}/stage_v") > a.index(data) and a.index(f"{sv}/inputs") > a.index(sv)


def test_stagev_sandbox_refusals():
    for mode in ("unlock", "v"):
        r = _sbx(mode, "true")  # the receipt is not visible here (and never exists in tests)
        assert r.returncode == 3 and "refused" in r.stderr
    assert _sbx("bogus", "true").returncode == 2


def test_regression_malformed_reference_is_a_recorded_fail(env, monkeypatch):
    args, _, _ = _orchestration(env, monkeypatch, corrupt_reference="malformed")
    with pytest.raises(SV.RegressionFail):
        SV.run_prep(args, encoder_factory=_Enc, lock_models_factory=lambda lk: _NoLeakLM())
    rd = SV.run_dir()
    nat = json.loads((rd / "native.json").read_text())
    assert nat["stage"] == "regression_fail" and "KeyError" in nat["error"] and (rd / SV.REG_FAIL).exists()
    assert SV.run_stats(args)["verdict"] == "REGRESSION_FAIL"


def test_recorded_regression_pass_is_never_recomputed_on_resume(env, monkeypatch):
    args, _, cids = _orchestration(env, monkeypatch, corrupt_reference=False)
    monkeypatch.setattr(SV, "vnative_context", lambda *a: (_ for _ in ()).throw(MemoryError()))  # host failure
    with pytest.raises(MemoryError):
        SV.run_prep(args, encoder_factory=_Enc, lock_models_factory=lambda lk: _NoLeakLM())
    rd = SV.run_dir()
    assert json.loads((rd / "native.json").read_text())["stage"] == "regression"  # PASS recorded right away
    assert not (rd / SV.PREP_DONE).exists()

    def no_regression(*a, **k):
        raise AssertionError("the recorded regression must not be recomputed")

    monkeypatch.setattr(SV, "r882_regression", no_regression)
    monkeypatch.setattr(SV, "vnative_context", lambda lm, c, f, v, p, u: {
        "u": float(u[0]), "per_seed": {str(s): {"top1": 0, "u": float(u[0])} for s in SV.LOCK_TRAIN_SEEDS}})
    args.resume = True
    done = SV.run_prep(args, encoder_factory=_Enc, lock_models_factory=lambda lk: _NoLeakLM())
    assert done["regression_pass"] and json.loads((rd / "native.json").read_text())["stage"] == "complete"
    ev = [e for e in SV.read_jsonl(rd / "events.jsonl")[0] if e["event"] == "prep_start"]
    assert [e["mode"] for e in ev] == ["first", "resume"]


def test_tampered_train_inputs_manifest_is_refused(locked):
    lock, sha, *_ = locked
    assert SV.cmd_unlock(None) == 0
    C._STATE["stage_v"] = False
    m = SV.train_inputs_manifest_path()
    os.chmod(m, 0o644)
    doc = json.loads(m.read_text())
    doc["tables"][sorted(doc["tables"])[0]] = "0" * 64
    m.write_text(json.dumps(doc))
    with pytest.raises(C.GuardViolation, match="changed after LOCK"):
        SV.enter_stage_v("audit")
    assert C._STATE["stage_v"] is False


def test_arm_a_lock_model_uses_the_a525_training_set(env):
    """MEDIUM-4: H0.1 = A (baseline) + a GP adoption: Stage V runs with the arm-A LOCK model (525 subset members)."""
    import rapid_hypothesis_lab as R

    dec = _decisions(env, w1="A", gp=("C", "evoB", "single"))
    arm = SV.locked_arm("A", "S", "opt1")
    assert arm["subset"] == "A525" and arm["views"] == "A4"
    lock_dir = C.DATA_ROOT / "neural" / "lock"
    _synthetic_lock_fit(lock_dir, arm, cids=R.arm_a_subset(C.train_rows()))
    lock = SV.build_lock(dec["H0.1"], dec["H0.2"], dec["H0.3"], dec["gp"], lock_dir)
    assert lock["stage_v_runs"] and not lock["adoptions"]["H0.1"] and lock["adoptions"]["H0.4"]
    assert all(v["n_train_cids"] == 525 for v in lock["neural"]["per_seed"].values())
    wrong = C.DATA_ROOT / "neural" / "lock_all"
    _synthetic_lock_fit(wrong, arm)  # an all-TRAIN fit is NOT the locked arm-A configuration
    with pytest.raises(SV.StageVError, match="locked arm's training set .subset 'A525': 525 contexts"):
        SV.build_lock(dec["H0.1"], dec["H0.2"], dec["H0.3"], dec["gp"], wrong)


def test_rerun_cap_records_a_repeating_host_failure(env):
    """A deterministic host failure (MemoryError) is re-executed on resume at most twice, then recorded failed."""
    _prep_done()
    cid = "0e" + "a" * 62
    calls = []

    def ex(name, c, seed, k):
        if name == "H257_REPLICA":
            calls.append(name)
            raise MemoryError()
        return _fake_unit(name, c, seed, k)

    for _ in range(SV.MAX_UNIT_ATTEMPTS):
        with pytest.raises(SV.StageVError):
            SV.run_audit_loop([cid], (317101,), [ex], (0, 1), "L" * 64)
    SV.run_audit_loop([cid], (317101,), [ex], (0, 1), "L" * 64)  # 3rd attempt: recorded, executor not called
    done, started, _ = SV.all_unit_records()
    r = done[(cid, 317101, "H257_REPLICA")]
    assert len(calls) == SV.MAX_UNIT_ATTEMPTS and r["status"] == "failed" and r["value"] == 0.0 and r["invalid"]
    assert r["error"] == "host failure on 2 prior attempts" and r["rerun_after_interruption"]
    assert len(done) == len(SV.PIPELINES) and not started


def test_transient_host_failure_classification():
    assert SV._infra_error(BlockingIOError(errno.EAGAIN, "resource temporarily unavailable"))
    assert SV._infra_error(BlockingIOError())
    assert SV._infra_error(OSError(errno.EAGAIN, "EAGAIN"))
    assert SV._infra_error(RuntimeError("can't start new thread"))
    assert SV._infra_error(MemoryError()) and SV._infra_error(OSError(errno.ENOSPC, "full"))
    assert not SV._infra_error(RuntimeError("some deterministic pipeline bug"))
    assert not SV._infra_error(ValueError("x")) and not SV._infra_error(OSError(errno.ENOENT, "missing"))
    assert SV._integrity_error(C.GuardViolation("x")) and not SV._integrity_error(OSError(errno.EIO, "io"))


def test_verify_train_inputs_refusals(env, monkeypatch):
    """Both refusals of verify_train_inputs: a TRAIN table differing from the manifest, and a selected.json sha not
    allowed for the phase (sha256_file monkeypatched; no table is read)."""
    rows = C.train_rows()[:3]
    tables = {r["context_sha256"]: f"{i:064x}" for i, r in enumerate(rows, 1)}
    red, orig = C.SELECTED_REDACTED_SHA256, C.SELECTED_SHA256
    SV.train_inputs_manifest_path().write_text(json.dumps(
        {"tables": tables, "selected_sha256": {"redacted": red, "original": orig},
         "selected_allowed_by_phase": SV.SELECTED_ALLOWED_BY_PHASE}))
    lock = {"train_inputs": {"manifest": str(SV.train_inputs_manifest_path())}}
    state = {"tables": dict(tables), "selected": red}

    def fake_sha(path):
        path = Path(path)
        if path.name == "selected.json":
            return state["selected"]
        if path.suffix == ".jtt1":
            return state["tables"][path.stem]
        raise AssertionError(f"unexpected hash of {path}")

    monkeypatch.setattr(C, "sha256_file", fake_sha)
    assert SV.verify_train_inputs(lock, "audit") == {"train_tables_verified": 3, "selected_sha256": red}
    state["tables"][rows[1]["context_sha256"]] = "f" * 64  # a staged TRAIN table differs from the manifest
    with pytest.raises(C.GuardViolation, match="1 TRAIN table.*differ from the LOCK manifest"):
        SV.verify_train_inputs(lock, "audit")
    state["tables"] = dict(tables)
    with pytest.raises(C.GuardViolation, match="not allowed in this phase"):
        SV.verify_train_inputs(lock, "prep")  # prep needs the original selected.json
    state["selected"] = orig
    assert SV.verify_train_inputs(lock, "prep")["selected_sha256"] == orig
    state["selected"] = "e" * 64  # neither pinned copy
    for phase in ("prep", "audit", "stats"):
        with pytest.raises(C.GuardViolation, match="not allowed in this phase"):
            SV.verify_train_inputs(lock, phase)
    assert SV.verify_train_inputs({}, "audit") == {}  # LOCKs without the inputs group are not checked
