"""Data-free H228 tests: AND-mask, GroupDRO, placebo density parity, lock ordering, and guards."""

import ast
import hashlib
import importlib.util
import io
import json
import math
import subprocess
from pathlib import Path
import types
import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(
        "h228_test_" + name, HERE / (name + ".py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


invariant = load("invariant")
sci = load("science")
backend = load("backend")
runner = load("runner")


def legacy_module():
    spec = importlib.util.spec_from_file_location(
        "h228_legacy_test", HERE.parent / "h109_family_balanced_accum/runner.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return sci.legacy_config(mod)


def test_andmask_exact_tau08_boundary_mean_and_family_min_guard():
    # 5 families, 5 coordinates:
    # col 0: 5/5 positive (1.0 >= 0.8) -> kept, mean = 2.0
    # col 1: 4/5 positive (0.8 >= 0.8) -> kept, mean = (1+1+1+1-1)/5 = 0.6
    # col 2: 3/5 positive (0.6 < 0.8)  -> zeroed
    # col 3: 4/5 negative (0.8 >= 0.8) -> kept, mean = (-2-2-2-2+3)/5 = -1.0
    # col 4: 0/5 nonzero  (0.0 < 0.8)  -> zeroed
    g = torch.tensor(
        [
            [2.0, 1.0, 1.0, -2.0, 0.0],
            [2.0, 1.0, 1.0, -2.0, 0.0],
            [2.0, 1.0, 1.0, -2.0, 0.0],
            [2.0, 1.0, -1.0, -2.0, 0.0],
            [2.0, -1.0, -1.0, 3.0, 0.0],
        ],
        dtype=torch.float32,
    )
    masked, mask, stats = invariant.andmask_combine(g, tau=0.8)
    assert mask.tolist() == [True, True, False, True, False]
    assert stats["n_families"] == 5
    assert stats["kept_count"] == 3
    assert stats["total_coords"] == 5
    assert stats["kept_fraction"] == pytest.approx(0.6)
    assert torch.allclose(masked, torch.tensor([2.0, 0.6, 0.0, -1.0, 0.0]))

    # Refuse < 4 families or non-0.8 tau or NaN
    with pytest.raises(ValueError):
        invariant.andmask_combine(g[:3], tau=0.8)
    with pytest.raises(ValueError):
        invariant.andmask_combine(g, tau=0.7)
    bad = g.clone()
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError):
        invariant.andmask_combine(bad, tau=0.8)


def test_andmask_shuf_exact_per_step_density_match_and_random_permutation():
    rng = np.random.default_rng(228999)
    for step in range(40):
        n_fam = int(rng.integers(4, 17))
        raw = rng.normal(loc=0.35, scale=1.0, size=(n_fam, 256)).astype(np.float32)
        g = torch.as_tensor(raw)
        _, and_mask, and_stats = invariant.andmask_combine(g, tau=0.8)
        shuf_grad, shuf_mask, shuf_stats = invariant.andmask_shuf_combine(
            g, seed=228000 + step, tau=0.8
        )
        assert shuf_stats["kept_count"] == and_stats["kept_count"]
        assert shuf_stats["kept_fraction"] == and_stats["kept_fraction"]
        assert shuf_stats["density_match"] is True
        assert int(shuf_mask.sum().item()) == int(and_mask.sum().item())
        # Reproducibility with same seed
        shuf_grad2, shuf_mask2, _ = invariant.andmask_shuf_combine(
            g, seed=228000 + step, tau=0.8
        )
        assert torch.equal(shuf_mask, shuf_mask2) and torch.equal(shuf_grad, shuf_grad2)
        if 0 < and_stats["kept_count"] < 256:
            assert not torch.equal(shuf_mask, and_mask)
        # Verify kept values are exact family means at the selected coordinates
        mean_g = g.mean(dim=0)
        assert torch.allclose(shuf_grad[shuf_mask], mean_g[shuf_mask])
        assert (shuf_grad[~shuf_mask] == 0).all()

    # Also test explicit kept_count_override parity
    g_test = torch.ones((6, 100), dtype=torch.float32)
    _, m_ov, s_ov = invariant.andmask_shuf_combine(
        g_test, seed=42, tau=0.8, kept_count_override=17
    )
    assert s_ov["kept_count"] == 17 and int(m_ov.sum().item()) == 17
    assert s_ov["kept_fraction"] == pytest.approx(0.17)


def test_gdro_sagawa2020_exponentiated_weights_and_equal_loss_reduction():
    fit_families = [f"fam-{i}" for i in range(10)]
    q0 = invariant.gdro_init_weights(len(fit_families))
    assert q0.shape == (10,) and math.fsum(q0.tolist()) == pytest.approx(1.0)

    step_families = ["fam-0", "fam-2", "fam-5", "fam-8"]
    g = torch.tensor(
        [
            [1.0, -2.0, 0.5],
            [3.0, 0.0, -0.5],
            [-1.0, 4.0, 1.5],
            [5.0, 2.0, 2.5],
        ],
        dtype=torch.float32,
    )
    # Equal losses -> uniform step weights -> exact mean of family gradients
    eq_losses = np.array([1.5, 1.5, 1.5, 1.5], dtype=np.float64)
    grad_eq, q1, stats_eq = invariant.gdro_step(
        q0, fit_families, step_families, eq_losses, g, eta=0.01
    )
    assert torch.allclose(grad_eq, g.mean(dim=0), atol=1e-6)
    assert stats_eq["max_step_weight"] == pytest.approx(0.25)
    assert stats_eq["min_step_weight"] == pytest.approx(0.25)
    # Observed families grew by exp(0.01 * 1.5) relative to unobserved families
    assert q1[0] > q1[1] and math.fsum(q1.tolist()) == pytest.approx(1.0)

    # Unequal losses from uniform q0 -> highest-loss family gets highest weight
    uneq_losses = np.array([0.0, 0.0, 0.0, 10.0], dtype=np.float64)
    grad_uneq, q2, stats_uneq = invariant.gdro_step(
        q0, fit_families, step_families, uneq_losses, g, eta=0.01
    )
    expected_w3 = math.exp(0.1) / (3.0 + math.exp(0.1))
    assert stats_uneq["max_step_weight"] == pytest.approx(expected_w3)
    # Since fam-8 has g[3, 0] = 5.0 (largest), upweighting fam-8 increases grad_uneq[0] above mean
    assert float(grad_uneq[0]) > float(g[:, 0].mean())

    with pytest.raises(ValueError):
        invariant.gdro_step(
            q0, fit_families, step_families[:3], uneq_losses[:3], g[:3], eta=0.01
        )
    with pytest.raises(ValueError):
        invariant.gdro_step(q0, fit_families, step_families, uneq_losses, g, eta=0.05)


def test_extract_family_gradients_and_accumulate_active_grad():
    params = [
        torch.nn.Parameter(torch.ones(10, dtype=torch.float32) * (i + 1) * 0.01)
        for i in range(47)
    ]
    batch_cids = [f"c{i}" for i in range(8)]
    family_by_cid = {f"c{i}": f"f{i // 2}" for i in range(8)}  # 4 families
    active_sum = sum(p.sum() for p in params[:41])
    ctx_losses = torch.stack(
        [active_sum * (i + 1) for i in range(8)]
    )  # 41 active, 6 unused
    step_fams, fam_losses, fam_grads, active_ix = invariant.extract_family_gradients(
        params, batch_cids, family_by_cid, ctx_losses
    )
    assert step_fams == ["f0", "f1", "f2", "f3"]
    assert len(active_ix) == 41 and fam_grads.shape == (4, 410)
    assert np.isfinite(fam_losses).all()
    for p in params:
        p.grad = None
    invariant.accumulate_active_grad(
        params, active_ix, torch.ones(410, dtype=torch.float32), scale=0.25
    )
    assert sum(p.grad is not None for p in params) == 41
    assert all(torch.allclose(p.grad, torch.full_like(p, 0.25)) for p in params[:41])
    assert all(p.grad is None for p in params[41:])

    # Fewer than 4 families in batch must refuse
    too_few = {f"c{i}": f"f{i // 3}" for i in range(8)}  # only 3 families
    with pytest.raises(RuntimeError, match="requires >= 4 families"):
        invariant.extract_family_gradients(params, batch_cids, too_few, ctx_losses)


def test_labels_refuse_wrong_role_or_unlocked_a_and_dl_val_before_any_read():
    e = backend.Engine.__new__(backend.Engine)
    e.progress = {"locked": False}
    e.labels_cache = {}
    e.pools = types.SimpleNamespace(blob=lambda cid: pytest.fail("unauthorized read"))
    with pytest.raises(RuntimeError):
        e.labels("x", {"y"}, "FIT")
    with pytest.raises(RuntimeError):
        e.labels("x", {"x"}, "UNKNOWN")
    with pytest.raises(RuntimeError):
        e.labels("x", {"x"}, "A")
    with pytest.raises(RuntimeError):
        e.labels("x", {"x"}, "DL-Val")


@pytest.mark.parametrize("name", ["runner", "launch"])
def test_guard_301_seconds_and_recovery(name):
    g = load(name).SustainedLoad()
    assert not g.overloaded(90, 1, 1, 100, 0)
    assert not g.overloaded(1, 1, 1, 100, 1)
    assert not g.overloaded(90, 1, 1, 100, 2)
    assert not g.overloaded(90, 1, 1, 100, 301)
    assert g.overloaded(90, 1, 1, 100, 303)
    assert load(name).SustainedLoad().overloaded(1, 86, 1, 100, 0)


def test_launch_load5_self_pause_hysteresis():
    launch = load("launch")
    assert not launch.should_pause_worker(84.9, paused=False)
    assert launch.should_pause_worker(85.1, paused=False)
    assert launch.should_pause_worker(86.0, paused=False)
    assert launch.should_pause_worker(81.0, paused=True)
    assert not launch.should_pause_worker(80.0, paused=True)


def test_source_calls_native_once_with_no_duplicate_token_kwargs():
    tree = ast.parse((HERE / "runner.py").read_text())
    native = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "native_forward"
    )
    kwargs = next(
        n.value
        for n in native.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "kwargs" for t in n.targets)
    )
    assert not {"mm_tokens_10x32", "mm_mask_10"} & {k.arg for k in kwargs.keywords}
    calls = [
        n
        for n in ast.walk(native)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "predict_costs"
    ]
    assert len(calls) == 1
    assert {"mm_tokens_10x32", "mm_mask_10"} <= {k.arg for k in calls[0].keywords}


def test_s3_create_only_and_exact_version_corruption_refusal(tmp_path):
    store = runner.Store.__new__(runner.Store)
    store.output = tmp_path
    store.prefix = "H228/unit/"
    store.receipts = {}

    class Fake:
        corrupt = False

        def put_object(self, **kw):
            assert kw["IfNoneMatch"] == "*"
            self.raw = kw["Body"]
            return {"VersionId": "v1"}

        def get_object(self, **kw):
            assert kw["VersionId"] == "v1"
            return {
                "VersionId": "v1",
                "Body": io.BytesIO(b"bad" if self.corrupt else self.raw),
            }

    store.client = Fake()
    store.put("x", b"proof")
    store.client.corrupt = True
    with pytest.raises(RuntimeError):
        store.put("y", b"proof")


def test_fifo_preserves_old_ticket_and_refuses_late_waiter(tmp_path, monkeypatch):
    launch = load("launch")
    queue = tmp_path / "queue"
    queue.mkdir()
    old = queue / "1-100-HOLD"
    old.write_text("keep")
    original_path = Path

    def path(value):
        return (
            queue
            if str(value).endswith("qualification.queue")
            else tmp_path / "lock"
            if str(value).endswith("qualification.lock")
            else original_path(value)
        )

    monkeypatch.setattr(launch, "Path", path)
    monkeypatch.setattr(launch.os, "getpid", lambda: 200)
    monkeypatch.setattr(
        launch.psutil,
        "Process",
        lambda pid: types.SimpleNamespace(
            is_running=lambda: True, status=lambda: "running"
        ),
    )
    monkeypatch.setattr(launch.time, "time_ns", lambda: 2)
    monkeypatch.setattr(launch.time, "time", lambda: launch.COMPUTE_DEADLINE_EPOCH + 1)
    with pytest.raises(RuntimeError, match="FIFO admission deadline"):
        with launch.fifo_lock():
            pytest.fail("jumped older ticket")
    assert (
        old.read_text() == "keep"
        and (queue / "2-200-H228").exists()
        and (queue / "2-200-H228.done").exists()
    )


def test_fresh_folds_exact_family_coverage_and_h109_algorithm():
    rows = [
        dict(
            family_id=str(i),
            context_sha256=str(i),
            dl_split="DL-Train",
            task="regression",
        )
        for i in range(205)
    ]
    legacy = legacy_module()
    assert sci.folds([r["family_id"] for r in rows], 10900) == legacy.family_folds(rows)
    partitions = []
    for seed in sci.SEEDS:
        seen = []
        for outer in range(5):
            fit, held = sci.partition(rows, seed, outer)
            f = {r["family_id"] for r in fit}
            h = {r["family_id"] for r in held}
            assert len(f) == 164 and len(h) == 41 and not f & h
            seen += sorted(h)
        assert len(seen) == len(set(seen)) == 205
        partitions.append(sci.folds([r["family_id"] for r in rows], seed + 40))
    assert partitions[0] != partitions[1] != partitions[2]
    report = sci.jaccard_report([r["family_id"] for r in rows])
    assert all(
        max(max(row) for row in matrix) < 1
        for old in report.values()
        for matrix in old.values()
    )


def test_fbal_arms_share_sampler_and_respect_4_family_minimum():
    legacy = legacy_module()
    groups = {f"fam-{i}": [f"cid-{i}-0", f"cid-{i}-1"] for i in range(30)}
    seed = sci.training_seed(228101, 0)
    ctrl_plan = legacy.sample_microbatches(groups, sci.LEGACY_ARMS["CTRL"], seed)
    and_plan = legacy.sample_microbatches(groups, sci.LEGACY_ARMS["ANDMASK"], seed)
    gdro_plan = legacy.sample_microbatches(groups, sci.LEGACY_ARMS["GDRO"], seed)
    shuf_plan = legacy.sample_microbatches(
        groups, sci.LEGACY_ARMS["ANDMASK_SHUF"], seed
    )
    assert and_plan == gdro_plan == shuf_plan
    assert len(and_plan) == 160 and all(len(x) == 16 for x in and_plan)
    assert len(ctrl_plan) == 160
    fam_by_cid = {c: f for f, cs in groups.items() for c in cs}
    assert min(len({fam_by_cid[c] for c in mb}) for mb in and_plan) >= 4


def test_exact_merged_pr_gate_refuses_open_pr_and_wrong_source():
    launch = load("launch")
    pr = dict(
        headRefName="rd/h228-family-invariant",
        baseRefName="main",
        isDraft=False,
        state="OPEN",
        headRefOid="x",
    )
    assert not launch.admitted_pr(pr, "x")
    pr.update(state="MERGED", mergeCommit={"oid": "y"})
    assert launch.admitted_pr(pr, "y") and not launch.admitted_pr(pr, "x")


def test_actual_resource_guard_uses_8gib_150gb_6gib_4cores_and_6gib_vram():
    sample = dict(
        load=1,
        load5=1,
        cores=4,
        host_cores=102,
        cpu=1,
        available=8 * 2**30,
        swap_free=0,
        disk_used_fraction=0.70,
        free=150_000_000_000,
        rss=1_000_000,
    )
    runner.guard(sample, state=runner.SustainedLoad(), now=0, admission=True)
    for key, value in [
        ("available", 8 * 2**30 - 1),
        ("free", 149_999_999_999),
        ("rss", 6_000_000_000),
        ("cores", 5),
        ("load5", 85.1),
    ]:
        with pytest.raises(RuntimeError):
            runner.guard(
                dict(sample, **{key: value}), state=runner.SustainedLoad(), now=1
            )

    # Verify pause_on_trip=True pauses/waits and persists exact violating_field + violating_value
    log_buf = io.StringIO()
    seq = [dict(sample, available=7 * 2**30), dict(sample, available=10 * 2**30)]
    it = iter(seq)
    original_snap = runner.snapshot
    original_sleep = runner.time.sleep
    try:
        runner.snapshot = lambda: next(it)
        runner.time.sleep = lambda s: None
        recovered = runner.guard(
            dict(sample, available=6 * 2**30),
            state=runner.SustainedLoad(),
            now=1,
            pause_on_trip=True,
            log_file=log_buf,
            started=0.0,
            deadline_epoch=9999999999,
        )
    finally:
        runner.snapshot = original_snap
        runner.time.sleep = original_sleep
    assert recovered["guard_trip"] is False
    logged = [json.loads(line) for line in log_buf.getvalue().splitlines()]
    assert len(logged) == 2
    assert logged[0]["paused_on_guard_trip"] is True
    assert logged[0]["violating_field"] == "available"
    assert logged[0]["violating_value"] == 6 * 2**30
    assert logged[1]["violating_field"] == "available"
    assert logged[1]["violating_value"] == 7 * 2**30

    # Also verify launch.guard_violations reports violating_field + violating_value and enforces gpu_free_mib >= 6144 without preempting flywheel
    launch = load("launch")
    ls = dict(
        load1=1.0,
        load5=86.5,
        effective_cores=4,
        host_cores=102,
        cpu_percent=10.0,
        available=12 * 2**30,
        swap_free=0,
        free=200_000_000_000,
        used_fraction=0.70,
        child_rss=1_000_000_000,
        effective_gpu_free_mib=6797,
        overloaded=False,
    )
    v = launch.guard_violations(ls, paused=False, other_gpu_pids={2847658})
    assert ls["guard_trip"] is True
    assert ls["violating_field"] == "load5"
    assert ls["violating_value"] == pytest.approx(86.5)
    assert len(v) == 1

    ls_vram = dict(ls, load5=10.0, effective_gpu_free_mib=6143)
    v_vram = launch.guard_violations(ls_vram, paused=False, other_gpu_pids={2847658})
    assert ls_vram["guard_trip"] is True
    assert ls_vram["violating_field"] == "gpu_free_mib"
    assert ls_vram["violating_value"] == 6143
    assert len(v_vram) == 1


def test_readout_strict_margin_002_exact_tie_rejection_holm2_and_breakthrough():
    families = list(map(str, range(205)))
    val_families = [str(i % 19) for i in range(91)]
    zero = np.zeros((3, 205), dtype=np.float64)
    val_zero = np.zeros((3, 91), dtype=np.float64)
    panels = {a: zero.copy() for a in ("R882",) + sci.ARMS}
    val_panels = {a: val_zero.copy() for a in ("R882",) + sci.ARMS}

    # 0.0 -> NO_ADVANCE
    assert (
        sci.readout(panels, families, val_panels, val_families)["verdict"]
        == "NO_ADVANCE"
    )

    # Exact +0.002 tie -> must REJECT (NO_ADVANCE)
    panels["ANDMASK"][:] = 0.002
    val_panels["ANDMASK"][:] = 0.002
    tie_res = sci.readout(panels, families, val_panels, val_families)
    assert tie_res["verdict"] == "NO_ADVANCE"
    assert not tie_res["admitted"]["ANDMASK"]
    assert not tie_res["dl_val91"]["passes"]["ANDMASK"]

    # TRAIN OOF passes (> +0.002, CI lower > 0), DL-Val91 stays at 0.002 -> CANDIDATE (not BREAKTHROUGH)
    panels["ANDMASK"][:] = 0.015
    cand_res = sci.readout(panels, families, val_panels, val_families)
    assert (
        cand_res["verdict"] == "CANDIDATE"
        and cand_res["admitted"]["ANDMASK"] is True
        and cand_res["holm_multiplicity"] == 2
        and cand_res["dl_val91"]["passes"]["ANDMASK"] is False
    )

    # Both TRAIN OOF and DL-Val91 pass (> +0.002, CI lower > 0, 3/3 seeds, harm <= 10%) -> BREAKTHROUGH
    val_panels["ANDMASK"][:] = 0.015
    bt_res = sci.readout(panels, families, val_panels, val_families)
    assert (
        bt_res["verdict"] == "BREAKTHROUGH"
        and bt_res["admitted"]["ANDMASK"] is True
        and bt_res["dl_val91"]["passes"]["ANDMASK"] is True
    )

    # One negative seed -> NO_ADVANCE
    panels["ANDMASK"][0, :] = -0.005
    assert (
        sci.readout(panels, families, val_panels, val_families)["verdict"]
        == "NO_ADVANCE"
    )

    # Excess harm (> 10% of 205 families = >= 21 families < -0.01) -> NO_ADVANCE
    panels["ANDMASK"][:] = 0.05
    panels["ANDMASK"][:, :25] = -0.02
    assert (
        sci.readout(panels, families, val_panels, val_families)["verdict"]
        == "NO_ADVANCE"
    )


def test_actual_training_all_4_arms_and_exact_shuf_density_parity(tmp_path):
    torch.set_num_threads(1)
    net = torch.nn.Module()
    net.p = torch.nn.ParameterList(
        [torch.nn.Parameter(torch.ones(621) * 0.1) for _ in range(46)]
        + [torch.nn.Parameter(torch.ones(645) * 0.1)]
    )
    e = backend.Engine.__new__(backend.Engine)
    e.net = net
    e.names = tuple(n for n, p in net.named_parameters())
    e.initial = {n: p.detach().clone() for n, p in net.named_parameters()}
    e.check = lambda: None
    e.progress = {"updates": 0}
    e.invariant = invariant

    def digest(state):
        return hashlib.sha256(
            b"".join(v.detach().numpy().tobytes() for v in state.values())
        ).hexdigest()

    e.rt = types.SimpleNamespace(
        tensor_hash=digest,
        loss=lambda pred, f, g: invariant.per_context_loss(pred, f, g).mean(),
    )
    e.treatment = types.SimpleNamespace(
        adapter_forward=lambda net, *args: (
            sum(p.mean() for p in list(net.p)[:41]) + args[0][:, :, 0] * 0.01
        )
    )
    e.score = lambda cid: np.array(
        [float(sum(p.mean() for p in list(net.p)[:41]).detach()), 0.0]
    )
    e.labels = lambda *a: (np.arange(30, dtype=float), np.ones(30, bool))
    e.anchor = lambda kw: (torch.zeros((16, 24)), torch.zeros((16, 24)))

    def batch(cids, indices, allowed):
        assert (
            len(cids) == 16
            and set(cids) <= allowed
            and all(len(ix) == 24 for ix in indices)
        )
        x = torch.tensor([float(cid) for cid in cids])[:, None]
        return (
            dict(
                prog_412=x[:, :, None].expand(-1, 24, 412),
                sos_prior=torch.zeros((16, 24, 2)),
                scale=torch.ones(16),
                revealed_indices=torch.arange(4)[None].expand(16, -1),
                revealed_raw_f=x.expand(-1, 4),
                revealed_gate=torch.ones((16, 4)),
            ),
            x[:, :, None].expand(-1, 10, 32),
            torch.ones((16, 10)),
            x.expand(-1, 24),
            torch.ones((16, 24), dtype=torch.bool),
        )

    e.batch = batch
    events = []
    e.store = types.SimpleNamespace(
        put=lambda name, raw: events.append((name, json.loads(raw)))
    )
    legacy = legacy_module()
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"synthetic")
    checkpoint.chmod(0o444)
    original = legacy.stage_r882_copy
    legacy.stage_r882_copy = lambda src, dst: original(
        src, dst, expected_sha256=hashlib.sha256(b"synthetic").hexdigest()
    )
    # 20 distinct families so every 16-context FBAL microbatch has >= 4 families
    rows = [dict(context_sha256=str(i), family_id=f"fam-{i}") for i in range(20)]
    _, p_ctrl = e.train(
        rows, "CTRL", 228101, 0, "ctrl", legacy, sci, tmp_path, checkpoint, steps=4
    )
    _, p_and = e.train(
        rows, "ANDMASK", 228101, 0, "and", legacy, sci, tmp_path, checkpoint, steps=4
    )
    _, p_gdro = e.train(
        rows, "GDRO", 228101, 0, "gdro", legacy, sci, tmp_path, checkpoint, steps=4
    )
    _, p_shuf = e.train(
        rows,
        "ANDMASK_SHUF",
        228101,
        0,
        "shuf",
        legacy,
        sci,
        tmp_path,
        checkpoint,
        steps=4,
        shuf_kept_counts=p_and["invariant"]["kept_counts"],
    )
    assert e.progress["updates"] == 16
    assert (
        p_ctrl["before_sha256"]
        == p_and["before_sha256"]
        == p_gdro["before_sha256"]
        == p_shuf["before_sha256"]
        == digest(e.initial)
    )
    assert (
        p_and["invariant"]["kept_counts"] == p_shuf["invariant"]["kept_counts"]
        and p_and["invariant"]["kept_fractions_per_step"]
        == p_shuf["invariant"]["kept_fractions_per_step"]
    )
    assert (
        p_gdro["invariant"]["gdro_final_max_weight"]
        > p_gdro["invariant"]["gdro_final_min_weight"]
    )
    assert (
        p_and["sampler_plan_sha256"]
        == p_gdro["sampler_plan_sha256"]
        == p_shuf["sampler_plan_sha256"]
    )


def test_full_protocol_locks_all_60_fits_before_any_a_or_dl_val91_utility(tmp_path):
    study = load("study")
    rows = [dict(context_sha256=str(i), family_id=str(i)) for i in range(205)]
    val_rows = [
        dict(context_sha256=f"val-{i}", family_id=f"vfam-{i % 19}") for i in range(91)
    ]
    events = []
    progress = {"updates": 0, "locked": False, "source_commit": "fixture"}

    class Store:
        def put(self, name, raw):
            events.append(name)

        def arrays(self, name, **arrays):
            events.append(name)

    class Engine:
        names = ()
        net = None
        inputs = {}
        native = {
            **{str(i): np.array([0.0, 1.0]) for i in range(205)},
            **{f"val-{i}": np.array([0.0, 1.0]) for i in range(91)},
        }

        def train(
            self,
            fit,
            arm,
            seed,
            outer,
            label,
            legacy,
            sci,
            output,
            checkpoint,
            steps=40,
            shuf_kept_counts=None,
        ):
            progress["updates"] += steps
            kc = [100] * (steps * 4) if shuf_kept_counts is None else shuf_kept_counts
            return {"fit": [r["context_sha256"] for r in fit]}, dict(
                arm=arm,
                updates=steps,
                seconds_per_update=[0.01] * steps,
                invariant=dict(
                    kept_counts=kc if arm in ("ANDMASK", "ANDMASK_SHUF") else None,
                    kept_fractions_per_step=[0.1] * steps
                    if arm in ("ANDMASK", "ANDMASK_SHUF")
                    else None,
                    mean_kept_fraction=0.1
                    if arm in ("ANDMASK", "ANDMASK_SHUF")
                    else None,
                    min_families_per_microbatch=12,
                    gdro_final_max_weight=0.01 if arm == "GDRO" else None,
                    gdro_final_min_weight=0.005 if arm == "GDRO" else None,
                ),
            )

        def restore(self, state):
            self.fit = set(state["fit"])

        def score(self, cid):
            assert cid not in self.fit
            return np.array([0.0, 1.0])

        def utility(self, cid, allowed, role):
            assert (
                progress["updates"] == 2404
                and progress["locked"]
                and "selection-lock.json" in events
                and role in ("A", "DL-Val")
                and cid in allowed
            )
            return np.array([0.0, -1.0])

    rt = types.SimpleNamespace(
        check_frozen=lambda *a: None,
        sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest(),
    )
    result = study.execute(
        Engine(),
        sci,
        None,
        types.SimpleNamespace(elo_vs_reference=lambda *a, **kw: {"median": 1500}),
        rows,
        Store(),
        rt,
        None,
        None,
        progress,
        runner.save,
        tmp_path,
        lambda: None,
        tmp_path / "unused",
        val_rows=val_rows,
        val_gpu_inputs={f"val-{i}": () for i in range(91)},
        native_val=[0.0] * 91,
    )
    assert (
        result["optimizer_updates"] == 2400
        and result["smoke_optimizer_updates"] == 4
        and result["total_optimizer_updates"] == 2404
        and result["member_fits"] == 60
        and result["verdict"] == "NO_ADVANCE"
        and result["invariant_exposure"]["exact_per_step_density_match"] is True
    )
    assert events.index("selection-lock.json") < events.index("A-cell/228101-0")
    assert events.index("selection-lock.json") < events.index("DL-Val91-panels")


def test_manifest_and_prereg_artifacts_integrity():
    manifest = json.loads((HERE / "manifest.json").read_text())
    assert manifest["wave"] == "H228"
    assert manifest["owner"] == "rd_argon"
    assert manifest["primary"] == "ANDMASK"
    assert manifest["holm_multiplicity"] == 2
    assert manifest["max_total_updates"] == 2404
    assert manifest["member_fits"] == 60
    assert manifest["strict_margin"] == pytest.approx(0.002)
    assert manifest["gpu_uuid"] == "GPU-784030c4-b1f9-d5a3-f0eb-9337f133f503"
    assert manifest["resource_limits"]["host"] == "xbabe1"
    assert manifest["resource_limits"]["min_gpu_free_vram_mib"] == 6144
    assert manifest["resource_limits"]["min_free_bytes"] == 150_000_000_000
    assert manifest["resource_limits"]["min_available_ram_bytes"] == 8 * 2**30
    assert manifest["resource_limits"]["max_process_tree_rss_bytes"] == 6_000_000_000
    assert manifest["resource_limits"]["max_affinity_cores"] == 4
    for rel, expected in manifest["source_sha256"].items():
        if manifest["status"] != "PREREGISTERED" and rel == "README.md":
            continue
        actual = hashlib.sha256((HERE / rel).read_bytes()).hexdigest()
        assert actual == expected, f"hash mismatch for {rel}"
    repo_root = HERE.parents[2]
    for rel, expected in manifest["parent_sources"].items():
        raw = subprocess.check_output(
            ["git", "-C", str(repo_root), "show", f"{manifest['source_base']}:{rel}"]
        )
        actual = hashlib.sha256(raw).hexdigest()
        assert actual == expected, f"parent_sources hash mismatch for {rel}"
