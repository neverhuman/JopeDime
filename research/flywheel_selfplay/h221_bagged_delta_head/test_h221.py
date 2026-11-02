"""Data-free H221 tests; no corpus, model checkpoint, GPU or real S3 calls."""

import ast

import importlib.util

import io

import json

from pathlib import Path

import types

import hashlib

import numpy as np

import pytest

import torch

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(
        "h221_test_" + name, HERE / (name + ".py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("name", ["runner", "launch"])
def test_guard_301_seconds_and_recovery(name):
    g = load(name).SustainedLoad()
    assert not g.overloaded(90, 1, 1, 100, 0)
    assert not g.overloaded(1, 1, 1, 100, 1)
    assert not g.overloaded(90, 1, 1, 100, 2)
    assert not g.overloaded(90, 1, 1, 100, 301)
    assert g.overloaded(90, 1, 1, 100, 303)
    assert load(name).SustainedLoad().overloaded(1, 86, 1, 100, 0)


def test_s3_create_only_and_exact_version_corruption_refusal(tmp_path):
    store = runner.Store.__new__(runner.Store)
    store.output = tmp_path
    store.prefix = "H221/unit/"
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
    monkeypatch.setattr(launch.time, "time", lambda: 1793631601)
    with pytest.raises(RuntimeError, match="FIFO admission deadline"):
        with launch.fifo_lock():
            pytest.fail("jumped older ticket")
    assert (
        old.read_text() == "keep"
        and (queue / "2-200-H221").exists()
        and (queue / "2-200-H221.done").exists()
    )


def test_fifo_holds_required_lock_and_retains_ticket_after_release(
    tmp_path, monkeypatch
):
    import fcntl

    launch = load("launch")
    queue = tmp_path / "queue"
    queue.mkdir()
    monkeypatch.setattr(
        launch,
        "Path",
        lambda value: (
            queue if str(value).endswith("qualification.queue") else tmp_path / "lock"
        ),
    )
    with launch.fifo_lock() as ticket:
        assert Path(ticket).exists()
        with (tmp_path / "lock").open("a") as other:
            with pytest.raises(BlockingIOError):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert Path(ticket).exists() and Path(ticket + ".done").exists()
    with (tmp_path / "lock").open("a") as other:
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)


def actual_check(resources, guard, now, tmp_path):
    tree = ast.parse((HERE / "runner.py").read_text())
    run = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run"
    )
    check = next(
        n for n in run.body if isinstance(n, ast.FunctionDef) and n.name == "check"
    )
    factory = ast.parse(
        "def factory():\n    checking=False\n    last_check=0.0\n    return check\n"
    ).body[0]
    factory.body.insert(2, check)
    ns = dict(
        resources=resources,
        guard=guard,
        snapshot=lambda: {},
        json=json,
        time=types.SimpleNamespace(monotonic=lambda: now[0], time=lambda: now[0]),
        absolute_deadline=5400,
        started=0.0,
        args=types.SimpleNamespace(output=tmp_path),
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[])),
            "actual-H221-check",
            "exec",
        ),
        ns,
    )
    return ns["factory"]()


def test_real_signal_reentry_remains_safe_with_poll_throttle(tmp_path):
    import signal

    class Raw(io.RawIOBase):
        fired = False

        def writable(self):
            return True

        def write(self, data):
            if not self.fired:
                self.fired = True
                signal.raise_signal(signal.SIGALRM)
            return len(data)

    raw = Raw()
    stream = io.TextIOWrapper(io.BufferedWriter(raw, buffer_size=1))
    calls = []
    now = [10.0]
    check = actual_check(stream, lambda s: calls.append(s), now, tmp_path)
    previous = signal.signal(signal.SIGALRM, check)
    try:
        check()
        assert raw.fired and len(calls) == 1
        check()
        assert len(calls) == 1
        now[0] += 2
        check()
        assert len(calls) == 2
    finally:
        signal.signal(signal.SIGALRM, previous)
        stream.close()


def test_guard_exception_releases_reentry_flag_after_poll_interval(tmp_path):
    now = [10.0]
    calls = []

    def refuse(s):
        calls.append(s)
        raise RuntimeError("guard stop")

    check = actual_check(io.StringIO(), refuse, now, tmp_path)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="guard stop"):
            check()
        now[0] += 2
    assert len(calls) == 2


sci = load("science")

runner = load("runner")


def test_exact_merged_pr_gate_refuses_open_pr_and_wrong_source():
    launch = load("launch")
    pr = dict(
        headRefName="rd/h221-bagged-delta",
        baseRefName="main",
        isDraft=False,
        state="OPEN",
        headRefOid="x",
    )
    assert not launch.admitted_pr(pr, "x")
    pr.update(state="MERGED", mergeCommit={"oid": "y"})
    assert launch.admitted_pr(pr, "y") and not launch.admitted_pr(pr, "x")


def test_actual_resource_guard_uses_20gib_and_200gb_not_percentage():
    sample = dict(
        load=1,
        load5=1,
        cores=100,
        cpu=1,
        available=20 * 2**30,
        swap_free=0,
        disk_used_fraction=0.97,
        free=200_000_000_000,
        rss=1_000_000,
    )
    runner.guard(sample, state=runner.SustainedLoad(), now=0, admission=True)
    for key, value in [
        ("available", 20 * 2**30 - 1),
        ("free", 199_999_999_999),
        ("rss", 20 * 2**30),
    ]:
        with pytest.raises(RuntimeError):
            runner.guard(
                dict(sample, **{key: value}), state=runner.SustainedLoad(), now=1
            )


def test_native91_requires_full_exact_reference_and_known_mean():
    expected = [1.1377941898164525] * 91
    good = np.zeros(91)
    good[0] = 1.1377941898164525 * 91
    sci.require_native_reference(good, good)
    # Constant-array averaging can shift one ULP, so use the original mean as a
    # separate negative gate here; exact historical-vector proof is a runtime gate.
    with pytest.raises(RuntimeError):
        sci.require_native_reference([1.0] * 91, [1.0] * 91)
    with pytest.raises(RuntimeError):
        sci.require_native_reference(expected[:-1], expected)
    with pytest.raises(RuntimeError):
        sci.require_native_reference([1.0] * 90 + [2.0], [1.0] * 91)


def test_real_optimizer_changes_future_forward_and_reference_stays_zero():
    torch.manual_seed(1)
    head = sci.DeltaHead()
    x = torch.randn(12, 744)
    u = torch.linspace(-1, 1, 12)
    before = {k: v.clone() for k, v in head.state_dict().items()}
    initial = head(x, 4).detach().clone()
    opt = torch.optim.AdamW(head.parameters(), lr=0.001, weight_decay=0)
    opt.zero_grad()
    sci.balanced_loss(head(x, 4), u, 4).backward()
    assert sum(float(p.grad.square().sum()) for p in head.parameters()) > 0
    opt.step()
    after = head(x, 4).detach()
    assert after[4] == 0 and not torch.equal(after, initial)
    assert any(not torch.equal(before[k], v) for k, v in head.state_dict().items())
    assert head.net[-1].bias is None


def test_fallback_ties_and_threshold():
    assert sci.pick(np.zeros(3), 2) == 2
    assert sci.pick(np.array([0.1, 0, 0.2]), 1, 0.2) == 1
    assert sci.pick(np.array([0.2, 0.2, 0]), 2, 0.1) == 0
    assert sci.pick(np.array([1.0, 0]), 1, float("inf")) == 1
    with pytest.raises(ValueError):
        sci.pick(np.array([np.nan]), 0)


def test_b_gate_prefers_safer_equal_gain_and_refuses_harm():
    families = [str(i) for i in range(20)]
    gains = {t: np.zeros(20) for t in sci.THRESHOLDS}
    gains[0.0] = np.ones(20) * 0.01
    gains[0.1] = gains[0.0].copy()
    chosen, _ = sci.select_threshold(gains, families)
    assert chosen == 0.1
    gains[0.0][:2] = -1
    gains[0.1][:2] = -1
    assert np.isinf(sci.select_threshold(gains, families)[0])
    gains[0.5] = np.ones(20) * 0.001
    assert np.isinf(sci.select_threshold(gains, families)[0])


def test_role_refusal_before_read(tmp_path):
    pools = runner.Pools([{"context_sha256": "x"}], tmp_path)
    with pytest.raises(RuntimeError, match="role/lock"):
        pools.utility("x", {"x"}, "A", locked=False)
    with pytest.raises(RuntimeError, match="role/lock"):
        pools.utility("x", {"y"}, "FIT")


def test_fold_disjointness_and_freshness():
    rows = [{"family_id": str(i), "context_sha256": str(i)} for i in range(205)]
    partitions = []
    for seed in sci.SEEDS:
        covered = []
        for outer in range(5):
            fit, b, a = sci.partition(rows, seed, outer)
            fs, bs, aa = ({x["family_id"] for x in group} for group in (fit, b, a))
            assert len(fs) == 123 and len(bs) == len(aa) == 41
            assert not (fs & bs or fs & aa or bs & aa)
            covered.extend(sorted(aa))
        assert len(covered) == len(set(covered)) == 205
        partitions.append(sci.partition(rows, seed, 0)[2])
    assert partitions[0] != partitions[1] != partitions[2]


def test_feature_values_and_native_reference_no_labels():
    x = sci.features(
        np.ones((4, 412)),
        np.ones((10, 32)),
        np.zeros(10, bool),
        np.array([2.0, 1.0, 3.0, 4.0]),
    )
    assert x.shape == (4, 744) and np.isfinite(x).all()
    assert np.count_nonzero(x[:, 412:742]) == 0
    assert x[1, -1] == 0


def test_native_gate_rejects_mean_only_and_permuted_panel():
    mean = 1.1377941898164525
    expected = np.array([mean] * 91)
    # Even if the aggregate matches, every context must match its reference.
    observed = expected.copy()
    observed[0] += 0.5
    observed[1] -= 0.5
    with pytest.raises(RuntimeError, match="native91"):
        sci.require_native_reference(observed, expected)
    with pytest.raises(RuntimeError, match="native91"):
        sci.require_native_reference(expected[:90], expected)


def test_native_gate_accepts_exact_vector():
    expected = np.zeros(91, np.float64)
    expected[-1] = 1.1377941898164525 * 91
    sci.require_native_reference(expected.copy(), expected)


def test_bag_family_bootstrap_counts_and_matched_placebo_plan():
    rows = [dict(family_id=str(i // 2), context_sha256=str(i)) for i in range(246)]
    populations = []
    initials = []
    for member in range(5):
        plan, init, pop = sci.fit_plan(rows, 221101, 0, member)
        again = sci.fit_plan(rows, 221101, 0, member)
        assert (plan, init, pop) == again and len(pop) == 123 and len(plan) == 128
        assert len(set(pop)) < 123 and all(len(batch) == 4 for batch in plan)
        family = {r["context_sha256"]: r["family_id"] for r in rows}
        assert all(family[cid] in pop for batch in plan for cid in batch)
        populations.append(pop)
        initials.append(init)
    assert len(set(initials)) == 5 and populations[0] != populations[1]
    _, _, single = sci.fit_plan(rows, 221101, 0, None)
    assert len(set(single)) == 123
    with pytest.raises(ValueError):
        sci.fit_plan(rows, 221101, 0, 5)


def test_mean_predictions_precede_picking_and_gate():
    members = [np.array([0.0, 0.5, -0.1])] * 2 + [np.array([0.0, -0.4, 0.3])] * 3
    mean = sci.ensemble_mean(members)
    assert mean[0] == 0 and sci.pick(mean, 0, 0) == 2
    assert sci.pick(mean, 0, 0.2) == 0
    with pytest.raises(ValueError):
        sci.ensemble_mean(members[:3])
    with pytest.raises(ValueError):
        sci.ensemble_mean([np.array([0.0, np.nan])])


def test_owner_gate_strict_point_margin_two_seeds_and_primary_only():
    families = list(map(str, range(205)))
    base = np.zeros((3, 205))
    c = sci.contrast(np.ones_like(base) * 0.002, base, families)
    assert sci.verdict(c) == "ADVANCE"
    assert sci.verdict(dict(c, median_gain=0.001)) == "NO_ADVANCE"
    assert sci.verdict(dict(c, positive_seeds=1)) == "NO_ADVANCE"
    assert sci.verdict(dict(c, positive_seeds=2)) == "ADVANCE"
    for bad in [dict(ci95=[0, 0.1]), dict(p=0.05), dict(harm=0.10001)]:
        assert sci.verdict(dict(c, **bad)) == "NO_ADVANCE"
    assert sci.verdict(sci.contrast(base, base, families)) == "NO_ADVANCE"


def test_actual_128_update_bag_fit_cache_and_shuffled_pair(tmp_path):
    backend = load("backend")
    torch.set_num_threads(1)
    rows = [dict(context_sha256=str(i), family_id=str(i // 2)) for i in range(6)]
    native = {str(i): np.arange(16, dtype=np.float32) for i in range(6)}
    calls = []

    class Pools:
        outcome_reads = []

        def predictors(self, cid):
            calls.append(cid)
            return (
                np.arange(16 * 412, dtype=np.float32).reshape(16, 412) / 1000
                + int(cid),
                None,
                None,
            )

        def utility(self, cid, allowed, role, locked=False):
            assert cid in allowed
            return np.linspace(-1, 1, 16)

    def digest(state):
        return hashlib.sha256(
            b"".join(v.detach().cpu().numpy().tobytes() for v in state.values())
        ).hexdigest()

    events = []
    store = types.SimpleNamespace(put=lambda name, raw: events.append(name))
    progress = dict(updates=0, locked=False)
    engine = backend.Engine(
        sci,
        types.SimpleNamespace(tensor_hash=digest),
        Pools(),
        native,
        np.ones((6, 10, 32), np.float32),
        np.ones((6, 10), bool),
        {str(i): i for i in range(6)},
        lambda: None,
        progress,
        store,
        device="cpu",
    )
    real, p = engine.fit(rows, 221101, 0, 0, False)
    shuf, q = engine.fit(rows, 221101, 0, 0, True)
    assert progress["updates"] == 256 and len(set(calls)) == len(calls)
    assert (
        p["before_sha256"] == q["before_sha256"]
        and p["context_plan_sha256"] == q["context_plan_sha256"]
        and p["bootstrap_families"] == q["bootstrap_families"]
    )
    assert p["after_sha256"] != q["after_sha256"]
    assert all(
        x["updates"] == 128 and x["gradient_squared_sum"] > 0 and x["forward_changed"]
        for x in [p, q]
    )
    engine.utility("0", {"0"}, "FIT")
    with pytest.raises(RuntimeError):
        engine.utility("0", {"0"}, "A")
    with pytest.raises(RuntimeError):
        engine.utility("0", {"other"}, "FIT")
    for cid, x in engine.features.items():
        expected = sci.features(
            Pools().predictors(cid)[0],
            np.ones((10, 32)),
            np.ones(10, bool),
            native[cid],
        )
        assert np.array_equal(x.numpy(), expected)
    assert len(events) == 4


def test_all_165_heads_and_45_b_gates_lock_before_a(tmp_path):
    study = load("study")
    rows = [dict(context_sha256=str(i), family_id=str(i)) for i in range(205)]
    events = []

    class Store:
        def put(self, name, raw):
            events.append(name)

        def arrays(self, name, **kwargs):
            events.append(name)

    class Engine:
        progress = dict(updates=0, locked=False)
        native = {str(i): np.array([0.0, 1.0]) for i in range(205)}
        rt = types.SimpleNamespace(
            tensor_hash=lambda s: hashlib.sha256(
                json.dumps(s, sort_keys=True).encode()
            ).hexdigest()
        )

        def fit(self, fit, seed, outer, member, shuffled):
            self.progress["updates"] += 128
            return {"fit": sorted(r["context_sha256"] for r in fit)}, dict(updates=128)

        def predict(self, states, held):
            assert len(states) in (1, 5)
            assert all(
                r["context_sha256"] not in s["fit"] for s in states for r in held
            )
            return {r["context_sha256"]: np.array([0.0, 0.02]) for r in held}

        def utility(self, cid, allowed, role):
            assert cid in allowed
            if role == "A":
                assert (
                    self.progress["updates"] == 21120
                    and self.progress["locked"]
                    and "selection-lock.json" in events
                )
            return np.array([0.0, 0.02])

    engine = Engine()
    res = study.execute(
        engine,
        sci,
        rows,
        Store(),
        tmp_path,
        "fixture",
        runner.save,
        lambda p: hashlib.sha256(p.read_bytes()).hexdigest(),
        lambda: None,
        types.SimpleNamespace(elo_vs_reference=lambda *a, **kw: {"median": 1500.0}),
    )
    assert (
        res["member_fits"] == 165
        and res["optimizer_updates"] == 21120
        and res["primary"] == "BAG5_SAFE"
        and res["holm_multiplicity"] == 1
        and res["verdict"] == "ADVANCE"
    )
    assert (
        len([x for x in events if x.startswith("B/")]) == 45
        and len([x for x in events if x.startswith("A/")]) == 45
    )


def test_h213_architecture_feature_loss_and_gate_formulas_are_unchanged():
    old = ast.parse((HERE.parent / "h213_gpu_delta_sustained/science.py").read_text())
    new = ast.parse((HERE / "science.py").read_text())
    for name in [
        "DeltaHead",
        "features",
        "balanced_loss",
        "pick",
        "select_threshold",
        "require_native_reference",
    ]:
        a = next(n for n in old.body if getattr(n, "name", None) == name)
        b = next(n for n in new.body if getattr(n, "name", None) == name)
        assert ast.dump(a) == ast.dump(b), name
