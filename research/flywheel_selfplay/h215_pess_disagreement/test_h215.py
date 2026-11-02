"""Data-free H215 protocol/guard tests; no checkpoints, corpora or GPU access."""

import ast
import importlib.util
import io
import json
from pathlib import Path
import types
import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(
        "h215_test_" + name, HERE / (name + ".py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sci = load("science")
backend = load("backend")
runner = load("runner")


def test_zero_penalty_preserves_native_first_argmin_including_ties():
    for dtype in [np.float32, np.float64]:
        c = np.array([1, -0.0, 0.0, 2], dtype=dtype)
        assert sci.pick(c, np.array([1e200, 1e200, 0, 0]), 0) == int(c.argmin()) == 1
    assert sci.pick(np.array([0.0, 1.0]), np.array([2.0, 0.0]), 1) == 1
    assert sci.pick(np.array([0.0, 1.0]), np.array([2.0, 0.0]), -1) == 0


def test_disagreement_is_member_std_of_within_context_zscores():
    c = np.tile([0.0, 1.0, 2.0], (10, 1))
    assert np.array_equal(sci.disagreement(c), np.zeros(3))
    c[-1] = [2, 1, 0]
    expected = ((c - c.mean(1, keepdims=True)) / c.std(1, keepdims=True)).std(0)
    assert np.allclose(sci.disagreement(c), expected)
    assert np.array_equal(sci.disagreement(np.ones((10, 4))), np.zeros(4))
    with pytest.raises(ValueError):
        sci.disagreement(c[:9])
    with pytest.raises(ValueError):
        sci.pick([0, np.nan], [0, 1], 1)


def test_family_outer_inner_disjointness_and_exact_coverage():
    rows = [dict(family_id=str(i), context_sha256=str(i)) for i in range(205)]
    partitions = []
    for seed in sci.SEEDS:
        all_a = []
        all_b = []
        for outer in range(5):
            fit, b, a = sci.partition(rows, seed, outer)
            f, bf, af = [{r["family_id"] for r in x} for x in [fit, b, a]]
            assert (len(f), len(bf), len(af)) == (123, 41, 41)
            assert not (f & bf or f & af or bf & af)
            seen = []
            for inner in range(3):
                train, held = sci.inner_partition(fit, seed, outer, inner)
                t, h = [{r["family_id"] for r in x} for x in [train, held]]
                assert len(t) == 82 and len(h) == 41 and not (t & h) and t | h == f
                seen += sorted(h)
            assert len(seen) == len(set(seen)) == 123
            all_a += sorted(af)
            all_b += sorted(bf)
        assert len(all_a) == len(set(all_a)) == len(all_b) == len(set(all_b)) == 205
        partitions.append(sci.partition(rows, seed, 0)[2])
    assert partitions[0] != partitions[1] != partitions[2]


def test_all_600_member_seeds_unique_and_budget_exact():
    seeds = [
        sci.member_seed(s, o, p, m)
        for s in sci.SEEDS
        for o in range(5)
        for p in range(4)
        for m in range(10)
    ]
    assert len(set(seeds)) == 600 and min(seeds) == 215201 and max(seeds) == 215800
    assert len(seeds) * 40 == 24000


def test_fit_only_grid_ties_fallback_and_equal_family_weight():
    fam = list(map(str, range(123)))
    gains = {k: np.zeros(123) for k in sci.KAPPAS}
    assert sci.choose_kappa(gains, fam)[0] == 0
    gains[0.5][:] = 0.1
    gains[1][:] = 0.1
    assert sci.choose_kappa(gains, fam)[0] == 0.5
    with pytest.raises(ValueError):
        sci.choose_kappa(gains, fam[:-1])


def test_primary_gate_cannot_be_won_by_diagnostic_or_elo():
    c = dict(median_gain=0.1, ci95=[0.01, 0.2], harm=0.1)
    assert sci.verdict(c, 0.01, True, True) == "CANDIDATE"
    for args in [
        (c, 0.05, True, True),
        (c, 0.01, False, True),
        (c, 0.01, True, False),
        (dict(c, ci95=[0, 0.2]), 0.01, True, True),
        (dict(c, harm=0.1001), 0.01, True, True),
    ]:
        assert sci.verdict(*args) == "NO_ADVANCE"


def test_labels_refuse_wrong_role_or_unlocked_a_before_any_read():
    e = backend.Engine.__new__(backend.Engine)
    e.progress = {"locked": False}
    e.labels_cache = {}
    e.pools = types.SimpleNamespace(blob=lambda cid: pytest.fail("unauthorized read"))
    with pytest.raises(RuntimeError):
        e.labels("x", {"y"}, "FIT")
    with pytest.raises(RuntimeError):
        e.labels("x", {"x"}, "DL-Val")
    with pytest.raises(RuntimeError):
        e.labels("x", {"x"}, "A")


@pytest.mark.parametrize("name", ["runner", "launch"])
def test_guard_301_seconds_and_recovery(name):
    g = load(name).SustainedLoad()
    assert not g.overloaded(90, 1, 1, 100, 0)
    assert not g.overloaded(1, 1, 1, 100, 1)
    assert not g.overloaded(90, 1, 1, 100, 2)
    assert not g.overloaded(90, 1, 1, 100, 301)
    assert g.overloaded(90, 1, 1, 100, 303)
    assert load(name).SustainedLoad().overloaded(1, 86, 1, 100, 0)


def test_actual_worker_guard_full_swap_not_an_admission_stop():
    sample = dict(
        load=1,
        load5=1,
        cores=100,
        cpu=1,
        available=32 * 2**30,
        swap_free=0,
        disk_used_fraction=0.9,
        free=50_000_000_000,
        rss=1_000_000,
    )
    runner.guard(sample, state=runner.SustainedLoad(), now=0, admission=True)
    for key, value in [
        ("available", 15 * 2**30),
        ("disk_used_fraction", 0.939),
        ("free", 39_000_000_000),
        ("rss", 6_000_000_000),
    ]:
        with pytest.raises(RuntimeError):
            runner.guard(
                dict(sample, **{key: value}), state=runner.SustainedLoad(), now=1
            )


def test_source_calls_native_once_with_no_duplicate_token_kwargs():
    # Inspect the actual nested production forward; duplicated token kwargs caused
    # a prelaunch defect in the first draft and must never consume the attempt.
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


def test_native_gate_requires_full_exact_vector():
    a = np.zeros(91)
    a[-1] = 1.1377941898164525 * 91
    sci.require_native_reference(a, a.copy())
    b = a.copy()
    b[0] += 1
    b[1] -= 1
    with pytest.raises(RuntimeError):
        sci.require_native_reference(b, a)


def test_fresh_adapter_is_new_and_restoration_preserves_anchor():
    class Toy(torch.nn.Module):
        def __init__(
            self,
            frozen_anchor=None,
            prog_mean=None,
            prog_std=None,
            d_token=32,
            n_modalities=10,
            modality_dropout_p=0,
        ):
            super().__init__()
            self.anchor = frozen_anchor
            self.register_buffer("prog_mean", torch.zeros(4))
            self.register_buffer("prog_std", torch.ones(4))
            self.d_token = d_token
            self.n_modalities = n_modalities
            self.modality_dropout_p = modality_dropout_p
            self.p = torch.nn.ParameterList(
                [torch.nn.Parameter(torch.randn(621)) for _ in range(46)]
                + [torch.nn.Parameter(torch.randn(645))]
            )

    net = torch.nn.Module()
    net.adapter_10tok = Toy(torch.nn.Linear(2, 2))
    names = [
        n
        for n, p in net.named_parameters()
        if not n.startswith("adapter_10tok.anchor.")
    ]
    anchor = {n: t.clone() for n, t in net.adapter_10tok.anchor.state_dict().items()}
    a = backend.random_adapter_state(net, names, 215201)
    b = backend.random_adapter_state(net, names, 215202)
    assert any(not torch.equal(a[n], b[n]) for n in names)
    backend.restore(net, names, a)
    assert all(torch.equal(dict(net.named_parameters())[n], a[n]) for n in names)
    assert all(
        torch.equal(t, net.adapter_10tok.anchor.state_dict()[n])
        for n, t in anchor.items()
    )


def test_s3_create_only_and_exact_version_corruption_refusal(tmp_path):
    store = runner.Store.__new__(runner.Store)
    store.output = tmp_path
    store.prefix = "H215/unit/"
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


def test_exact_open_pr_binding():
    launch = load("launch")
    pr = dict(
        state="OPEN",
        headRefName="rd/h215-pess-disagreement",
        baseRefName="main",
        isDraft=False,
        headRefOid="a" * 40,
    )
    assert launch.admitted_pr(pr, "a" * 40)
    for key, value in [
        ("isDraft", True),
        ("headRefOid", "b" * 40),
        ("state", "CLOSED"),
        ("headRefName", "other"),
    ]:
        assert not launch.admitted_pr(dict(pr, **{key: value}), "a" * 40)


def test_complete_study_role_order_and_fit_budget_without_data_or_gpu(tmp_path):
    study = load("study")
    rows = [dict(family_id=str(i), context_sha256=str(i)) for i in range(205)]
    progress = dict(updates=0, locked=False, source_commit="a" * 40)
    events = []

    class Store:
        def put(self, name, raw):
            events.append(name)

        def arrays(self, name, **arrays):
            events.append(name)

    class Engine:
        names = ()
        net = None
        native = {str(i): np.array([0.0, 1.0, 2.0]) for i in range(205)}
        fit = set()

        def train(self, fit, seed, label):
            self.fit = {r["context_sha256"] for r in fit}
            progress["updates"] += 40
            return {"fit": sorted(self.fit)}, dict(updates=40, member_seed=seed)

        def score(self, cid):
            assert cid not in self.fit
            return np.array([0.0, 1.0, 2.0])

        def utility(self, cid, allowed, role):
            assert cid in allowed
            if role == "A":
                assert progress["locked"] and "selection-lock.json" in events
            return np.array([0.0, -1.0, -2.0])

        def restore(self, state):
            self.fit = set(state["fit"])

    selector = types.SimpleNamespace(
        resolve_h177_verdict=lambda *a: dict(verdict="ADVANCE"),
        H177_DECISION_RULE=types.SimpleNamespace(
            select=lambda rows, base, arms, seed: {
                a: dict(admitted=False) for a in arms
            }
        ),
        holm=lambda x: x,
    )
    rt = types.SimpleNamespace(
        check_frozen=lambda *a: None,
        sha=lambda p: __import__("hashlib").sha256(p.read_bytes()).hexdigest(),
    )
    elo = types.SimpleNamespace(elo_vs_reference=lambda *a, **k: dict(median=1500.0))

    def save(path, obj):
        path.write_text(json.dumps(obj))

    result = study.execute(
        Engine(),
        sci,
        selector,
        {"h177_analysis_sha256": "x"},
        elo,
        rows,
        Store(),
        rt,
        None,
        None,
        progress,
        save,
        tmp_path,
        lambda: None,
    )
    assert result["verdict"] == "NO_ADVANCE" and progress["updates"] == 24000
    assert result["member_fits"] == 600 and all(
        x == [0, 0, 0] for x in result["adoption_counts"].values()
    )
    assert (
        events.index("B-panels")
        < events.index("selection-lock.json")
        < next(i for i, e in enumerate(events) if e.startswith("A-sigma/"))
    )


def test_exact_calibrated_selector_requires_four_panels():
    spec = importlib.util.spec_from_file_location(
        "h215_selector_test", HERE.parent / "h177_pooled_selector/analysis.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [dict(family_id=str(i)) for i in range(205)]
    base = np.zeros((3, 205))
    with pytest.raises(ValueError, match="4-arm"):
        module.H177_DECISION_RULE.select(rows, base, {"CTRL": base}, seed=215150)
    decisions = module.H177_DECISION_RULE.select(
        rows, base, {sci.SELECTOR_NAMES[a]: base for a in sci.ARMS}, seed=215150
    )
    assert all(
        not x["admitted"] and x["holm_multiplicity"] == 4 for x in decisions.values()
    )


def test_actual_member_training_changes_state_forward_and_persists_proof():
    class Toy(torch.nn.Module):
        def __init__(
            self,
            frozen_anchor=None,
            prog_mean=None,
            prog_std=None,
            d_token=32,
            n_modalities=10,
            modality_dropout_p=0,
        ):
            super().__init__()
            self.anchor = frozen_anchor
            self.register_buffer("prog_mean", torch.zeros(4))
            self.register_buffer("prog_std", torch.ones(4))
            self.d_token = d_token
            self.n_modalities = n_modalities
            self.modality_dropout_p = modality_dropout_p
            self.p = torch.nn.ParameterList(
                [torch.nn.Parameter(torch.randn(621)) for _ in range(46)]
                + [torch.nn.Parameter(torch.randn(645))]
            )

    net = torch.nn.Module()
    net.adapter_10tok = Toy()
    e = backend.Engine.__new__(backend.Engine)
    e.net = net
    e.names = tuple(n for n, p in net.named_parameters())
    e.check = lambda: None
    e.progress = {"updates": 0}

    def cost(net, *unused):
        value = sum(p.mean() for p in list(net.adapter_10tok.p)[:41])
        return value.expand(8, 24)

    e.treatment = types.SimpleNamespace(adapter_forward=cost)

    def hash_state(state):
        return (
            __import__("hashlib")
            .sha256(b"".join(v.detach().numpy().tobytes() for v in state.values()))
            .hexdigest()
        )

    e.rt = types.SimpleNamespace(
        tensor_hash=hash_state, loss=lambda pred, f, g: ((pred - f) ** 2).mean()
    )
    e.labels = lambda cid, allowed, role: (np.linspace(-1, 1, 30), np.ones(30, bool))
    e.score = lambda cid: np.array([float(cost(net)[0, 0].detach())])
    kwargs = {
        key: None
        for key in [
            "prog_412",
            "sos_prior",
            "scale",
            "revealed_indices",
            "revealed_raw_f",
            "revealed_gate",
        ]
    }
    e.batch = lambda *a: (
        kwargs,
        None,
        None,
        torch.ones((8, 24)) * 0.25,
        torch.ones((8, 24), dtype=torch.bool),
    )
    e.anchor = lambda kw: (None, None)
    saved = {}
    e.store = types.SimpleNamespace(
        put=lambda name, raw: saved.update({name: json.loads(raw)})
    )
    rows = [dict(context_sha256=str(i), family_id=str(i)) for i in range(8)]
    _, proof = e.train(rows, 215201, "unit")
    assert (
        e.progress["updates"] == 40
        and proof["gradient_squared_sum"] > 0
        and proof["forward_changed"]
    )
    assert proof["before_sha256"] != proof["after_sha256"]
    assert saved["proofs/unit.json"] == proof


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
    monkeypatch.setattr(launch.time, "time", lambda: 1793615401)
    with pytest.raises(RuntimeError, match="FIFO admission deadline"):
        with launch.fifo_lock():
            pytest.fail("jumped older ticket")
    assert (
        old.read_text() == "keep"
        and (queue / "2-200-H215").exists()
        and (queue / "2-200-H215.done").exists()
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
    """Use the production check factory with a deterministic clock."""
    return runner.make_resource_check(
        resources, guard, lambda: {}, tmp_path, 0.0, clock=lambda: now[0]
    )


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
