"""Data-free H220 tests; no corpus, model checkpoint, GPU or real S3 calls."""

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
        "h220_test_" + name, HERE / (name + ".py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def test_s3_create_only_and_exact_version_corruption_refusal(tmp_path):
    store = runner.Store.__new__(runner.Store)
    store.output = tmp_path
    store.prefix = "H220/unit/"
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
    monkeypatch.setattr(launch.time, "time", lambda: 1793628001)
    with pytest.raises(RuntimeError, match="FIFO admission deadline"):
        with launch.fifo_lock():
            pytest.fail("jumped older ticket")
    assert (
        old.read_text() == "keep"
        and (queue / "2-200-H220").exists()
        and (queue / "2-200-H220.done").exists()
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
            "actual-H220-check",
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
backend = load("backend")
runner = load("runner")


def legacy_module():
    spec = importlib.util.spec_from_file_location(
        "h220_legacy_test", HERE.parent / "h109_family_balanced_accum/runner.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return sci.legacy_config(mod)


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


def test_control_and_mix_share_exact_h109_sampler_and_schedule():
    legacy = legacy_module()
    groups = {"large": [str(i) for i in range(100)], "small": ["small"]}
    seed = sci.training_seed(220101, 0)
    a = legacy.sample_microbatches(groups, sci.LEGACY_ARMS["CONTROL"], seed)
    b = legacy.sample_microbatches(groups, sci.LEGACY_ARMS["MIX"], seed)
    assert a == b and len(a) == 160 and all(len(x) == 16 for x in a)
    assert sum(c == "small" for row in a for c in row) < 100
    assert all(
        legacy.learning_rate(sci.LEGACY_ARMS[arm], i) == 1e-4
        for arm in sci.ARMS
        for i in range(40)
    )


def test_exact_merged_pr_gate_refuses_open_pr_and_wrong_source():
    launch = load("launch")
    pr = dict(
        headRefName="rd/h220-mixup",
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
        ("rss", 6_000_000_000),
    ]:
        with pytest.raises(RuntimeError):
            runner.guard(
                dict(sample, **{key: value}), state=runner.SustainedLoad(), now=1
            )


def test_feature_reader_selects_only_allowed_rows(tmp_path):
    p = tmp_path / "features.npz"
    a = np.arange(30, dtype=np.float32).reshape(3, 2, 5)
    mask = np.ones((3, 2), np.float32)
    np.savez_compressed(
        p, cids=np.array(["train0", "excluded", "train1"]), tokens_10x32=a, mask_10=mask
    )
    ids, tokens, masks = runner.train_features(p, {"train0", "train1"})
    assert (
        ids == ["train0", "train1"]
        and np.array_equal(tokens, a[[0, 2]])
        and np.array_equal(masks, mask[[0, 2]])
    )
    with pytest.raises(RuntimeError):
        runner.train_features(p, {"missing"})


def test_readout_strict_margin_holm_harm_all_seeds_and_control_reference():
    families = list(map(str, range(205)))
    zero = np.zeros((3, 205))
    panels = {a: zero.copy() for a in ("R882",) + sci.ARMS}
    assert sci.readout(panels, families)["verdict"] == "NO_ADVANCE"
    panels["MIX"][:] = 0.001
    assert sci.readout(panels, families)["verdict"] == "NO_ADVANCE"
    panels["MIX"][:] = 0.02
    good = sci.readout(panels, families)
    assert (
        good["verdict"] == "CANDIDATE"
        and good["holm_multiplicity"] == 1
        and good["contrasts"]["MIX_vs_CONTROL"]["median_gain"] == pytest.approx(0.02)
    )
    panels["MIX"][0, :] = -0.01
    assert sci.readout(panels, families)["verdict"] == "NO_ADVANCE"
    panels["MIX"][:] = 0.1
    panels["MIX"][:, :30] = -0.02
    assert sci.readout(panels, families)["verdict"] == "NO_ADVANCE"
    panels["MIX"][0, 0] = np.nan
    with pytest.raises(ValueError):
        sci.readout(panels, families)


def test_actual_training_accumulates_160_microbatches_and_resets_r882(tmp_path):
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

    def digest(state):
        return hashlib.sha256(
            b"".join(v.detach().numpy().tobytes() for v in state.values())
        ).hexdigest()

    e.rt = types.SimpleNamespace(
        tensor_hash=digest, loss=lambda pred, f, g: pred.square().mean()
    )
    e.treatment = types.SimpleNamespace(
        adapter_forward=lambda net, *args: sum(
            p.mean() for p in list(net.p)[:41]
        ).expand(16, 24)
    )
    e.score = lambda cid: np.array(
        [float(sum(p.mean() for p in list(net.p)[:41]).detach()), 0.0]
    )
    e.labels = lambda *a: (np.arange(30, dtype=float), np.ones(30, bool))
    e.anchor = lambda kw: (torch.zeros((16, 24)), torch.zeros((16, 24)))
    e.mixup = load("mixup")
    calls = []

    def batch(cids, indices, allowed):
        assert (
            len(cids) == 16
            and set(cids) <= allowed
            and all(len(ix) == 24 for ix in indices)
        )
        calls.append(cids)
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
    rows = [dict(context_sha256=str(i), family_id=str(i // 2)) for i in range(8)]
    _, a = e.train(
        rows, "CONTROL", 220101, 0, "control", legacy, sci, tmp_path, checkpoint
    )
    _, b = e.train(rows, "MIX", 220101, 0, "wcr", legacy, sci, tmp_path, checkpoint)
    assert len(calls) == 480 and e.progress["updates"] == 80
    assert a["before_sha256"] == b["before_sha256"] == digest(e.initial)
    assert all(
        p["updates"] == 40
        and p["microbatches"] == 160
        and p["gradient_squared_sum"] > 0
        and p["forward_changed"]
        and p["before_sha256"] != p["after_sha256"]
        for p in [a, b]
    )
    assert len(events) == 2 and b["learning_rates"][0] == 1e-4
    assert 1000 < b["mixup"]["mixed_rows"] < 1550
    assert a["sampler_plan_sha256"] == b["sampler_plan_sha256"]


def test_full_protocol_locks_all_30_fits_before_any_held_out_utility(tmp_path):
    study = load("study")
    rows = [dict(context_sha256=str(i), family_id=str(i)) for i in range(205)]
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
        native = {str(i): np.array([0.0, 1.0]) for i in range(205)}

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
        ):
            progress["updates"] += steps
            return {"fit": [r["context_sha256"] for r in fit]}, dict(
                updates=steps, seconds_per_update=[0.01] * steps
            )

        def restore(self, state):
            self.fit = set(state["fit"])

        def score(self, cid):
            assert cid not in self.fit
            return np.array([0.0, 1.0])

        def utility(self, cid, allowed, role):
            assert (
                progress["updates"] == 1204
                and progress["locked"]
                and "selection-lock.json" in events
                and role == "A"
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
    )
    assert (
        result["optimizer_updates"] == 1200
        and result["smoke_optimizer_updates"] == 4
        and result["member_fits"] == 30
        and result["verdict"] == "NO_ADVANCE"
    )
    assert events.index("selection-lock.json") < events.index("A-cell/220101-0")


def test_mixup_partners_stay_in_fit_family_singletons_identity_and_determinism():
    mix = load("mixup")
    groups = {"a": ["a0", "a1", "a2"], "b": ["b0"], "c": ["c0", "c1"]}
    family = {c: f for f, cs in groups.items() for c in cs}
    cids = ["a0", "b0", "c0"] * 2000
    partners, weights = mix.plan(cids, groups, family, 220777)
    p2, w2 = mix.plan(cids, groups, family, 220777)
    assert partners == p2 and np.array_equal(weights, w2)
    assert all(family[a] == family[b] for a, b in zip(cids, partners, strict=True))
    assert all(
        b == "b0" and w == 1
        for a, b, w in zip(cids, partners, weights, strict=True)
        if a == "b0"
    )
    actual = [
        (a, b, w) for a, b, w in zip(cids, partners, weights, strict=True) if a != "b0"
    ]
    assert 1800 < sum(a != b for a, b, w in actual) < 2200
    assert all(0 <= w <= 1 for w in weights)


def test_mixup_convex_inputs_targets_frozen_outputs_and_no_input_mutation():
    mix = load("mixup")

    def batch(value, gate):
        x = torch.full((2, 3), float(value))
        return (
            dict(
                prog_412=x,
                revealed_indices=torch.tensor([[0, 1], [0, 1]]),
                revealed_raw_f=x,
                revealed_gate=torch.full((2, 3), float(gate)),
            ),
            x[:, :, None],
            torch.full((2, 3), float(gate)),
            x,
            torch.full((2, 3), gate, dtype=torch.bool),
        )

    left, right = batch(-6, True), batch(10, False)
    a = (torch.zeros((2, 3)), torch.ones((2, 3)))
    b = (torch.ones((2, 3)) * 10, torch.ones((2, 3)) * 5)
    result = mix.batch(left, right, a, b, np.array([0.25, 1.0]))
    kw, tok, mask, target, gate, anchors = result
    assert torch.equal(kw["prog_412"][0], torch.full((3,), 6.0))
    assert torch.equal(target[0], torch.full((3,), 0.5))  # .25*(-4)+.75*2
    assert torch.equal(target[1], torch.full((3,), -4.0))
    assert torch.equal(anchors[0][0], torch.full((3,), 7.5))
    assert torch.equal(kw["revealed_gate"][0], torch.full((3,), 0.25))
    assert gate.all() and torch.equal(left[3], torch.full((2, 3), -6.0))
    assert torch.equal(kw["revealed_indices"], left[0]["revealed_indices"])
    with pytest.raises(ValueError):
        mix.blend(torch.zeros(2, 3), torch.zeros(3, 3), [0.5, 0.5])


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


def test_actual_cohort_uses_dev_key_and_refuses_overlap_and_protected_rows():
    def row(i, split):
        return dict(
            family_id=str(i), dl_split=split, task="regression", source_split="inner"
        )

    packet = dict(
        train=[row(i % 205, "DL-Train") for i in range(1219)],
        dev=[row(205 + i % 19, "DL-Val") for i in range(91)],
    )
    train, val = runner.cohort(packet)
    assert train is packet["train"] and val is packet["dev"]
    packet["dev"][0]["family_id"] = "0"
    with pytest.raises(RuntimeError):
        runner.cohort(packet)
    packet["dev"][0]["family_id"] = "205"
    packet["train"][0]["has_meta_test"] = True
    with pytest.raises(RuntimeError):
        runner.cohort(packet)
