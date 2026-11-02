"""Data-free H219 tests; no corpus, model checkpoint, GPU or real S3 calls."""

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
        "h219_test_" + name, HERE / (name + ".py")
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
    store.prefix = "H219/unit/"
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
    monkeypatch.setattr(launch.time, "time", lambda: 1793622601)
    with pytest.raises(RuntimeError, match="FIFO admission deadline"):
        with launch.fifo_lock():
            pytest.fail("jumped older ticket")
    assert (
        old.read_text() == "keep"
        and (queue / "2-200-H219").exists()
        and (queue / "2-200-H219.done").exists()
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


sci = load("science")
backend = load("backend")
runner = load("runner")


def legacy_module():
    spec = importlib.util.spec_from_file_location(
        "h219_legacy_test", HERE.parent / "h109_family_balanced_accum/runner.py"
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


def test_h109_sampler_balances_families_and_pairs_fbal_schedule():
    legacy = legacy_module()
    groups = {"large": [str(i) for i in range(100)], "small": ["small"]}
    seed = sci.training_seed(219101, 0)
    control = legacy.sample_microbatches(groups, "CONTROL", seed)
    balanced = legacy.sample_microbatches(groups, "FBAL", seed)
    assert balanced == legacy.sample_microbatches(groups, "FBAL+WCR", seed)
    assert len(balanced) == 160 and all(len(b) == 16 for b in balanced)
    assert sum(c == "small" for b in control for c in b) < 100
    assert 1100 < sum(c == "small" for b in balanced for c in b) < 1450
    assert [legacy.learning_rate("FBAL+WCR", i) for i in range(3)] == [5e-5, 1e-4, 1e-4]
    assert legacy.learning_rate("FBAL+WCR", 16) > legacy.learning_rate("FBAL+WCR", 15)


def test_exact_merged_pr_gate_refuses_open_pr_and_wrong_source():
    launch = load("launch")
    pr = dict(
        headRefName="rd/h219-fbal",
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
    panels["FBAL"][:] = 0.001
    assert sci.readout(panels, families)["verdict"] == "NO_ADVANCE"
    panels["FBAL"][:] = 0.02
    panels["FBAL_WCR"][:] = 0.03
    good = sci.readout(panels, families)
    assert (
        good["verdict"] == "CANDIDATE"
        and good["holm_multiplicity"] == 2
        and good["contrasts"]["FBAL_vs_CONTROL"]["median_gain"] == pytest.approx(0.02)
    )
    panels["FBAL"][0, :] = -0.01
    assert sci.readout(panels, families)["verdict"] == "NO_ADVANCE"
    panels["FBAL"][:] = 0.1
    panels["FBAL"][:, :30] = -0.02
    assert sci.readout(panels, families)["verdict"] == "NO_ADVANCE"
    panels["FBAL"][0, 0] = np.nan
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
    e.anchor = lambda kw: ()
    calls = []

    def batch(cids, indices, allowed):
        assert (
            len(cids) == 16
            and set(cids) <= allowed
            and all(len(ix) == 24 for ix in indices)
        )
        calls.append(cids)
        return (
            dict(
                prog_412=None,
                sos_prior=None,
                scale=None,
                revealed_indices=None,
                revealed_raw_f=None,
                revealed_gate=None,
            ),
            None,
            None,
            None,
            None,
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
    rows = [dict(context_sha256=str(i), family_id=str(i)) for i in range(8)]
    _, a = e.train(
        rows, "CONTROL", 219101, 0, "control", legacy, sci, tmp_path, checkpoint
    )
    _, b = e.train(
        rows, "FBAL_WCR", 219101, 0, "wcr", legacy, sci, tmp_path, checkpoint
    )
    assert len(calls) == 320 and e.progress["updates"] == 80
    assert a["before_sha256"] == b["before_sha256"] == digest(e.initial)
    assert all(
        p["updates"] == 40
        and p["microbatches"] == 160
        and p["gradient_squared_sum"] > 0
        and p["forward_changed"]
        and p["before_sha256"] != p["after_sha256"]
        for p in [a, b]
    )
    assert len(events) == 2 and b["learning_rates"][0] == 5e-5


def test_full_protocol_locks_all_45_fits_before_any_held_out_utility(tmp_path):
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
                progress["updates"] == 1804
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
        result["optimizer_updates"] == 1800
        and result["smoke_optimizer_updates"] == 4
        and result["member_fits"] == 45
        and result["verdict"] == "NO_ADVANCE"
    )
    assert events.index("selection-lock.json") < events.index("A-cell/219101-0")
