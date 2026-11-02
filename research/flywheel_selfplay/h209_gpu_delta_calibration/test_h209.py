"""Data-free H209 refusal, gradient/forward and role controls."""

import importlib.util
from pathlib import Path
import io
import zipfile

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(
        "h209_test_" + name, HERE / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sci, runner = load("science"), load("runner")


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


def test_reveal_reader_limits_decoded_values():
    b = io.BytesIO()
    np.savez_compressed(b, fitness=np.arange(10, dtype=np.float64))
    with zipfile.ZipFile(io.BytesIO(b.getvalue())) as z:
        assert np.array_equal(runner.array_prefix(z, "fitness", 4), np.arange(4))
        with pytest.raises(ValueError):
            runner.array_prefix(z, "fitness", 11)


def test_resource_guard_uses_current_load_and_reserved_blocks():
    sample = dict(
        load=1,
        cores=100,
        cpu=1,
        available=32 * 2**30,
        disk_used_fraction=0.9,
        free=50_000_000_000,
        rss=1000000,
    )
    runner.guard(sample)
    for key, value in (
        ("load", 86),
        ("cpu", 86),
        ("disk_used_fraction", 920 / (920 + 50)),
        ("free", 39_000_000_000),
        ("rss", 6_000_000_000),
    ):
        with pytest.raises(RuntimeError):
            runner.guard(dict(sample, **{key: value}))


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


def test_primary_gate_and_pooled_constant_gain():
    families = [str(i) for i in range(20)]
    base = np.zeros((3, 20))
    result = sci.contrast(base + 0.1, base, families)
    assert result["ci95"][0] > 0 and result["harm"] == 0
    assert sci.verdict(result) == "CANDIDATE"
    assert sci.verdict(dict(result, harm=0.10001)) == "NO_ADVANCE"
    assert sci.verdict(dict(result, p=0.05)) == "NO_ADVANCE"


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


def test_s3_create_only_exact_version_and_corruption_refusal(tmp_path):
    store = runner.Store.__new__(runner.Store)
    store.output, store.prefix, store.receipts = tmp_path, "H209/test/", {}

    class Fake:
        corrupt = False

        def put_object(self, **kwargs):
            assert kwargs["IfNoneMatch"] == "*"
            self.body = kwargs["Body"]
            return {"VersionId": "immutable-v1"}

        def get_object(self, **kwargs):
            assert kwargs["VersionId"] == "immutable-v1"
            return {
                "VersionId": "immutable-v1",
                "Body": io.BytesIO(b"wrong" if self.corrupt else self.body),
            }

    store.client = Fake()
    receipt = store.put("good", b"proof")
    assert receipt["sha256"] == runner.digest(b"proof")
    store.client.corrupt = True
    with pytest.raises(RuntimeError, match="recovery"):
        store.put("bad", b"proof")
    assert not (tmp_path / "receipt-bad.json").exists()


def test_unmerged_launch_refused_before_gpu_or_claim(monkeypatch):
    import types

    launch = load("launch")
    monkeypatch.setattr(launch.os, "nice", lambda n: None)
    monkeypatch.setattr(
        launch.os, "uname", lambda: types.SimpleNamespace(nodename="xbabe2")
    )
    monkeypatch.setattr(
        launch.subprocess,
        "check_output",
        lambda *a, **k: '{"state":"OPEN","mergeCommit":null}',
    )
    monkeypatch.setattr(
        launch,
        "process_gpu_pids",
        lambda: pytest.fail("GPU probe before merged source refusal"),
    )
    with pytest.raises(RuntimeError, match="merged preregistration"):
        launch.main(types.SimpleNamespace(pr=1, commit="a" * 40))


def make_actual_resource_check(resources, guard):
    """Compile the production closure, without starting a model or worker."""
    import ast
    import json
    import time
    import types

    tree = ast.parse((HERE / "runner.py").read_text())
    run = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run"
    )
    check = next(
        n for n in run.body if isinstance(n, ast.FunctionDef) and n.name == "check"
    )
    factory = ast.parse(
        "def factory():\n    checking = False\n    return check\n"
    ).body[0]
    factory.body.insert(1, check)
    ns = dict(
        resources=resources,
        guard=guard,
        snapshot=lambda: {},
        json=json,
        time=time,
        started=time.monotonic(),
        args=types.SimpleNamespace(output=HERE),
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[])),
            str(HERE / "runner.py"),
            "exec",
        ),
        ns,
    )
    return ns["factory"]()


def test_alarm_during_buffered_write_does_not_reenter_resource_check():
    import signal

    class AlarmDuringWrite(io.RawIOBase):
        fired = False

        def writable(self):
            return True

        def write(self, data):
            if not self.fired:
                self.fired = True
                signal.raise_signal(signal.SIGALRM)
            return len(data)

    raw = AlarmDuringWrite()
    stream = io.TextIOWrapper(io.BufferedWriter(raw, buffer_size=1))
    calls = []
    check = make_actual_resource_check(stream, lambda sample: calls.append(sample))
    previous = signal.signal(signal.SIGALRM, check)
    try:
        check()
        assert raw.fired and len(calls) == 1
        check()
        assert len(calls) == 2
    finally:
        signal.signal(signal.SIGALRM, previous)
        stream.close()


def test_resource_guard_failure_releases_reentry_flag():
    calls = []

    def refuse(sample):
        calls.append(sample)
        raise RuntimeError("resource gate")

    check = make_actual_resource_check(io.StringIO(), refuse)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="resource gate"):
            check()
    assert len(calls) == 2
