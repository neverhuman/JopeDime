"""CPU-only numerical, leakage and budget tests; no real corpus payloads."""
import fcntl
import importlib.util
import subprocess
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location('h76_test_runner', HERE / 'runner.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
import run_wave as wave  # noqa: E402


@pytest.mark.parametrize('arm', r.ARMS)
def test_step_matches_independent_adamw_reference_and_freezes_other_weights(arm):
    weight = torch.nn.Parameter(torch.tensor([.4, -.7]))
    bias = torch.nn.Parameter(torch.tensor([.2]))
    frozen = torch.nn.Parameter(torch.tensor([3.]), requires_grad=False)
    original = [p.detach().clone() for p in (weight, bias)]
    optimizer = torch.optim.AdamW([weight, bias], lr=1e-4, weight_decay=1e-4)
    calls = []
    def closure():
        calls.append([weight.detach().clone(), bias.detach().clone()])
        return (weight.square().sum() + bias.square().sum()) / 2 + frozen.sum()
    r.optimizer_step([('weight', weight), ('bias', bias), ('frozen', frozen)], optimizer, closure, arm)
    reference = [torch.nn.Parameter(x.clone()) for x in original]
    refopt = torch.optim.AdamW(reference, lr=1e-4, weight_decay=1e-4)
    if arm in ('SAM', 'ASAM'):
        scales = [original[0].abs() + .01 if arm == 'ASAM' else torch.ones_like(original[0]),
                  torch.ones_like(original[1])]
        norm = torch.cat([(x*t).flatten() for x, t in zip(original, scales, strict=True)]).norm()
        rho = .5 if arm == 'ASAM' else .05
        perturbed = [x + rho*t*t*x/(norm+1e-16) for x, t in zip(original, scales, strict=True)]
        for actual, expected in zip(calls[1], perturbed, strict=True):
            torch.testing.assert_close(actual, expected)
        for p, g in zip(reference, perturbed, strict=True):
            p.grad = g.clone()
    else:
        for p, g in zip(reference, original, strict=True):
            p.grad = g.clone()
    torch.nn.utils.clip_grad_norm_(reference, 1.)
    refopt.step()
    for p, expected in zip((weight, bias), reference, strict=True):
        torch.testing.assert_close(p, expected, rtol=0, atol=0)
    assert len(calls) == (2 if arm in ('SAM', 'ASAM') else 1)
    assert frozen.item() == 3
    assert optimizer.state[weight]['step'] == 1


@pytest.mark.parametrize('arm', ['SAM', 'ASAM'])
def test_second_pass_failure_restores_exact_weights_and_skips_update(arm):
    p = torch.nn.Parameter(torch.tensor([.1234567]))
    original = p.detach().clone()
    opt = torch.optim.AdamW([p])
    count = 0
    def closure():
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError('second pass fails')
        return p.square().sum()
    with pytest.raises(RuntimeError, match='second pass'):
        r.optimizer_step([('weight', p)], opt, closure, arm)
    assert torch.equal(original, p)
    assert not opt.state


def test_dropout_randomness_replayed_for_both_passes():
    p = torch.nn.Parameter(torch.ones(10))
    draws = []
    def closure():
        x = torch.rand(10)
        draws.append(x)
        return (p*x).sum()
    r.optimizer_step([('weight', p)], torch.optim.AdamW([p]), closure, 'SAM')
    assert torch.equal(*draws)


def test_batch_pairing_preserves_h59_first_two_keys_and_round_candidate_streams():
    pools = {str(i): {'fitness': np.arange(200)} for i in range(20)}
    def sampler(pool, order, size):
        return np.random.choice(len(order), size, replace=False)
    before = np.random.get_state()
    a = r.batch_plan(pools, sampler, 12)
    b = r.batch_plan(pools, sampler, 12)
    after = np.random.get_state()
    assert np.array_equal(before[1], after[1])
    assert len(a) == 160
    for x, y in zip(a, b, strict=True):
        assert x.context_ids == y.context_ids
        assert len(x.context_ids) == 16
        assert all(np.array_equal(i, j) for i, j in zip(x.candidate_indices, y.candidate_indices, strict=True))
        if x.step_index < 2:
            assert x.seed == r.h59.paired_stream_seed(12, x.round_index, x.step_index)


@pytest.mark.parametrize('path', ['sealed/file', 'holdout/file', 'meta-test/file', 'flywheel-full-20261030/r882.pt'])
def test_protected_paths_rejected_before_read(tmp_path, path):
    with pytest.raises(RuntimeError, match='protected/live'):
        r.safe_path(tmp_path / path)


def test_symlink_alias_to_protected_payload_rejected(tmp_path):
    target = tmp_path / 'sealed'
    target.mkdir()
    alias = tmp_path / 'ordinary'
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(RuntimeError):
        r.safe_path(alias / 'file')


def fixture_cells(gain):
    result = []
    for arm in r.ARMS:
        for seed in r.SEEDS:
            value = gain if arm != 'CONTROL' else 0
            score = {'per_family': {str(i): value for i in range(19)}, 'mean': value}
            result.append({'arm': arm, 'seed': seed, 'train': score, 'dev': score,
                           'timing_only': False, 'updates': 160 if arm == 'K8' else 40,
                           'trainable_count': 29211, 'dope_before': r.h59.DOPE_SHA,
                           'dope_after': r.h59.DOPE_SHA, 'checkpoint_sha256': r.h59.R882_SHA,
                           'manifest_sha256': r.h59.MANIFEST_SHA, 'selected_sha256': r.SELECTED_SHA,
                           'initial_adapter_sha256': 'before', 'final_adapter_sha256': 'after',
                           'prereg_sha256': 'prereg', 'source_commit': 'source', 'folds': 'folds'})
    return result


@pytest.mark.parametrize('gain, admitted', [(0, False), (r.MARGIN, False), (.002, True), (-.002, False)])
def test_admission_strict_margin_and_ties(gain, admitted):
    metrics = r.summarize(fixture_cells(gain), draws=100)
    assert metrics['arms']['SAM']['admitted'] is admitted
    assert metrics['arms']['CONTROL']['elo'] == 1500
    assert metrics['arms']['SAM']['median_gain'] == pytest.approx(gain)


def test_missing_duplicate_or_changed_family_prevents_admission():
    cells = fixture_cells(.1)
    with pytest.raises(RuntimeError):
        r.summarize(cells[:-1])
    cells[-1] = cells[0]
    with pytest.raises(RuntimeError):
        r.summarize(cells)
    cells = fixture_cells(.1)
    cells[-1]['dev'] = {'per_family': {'other': .1}, 'mean': .1}
    with pytest.raises(RuntimeError):
        r.summarize(cells)


def test_elo_and_extremes_are_json_safe():
    assert r.elo_delta(.5) == 0
    assert r.elo_delta(.75) == pytest.approx(400*math.log10(3))
    assert r.elo_delta(0) == '-Infinity'
    assert r.elo_delta(1) == 'Infinity'


def test_budget_includes_voids_and_refuses_uncertain_resume():
    ledger = {'cells': [{'wall_seconds': 100, 'error': 'void'}]}
    assert wave.remaining(ledger) == 2600
    with pytest.raises(RuntimeError, match='unaccounted'):
        wave.remaining(dict(ledger, in_progress={'arm': 'SAM'}))
    with pytest.raises(RuntimeError, match='exhausted'):
        wave.remaining({'cells': [{'wall_seconds': 2700}]})
    assert wave.feasible(100)
    assert not wave.feasible(200)


def test_unheld_lock_descriptor_rejected(tmp_path):
    with (tmp_path / 'unrelated.lock').open('w') as handle:
        with pytest.raises(RuntimeError):
            r.assert_lease(handle.fileno())


def test_revealed_anchor_cannot_win_terminal_metric():
    h61 = r.load('h76_test_h61', HERE.parent / 'h61/runner.py')
    pools = {'x': {'fitness': np.array([-4, -4, -4, -4, -.2, -.8]),
                   'gate': np.ones(6, dtype=bool), 'rec': {'family_id': 'f'}}}
    evaluated = h61.evaluate_pool_predictions(pools, pools, {'x': np.array([-10, -9, -8, -7, 0, 1])})
    assert evaluated['chosen_indices']['x'] == 4
    assert evaluated['mean'] == .2


def test_actual_inherited_parent_flock_is_required(tmp_path):
    path = tmp_path / 'qualification.lock'
    with path.open('a+') as handle:
        script = ("import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
                  "from runner import assert_lease; assert_lease(int(sys.argv[2]), Path(sys.argv[3]))")
        command = [sys.executable, '-c', script, str(HERE), str(handle.fileno()), str(path)]
        unheld = subprocess.run(command, pass_fds=(handle.fileno(),), capture_output=True, timeout=20)
        assert unheld.returncode != 0
        assert b'parent does not own' in unheld.stderr
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        held = subprocess.run(command, pass_fds=(handle.fileno(),), capture_output=True, timeout=20)
        assert held.returncode == 0, held.stderr


@pytest.mark.parametrize('field,value', [('dope_after', 'changed'), ('updates', 1),
                                       ('prereg_sha256', 'different')])
def test_mixed_or_incomplete_cell_proofs_fail(field, value):
    cells = fixture_cells(.01)
    cells[-1][field] = value
    with pytest.raises(RuntimeError):
        r.summarize(cells)


def test_family_bootstrap_preserves_seed_vectors():
    cells = fixture_cells(0)
    gains = np.linspace(-.02, .03, 19)
    for cell in cells:
        if cell['arm'] == 'SAM':
            cell['dev'] = {'mean': float(gains.mean()),
                           'per_family': {str(i): float(gains[i]) for i in range(19)}}
    result = r.summarize(cells, draws=100)['arms']['SAM']
    # Independent scalar replay of the specified cluster draws and aggregation.
    names = sorted(str(i) for i in range(19))
    draws = np.random.default_rng(76000).integers(0, 19, size=(100, 19))
    means = [np.mean([gains[int(names[i])] for i in draw]) for draw in draws]
    assert result['ci95'] == pytest.approx(np.quantile(means, [.025, .975]))
    assert result['lcb95'] == pytest.approx(np.quantile(means, .05))
