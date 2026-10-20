"""Real-input structure and actual runner boundaries; no model/device work.

The legacy AST adapters execute only the two reviewed old boundary statements,
so the same tests reproduce the original defects without a GPU training run.
"""
import ast
import copy
import hashlib
import json
import os
import struct
import unittest
import zipfile
from pathlib import Path

import numpy as np
import runner
from protocol import BASELINES


def native_scores(row, count):
    if hasattr(runner, 'native_control_scores'):
        return runner.native_control_scores(row, count)
    tree = ast.parse(Path(runner.__file__).read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    block = next(n for n in ast.walk(main) if isinstance(n, ast.For) and
                 isinstance(n.target, ast.Name) and n.target.id == 'native')
    scope = {'scores': {}, 'controls': {'c': row}, 'cid': 'c'}
    exec(compile(ast.Module(body=[block], type_ignores=[]), '<reviewed-native-boundary>', 'exec'), scope)
    return scope['scores']


def initial_pairs(rng, count):
    if hasattr(runner, 'initial_uniform_acquisitions'):
        return runner.initial_uniform_acquisitions(rng, count)
    tree = ast.parse(Path(runner.__file__).read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    block = next(n for n in ast.walk(main) if isinstance(n, ast.For) and
                 isinstance(n.target, ast.Name) and n.target.id == 'cid' and
                 isinstance(n.iter, ast.Name) and n.iter.id == 'train_ids')
    result = []
    scope = dict(rng=rng, train_ids=['c'], selected_rows={'c': {'members': list(range(count))}},
                 initial=None, acquire=lambda ledger, cid, member, **kw: result.append((member, kw['propensity'])))
    exec(compile(ast.Module(body=[block], type_ignores=[]), '<reviewed-initial-boundary>', 'exec'), scope)
    return result


def authenticated_inputs():
    root = Path(os.environ['JOPE_CONTRACT_INPUTS'])
    expected = {'controls.json': '53c149d7c6dc852eff1893520a08e5b912cc3adbf1a2168573501ce09d7ccb17',
                'features.npz': '9a65e29a56a7c0da6a1b768a841518d9ee4cccc65fa431ceb44b7129db212fc4',
                'preregistration.json': 'bd792588b43d8593ff2e51d9be9c7755ba5c89ba98e81cee702b9fa101bc33b9'}
    for name, sha in expected.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != sha:
            raise AssertionError('real input drift: ' + name)
    controls = json.loads((root / 'controls.json').read_text())
    split = json.loads((root / 'preregistration.json').read_text())['split']
    counts = {}
    # Decode only .npy shape headers, never the feature values or pool outcomes.
    with zipfile.ZipFile(root / 'features.npz') as z:
        for cid, item in split.items():
            if item['partition'] not in ('train', 'calibration'):
                continue
            with z.open(cid + '.npy') as f:
                if f.read(8) != b'\x93NUMPY\x01\x00':
                    raise AssertionError('unexpected frozen NPY header version')
                header = ast.literal_eval(f.read(struct.unpack('<H', f.read(2))[0]).decode())
                counts[cid] = header['shape'][0]
    return controls, split, counts


class RunnerContractTests(unittest.TestCase):
    def setUp(self):
        self.row = {name: [0.1, 0.2, 0.3] for name in BASELINES}

    def test_authentic_control_keys_positive(self):
        controls, split, counts = authenticated_inputs()
        self.assertEqual(len(counts), 713 + 209)
        for cid, count in counts.items():
            try:
                scores = native_scores(controls[cid], count)
            except (KeyError, ValueError) as exc:
                self.fail('actual frozen input must support all four native controls: ' + str(exc))
            self.assertEqual(set(scores), BASELINES)
            self.assertTrue(all(len(v) == count for v in scores.values()))

    def test_missing_or_renamed_control_refuses(self):
        for name in BASELINES:
            row = copy.deepcopy(self.row)
            row[name.replace('_', '.safetensors_', 1)] = row.pop(name)
            with self.assertRaises(ValueError): native_scores(row, 3)

    def test_control_length_refuses(self):
        for count in (2, 4):
            with self.assertRaises(ValueError): native_scores(self.row, count)

    def test_nonfinite_nested_and_boolean_controls_refuse(self):
        for bad in (float('nan'), float('inf'), True, [0.1]):
            row = copy.deepcopy(self.row); row['native429_qhead'][1] = bad
            with self.assertRaises(ValueError): native_scores(row, 3)

    def test_invalid_control_container_or_candidate_count_refuses(self):
        for row, count in (([], 3), (self.row, 0), (self.row, True)):
            with self.assertRaises(ValueError): native_scores(row, count)

    def test_initial_second_conditional_propensity(self):
        pairs = initial_pairs(np.random.default_rng(8102026), 7)
        self.assertEqual([p for _, p in pairs], [1 / 7, 1 / 6])
        self.assertNotEqual(pairs[1][1], 1 / 7)
        self.assertEqual(pairs[0][1] * pairs[1][1], 1 / (7 * 6))

    def test_authentic_training_draws_and_rng_state_unchanged(self):
        _, split, counts = authenticated_inputs()
        reference = np.random.default_rng(8102026); actual = np.random.default_rng(8102026)
        for cid in sorted(c for c in counts if split[c]['partition'] == 'train'):
            n = counts[cid]
            expected = reference.choice(n, 2, replace=False).tolist()
            pairs = initial_pairs(actual, n)
            self.assertEqual([k for k, _ in pairs], expected)
            self.assertEqual([p for _, p in pairs], [1 / n, 1 / (n - 1)])
        self.assertEqual(actual.bit_generator.state, reference.bit_generator.state)

    def test_too_small_initial_pool_refuses(self):
        for n in (0, 1, True):
            with self.assertRaises(ValueError): initial_pairs(np.random.default_rng(41), n)

    def test_actual_main_consumes_same_validators(self):
        tree = ast.parse(Path(runner.__file__).read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        calls = [n for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        controls = [n.lineno for n in calls if n.func.id == 'native_control_scores']
        self.assertEqual(len(controls), 2, 'shared helper must serve preflight AND final evaluation')
        self.assertLess(min(controls), min(n.lineno for n in calls if n.func.id == 'ValueAdapter'))
        self.assertEqual(sum(n.func.id == 'initial_uniform_acquisitions' for n in calls), 1)


if __name__ == '__main__': unittest.main()
