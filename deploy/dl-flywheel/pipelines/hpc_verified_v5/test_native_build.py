import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import textwrap
import unittest
from unittest.mock import patch

import native_build as nb


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def actual_fixture():
    """Captured receipt BODY, and reconstructed bytes matching the actual plan SHA.

    This fixture does not claim that reserializing the receipt reproduces its
    original SHA. Production callers independently read its original raw bytes.
    """
    captured = json.loads((Path(__file__).parent/'fixtures/actual-native-build-and-plan.json').read_bytes())
    plan = captured['actual_plan']; raw = encoded(plan)
    assert len(raw) == 1119
    assert hashlib.sha256(raw).hexdigest() == '7b200996636f3b8eead135dee30ca90339e9b5ca384299f64bc45c95c917346c'
    config = copy.deepcopy(plan)
    config['worker']['bytes'] = 3649816
    config['native_build_receipt']['bytes'] = 67916
    config['source_build_plan_sha256'] = hashlib.sha256(raw).hexdigest()
    ref = {'path': '/fixture/native-v10/build-plan.json', 'sha256': config['source_build_plan_sha256'], 'bytes': len(raw)}
    config['source_pins'] = [ref]
    def reader(request):
        assert request == ref
        return raw
    return config, captured['receipt'], reader


class NativeBuildTests(unittest.TestCase):
    def test_actual_receipt_body_and_exact_plan_bytes_shared_callers(self):
        import native_dispatch as nd
        import operations_launch as op
        c, receipt, reader = actual_fixture()
        self.assertNotIn('plan_sha256', receipt['source'])
        self.assertIs(nd.verify_native_build, op.verify_native_build)
        with patch.object(nd, 'bound', side_effect=reader):
            proof = nd.validate_build(c, receipt)
        self.assertEqual(proof, op.verify_native_build(c, receipt, reader))
        self.assertEqual(proof['source_build_plan_sha256'], c['source_build_plan_sha256'])

    def test_raw_plan_corruption_and_length_rejected_before_join(self):
        for change in (lambda b: b+b'\n', lambda b: b[:-1]+b'!'):
            c, receipt, reader = actual_fixture(); raw = reader(c['source_pins'][0])
            with self.assertRaisesRegex(ValueError, 'raw build-plan'):
                nb.verify_native_build(c, receipt, lambda ref: change(raw))

    def test_wrong_hash_missing_or_ambiguous_plan_ref(self):
        for kind in ('wrong_hash', 'missing', 'duplicate'):
            c, receipt, reader = actual_fixture()
            if kind == 'wrong_hash': c['source_build_plan_sha256'] = '0'*64
            elif kind == 'missing': c['source_pins'] = []
            else: c['source_pins'] += [copy.deepcopy(c['source_pins'][0])]
            with self.assertRaisesRegex(ValueError, 'uniquely pinned'):
                nb.verify_native_build(c, receipt, reader)

    def test_rehashed_plan_must_join_all_current_descriptors_and_source(self):
        for key in sorted(nb.PLAN_FIELDS):
            with self.subTest(key=key):
                c, receipt, reader = actual_fixture(); plan = json.loads(reader(c['source_pins'][0]))
                if key == 'source_commit': plan[key] = '0'*40
                else: plan[key]['sha256'] = '0'*64
                raw = encoded(plan); c['source_build_plan_sha256'] = hashlib.sha256(raw).hexdigest()
                c['source_pins'][0].update(sha256=c['source_build_plan_sha256'], bytes=len(raw))
                with self.assertRaisesRegex(ValueError, 'plan (source|descriptor)'):
                    nb.verify_native_build(c, receipt, lambda ref: raw)

    def test_receipt_source_worker_and_canonicalizer_are_independent(self):
        for key in ('source', 'worker_honest', 'proposal'):
            c, receipt, reader = actual_fixture()
            if key == 'source': receipt['source']['pin'] = '0'*40
            else: receipt['binaries'][key]['sha256'] = '0'*64
            with self.assertRaisesRegex(ValueError, 'receipt (source|binary)'):
                nb.verify_native_build(c, receipt, reader)

    def test_optional_plan_counts_cannot_conflict_or_be_dropped(self):
        c, receipt, reader = actual_fixture()
        # Historical receipt/worker lack counts; current count enrichment passes.
        nb.verify_native_build(c, receipt, reader)
        for key in ('worker', 'native_canonicalizer'):
            wrong = copy.deepcopy(c); wrong[key]['bytes'] += 1
            with self.assertRaises(ValueError): nb.verify_native_build(wrong, receipt, reader)
        wrong = copy.deepcopy(c); del wrong['rustc']['bytes']
        with self.assertRaisesRegex(ValueError, 'plan descriptor'):
            nb.verify_native_build(wrong, receipt, reader)

    def test_descriptor_paths_are_bound_even_with_same_binary_hash(self):
        for key in nb.PLAN_FIELDS - {'source_commit'}:
            c, receipt, reader = actual_fixture(); c[key]['path'] += '.other'
            with self.assertRaisesRegex(ValueError, 'plan descriptor'):
                nb.verify_native_build(c, receipt, reader)

    def test_closed_plan_and_duplicate_json_fields(self):
        for kind in ('extra', 'missing', 'duplicate'):
            c, receipt, reader = actual_fixture(); raw = reader(c['source_pins'][0]); plan = json.loads(raw)
            if kind == 'extra': plan['extra'] = True; raw = encoded(plan)
            elif kind == 'missing': del plan['rustc']; raw = encoded(plan)
            else: raw = raw[:-1]+b',"source_commit":"'+c['source_commit'].encode()+b'"}'
            c['source_build_plan_sha256'] = hashlib.sha256(raw).hexdigest()
            c['source_pins'][0].update(sha256=c['source_build_plan_sha256'], bytes=len(raw))
            with self.assertRaisesRegex(ValueError, '(closed six-field|duplicate plan)'):
                nb.verify_native_build(c, receipt, lambda ref: raw)

    def test_actual_post_execution_provenance_uses_verified_config_plan(self):
        from native_evaluate import NativeEvaluator
        c, receipt, reader = actual_fixture(); nb.verify_native_build(c, receipt, reader)
        self.assertNotIn('plan_sha256', receipt['source'])
        tree = ast.parse(textwrap.dedent(inspect.getsource(NativeEvaluator.execute)))
        value = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign) and
                     any(isinstance(t, ast.Name) and t.id == 'provenance' for t in n.targets))
        # Evaluate the actual result-metadata expression without running GP,
        # compiling Rust, or replacing the evaluator's scientific implementation.
        import time
        env = {'self': SimpleNamespace(config=c, plan='1'*64), 'context': SimpleNamespace(evaluator_sha256='2'*64),
               'raw': b'input', 'candidate': {'program_sha256': '3'*64}, 'evaluation_bytes': b'evaluation',
               'sha': lambda b: hashlib.sha256(b).hexdigest(), 'time': time, 'commands': {},
               'descriptor': {}, 'manifest_bytes': b'{}', 'predictions': [], 'checks': []}
        provenance = eval(compile(ast.Expression(value), '<actual-provenance>', 'eval'), env)
        self.assertEqual(provenance['source_build_plan_sha256'], c['source_build_plan_sha256'])


if __name__ == '__main__':
    unittest.main()
