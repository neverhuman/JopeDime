"""New portable-adapter refusals; no historical fixture execution in these cases."""
from pathlib import Path
import tempfile
import unittest

from offline_adapter import prepare_output


class OutputCases(unittest.TestCase):
    def test_existing_receipt_directory_refused_without_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source, output = base / 'source', base / 'old-output'
            source.mkdir(); output.mkdir()
            receipt = output / 'receipt.json'
            receipt.write_bytes(b'old exact receipt\n')
            with self.assertRaisesRegex(ValueError, 'fresh output'):
                prepare_output(output, source, 'adapter_v1')
            self.assertEqual(receipt.read_bytes(), b'old exact receipt\n')
            self.assertEqual(list(output.iterdir()), [receipt])

    def test_symlink_output_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary); source = base / 'source'; source.mkdir()
            target = base / 'target'; target.mkdir()
            output = base / 'linked-output'; output.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'symlink'):
                prepare_output(output, source, 'adapter_v1')
            self.assertEqual(list(target.iterdir()), [])

    def test_source_contained_output_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'source'; source.mkdir()
            output = source / 'receipt-output'
            with self.assertRaisesRegex(ValueError, 'outside immutable source'):
                prepare_output(output, source, 'adapter_v1')
            self.assertFalse(output.exists())

    def test_unknown_family_refused_before_output_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary); source = base / 'source'; source.mkdir()
            output = base / 'new-output'
            with self.assertRaisesRegex(ValueError, 'known closed family'):
                prepare_output(output, source, 'unknown')
            self.assertFalse(output.exists())



class CIPathCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import the guarded-main, stdlib-only classifier by exact source identity.
        # No CLI main, subprocess, Git, compiler or project runtime is invoked.
        import hashlib
        import importlib.util
        import json
        import sys
        root = Path(__file__).resolve().parents[1]
        declaration = json.loads((root / 'CI_SOURCE_PIN.json').read_bytes())
        if set(declaration) != {'schema', 'path', 'sha256', 'bytes'} or declaration['schema'] != 'jope.standalone-monitor-ci-source-pin/v1':
            raise ValueError('closed exact CI source declaration')
        if declaration['path'] != 'deploy/ci/ci-plan.py':
            raise ValueError('exact owning CI classifier path')
        path = Path(__file__).resolve().parents[5] / declaration['path']
        if not path.is_file() or path.is_symlink():
            raise ValueError('regular CI source')
        raw = path.read_bytes()
        if len(raw) != declaration['bytes'] or hashlib.sha256(raw).hexdigest() != declaration['sha256']:
            raise ValueError('exact external CI source bytes')
        alias = 'monitor_ci_plan_adapter_v1'
        if alias in sys.modules:
            raise ValueError('fresh non-main CI source alias')
        spec = importlib.util.spec_from_file_location(alias, path)
        if spec is None or spec.loader is None:
            raise ValueError('explicit guarded CI loader')
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        spec.loader.exec_module(module)
        cls.classifier = module

    def check_python_only(self, path):
        lanes, gates = self.classifier.classify([path], 'pull_request')
        self.assertTrue(lanes['python'])
        self.assertTrue(gates['python'])
        self.assertTrue(lanes['changes'])
        self.assertTrue(lanes['hygiene'])
        for lane in ('rust-check','rust-test','emitters','inputs','e2e-smoke'):
            self.assertFalse(lanes[lane], lane)

    def test_standalone_only_pr_selects_python(self):
        self.check_python_only('deploy/dl-flywheel/pipelines/standalone_paper_monitor_v1/README.md')

    def test_dashboard_only_pr_selects_python(self):
        self.check_python_only('deploy/dl-flywheel/dashboard/index.html')

    def test_unrelated_docs_skip_python_dispatch_selects_all(self):
        paths = ['docs/user-guides/flywheel-renderer-notes.md']
        lanes, gates = self.classifier.classify(paths, 'pull_request')
        self.assertFalse(lanes['python'])
        self.assertFalse(gates['python'])
        self.assertTrue(lanes['changes'])
        self.assertTrue(lanes['hygiene'])
        self.assertTrue(all(not value for value in gates.values()))
        dispatch, dispatch_gates = self.classifier.classify(paths, 'workflow_dispatch')
        self.assertEqual(set(dispatch), {'changes','hygiene','rust-check','rust-test','emitters','inputs','python','e2e-smoke'})
        self.assertTrue(all(dispatch.values()))
        self.assertTrue(all(dispatch_gates.values()))
