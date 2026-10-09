"""New regression cases for the closed, source-pinned portable runner."""
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest

from .. import run_offline as runner


class RunnerTests(unittest.TestCase):
    def test_expected_and_unexpected_failures_skips_and_counts_are_not_passes(self):
        result = SimpleNamespace(testsRun=runner.EXPECTED_FIXTURE_COUNT, failures=[], errors=[], skipped=[],
                                 expectedFailures=[], unexpectedSuccesses=[], wasSuccessful=lambda: True)
        self.assertTrue(runner._result_is_clean(result))
        for name in ('failures', 'errors', 'skipped', 'expectedFailures', 'unexpectedSuccesses'):
            with self.subTest(name=name):
                setattr(result, name, ['reported-outcome'])
                self.assertFalse(runner._result_is_clean(result))
                setattr(result, name, [])
        result.testsRun -= 1
        self.assertFalse(runner._result_is_clean(result))

    def test_missing_extra_duplicate_replaced_or_reordered_fixtures_refused(self):
        ids = list(runner.EXPECTED_FIXTURE_IDS)
        runner._check_fixture_roster(ids)
        cases = [ids[:-1], ids + ['invented'], ids[:-1] + [ids[0]], ids[:-1] + ['invented'], list(reversed(ids))]
        for case in cases:
            with self.subTest(case=case[-1]), self.assertRaisesRegex(RuntimeError, 'frozen fixture roster'):
                runner._check_fixture_roster(case)

    def test_source_pin_and_import_origin_tampering_refused(self):
        root = Path(runner.__file__).resolve().parent
        with tempfile.TemporaryDirectory() as temp:
            copied = Path(temp).resolve() / 'source'
            copied.mkdir()
            for rel in set(runner.EXPECTED_IMPORTS.values()) | {'SOURCE_PINS.json'}:
                path = copied / rel
                path.parent.mkdir(exist_ok=True)
                shutil.copyfile(root / rel, path)
            runner._pinned_sources(copied)
            changed = copied / 'export_verified.py'
            changed.write_bytes(changed.read_bytes() + b'\n# unexpected source change\n')
            with self.assertRaisesRegex(RuntimeError, 'source bytes changed'):
                runner._pinned_sources(copied)
        modules = {name: SimpleNamespace(__file__=str(root / rel), __spec__=SimpleNamespace(origin=str(root / rel)))
                   for name, rel in runner.EXPECTED_IMPORTS.items()}
        runner._check_import_origins(root, modules)
        name = runner.PREFIX + '.common'
        modules[name].__spec__.origin = '/foreign/package/common.py'
        with self.assertRaisesRegex(RuntimeError, 'import origin mismatch'):
            runner._check_import_origins(root, modules)
        modules.pop(name)
        with self.assertRaisesRegex(RuntimeError, 'import module roster'):
            runner._check_import_origins(root, modules)
