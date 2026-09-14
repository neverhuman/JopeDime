from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import qualification_budget as q
from qualification_budget import Budget
from promotion_gate import immutable
from resume_expired_cycle import sha


@contextmanager
def synthetic_domain(root, history=()):
    """Only test fixtures may replace the fixed workspace ledger location."""
    root = Path(root)
    authorization = root / 'synthetic-budget-authorization.json'
    immutable(authorization, dict(schema='parallel.budget_authorization.v2', domain_id='synthetic-tests',
        approved_by='isolated-test-fixture', scope='private_synthetic_qualification', expires_at=time.time()+10000,
        core_seconds=64*3600, node_seconds=2*3600,
        capabilities=['qualification', 'regression_training'],
        historical_ledgers=[dict(path=str(p), sha256=sha(p)) for p in history]))
    with patch.object(q, 'CANONICAL_DOMAIN', root / 'canonical-domain'):
        q.initialize_domain(authorization, sha(authorization))
        yield


class BudgetTests(unittest.TestCase):
    def test_absent_domain_is_denied_without_creating_any_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(q, 'CANONICAL_DOMAIN', root/'missing'):
                with self.assertRaisesRegex(RuntimeError, 'canonical budget domain unavailable'):
                    Budget(root/'new-output')
            self.assertEqual(list(root.iterdir()), [])

    def test_self_declared_empty_history_cannot_initialize_production(self):
        with tempfile.TemporaryDirectory() as folder, synthetic_domain(folder):
            authorization=Path(folder)/'synthetic-budget-authorization.json'
            with patch.object(q, 'WORKSPACE_DOMAIN', q.CANONICAL_DOMAIN):
                with self.assertRaisesRegex(RuntimeError, 'two-host ledger coverage required'):
                    q.initialize_domain(authorization, sha(authorization))

    def test_planted_production_files_deny_at_construction_and_replay(self):
        with tempfile.TemporaryDirectory() as folder, synthetic_domain(folder):
            budget=Budget(folder)
            with patch.object(q, 'WORKSPACE_DOMAIN', q.CANONICAL_DOMAIN):
                with self.assertRaisesRegex(RuntimeError, 'production budget authorization verifier unavailable'):
                    Budget(Path(folder)/'new-output')
                with self.assertRaisesRegex(RuntimeError, 'production budget authorization verifier unavailable'):
                    budget.reserve('planted-grant',core_seconds=1)

    def test_source_relocation_does_not_select_new_domain(self):
        import importlib.util
        with tempfile.TemporaryDirectory() as folder:
            copied=Path(folder)/'release-copy.py'
            copied.write_bytes(Path(q.__file__).read_bytes())
            spec=importlib.util.spec_from_file_location('isolated_relocated_budget',copied)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            self.assertEqual(module.WORKSPACE_DOMAIN, q.WORKSPACE_DOMAIN)
            with self.assertRaisesRegex(RuntimeError, 'production budget authorization verifier unavailable'):
                module.Budget(Path(folder)/'new-output')

    def test_caps_survive_new_output_restart_and_completed_work_is_not_refunded(self):
        with tempfile.TemporaryDirectory() as folder, synthetic_domain(folder):
            b = Budget(Path(folder)/'first-output')
            b.reserve('fits', core_seconds=64*3600, searches=16)
            with self.assertRaisesRegex(RuntimeError, 'CPU qualification cap'):
                Budget(Path(folder)/'different-output').reserve('extra', core_seconds=1)
            receipt = Path(folder) / 'finished.json'
            immutable(receipt, {'finished':True})
            b.complete('fits', receipt)
            with self.assertRaisesRegex(RuntimeError, 'CPU qualification cap'):
                b.reserve('extra', core_seconds=1)
            b.reserve('gpu', node_seconds=7200)
            with self.assertRaisesRegex(RuntimeError, 'GPU qualification cap'):
                b.reserve('gpu-extra', node_seconds=1)

    def test_concurrency_and_interrupted_attempt_reservations(self):
        with tempfile.TemporaryDirectory() as folder, synthetic_domain(folder):
            b = Budget(folder)
            b.reserve('searches', core_seconds=9600, searches=16)
            with self.assertRaisesRegex(RuntimeError, 'allocation exhausted'):
                Budget(folder).reserve('extra', core_seconds=600, searches=1)
            with self.assertRaisesRegex(RuntimeError, 'already consumed'):
                b.reserve('searches', core_seconds=9600, searches=16)

    def test_concurrent_reservations_cannot_overbook(self):
        with tempfile.TemporaryDirectory() as folder, synthetic_domain(folder):
            def reserve(i):
                try:
                    Budget(Path(folder)/str(i)).reserve(str(i), core_seconds=1, searches=1)
                    return True
                except RuntimeError:
                    return False
            with ThreadPoolExecutor(max_workers=8) as pool:
                self.assertEqual(sum(pool.map(reserve, range(32))), 16)

    def test_unknown_historical_consumption_remains_charged(self):
        with tempfile.TemporaryDirectory() as folder:
            old = Path(folder)/'legacy.json'
            immutable(old, dict(schema='parallel.qualification_budget.v1', reservations={
                'unknown':dict(core_seconds=64*3600, node_seconds=0, searches=16, complete=False)}))
            with synthetic_domain(folder, [old]):
                with self.assertRaisesRegex(RuntimeError, 'CPU qualification cap'):
                    Budget(Path(folder)/'new-output').reserve('extra', core_seconds=1)

    def test_nonfinite_boolean_and_negative_reservations_are_denied(self):
        with tempfile.TemporaryDirectory() as folder, synthetic_domain(folder):
            for invalid in (float('nan'), float('inf'), -1, True):
                with self.subTest(invalid=invalid), self.assertRaisesRegex(RuntimeError, 'invalid reservation'):
                    Budget(folder).reserve('invalid', core_seconds=invalid)

    def test_task_reservation_bound_deadline_and_one_consumption(self):
        with tempfile.TemporaryDirectory() as folder, synthetic_domain(folder):
            budget=Budget(folder)
            budget.reserve('train', node_seconds=300, capability='regression_training',
                task_sha256='a'*64, deadline=time.time()+250)
            ref=dict(domain_id='synthetic-tests', identity='train', task_sha256='a'*64)
            for bad in (None, True, False, {}, dict(ref, task_sha256='b'*64)):
                with self.subTest(ref=bad), self.assertRaises(RuntimeError):
                    budget.verify(bad, 'a'*64)
            budget.verify(ref, 'a'*64)
            budget.verify(ref, 'a'*64, consume=True)
            with self.assertRaisesRegex(RuntimeError, 'already consumed'):
                budget.verify(ref, 'a'*64, consume=True)
            with patch.object(q.time, 'time', return_value=time.time()+20000):
                with self.assertRaisesRegex(RuntimeError, 'expired'):
                    budget.reserve('late', core_seconds=1)

    def test_journal_corruption_and_missing_ledger_cannot_reset_funding(self):
        with tempfile.TemporaryDirectory() as folder, synthetic_domain(folder):
            budget=Budget(folder)
            budget.reserve('old', core_seconds=1)
            with budget.path.open('ab') as stream:
                stream.write(b'{broken}\n')
            with self.assertRaisesRegex(ValueError, 'corrupt complete journal'):
                Budget(folder).reserve('new', core_seconds=1)
            budget.path.unlink()
            with self.assertRaisesRegex(RuntimeError, 'canonical budget domain unavailable'):
                Budget(Path(folder)/'another-output')
            authorization=Path(folder)/'synthetic-budget-authorization.json'
            with self.assertRaisesRegex(RuntimeError, 'reset forbidden'):
                q.initialize_domain(authorization, sha(authorization))


if __name__ == '__main__':
    unittest.main()
