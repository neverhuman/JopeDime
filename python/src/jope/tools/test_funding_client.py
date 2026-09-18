"""No authority installation or numerical execution; negative bridge boundaries."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import funding_client
import qualification_budget


class FundingClientTests(unittest.TestCase):
    def test_caller_environment_cannot_redirect_production_trust(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / 'self-declared-client.json'
            candidate.write_text('{"schema":"parallel.funding_client.v1","eligible":true}')
            with patch.dict('os.environ', {'JOPE_FUNDING_CONFIG': str(candidate)}), \
                    patch.object(funding_client, '_protected', side_effect=FileNotFoundError('installation missing')) as opened:
                with self.assertRaisesRegex(RuntimeError, 'production budget authorization verifier unavailable'):
                    funding_client.NativeFundingClient()
            self.assertEqual(opened.call_args.args[0], Path('/etc/jope-dime/funding/client.json'))

    def test_unprotected_or_symlink_installation_is_not_a_trust_root(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text('{}')
            alias = Path(directory) / 'alias.json'
            alias.symlink_to(path)
            for candidate in (path, alias):
                with self.assertRaisesRegex(RuntimeError, 'independently protected'):
                    funding_client._protected(candidate)

    def test_duplicate_or_nonfinite_funding_json_is_rejected(self):
        for raw in ('{"domain_id":"a","domain_id":"b"}', '{"node_ms":NaN}', '{"core_ms":Infinity}', '{"core_ms":1e400}'):
            with self.assertRaises(ValueError):
                funding_client._json(raw)

    def test_python_client_cannot_consume_native_start_or_rank_slots(self):
        client = funding_client.NativeFundingClient.__new__(funding_client.NativeFundingClient)
        for operation in ('start', 'participant', 'complete', 'reset'):
            with patch.object(funding_client, '_protected', side_effect=AssertionError('no installation opens')):
                with self.assertRaisesRegex(RuntimeError, 'cannot consume numerical execution'):
                    client.request(operation, eligible=True)

    def test_scalar_python_reservations_and_completions_cannot_bypass_native_custody(self):
        budget = qualification_budget.Budget.__new__(qualification_budget.Budget)
        budget.native = object()
        with self.assertRaisesRegex(RuntimeError, 'reserve_task'):
            budget.reserve('new-name', core_seconds=1, node_seconds=1)
        with self.assertRaisesRegex(RuntimeError, 'preflight_task'):
            budget.verify({}, 'a' * 64, consume=True)
        with self.assertRaisesRegex(RuntimeError, 'witnessed termination'):
            budget.complete('old-task', Path('/caller/claimed-completion.json'))

    def test_unsigned_legacy_authorization_cannot_initialize_production(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'unsigned.json'
            path.write_text('{"schema":"parallel.budget_authorization.v2","eligible":true,"scope":"production"}')
            with patch.object(qualification_budget, 'CANONICAL_DOMAIN', qualification_budget.WORKSPACE_DOMAIN), \
                    patch.object(qualification_budget, 'NativeFundingClient', side_effect=AssertionError('unsigned legacy grant must not reach native import')):
                with self.assertRaisesRegex(RuntimeError, 'two-host ledger coverage required'):
                    qualification_budget.initialize_domain(path, qualification_budget.sha(path))


if __name__ == '__main__':
    unittest.main()
