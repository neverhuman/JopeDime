import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from admission import check, protected_json, validate_status


class AdmissionTests(unittest.TestCase):
    def status_fixture(self):
        return dict(grant_sha256='hash', cgroup='/own', sample_unix=100, tick_seconds=1, state='admitted',
                    host_cpu_peak_percent=20, available_ram_bytes=40*1024**3, free_disk_bytes=200*1024**3,
                    temperature_celsius=50, swapin_bytes_per_second=0)

    def test_valid_telemetry_contract_only(self):
        validate_status(self.status_fixture(), {'cgroup': '/own'}, 'hash', 100.5)

    def test_nonfinite_resource_never_means_headroom(self):
        for key in ('host_cpu_peak_percent', 'available_ram_bytes', 'free_disk_bytes', 'temperature_celsius'):
            s = self.status_fixture(); s[key] = float('nan')
            with self.assertRaises(ValueError): validate_status(s, {'cgroup': '/own'}, 'hash', 100.5)

    def test_guard_stale_hold_or_wrong_leaf_refused(self):
        for delta in ({'sample_unix': 90}, {'state': 'hold'}, {'cgroup': '/foreign'}, {'host_cpu_peak_percent': 65}):
            s = {**self.status_fixture(), **delta}
            with self.assertRaises(ValueError): validate_status(s, {'cgroup': '/own'}, 'hash', 100.5)

    def test_user_written_grant_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'grant.json'; p.write_text('{}')
            with self.assertRaises(ValueError): protected_json(p)

    def test_no_old_inert_grant_substitution(self):
        with patch('admission.protected_json', return_value={'schema': 'jope.inert-owner-entry/v2'}):
            with self.assertRaises(ValueError): check('unused', plan_sha256='x', source_commit='y')

    def test_wrong_node_refused_before_kernel_or_device(self):
        g = {'schema': 'jope.gpu-research-admission/v1', 'node': 'xbabe1'}
        with patch('admission.protected_json', return_value=g), patch('admission.socket.gethostname', return_value='xbabe2'):
            with self.assertRaises(ValueError): check('unused', plan_sha256='x', source_commit='y')

    def test_old_source_receipt_does_not_admit_successor(self):
        g = {'schema': 'jope.gpu-research-admission/v1', 'node': 'xbabe2', 'source_commit': 'old', 'preregistration_sha256': 'old'}
        with patch('admission.protected_json', return_value=g), patch('admission.socket.gethostname', return_value='xbabe2'):
            with self.assertRaises(ValueError): check('unused', plan_sha256='new', source_commit='new')


if __name__ == '__main__': unittest.main()
