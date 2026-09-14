import copy
import hashlib
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import kernel_remote_state as k


class MigrationHealthTests(unittest.TestCase):
    def state(self):
        services = {}
        for unit in k.UNITS:
            stopped = unit in k.FENCED_UNITS
            services[unit] = dict(definition='bound service definition', exe_sha256='a' * 64,
                                  properties=dict(MainPID='0' if stopped else '123',
                                                  ActiveState='inactive' if stopped else 'active'))
        services['serving_reference'] = dict(sha256=k.RECOVERY_SHA256)
        root = Path('/home/ubuntu/jope-dime/parallel')
        original_marker = str(root / Path(k.FENCE_COMPLETE).parent / 'REGISTRY_HANDOFF_REQUIRED')
        services['jope-parallel-gp']['properties'].update(Restart='no', DropInPaths=str(k.FENCE_DROPIN))
        services['jope-parallel-gp']['effective_conditions'] = [['ConditionPathExists', False, False, original_marker]]
        for unit in k.FENCED_UNITS[1:]:
            marker = str(root / k.COORDINATED_MARKER)
            dropin = str(Path('/etc/systemd/system') / (unit + '.service.d') / k.COORDINATED_DROPIN)
            contents = f'[Unit]\nConditionPathExists={marker}\n[Service]\nRestart=no\n'.encode()
            services[unit]['properties'].update(Restart='no', DropInPaths=dropin)
            services[unit]['coordinated_fence'] = dict(dropin=dropin, dropin_sha256=hashlib.sha256(contents).hexdigest(),
                marker=marker, marker_exists=False, effective_conditions=[['ConditionPathExists', False, False, marker]])
        return dict(schema='kernel.live-state.v2', custody='planned_migration',
                    timestamp=100, stable=services, capture_errors=[], dispatch_processes=[],
                    telemetry={'hardware.json': dict(timestamp=100, errors=[])},
                    fence=dict(receipt_sha256=k.FENCE_RECEIPT_SHA256,
                               marker=original_marker, marker_exists=False,
                               override_sha256='67940194f08f23503b44d75b13e5f6add51c4abb06a32487e13fa9d106c8c7f5',
                               registry_dispatch_enabled=False))

    def test_stopped_roles_do_not_excuse_other_failures(self):
        before = self.state()
        self.assertEqual(k.compare_health(before, before)['status'], 'passed')
        for fault in ('monitor', 'hardware', 'stale', 'fence', 'worker', 'definition', 'serving', 'incomplete'):
            after = copy.deepcopy(before)
            if fault == 'monitor':
                after['stable']['jope-parallel-monitor']['properties']['ActiveState'] = 'inactive'
            elif fault == 'hardware':
                after['telemetry']['hardware.json']['errors'] = ['device error']
            elif fault == 'stale':
                after['telemetry']['hardware.json']['timestamp'] = 0
            elif fault == 'fence':
                after['fence']['receipt_sha256'] = 'b' * 64
            elif fault == 'worker':
                after['dispatch_processes'] = [dict(pid=234)]
            elif fault == 'definition':
                after['stable']['jope-parallel-gp']['definition'] = 'changed'
            elif fault == 'serving':
                after['stable']['serving_reference']['sha256'] = 'b' * 64
            else:
                del after['capture_errors']
            with self.subTest(fault=fault):
                self.assertNotEqual(k.compare_health(before, after)['status'], 'passed')

    def test_running_alternate_identity_remains_fenced(self):
        state = self.state()
        state['stable']['jope-parallel-gp-next']['properties'].update(MainPID='234', ActiveState='active')
        self.assertEqual(k.compare_health(state, state)['status'], 'failed')

    def test_clearing_errors_cannot_replace_effective_fence_evidence(self):
        for unit in k.FENCED_UNITS:
            for replacement in (None, {}, []):
                state = self.state()
                key = 'effective_conditions' if unit == 'jope-parallel-gp' else 'coordinated_fence'
                state['stable'][unit][key] = replacement
                state['capture_errors'] = []
                self.assertEqual(k.compare_health(state, state)['status'], 'failed')

    def test_changed_fence_bytes_or_active_marker_cannot_pass(self):
        for key, value in [('dropin_sha256', 'a'*64), ('marker_exists', True), ('effective_conditions', [])]:
            state = self.state()
            state['stable'][k.FENCED_UNITS[2]]['coordinated_fence'][key] = value
            self.assertEqual(k.compare_health(state, state)['status'], 'failed')

    def test_effective_coordinated_condition_requires_bound_missing_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / k.COORDINATED_MARKER
            marker.parent.mkdir(parents=True)
            dropin = root / 'fence.conf'
            dropin.write_text(f'[Unit]\nConditionPathExists={marker}\n[Service]\nRestart=no\n')
            actual_path = Path('/etc/systemd/system/jope-parallel-gp-next.service.d') / k.COORDINATED_DROPIN
            props = dict(Restart='no', DropInPaths=str(actual_path))
            condition = ['ConditionPathExists', False, False, str(marker)]
            original_file, original_read = Path.is_file, Path.read_text
            with patch.object(Path, 'is_file', lambda p: True if p == actual_path else original_file(p)), \
                 patch.object(Path, 'read_text', lambda p: dropin.read_bytes().decode() if p == actual_path else original_read(p)), \
                 patch.object(k, 'sha', return_value='a'*64), \
                 patch.object(k, 'effective_conditions', return_value=[condition]) as loaded:
                self.assertIsNotNone(k.coordinated_dispatch_fence(root, 'jope-parallel-gp-next', props))
                for wrong in ([], [['ConditionPathExists', True, False, str(marker)]],
                              [['ConditionPathExists', False, True, str(marker)]]):
                    loaded.return_value = wrong
                    self.assertIsNone(k.coordinated_dispatch_fence(root, 'jope-parallel-gp-next', props))
                loaded.return_value = [condition]
                marker.touch()
                self.assertIsNone(k.coordinated_dispatch_fence(root, 'jope-parallel-gp-next', props))
                marker.unlink()
                dropin.write_text('[Service]\nRestart=no\n')
                self.assertIsNone(k.coordinated_dispatch_fence(root, 'jope-parallel-gp-next', props))


if __name__ == '__main__':
    unittest.main()
