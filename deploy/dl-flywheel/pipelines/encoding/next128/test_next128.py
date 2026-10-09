"""Successor-only tests. Never imports/reruns the closed fresh32 test module."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import run_next128 as run
import freeze_next128 as freeze

HERE = Path(__file__).resolve().parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SharedClock(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.template = run.file_ref(HERE / 'config.template.json')
        self.review = run.save(self.root / 'review.json', {'state': 'passed', 'template_sha256': self.template['sha256']})

    def tearDown(self):
        self.tmp.cleanup()

    def frozen(self):
        ref = freeze.freeze(self.template, self.review, self.root / 'frozen', wall=1000., monotonic=100.)
        return json.loads(run.read(ref))

    def test_all_four_children_share_one_clock(self):
        config = self.frozen()
        with patch.object(run.time, 'time', return_value=1001), patch.object(run.time, 'monotonic', return_value=101):
            snapshot = run.validate_config(config)
        self.assertEqual(len(snapshot['rows']), 128)
        for item in config['packets']:
            child = json.loads(run.read(item['config']))
            self.assertEqual((child['created_unix'], child['deadline_unix'], child['shared_monotonic_deadline']), (1000., 2800., 1900.))

    def test_clock_cannot_be_frozen_twice(self):
        self.frozen()
        with self.assertRaises(FileExistsError):
            freeze.freeze(self.template, self.review, self.root / 'frozen', wall=2000., monotonic=200.)

    def test_wrong_review_refused_before_clock_directory(self):
        wrong = run.save(self.root / 'wrong.json', {'state': 'passed', 'template_sha256': '0' * 64})
        with self.assertRaisesRegex(ValueError, 'independent review'):
            freeze.freeze(self.template, wrong, self.root / 'absent', wall=1000., monotonic=100.)
        self.assertFalse((self.root / 'absent').exists())

    def test_global_deadline_cannot_be_extended(self):
        with self.assertRaisesRegex(ValueError, 'global ceiling'):
            freeze.freeze(self.template, self.review, self.root / 'absent', wall=run.GLOBAL - 1799, monotonic=100.)

    def test_backward_wall_clock_never_extends_monotonic_budget(self):
        config = {'deadline_unix': 2800., 'shared_monotonic_deadline': 1900.}
        self.assertEqual(run.remaining(config, wall=-100000, monotonic=1800), 100)
        self.assertLessEqual(run.remaining(config, wall=1000, monotonic=1901), 0)

    def test_expired_config_refused(self):
        config = self.frozen()
        with patch.object(run.time, 'time', return_value=2000), patch.object(run.time, 'monotonic', return_value=1901):
            with self.assertRaisesRegex(ValueError, 'window expired'):
                run.validate_config(config)

    def test_child_clock_reset_and_numeric_boolean_approval_refused(self):
        config = self.frozen()
        child = json.loads(run.read(config['packets'][1]['config']))
        child['deadline_unix'] += 1
        config['packets'][1]['config'] = run.save(self.root / 'wrong-child.json', child)
        with patch.object(run.time, 'time', return_value=1001), patch.object(run.time, 'monotonic', return_value=101):
            with self.assertRaisesRegex(ValueError, 'reset its clock'):
                run.validate_config(config)
        config['approved_for_launch'] = 1
        with self.assertRaisesRegex(ValueError, 'approved config'):
            run.validate_config(config)

    def test_source_drift_refused(self):
        config = json.loads(run.read(self.template))
        config['packets'][0]['controller']['sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            run.validate_template(config)

    def test_duplicate_packet_and_changed_support_refused(self):
        config = json.loads(run.read(self.template))
        config['packets'][1] = copy.deepcopy(config['packets'][0])
        with self.assertRaisesRegex(ValueError, 'packet order'):
            run.validate_template(config)
        config = json.loads(run.read(self.template))
        config['selection']['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'snapshot identity'):
            run.validate_template(config)


class SequenceLifecycle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config = json.loads((HERE / 'config.template.json').read_bytes())
        self.config.update(deadline_unix=time.time() + 1500, shared_monotonic_deadline=time.monotonic() + 1500)
        self.snapshot = json.loads(run.read(self.config['selection']))

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def verify(descriptor, result, support):
        run.need(result['state'] == 'stage_committed', 'not committed')
        return {'ordinal': descriptor['ordinal'], 'state': 'stage_committed', 'variants': len(support)}

    def test_only_four_fixed_packets_no_fifth(self):
        calls = []
        def invoke(descriptor, config, output):
            calls.append((descriptor['ordinal'], config['deadline_unix'], config['shared_monotonic_deadline']))
            return {'state': 'stage_committed'}
        result = run.sequence(self.config, self.snapshot, self.root, invoke, self.verify)
        self.assertEqual([r[0] for r in calls], [0, 1, 2, 3])
        self.assertEqual(len({r[1:] for r in calls}), 1)
        self.assertEqual((result['completed_variants'], result['verified_objects']), (128, 256))

    def test_blocked_claim_empty_failure_unknown_stop_without_next(self):
        for state in ('blocked_claim', 'empty', 'packet_already_exists', 'failed', 'unknown'):
            with self.subTest(state=state):
                output = self.root / state
                output.mkdir()
                calls = []
                def invoke(descriptor, config, output):
                    calls.append(descriptor['ordinal'])
                    return {'state': state}
                with self.assertRaises(ValueError):
                    run.sequence(self.config, self.snapshot, output, invoke, self.verify)
                self.assertEqual(calls, [0])

    def test_completed_first_packet_retained_on_second_failure(self):
        calls = []
        def invoke(descriptor, config, output):
            calls.append(descriptor['ordinal'])
            if descriptor['ordinal'] == 1:
                raise RuntimeError('uncertain write retained')
            return {'state': 'stage_committed'}
        with self.assertRaises(RuntimeError):
            run.sequence(self.config, self.snapshot, self.root, invoke, self.verify)
        self.assertEqual(calls, [0, 1])
        self.assertTrue((self.root / 'packet0/verified.json').exists())
        self.assertFalse((self.root / 'packet2').exists())

    def test_insufficient_shared_window_never_invokes_or_claims(self):
        self.config['shared_monotonic_deadline'] = time.monotonic() + 650
        calls = []
        with self.assertRaisesRegex(ValueError, 'before next claim'):
            run.sequence(self.config, self.snapshot, self.root, lambda *args: calls.append(args), self.verify)
        self.assertEqual(calls, [])

    def test_owned_group_cleanup_after_parent_has_exited(self):
        # Regression for a terminated parent whose descendant still lives.
        child = subprocess.Popen([sys.executable, '-c', 'import subprocess; subprocess.Popen(["sleep", "20"])'], start_new_session=True)
        child.wait(timeout=5)
        run.terminate_owned_group(child)
        # The group may retain a short-lived reparented zombie. A second cleanup
        # must also be safe and cannot target a process group we did not create.
        run.terminate_owned_group(child)


class ExistingQueueBoundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.campaign = load(HERE / 'packet0/campaign.py', 'next_campaign')
        previous = sys.modules.get('campaign')
        sys.modules['campaign'] = cls.campaign
        try:
            cls.queue = load(HERE / 'packet0/kumo_queue.py', 'next_queue')
        finally:
            if previous is None:
                del sys.modules['campaign']
            else:
                sys.modules['campaign'] = previous
        cls.selection = json.loads((HERE / 'packet0/selection.json').read_bytes())
        cls.snapshot = json.loads((HERE / 'selection-readonly.json').read_bytes())

    def database(self):
        db = sqlite3.connect(':memory:', isolation_level=None)
        db.row_factory = sqlite3.Row
        row = self.selection['rows'][0]
        pub = self.snapshot['publications'][0]
        for table, sample in [('variants', row), (self.queue.PUB, pub)]:
            db.execute('CREATE TABLE ' + table + '(' + ','.join('"' + k + '"' for k in sample) + ')')
        for item in self.selection['rows']:
            db.execute('INSERT INTO variants VALUES(' + ','.join('?' for _ in item) + ')', list(item.values()))
        for item in self.snapshot['publications'][:32]:
            db.execute('INSERT INTO ' + self.queue.PUB + ' VALUES(' + ','.join('?' for _ in item) + ')', list(item.values()))
        self.queue.ensure(db)
        self.addCleanup(db.close)
        return db

    def claim(self, db, role=lambda row: True, deadline=None):
        return self.queue.claim(db, self.campaign.PIPELINE_SHA, self.campaign.PACKET_ID, 32, time.time() + 1000 if deadline is None else deadline, self.selection, role)

    def test_new32_claim_excludes_held_old_identity_globally(self):
        db = self.database()
        row = self.selection['rows'][0]
        db.execute('INSERT INTO ' + self.queue.ITEMS + ' VALUES(?,?,?,?,?)', (row['identity'], 'old-held', row['sequence'], '{}', '{}'))
        result = self.claim(db)
        self.assertEqual(result['state'], 'blocked_claim')
        self.assertEqual(db.execute('SELECT count(*) FROM ' + self.queue.PACKETS).fetchone()[0], 0)

    def test_active_same_host_claim_is_blocked_not_empty(self):
        db = self.database()
        db.execute('INSERT INTO ' + self.queue.PACKETS + ' VALUES(?,?,?,?,?,?,?)', ('old', 'xbabe3', self.campaign.PIPELINE_SHA, 0, 1, 'claimed', None))
        self.assertEqual(self.claim(db)['state'], 'blocked_claim')
        self.assertEqual(db.execute('SELECT state FROM ' + self.queue.PACKETS).fetchone()[0], 'claimed')

    def test_current_role_refusal_rolls_back_all_claims(self):
        db = self.database()
        with self.assertRaisesRegex(ValueError, 'currently protected'):
            self.claim(db, role=lambda row: False)
        self.assertEqual(db.execute('SELECT count(*) FROM ' + self.queue.PACKETS).fetchone()[0], 0)
        self.assertEqual(db.execute('SELECT count(*) FROM ' + self.queue.ITEMS).fetchone()[0], 0)

    def test_expired_window_refused_before_claim(self):
        db = self.database()
        with self.assertRaisesRegex(ValueError, 'claim window'):
            self.claim(db, deadline=time.time() - 1)

    def test_role_lookup_delay_leaves_zero_real_claim_inserts(self):
        db = self.database()
        clock = [1000.]
        def delayed_role(row):
            # Simulate staging/authority work consuming the useful GPU reserve
            # after the entry guard but inside the real claim transaction.
            clock[0] = 1500.
            return True
        with patch.object(self.queue.time, 'time', side_effect=lambda: clock[0]):
            with self.assertRaisesRegex(ValueError, 'immediately before claim INSERT'):
                self.claim(db, role=delayed_role, deadline=2000.)
        self.assertEqual(db.execute('SELECT count(*) FROM ' + self.queue.PACKETS).fetchone()[0], 0)
        self.assertEqual(db.execute('SELECT count(*) FROM ' + self.queue.ITEMS).fetchone()[0], 0)
        self.assertFalse(db.in_transaction)

    def test_resource_delay_refuses_before_actual_intake_dispatch(self):
        previous = sys.modules.get('campaign')
        sys.modules['campaign'] = self.campaign
        try:
            controller = load(HERE / 'packet0/kumo_controller.py', 'delayed_controller')
        finally:
            if previous is None:
                del sys.modules['campaign']
            else:
                sys.modules['campaign'] = previous
        clock = [1000.]
        def resource(*args):
            clock[0] = 2150.
            return {'passed': True}
        config = {'created_unix': 1000., 'deadline_unix': 2800., 'shared_monotonic_deadline': 1900.}
        with tempfile.TemporaryDirectory() as temp, patch.object(controller, 'resource_precheck', side_effect=resource), patch.object(controller, 'ssh') as ssh, patch.object(controller.time, 'time', side_effect=lambda: clock[0]), patch.object(controller.time, 'monotonic', side_effect=lambda: clock[0] - 900.):
            with self.assertRaisesRegex(ValueError, 'before actual intake'):
                controller.packet(config, 'unused-pin', '/unused', Path(temp))
            ssh.assert_not_called()

    def test_success_then_same_packet_never_replays(self):
        db = self.database()
        self.assertEqual(self.claim(db)['state'], 'claimed')
        self.assertEqual(self.claim(db)['state'], 'packet_already_exists')
        self.assertEqual(db.execute('SELECT count(*) FROM ' + self.queue.ITEMS).fetchone()[0], 32)


class SourcePreservation(unittest.TestCase):
    def test_all_four_scientific_workers_are_exact_successful_bytes(self):
        for ordinal in range(4):
            raw = (HERE / ('packet' + str(ordinal)) / 'evergreen_kumos_packet.py').read_bytes()
            self.assertEqual(run.sha(raw), run.WORKER_SHA)

    def test_queue_changes_only_fixed_sequence_literal(self):
        original = HERE.parent.parent / 'kumo-xbabe3-fresh32/kumo_queue.py'
        before = ast.dump(ast.parse(original.read_text()), include_attributes=False)
        for ordinal in range(4):
            text = (HERE / ('packet' + str(ordinal)) / 'kumo_queue.py').read_text()
            text = text.replace("need(len(selected)==COUNT and deadline-time.time()>=660,'complete fixed support and660 seconds immediately before claim INSERT')", "need(len(selected)==COUNT and time.time()<deadline,'complete fixed support/deadline before claim')")
            text = text.replace('list(range(' + str(938 + ordinal * 32) + ',' + str(970 + ordinal * 32) + '))', 'list(range(906,938))')
            self.assertEqual(ast.dump(ast.parse(text), include_attributes=False), before)

    def test_controller_only_fixed_root_and_shared_clock_deltas(self):
        original = HERE.parent.parent / 'kumo-xbabe3-fresh32/kumo_controller.py'
        before = ast.dump(ast.parse(original.read_text()), include_attributes=False)
        original_root = '/home/ubuntu/JopeDime/.agent/proposals/dl-flywheel-kumo-xbabe3-fresh32-20261009/runtime'
        added_guard = "need(type(c['shared_monotonic_deadline'])in(int,float)and 0<c['shared_monotonic_deadline']-time.monotonic()<=1800,'shared monotonic deadline');need(c['shared_scope']=='next128-four-fixed32-v1','shared scope');"
        for ordinal in range(4):
            text = (HERE / ('packet' + str(ordinal)) / 'kumo_controller.py').read_text()
            text = text.replace(run.SOURCE_ROOT + '/packet' + str(ordinal) + '/runtime', original_root)
            text = text.replace("mono=config['shared_monotonic_deadline'];need(time.monotonic()<mono,'unchanged shared monotonic deadline')", "mono=time.monotonic()+deadline-time.time()")
            text = text.replace(added_guard, '')
            text = text.replace("need(remaining()>=660,'shared remaining before actual intake must reserve660 seconds');", '')
            self.assertEqual(ast.dump(ast.parse(text), include_attributes=False), before)

    def test_no_source_replays_first32_or_original_held32(self):
        old = json.loads((HERE.parent.parent / 'kumo-xbabe3-fresh32/selection.json').read_bytes())
        old_ids = {r['identity'] for r in old['rows']}
        all_ids = []
        for ordinal in range(4):
            rows = json.loads((HERE / ('packet' + str(ordinal)) / 'selection.json').read_bytes())['rows']
            self.assertTrue(all(r['sequence'] >= 938 for r in rows))
            self.assertFalse(old_ids.intersection(r['identity'] for r in rows))
            all_ids.extend(r['identity'] for r in rows)
        self.assertEqual(len(set(all_ids)), 128)


if __name__ == '__main__':
    unittest.main(verbosity=2)
