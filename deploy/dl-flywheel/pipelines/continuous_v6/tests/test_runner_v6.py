"""runner_v6 / genesis / resume private CPU fixtures on a real W10 r2 capture of a v1-shaped run (no live data).

The W10 ``idle`` world runs the frozen v1 runner until it waits for GP capacity, simulates the RuntimeMaxSec kill
inside the closure and is captured with the R17 class.  Genesis builds a v2 campaign root from that capture and
runner_v6 trains from it.  No S3, GPU, GP/program execution or live file is involved.
"""
import copy
import json
import os
import shutil
import sqlite3
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
from safetensors.torch import load

from jope_flywheel.protocol import canonical, digest
from jope_flywheel.program_io import sha
from neural_native_stream import BatchStream
from pipelines.boundary_admission_v2 import boundary as Bd
from pipelines.continuous_v6 import genesis as G
from pipelines.continuous_v6 import resume_v6 as RV
from pipelines.continuous_v6 import runner_v6 as R6
from pipelines.continuous_v6 import anchored_step as AS
from pipelines.final_capture_v1 import capture as FC
from pipelines.final_capture_v1 import probe as FP
from pipelines.final_capture_v1.tests import world as W10
from pipelines.generated_target_preparation_v2.tests import world as CW

PARITY = {'cpu': {'passed': True, 'sha256': digest({'fixture-cpu-parity': 1})},
          'cuda': {'passed': True, 'sha256': digest({'fixture-cuda-parity': 1})}}
EXEC = digest({'exec': 1})


def parity_receipt(capture, **over):
    """A fixture next-update parity receipt shaped like parity_probe.compare's output (the real one comes from GPU)."""
    raw = (capture / 'final-capture.json').read_bytes(); v1c = json.loads(raw)['control']['body']
    r = {'schema': G.PARITY_SCHEMA, 'passed': True, 'capture_sha256': sha(raw), 'device': 'cuda:0',
         'v1_control_sha256': sha(canonical(v1c)), 'control_applied': 'v1 unchanged (pre-R14)',
         'bstar': {'source': 'v1 planner backup (byte copy)'}, 'sides': {'A': {}, 'B1': {}, 'B2': {}},
         'comparisons': {k: {'bitwise_equal': True, 'groups': {'model': {}, 'adamw': {}, 'rng': {}}} for k in ('A_vs_B1', 'A_vs_B2')}}
    r.update(over); return r
R14_RECEIPT = digest({'fixture-r14-receipt': 1})
CAMPAIGN = digest({'fixture-v2-campaign': 1})
SLEEPLESS = types.SimpleNamespace(time=time.time, monotonic=time.monotonic, sleep=lambda s: None)


class ClosingStream(BatchStream):
    """Fixture planner that reports epoch 1 closed after its second commit.  The 4-context fixture world cannot close
    an epoch under v1's seed (its last member would repeat an inherited singleton membership), so the runner's
    closed-epoch control flow is exercised through this summary only; the journal itself is unchanged."""
    commits = 0

    def commit(self, *a, **k):
        out = super().commit(*a, **k); type(self).commits += 1; return out

    def summary(self):
        s = super().summary()
        return {**s, 'completed_epochs': s['completed_epochs'] + 1} if type(self).commits >= 2 else s


class FakeBoundary:
    def __init__(self): self.calls = []; self.seq = 0; self.status = {'state': 'fake'}
    def step(self, planner, snapshot, checkpoint_sha256, *, at_step):
        self.calls.append(at_step); return snapshot


class V6Fixture:
    """Shared W10-world fixture helpers (no tests here; mixed into each TestCase)."""
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.w = W10.world('idle')

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name).resolve()
        copy_root = self.w.private_copy(self.base / 'copy')
        self.capture = self.base / 'capture'
        self.w.capture(copy_root, self.capture)

    def genesis(self, name='campaign', **over):
        args = dict(preprocessing_raw=self.w.pre_raw, campaign_sha256=CAMPAIGN, contract=CW.contract(self.w.pre_raw),
                    r14_receipt_sha256=R14_RECEIPT, parity_receipts=PARITY, outbox_max_pending=100,
                    training_objective_sha256=digest(AS.training_objective(self.anchor_kl())),
                    next_update_parity=None if over.get('probe') else parity_receipt(self.capture))
        args.update(over)
        return G.build(self.capture, self.base / name, **args)

    def config(self, **over):
        n = copy.deepcopy(self.w.config['neural'])
        n.update(max_new_updates=2, emissions_per_step=1, step_budget_floor_seconds=0.001, memory_floor_fraction=0.0,
                 memory_exit_fraction=-1.0, memory_exit_checks=3, max_pending_emissions=100, run_id='fixture-v2',
                 epoch_boundary_budget_seconds=0.001, watermark_budget_seconds=60, closure_budget_seconds=60,
                 regression_guard={'reference_loss': 1e9, 'ratio': 1.02, 'consecutive_epochs': 3},
                 kl_guard=self.kl_policy(), anchor_kl=self.anchor_kl())
        n.update(over)
        n.setdefault('training_objective_sha256', digest(AS.training_objective(n['anchor_kl'])))
        return {'neural': n, 'inherited_backlog': {'fixture': True}}

    def anchor_kl(self, **over):
        """R23 anchor-KL block exactly as the builder emits it (the W10 deployed trie is one program: KL is 0 there)."""
        _, _, ancestry = FC.load_capture(self.capture)
        return {'weight': 0.2, 'contexts_per_step': 1, 'anchor_parameter_sha256': ancestry['checkpoint_sha256'],
                'form': AS.FORM, 'temperature': 1.0, 'max_tokens_per_chunk': 16384, **over}

    def kl_policy(self, **over):
        """T12 KL-guard policy; permissive by default (like regression_guard's 1e9 reference) so only KL tests trip it."""
        _, _, ancestry = FC.load_capture(self.capture)
        return {'probe_contexts': 2, 'alarm_mean': 1e8, 'hold_mean': 1e9, 'hold_max': 1e9,
                'anchor_parameter_sha256': ancestry['checkpoint_sha256'], **over}

    def run_v6(self, root, cfg, resume, *, boundary=None, deadline=None, memory=lambda: 0.5, ancestry=None, **kw):
        if ancestry is None: _, _, ancestry = FC.load_capture(self.capture)
        kw.setdefault('kl_anchor', FP.captured_inputs(self.capture)['checkpoint'])
        boundary = boundary or FakeBoundary()
        with mock.patch.object(R6, 'time', SLEEPLESS):
            out = R6.run(self.w.native, self.w.active, cfg, root=root / 'training', campaign_sha=CAMPAIGN, ancestry=ancestry,
                         preprocessing=self.w.pre_raw, deadline=deadline or time.time() + 90, authority_check=lambda s: None,
                         boundary=boundary, resume=resume, memory=memory,
                         genesis_sha256=sha((root / 'genesis.json').read_bytes()), **kw)
        return out, boundary

    def completed(self, name='campaign', updates=2):
        self.genesis(name); root = self.base / name
        self.run_v6(root, self.config(max_new_updates=updates), G.genesis_resume(root, execution_config_sha256=EXEC))
        return root


class RunnerV6Tests(V6Fixture, unittest.TestCase):
    # ------------------------------------------------------------------ genesis
    def test_genesis_inherits_lineage_indexes_and_logs_r14_after_parity(self):
        g = self.genesis()
        root = self.base / 'campaign'; t = root / 'training'
        _, _, ancestry = FC.load_capture(self.capture)
        self.assertEqual(g['origin_absolute_step'], ancestry['optimizer_steps'])
        manifest_raw, active = Bd.active_manifest(t / 'batches.sqlite')
        self.assertEqual(manifest_raw, (self.capture / 'active-manifest.json').read_bytes())
        self.assertEqual(json.loads(manifest_raw)['snapshot_sha256'], self.w.active.identity)
        db = sqlite3.connect(t / 'batches.sqlite'); self.assertTrue(R6.has_indexes(db, R6.PLANNER_INDEXES)); db.close()
        db = sqlite3.connect(t / 'outbox.sqlite'); self.assertTrue(R6.has_indexes(db, R6.OUTBOX_INDEXES)); db.close()
        event = json.loads((t / 'control-event-r14.json').read_bytes())
        self.assertEqual(event['after']['group_learning_rates'], [3.75e-5, 3.75e-5])
        before = event['before']['plateau']
        self.assertEqual(event['after']['plateau'], {'best': None, 'bad': 0, 'reductions': 0, 'patience': before['patience'],
                                                     'floor': before['floor']})
        self.assertEqual(event['before'], json.loads((t / 'v1-final-optimizer-control.json').read_bytes()))
        self.assertEqual(event['adamw_moments'], 'kept')
        state = load((t / 'rolling' / f'{g["origin_absolute_step"]:09d}' / 'state.safetensors').read_bytes())
        self.assertTrue(any(k.startswith('optimizer.') for k in state))
        with self.assertRaisesRegex(ValueError, 'parity gates'):
            self.genesis('no-parity', parity_receipts={'cpu': PARITY['cpu']})
        with self.assertRaisesRegex(ValueError, 'normalizer'):
            self.genesis('bad-pre', preprocessing_raw=self.w.pre_raw + b' ')
        with self.assertRaisesRegex(ValueError, 'new campaign root'):
            self.genesis()

    # ------------------------------------------------------------------ runner
    def test_runner_v6_trains_from_genesis_with_origin_proofs_and_bounded_closure(self):
        g = self.genesis(); root = self.base / 'campaign'; t = root / 'training'
        resume = G.genesis_resume(root, execution_config_sha256=EXEC)
        ClosingStream.commits = 0
        out, boundary = self.run_v6(root, self.config(max_new_updates=2), resume, planner_factory=ClosingStream)
        origin = g['origin_absolute_step']
        self.assertEqual(out, {'state': 'maximum_updates_reached', 'new_optimizer_steps': 2, 'absolute_optimizer_steps': origin + 2})
        for step in range(origin + 1, origin + 3):
            proof = json.loads((t / f'batch-{step:09d}.json').read_bytes())
            self.assertEqual((proof['schema'], proof['absolute_optimizer_step'], proof['snapshot_sha256'], proof['adoption_seq']),
                             ('jope.continuous-neural-batch-proof/v2', step, self.w.active.identity, 0))
            self.assertFalse(proof['fresh_generated_rewards_used'])
            self.assertEqual(len(proof['emission_ack']['emission_ids']), 1)
        result = json.loads((t / 'result.json').read_bytes()); ver = json.loads((t / 'result-verification.json').read_bytes())
        self.assertEqual((result['state'], result['absolute_optimizer_steps']), ('maximum_updates_reached', origin + 2))
        self.assertEqual((ver['planner_verification']['committed_batches'], ver['planner_verification']['unresolved_batches']), (2, 0))
        self.assertEqual(ver['outbox_verification']['observations'], 2)
        self.assertLess(ver['seconds']['closure'], 60)
        self.assertEqual((result['schema'], result['verified'], result['verification_sha256']),
                         ('jope.continuous-neural-result/v3', True, sha((t / 'result-verification.json').read_bytes())))
        self.assertEqual(ver['intent_sha256'], sha((t / 'terminal-intent.json').read_bytes()))
        mark = json.loads((t / 'verified-watermark.json').read_bytes())
        self.assertEqual((mark['source'], mark['watermark']['local_step']), ('closed epoch', 2), 'epoch watermark persisted')
        self.assertEqual((ver['watermark_report']['from_step'], ver['watermark_report']['joined_steps']), (2, 0),
                         'the closure joins only the suffix after the last epoch watermark')
        control = json.loads((t / 'optimizer-control.json').read_bytes())
        self.assertEqual(control['absolute_optimizer_step'], origin + 2)
        self.assertEqual(control['group_learning_rates'], [3.75e-5, 3.75e-5])
        self.assertEqual(boundary.calls, [origin + 2], 'boundary driven exactly at the closed epoch (no latch)')
        self.assertTrue(list((t / 'epochs').iterdir()), 'closed-epoch checkpoint retained')
        self.assertTrue(list(t.glob('validation-original-epoch-*.json')), 'full original validation at the closed epoch')

    def test_restart_continues_bitwise_from_own_state(self):
        self.genesis('a'); self.genesis('b')
        ra, rb = self.base / 'a', self.base / 'b'
        self.run_v6(ra, self.config(max_new_updates=2), G.genesis_resume(ra, execution_config_sha256=EXEC))
        self.run_v6(rb, self.config(max_new_updates=1), G.genesis_resume(rb, execution_config_sha256=EXEC))
        resume = RV.v2_resume(rb, execution_config_sha256=EXEC)
        self.assertEqual(resume['local_step'], 1)
        self.run_v6(rb, self.config(max_new_updates=2), resume)
        origin = json.loads((ra / 'genesis.json').read_bytes())['origin_absolute_step']
        last = f'{origin + 2:09d}'
        left = load((ra / 'training' / 'rolling' / last / 'state.safetensors').read_bytes())
        right = load((rb / 'training' / 'rolling' / last / 'state.safetensors').read_bytes())
        self.assertEqual(set(left), set(right))
        for key in left:
            self.assertTrue(torch.equal(left[key], right[key]), key)
        a = json.loads((ra / 'training' / f'batch-{origin + 2:09d}.json').read_bytes())
        b = json.loads((rb / 'training' / f'batch-{origin + 2:09d}.json').read_bytes())
        self.assertEqual(a['batch']['ordered_data_run_ids'], b['batch']['ordered_data_run_ids'])

    def test_restart_refuses_failure_or_mismatched_state(self):
        self.genesis(); root = self.base / 'campaign'
        self.run_v6(root, self.config(max_new_updates=2), G.genesis_resume(root, execution_config_sha256=EXEC))
        (root / 'training' / 'failure.json').write_bytes(b'{}')
        with self.assertRaisesRegex(ValueError, 'review required'):
            RV.v2_resume(root, execution_config_sha256=EXEC)

    def test_tick_safe_stop_never_reserves_inside_the_budget(self):
        self.genesis(); root = self.base / 'campaign'; t = root / 'training'
        resume = G.genesis_resume(root, execution_config_sha256=EXEC)
        out, _ = self.run_v6(root, self.config(max_new_updates=50, step_budget_floor_seconds=30), resume, deadline=time.time() + 20)
        self.assertEqual((out['state'], out['new_optimizer_steps']), ('global_deadline_reached', 0))
        self.assertFalse((t / 'failure.json').exists()); self.assertFalse((t / 'uncertain-physical-state').exists())
        self.assertEqual(json.loads((t / 'result.json').read_bytes())['state'], 'global_deadline_reached')
        self.assertTrue((t / 'result-verification.json').exists())

    def test_memory_gate_waits_then_exits_cleanly_without_reservation(self):
        self.genesis(); root = self.base / 'campaign'; t = root / 'training'
        resume = G.genesis_resume(root, execution_config_sha256=EXEC)
        readings = iter([0.10, 0.10, 0.05, 0.05, 0.05])
        out, _ = self.run_v6(root, self.config(max_new_updates=5, memory_floor_fraction=0.15, memory_exit_fraction=0.13,
                                               memory_exit_checks=3), resume, memory=lambda: next(readings, 0.05))
        self.assertEqual((out['state'], out['new_optimizer_steps']), ('memory_floor_exit', 0))
        db = sqlite3.connect(t / 'batches.sqlite'); n = db.execute('SELECT count(*) FROM batches').fetchone()[0]; db.close()
        self.assertEqual(n, 0, 'no reservation while waiting for memory')

    def test_own_row_cap_waits_without_reservation(self):
        self.genesis(outbox_max_pending=2); root = self.base / 'campaign'
        resume = G.genesis_resume(root, execution_config_sha256=EXEC)
        calls = []
        deadline = time.time() + 600
        def remaining(d):
            calls.append(1); return 600 if len(calls) < 40 else -1
        with mock.patch.object(R6, 'remaining', side_effect=remaining):
            out, _ = self.run_v6(root, self.config(max_new_updates=10, max_pending_emissions=2), resume, deadline=deadline)
        self.assertEqual(out['state'], 'global_deadline_reached')
        self.assertLessEqual(out['new_optimizer_steps'], 2)
        p = json.loads((root / 'training' / 'progress.json').read_bytes())
        self.assertLessEqual(p['unresolved_programs'], 2)

    def test_runner_refuses_planner_without_bounded_verify_index(self):
        self.genesis(); root = self.base / 'campaign'
        db = sqlite3.connect(root / 'training' / 'batches.sqlite', isolation_level=None)
        db.execute('DROP INDEX v2_members_batch'); db.close()
        resume = G.genesis_resume(root, execution_config_sha256=EXEC)
        with self.assertRaisesRegex(ValueError, 'bounded-verify planner indexes'):
            self.run_v6(root, self.config(), resume)

    def test_real_empty_inbox_boundary_reports_idle(self):
        self.genesis(); root = self.base / 'campaign'; t = root / 'training'
        resume = G.genesis_resume(root, execution_config_sha256=EXEC)
        self.run_v6(root, self.config(max_new_updates=2), resume)
        inbox = Bd.Inbox(root / 'inbox')
        driver = R6.BoundaryDriver(inbox, self.w.native, root=t, campaign_sha256=CAMPAIGN,
                                   preprocessing_sha256=sha(self.w.pre_raw), registry_sha256='0' * 64,
                                   fresh_authority=lambda: b'{}', authority_check=lambda raw, proof: None)
        raw, active = Bd.active_manifest(t / 'batches.sqlite')
        _, _, ancestry = FC.load_capture(self.capture)
        planner = BatchStream(t / 'batches.sqlite', raw, active, seed=self.w.config['neural']['seed'],
                              batch_size=self.w.config['neural']['batch_size'], initial_checkpoint_sha256=ancestry['checkpoint_sha256'])
        try:
            state = inbox.reconcile(planner)
            self.assertEqual((state['records'], state['holds']), ([], []))
            same = driver.step(planner, self.w.active, planner.summary()['checkpoint_sha256'], at_step=0)
        finally:
            planner.close()
        self.assertIs(same, self.w.active)
        status = json.loads((t / 'boundary-status.json').read_bytes())
        self.assertEqual((status['state'], status['adoption_seq']), ('idle', 0))

if __name__ == '__main__':
    unittest.main()
