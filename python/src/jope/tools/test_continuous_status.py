import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import continuous_status as status


class ContinuousStatusTests(unittest.TestCase):
    def test_process_bound_gp_activity_expires_without_erasing_history(self):
        process = status.process_identity(os.getpid())
        metadata = Path('/proc/self/exe').stat()
        binding = dict(schema='parallel.status_binding.v1', controller_generation='a' * 64,
            executable_sha256='b' * 64,
            process=dict(pid=os.getpid(), birth_ticks=int(process['birth']), boot_id=status.boot_identity()),
            executable=dict(device=metadata.st_dev, inode=metadata.st_ino, bytes=metadata.st_size,
                mtime=metadata.st_mtime_ns // 1_000_000_000, mtime_ns=metadata.st_mtime_ns % 1_000_000_000,
                ctime=metadata.st_ctime_ns // 1_000_000_000, ctime_ns=metadata.st_ctime_ns % 1_000_000_000))
        receipt = dict(timestamp=100, expires_at=110, last_progress_at=100, phase='running', active=104,
            completed_views=50801, status_binding=binding)
        self.assertEqual(status.observed_gp_status(receipt, 109)['current_activity'], 'current')
        for now in (99, 110, 1000):
            stale = status.observed_gp_status(receipt, now)
            self.assertEqual(stale['current_activity'], 'stale')
            self.assertIsNone(stale['active'])
            self.assertEqual(stale['completed_views'], 50801)
            self.assertEqual(stale['last_recorded_activity']['active'], 104)
            self.assertEqual(status.observed_gp_status(stale, now), stale)
        for changes in ({'process': dict(binding['process'], birth_ticks=1)},
                        {'process': dict(binding['process'], boot_id='another boot')},
                        {'executable': dict(binding['executable'], inode=0)}):
            changed = dict(receipt, status_binding=dict(binding, **changes))
            self.assertEqual(status.observed_gp_status(changed, 105)['current_activity'], 'stale')
        self.assertEqual(status.observed_gp_status(dict(receipt, status_binding={}), 105)['phase'], 'stale')
        self.assertEqual(status.observed_gp_status(dict(receipt, timestamp=200, expires_at=210), 205)['phase'], 'stale')
        terminal = status.observed_gp_status(dict(receipt, terminal=True), 1000)
        self.assertEqual(terminal['current_activity'], 'terminal')
        self.assertIsNone(terminal['active'])

    def test_live_gp_service_reports_empty_queue_and_recovers(self):
        gp=dict(phase='waiting',active=0,queued=0,timestamp=100)
        since,alert=status.gp_queue_health(gp,False,100,{})
        self.assertEqual(since,100)
        self.assertEqual(alert['severity'],'warning')
        since,alert=status.gp_queue_health(gp,False,220,dict(gp_queue_empty_since=since))
        self.assertEqual(alert['severity'],'incident')
        for change in (dict(active=104),dict(queued=420)):
            self.assertEqual(status.gp_queue_health(dict(gp,**change),False,221,
                dict(gp_queue_empty_since=since)),(None,None))
        self.assertEqual(status.gp_queue_health(gp,True,221,{}),(None,None))

    def test_live_process_cannot_hide_stale_receipt_or_nonadvancing_source(self):
        teacher=dict(schema='parallel.continuous_teacher_progress.v1', manifest_sha256='m',
                     timestamp=100, reporter_pid=123, process_alive=True, source_stale=False)
        self.assertIsNone(status.teacher_health(teacher,'m',200))
        self.assertEqual(status.teacher_health(teacher,'m',301)[0],'teacher_reporter_stale')
        teacher.update(timestamp=301,source_stale=True)
        self.assertEqual(status.teacher_health(teacher,'m',301)[0],'teacher_source_stale')

    def monitor(self, **changes):
        value = dict(schema='parallel.frozen_monitoring.v1', binding=dict(panel_sha256='panel',
            corpus_sha256='corpus', executable_sha256='exe'), checkpoint='parent', timestamp=1,
            sampled_examples=2048, family_balanced=True, independent_evaluation=False,
            training_selection_used=False, final_tests_opened=False,
            sequence_nll=1.0, ranking_bt=0.5, ranking_listwise=2.0, runtime_huber=0.1)
        value.update(changes)
        return value

    def quality(self):
        rows = [dict(output=arm, dataset='dataset', family='family', partition='meta_validation',
            cut=0, arm=arm, rng_seed=123, metric='rmse', units='original_target',
            best_final_gp=score, best_retained_teacher=3.0, best_initial_neural_seed=2.5)
            for arm, score in [('neural', 1.0), ('generic', 2.0), ('shuffled', 3.0)]]
        return dict(schema='parallel.continuous_quality.v1', rows=rows,
                    final_tests_opened=False, independent_evaluation=False), dict(jobs=[
                        dict(output=r['output'], rng_seed=123) for r in rows])

    def test_monitors_sort_by_time_and_show_regression(self):
        result = status.compare_monitoring(self.monitor(), [self.monitor(timestamp=3, sequence_nll=1.2),
            self.monitor(timestamp=2, sequence_nll=0.9)])
        self.assertEqual(result['latest']['timestamp'], 3)
        self.assertEqual(result['best_nll']['sequence_nll'], 0.9)
        self.assertEqual(result['regressed_metrics'], ['sequence_nll'])

    def test_executable_upgrade_retains_historical_regressions_in_separate_series(self):
        baseline=self.monitor();baseline['binding']['executable_sha256']='new-exe'
        old=self.monitor(timestamp=2,sequence_nll=1.2)
        result=status.monitoring_series(baseline,[self.monitor(),old])
        self.assertEqual(result['binding']['executable_sha256'],'new-exe')
        self.assertEqual(result['latest']['sequence_nll'],1)
        self.assertEqual(result['archived_policies'][0]['regressed_metrics'],['sequence_nll'])
        wrong=self.monitor();wrong['binding']['panel_sha256']='changed'
        with self.assertRaisesRegex(ValueError,'incomparable'):status.monitoring_series(baseline,[wrong])

    def test_changed_panel_corpus_or_executable_is_rejected(self):
        for key in ('panel_sha256', 'corpus_sha256', 'executable_sha256'):
            receipt = self.monitor()
            receipt['binding'][key] = 'changed'
            with self.assertRaisesRegex(ValueError, 'incomparable'):
                status.compare_monitoring(self.monitor(), [receipt])

    def test_nonfinite_loss_is_not_improvement(self):
        for score in (float('nan'), float('inf'), None, True):
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                status.compare_monitoring(self.monitor(), [self.monitor(sequence_nll=score)])

    def test_changed_sampling_and_selection_scope_are_rejected(self):
        for change in (dict(sampled_examples=1024), dict(training_selection_used=True),
                       dict(final_tests_opened=True)):
            with self.assertRaises(ValueError):
                status.compare_monitoring(self.monitor(), [self.monitor(**change)])

    def test_initial_seeds_are_distinct_from_later_gp(self):
        report, jobs = self.quality()
        report['rows'][0]['best_initial_neural_seed'] = 4.0
        counts = status.summarize_quality(report, jobs)['meta_validation']
        self.assertEqual(counts['neural_exclusive_wins'], 1)
        self.assertEqual(counts['direct_seed_beats_teacher'], 0)
        self.assertEqual(counts['neural_final_beats_teacher'], 1)

    def test_missing_metrics_are_not_wins_or_imputed_pairs(self):
        report, jobs = self.quality()
        report['rows'][1]['best_final_gp'] = None
        report['rows'][0]['best_initial_neural_seed'] = None
        counts = status.summarize_quality(report, jobs)['meta_validation']
        self.assertEqual(counts['missing_final'], 1)
        self.assertEqual(counts['comparable'], 0)
        self.assertEqual(counts['direct_seed_teacher_pairs'], 0)

    def test_missing_duplicate_or_mismatched_arms_are_rejected(self):
        for mutation in ('missing', 'duplicate', 'rng'):
            report, jobs = self.quality()
            if mutation == 'missing':
                report['rows'].pop()
            elif mutation == 'duplicate':
                report['rows'][1] = copy.deepcopy(report['rows'][0])
            else:
                report['rows'][1]['rng_seed'] = 124
            with self.assertRaises(ValueError):
                status.summarize_quality(report, jobs)

    def test_float_noise_is_a_tie(self):
        report, jobs = self.quality()
        report['rows'][1]['best_final_gp'] = 1.0 + 1e-11
        counts = status.summarize_quality(report, jobs)['meta_validation']
        self.assertEqual(counts['neural_best_ties'], 1)
        self.assertEqual(counts['neural_exclusive_wins'], 0)

    def test_training_and_validation_have_separate_denominators(self):
        report, jobs = self.quality()
        extra = copy.deepcopy(report['rows'])
        for row in extra:
            row.update(output=row['output'] + '-train', partition='meta_train', dataset='train')
        report['rows'].extend(extra)
        jobs['jobs'].extend(dict(output=r['output'], rng_seed=123) for r in extra)
        result = status.summarize_quality(report, jobs)
        self.assertEqual(result['meta_train']['triplets'], 1)
        self.assertEqual(result['meta_validation']['triplets'], 1)

    def test_partial_metric_write_keeps_last_complete_record(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'metrics.jsonl'
            path.write_bytes(b'{"step":10}\n{"step":11')
            self.assertEqual(status.last_metric(path), dict(step=10))
            path.write_bytes(b'{"step":10}\ninvalid\n')
            with self.assertRaises(json.JSONDecodeError):
                status.last_metric(path)

    def test_long_metrics_tail_discards_only_partial_first_record(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'metrics.jsonl'
            path.write_text(''.join(json.dumps(dict(step=i, padding='x' * 100)) + '\n' for i in range(1000)))
            self.assertEqual(status.last_metric(path)['step'], 999)

    def test_receipt_paths_cannot_escape_workspace(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'outside workspace'):
                status.within(Path(folder), '../another-workspace')


class MigrationFenceStatusTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.directory=self.root/'runs/migration/live-fence'
        self.override=self.root/'systemd/jope-parallel-gp.service.d/fence.conf'
        self.directory.mkdir(parents=True);self.override.parent.mkdir(parents=True)
        self.permit=self.directory/'REGISTRY_HANDOFF_REQUIRED'
        self.override.write_text(f'# Registry migration fence: {self.directory}\n[Unit]\nConditionPathExists={self.permit}\n[Service]\nRestart=no\n')
        self.request=dict(schema='parallel.legacy_fit_fence.v1',root=str(self.root),unit=status.GP_UNIT,
            boot='original-boot',pid=4123,process=dict(pid=4123,birth='7'),
            properties=dict(ControlGroup='/system.slice/'+status.GP_UNIT))
        self.write(self.directory/'request.json',self.request)
        self.snapshot=dict(request_sha256=status.digest(self.directory/'request.json'),
            cgroup='/system.slice/'+status.GP_UNIT,processes=[dict(pid=4123,birth='7')])
        self.write(self.directory/'frozen.json',self.snapshot)
        frozen=status.digest(self.directory/'frozen.json')
        self.write(self.directory/'kill-intent.json',dict(frozen_sha256=frozen))
        self.completion=dict(schema='parallel.legacy_fit_fence_completion.v1',timestamp=100,
            request_sha256=status.digest(self.directory/'request.json'),frozen_sha256=frozen,
            permanent_override=str(self.override),override_sha256=status.digest(self.override),
            legacy_dispatch_fenced=True,descendants_reconciled=True,registry_dispatch_enabled=False,
            historical_import_complete=False)
        self.write(self.directory/'complete.json',self.completion)
        self.gp=dict(MainPID='0',ControlPID='0',ActiveState='failed',SubState='failed',ControlGroup='',
            Restart='no',DropInPaths=str(self.override),Requires='')
        self.controller=dict(self.gp,ActiveState='inactive',SubState='dead',DropInPaths='',Requires=status.GP_UNIT)
        self.conditions=[['ConditionPathExists',False,False,str(self.permit),0]]

    @staticmethod
    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))

    def verify(self,process=None,pids=None,boot='original-boot'):
        with patch.object(status,'systemd_properties',side_effect=lambda unit:self.gp if unit==status.GP_UNIT else self.controller), \
            patch.object(status,'systemd_conditions',return_value=self.conditions), \
            patch.object(status,'process_identity',return_value=process), \
            patch.object(status,'cgroup_pids',return_value=pids or []), \
            patch.object(status,'boot_identity',return_value=boot):
            return status.verified_migration_fence(self.root,1000,self.override)

    def test_receipts_current_condition_and_empty_process_scope_establish_only_planned_stop(self):
        result=self.verify()
        self.assertTrue(result['verified']);self.assertEqual(result['phase'],'planned_migration')
        self.assertTrue(result['controller_stopped_by_dependency'])
        self.assertFalse(result['historical_import_complete'])
        self.assertFalse(result['collection_active']);self.assertFalse(result['useful_gpu_work_established'])

    def test_absent_override_never_infers_plan_from_old_receipt(self):
        self.override.unlink();self.assertIsNone(self.verify())

    def test_incomplete_or_tampered_original_receipts_are_unverified(self):
        (self.directory/'complete.json').unlink();self.assertFalse(self.verify()['verified'])
        self.write(self.directory/'complete.json',self.completion)
        self.snapshot['processes'][0]['birth']='changed';self.write(self.directory/'frozen.json',self.snapshot)
        self.assertFalse(self.verify()['verified'])

    def test_live_or_restarted_dispatcher_cannot_be_hidden_by_saved_completion(self):
        self.assertFalse(self.verify(process=dict(pid=4123,birth='7',state='S'))['verified'])
        self.assertTrue(self.verify(process=dict(pid=4123,birth='8',state='S'))['verified'])
        self.assertFalse(self.verify(pids=[5000])['verified'])
        self.gp.update(MainPID='5000',ActiveState='active')
        self.assertFalse(self.verify()['verified'])

    def test_loaded_condition_absent_negated_triggered_or_satisfied_is_not_a_fence(self):
        original=copy.deepcopy(self.conditions)
        for row in [[],[['ConditionPathExists',False,True,str(self.permit),0]],
                    [['ConditionPathExists',True,False,str(self.permit),0]]]:
            self.conditions=row;self.assertFalse(self.verify()['verified'])
        self.conditions=original;self.permit.touch();self.assertFalse(self.verify()['verified'])
        self.permit.unlink();self.gp['DropInPaths']='';self.assertFalse(self.verify()['verified'])

    def test_reboot_and_restart_policy_changes_require_reconciliation(self):
        self.assertFalse(self.verify(boot='new-boot')['verified'])
        self.gp['Restart']='always';self.assertFalse(self.verify()['verified'])

    def test_unrelated_controller_stop_does_not_suppress_its_health_incident(self):
        self.controller['Requires']='';result=self.verify()
        self.assertTrue(result['verified']);self.assertFalse(result['controller_stopped_by_dependency'])

    def observation_fixture(self):
        manifest_path=self.root/'manifest.json';baseline_path=self.root/'baseline.json'
        manifest=dict(output='runs/current',monitor_panel='panel.json',cpu_executable='cpu',
            monitor_executable='monitor',bindings=dict(cpu='cpu-sha',monitor='monitor-sha'),world=2)
        self.write(manifest_path,manifest);manifest_sha=status.digest(manifest_path)
        checkpoint='runs/checkpoint';self.write(self.root/checkpoint/'state.json',dict(step=10))
        self.write(self.root/'panel.json',dict(panel_sha256='panel',corpus_sha256='corpus'))
        baseline=ContinuousStatusTests().monitor(checkpoint=checkpoint,timestamp=900)
        baseline['binding'].update(executable_sha256='monitor-sha',
            checkpoint_sha256=status.digest(self.root/checkpoint/'state.json'))
        self.write(baseline_path,baseline)
        self.write(self.root/'runs/current/state.json',dict(manifest_sha256=manifest_sha,tasks={},
            checkpoint=checkpoint,completed_blocks=4,generations=[],snapshots=[],trained_snapshots=[],
            serving_checkpoint=checkpoint,serving_status='rolled_back_to_recovery'))
        self.write(self.root/'runs/continuous-progress.json',dict(manifest_sha256=manifest_sha,
            output=manifest['output'],timestamp=100,pid=4124))
        self.write(self.root/'runs/gp-service.json',dict(timestamp=100,phase='running',active=104,queued=560,
            completed_views=30,completed_evaluations=4000,failed_views=104,retry_pending=0,executable_sha256='cpu-sha'))
        self.write(self.root/'runs/continuous-control.json',dict(manifest_sha256=manifest_sha,paused=False))
        self.write(self.root/'runs/teacher-progress.json',dict(schema='parallel.continuous_teacher_progress.v1',
            manifest_sha256=manifest_sha,timestamp=990,source_stale=False,last_meaningful_improvement_at=200))
        self.write(self.root/'runs/teacher-reporter.json',dict(timestamp=995,pending_searches=2,oldest_pending_completed_at=800))
        return manifest_path,baseline_path

    def observe(self,migration):
        manifest,baseline=self.observation_fixture()
        with patch.object(status,'verified_migration_fence',return_value=migration), \
            patch.object(status.time,'time',return_value=1000), \
            patch.object(status.subprocess,'check_output',return_value='0, 0, 16000\n1, 0, 16000\n'):
            return status.observe(self.root,manifest,baseline,{})

    def test_planned_migration_replaces_live_phase_but_preserves_history_and_distinct_clocks(self):
        result=self.observe(self.verify())
        self.assertEqual(result['health'],'planned_migration');self.assertEqual(result['operational_status'],'planned_migration')
        self.assertEqual(result['gp']['phase'],'planned_migration');self.assertEqual(result['gp']['active'],0)
        self.assertEqual(result['gp']['last_receipt']['phase'],'running')
        self.assertEqual(result['gp']['last_receipt']['active'],104)
        self.assertEqual(result['gp']['last_receipt']['failed_views'],104)
        self.assertEqual(result['controller_pid'],0);self.assertEqual(result['controller_last_recorded_pid'],4124)
        self.assertNotIn('gp_health',{a['code'] for a in result['alerts']})
        self.assertNotIn('controller_heartbeat',{a['code'] for a in result['alerts']})
        self.assertEqual(result['progress_ages']['gp_heartbeat_seconds'],900)
        self.assertEqual(result['progress_ages']['teacher_reporter_heartbeat_seconds'],5)
        self.assertEqual(result['progress_ages']['oldest_unprocessed_receipt_seconds'],200)
        self.assertEqual(result['progress_ages']['seconds_since_meaningful_improvement'],800)
        self.assertEqual(result['scientific_status'],'transfer_gain_not_established')
        self.assertEqual(result['learning_status'],'controller_stopped_for_migration')
        self.assertFalse(result['useful_gpu_work_established'])

    def test_unverified_fence_remains_an_incident(self):
        result=self.observe(dict(verified=False,reason='descendant survives'))
        self.assertEqual(result['health'],'incident')
        self.assertIn('migration_fence_unverified',{a['code'] for a in result['alerts']})
        self.assertIn('gp_activity_unknown',{a['code'] for a in result['alerts']})

    def test_unknown_progress_times_are_not_zero_or_recent_improvement(self):
        ages=status.progress_ages({}, {}, {}, {},1000)
        self.assertIsNone(ages['oldest_unprocessed_receipt_seconds'])
        self.assertIsNone(ages['seconds_since_meaningful_improvement'])

    def test_planned_stop_does_not_mask_an_unrelated_controller_incident(self):
        migration=self.verify();migration['controller_stopped_by_dependency']=False
        result=self.observe(migration)
        self.assertEqual(result['health'],'incident')
        self.assertIn('controller_heartbeat',{a['code'] for a in result['alerts']})
        self.assertEqual(result['controller_pid'],4124)


if __name__ == '__main__':
    unittest.main()
