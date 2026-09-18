import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import activate_promotion_safeguards as activate
from resume_expired_cycle import atomic, read, sha


class ActivationRecoveryTests(unittest.TestCase):
    def test_health_keeps_effective_baseline_and_waits_for_new_manifest_progress(self):
        unit='ExecStart=/old --baseline old --output status\n[Service]\nExecStart=\nExecStart=/new --baseline qualified --output status\n'
        self.assertEqual(activate.health_launch(unit),'ExecStart=/new --baseline qualified --output status')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'runs').mkdir();manifest=root/'manifest.json';atomic(manifest,{})
            atomic(root/'runs/continuous-progress.json',dict(manifest_sha256='old'))
            with self.assertRaisesRegex(RuntimeError,'has not published'):
                activate.wait_controller_identity(root,manifest,timeout=0)
            atomic(root/'runs/continuous-progress.json',dict(manifest_sha256=sha(manifest)))
            activate.wait_controller_identity(root,manifest,timeout=0)

    def test_superseding_launches_are_reviewed_and_preserved_before_removal(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);system=root/'system';output=root/'output';output.mkdir()
            service='jope-parallel-campaign.service';directory=system/(service+'.d');directory.mkdir(parents=True)
            old='[Service]\nExecStart=\nExecStart=/old/controller\n'
            (directory/activate.OVERRIDE).write_text(old)
            later=directory/'zzzz-flywheel-v3.conf';later.write_text(old+'# peer release\n')
            entries=activate.prepare_overrides(output,[service],system)
            with patch.object(activate.subprocess,'run') as run:
                activate.remove_superseded(entries)
                self.assertEqual(run.call_args.args[0],['sudo','-n','rm','--',str(later)])
                later.write_text(old+'Environment=UNREVIEWED=1\n')
                with self.assertRaisesRegex(RuntimeError,'changed during activation'):
                    activate.remove_superseded(entries)
            self.assertEqual(read(output/'service-overrides.before.json'),entries)
            # A subsequent resume uses the original journal even after removal.
            later.unlink()
            self.assertEqual(activate.prepare_overrides(output,[service],system),entries)

    def test_effective_launch_must_match_reviewed_source_and_manifest(self):
        content='[Service]\nExecStart=\nExecStart=/usr/bin/python3 reviewed.py --manifest new.json\n'
        with patch.object(activate.subprocess,'check_output',return_value='{ argv[]=/usr/bin/python3 stale.py --manifest old.json ; }'):
            with self.assertRaisesRegex(RuntimeError,'effective service launch differs'):
                activate.verify_effective_launches({'campaign':content})

    def test_interrupted_configuration_resumes_without_replacing_newer_state(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'tools';source.mkdir()
            (source/'activate_promotion_safeguards.py').write_text('# frozen activation source\n')
            (root/'runs').mkdir();(root/'state').mkdir()
            (root/'binary').write_bytes(b'qualified executable')
            old=root/'old.json';atomic(old,dict(output='state',bindings={},cpu_executable='reference'))
            control=root/'runs/continuous-control.json'
            atomic(control,dict(schema='parallel.continuous_control.v1',paused=False,manifest_sha256=sha(old)))
            state_path=root/'state/state.json'
            initial=dict(manifest_sha256=sha(old),checkpoint='recovery',serving_checkpoint='recovery',
                serving_checkpoint_sha256='recovery-sha',serving_status='rolled_back_to_recovery',
                tasks={},candidates=[],promotion_decisions=[],consumed_training_snapshots=[])
            atomic(state_path,initial)
            release_path=root/'release.json';atomic(release_path,{})
            (root/'workers').mkdir()
            atomic(root/'workers/monitor-0.json',{'qualified':'baseline'})
            atomic(root/'workers/complete.json',{'stages':{'monitor-0':{'receipt_sha256':sha(root/'workers/monitor-0.json')}}})
            release=dict(settings=dict(training_executable='binary',packing_executable='binary',
                monitor_executable='binary',gpu_pool='runs/pool',teaching_store='runs/store'),teaching_sources=['first'],
                workers={'path':'workers/complete.json'})
            deployment=root/'deployment';phase={'interrupted':False}

            def output(command,**kwargs):
                if command[:2]==['systemctl','cat']:
                    return 'ExecStart=/usr/bin/python3 observer.py --baseline baseline --output health\n'
                if 'jope-parallel-gp.service' in command:return '2342\n'
                if 'ExecStart' in command:
                    content=(deployment/(command[2]+'.conf')).read_text()
                    start=next(line.partition('=')[2] for line in content.splitlines()
                               if line.startswith('ExecStart=') and line!='ExecStart=')
                    return '{ argv[]='+start+' ; }'
                if 'MainPID' in command:return '0\n' if phase['interrupted'] else '123\n'
                if 'ActiveState' in command:return 'inactive\n'
                if 'ControlGroup' in command:return '' if phase['interrupted'] else '/controller\n'
                raise AssertionError(command)

            def run(command,**kwargs):
                if command[:4]==['sudo','-n','systemctl','start'] and command[-1]=='jope-parallel-campaign.service':
                    atomic(root/'runs/continuous-progress.json',dict(manifest_sha256=sha(deployment/'manifest.json')))
                if command[:3]==['sudo','-n','install'] and not phase['interrupted']:
                    phase['interrupted']=True
                    raise RuntimeError('injected configuration interruption')

            args=['activate','--root',str(root),'--manifest',str(old),'--output',str(deployment),
                  '--operational-release',str(release_path)]
            with patch.object(activate,'__file__',str(source/'activate_promotion_safeguards.py')), \
                 patch.object(activate.operational_release,'verify',return_value=release), \
                 patch.object(activate.operational_release,'stop_controller'), \
                 patch.object(activate.subprocess,'run',side_effect=run), \
                 patch.object(activate.subprocess,'check_output',side_effect=output), \
                 patch.object(activate.time,'sleep'):
                with patch('sys.argv',args),self.assertRaisesRegex(RuntimeError,'configuration interruption'):
                    activate.main()
                newer=read(state_path)
                self.assertEqual(newer['manifest_sha256'],sha(deployment/'manifest.json'))
                self.assertFalse(read(control)['paused'])
                newer.update(checkpoint='newer',serving_checkpoint='newer',serving_checkpoint_sha256='newer-sha',
                    serving_status='qualified',teaching_sources=['first','second'],
                    candidates=[{'checkpoint':'candidate'}],promotion_decisions=[{'outcome':'rejected'}],
                    consumed_training_snapshots=['consumed'])
                atomic(state_path,newer)
                with patch('sys.argv',args+['--recover-stopped']):activate.main()
            self.assertEqual(read(state_path),newer)
            self.assertEqual(read(deployment/'state.before.json'),initial)
            self.assertFalse(read(control)['paused'])
            self.assertEqual(read(deployment/'complete.json')['serving_checkpoint'],'newer')
