import unittest
import signal
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch
import operational_release as release
from operational_release import migrate,fallback


class OperationalMigrationTests(unittest.TestCase):
    def test_controller_handoff_delegates_only_after_the_durable_boundary(self):
        events=[]
        with patch.object(release,'read',return_value={'tasks':{}}), \
             patch.object(release,'controller_process',return_value=dict(state='T',birth='1')), \
             patch.object(release.os,'kill',side_effect=lambda pid,sig:events.append(sig)), \
             patch.object(release,'wait_gpu_boundaries',side_effect=lambda *args:events.append('durable')), \
             patch('stopped_unit.stop',side_effect=lambda *args:events.append('safe_stop')):
            release.stop_controller(Path('/workspace'),Path('/state'),123,'/group',100)
        self.assertEqual(events,[signal.SIGSTOP,'durable','safe_stop'])

    def test_handoff_timeout_resumes_controller_without_terminating_it(self):
        events=[]
        with patch.object(release,'read',return_value={'tasks':{}}), \
             patch.object(release,'controller_process',return_value=dict(state='T',birth='1')), \
             patch.object(release.os,'kill',side_effect=lambda pid,sig:events.append(sig)), \
             patch.object(release,'wait_gpu_boundaries',side_effect=RuntimeError('timeout')), \
             patch.object(release.subprocess,'run',side_effect=AssertionError('must not stop')):
            with self.assertRaisesRegex(RuntimeError,'timeout'):
                release.stop_controller(Path('/workspace'),Path('/state'),123,'/group',100)
        self.assertEqual(events,[signal.SIGSTOP,signal.SIGCONT])

    def test_real_controller_cannot_dispatch_from_its_termination_handler(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);state=root/'state.json';state.write_text('{"tasks":{}}')
            marker=root/'new-task'
            code='''import signal,sys,time
signal.signal(signal.SIGTERM,lambda *_:open(sys.argv[1],'w').write('resumed'))
print('ready',flush=True)
time.sleep(100)
'''
            parent=subprocess.Popen([sys.executable,'-c',code,str(marker)],stdout=subprocess.PIPE,text=True)
            parent.stdout.readline()
            from test_stopped_unit import Manager
            manager=Manager(root/'runtime',parent)
            try:
                with patch.object(release,'wait_gpu_boundaries'),manager.patch():
                    release.stop_controller(root,state,parent.pid,'/fixture',time.monotonic()+10)
                self.assertEqual(parent.returncode,-signal.SIGKILL)
                self.assertFalse(marker.exists())
                self.assertEqual(len(list(root.glob('controller-stop-*/complete.json'))),1)
            finally:
                if parent.poll() is None:parent.kill();parent.wait()
                parent.stdout.close()

    def test_rollout_and_rollback_preserve_newer_scientific_state_and_pending_receipts(self):
        state=dict(manifest_sha256='old',serving_checkpoint='recovery',serving_status='rolled_back_to_recovery',
            candidates=[{'checkpoint':'candidate'}],promotion_decisions=[{'outcome':'rejected'}],
            consumed_training_snapshots=['shard-v1'],tasks={'train':{'receipt':'pending/complete.json'}})
        rolled=migrate(state,'old','new',['ledger-a'])
        rolled['serving_checkpoint']='qualified-newer-serving'
        rolled['promotion_decisions'].append({'outcome':'promoted'})
        restored=migrate(rolled,'new','fallback')
        self.assertEqual({k:v for k,v in restored.items() if k!='manifest_sha256'},
                         {k:v for k,v in rolled.items() if k!='manifest_sha256'})
        self.assertEqual(state['serving_checkpoint'],'recovery')
        with self.assertRaisesRegex(RuntimeError,'manifest changed'):migrate(rolled,'old','fallback')

    def test_fallback_keeps_shard_reader_and_uses_reference_packer(self):
        manifest=dict(gpu_pool='pool',teaching_store='store',cpu_executable='reference',
                      packing_executable='incremental',training_executable='compact-reader',neural_executable='serving')
        result=fallback(manifest)
        self.assertNotIn('gpu_pool',result);self.assertNotIn('teaching_store',result)
        self.assertEqual(result['packing_executable'],'reference')
        self.assertEqual(result['training_executable'],'compact-reader')
        self.assertEqual(result['neural_executable'],'serving')
