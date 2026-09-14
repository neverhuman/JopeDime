import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from gp_emission_handoff import committed,drain,process,resume,same_process,stop_drained
from promotion_gate import immutable
from resume_expired_cycle import read


class DrainTests(unittest.TestCase):
    def parent(self,root,seconds=1):
        output=root/'search';output.mkdir()
        immutable(output/'observations.json',[{'fixture':True}])
        immutable(output/'search.json',{'schema':'parallel.search.v1','completed_evaluations':1})
        script='import subprocess,sys,time; c=subprocess.Popen([sys.executable,"-c","import time;time.sleep('+str(seconds)+')","search","--output",sys.argv[1]]);print(c.pid,flush=True);c.wait();time.sleep(100)'
        parent=subprocess.Popen([sys.executable,'-c',script,str(output)],stdout=subprocess.PIPE,text=True)
        child=int(parent.stdout.readline());return parent,child

    def test_stopped_supervisor_allows_search_to_finish_and_hashes_committed_ledgers(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);out=root/'handoff';out.mkdir();parent,child=self.parent(root)
            try:
                request={'supervisor':process(parent.pid),'executable':sys.executable}
                immutable(out/'request.json',request)
                rows=drain(root,out,request,seconds=5)
                self.assertEqual(rows[0]['pid'],child);self.assertEqual(rows[0]['completed_evaluations'],1)
                self.assertEqual(process(parent.pid)['state'],'T')
                self.assertIsNotNone(committed(root,read(out/'children.json')))
                resume(out)
            finally:
                os.kill(parent.pid,signal.SIGCONT);parent.terminate();parent.wait();parent.stdout.close()

    def test_timeout_resumes_dispatch_without_killing_active_search(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);out=root/'handoff';out.mkdir();parent,child=self.parent(root,seconds=10)
            try:
                request={'supervisor':process(parent.pid),'executable':sys.executable}
                immutable(out/'request.json',request)
                with self.assertRaisesRegex(RuntimeError,'deadline exceeded'):
                    drain(root,out,request,seconds=.1)
                time.sleep(.05)
                self.assertNotEqual(process(parent.pid)['state'],'T')
                self.assertNotEqual(process(child)['state'],'Z')
                self.assertFalse((out/'drained.json').exists())
                wrong=dict(request['supervisor'],birth=0);self.assertFalse(same_process(wrong))
            finally:
                os.kill(child,signal.SIGTERM);os.kill(parent.pid,signal.SIGCONT)
                parent.terminate();parent.wait();parent.stdout.close()

    def test_shutdown_never_resumes_a_dispatch_loop_after_term(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);out=root/'handoff';out.mkdir();destination=root/'search';destination.mkdir()
            immutable(destination/'observations.json',[{'fixture':True}])
            immutable(destination/'search.json',dict(schema='parallel.search.v1',completed_evaluations=1))
            marker=root/'dispatched-again'
            script='''import signal,subprocess,sys,time
signal.signal(signal.SIGTERM,lambda *_:None)
c=subprocess.Popen([sys.executable,'-c','import time;time.sleep(1)','search','--output',sys.argv[1]])
print(c.pid,flush=True)
c.wait()
open(sys.argv[2],'w').write('new search')
time.sleep(100)
'''
            parent=subprocess.Popen([sys.executable,'-c',script,str(destination),str(marker)],stdout=subprocess.PIPE,text=True)
            parent.stdout.readline()
            try:
                request=dict(supervisor=process(parent.pid),executable=sys.executable)
                immutable(out/'request.json',request);drain(root,out,request,seconds=5)
                from test_stopped_unit import Manager
                manager=Manager(root/'runtime',parent)
                with manager.patch():
                    stop_drained(root,out,request)
                resume(out)
                self.assertEqual(parent.returncode,-signal.SIGKILL)
                self.assertFalse(marker.exists())
            finally:
                if parent.poll() is None:parent.kill();parent.wait()
                parent.stdout.close()
