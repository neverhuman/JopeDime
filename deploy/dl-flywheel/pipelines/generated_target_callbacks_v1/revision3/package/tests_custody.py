"""New durable pre-work custody fixtures; no live services or datasets."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from . import callbacks as C
from .durable import EventSink
from .readonly_io import Window,ExactReader,NativeCanonical,canonical,sha

class CustodyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
    def sink(self):return EventSink(self.root/'output','a'*64)
    def events(self):
        p=self.root/'output/events.jsonl';raw=p.read_bytes();self.assertTrue(raw.endswith(b'\n'))
        docs=[json.loads(x)for x in raw.splitlines()];previous='0'*64
        for i,d in enumerate(docs,1):
            self.assertEqual(d['seq'],i);self.assertEqual(d['previous_sha256'],previous);previous=sha(canonical(d))
        return docs
    def test_real_configure_failure_has_durable_attempt_and_final_receipt(self):
        raw=canonical({'output_root':str(self.root/'output'),'schema':'bad'})
        with patch.multiple(C,_JOURNAL=None,_CONFIGURED=False,_STATE=None,_WINDOW=None,QUALIFICATION_PREFIX=str(self.root)):
            C.start_evidence(raw,sha(raw))
            try:C.configure(raw,sha(raw))
            except ValueError as error:result=C.failure_custody(error)
            else:self.fail('bad config accepted')
        self.assertEqual(result['state'],'failed');self.assertIsNone(result['timing']);self.assertFalse(result['optimizer_admission'])
        self.assertEqual([e['kind']for e in self.events()],['configuration.started','configure.source_and_runtime_checks.attempt','qualification.failed'])
        self.assertTrue((self.root/'output/result.json').is_file())
    def test_get_attempt_is_fsynced_before_transport_failure(self):
        sink=self.sink();window=Window(time.time()+10,1,10,sink);outer=self
        class Transport:
            def get_object(self,**unused):
                self.events=outer.events();outer.assertEqual(self.events[-1]['kind'],'s3.GET.attempt')
                raise OSError('controlled transport failure')
        ref={'bucket':'veox-jopedime-learning-use2','key':'experiments/gp-policy-dl/v2-trainonly/program-results/flywheel/v1/a/input','version_id':'v1','sha256':sha(b'x'),'bytes':1}
        with self.assertRaises(OSError):ExactReader(Transport(),window).get_version(ref)
        self.assertEqual(window.gets,1);self.assertEqual(window.bytes,0)
        sink.finish({'state':'failed','timing':window.counters()})
        self.assertEqual(self.events()[-1]['payload']['gets'],1)
    def test_partial_read_counts_survive_known_failure(self):
        sink=self.sink();window=Window(time.time()+10,1,10,sink)
        class Body:
            n=0
            def read(self,size):
                self.n+=1
                if self.n==1:return b'x'
                raise OSError('controlled interrupted stream')
            def close(self):self.closed=True
        body=Body()
        class Transport:
            def get_object(self,**unused):return {'VersionId':'v1','ContentLength':2,'Body':body}
        ref={'bucket':'veox-jopedime-learning-use2','key':'experiments/gp-policy-dl/v2-trainonly/program-results/flywheel/v1/a/input','version_id':'v1','sha256':sha(b'xy'),'bytes':2}
        with self.assertRaises(OSError):ExactReader(Transport(),window).get_version(ref)
        self.assertTrue(body.closed);self.assertEqual(window.bytes,1)
        sink.finish({'state':'failed'});self.assertEqual(self.events()[-1]['payload']['actual_bytes'],1)
    def test_native_attempt_and_timing_are_durable_on_nonzero_exit(self):
        sink=self.sink();native=self.root/'native';native.write_text('#!'+sys.executable+'\nraise SystemExit(9)\n');native.chmod(0o700)
        window=Window(time.time()+10,1,10,sink)
        tool=NativeCanonical({'path':str(native),'sha256':sha(native.read_bytes())},window)
        with self.assertRaises(ValueError):tool.canonical_json({'x':1})
        sink.finish({'state':'failed'});events=self.events();self.assertEqual(events[1]['kind'],'native.canonicalization.attempt')
        self.assertIn('stdin_sha256',events[1]['payload']);self.assertNotIn('stdin',events[1]['payload'])
        timed=events[-1]['payload']['timing'];self.assertGreaterEqual(timed['end_monotonic_ns'],timed['start_monotonic_ns'])
        for key in('cpu_user_seconds','cpu_system_seconds','child_user_seconds','child_system_seconds'):self.assertGreaterEqual(timed[key],0)
    def test_hard_exit_retains_attempt_without_invented_completion(self):
        pid=os.fork()
        if pid==0:
            sink=EventSink(self.root/'output','a'*64);sink.emit('s3.GET.attempt',{'gets':1,'actual_bytes':0});os._exit(137)
        _,status=os.waitpid(pid,0);self.assertEqual(os.WEXITSTATUS(status),137)
        self.assertEqual([e['kind']for e in self.events()],['configuration.started','s3.GET.attempt'])
        self.assertFalse((self.root/'output/result.json').exists())
    def test_existing_output_cannot_be_overwritten_or_resumed(self):
        sink=self.sink();before=(self.root/'output/events.jsonl').read_bytes()
        with self.assertRaises(ValueError):EventSink(self.root/'output','a'*64)
        self.assertEqual((self.root/'output/events.jsonl').read_bytes(),before);sink.finish({'state':'closed'})
    def test_event_bound_failure_retains_prior_hash_chain(self):
        sink=self.sink();before=(self.root/'output/events.jsonl').read_bytes()
        with self.assertRaises(ValueError):sink.emit('too-large',{'data':'x'*17000})
        self.assertEqual((self.root/'output/events.jsonl').read_bytes(),before)
        result=sink.finish({'state':'failed'});self.assertEqual(result['durable_event_head']['events'],1);self.events()
