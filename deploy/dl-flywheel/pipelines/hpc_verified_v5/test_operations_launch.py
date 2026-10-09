"""Local fixture files and mocked command runners only; no systemd/SSH/S3/GP."""
from pathlib import Path
import json
import contextlib
import io
import os
import shutil
import shlex
import sqlite3
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import operations_launch as op


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.counter=0;self.at=op.GLOBAL_DEADLINE-3600
        self.resources=lambda root:{'fixture_only':True,'available_memory_bytes':40*1024**3}
        self.src=self.root/'src';self.src.mkdir()
        self.original=self.root/'original.sqlite';self.continuation=self.root/'continuation.sqlite'
        self.conf={'schema':'fixture-outbox/v1','origin':'fixture'}
        for path in (self.original,self.continuation):
            db=sqlite3.connect(path);db.execute('CREATE TABLE config(body BLOB)')
            db.execute('INSERT INTO config VALUES(?)',(op.canonical(self.conf),))
            db.execute('CREATE TABLE emissions(state TEXT,body BLOB)');db.commit();db.close()
        artifact=self.save('artifact',b'fixture input')
        worker=self.save('worker',b'fixture worker')
        canonicalizer=self.save('canonicalizer',b'fixture canonicalizer')
        build=self.save('build.json',op.canonical({'schema':'jope.dl-s1-control-receipt/v1','source':{'pin':'a'*40},
             'binaries':{'worker_honest':{'sha256':worker['sha256'],'bytes':worker['bytes']},
                         'proposal':{'sha256':canonicalizer['sha256'],'bytes':canonicalizer['bytes']}}}))
        template={key:None for key in op.PACKET_FIELDS}
        template.update(schema='jope.native-original-drain/v4',input_lists=[artifact],source_pins=[artifact],
          native_build_receipt=build,native_canonicalizer=canonicalizer,snapshot=artifact,training_config=artifact,
          source_commit='a'*40,source_build_plan_sha256='b'*64,worker=worker,emitter=artifact,rustc=artifact)
        plan=self.save('build-plan.json',op.canonical({k:template[k] for k in
            ('source_commit','native_build_receipt','worker','emitter','native_canonicalizer','rustc')}))
        template['source_build_plan_sha256']=plan['sha256'];template['source_pins'].append(plan)
        self.template=self.save('template.json',op.canonical(template))
        sources=[{'path':str(path),'kind':'outbox','config':self.conf} for path in (self.original,self.continuation)]
        sources.sort(key=lambda x:x['path'])
        prepared={'schema':'jope.native-v5-prepared/v1','sources':sources,'census':self.save('census.json',b'{}'),
          'scope_sha256':'c'*64,'canonicalizer_sha256':canonicalizer['sha256'],'original_outbox_empty':True,
          'original_outbox':{'path':str(self.original),'config':self.conf},
          'continuation_outbox':{'path':str(self.continuation),'config':self.conf},
          'legacy_closure_refs':[self.save('closure.json',b'{}')],
          'legacy_units':[{'host':'xbabe2','unit':'old-native.service'},{'host':'xbabe3','unit':'old-gp.service'}]}
        self.prepared=self.save('prepared.json',op.canonical(prepared))
        names=['native_dispatch.py','native_build.py','native_global_guard.py','readonly_snapshot.py','neural_outbox.py',
          'native_evaluate.py','native_inputs.py','native_failure.py','verified_cache.py',
          'native_corpus_authority.py','jope_flywheel/__init__.py']
        files=[{'relative_path':name,'ref':self.save('source'+str(i)+'.py',b'# inert fixture\n')}
               for i,name in enumerate(names)]
        files.sort(key=lambda x:x['relative_path'])
        python=self.root/'fixture-python';shutil.copyfile(Path(sys.executable).resolve(),python);python.chmod(0o700)
        self.manifest={'schema':'jope.native-v5-launch-input/v1','prepared':self.prepared,'worker_template':self.template,
          'source_files':files,'root':str(self.root/'new-packet'),'registry_path':str(self.root/'permanent.sqlite'),
          'python':op.reference(python),'unit_prefix':'jope-native-v5-fixture'}
        self.manifest_ref=self.save('manifest.json',op.canonical(self.manifest))

    def save(self,name,raw):
        p=self.src/name;p.write_bytes(raw);return op.reference(p)

    def stage_and_freeze(self):
        stage=op.stage(self.manifest_ref)
        return op.freeze(stage,now=lambda:self.at,resources=self.resources)

    def runner(self,argv,**kwargs):
        self.calls.append(argv)
        if argv[0]=='systemd-run':
            if '--preflight-only' in argv:
                prepared=json.loads(op.read(self.prepared))
                return {'returncode':0,'output':json.dumps({'state':'preflight_pass','global_guard':{
                    'schema':'jope.native-global-execution-guard/v1','sources':prepared['sources'],
                    'registry':{'claims':0,'head_sha256':'0'*64},'historical_census_sha256':prepared['census']['sha256'],
                    'automatic_retry':False}})}
            return {'returncode':0,'output':'Running as unit\n'}
        if argv[-1]=='--property=LoadState':return {'returncode':0,'output':'LoadState=not-found\n'}
        args=shlex.split(argv[-1]) if argv[0]=='ssh' else argv
        return {'returncode':0,'output':json.dumps({'schema':'jope.native-v5-live-unit/v1','host':args[-1],
            'unit':args[-2].removesuffix('.service')+'.service','checked_at_unix':self.at,
            'systemd':{'ActiveState':'inactive','SubState':'dead','MainPID':'0','ControlPID':'0'},
            'cgroups':[{'path':'/sys/fs/cgroup/fixture.service','exists':True,'pids':[],'populated':0}]})}

    def test_dry_plan_has_no_writes_or_commands(self):
        before=set(self.root.iterdir())
        with patch.object(op.subprocess,'Popen',side_effect=AssertionError('no child expected')):
            result=op.dry_plan(self.manifest_ref)
        self.assertEqual(result['workers'],4);self.assertEqual(result['external_calls'],0)
        self.assertEqual(set(self.root.iterdir()),before)

    def test_stage_createonly_and_source_tamper_refused(self):
        stage=op.stage(self.manifest_ref);s=json.loads(op.read(stage));self.assertEqual(len(s['workers']),4)
        with self.assertRaises(FileExistsError):op.stage(self.manifest_ref)
        p=Path(s['workers'][0]['source_pins'][0]['path']);p.write_bytes(b'tampered')
        with self.assertRaises(ValueError):op.freeze(stage,now=lambda:self.at,resources=self.resources)
        self.assertFalse((Path(self.manifest['root'])/'freeze-intent.json').exists())

    def test_freeze_shared_guard_and_fixed_full_window(self):
        stage=op.stage(self.manifest_ref)
        with self.assertRaisesRegex(ValueError,'full fresh'):
            op.freeze(stage,now=lambda:op.GLOBAL_DEADLINE-1799,resources=self.resources)
        ref=op.freeze(stage,now=lambda:self.at,resources=self.resources);plan=json.loads(op.read(ref))
        configs=[json.loads(op.read(w['config'])) for w in plan['workers']]
        self.assertTrue(all(c['global_guard']==configs[0]['global_guard'] for c in configs))
        self.assertTrue(all(c['deadline_unix']==self.at+1800 and c['created_unix']==self.at and
                            c['maximum_emissions']==128 and c['threads']==1 for c in configs))
        self.assertTrue(all(c['outbox_path']==str(self.continuation) for c in configs))
        self.assertEqual(len({c['worker_id'] for c in configs}),4)
        with self.assertRaises(FileExistsError):op.freeze(stage,now=lambda:self.at+1,resources=self.resources)

    def test_resource_gate_and_original_queue_checked_before_freeze(self):
        stage=op.stage(self.manifest_ref)
        with self.assertRaisesRegex(ValueError,'fixture resource hold'):
            op.freeze(stage,now=lambda:self.at,resources=lambda _:op.need(False,'fixture resource hold'))
        db=sqlite3.connect(self.original)
        db.execute('INSERT INTO emissions VALUES(?,?)',('recorded',op.canonical({'emission':{'input_descriptor':{'source_kind':'native_s1_context'}}})))
        db.commit();db.close()
        with self.assertRaisesRegex(ValueError,'no longer empty'):
            op.freeze(stage,now=lambda:self.at,resources=self.resources)
        self.assertFalse((Path(self.manifest['root'])/'freeze-intent.json').exists())

    def test_all_gates_and_preflights_precede_four_native_starts(self):
        plan=self.stage_and_freeze();self.calls=[]
        ref=op.launch(plan,runner=self.runner,resources=self.resources,now=lambda:self.at+1)
        result=json.loads(op.read(ref));self.assertEqual(len(result['receipts']),4)
        self.assertFalse(result['native_completion_verified'])
        pre=[i for i,a in enumerate(self.calls) if '--preflight-only' in a]
        native=[i for i,a in enumerate(self.calls) if a[0]=='systemd-run' and '--preflight-only' not in a]
        self.assertEqual(len(pre),4);self.assertEqual(len(native),4);self.assertLess(max(pre),min(native))
        self.assertTrue(any(a[0]=='ssh' and 'xbabe3' in a for a in self.calls))
        for i in native:
            argv=self.calls[i]
            for value in ('--property=CPUQuota=100%','--property=MemoryMax=1G','--property=MemorySwapMax=0',
                          '--property=TasksMax=32','--property=Restart=no','--setenv=CUDA_VISIBLE_DEVICES='):
                self.assertIn(value,argv)
        calls=len(self.calls)
        with self.assertRaises(FileExistsError):op.launch(plan,runner=self.runner,resources=self.resources,now=lambda:self.at+2)
        self.assertEqual(len(self.calls),calls)

    def test_partial_native_launch_is_durable_and_never_retried(self):
        plan=self.stage_and_freeze();self.calls=[];started=[]
        def runner(argv,**kw):
            if argv[0]=='systemd-run' and '--preflight-only' not in argv:
                started.append(argv)
                if len(started)==2:return {'returncode':1,'output':'fixture uncertain launch'}
            return self.runner(argv,**kw)
        with self.assertRaises(ValueError):op.launch(plan,runner=runner,resources=self.resources,now=lambda:self.at+1)
        root=Path(self.manifest['root'])
        self.assertTrue((root/'launch-intent.json').exists());self.assertTrue((root/'operations-failure.json').exists())
        self.assertEqual(len(started),2)
        with self.assertRaises(FileExistsError):op.launch(plan,runner=runner,resources=self.resources,now=lambda:self.at+2)
        self.assertEqual(len(started),2)

    def test_live_legacy_hold_blocks_even_with_pinned_closed_receipt(self):
        plan=self.stage_and_freeze();self.calls=[]
        def active(argv,**kw):
            result=self.runner(argv,**kw);proof=json.loads(result['output'])
            proof['systemd'].update(ActiveState='active',MainPID='42')
            result['output']=json.dumps(proof);return result
        with self.assertRaisesRegex(ValueError,'physically empty'):
            op.launch(plan,runner=active,resources=self.resources,now=lambda:self.at+1)
        self.assertFalse((Path(self.manifest['root'])/'launch-intent.json').exists())

    def test_input_extra_keys_wrong_build_and_traversal_refused(self):
        for change in ({'extra':True},{'source_files':[{'relative_path':'../escape.py','ref':self.manifest['source_files'][0]['ref']}]}):
            ref=self.save('bad.json',op.canonical({**self.manifest,**change}))
            with self.assertRaises(ValueError):op.dry_plan(ref)
        template=json.loads(op.read(self.template));template['source_build_plan_sha256']='e'*64
        bad=self.save('bad-template.json',op.canonical(template))
        ref=self.save('bad.json',op.canonical({**self.manifest,'worker_template':bad}))
        with self.assertRaisesRegex(ValueError,'exact native build'):op.dry_plan(ref)

    def test_bounded_local_child_output_and_timeout(self):
        with patch.object(op.os,'killpg',wraps=os.killpg) as kill:
            r=op.bounded_run([sys.executable,'-c','print("fixture")'],timeout=2)
            self.assertEqual(kill.call_count,1,'close process group even after parent exits')
        self.assertEqual(r,{'returncode':0,'output':'fixture\n'})
        with self.assertRaisesRegex(ValueError,'output ceiling'):
            op.bounded_run([sys.executable,'-c','print("x"*10000)'],timeout=2,limit=100)
        with self.assertRaisesRegex(ValueError,'deadline'):
            op.bounded_run([sys.executable,'-c','import time;time.sleep(5)'],timeout=0.05)

    def test_resource_gate_preserves_foreign_and_future_trainer_reserves(self):
        with patch.object(op.os,'uname',return_value=SimpleNamespace(nodename='xbabe2')), \
             patch.object(op.os,'cpu_count',return_value=4), \
             patch.object(op.shutil,'disk_usage',return_value=SimpleNamespace(free=40*1024**3)):
            with patch.object(Path,'read_text',return_value='MemAvailable: '+str(35*1024**2)+' kB\n'):
                with self.assertRaisesRegex(ValueError,'36GiB'):op.resource_gate(self.root)
            with patch.object(Path,'read_text',return_value='MemAvailable: '+str(36*1024**2)+' kB\n'):
                self.assertEqual(op.resource_gate(self.root)['available_memory_bytes'],36*1024**3)

    def test_exact_preflight_census_mismatch_prevents_native_launch(self):
        plan=self.stage_and_freeze();self.calls=[]
        def wrong(argv,**kw):
            result=self.runner(argv,**kw)
            if '--preflight-only' in argv:
                proof=json.loads(result['output']);proof['global_guard']['historical_census_sha256']='0'*64
                result['output']=json.dumps(proof)
            return result
        with self.assertRaisesRegex(ValueError,'exact preflight census'):
            op.launch(plan,runner=wrong,resources=self.resources,now=lambda:self.at+1)
        self.assertFalse(any(a[0]=='systemd-run' and '--preflight-only' not in a for a in self.calls))

    def test_physical_unit_literal_checks_real_fixture_cgroup_files(self):
        cgroups=self.root/'cgroups';name='old-native.service';uid=os.getuid()
        directory=cgroups/'user.slice'/('user-'+str(uid)+'.slice')/('user@'+str(uid)+'.service')/'app.slice'/name
        directory.mkdir(parents=True);(directory/'cgroup.procs').write_text('999\n');(directory/'cgroup.events').write_text('populated 1\n')
        fields='Id='+name+'\nLoadState=loaded\nActiveState=inactive\nSubState=dead\nMainPID=0\nControlPID=0\nControlGroup=\n'
        real_path=Path
        def redirect(value,*args):return cgroups if value=='/sys/fs/cgroup' else real_path(value,*args)
        with patch('pathlib.Path',side_effect=redirect),patch('socket.gethostname',return_value='xbabe2'), \
             patch.object(sys,'argv',['-c',name,'xbabe2']), \
             patch('subprocess.run',return_value=SimpleNamespace(returncode=0,stdout=fields.encode(),stderr=b'')):
            with self.assertRaises(AssertionError):exec(op.PHYSICAL_UNIT,{})
            (directory/'cgroup.procs').write_text('');(directory/'cgroup.events').write_text('populated 0\n')
            output=io.StringIO()
            with contextlib.redirect_stdout(output):exec(op.PHYSICAL_UNIT,{})
            proof=json.loads(output.getvalue());self.assertEqual(proof['cgroups'][0]['pids'],[])


if __name__=='__main__':unittest.main()
