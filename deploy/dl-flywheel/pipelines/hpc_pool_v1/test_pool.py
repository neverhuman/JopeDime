"""New orchestration fixtures only. No systemd, SSH, S3 or native GP execution."""
from pathlib import Path
import hashlib
import json
import os
import shlex
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import operations_pool as op


class PoolTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.src=self.root/'sources';self.src.mkdir()
        self.at=op.GLOBAL_DEADLINE-4000;self.calls=[]
        self.original=self.root/'original.sqlite';self.continuation=self.root/'continuation.sqlite'
        self.control={'schema':'fixture-control/v1'}
        for p in (self.original,self.continuation):
            db=sqlite3.connect(p);db.executescript('CREATE TABLE config(body BLOB);CREATE TABLE emissions(state TEXT,body BLOB);')
            db.execute('INSERT INTO config VALUES(?)',(op.canonical(self.control),));db.commit();db.close()
        sources=sorted([{'path':str(p),'kind':'outbox','config':self.control}for p in (self.original,self.continuation)],key=lambda x:x['path'])
        physical={s['path']:{'device':Path(s['path']).stat().st_dev,'inode':Path(s['path']).stat().st_ino}for s in sources}
        artifact=self.save('artifact',b'fixture');worker=self.save('worker',b'worker');canon=self.save('canonicalizer',b'canonicalizer')
        build=self.save('build.json',op.canonical({'schema':'jope.dl-s1-control-receipt/v1','source':{'pin':'a'*40},
            'binaries':{'worker_honest':{'sha256':worker['sha256'],'bytes':worker['bytes']},
                        'proposal':{'sha256':canon['sha256'],'bytes':canon['bytes']}}}))
        self.prepared_value={'schema':'jope.native-v5-prepared/v1','sources':sources,
            'census':self.save('census.json',op.canonical({'source_identities':physical})),
            'scope_sha256':'c'*64,'canonicalizer_sha256':canon['sha256'],'original_outbox_empty':True,
            'original_outbox':{'path':str(self.original),'config':self.control},
            'continuation_outbox':{'path':str(self.continuation),'config':self.control},
            'legacy_closure_refs':[self.save('closure.json',b'{}')],
            'legacy_units':[{'host':'xbabe2','unit':'closed-native'},{'host':'xbabe3','unit':'closed-variant'}]}
        self.prepared=self.save('prepared.json',op.canonical(self.prepared_value))
        self.registry=self.root/'permanent.sqlite'
        self.patch_registry=patch.object(op,'PERMANENT_REGISTRY',str(self.registry));self.patch_registry.start();self.addCleanup(self.patch_registry.stop)
        python=self.root/'python3.12';shutil.copyfile(Path(sys.executable).resolve(),python);python.chmod(0o700)
        names=['native_dispatch.py','native_build.py','native_global_guard.py','readonly_snapshot.py','neural_outbox.py',
               'native_evaluate.py','native_inputs.py','native_failure.py','verified_cache.py','native_corpus_authority.py','jope_flywheel/__init__.py']
        names += ['support'+str(i)+'.py'for i in range(23)]
        files=sorted([{'relative_path':n,'ref':self.save('source'+str(i)+'.py',b'# inert source fixture\n')}for i,n in enumerate(names)],key=lambda x:x['relative_path'])
        parent_root=self.root/'parent';parent_root.mkdir()
        parent={'schema':'jope.native-v5-launch-input/v1','prepared':self.prepared,'worker_template':artifact,
            'source_files':files,'root':str(parent_root),'registry_path':str(self.registry),'python':op.reference(python),'unit_prefix':'jope-native-v5-parent'}
        parent_ref=self.save('parent.json',op.canonical(parent))
        m={'schema':'jope.native-v5-pool-input/v1','prepared':self.prepared,'worker_template':None,'source_files':files,
            'root':str(self.root/'new-pool'),'registry_path':str(self.registry),'python':parent['python'],
            'unit_prefix':'jope-native-v5-added','worker_count':12,'parent_manifest':parent_ref,'parent_plan':None,
            'other_allocations':[{'unit':'owned-learner','kind':'learner','memory_cap_bytes':8*op.GIB,
                'cpu_quota_percent':800,'expected_argv':['/usr/bin/python3','learner.py']}],'registry_identity':None}
        guard=op.parent_guard(m,self.prepared_value)
        template={k:None for k in op.PACKET_FIELDS}
        template.update(schema='jope.native-original-successor/v5',input_lists=[artifact],source_pins=[artifact],
            native_build_receipt=build,native_canonicalizer=canon,snapshot=artifact,training_config=artifact,
            source_commit='a'*40,source_build_plan_sha256='b'*64,worker=worker,emitter=artifact,rustc=artifact,
            global_guard=guard,created_unix=self.at-10,deadline_unix=self.at+1790,global_deadline_unix=op.GLOBAL_DEADLINE,
            outbox_path=str(self.continuation),outbox_config=self.control,threads=1,maximum_emissions=128)
        bp=self.save('build-plan.json',op.canonical({k:template[k]for k in('source_commit','native_build_receipt','worker','emitter','native_canonicalizer','rustc')}))
        template['source_build_plan_sha256']=bp['sha256'];template['source_pins'] += [bp]+[x['ref']for x in files]
        workers=[]
        for i in range(4):
            d=parent_root/('worker'+str(i));d.mkdir();c=dict(template,out=str(d),worker_id=parent['unit_prefix']+'-worker'+str(i))
            (d/'config.json').write_bytes(op.canonical(c));workers.append({'index':i,'unit':c['worker_id'],'config':op.reference(d/'config.json')})
        (parent_root/'stage.json').write_bytes(op.canonical({'manifest':parent_ref}))
        plan={'schema':'jope.native-v5-launch-plan/v1','stage':op.reference(parent_root/'stage.json'),
              'created_unix':self.at-10,'deadline_unix':self.at+1790,'workers':workers}
        (parent_root/'plan.json').write_bytes(op.canonical(plan));m['parent_plan']=op.reference(parent_root/'plan.json');m['worker_template']=workers[0]['config']
        identity={'schema':guard['schema'],'scope_sha256':guard['scope_sha256'],'canonicalizer_sha256':canon['sha256'],
            'sources':sources,'physical_sources':physical,'census_sha256':guard['census']['sha256'],'global_deadline_unix':op.GLOBAL_DEADLINE}
        db=sqlite3.connect(self.registry);db.executescript('CREATE TABLE config(body BLOB);CREATE TABLE claims(key TEXT,state TEXT);CREATE TABLE events(seq INTEGER,body BLOB);')
        db.execute('INSERT INTO config VALUES(?)',(op.canonical(identity),));db.execute('INSERT INTO claims VALUES(?,?)',('unknown-canonical-key','unknown'))
        db.execute('INSERT INTO events VALUES(?,?)',(1,b'fixture historical unknown'));db.commit();db.close()
        st=self.registry.stat();m['registry_identity']={'device':st.st_dev,'inode':st.st_ino,'config_sha256':op.sha(op.canonical(identity))}
        self.m=m;self.mref=self.save('manifest.json',op.canonical(m));self.resources=lambda m,r:{'fixture_resources':True,'observed_at_unix':self.at+1}

    def save(self,name,raw):
        p=self.src/name;p.write_bytes(raw);return op.reference(p)

    def mcopy(self,**changes):
        return self.save('modified.json',op.canonical(dict(self.m,**changes)))

    def staged(self):return op.stage(self.mref)
    def frozen(self):return op.freeze(self.staged(),now=lambda:self.at,resources=self.resources)

    def runner(self,argv,**kw):
        self.calls.append(argv)
        if argv[0]==self.m['python']['path']:
            return {'returncode':0,'output':json.dumps({'schema':'jope.native-pool-python/v1','version':[3,12,3],**self.m['python']})}
        if argv[0]=='systemd-run':
            if '--preflight-only'in argv:
                return {'returncode':0,'output':json.dumps({'state':'preflight_pass','global_guard':{
                    'schema':'jope.native-global-execution-guard/v1','sources':self.prepared_value['sources'],
                    'registry':{'claims':1,'head_sha256':'d'*64},'historical_census_sha256':self.prepared_value['census']['sha256'],'automatic_retry':False}})}
            return {'returncode':0,'output':'Started owned fixture\n'}
        if argv[-1]=='--property=LoadState':return {'returncode':0,'output':'LoadState=not-found\n'}
        if op.ALLOCATION_PROBE in argv:
            a=next(x for x in op.allocations(self.m)if x['unit']==argv[-1]);return {'returncode':0,'output':json.dumps({
                'unit':a['unit']+'.service','active':True,'memory_cap_bytes':a['memory_cap_bytes'],
                'memory_current_bytes':a['memory_cap_bytes']//2,'swap_cap_bytes':0,'cpu_quota':a['cpu_quota_percent']*1000,
                'cpu_period':100000,'argv':a['expected_argv']})}
        a=shlex.split(argv[-1])if argv[0]=='ssh'else argv
        return {'returncode':0,'output':json.dumps({'schema':'jope.native-v5-live-unit/v1','host':a[-1],'unit':a[-2]+'.service',
            'checked_at_unix':self.at,'systemd':{'ActiveState':'inactive','SubState':'dead','MainPID':'0','ControlPID':'0'},
            'cgroups':[{'path':'/fixture','exists':False}]})}

    def test_dry_is_readonly_and_count_is_explicit_8_12_or_24(self):
        before={p:op.sha(p.read_bytes())for p in (self.registry,self.original,self.continuation)}
        with patch.object(op.subprocess,'Popen',side_effect=AssertionError('no subprocess in dry')):
            self.assertEqual(op.dry_plan(self.mref)['workers'],12)
            self.assertEqual(op.dry_plan(self.mcopy(worker_count=8))['workers'],8)
            self.assertEqual(op.dry_plan(self.mcopy(worker_count=24))['workers'],24)
            for count in (True,0,4,13,25,128):
                with self.assertRaisesRegex(ValueError,'8, 12 or 24'):op.dry_plan(self.mcopy(worker_count=count))
        self.assertEqual(before,{p:op.sha(p.read_bytes())for p in before})

    def test_same_registry_inode_and_scope_required(self):
        db=sqlite3.connect(self.registry);db.execute('UPDATE config SET body=?',(b'{}',));db.commit();db.close()
        with self.assertRaisesRegex(ValueError,'config bytes'):op.dry_plan(self.mref)
        self.registry.unlink()
        with self.assertRaisesRegex(ValueError,'existing permanent'):op.dry_plan(self.mref)
        self.assertFalse(self.registry.exists())

    def test_copy_of_registry_cannot_replace_physical_authority(self):
        p=self.root/'copy.sqlite';shutil.copyfile(self.registry,p);os.replace(p,self.registry)
        with self.assertRaisesRegex(ValueError,'same permanent registry inode'):op.dry_plan(self.mref)

    def test_parent_source_and_guard_changes_rejected(self):
        files=json.loads(json.dumps(self.m['source_files']));files[0]['ref']=self.save('replacement.py',b'# changed\n')
        with self.assertRaisesRegex(ValueError,'34 parent'):op.dry_plan(self.mcopy(source_files=files))
        parent=json.loads(op.read(self.m['parent_plan']));c=json.loads(op.read(parent['workers'][1]['config']));c['global_guard']['scope_sha256']='0'*64
        p=Path(parent['workers'][1]['config']['path']);p.write_bytes(op.canonical(c));parent['workers'][1]['config']=op.reference(p)
        p=Path(self.m['parent_plan']['path']);p.write_bytes(op.canonical(parent))
        with self.assertRaisesRegex(ValueError,'same parent global guard'):op.dry_plan(self.mcopy(parent_plan=op.reference(p)))

    def test_stage_and_freeze_do_not_reset_unknowns_or_parent_clock(self):
        before=self.registry.read_bytes();parent=op.read(self.m['parent_plan']);plan=json.loads(op.read(self.frozen()))
        self.assertEqual(len(plan['workers']),12);self.assertEqual(self.registry.read_bytes(),before);self.assertEqual(op.read(self.m['parent_plan']),parent)
        configs=[json.loads(op.read(w['config']))for w in plan['workers']]
        self.assertEqual(len({c['worker_id']for c in configs}),12)
        self.assertTrue(all(c['global_guard']==configs[0]['global_guard']and c['maximum_emissions']==128 for c in configs))
        self.assertEqual(plan['deadline_unix'],self.at+1800)
        db=sqlite3.connect(self.registry);self.assertEqual(db.execute('SELECT state FROM claims').fetchone()[0],'unknown');db.close()
        with self.assertRaises(FileExistsError):op.stage(self.mref)

    def test_original_deadline_refuses_fresh_window(self):
        stage=self.staged()
        with self.assertRaisesRegex(ValueError,'full fresh'):op.freeze(stage,now=lambda:op.GLOBAL_DEADLINE-1799,resources=self.resources)
        self.assertFalse((Path(self.m['root'])/'freeze-intent.json').exists())

    def test_resource_gate_conservatively_reserves_parent_full_caps_and_learner(self):
        Path(self.m['root']).mkdir()
        with patch.object(op.os,'uname',return_value=SimpleNamespace(nodename='xbabe2')),patch.object(op.os,'cpu_count',return_value=128), \
             patch.object(op.shutil,'disk_usage',return_value=SimpleNamespace(free=100*op.GIB)):
            # Four1GiB parents +8GiB learner +12new +24foreign=48GiB, without racy current-memory credit.
            with patch.object(Path,'read_text',return_value='MemAvailable: '+str(48*1024**2)+' kB\n'):
                r=op.resource_gate(self.m,self.runner);self.assertEqual(r['required_available_bytes'],48*op.GIB)
            with patch.object(Path,'read_text',return_value='MemAvailable: '+str(47*1024**2)+' kB\n'):
                with self.assertRaisesRegex(ValueError,'resource hold'):op.resource_gate(self.m,self.runner)

    def test_owned_command_drift_cannot_reduce_reserved_memory(self):
        with patch.object(op.os,'uname',return_value=SimpleNamespace(nodename='xbabe2')):
            def wrong(argv,**kw):
                r=self.runner(argv,**kw);j=json.loads(r['output']);j['argv']=['wrong'];r['output']=json.dumps(j);return r
            with self.assertRaisesRegex(ValueError,'active owned command'):op.resource_gate(self.m,wrong)

    def test_small_active_owned_allocation_exact_caps_and_full_reservation(self):
        self.m['other_allocations'].append({'unit':'owned-publisher42','kind':'other_owned',
            'memory_cap_bytes':128*1024**2,'cpu_quota_percent':25,
            'expected_argv':['/usr/bin/python3','publisher.py']})
        mref=self.save('small.json',op.canonical(self.m));self.assertEqual(op.dry_plan(mref)['workers'],12)
        Path(self.m['root']).mkdir()
        with patch.object(op.os,'uname',return_value=SimpleNamespace(nodename='xbabe2')),patch.object(op.os,'cpu_count',return_value=128), \
             patch.object(op.shutil,'disk_usage',return_value=SimpleNamespace(free=100*op.GIB)):
            with patch.object(Path,'read_text',return_value='MemAvailable: '+str(48*1024**2+128*1024)+' kB\n'):
                r=op.resource_gate(self.m,self.runner);self.assertEqual(r['required_available_bytes'],48*op.GIB+128*1024**2)
                proof=r['owned_units'][-1];self.assertEqual(proof['allocation']['cpu_quota_percent'],25)
                self.assertEqual(proof['physical']['memory_cap_bytes'],128*1024**2)
            with patch.object(Path,'read_text',return_value='MemAvailable: '+str(48*1024**2)+' kB\n'):
                with self.assertRaisesRegex(ValueError,'resource hold'):op.resource_gate(self.m,self.runner)
            def understated_cpu(argv,**kw):
                result=self.runner(argv,**kw)
                if argv[-1]=='owned-publisher42':
                    body=json.loads(result['output']);body['cpu_quota']=12500;result['output']=json.dumps(body)
                return result
            with self.assertRaisesRegex(ValueError,'exact active owned command'):op.resource_gate(self.m,understated_cpu)
        for field,bad in [('memory_cap_bytes',31*1024**2),('cpu_quota_percent',0)]:
            changed=json.loads(json.dumps(self.m));changed['other_allocations'][-1][field]=bad
            with self.assertRaisesRegex(ValueError,'closed owned resource'):op.dry_plan(self.save('bad-small.json',op.canonical(changed)))
        changed=json.loads(json.dumps(self.m));changed['other_allocations'][0]['cpu_quota_percent']=400
        with self.assertRaisesRegex(ValueError,'one protected learner'):op.dry_plan(self.save('bad-learner.json',op.canonical(changed)))

    def test_all_preflights_and_python312_precede_native_starts(self):
        p=self.frozen();r=json.loads(op.read(op.launch(p,runner=self.runner,resources=self.resources,now=lambda:self.at+1)))
        self.assertEqual(len(r['receipts']),12)
        pre=[i for i,a in enumerate(self.calls)if'--preflight-only'in a];run=[i for i,a in enumerate(self.calls)if a[0]=='systemd-run'and'--preflight-only'not in a]
        self.assertEqual(len(pre),12);self.assertEqual(len(run),12);self.assertLess(max(pre),min(run))
        self.assertTrue(any(a[0]==self.m['python']['path']for a in self.calls[:min(pre)]))
        for i in run:self.assertIn('--property=MemoryMax=1G',self.calls[i]);self.assertIn('--property=CPUQuota=100%',self.calls[i])
        count=len(self.calls)
        with self.assertRaises(FileExistsError):op.launch(p,runner=self.runner,resources=self.resources,now=lambda:self.at+2)
        self.assertEqual(len(self.calls),count)

    def test_wrong_physical_python_fails_before_native_preflight(self):
        p=self.frozen()
        def wrong(argv,**kw):
            r=self.runner(argv,**kw)
            if argv[0]==self.m['python']['path']:
                j=json.loads(r['output']);j['version']=[3,10,12];r['output']=json.dumps(j)
            return r
        with self.assertRaisesRegex(ValueError,'Python 3.12'):op.launch(p,runner=wrong,resources=self.resources,now=lambda:self.at+1)
        self.assertFalse(any(a[0]=='systemd-run'for a in self.calls))

    def test_preflight_registry_census_mismatch_blocks_all_starts(self):
        p=self.frozen()
        def wrong(argv,**kw):
            r=self.runner(argv,**kw)
            if'--preflight-only'in argv:
                j=json.loads(r['output']);j['global_guard']['historical_census_sha256']='f'*64;r['output']=json.dumps(j)
            return r
        with self.assertRaisesRegex(ValueError,'exact preflight census'):op.launch(p,runner=wrong,resources=self.resources,now=lambda:self.at+1)
        self.assertFalse(any(a[0]=='systemd-run'and'--preflight-only'not in a for a in self.calls))

    def test_partial_launch_never_retries_or_resets(self):
        p=self.frozen();started=[]
        def fail(argv,**kw):
            if argv[0]=='systemd-run'and'--preflight-only'not in argv:
                started.append(argv)
                if len(started)==3:return {'returncode':1,'output':'uncertain dispatch'}
            return self.runner(argv,**kw)
        with self.assertRaises(ValueError):op.launch(p,runner=fail,resources=self.resources,now=lambda:self.at+1)
        self.assertEqual(len(started),3);self.assertTrue((Path(self.m['root'])/'operations-failure.json').exists())
        with self.assertRaises(FileExistsError):op.launch(p,runner=fail,resources=self.resources,now=lambda:self.at+2)
        self.assertEqual(len(started),3)

    def test_24_workers_parallel_preflight_is_bounded_and_joined(self):
        self.m['worker_count']=24;self.mref=self.save('manifest24.json',op.canonical(self.m))
        p=self.frozen();lock=threading.Lock();state={'running':0,'peak':0,'finished':0};barrier=threading.Barrier(4)
        def observed(argv,**kw):
            if '--preflight-only'in argv:
                with lock:state['running']+=1;state['peak']=max(state['peak'],state['running'])
                barrier.wait(timeout=2);time.sleep(.01)
                result=self.runner(argv,**kw)
                with lock:state['running']-=1;state['finished']+=1
                return result
            if argv[0]=='systemd-run':
                self.assertEqual(state['running'],0);self.assertEqual(state['finished'],24)
            return self.runner(argv,**kw)
        result=json.loads(op.read(op.launch(p,runner=observed,resources=self.resources,now=lambda:self.at+1)))
        self.assertEqual(len(result['receipts']),24);self.assertEqual(state['peak'],4)

    def test_explicit8_launch_has_exact8_and_never_downsizes_other_counts(self):
        self.m['worker_count']=8;self.mref=self.save('manifest8.json',op.canonical(self.m))
        p=self.frozen();plan=json.loads(op.read(p));self.assertEqual(len(plan['workers']),8)
        result=json.loads(op.read(op.launch(p,runner=self.runner,resources=self.resources,now=lambda:self.at+1)))
        starts=[a for a in self.calls if a[0]=='systemd-run'and'--preflight-only'not in a]
        self.assertEqual(len(starts),8);self.assertEqual(len(result['receipts']),8)
        self.assertEqual({w['index']for w in plan['workers']},set(range(8)))

    def test_inactive_learner_keeps_full_reserve_and_cpu_disk_hold(self):
        Path(self.m['root']).mkdir()
        def inactive(argv,**kw):
            if op.ALLOCATION_PROBE in argv:return {'returncode':0,'output':json.dumps({'unit':argv[-1]+'.service','active':False})}
            return self.runner(argv,**kw)
        with patch.object(op.os,'uname',return_value=SimpleNamespace(nodename='xbabe2')), \
             patch.object(Path,'read_text',return_value='MemAvailable: '+str(100*1024**2)+' kB\n'), \
             patch.object(op.shutil,'disk_usage',return_value=SimpleNamespace(free=100*op.GIB)):
            with patch.object(op.os,'cpu_count',return_value=128):
                r=op.resource_gate(self.m,inactive);self.assertEqual(r['required_available_bytes'],44*op.GIB)
                eight=dict(self.m,worker_count=8)
                with patch.object(Path,'read_text',return_value='MemAvailable: '+str(40*1024**2)+' kB\n'):
                    r=op.resource_gate(eight,inactive);self.assertEqual(r['required_available_bytes'],40*op.GIB)
                with patch.object(Path,'read_text',return_value='MemAvailable: '+str(39*1024**2)+' kB\n'):
                    with self.assertRaisesRegex(ValueError,'resource hold'):op.resource_gate(eight,inactive)
            with patch.object(op.os,'cpu_count',return_value=12):
                with self.assertRaisesRegex(ValueError,'resource hold'):op.resource_gate(self.m,inactive)

    def test_python_probe_checks_actual_binary_identity_not_version_alone(self):
        def wrong(argv,**kw):
            r=self.runner(argv,**kw);p=json.loads(r['output']);p['sha256']='0'*64;r['output']=json.dumps(p);return r
        with self.assertRaisesRegex(ValueError,'Python 3.12'):op.python_probe(self.m['python'],wrong)

    def test_registry_replacement_during_preflight_prevents_every_native_start(self):
        p=self.frozen();lock=threading.Lock();replaced=[]
        def replace(argv,**kw):
            result=self.runner(argv,**kw)
            if '--preflight-only'in argv:
                with lock:
                    if not replaced:
                        copy=self.root/'other-registry';shutil.copyfile(self.registry,copy);os.replace(copy,self.registry);replaced.append(True)
            return result
        with self.assertRaisesRegex(ValueError,'same permanent registry inode'):
            op.launch(p,runner=replace,resources=self.resources,now=lambda:self.at+1)
        self.assertFalse(any(a[0]=='systemd-run'and'--preflight-only'not in a for a in self.calls))

    def test_registry_replaced_after_first_start_prevents_next_start(self):
        p=self.frozen();started=[]
        def replace(argv,**kw):
            result=self.runner(argv,**kw)
            if argv[0]=='systemd-run'and'--preflight-only'not in argv:
                started.append(argv)
                if len(started)==1:
                    copy=self.root/'other-registry';shutil.copyfile(self.registry,copy);os.replace(copy,self.registry)
            return result
        with self.assertRaisesRegex(ValueError,'same permanent registry inode'):
            op.launch(p,runner=replace,resources=self.resources,now=lambda:self.at+1)
        self.assertEqual(len(started),1)

    def test_stale_resource_probe_blocks_preflight(self):
        p=self.frozen()
        with self.assertRaisesRegex(ValueError,'within five seconds'):
            op.launch(p,runner=self.runner,resources=lambda m,r:{'observed_at_unix':self.at-10},now=lambda:self.at+1)
        self.assertFalse(any(a[0]=='systemd-run'for a in self.calls))

    def test_resource_observation_occurs_after_slow_closure_and_each_start(self):
        p=self.frozen();events=[]
        def runner(argv,**kw):
            if argv[0]=='ssh':events.append('closure')
            if argv[-1]=='--property=LoadState':events.append('unitcheck')
            if argv[0]=='systemd-run'and'--preflight-only'not in argv:events.append('start')
            return self.runner(argv,**kw)
        def resources(m,r):events.append('resources');return {'observed_at_unix':self.at+1}
        op.launch(p,runner=runner,resources=resources,now=lambda:self.at+1)
        for i,e in enumerate(events):
            if e=='start':self.assertEqual(events[i-1],'resources')
        self.assertGreater(events.index('resources'),events.index('closure'))
        self.assertGreater(events.index('resources'),max(i for i,e in enumerate(events)if e=='unitcheck'))


if __name__=='__main__':unittest.main()
