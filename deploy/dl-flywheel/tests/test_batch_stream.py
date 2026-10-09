import copy,json,os,sqlite3,tempfile,threading,unittest
from pathlib import Path
from jope_flywheel.batch_stream import BatchStream,Batch,Waiting,data_run_id,fingerprint,validate_manifest
from jope_flywheel.protocol import canonical,digest


def fixture(n=7,children=1,extra=True):
    rows=[]
    for i in range(n):
        for j in range(children):
            run={'schema':'jope.flywheel-data-run/v1','variant_identity':digest({'v':[i,j]}),'context_id':digest({'c':[i,j]}),
                'dataset_sha256':digest({'input':[i,j]}),'target_sha256':digest({'target':[i,j]}),'split_sha256':'2'*64,'protocol_sha256':'3'*64,'cut_id':'cut'}
            rows.append({'data_run':run,'data_run_id':data_run_id(run),'parent_lineage_sha256':digest({'family':i}),'partition':'train',
                'status':'eligible','reason':None,'payload':{'context_id':run['context_id'],'programs':['4'*64]}})
    if extra:
        for i,status,role in[(n,'eligible','validation'),(n+1,'pending','train'),(n+2,'refused','train')]:
            base=copy.deepcopy(rows[0]);base['parent_lineage_sha256']=digest({'family':i});base['data_run']['variant_identity']=digest({'v':i});base['data_run']['context_id']=digest({'c':i})
            base['data_run_id']=data_run_id(base['data_run']);base.update(status=status,partition=role,reason=None if status=='eligible'else status+' reason');rows.append(base)
    return {'schema':'jope.flywheel-batch-manifest/v1','snapshot_sha256':'a'*64,'role_authority_sha256':'b'*64,'rows':rows,'metadata':{'fixture':True}}

class StreamTests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name).resolve()/'stream.sqlite';self.streams=[]
    def tearDown(self):
        for s in self.streams:
            try:s.close()
            except Exception:pass
        self.tmp.cleanup()
    def stream(self,m=None,size=3,seed=123,path=None):
        m=fixture()if m is None else m;raw=canonical(m);s=BatchStream(path or self.path,raw,digest(m),seed=seed,batch_size=size,initial_checkpoint_sha256='0'*64);self.streams.append(s);return s
    def commit(self,s,b,step=None):
        step=s.summary()['optimizer_steps_reported']+1 if step is None else step
        after=digest({'checkpoint':step})
        kw={'checkpoint_before':b.checkpoint_before,'checkpoint_after':after,'optimizer_step':step,'ordered_data_run_ids':b.ordered_data_run_ids,
            'evidence':{'origin':'native_optimizer','event_sha256':digest({'event':step})}}
        s.commit(b.batch_id,**kw);return after,kw
    def test_exhaustive_epoch_distinct_lineages_validation_pending_refused_absent(self):
        m=fixture(5,3);s=self.stream(m,size=4);checkpoint='0'*64;ids=[]
        while s.summary()['completed_epochs']==0:
            b=s.reserve(checkpoint);self.assertIsInstance(b,Batch)
            self.assertEqual(len({r['parent_lineage_sha256']for r in b.rows}),len(b.rows));self.assertTrue(all(r['partition']=='train'and r['status']=='eligible'for r in b.rows))
            ids+=list(b.ordered_data_run_ids);checkpoint,_=self.commit(s,b)
        expected={r['data_run_id']for r in m['rows']if r['partition']=='train'and r['status']=='eligible'}
        self.assertEqual(set(ids),expected);self.assertEqual(len(ids),len(expected));self.assertEqual(s.summary()['unique_exposed_data_runs'],15)
        self.assertEqual(s.summary()['membership_counts']['train:refused'],1);s.verify()
    def test_all_pending_waits_without_fictional_epoch(self):
        m=fixture(1,extra=False);m['rows'][0].update(status='pending',reason='awaiting exact scores')
        s=self.stream(m);w=s.reserve('0'*64);self.assertIsInstance(w,Waiting);self.assertEqual(w.reason,'no_eligible_training_data_runs');self.assertEqual(s.summary()['epoch'],0)
    def test_reproducible_seed_order_and_changed_seed(self):
        m=fixture(15);a=self.stream(m,path=self.path);b=self.stream(m,path=self.path.parent/'b.sqlite');c=self.stream(m,seed=124,path=self.path.parent/'c.sqlite')
        x=a.reserve('0'*64);y=b.reserve('0'*64);z=c.reserve('0'*64)
        self.assertEqual(x.ordered_data_run_ids,y.ordered_data_run_ids);self.assertNotEqual(x.ordered_data_run_ids,z.ordered_data_run_ids)
    def test_pending_crash_resume_never_silent_redraw(self):
        s=self.stream();b=s.reserve('0'*64);s.close();r=self.stream()
        self.assertEqual(r.pending()['batch_id'],b.batch_id)
        with self.assertRaisesRegex(ValueError,'unresolved'):r.reserve('0'*64)
        after,kw=self.commit(r,b);self.assertIsInstance(r.reserve(after),Batch)
    def test_exact_commit_retry_idempotent_conflict_refused(self):
        s=self.stream();b=s.reserve('0'*64);_,kw=self.commit(s,b);s.commit(b.batch_id,**kw)
        self.assertEqual(s.summary()['optimizer_steps_reported'],1);kw['checkpoint_after']='9'*64
        with self.assertRaisesRegex(ValueError,'retry differs'):s.commit(b.batch_id,**kw)
    def test_wrong_actual_order_rolls_back(self):
        s=self.stream();b=s.reserve('0'*64)
        with self.assertRaisesRegex(ValueError,'actual ordered'):s.commit(b.batch_id,checkpoint_before='0'*64,checkpoint_after='1'*64,optimizer_step=1,
            ordered_data_run_ids=list(reversed(b.ordered_data_run_ids)),evidence={'origin':'native_optimizer','event_sha256':'e'*64})
        self.assertEqual(s.summary()['unique_exposed_data_runs'],0);self.assertEqual(s.pending()['state'],'reserved')
    def test_never_repeat_membership_even_new_epoch_or_order(self):
        s=self.stream(fixture(1,extra=False),size=1);b=s.reserve('0'*64);after,_=self.commit(s,b);w=s.reserve(after)
        self.assertIsInstance(w,Waiting);self.assertEqual(w.remaining,1);self.assertEqual(s.summary()['completed_epochs'],1)
        self.assertEqual(fingerprint(['1'*64,'2'*64]),fingerprint(['2'*64,'1'*64]))
    def test_manifest_checkpoint_seed_resume_guards(self):
        m=fixture();s=self.stream(m);b=s.reserve('0'*64)
        with self.assertRaisesRegex(ValueError,'checkpoint'):s.reserve('1'*64)
        with self.assertRaisesRegex(ValueError,'seed/batch'):self.stream(m,seed=124)
        m['metadata']['changed']=True
        with self.assertRaisesRegex(ValueError,'manifest mismatch'):self.stream(m)
    def test_unknown_update_blocks_and_reconciles_without_second_batch(self):
        s=self.stream();b=s.reserve('0'*64);s.mark_uncertain(b.batch_id,'process died after model step')
        with self.assertRaisesRegex(ValueError,'unresolved'):s.reserve('0'*64)
        self.commit(s,b);self.assertIsNone(s.pending());self.assertEqual(s.verify()['committed_batches'],1)
    def test_new_manifest_preserves_old_rows_and_unique_fingerprints(self):
        m=fixture(1,extra=False);s=self.stream(m,size=1);b=s.reserve('0'*64);after,_=self.commit(s,b);self.assertIsInstance(s.reserve(after),Waiting)
        new=fixture(2,extra=False);s.adopt_manifest(canonical(new),digest(new),checkpoint_sha256=after);n=s.reserve(after)
        self.assertNotEqual(n.fingerprint,b.fingerprint);self.assertNotEqual(n.manifest_sha256,b.manifest_sha256)
        s.verify()
    def test_stale_instance_refuses_after_other_connection_adopts(self):
        m=fixture(2,extra=False);a=self.stream(m);b=self.stream(m);new=fixture(3,extra=False)
        a.adopt_manifest(canonical(new),digest(new),checkpoint_sha256='0'*64)
        with self.assertRaisesRegex(ValueError,'stale planner'):b.reserve('0'*64)
        with self.assertRaisesRegex(ValueError,'stale planner'):b.summary()
        with self.assertRaisesRegex(ValueError,'stale planner'):b.adopt_manifest(canonical(new),digest(new),checkpoint_sha256='0'*64)
        c=self.stream(new);self.assertIsInstance(c.reserve('0'*64),Batch)
    def test_dropped_old_membership_and_role_conflicts_refused(self):
        m=fixture(2);s=self.stream(m);new=fixture(1)
        with self.assertRaisesRegex(ValueError,'retain old'):s.adopt_manifest(canonical(new),digest(new),checkpoint_sha256='0'*64)
        conflict=fixture(2);conflict['rows'][1]['parent_lineage_sha256']=conflict['rows'][0]['parent_lineage_sha256'];conflict['rows'][1]['partition']='validation'
        with self.assertRaisesRegex(ValueError,'role conflict'):validate_manifest(canonical(conflict),digest(conflict))
    def test_tamper_event_exposure_and_missing_epoch_row_detected(self):
        s=self.stream();b=s.reserve('0'*64);self.commit(s,b)
        s.db.execute("UPDATE events SET body='{}'WHERE seq=1")
        with self.assertRaisesRegex(ValueError,'chain tampered'):s.verify()
        t=self.stream(path=self.path.parent/'other.sqlite');b=t.reserve('0'*64);self.commit(t,b)
        t.db.execute('DELETE FROM members WHERE epoch=1 AND run=?',(b.ordered_data_run_ids[0],))
        with self.assertRaisesRegex(ValueError,'membership'):t.verify()
    def test_sql_transaction_commit_fault_leaves_pending_no_exposure(self):
        s=self.stream();b=s.reserve('0'*64);original=s._event
        def fail(kind,body):
            if kind=='batch.committed':raise OSError('simulated crash before SQL COMMIT')
            return original(kind,body)
        s._event=fail
        with self.assertRaises(OSError):self.commit(s,b)
        self.assertEqual(s.summary()['unique_exposed_data_runs'],0);self.assertEqual(s.pending()['state'],'reserved');s._event=original;s.verify()
    def test_two_connections_only_one_reservation(self):
        m=fixture();self.stream(m).close();barrier=threading.Barrier(2);results=[]
        def reserve():
            s=BatchStream(self.path,canonical(m),digest(m),seed=123,batch_size=3,initial_checkpoint_sha256='0'*64)
            try:
                barrier.wait();results.append(('ok',s.reserve('0'*64).batch_id))
            except ValueError as e:results.append(('blocked',str(e)))
            finally:s.close()
        ts=[threading.Thread(target=reserve)for _ in range(2)]
        for t in ts:t.start()
        for t in ts:t.join(10);self.assertFalse(t.is_alive())
        self.assertEqual(sorted(x[0]for x in results),['blocked','ok'])
    def test_no_aggregate_import_claim_and_symlink(self):
        s=self.stream();b=s.reserve('0'*64)
        with self.assertRaisesRegex(ValueError,'native optimizer'):s.commit(b.batch_id,checkpoint_before='0'*64,checkpoint_after='1'*64,optimizer_step=1,
            ordered_data_run_ids=b.ordered_data_run_ids,evidence={'origin':'imported_aggregate','event_sha256':'e'*64})
        link=self.path.parent/'link.sqlite';link.symlink_to(self.path)
        with self.assertRaisesRegex(ValueError,'symlink'):self.stream(path=link)

if __name__=='__main__':unittest.main()
