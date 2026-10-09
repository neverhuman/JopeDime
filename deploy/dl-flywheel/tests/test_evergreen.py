import gzip,json,struct,tempfile,unittest
from pathlib import Path
import numpy as np
from jope_flywheel.augmentation import array_sha256
from jope_flywheel.protocol import partition
from jope_flywheel.evergreen import byte_sha,encode,decode,source,generate,recipe,Expansion

class EvergreenTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve()
        self.x=np.random.default_rng(7).normal(size=(60,20));self.y=np.arange(60,dtype='<f8')*.1
        self.folds=np.array([0]*30+[1]*30,dtype=np.uint8);raw=encode(self.x,self.y,self.folds)
        context=self.root/'context.json';context.write_bytes(b'{}');z=self.root/'encoder.jtt1.gz';z.write_bytes(gzip.compress(raw,mtime=0))
        self.parent={'schema':'jope.flywheel-prepared-parent/v1','parent_admission':'prepared-research-transforms-only',
          'rights':'research','raw_units':False,'prepared':True,'role_authority':{'eligible_target_all_views':True,'excluded_context_or_view':False},
          'parent_lineage_sha256':'a'*64,'parent_context_sha256':'b'*64,'split_namespace_sha256':'c'*64,
          'partition':partition('a'*64,'c'*64),'allowed_transforms':{'feature_gaussian_noise':False,'resample_rows':False,'feature_drop_max_fraction':.1,'nuisance_max_fraction':.1,'nuisance_max_columns':8,'max_output_features':512,'max_train_rows':1999,'numeric_target_noise':True},
          'context':{'path':str(context),'sha256':byte_sha(b'{}'),'bytes':2},
          'encoder':{'path':str(z),'sha256':byte_sha(z.read_bytes()),'bytes':z.stat().st_size},'raw_payload_sha256':byte_sha(raw),
          'features':20,'rows':60,'feature_matrix_sha256':array_sha256(self.x),'target_sha256':array_sha256(self.y),
          'inner_labels_sha256':byte_sha(self.folds.tobytes()),'parent_cut_id':'cut-parent'}
    def tearDown(self):self.tmp.cleanup()
    def test_source_hash_and_outer_test_refusal(self):
        x,y,f=source(self.parent);np.testing.assert_array_equal(x,self.x)
        raw=encode(x,y,f);mut=raw[:8]+struct.pack('<I',1)+raw[12:]
        with self.assertRaises(ValueError):decode(mut)
        Path(self.parent['encoder']['path']).write_bytes(b'changed')
        with self.assertRaises(ValueError):source(self.parent)
    def test_deterministic_distinct_variants_full_rows_and_lineage(self):
        a=generate(self.parent,0,self.x,self.y,self.folds);b=generate(self.parent,0,self.x,self.y,self.folds)
        self.assertEqual(a,b);self.assertNotEqual(a[0],generate(self.parent,1,self.x,self.y,self.folds)[0])
        x,y,f=decode(a[1]);np.testing.assert_array_equal(y,self.y);np.testing.assert_array_equal(f,self.folds)
        self.assertEqual(a[2]['partition'],self.parent['partition']);self.assertFalse(a[2]['teaching_eligible'])
        self.assertFalse(a[2]['inherited_gp_rewards']);self.assertEqual(a[2]['parent_record_sha256'],b[2]['parent_record_sha256'])
    def test_noise_fit_never_consumes_validation_targets(self):
        a=generate(self.parent,3,self.x,self.y,self.folds)[2]
        changed=self.y.copy();changed[30:]+=1000;p={**self.parent,'target_sha256':array_sha256(changed)}
        # Target identity changes RNG identity as intended; fit scale stays strictly fit-derived.
        b=generate(p,3,self.x,changed,self.folds)[2]
        self.assertEqual(a['target_noise_fit']['state']['train_std'],b['target_noise_fit']['state']['train_std'])
        self.assertNotEqual(a['target_sha256'],b['target_sha256']);self.assertEqual(a['recipe']['target_noise_fraction'],.05)
    def test_content_provenance_and_fresh_labels(self):
        _,raw,m=generate(self.parent,1,self.x,self.y,self.folds);_,yy,_=decode(raw)
        self.assertNotEqual(array_sha256(yy),self.parent['target_sha256']);self.assertEqual(m['target_sha256'],array_sha256(yy))
        self.assertEqual(m['content_sha256'],byte_sha(raw));self.assertTrue(m['fresh_gp_evaluations_required'])
        self.assertTrue(m['fresh_encodings_required']);self.assertEqual(m['column_mapping'][0]['kind'],'prepared_numeric_unknown_original_type')
    def test_small_feature_fallback_and_native_width_limit(self):
        self.assertEqual(recipe({**self.parent,'features':1},0)['target_noise_fraction'],.01)
        for i in range(16):
            r=recipe({**self.parent,'features':512},i)
            self.assertLessEqual(512-r['drop_count']+r['nuisance_count'],512)
            self.assertLessEqual(r['target_noise_fraction'],.05)
    def test_parent_input_drift_refused(self):
        x=self.x.copy();x[0,0]+=1
        with self.assertRaises(ValueError):generate(self.parent,0,x,self.y,self.folds)
    def test_balanced_schedule_backpressure_and_resume(self):
        p={**self.parent,'parent_lineage_sha256':'d'*64};p['partition']=partition('d'*64,p['split_namespace_sha256'])
        e=Expansion(self.root/'out',[self.parent,p],target=100,backlog=2,config_sha='e'*64)
        try:
            a=e.reserve();b=e.reserve();self.assertNotEqual(a[1]['parent_lineage_sha256'],b[1]['parent_lineage_sha256']);self.assertEqual(a[2],0);self.assertEqual(b[2],0)
            self.assertIsNone(e.reserve());self.assertEqual(e.snapshot()['states'],{'reserved':2})
        finally:e.close()
        resumed=Expansion(self.root/'out',[self.parent,p],target=100,backlog=2,config_sha='e'*64)
        try:self.assertIsNone(resumed.reserve())
        finally:resumed.close()
        with self.assertRaises(ValueError):Expansion(self.root/'out',[self.parent,p],target=100,backlog=2,config_sha='f'*64)
    def test_actual_step_creates_immutable_artifacts(self):
        e=Expansion(self.root/'out',[self.parent],target=100,backlog=2,config_sha='e'*64)
        try:
            self.assertTrue(e.step());v=e.db.execute('SELECT * FROM variants').fetchone();dest=self.root/'out'/'variants'/v['identity']
            self.assertEqual(v['state'],'generated');self.assertEqual(byte_sha((dest/'metadata.json').read_bytes()),v['metadata_sha'])
            self.assertEqual(byte_sha(gzip.decompress((dest/'encoder.jtt1.gz').read_bytes())),v['content_sha'])
            self.assertEqual(e.snapshot()['published'],0);self.assertEqual(e.snapshot()['admitted'],0)
        finally:e.close()
    def test_oversized_and_unbalanced_rows_refused(self):
        with self.assertRaises(ValueError):decode(encode(self.x,self.y,np.array([0]+[1]*59,dtype=np.uint8)))
        with self.assertRaises(ValueError):Expansion(self.root/'out',[self.parent],target=100000,backlog=2,config_sha='e'*64)
    def test_output_symlink_ancestors_database_and_variants_refused(self):
        actual=self.root/'actual';actual.mkdir();link=self.root/'link';link.symlink_to(actual,target_is_directory=True)
        with self.assertRaises(ValueError):Expansion(link/'child',[self.parent],target=100,backlog=2,config_sha='e'*64)
        for name in ('expansion.sqlite','expansion.sqlite-wal','variants','progress.new'):
            root=self.root/('case-'+name);root.mkdir();(root/name).symlink_to(actual)
            with self.assertRaises(ValueError):Expansion(root,[self.parent],target=100,backlog=2,config_sha='e'*64)
    def test_configuration_settings_cannot_change_under_same_digest(self):
        e=Expansion(self.root/'out',[self.parent],target=100,backlog=2,config_sha='e'*64);e.close()
        with self.assertRaises(ValueError):Expansion(self.root/'out',[self.parent],target=101,backlog=2,config_sha='e'*64)
    def test_actual_registry_null_and_pinned_preparation(self):
        source({**self.parent,'preparation_receipt':None})
        receipt=self.root/'preparation.json';receipt.write_bytes(b'{}')
        ref={'path':str(receipt),'sha256':byte_sha(b'{}'),'bytes':2}
        source({**self.parent,'preparation_receipt':ref})
        receipt.write_bytes(b'drift')
        with self.assertRaises(ValueError):source({**self.parent,'preparation_receipt':ref})
if __name__=='__main__':unittest.main()
