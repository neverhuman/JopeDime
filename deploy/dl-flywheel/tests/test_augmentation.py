import dataclasses
import hashlib
import json
import unittest
import numpy as np
from jope_flywheel import augmentation as a
from jope_flywheel.protocol import partition

def h(x):return hashlib.sha256(str(x).encode()).hexdigest()

class AugmentationTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(7)
        self.x=rng.normal(size=(30,20)).astype(np.float32)
        self.x[:,0]=7 # constant continuous column
        self.x[:,17]=np.arange(30)%6;self.x[:,18]=np.arange(30)%2;self.x[:,19]=np.arange(30)%4
        self.y=np.arange(30,dtype=np.float64)-7
        self.features=tuple(a.Feature('x'+str(i),'continuous','float32')for i in range(17))+(
            a.Feature('x17','integer','int32'),a.Feature('x18','binary','bool'),a.Feature('x19','categorical','int16'))
        self.roles={i:'fit'if i<20 else'validation'if i<25 else'test'for i in range(30)}
        self.parent=a.Parent(dataset_sha256=h('dataset'),context_sha256=h('context'),target_sha256=h('target'),
            target_name='y',parent_lineage_sha256=h('family'),split_namespace_sha256=h('split'),
            partition=partition(h('family'),h('split')),cut_id='fold1-inner0',
            feature_schema_sha256=a.schema_sha256(self.features),row_roles_sha256=a.roles_sha256(self.roles),
            admission_receipt_sha256=h('admitted'))
    def fit(self,spec=None,x=None,features=None,parent=None):
        return a.fit((self.x if x is None else x)[:20],features=self.features if features is None else features,
            row_ids=np.arange(20),row_roles=self.roles,parent=self.parent if parent is None else parent,
            spec=spec or a.Spec(42,.1,.1,.03))
    def test_reconstruction_reproducible_and_input_immutable(self):
        before=self.x.copy();y=self.y.copy();one=self.fit();two=self.fit()
        self.assertEqual(one.identity,two.identity);self.assertEqual(one.manifest(),two.manifest())
        one_batch=a.apply(self.x,self.y,row_ids=range(30),state=one)
        two_batch=a.apply(self.x,self.y,row_ids=range(30),state=two)
        np.testing.assert_array_equal(one_batch.x,two_batch.x)
        np.testing.assert_array_equal(self.x,before);np.testing.assert_array_equal(self.y,y)
        self.assertEqual(one_batch.receipt,two_batch.receipt)
        json.dumps(one.manifest(),allow_nan=False);json.dumps(one_batch.receipt,allow_nan=False)
    def test_train_only_scales_and_heldout_changes_cannot_change_fit(self):
        altered=self.x.copy();altered[20:,:17]*=1000
        one=self.fit();two=self.fit(x=altered)
        self.assertEqual(one.identity,two.identity)
        expected=np.std(self.x[:20,:17],axis=0,dtype=np.float64)
        np.testing.assert_allclose(one.train_std[:17],expected,rtol=1e-14,atol=0)
        with self.assertRaises(ValueError):a.fit(self.x[:21],features=self.features,row_ids=range(21),row_roles=self.roles,parent=self.parent,spec=a.Spec(1))
    def test_rows_targets_and_roles_unchanged(self):
        state=self.fit();order=np.array([28,2,22,1,25])
        result=a.apply(self.x[order],self.y[order],row_ids=order,state=state)
        np.testing.assert_array_equal(result.y,self.y[order]);self.assertEqual(result.y.dtype,self.y.dtype)
        self.assertEqual(result.row_ids,tuple(order));self.assertEqual(result.row_roles,tuple(self.roles[int(i)]for i in order))
        self.assertTrue(result.receipt['y_unchanged']);self.assertTrue(result.receipt['source_row_order_unchanged'])
        result.y[0]=999;self.assertNotEqual(self.y[28],999)
    def test_arbitrary_subsets_and_order_have_exact_same_features(self):
        state=self.fit();whole=a.apply(self.x,self.y,row_ids=range(30),state=state)
        for order in (np.array([29,0,8]),np.array([15]),np.arange(30)[::-1]):
            batch=a.apply(self.x[order],self.y[order],row_ids=order,state=state)
            np.testing.assert_array_equal(batch.x,whole.x[order])
    def test_small_drop_nuisance_mapping_and_minimum_width(self):
        state=self.fit();self.assertEqual(len(state.dropped_columns),2);self.assertEqual(len(state.nuisance_names),2)
        batch=a.apply(self.x,self.y,row_ids=range(30),state=state);self.assertEqual(batch.x.shape,(30,20))
        self.assertEqual([m['source_index']for m in batch.column_mapping[:18]],list(state.retained_columns))
        self.assertTrue(all(m['source_index']is None for m in batch.column_mapping[18:]))
        for p in (1,2,9):
            features=self.features[:p];parent=dataclasses.replace(self.parent,feature_schema_sha256=a.schema_sha256(features))
            state=a.fit(self.x[:20,:p],features=features,row_ids=range(20),row_roles=self.roles,parent=parent,spec=a.Spec(9,.1,.1,0))
            self.assertEqual(len(state.dropped_columns),0);self.assertEqual(len(state.nuisance_names),0)
            self.assertEqual(len(state.retained_columns),p)
    def test_noise_only_continuous_nonconstant(self):
        state=self.fit(a.Spec(42,0,0,.03));batch=a.apply(self.x,self.y,row_ids=range(30),state=state)
        np.testing.assert_array_equal(batch.x[:,0],self.x[:,0])
        np.testing.assert_array_equal(batch.x[:,17:],self.x[:,17:])
        self.assertFalse(np.array_equal(batch.x[:,1:17],self.x[:,1:17]));self.assertEqual(state.train_std[17:],(0.,0.,0.))
    def test_seed_parameters_target_schema_and_fit_data_bind_identity(self):
        base=self.fit();different=self.fit(a.Spec(43,.1,.1,.03));self.assertNotEqual(base.identity,different.identity)
        self.assertFalse(np.array_equal(a.apply(self.x,self.y,row_ids=range(30),state=base).x,a.apply(self.x,self.y,row_ids=range(30),state=different).x))
        for spec in (a.Spec(42,.05,.1,.03),a.Spec(42,.1,.05,.03),a.Spec(42,.1,.1,.01),a.Spec(42,.1,.1,.03,1)):
            self.assertNotEqual(base.identity,self.fit(spec).identity)
        self.assertNotEqual(base.identity,self.fit(parent=dataclasses.replace(self.parent,target_sha256=h('target2'))).identity)
        changed=self.x.copy();changed[0,1]+=1;self.assertNotEqual(base.identity,self.fit(x=changed).identity)
        renamed=(dataclasses.replace(self.features[0],name='renamed'),)+self.features[1:]
        self.assertNotEqual(base.identity,self.fit(features=renamed,parent=dataclasses.replace(self.parent,feature_schema_sha256=a.schema_sha256(renamed))).identity)
    def test_noop_variants_have_same_content_despite_distinct_seeds(self):
        s1=self.fit(a.Spec(1));s2=self.fit(a.Spec(2));self.assertNotEqual(s1.identity,s2.identity)
        b1=a.apply(self.x,self.y,row_ids=range(30),state=s1);b2=a.apply(self.x,self.y,row_ids=range(30),state=s2)
        self.assertEqual(b1.receipt['content_sha256'],b2.receipt['content_sha256']);self.assertFalse(b1.receipt['feature_values_changed'])
    def test_invalid_schema_target_leak_and_protected_role_rejected(self):
        with self.assertRaises(ValueError):self.fit(parent=dataclasses.replace(self.parent,target_name='x1'))
        with self.assertRaises(ValueError):self.fit(parent=dataclasses.replace(self.parent,feature_schema_sha256=h('bad')))
        with self.assertRaises(ValueError):dataclasses.replace(self.parent,partition='sealed')
        with self.assertRaises(ValueError):dataclasses.replace(self.parent,partition='train'if self.parent.partition=='validation'else'validation')
        for kind in ('target','unknown'):
            with self.assertRaises(ValueError):a.Feature('x',kind,'float32')
        with self.assertRaises(ValueError):a.Feature('x','categorical','float32')
    def test_nonfinite_wrong_type_rows_and_targets_refused(self):
        for value in (np.nan,np.inf):
            bad=self.x.copy();bad[0,1]=value
            with self.assertRaises(ValueError):self.fit(x=bad)
        state=self.fit()
        for y in (self.y[:,None],self.y[:-1],np.full(30,np.nan),np.ones(30,dtype=bool)):
            with self.assertRaises(ValueError):a.apply(self.x,y,row_ids=range(30),state=state)
        with self.assertRaises(ValueError):a.apply(self.x,self.y,row_ids=list(range(29))+[999],state=state)
        with self.assertRaises(ValueError):a.apply(self.x,self.y,row_ids=[0]*30,state=state)
        with self.assertRaises(ValueError):self.fit(x=self.x.astype(np.float16))
        bad=self.x.copy();bad[0,19]=.5
        with self.assertRaises(ValueError):self.fit(x=bad)
        bad=self.x.copy();bad[0,19]=2**15
        with self.assertRaises(ValueError):self.fit(x=bad)
    def test_configuration_bounds_and_unknown_versions_refused(self):
        for kw in ({'drop_fraction':.11},{'nuisance_fraction':.11},{'noise_fraction':.1},{'nuisance_max':9},{'seed':True},{'seed':-1},{'version':'future'}):
            with self.assertRaises(ValueError):a.Spec(**{'seed':1,**kw})
        with self.assertRaises(ValueError):a.apply(self.x,self.y,row_ids=range(30),state=dataclasses.replace(self.fit(),numpy_version='different'))
    def test_nuisance_cap_eight_and_target_never_scales_noise(self):
        features=tuple(a.Feature('p'+str(i),'continuous','float32')for i in range(100))
        x=np.ones((30,100),dtype=np.float32);parent=dataclasses.replace(self.parent,feature_schema_sha256=a.schema_sha256(features))
        state=a.fit(x[:20],features=features,row_ids=range(20),row_roles=self.roles,parent=parent,spec=a.Spec(3,0,.1,.03))
        self.assertEqual(len(state.nuisance_names),8)
        one=a.apply(x,self.y,row_ids=range(30),state=state);two=a.apply(x,self.y*100000,row_ids=range(30),state=state)
        np.testing.assert_array_equal(one.x,two.x);np.testing.assert_array_equal(two.y,self.y*100000)

if __name__=='__main__':unittest.main()
