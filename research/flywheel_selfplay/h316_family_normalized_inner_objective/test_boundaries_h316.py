"""Poisoned-tail loader controls and direct outer-score gate controls."""
import json,sys,tempfile,unittest,zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import run_h316 as run
sys.path.insert(0,str(run.FS/'h308_ensvote2_trainonly_repartition'))
import trainonly_loader as tl

class Boundaries(unittest.TestCase):
    def setUp(self):
        parent=run.CUSTODY/'test-fixtures';parent.mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=parent);self.data=Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
    def test_poisoned_mixed_tail_never_read(self):
        p=self.data/'context_features.npz';a=np.arange(28,dtype=np.float32).reshape(7,4)
        np.savez(p,cids=np.array(['t0','t1','t2','t3','t4','v0','v1']),lm48_z=a)
        off,_,_=tl._member_layout(p,'lm48_z.npy')
        with p.open('r+b') as f:
            f.seek(off+5*a[0].nbytes);f.write(np.full((2,4),np.nan,dtype=np.float32).tobytes())
        with np.load(p) as z:
            with self.assertRaises(zipfile.BadZipFile):z['lm48_z']
        with tl.TrainOnlyNumpy({'t'+str(i) for i in range(5)}).load(p) as z:
            np.testing.assert_array_equal(z['lm48_z'],a[:5])
    def test_nonprefix_train_refused(self):
        p=self.data/'context_features.npz';np.savez(p,cids=np.array(['t0','v0','t1']),lm48_z=np.zeros((3,2)))
        with self.assertRaises(RuntimeError):tl.TrainOnlyNumpy({'t0','t1'}).load(p)
    def fake(self):
        class Packets(dict):
            reads=0
            def __getitem__(self,key):
                self.reads+=1;raise AssertionError('outer packet read before gate')
        packets=Packets()
        r=SimpleNamespace(G=dict(train_rows=[dict(context_sha256='outer',family_id='F')],fam_by_cid={'outer':'F'},
                                   a302=SimpleNamespace(outer_fold_of_family=lambda f:0),P8=packets))
        return r,packets
    def test_missing_fit_lock_prevents_outer_read(self):
        r,p=self.fake()
        with patch.object(run,'DATA',self.data),patch.dict(run.R,{'r303':r},clear=True):
            with self.assertRaises(FileNotFoundError):run.score('P0','BASE')
        self.assertEqual(p.reads,0)
    def test_second_scoring_attempt_prevents_outer_read(self):
        r,p=self.fake();d=self.data/'P0/fold_0/root_30001';d.mkdir(parents=True)
        (d/'fit_lock.json').write_text('{}');(d/'base_score_once.json').write_text('{}')
        with patch.object(run,'DATA',self.data),patch.dict(run.R,{'r303':r},clear=True):
            with self.assertRaisesRegex(AssertionError,'already scored'):run.score('P0','BASE')
        self.assertEqual(p.reads,0)

if __name__=='__main__':unittest.main()
