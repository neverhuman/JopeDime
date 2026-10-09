import unittest,struct,hashlib
import numpy as np
from native_inputs import masked_training,OriginalScoreContext,sha
class NativeInputsTest(unittest.TestCase):
 def setUp(self):
  n,nt,p=50,10,2;self.f=np.array([0,1]*25,dtype='u1');self.data=np.arange(n*(p+1),dtype='<f8').reshape(n,p+1);self.test=np.full((nt,p+1),np.nan,dtype='<f8')
  self.raw=b'JTT1'+struct.pack('<IIIB',n,nt,p,2)+self.f.tobytes()+self.data.tobytes()+self.test.tobytes()
  mask=self.test.copy();mask[:,-1]=0;self.masked=self.raw[:17+n+n*(p+1)*8]+mask.tobytes()
  self.row={'shape':{'n_train':n,'n_test':nt,'p':p,'k':2,'fold_sizes':[25,25]},'s1_input':{'bytes':len(self.masked),'sha256':sha(self.masked)}}
 def test_outer_values_remain_opaque(self):
  raw,x,y,f=masked_training(self.raw,self.row);self.assertEqual(raw,self.masked);np.testing.assert_array_equal(x,self.data[:,:2]);self.assertFalse(y.flags.writeable);self.assertFalse(f.flags.writeable)
 def test_reject_target_or_fold_mutation(self):
  b=bytearray(self.raw);b[17+50+16]^=1
  with self.assertRaises(ValueError):masked_training(bytes(b),self.row)
  b=bytearray(self.raw);b[17]=1
  with self.assertRaises(ValueError):masked_training(bytes(b),self.row)
 def test_reject_incorrect_registered_input(self):
  self.row['s1_input']['sha256']='0'*64
  with self.assertRaises(ValueError):masked_training(self.raw,self.row)
 def test_reject_stripped_synthetic_shape(self):
  raw=bytearray(self.raw);raw[8:12]=struct.pack('<I',0)
  with self.assertRaises(ValueError):masked_training(bytes(raw),self.row)
if __name__=='__main__':unittest.main()
