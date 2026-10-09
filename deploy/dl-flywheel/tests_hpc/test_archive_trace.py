import io,json,tempfile,unittest,time
from pathlib import Path
from archive_trace import canonical,sha,verify_segment,put_verified,PREFIX

class TraceTests(unittest.TestCase):
 def test_chain_and_tampering(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);p=root/'events-00000.jsonl';s=root/'events-00000.sealed.json'
   event={'host':'xbabe1','seq':1,'previous':'0'*64,'payload':{'n':1}}
   head=sha(canonical(event));body=canonical({**event,'sha256':head})+b'\n';p.write_bytes(body)
   seal={'path':str(p),'bytes':len(body),'sha256':sha(body),'last_seq':1,'event_head':head};s.write_bytes(canonical(seal))
   self.assertEqual(verify_segment(root,s,'0'*64,0,'xbabe1')[2:],(head,1))
   with self.assertRaises(ValueError):verify_segment(root,s,'f'*64,0,'xbabe1')
   with self.assertRaises(ValueError):verify_segment(root,s,'0'*64,0,'xbabe2')
   p.write_bytes(body.replace(b'"n":1',b'"n":2'))
   with self.assertRaises(ValueError):verify_segment(root,s,'0'*64,0,'xbabe1')
 def test_path_escape(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);s=root/'events-00000.sealed.json';s.write_bytes(canonical({'path':'/tmp/wrong.jsonl'}))
   with self.assertRaises(ValueError):verify_segment(root,s,'0'*64,0,'xbabe1')
 def test_exact_version_readback(self):
  class Client:
   def put_object(self,**kw):self.kw=kw;return {'VersionId':'v1'}
   def get_object(self,**kw):
    self.get=kw;return {'VersionId':'v1','ContentLength':3,'Body':io.BytesIO(self.body)}
  c=Client();c.body=b'yes';r=put_verified(c,PREFIX+'xbabe1/test',b'yes',time.time()+10)
  self.assertEqual(c.kw['IfNoneMatch'],'*');self.assertEqual(c.get['VersionId'],'v1');self.assertEqual(r['sha256'],sha(b'yes'))
  c.body=b'bad'
  with self.assertRaises(ValueError):put_verified(c,PREFIX+'xbabe1/test',b'yes',time.time()+10)
  with self.assertRaises(ValueError):put_verified(c,'foreign',b'yes',time.time()+10)
  with self.assertRaises(ValueError):put_verified(c,PREFIX+'xbabe1/test',b'yes',time.time()-1)
if __name__=='__main__':unittest.main()
