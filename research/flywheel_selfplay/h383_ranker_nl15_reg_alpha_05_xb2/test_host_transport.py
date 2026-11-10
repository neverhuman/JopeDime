"""Exercise credential-free receipt transport and own-unit guard targeting without models."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import hashlib,inspect,json,os,tempfile,unittest
import archive_relay as relay
import archive_h383 as archive
import guard_h383 as guard

class TransportTests(unittest.TestCase):
 def test_AWS_client_refused_on_xbabe1(self):
  with patch.object(archive.os,'uname',return_value=SimpleNamespace(nodename='xbabe1')):
   with self.assertRaises(AssertionError):archive.client()
 def test_xbabe1_upload_dispatch_never_enters_AWS(self):
  receipt={'sent':'via-file-protocol'}
  with patch.object(archive.os,'uname',return_value=SimpleNamespace(nodename='xbabe1')),patch.object(relay,'request_upload',return_value=receipt) as send,patch.object(archive,'client',side_effect=AssertionError('AWS touched')):
   self.assertEqual(archive.upload_verify('/unused','l1a05_members/test',{'member':'sha'}),receipt);send.assert_called_once()
 def test_guard_targets_only_H383_x1_unit(self):self.assertEqual(inspect.signature(guard.monitor).parameters['unit'].default,'h383-run.service')
 def exercise(self,bad=False):
  parent=Path(__file__).resolve().parent.parent/'transport-test-fixtures';parent.mkdir(exist_ok=True)
  with tempfile.TemporaryDirectory(dir=parent) as temp:
   d=Path(temp);p=d/'P0/fold_0/root_30001/inner_members.tar.gz';p.parent.mkdir(parents=True);p.write_bytes(b'archive-copy');h=hashlib.sha256(p.read_bytes()).hexdigest();rename=os.replace
   def answer(src,dst):
    rename(src,dst);request=json.loads(Path(dst).read_text());response=d/'transfer/responses'/Path(dst).name
    response.write_text(json.dumps(dict(sha256='wrong' if bad else h,bytes=len(b'archive-copy'),key='H383/'+request['key'],exact_version_get_sha256_verified=True,all_members_exact_version_verified=True,version_id='exact-version')))
   with patch.object(relay,'D',d),patch.object(relay.os,'uname',return_value=SimpleNamespace(nodename='xbabe1')),patch.object(relay.os,'replace',side_effect=answer):
    if bad:
     with self.assertRaises(AssertionError):relay.request_upload(p,'l1a05_members/task.tar.gz',{'member':'sha'})
    else:self.assertEqual(relay.request_upload(p,'l1a05_members/task.tar.gz',{'member':'sha'})['sha256'],h)
   self.assertEqual(p.read_bytes(),b'archive-copy')
 def test_exact_receipt_unblocks_with_raw_copy_retained(self):self.exercise()
 def test_mismatched_receipt_refused_with_raw_copy_retained(self):self.exercise(True)
if __name__=='__main__':unittest.main()
