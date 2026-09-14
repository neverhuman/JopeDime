import tempfile
import unittest
from pathlib import Path

from qualify_teaching_store import prepare_request
from resume_expired_cycle import atomic,sha


class PackingResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.prior=self.root/'prior';self.prior.mkdir()
        self.binary=self.root/'binary';self.binary.write_bytes(b'frozen binary')
        (self.root/'contexts').write_bytes(b'frozen contexts')
        (self.root/'reference').mkdir();(self.root/'reference/index.json').write_bytes(b'frozen reference')
        self.request=dict(schema='parallel.teaching_store_qualification.v1',sources=['ledger-a'],
            reference='reference',reference_index_sha256=sha(self.root/'reference/index.json'),
            contexts='contexts',contexts_sha256=sha(self.root/'contexts'),executable_sha256=sha(self.binary),
            final_tests_opened=False)
        atomic(self.prior/'request.json',self.request)
        atomic(self.prior/'complete.json',dict(status='inconclusive',request_sha256=sha(self.prior/'request.json')))

    def resume(self):
        # No current manifest or state is needed; live collection may have advanced.
        return prepare_request(self.root,self.root/'absent-manifest',self.binary,self.root/'next',300,self.prior)

    def test_resume_preserves_sources_store_and_original_receipts(self):
        previous=sha(self.prior/'complete.json');request=self.resume()
        self.assertEqual(request['sources'],['ledger-a'])
        self.assertEqual(request['store'],'prior/store')
        self.assertEqual(request['ceiling_seconds'],300)
        self.assertEqual(request['resume']['sha256'],previous)
        self.assertEqual(sha(self.prior/'complete.json'),previous)

    def test_active_or_changed_attempt_cannot_resume(self):
        (self.prior/'complete.json').unlink()
        with self.assertRaises(FileNotFoundError):self.resume()
        atomic(self.prior/'complete.json',dict(status='inconclusive',request_sha256='changed'))
        with self.assertRaisesRegex(RuntimeError,'request changed'):self.resume()

    def test_changed_inputs_cannot_resume(self):
        for path,reason in [(self.binary,'executable'),(self.root/'contexts','contexts'),
                            (self.root/'reference/index.json','reference')]:
            with self.subTest(reason=reason):
                original=path.read_bytes();path.write_bytes(b'changed')
                with self.assertRaisesRegex(RuntimeError,reason+' changed'):self.resume()
                path.write_bytes(original)
