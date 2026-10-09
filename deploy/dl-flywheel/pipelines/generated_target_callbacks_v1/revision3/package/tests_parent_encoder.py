"""New actual-shape gzip regression tests; synthetic payload, no live reads."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from . import callbacks as C
from .parent_encoder import recheck,MAX_RAW
from .readonly_io import canonical,sha

class ParentEncoderTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
        self.lineage='a'*64;self.raw=b'prepared-parent-raw-fixture'
        self.path=self.root/'encoder.jtt1.gz';self.path.write_bytes(gzip.compress(self.raw,mtime=0));self.refresh()
    def refresh(self):
        zipped=self.path.read_bytes();self.ref={'path':str(self.path),'sha256':sha(zipped),'bytes':len(zipped),'raw_sha256':sha(self.raw)}
        self.parent={'parent_lineage_sha256':self.lineage,'encoder':copy.deepcopy(self.ref),'raw_payload_sha256':sha(self.raw),
                     'partition':'train','view_id':'allowed','retained_lineage_context_sha256':['b'*64]}
    def check(self,local=None,encoders=None,parents=None,roles=None):
        return recheck([]if local is None else local,{self.lineage:self.ref}if encoders is None else encoders,
                       {self.lineage:self.parent}if parents is None else parents,{self.lineage}if roles is None else roles)
    def test_actual_four_field_shape_verifies_both_hashes_without_mutation(self):
        evidence=json.loads((Path(__file__).parent/'fixtures/actual-parent-pin-shape.json').read_bytes())
        self.assertEqual(sorted(self.ref),evidence['refs'][0]['keys'])
        before=(self.path.read_bytes(),self.path.stat().st_mtime_ns,list(self.root.iterdir()))
        self.check();self.assertEqual(before,(self.path.read_bytes(),self.path.stat().st_mtime_ns,list(self.root.iterdir())))
    def test_wrong_decompressed_hash_refused_even_when_annotation_matches_registry(self):
        self.ref['raw_sha256']='c'*64;self.parent['encoder']=dict(self.ref);self.parent['raw_payload_sha256']='c'*64
        with self.assertRaisesRegex(ValueError,'decompressed parent raw SHA'):self.check()
    def test_wrong_compressed_hash_or_bytes_refused(self):
        for key,value in [('sha256','c'*64),('bytes',self.ref['bytes']+1)]:
            original=copy.deepcopy(self.ref);self.ref[key]=value;self.parent['encoder']=dict(self.ref)
            with self.assertRaisesRegex(ValueError,'exact hash/bytes'):self.check()
            self.ref=original
    def test_raw_annotation_must_equal_parent_registry_payload_identity(self):
        self.parent['raw_payload_sha256']='c'*64
        with self.assertRaisesRegex(ValueError,'annotation/registry'):self.check()
    def test_corrupt_gzip_is_not_accepted_by_compressed_hash_alone(self):
        self.path.write_bytes(b'not-gzip');self.refresh()
        with self.assertRaises((ValueError,OSError,EOFError)):self.check()
    def test_decompression_overflow_is_bounded(self):
        self.raw=b'x'*(MAX_RAW+1);self.path.write_bytes(gzip.compress(self.raw,mtime=0));self.refresh()
        with self.assertRaisesRegex(ValueError,'bounded decompressed'):self.check()
    def test_unknown_parent_annotation_is_refused(self):
        self.ref['extra']='unapproved';self.parent['encoder']=dict(self.ref)
        with self.assertRaisesRegex(ValueError,'closed role-specific'):self.check()
    def test_annotation_is_not_allowed_on_generic_or_unreplayed_refs(self):
        with self.assertRaisesRegex(ValueError,'closed local pin'):self.check(local=[self.ref])
        with self.assertRaisesRegex(ValueError,'exact replayed parent'):self.check(roles=set())
        with self.assertRaisesRegex(ValueError,'exact replayed parent'):self.check(encoders={})
    def test_parent_encoder_role_cannot_be_substituted_or_transferred(self):
        self.parent['encoder']=dict(self.ref,raw_sha256='c'*64)
        with self.assertRaisesRegex(ValueError,'registry role'):self.check()
        self.parent['encoder']=dict(self.ref);self.parent['parent_lineage_sha256']='d'*64
        with self.assertRaisesRegex(ValueError,'parent identity'):self.check()
    def test_generic_two_and_three_field_pins_remain_strict_and_supported(self):
        plain=self.root/'source.py';plain.write_bytes(b'unchanged')
        two={'path':str(plain),'sha256':sha(b'unchanged')};three=dict(two,bytes=9)
        self.check(local=[two,three])
        with self.assertRaisesRegex(ValueError,'closed local pin'):self.check(local=[dict(three,raw_sha256=sha(b'unchanged'))])
    def test_physical_recheck_uses_distinct_parent_role_and_keeps_generic_guard(self):
        descriptors={'e'*64:{'parent_lineage_sha256':self.lineage,'partition':'train'}}
        state=SimpleNamespace(config=lambda:{'refs':{'kumo_audit':{},'weight_license':{},'code_license':{}}},
             window=SimpleNamespace(tick=lambda:None),inputs=SimpleNamespace(parents={self.lineage:self.parent},generator={}),
             descriptors_bytes=canonical(descriptors),replayed=frozenset(descriptors),parent_roles_bytes=canonical({self.lineage:'train'}),
             local_pins_bytes=canonical([]),parent_encoder_pins_bytes=canonical({self.lineage:self.ref}))
        with patch.object(C,'_package_check'),patch.object(C,'_import_origins'),patch.object(C.B,'license_contract'),             patch.object(C.B,'current_roles',return_value=({'allowed'},set(),set())):
            C._physical_checks(state)
            state.local_pins_bytes=canonical([self.ref])
            with self.assertRaisesRegex(ValueError,'closed local pin'):C._physical_checks(state)
