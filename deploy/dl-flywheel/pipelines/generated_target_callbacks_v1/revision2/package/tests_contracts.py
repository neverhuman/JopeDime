"""Physical role and actual captured publication-shape regression fixtures."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from . import bindings as B
from .readonly_io import canonical,sha

class AuthorityContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
        (self.root/'gp-hwm-index').mkdir();(self.root/'exclusions').mkdir()
        self.pins=[]
        self.role=self.write('role.json',{'schema':'jope.dltrain-one-holdout-role-map/v1'})
        self.layer=self.write('exclusions/layer.json',{'schema':'jope.dltrain-training-exclusion-layer/v1','entries':[{'context_sha256':'a'*64,'view_ids':['excluded']}]})
        self.layers=self.write('exclusions/TRAINING-EXCLUSION-LAYERS-v4-test.json',{'layers':[self.role,self.layer]})
        self.target=self.write('gp-hwm-index/gp-target-one-holdout-after-layers-v4-test.json',{'schema':'jope.dltrain-gp-target-after-layers/v1',
              'registry':self.layers,'inputs':{self.role['path']:self.role['sha256']},'views':['allowed'],'gp_target':1})
    def write(self,name,value):
        p=self.root/name;raw=canonical(value);p.write_bytes(raw);r={'path':str(p),'sha256':sha(raw),'bytes':len(raw)};self.pins.append(r);return r
    def check(self):return B.current_roles({'authority_pins':self.pins},self.root)
    def test_current_physical_full_role_closure(self):
        self.assertEqual(self.check(),({'allowed'},{'a'*64},{'excluded'}))
    def test_newer_authority_pointer_refuses_frozen_copy(self):
        newer=self.root/'gp-hwm-index/gp-target-one-holdout-after-layers-v5-new.json';newer.write_text('{}')
        os.utime(newer,ns=(999999999999999999,9999999999999999999))
        with self.assertRaisesRegex(ValueError,'newer physical'):self.check()
    def test_target_role_join_cannot_echo_wrong_sha(self):
        target=json.loads(Path(self.target['path']).read_bytes());target['inputs'][self.role['path']]='b'*64
        raw=canonical(target);Path(self.target['path']).write_bytes(raw);self.target.update(sha256=sha(raw),bytes=len(raw))
        with self.assertRaisesRegex(ValueError,'role hash joins'):self.check()
    def test_removed_or_changed_layer_fails(self):
        Path(self.layer['path']).write_text('{}')
        with self.assertRaises(ValueError):self.check()
    def test_copied_authority_root_is_not_a_live_role_proof(self):
        with self.assertRaisesRegex(ValueError,'designated physical root'):
            B.current_roles({'authority_pins':self.pins},Path('/home/ubuntu/scratch/jopedime-dl-training-v1/receipts'))
    def test_actual_annotated_ref_projection_preserves_original(self):
        fixture=json.loads((Path(__file__).parent/'fixtures/actual-publication-ref-shape.json').read_bytes());before=copy.deepcopy(fixture)
        for name,ref in fixture['task_refs'].items():
            self.assertEqual(B.publication_identity(ref),fixture['normalized_descriptor_refs'][name])
        self.assertEqual(fixture,before)
    def test_unknown_custody_annotation_and_version_are_refused(self):
        fixture=json.loads((Path(__file__).parent/'fixtures/actual-publication-ref-shape.json').read_bytes());ref=fixture['task_refs']['encoder_publication']
        for key,value in [('state','passed'),('verification','HEAD only'),('version_id','null')]:
            bad=dict(ref);bad[key]=value
            with self.assertRaises(ValueError):B.publication_identity(bad)
