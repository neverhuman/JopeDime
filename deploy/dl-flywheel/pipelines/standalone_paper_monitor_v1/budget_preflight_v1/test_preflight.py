import copy
import hashlib
import json
import unittest
from unittest.mock import patch
from preflight import ExternalPlanPin, Refusal, project


def plan():
    return {'schema':'codex.legacy-producer-budget-preflight/v1',
            'roster':[row('/captured/v1/body',100)],
            'metadata':[{'id':'authority','sha256':'a'*64,'bytes':100,'read_count':1,'retained_copies':1,'serialized_copies':1}],
            'output_reservations':[{'id':'summary','maximum_body_bytes':128,'maximum_readbacks':1,'maximum_retained_copies':1,'maximum_transport_bytes':256}],
            'transport_envelope_reservation_bytes':1024}

def row(path,size):
    return {'namespace':'synthetic','original_path':path,'sha256':'b'*64,'bytes':size,
            'read_count':1,'retained_copies':1,'serialized_copies':1}

def wire_and_pin(p):
    wire=json.dumps(p,sort_keys=True,separators=(',',':')).encode()
    return wire, ExternalPlanPin(hashlib.sha256(wire).hexdigest(),len(wire))

class PreflightCases(unittest.TestCase):
    def call(self,p):
        wire,pin=wire_and_pin(p)
        with patch('builtins.open',side_effect=AssertionError('payload I/O forbidden')):
            return project(wire,pin)

    def test_small_projection_has_no_admission_or_payload_io(self):
        p=plan(); result=self.call(p)
        self.assertEqual(result.state,'WITHIN_BUDGET_NOT_ADMISSION')
        self.assertFalse(result.admission_authorized)
        self.assertEqual(result.source_body_reads,0)
        self.assertEqual(result.fresh_read_bytes_reserved,len(wire_and_pin(p)[0])+328)

    def test_external_pin_wrong_type_refuses_before_attributes(self):
        class Fake:
            def __getattr__(self,n):raise AssertionError('fake authority used')
        with self.assertRaisesRegex(Refusal,'typed plan pin'):project(b'{}',Fake())

    def test_missing_member_cannot_rebind_external_plan(self):
        p=plan();p['roster'].append(row('/captured/v2/body',100))
        original,pin=wire_and_pin(p);p['roster'].pop();changed,_=wire_and_pin(p)
        with self.assertRaisesRegex(Refusal,'exact plan bytes|external plan hash'):project(changed,pin)
        self.assertGreater(len(original),len(changed))

    def test_six_required_roots_exceed_fixed_cap_before_io(self):
        p=plan();p['roster']=[row('/synthetic/v'+str(i)+'/body',n)for i,n in enumerate([6091026,3422304,6515258,6579566,6457341,6608643])]
        self.assertEqual(sum(r['bytes']for r in p['roster']),35674138)
        with self.assertRaisesRegex(Refusal,'read budget exceeded'):self.call(p)

    def test_single_body_above_eight_mib_refuses(self):
        p=plan();p['roster'][0]['bytes']=8*1024**2+1
        with self.assertRaisesRegex(Refusal,'single body bound'):self.call(p)

    def test_metadata_body_above_one_mib_refuses(self):
        p=plan();p['metadata'][0]['bytes']=1024**2+1
        with self.assertRaisesRegex(Refusal,'metadata body bound'):self.call(p)

    def test_future_output_readbacks_are_reserved_before_creation(self):
        p=plan();p['output_reservations'][0].update(maximum_body_bytes=8*1024**2,maximum_readbacks=4,maximum_transport_bytes=8*1024**2)
        with self.assertRaisesRegex(Refusal,'read budget exceeded'):self.call(p)

    def test_future_output_cannot_fabricate_inode(self):
        p=plan();p['output_reservations'][0]['inode']=1
        with self.assertRaisesRegex(Refusal,'no fabricated identity'):self.call(p)

    def test_equal_content_on_different_paths_not_deduplicated(self):
        p=plan();p['roster']=[row('/captured/'+str(i),8*1024**2)for i in range(4)]
        with self.assertRaisesRegex(Refusal,'read budget exceeded'):self.call(p)

    def test_duplicate_original_path_refuses(self):
        p=plan();p['roster'].append(copy.deepcopy(p['roster'][0]));p['roster'][1]['sha256']='c'*64
        with self.assertRaisesRegex(Refusal,'path duplicate'):self.call(p)

    def test_repeat_reads_cannot_reset_budget(self):
        p=plan();p['roster'][0].update(bytes=6*1024**2,read_count=6)
        with self.assertRaisesRegex(Refusal,'read budget exceeded'):self.call(p)

    def test_retained_copies_have_separate_bound(self):
        p=plan();p['roster'][0].update(bytes=6*1024**2,retained_copies=6)
        with self.assertRaisesRegex(Refusal,'retention exceeded'):self.call(p)

    def test_transport_envelope_charged_beyond_body_lower_bound(self):
        p=plan();p['transport_envelope_reservation_bytes']=48*1024**2
        with self.assertRaisesRegex(Refusal,'transport reservation exceeded'):self.call(p)

    def test_boolean_bytes_and_request_cap_override_refuse(self):
        p=plan();p['roster'][0]['bytes']=True
        with self.assertRaisesRegex(Refusal,'single body bound'):self.call(p)
        p=plan();p['read_limit']=10**12
        with self.assertRaisesRegex(Refusal,'closed plan'):self.call(p)

    def test_duplicate_json_key_refuses(self):
        w=b'{"schema":1,"schema":2}';pin=ExternalPlanPin(hashlib.sha256(w).hexdigest(),len(w))
        with self.assertRaisesRegex(Refusal,'unique-key JSON'):project(w,pin)
