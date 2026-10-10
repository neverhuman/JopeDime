"""New adversarial certificate/reader cases; all bodies are synthetic temp files."""
import dataclasses
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
from monitor_capture_reuse_v1.vendor.common import canonical,digest,Refusal,FRESH_LIMIT
from monitor_capture_reuse_v1.reuse import authenticate_history,verify_fresh_manifest,verify_normal_current
from monitor_capture_reuse_v1.reader import ReadSession,read_external_seal
from .fixtures import scenario,verified,request,reseal,phase,pin

class ReuseTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.f=scenario(self.tmp.name)
    def test_valid_reused_graph_and_fresh_bodies(self):
        f=self.f;h=verified(f);s=ReadSession(h)
        with patch('socket.socket',side_effect=AssertionError('no network')):
            out=s.capture_fresh(canonical(request(f)))
        self.assertEqual(len(out['body_wires']),3);self.assertEqual(out['historical_members_preserved'],4)
        self.assertFalse(out['monitor_or_publication_acceptance']);self.assertFalse((Path(self.tmp.name)/'custody').exists())
    def test_request_cannot_supply_its_own_anchor(self):
        d=request(self.f);d['root_seal']=self.f['seal'];h=verified(self.f)
        with self.assertRaises(Refusal):verify_fresh_manifest(canonical(d),h)
    def test_mutated_seal_against_external_sha(self):
        f=self.f;mut=deepcopy(f['seal']);mut['generation']=16
        with self.assertRaises(Refusal):authenticate_history(canonical(mut),f['external'],f['seal_pin'],f['seal_pin'],f['graph_wire'],f['aw'],f['evidence'])
    def test_missing_capture_authority_evidence(self):
        f=self.f;del f['evidence'][f['acceptance']['capture']['sha256']]
        with self.assertRaises(Refusal):verified(f)
    def test_missing_independent_review_authority(self):
        f=self.f;f['acceptance']['independent_review']=pin(b'unavailable');reseal(f)
        with self.assertRaises(Refusal):verified(f)
    def test_skipped_member_even_with_internal_graph_rehash(self):
        f=self.f;f['graph']['namespaces'][0]['members'].pop();reseal(f)
        with self.assertRaises(Refusal):verified(f)
    def test_wrong_source_inventory_commitment(self):
        f=self.f;f['acceptance']['source_inventory_sha256']='9'*64;reseal(f)
        with self.assertRaises(Refusal):verified(f)
    def test_wrong_acceptance_parent_chain(self):
        f=self.f;f['acceptance']['previous_acceptance_sha256']='8'*64;reseal(f)
        with self.assertRaises(Refusal):verified(f)
    def test_old_generation_refused(self):
        f=self.f;f['graph']['generation_index']=13;reseal(f)
        with self.assertRaises(Refusal):verified(f)
    def test_active_prefix_cannot_be_normal_terminal(self):
        f=self.f;f['graph']['namespaces'][0]['status']='pending_verification';f['acceptance']['status']='pending_verification';reseal(f)
        with self.assertRaises(Refusal):verified(f)
    def test_terminal_capture_after_original_deadline(self):
        f=self.f;f['acceptance']['terminal_capture']['captured_unix']=4601.0;reseal(f)
        with self.assertRaises(Refusal):verified(f)
    def test_missing_fsync_does_not_become_zero(self):
        f=self.f;f['acceptance']['terminal_capture']['fsync_receipt']=None;reseal(f)
        with self.assertRaises(Refusal):verified(f)
    def test_wrong_paired_terminal_clock(self):
        f=self.f;f['acceptance']['terminal_capture']['completed_monotonic_ns']=4601*10**9;reseal(f)
        with self.assertRaises(Refusal):verified(f)
    def test_fresh_generation_cannot_precede_terminal(self):
        f=self.f;f['seal']['current']['clock']['created_unix']=4000.0;f['seal']['current']['clock']['deadline_unix']=7600.0;f['seal']['current']['clock']['created_monotonic_ns']=4000*10**9;f['seal']['current']['clock']['deadline_monotonic_ns']=7600*10**9;reseal(f)
        with self.assertRaises(Refusal):verified(f)
    def test_crlf_physical_event_stream_refused(self):
        c,e,r,s,b=phase()
        with self.assertRaises(ValueError):verify_normal_current(c,e.replace(b'\n',b'\r\n'),r,s,b,'live')
    def test_embedded_non_lf_separator_refused(self):
        c,e,r,s,b=phase()
        for separator in (b'\v',b'\f',b'\x1c',b'\xc2\x85',b'\xe2\x80\xa8'):
            with self.subTest(separator=separator),self.assertRaises(ValueError):verify_normal_current(c,e.replace(b'\n',separator,1),r,s,b,'live')
    def test_exact_lf_phase_and_prefix_pass(self):
        c,e,r,s,b=phase();proof=verify_normal_current(c,e,r,s,b,'live');self.assertEqual(proof.event_count,4)
    def test_singleton_bound_refuses_before_open(self):
        f=self.f;d=request(f);d['members'][0]['bytes']=8*1024**2+1
        with patch('os.open',side_effect=AssertionError('opened before singleton refusal')):
            with self.assertRaises(Refusal):ReadSession(verified(f)).capture_fresh(canonical(d))
    def test_aggregate_projection_refuses_before_open(self):
        f=self.f;d=request(f);base=deepcopy(d['members'][0]);d['members']=[]
        for i in range(5):
            r=deepcopy(base);r.update(path=str(Path(f['seal']['current']['root'])/('big'+str(i))),inode=700+i,bytes=8*1024**2);d['members'].append(r)
        d['members'].extend(f['fresh']);d['members'].sort(key=lambda r:r['path'])
        with patch('os.open',side_effect=AssertionError('opened before aggregate refusal')):
            with self.assertRaises(Refusal):ReadSession(verified(f)).capture_fresh(canonical(d))
    def test_fresh_alias_refused(self):
        f=self.f;d=request(f);d['members'][1]['inode']=d['members'][0]['inode']
        with self.assertRaises(Refusal):verify_fresh_manifest(canonical(d),verified(f))
    def test_fresh_hash_corruption_fails_and_burns_session(self):
        f=self.f;s=ReadSession(verified(f));p=Path(f['fresh'][0]['path']);p.write_bytes(b'bad')
        with self.assertRaises(Refusal):s.capture_fresh(canonical(request(f)))
        with self.assertRaises(Refusal):s.capture_fresh(canonical(request(f)))
        self.assertTrue(s.failed)
    def test_second_capture_refused_not_budget_reset(self):
        f=self.f;s=ReadSession(verified(f));s.capture_fresh(canonical(request(f)));before=s.read_charge
        with self.assertRaises(Refusal):s.capture_fresh(canonical(request(f)))
        self.assertEqual(s.read_charge,before)
    def test_later_missing_historical_body_is_hold_no_retry(self):
        s=ReadSession(verified(self.f))
        with self.assertRaises(FileNotFoundError):s.later_body_use('old14','source.json')
        with self.assertRaises(Refusal):s.later_body_use('old14','source.json')
        self.assertTrue(s.failed)
    def test_shared_budget_applies_to_later_body_use(self):
        s=ReadSession(verified(self.f));s.read_charge=FRESH_LIMIT
        with patch('os.open',side_effect=AssertionError('opened over shared budget')):
            with self.assertRaises(Refusal):s.later_body_use('old14','source.json')
    def test_local_external_seal_read_and_mutation_refusal(self):
        f=self.f;wire,before,after=read_external_seal(f['external']);self.assertEqual(wire,f['seal_wire']);self.assertEqual(before,after)
        p=f['seal_path'];p.chmod(0o600)
        with self.assertRaises(Refusal):read_external_seal(f['external'])
    def test_authenticated_proof_field_substitution_refused(self):
        h=dataclasses.replace(verified(self.f),historical_bytes_not_read=0)
        with self.assertRaises(Refusal):h.check()

    def test_fake_history_refused_before_any_callback_or_io(self):
        class Forged:
            def check(self):raise AssertionError('must not invoke unverified object')
        with patch('os.open',side_effect=AssertionError('unverified IO')):
            with self.assertRaises(Refusal):ReadSession(Forged())

    def test_complete_current_inventory_is_externally_committed(self):
        f=self.f;h=verified(f);d=request(f)
        self.assertEqual(verify_fresh_manifest(canonical(d),h)['state'],'PENDING_FRESH_BODY_VERIFICATION')
        d['members'].pop()
        with patch('os.open',side_effect=AssertionError('opened before completeness refusal')):
            with self.assertRaises(Refusal):ReadSession(h).capture_fresh(canonical(d))
        d=request(f);r=deepcopy(d['members'][0]);r['path']+='.extra';r['inode']+=99999;d['members'].append(r);d['members'].sort(key=lambda r:r['path'])
        with self.assertRaises(Refusal):verify_fresh_manifest(canonical(d),h)
