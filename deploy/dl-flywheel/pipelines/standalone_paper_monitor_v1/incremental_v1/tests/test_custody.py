import json
import unittest
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch
from monitor_incremental_custody_v1.common import (Refusal, canonical, digest, document, RETAINED_LIMIT, FRESH_LIMIT, TRANSPORT_LIMIT)
from monitor_incremental_custody_v1.authority import (ExternalAnchor, VerifiedAuthority, verify_authority)
from monitor_incremental_custody_v1.custody_manifest import (BodyWitness, verify_accepted_history, verify_candidate)
from monitor_incremental_custody_v1.phase_verifier import verify_normal_phase
from .fixtures import (family, authenticated, verified, candidate, reseal_authority, reseal_graph, phase_case, rechain)

class AuthorityCases(unittest.TestCase):
    def test_external_seal_exact_valid(self):
        f=family(); a=authenticated(f)
        self.assertEqual(a.sha256,digest(f['aw'])); self.assertEqual(a.data(),f['authority'])
        self.assertFalse(hasattr(a,'admission_allowed'))

    def test_request_echo_cannot_replace_root_seal(self):
        f=family(); changed=deepcopy(f['authority']); changed['engine_source']['sha256']='b'*64
        with self.assertRaises(Refusal): verify_authority(canonical(changed),f['anchor'],f['seal'],f['seal'])
        with self.assertRaises(Refusal): VerifiedAuthority(f['aw'],digest(f['aw']),'0'*64)

    def test_rollback_even_with_self_consistent_new_bytes(self):
        f=family(); f['authority']['generation_index']=3; reseal_authority(f)
        anchor=replace(f['anchor'],generation_index=4)
        with self.assertRaises(Refusal): verify_authority(f['aw'],anchor,f['seal'],f['seal'])

    def test_wrong_external_head_and_policy(self):
        f=family()
        for field in ('predecessor_graph_sha256','predecessor_acceptance_sha256','policy_sha256'):
            with self.subTest(field=field),self.assertRaises(Refusal):
                verify_authority(f['aw'],replace(f['anchor'],**{field:'b'*64}),f['seal'],f['seal'])

    def test_original_clock_caps_and_guard_refusal(self):
        for mutation in ('clock','clock_tolerance','caps','boolean_caps','guard','limit'):
            f=family()
            if mutation=='clock': f['authority']['clock']['deadline_unix']+=1
            if mutation=='clock_tolerance': f['authority']['clock']['deadline_monotonic_ns']-=20_000
            if mutation=='caps': f['authority']['policy']['caps']['memory_max']*=2
            if mutation=='boolean_caps': f['authority']['policy']['caps']['memory_swap_max']=False
            if mutation=='guard': f['authority']['fresh_guards'].pop()
            if mutation=='limit': f['authority']['limits']['fresh_read_bytes']+=1
            reseal_authority(f)
            with self.subTest(mutation=mutation), self.assertRaises(Refusal): authenticated(f)

    def test_physical_seal_replacement_and_writable_refused(self):
        f=family(); other=deepcopy(f['seal']); other['inode']+=1
        with self.assertRaises(Refusal): verify_authority(f['aw'],f['anchor'],f['seal'],other)
        other=deepcopy(f['seal']); other['mode']=0o600
        with self.assertRaises(Refusal): verify_authority(f['aw'],f['anchor'],other,other)

    def test_json_duplicate_noncanonical_and_unknown_fields(self):
        for wire in (b'{"a":1,"a":2}', b'{ "a": 1}', b'{"a":NaN}'):
            with self.subTest(wire=wire),self.assertRaises(Refusal): document(wire)
        f=family(); f['authority']['approved']=True; reseal_authority(f)
        with self.assertRaises(Refusal): authenticated(f)

class CustodyCases(unittest.TestCase):
    def test_complete_reuse_preserves_failed_unknown_and_pending(self):
        f=family(); a,h=verified(f); d,c=candidate(f,a,h)
        out=verify_candidate(canonical(d),h,f['new_witnesses'],a,c)
        got=document(out.wire)
        self.assertEqual([x['status'] for x in got['namespaces']],['failed','unknown','normal_closed','pending_verification'])
        self.assertEqual(got['namespaces'][:-1],f['graph']['namespaces'])
        self.assertIsNone(got['namespaces'][-1]['acceptance']); self.assertFalse(hasattr(out,'admission_allowed'))

    def test_held_source_only_preserves_absent_clock_and_config(self):
        f=family(('held_source_only','failed','normal_closed')); a,h=verified(f); d,c=candidate(f,a,h)
        out=verify_candidate(canonical(d),h,f['new_witnesses'],a,c)
        held=document(out.wire)['namespaces'][0]
        self.assertEqual(held['window_state'],'not_created'); self.assertIsNone(held['clock']); self.assertIsNone(held['config_sha256'])
        d['namespaces'][0]['clock']=deepcopy(f['authority']['clock'])
        with self.assertRaises(Refusal): verify_candidate(canonical(d),h,f['new_witnesses'],a,c)

    def test_copied_proof_cannot_launder_history_or_authority(self):
        f=family(); a,h=verified(f)
        for mutation in ({'witness_bytes':0},{'graph_wire':b'{}'}, {'member_count':0}):
            with self.subTest(mutation=list(mutation)),self.assertRaises(Refusal): replace(h,**mutation)
        changed=deepcopy(f['authority']); changed['new_namespace']['id']='elsewhere'; raw=canonical(changed)
        with self.assertRaises(Refusal): replace(a,wire=raw,sha256=digest(raw))

    def test_missing_and_extra_body_are_refused(self):
        f=family(); a=authenticated(f)
        for ws in (f['witnesses'][:-1], f['witnesses']+(f['witnesses'][0],)):
            with self.subTest(count=len(ws)),self.assertRaises(Refusal):
                verify_accepted_history(f['graph_wire'],f['acceptances'],ws,a)

    def test_corrupted_body_not_stat_only(self):
        f=family(); a=authenticated(f); ws=list(f['witnesses']); ws[0]=replace(ws[0],wire=b'X'*len(ws[0].wire))
        with self.assertRaises(Refusal): verify_accepted_history(f['graph_wire'],f['acceptances'],ws,a)

    def test_custody_inode_replacement_after_read(self):
        f=family(); a=authenticated(f); ws=list(f['witnesses']); after=document(ws[0].after_json); after['inode']+=1
        ws[0]=replace(ws[0],after_json=canonical(after))
        with self.assertRaises(Refusal): verify_accepted_history(f['graph_wire'],f['acceptances'],ws,a)

    def test_acceptance_chain_wrong_version_or_missing_link(self):
        f=family(); a=authenticated(f)
        with self.assertRaises(Refusal): verify_accepted_history(f['graph_wire'],f['acceptances'][1:],f['witnesses'],a)
        rs=list(f['acceptances']); r=document(rs[1]); r['schema']='codex.monitor-custody-acceptance/v2'; rs[1]=canonical(r)
        f['graph']['namespaces'][1]['acceptance']={'sha256':digest(rs[1]),'bytes':len(rs[1])}; reseal_graph(f)
        with self.assertRaises(Refusal): verify_accepted_history(f['graph_wire'],tuple(rs),f['witnesses'],authenticated(f))

    def test_old_refs_roles_status_and_clocks_cannot_change(self):
        f=family(); a,h=verified(f)
        for field in ('remove','role','status','clock','ref'):
            d,c=candidate(f,a,h)
            if field=='remove': d['namespaces'].pop(0)
            if field=='role': d['namespaces'][0]['members'][0]['role']='other'
            if field=='status': d['namespaces'][1]['status']='normal_closed'
            if field=='clock': d['namespaces'][0]['clock']['deadline_unix']+=1
            if field=='ref': d['namespaces'][0]['members'][0]['custody']['sha256']='e'*64
            with self.subTest(field=field),self.assertRaises(Refusal): verify_candidate(canonical(d),h,f['new_witnesses'],a,c)

    def test_immediate_unknown_predecessor_cannot_normal_advance(self):
        f=family(('failed','normal_closed','unknown')); a,h=verified(f); d,c=candidate(f,a,h)
        with self.assertRaises(Refusal): verify_candidate(canonical(d),h,f['new_witnesses'],a,c)

    def test_new_namespace_alias_and_pending_acceptance_refused(self):
        f=family(); a,h=verified(f)
        for field in ('id','root','acceptance'):
            d,c=candidate(f,a,h)
            if field=='acceptance': d['namespaces'][-1][field]={'sha256':'a'*64,'bytes':1}
            else: d['namespaces'][-1][field]=d['namespaces'][0][field]
            with self.subTest(field=field),self.assertRaises(Refusal): verify_candidate(canonical(d),h,f['new_witnesses'],a,c)

    def test_physical_alias_with_different_paths_refused(self):
        f=family(); a,h=verified(f); d,c=candidate(f,a,h)
        new=d['namespaces'][-1]; new['members'][1]['custody']['inode']=new['members'][0]['custody']['inode']
        new['member_manifest_sha256']=digest(canonical(new['members']))
        ws=list(f['new_witnesses']); ws[1]=replace(ws[1],before_json=canonical(new['members'][1]['custody']),
                                                 after_json=canonical(new['members'][1]['custody']))
        c['transport_bytes']=len(canonical(d))+c['decoded_witness_bytes']
        with self.assertRaisesRegex(Refusal,'physical inode alias'): verify_candidate(canonical(d),h,ws,a,c)

    def test_custody_inside_old_namespace_refused(self):
        f=family(); a,h=verified(f); d,c=candidate(f,a,h); new=d['namespaces'][-1]
        new['members'][0]['custody']['path']=d['namespaces'][0]['root']+'/extra-custody.json'
        new['member_manifest_sha256']=digest(canonical(new['members'])); ws=list(f['new_witnesses'])
        ws[0]=replace(ws[0],before_json=canonical(new['members'][0]['custody']),after_json=canonical(new['members'][0]['custody']))
        c['transport_bytes']=len(canonical(d))+c['decoded_witness_bytes']
        with self.assertRaisesRegex(Refusal,'custody path overlaps'): verify_candidate(canonical(d),h,ws,a,c)

    def test_projection_refuses_before_consuming_witness_iterator(self):
        f=family(); a,h=verified(f); d,c=candidate(f,a,h); ns=d['namespaces'][-1]
        template=deepcopy(ns['members'][0])
        for i in range(5):
            m=deepcopy(template); m['relative_path']='z'+str(i)+'.bin'; m['role']='other'
            m['original'].update(path=ns['root']+'/'+m['relative_path'],inode=20000+i,bytes=8*1024**2)
            m['custody'].update(path='/custody/new4/'+m['relative_path'],inode=21000+i,bytes=8*1024**2)
            ns['members'].append(m)
        ns['member_manifest_sha256']=digest(canonical(ns['members']))
        touched=[]
        def forbidden_witnesses():
            touched.append(True)
            yield None
        with self.assertRaisesRegex(Refusal,'pre-witness cumulative decoded budget'): verify_candidate(canonical(d),h,forbidden_witnesses(),a,c)
        self.assertEqual(touched,[])

    def test_fresh_retained_transport_counters_separate(self):
        f=family(); a,h=verified(f); d,c=candidate(f,a,h)
        for key,wrong in (('fresh_read_bytes',0),('decoded_witness_bytes',0),('transport_bytes',0),
                          ('fresh_read_bytes',FRESH_LIMIT+1),('transport_bytes',TRANSPORT_LIMIT+1)):
            c2=dict(c); c2[key]=wrong
            with self.subTest(key=key,wrong=wrong),self.assertRaises(Refusal):
                verify_candidate(canonical(d),h,f['new_witnesses'],a,c2)

    def test_pure_verification_has_no_io_or_mutation(self):
        f=family(); before=canonical(f['graph'])
        with patch('builtins.open',side_effect=AssertionError('no file IO')), \
             patch('subprocess.run',side_effect=AssertionError('no process')), \
             patch('socket.socket',side_effect=AssertionError('no network')):
            a,h=verified(f); d,c=candidate(f,a,h); verify_candidate(canonical(d),h,f['new_witnesses'],a,c)
        self.assertEqual(before,canonical(f['graph']))

class PhaseCases(unittest.TestCase):
    def run_case(self,f):
        return verify_normal_phase(f['config'],f['events'],canonical(f['result']),f['snapshot'],f['binding'],f['phase'])

    def test_valid_dry_live_parity_summary(self):
        for phase in ('qualification','live'):
            f=phase_case(phase); out=self.run_case(f)
            self.assertEqual(out.event_count,4); self.assertEqual(out.verified_publications,1)
            self.assertEqual(out.event_head_sha256,f['result']['event_head_sha256'])
            self.assertFalse(hasattr(out,'physically_closed')); self.assertFalse(hasattr(out,'process_exit0'))

    def test_actual_legacy_history_descriptor_shape_not_boolean(self):
        f=phase_case(); legacy={'path':'/retained/flywheel_history_v1.jsonl','sha256':'b'*64,'head':'c'*64,'bytes':14}
        events=[json.loads(x) for x in f['events'].splitlines()]; events[0]['body']['legacy_history_readonly']=legacy
        f['events']=rechain(events); f['binding']=replace(f['binding'],prefixes=(),legacy_history_json=canonical(legacy))
        f['result']['event_head_sha256']=digest(f['events'].splitlines()[-1]); self.run_case(f)
        f['binding']=replace(f['binding'],legacy_history_json=b'true')
        with self.assertRaises(Refusal): self.run_case(f)

    def test_prefix_replacement_even_valid_rechain(self):
        f=phase_case(); events=[json.loads(x) for x in f['events'].splitlines()]; events[0]['utc_ns']+=1
        f['events']=rechain(events); f['result']['event_head_sha256']=digest(f['events'].splitlines()[-1])
        with self.assertRaises(Refusal): self.run_case(f)

    def test_clock_rewind_and_outside_endpoint(self):
        for change in ('rewind','wall_end','mono_end'):
            f=phase_case(); f['binding']=replace(f['binding'],prefixes=()); events=[json.loads(x) for x in f['events'].splitlines()]
            if change=='rewind': events[2]['monotonic_ns']=events[1]['monotonic_ns']-1
            if change=='wall_end': events[2]['utc_ns']=4600*10**9
            if change=='mono_end': events[2]['monotonic_ns']=4600*10**9
            f['events']=rechain(events); f['result']['event_head_sha256']=digest(f['events'].splitlines()[-1])
            with self.subTest(change=change),self.assertRaises(Refusal): self.run_case(f)

    def test_unknown_parity_failure_or_partial_prefix_not_normal(self):
        for kind in ('monitor.failure','parity.failure','unknown','partial'):
            f=phase_case(); f['binding']=replace(f['binding'],prefixes=()); events=[json.loads(x) for x in f['events'].splitlines()]
            if kind=='partial': events.pop()
            else: events[-1]['kind']=kind
            f['events']=rechain(events); f['result']['event_head_sha256']=digest(f['events'].splitlines()[-1])
            with self.subTest(kind=kind),self.assertRaises(Refusal): self.run_case(f)

    def test_false_body_boolean_attempt_and_nonhex_snapshot_refused(self):
        for change in ('body','bool_attempt','bad_hash','not_readonly'):
            f=phase_case(); f['binding']=replace(f['binding'],prefixes=()); events=[json.loads(x) for x in f['events'].splitlines()]
            if change=='body': events[0]['body']=['legacy_history_readonly']
            if change=='bool_attempt': events[2]['body']['attempt']=True
            if change=='bad_hash': events[3]['body']['snapshot']['sha256']='z'*64
            if change=='not_readonly': events[0]['body']['legacy_history_readonly']=True
            f['events']=rechain(events); f['result']['event_head_sha256']=digest(f['events'].splitlines()[-1])
            with self.subTest(change=change),self.assertRaises(Refusal): self.run_case(f)

    def test_result_numeric_counter_and_retry_refused(self):
        for key,value in (('GP_executions',1),('S3_PUT',True),('retry_allowed',True),('old_windows_mutated',True)):
            f=phase_case(); f['result'][key]=value
            with self.subTest(key=key),self.assertRaises(Refusal): self.run_case(f)

    def test_final_snapshot_body_and_provenance_not_echo(self):
        f=phase_case(); f['snapshot']=f['snapshot'].replace(b'fixture',b'changed')
        with self.assertRaises(Refusal): self.run_case(f)
        f=phase_case(); f['binding']=replace(f['binding'],source_manifest_sha256='a'*64)
        with self.assertRaises(Refusal): self.run_case(f)

    def test_result_head_custody_and_early_closure_refused(self):
        for change in ('head','custody','early'):
            f=phase_case()
            if change=='head': f['result']['event_head_sha256']='c'*64
            if change=='custody': f['result']['last_publication_custody']['attempt']=True
            if change=='early':
                events=[json.loads(x) for x in f['events'].splitlines()]
                for i,e in enumerate(events): e['utc_ns']=(4000+i)*10**9; e['monotonic_ns']=(4000+i)*10**9
                f['events']=rechain(events); f['result']['event_head_sha256']=digest(f['events'].splitlines()[-1])
                f['binding']=replace(f['binding'],prefixes=()); f['result']['at']=4100.0
            with self.subTest(change=change),self.assertRaises(Refusal): self.run_case(f)

    def test_original_config_bytes_and_snapshot_path_exact(self):
        f=phase_case(); f['config']=b' '+f['config']
        with self.assertRaises(Refusal): self.run_case(f)
        f=phase_case(); f['binding']=replace(f['binding'],snapshot_path='/different/snapshot.json')
        with self.assertRaises(Refusal): self.run_case(f)
