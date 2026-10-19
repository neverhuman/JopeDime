#!/usr/bin/env python3
"""Validate planning inventory; never produces an implementation PASS receipt."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-dir', type=Path)
parser.add_argument('--baseline-json', type=Path)
parser.add_argument('--baseline-ref', default='118f25cbdf3e148dc96abbaa2cbd5fb5c46090b6')
args = parser.parse_args()
here = Path(__file__).resolve().parent
tracker = json.loads((here/'MAJOR_RELEASE_TRACKER.json').read_text())
coverage_raw = (here/'GP_SPEC_COVERAGE.json').read_bytes()
coverage = json.loads(coverage_raw)
tasks = {x['id']:x for x in tracker['tasks']}
assert len(tasks) == len(tracker['tasks']) == tracker['total_tasks'] == 47
assert set(tasks) == {f'MR{i:02}' for i in range(1,47)} | {'MRFV'}
assert tracker['pinned_baseline_main_sha'] == 'e0e14ddb8646deff34e3c9d45b46cc236c2fdcc2'
if args.baseline_json:
    old = json.loads(args.baseline_json.read_text())
else:
    cmd = ['git','--no-optional-locks','-c','gc.auto=0','-c','maintenance.auto=false',
           'show',args.baseline_ref+':docs/plans/major-release-20261018/MAJOR_RELEASE_TRACKER.json']
    old = json.loads(subprocess.run(cmd,check=True,text=True,capture_output=True).stdout)
for before in old['tasks']:
    after = tasks[before['id']]
    assert after['owner_screen'] == before['owner_screen'], before['id']
    assert after['gate'] == before['gate'], before['id']
    assert after['release'] == before['release'], before['id']
    assert set(before['acceptance_cases']) <= set(after['acceptance_cases']), before['id']
    assert after['completion_criterion'] == before['completion_criterion'], before['id']
    if after['exact_acceptance_test'] != before['exact_acceptance_test']:
        if before['id'] == 'MR09':
            # SC-DEC-184Q: preserve the old blueprint and accept exactly this reviewed
            # real example route. This is not permission for arbitrary command changes.
            assert before['exact_acceptance_test'] == 'cargo run --release -p feat-math --bin gpr_cuda_bench -- --cases benchmarks/gpr_cuda_bench/cases.json --verify-envelope', 'MR09_before'
            assert after.get('baseline_blueprint_acceptance_test') == before['exact_acceptance_test'], 'MR09_blueprint_retained'
            assert after['exact_acceptance_test'] == 'cargo run --release -p jope-parallel --features neural --example gpr_cuda_bench -- --root "${JOPE_GPR_BENCH_ROOT:?set the admitted prepared-data root}" --cases benchmarks/gpr_cuda_bench/cases.json --output "${JOPE_GPR_BENCH_OUTPUT:?set a fresh receipt output path}" --device cuda:0 --verify-envelope', 'MR09_runtime_command'
            transition = after.get('acceptance_command_transition', {})
            assert transition.get('schema') == 'jope.acceptance-command-transition/v1', 'MR09_transition_schema'
            assert transition.get('decision') == 'SC-DEC-184Q', 'MR09_transition_authority'
            assert transition.get('decision_sha256') == '50ec2257659f1d1d03abc6da69956eb2b18596923d1bcba2b08fd36244279d9a', 'MR09_transition_authority_digest'
            assert transition.get('baseline_ref') == '118f25cbdf3e148dc96abbaa2cbd5fb5c46090b6', 'MR09_transition_baseline'
            runtime = 'crates/jope-engine/examples/gpr_cuda_bench.rs'
            assert transition.get('runtime_source') == runtime, 'MR09_runtime_path'
            assert transition.get('runtime_source_sha256') == hashlib.sha256((here.parents[2]/runtime).read_bytes()).hexdigest(), 'MR09_runtime_source_identity'
            assert after['status'] == 'partial', 'MR09_status_requires_new_qualification'
        else:
            assert after['exact_acceptance_test'] == before['exact_acceptance_test'].replace('-p jope-engine ','-p jope-parallel '), before['id']
    assert set(before['deps']) <= set(after['deps']), before['id']

visiting, visited = set(), set()
def visit(ident):
    assert ident in tasks, ident
    assert ident not in visiting, ('dependency_cycle',ident)
    if ident in visited:
        return
    visiting.add(ident)
    for d in tasks[ident]['deps']:
        visit(d)
    visiting.remove(ident)
    visited.add(ident)
for ident in tasks:
    visit(ident)

expected = {
    'spec_a':{f'P{i:02}' for i in range(20)},
    'spec_b':{f'GP-{i:02}' for i in range(26)},
    'spec_c':{f'WP{i:02}' for i in range(13)} | {'WP13A','WP13B'} | {f'WP{i:02}' for i in range(14,22)},
}
seen={k:set() for k in expected}
for row in coverage['packets']:
    key, packet = row['source'], row['packet']
    assert packet not in seen[key], (key,packet)
    seen[key].add(packet)
    assert hashlib.sha256(row['source_text'].encode()).hexdigest() == row['source_text_sha256']
    assert row['required_tasks_all'] and len(set(row['required_tasks_all'])) == len(row['required_tasks_all'])
    for ident in row['required_tasks_all']:
        assert packet in tasks[ident]['spec_mapping'][key].split(','), (key,packet,ident)
    if args.source_dir:
        original = args.source_dir/coverage['source_documents'][key]['name']
        raw = original.read_bytes()
        assert len(raw) == coverage['source_documents'][key]['bytes']
        assert hashlib.sha256(raw).hexdigest() == coverage['source_documents'][key]['sha256']
        a,b = row['lines']
        text = '\n'.join(raw.decode().splitlines()[a-1:b]).rstrip()
        assert text == row['source_text'], (key,packet,'source_lines')
assert seen == expected
assert len(coverage['packets']) == coverage['total_source_work_packets'] == 69
for ident,t in tasks.items():
    assert t['acceptance_cases'] and t['exact_acceptance_test']
    assert t['receipt_format']['verify_path'].endswith('/'+ident+'.json')
    assert t['receipt_format']['must_match_current_origin_main'] is True
    for key in expected:
        for packet in filter(None,(t['spec_mapping'].get(key) or '').split(',')):
            if packet not in expected[key] and not packet.startswith('Sec-'):
                assert re.fullmatch(r'(W\d{2}|E\d{2}|G\d{1,2})',packet), (ident,key,packet)
                continue
            if packet.startswith('Sec-'):
                assert ident == 'MR02' and key == 'spec_c' and packet == 'Sec-6.1'
                continue
            matching=[r for r in coverage['packets'] if r['source']==key and r['packet']==packet]
            assert len(matching)==1 and ident in matching[0]['required_tasks_all'], (ident,key,packet)
for ident in ['MR41','MR42','MR43','MR44','MR45','MR46','MRFV']:
    assert tasks[ident]['status'] == 'todo'
assert 'MR41' in tasks['MR30']['deps']
assert 'MR42' in tasks['MR31']['deps']
assert 'MR43' in tasks['MR33']['deps'] and 'MR43' in tasks['MR35']['deps']
assert 'MR44' in tasks['MR21']['deps']
md=(here/'MAJOR_RELEASE_TRACKER.md').read_text()
for ident in tasks:
    assert sum(line.startswith('| `'+ident+'` |') for line in md.splitlines()) == 1, ident
print(json.dumps({'scope_validation':'PASS','implementation_qualification':False,
                  'tasks':47,'prior_pr_tasks_retained':47,'source_packets':69,
                  'source_packet_counts':{k:len(v) for k,v in seen.items()},
                  'source_payload_hashes_checked':bool(args.source_dir),
                  'dependency_graph':'acyclic',
                  'coverage_file_sha256':hashlib.sha256(coverage_raw).hexdigest()},sort_keys=True))
