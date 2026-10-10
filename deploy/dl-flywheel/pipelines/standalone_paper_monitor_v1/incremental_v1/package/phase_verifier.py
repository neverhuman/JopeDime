"""Pure replay of a complete original normal phase, with explicit tightening.

No failure-prefix bypass, IO, worker import, current clock, exit inference or
runtime admission. Config/source/clock/prefix pins come from a separately
verified namespace. Historical event schema has no terminal monotonic timestamp.
"""
from dataclasses import dataclass
from .common import (EVENT_LIMIT, SNAPSHOT_LIMIT, ZERO, canonical, clock, closed, content_pin,
                     digest, document, exact_body, integer, need, number, sha, absolute)

@dataclass(frozen=True)
class PhaseBinding:
    config_sha256: str
    source_manifest_sha256: str
    clock_json: bytes
    snapshot_path: str
    legacy_history_json: bytes
    # Exact accepted/observed prefixes: (number of complete lines, SHA of lines + LF).
    prefixes: tuple

@dataclass(frozen=True)
class PhaseProof:
    phase: str
    event_count: int
    event_head_sha256: str
    verified_publications: int
    final_snapshot_json: bytes
    result_sha256: str
    config_sha256: str
    source_manifest_sha256: str
    terminal_at: str
    last_event_monotonic_ns: int
    # Deliberately absent: process_exit0, physically_closed, admission_allowed.

def verify_normal_phase(config_wire, event_wire, result_wire, final_snapshot_wire, binding, phase):
    need(type(binding) is PhaseBinding and phase in ('qualification', 'live'), 'typed normal phase binding')
    sha(binding.config_sha256); sha(binding.source_manifest_sha256); absolute(binding.snapshot_path)
    expected_clock = clock(document(binding.clock_json))
    legacy = document(binding.legacy_history_json)
    if legacy is not None:
        closed(legacy, ('path','sha256','head','bytes'), 'legacy history descriptor')
        absolute(legacy['path']); sha(legacy['sha256']); sha(legacy['head'])
        integer(legacy['bytes'], 0, 8*1024**2, 'legacy history bytes')
    need(digest(config_wire) == binding.config_sha256, 'exact original config bytes')
    config = document(config_wire, canonical_required=False)
    need(type(config) is dict and all(config.get(k) == v and type(config.get(k)) is type(v)
                                    for k,v in expected_clock.items()), 'original clock joins')
    need(config.get('source_manifest_sha256') == binding.source_manifest_sha256, 'original source join')
    need(type(event_wire) is bytes and 0 < len(event_wire) <= EVENT_LIMIT and event_wire.endswith(b'\n'),
         'bounded complete event stream')
    lines = event_wire.splitlines(); need(4 <= len(lines) <= 1024, 'complete event count')
    need(type(binding.prefixes) is tuple, 'immutable prefix roster')
    seen_prefixes = set()
    for row in binding.prefixes:
        need(type(row) is tuple and len(row) == 2, 'prefix tuple')
        count, expected = row; integer(count, 1, len(lines), 'accepted prefix length'); sha(expected)
        need(count not in seen_prefixes, 'duplicate prefix'); seen_prefixes.add(count)
        need(digest(b'\n'.join(lines[:count])+b'\n') == expected, 'accepted prefix replaced')
    previous=ZERO; state='start'; attempt=0; verified=0; last=None; wall=None; mono=None
    for seq, line in enumerate(lines, 1):
        need(0 < len(line) < 65536, 'event line bound'); e=document(line, 65535)
        closed(e, ('seq','previous_sha256','kind','utc_ns','monotonic_ns','config_sha256','body'), 'event fields')
        need(type(e['seq']) is int and e['seq'] == seq and e['previous_sha256'] == previous and
             e['config_sha256'] == binding.config_sha256, 'event head/sequence/config')
        wn=integer(e['utc_ns'], 0, 2**63-1, 'event wall'); mn=integer(e['monotonic_ns'], 1, 2**63-1, 'event mono')
        need(number(expected_clock['created_unix'])*10**9 <= wn < number(expected_clock['deadline_unix'])*10**9
             and expected_clock['created_monotonic_ns'] <= mn < expected_clock['deadline_monotonic_ns'],
             'event outside original paired clock')
        need((wall is None or wn >= wall) and (mono is None or mn >= mono), 'clock rewind')
        wall,mono=wn,mn; previous=digest(line); body=e['body']; kind=e['kind']
        need(type(body) is dict, 'event body object')
        if state == 'start':
            closed(body, ('legacy_history_readonly',), 'start fields')
            need(kind == 'monitor.started' and canonical(body['legacy_history_readonly']) == binding.legacy_history_json,
                 'exact actual legacy history descriptor, not boolean authority')
            state='ready'
        elif state == 'ready':
            closed(body, ('attempt',), 'collection fields')
            need(kind == 'collection.attempt' and type(body['attempt']) is int and body['attempt'] == attempt+1,
                 'one next collection'); attempt+=1; state='collecting'
        elif state == 'collecting':
            closed(body, ('attempt','qualification_only'), 'publication fields')
            need(kind == 'publication.attempt' and type(body['attempt']) is int and body['attempt'] == attempt
                 and type(body['qualification_only']) is bool and body['qualification_only'] == (phase=='qualification'),
                 'exact phase publication'); state='publishing'
        else:
            closed(body, ('attempt','snapshot','errors'), 'verified fields')
            need(state == 'publishing' and kind == 'publication.verified' and type(body['attempt']) is int
                 and body['attempt'] == attempt and body['errors'] == [], 'no failure or unknown publication')
            snapshot=body['snapshot']; closed(snapshot, ('path','sha256','bytes'), 'snapshot fields')
            need(snapshot['path'] == binding.snapshot_path, 'exact snapshot path')
            content_pin({k:snapshot[k] for k in ('sha256','bytes')}, SNAPSHOT_LIMIT)
            last=snapshot; verified+=1; state='ready'
    need(state == 'ready' and attempt == verified and verified > 0 and (phase!='qualification' or verified==1),
         'complete phase without outstanding attempts')
    r=document(result_wire, canonical_required=False)
    closed(r, ('schema','state','at','config_sha256','attempts','last_verified_snapshot','last_publication_custody',
               'event_count','event_head_sha256','optimizer_updates','GP_executions','S3_PUT','old_windows_mutated',
               'retry_allowed'), 'result fields')
    expected='qualification_pass' if phase=='qualification' else 'window_closed_before_next_collection'
    need(r['schema']=='jope.paper-monitor-result/v1' and r['state']==expected and
         r['config_sha256']==binding.config_sha256, 'clean exact result')
    need(type(r['attempts']) is int and r['attempts']==verified and type(r['event_count']) is int
         and r['event_count']==len(lines) and r['event_head_sha256']==previous and r['last_verified_snapshot']==last,
         'derived result joins')
    custody=r['last_publication_custody']; closed(custody, ('attempt','phase','snapshot_may_have_been_replaced',
                                                          'attempted_snapshot'), 'custody fields')
    need(type(custody['attempt']) is int and custody['attempt']==verified and custody['phase']=='verified'
         and custody['snapshot_may_have_been_replaced'] is True and custody['attempted_snapshot']==last,
         'verified final custody')
    need(all(type(r[k]) is int and r[k]==0 for k in ('optimizer_updates','GP_executions','S3_PUT'))
         and r['old_windows_mutated'] is False and r['retry_allowed'] is False, 'zero science and no retries')
    terminal=number(r['at']); deadline=number(expected_clock['deadline_unix'])
    need(number(wall)/10**9 <= terminal < deadline, 'terminal after publication within original wall clock')
    if phase=='live': need(0 < deadline-terminal < 100, 'original near-end normal closure')
    exact_body(final_snapshot_wire, last)
    snapshot=document(final_snapshot_wire, SNAPSHOT_LIMIT, canonical_required=False)
    need(type(snapshot) is dict and type(snapshot.get('standalone_monitor')) is dict, 'snapshot provenance')
    provenance=snapshot['standalone_monitor']
    need(provenance.get('config_sha256')==binding.config_sha256 and
         provenance.get('source_manifest_sha256')==binding.source_manifest_sha256 and
         type(provenance.get('attempt')) is int and provenance['attempt']==verified, 'snapshot config/source/attempt')
    return PhaseProof(phase,len(lines),previous,verified,canonical(last),digest(result_wire),binding.config_sha256,
                      binding.source_manifest_sha256,str(terminal),mono)
