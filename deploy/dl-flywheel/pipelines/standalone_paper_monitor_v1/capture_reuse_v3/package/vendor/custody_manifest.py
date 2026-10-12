"""Complete accepted-history joins and a pending candidate manifest. Pure data only.

Witnesses are exact captured bytes with independently obtained same-FD before/after
metadata. They never prove current remote freshness. No body reader or IO callback
is accepted. The conservative per-invocation decoded witness total is 32 MiB.
"""
from dataclasses import dataclass, field
from .authority import VerifiedAuthority
from .common import (BODY_LIMIT, FRESH_LIMIT, RETAINED_LIMIT, TRANSPORT_LIMIT, ZERO,
                     absolute, canonical, clock, closed, content_pin, digest, document,
                     exact_body, file_pin, identifier, integer, need, relative, sha)

_HISTORY_VERIFIED = object()

ROLES = {'source', 'config', 'events', 'result', 'snapshot', 'failure', 'unknown', 'receipt', 'other'}
STATES = {'normal_closed', 'failed', 'unknown', 'held_source_only'}

@dataclass(frozen=True)
class BodyWitness:
    namespace: str
    relative_path: str
    wire: bytes
    before_json: bytes
    after_json: bytes

@dataclass(frozen=True)
class VerifiedHistory:
    graph_wire: bytes
    graph_sha256: str
    acceptance_head: str
    authority_sha256: str
    witness_bytes: int
    member_count: int
    _token: object = field(default=None, repr=False, compare=False)
    def __post_init__(self):
        expected = (digest(self.graph_wire), self.graph_sha256, self.acceptance_head,
                    self.authority_sha256, self.witness_bytes, self.member_count)
        need(type(self._token) is tuple and self._token == (_HISTORY_VERIFIED, expected)
             and digest(self.graph_wire) == self.graph_sha256, 'history immutable proof binding')

@dataclass(frozen=True)
class CandidateManifest:
    wire: bytes
    sha256: str
    reused_bytes: int
    fresh_bytes: int
    decoded_witness_bytes: int
    transport_bytes: int
    # Deliberately no accepted/admitted field or optimizer interface.

def _member(m, root):
    closed(m, ('relative_path', 'role', 'original', 'custody'), 'member fields')
    relative(m['relative_path']); need(type(m['role']) is str and m['role'] in ROLES, 'member role')
    original = file_pin(m['original']); custody = file_pin(m['custody'], immutable=True)
    need(original['path'] == root + '/' + m['relative_path'], 'original namespace path')
    need(original['sha256'] == custody['sha256'] and original['bytes'] == custody['bytes'],
         'custody preserves exact original bytes')
    need(custody['path'] != original['path'], 'separate captured-body custody')
    return m

def _namespace(ns, accepted):
    closed(ns, ('id', 'root', 'source_manifest_sha256', 'config_sha256', 'clock', 'status',
                'members', 'member_manifest_sha256', 'acceptance', 'window_state'), 'namespace fields')
    identifier(ns['id']); absolute(ns['root']); sha(ns['source_manifest_sha256']); sha(ns['member_manifest_sha256'])
    need(ns['window_state'] in ('frozen', 'not_created'), 'window creation state')
    if ns['window_state'] == 'not_created':
        need(accepted and ns['status'] in ('failed','held_source_only') and ns['config_sha256'] is None
             and ns['clock'] is None, 'prewindow custody must not invent config or clock')
    else:
        need(ns['status'] != 'held_source_only', 'held source-only window not created')
        sha(ns['config_sha256']); clock(ns['clock'])
    need(type(ns['status']) is str and (ns['status'] in STATES if accepted else ns['status'] == 'pending_verification'), 'namespace status')
    rows = ns['members']; need(type(rows) is list and 1 <= len(rows) <= 4096, 'complete bounded members')
    names = []; originals = set(); custodies = set()
    for m in rows:
        _member(m, ns['root']); names.append(m['relative_path'])
        need(m['original']['path'] not in originals and m['custody']['path'] not in custodies,
             'member physical alias or duplicate')
        originals.add(m['original']['path']); custodies.add(m['custody']['path'])
    need(names == sorted(set(names)), 'unique ordered member roster')
    need(digest(canonical(rows)) == ns['member_manifest_sha256'], 'full role/member commitment')
    need(any(m['role'] == 'source' and m['original']['sha256'] == ns['source_manifest_sha256'] for m in rows),
         'source manifest body retained')
    if ns['window_state'] == 'frozen':
        need(any(m['role'] == 'config' and m['original']['sha256'] == ns['config_sha256'] for m in rows),
             'original config body retained')
    else:
        need(not any(m['role'] == 'config' for m in rows), 'no created config in prewindow inventory')
    if ns['status'] == 'failed':
        need(any(m['role'] == 'failure' for m in rows), 'failure custody retained')
    if ns['status'] == 'unknown':
        need(any(m['role'] == 'unknown' for m in rows), 'unknown custody retained')
    if accepted:
        content_pin(ns['acceptance'])
    else:
        need(ns['acceptance'] is None, 'candidate cannot mint acceptance')
    return ns

def _no_aliases(namespaces):
    ids = set(); roots = []; originals = set(); custodies = set(); physical = set()
    for ns in namespaces:
        need(ns['id'] not in ids, 'namespace alias'); ids.add(ns['id'])
        need(all(ns['root'] != x and not ns['root'].startswith(x+'/') and
                 not x.startswith(ns['root']+'/') for x in roots), 'overlapping namespace')
        roots.append(ns['root'])
        for m in ns['members']:
            need(m['original']['path'] not in originals and m['custody']['path'] not in custodies,
                 'cross-namespace physical alias')
            for label in ('original', 'custody'):
                pair = (m[label]['device'], m[label]['inode'])
                need(pair not in physical, 'physical inode alias')
                physical.add(pair)
            originals.add(m['original']['path']); custodies.add(m['custody']['path'])
    need(not originals.intersection(custodies), 'custody path aliases original namespace')
    for path in custodies:
        need(all(path != root and not path.startswith(root+'/') and not root.startswith(path+'/')
                 for root in roots), 'custody path overlaps original namespace')

def _witnesses(namespaces, witnesses, prior_bytes=0):
    expected = {(ns['id'], m['relative_path']): m for ns in namespaces for m in ns['members']}
    projected = prior_bytes + sum(m['custody']['bytes'] for m in expected.values())
    need(projected <= RETAINED_LIMIT, 'pre-witness decoded budget')
    count = 0; seen = set()
    for w in witnesses:
        need(type(w) is BodyWitness, 'typed body witness, not callback success')
        key = (w.namespace, w.relative_path)
        need(key in expected and key not in seen, 'extra/duplicate body witness')
        m = expected[key]; before = file_pin(document(w.before_json), immutable=True)
        after = file_pin(document(w.after_json), immutable=True)
        need(before == after == m['custody'], 'captured-body physical replacement')
        exact_body(w.wire, m['custody'])
        seen.add(key); count += len(w.wire)
    need(seen == set(expected), 'missing exact captured body')
    return count, len(seen)

def verify_accepted_history(graph_wire, acceptance_wires, witnesses, authority):
    need(type(authority) is VerifiedAuthority, 'verified external authority required')
    a = authority.data(); need(digest(graph_wire) == a['predecessor_graph_sha256'], 'history rollback/replacement')
    graph = document(graph_wire)
    closed(graph, ('schema', 'generation_index', 'namespaces', 'acceptance_head'), 'accepted graph fields')
    need(graph['schema'] == 'codex.monitor-accepted-custody-graph/v1', 'accepted graph schema')
    integer(graph['generation_index'], 0, 999999, 'prior generation')
    need(graph['generation_index']+1 == a['generation_index'], 'exact next generation, no skipped ancestry')
    rows = graph['namespaces']; need(type(rows) is list and 1 <= len(rows) <= 128, 'bounded complete namespaces')
    for row in rows: _namespace(row, True)
    _no_aliases(rows)
    need(type(acceptance_wires) is tuple and len(acceptance_wires) == len(rows), 'complete acceptance chain')
    previous = ZERO
    for ns, wire in zip(rows, acceptance_wires):
        exact_body(wire, ns['acceptance']); acc = document(wire)
        closed(acc, ('schema', 'namespace', 'member_manifest_sha256', 'source_manifest_sha256',
                     'config_sha256', 'clock_sha256', 'status', 'previous_acceptance_sha256',
                     'capture', 'verifier_source', 'window_state'), 'acceptance receipt fields')
        need(acc['schema'] == 'codex.monitor-custody-acceptance/v1' and acc['namespace'] == ns['id'],
             'acceptance namespace')
        for field in ('member_manifest_sha256', 'source_manifest_sha256', 'config_sha256', 'status', 'window_state'):
            need(acc[field] == ns[field], 'acceptance identity/role/status join')
        need(acc['clock_sha256'] == digest(canonical(ns['clock'])) and
             acc['previous_acceptance_sha256'] == previous, 'acceptance clock/head chain')
        content_pin(acc['capture']); content_pin(acc['verifier_source'])
        previous = digest(wire)
    need(previous == graph['acceptance_head'] == a['predecessor_acceptance_sha256'], 'external acceptance head')
    size, count = _witnesses(rows, witnesses)
    return VerifiedHistory(graph_wire, digest(graph_wire), previous, authority.sha256, size, count,
                           (_HISTORY_VERIFIED, (digest(graph_wire), digest(graph_wire), previous,
                                                authority.sha256, size, count)))

def verify_candidate(candidate_wire, history, fresh_witnesses, authority, accounting):
    """Return a checked PENDING manifest, never publication/launch/admission authority.

Fresh counters are externally measured by a future qualified reader. This function
checks bounds/projections; it cannot attest that counter producers are truthful.
"""
    need(type(history) is VerifiedHistory and type(authority) is VerifiedAuthority, 'verified inputs required')
    history.__post_init__()
    a = authority.data(); old = document(history.graph_wire)
    expected_members = [m for ns in old['namespaces'] for m in ns['members']]
    need(sum(m['custody']['bytes'] for m in expected_members) == history.witness_bytes
         and len(expected_members) == history.member_count, 'derived history accounting')
    need(old['namespaces'][-1]['status'] == 'normal_closed', 'normal successor requires normal immediate predecessor')
    need(history.authority_sha256 == authority.sha256 and history.graph_sha256 == a['predecessor_graph_sha256']
         and history.acceptance_head == a['predecessor_acceptance_sha256'], 'same external history authority')
    d = document(candidate_wire)
    closed(d, ('schema', 'generation_index', 'authority_sha256', 'predecessor_graph_sha256',
               'predecessor_acceptance_sha256', 'namespaces'), 'candidate fields')
    need(d['schema'] == 'codex.monitor-pending-custody-extension/v1', 'pending extension schema')
    need(type(d['generation_index']) is int and d['generation_index'] == a['generation_index']
         and d['authority_sha256'] == authority.sha256, 'candidate authority/sequence')
    need(d['predecessor_graph_sha256'] == history.graph_sha256 and
         d['predecessor_acceptance_sha256'] == history.acceptance_head, 'candidate previous chain')
    rows = d['namespaces']; need(type(rows) is list and len(rows) == len(old['namespaces'])+1,
                                'exact one new namespace')
    need(canonical(rows[:-1]) == canonical(old['namespaces']), 'old body/role/state/clock/ref removal or mutation')
    new = _namespace(rows[-1], False); _no_aliases(rows)
    need({k:new[k] for k in a['new_namespace']} == a['new_namespace'] and new['clock'] == a['clock'],
         'new exact source/config/root/clock authority')
    closed(accounting, ('fresh_read_bytes', 'decoded_witness_bytes', 'transport_bytes'), 'counter fields')
    for key, limit in (('fresh_read_bytes', FRESH_LIMIT), ('decoded_witness_bytes', RETAINED_LIMIT),
                       ('transport_bytes', TRANSPORT_LIMIT)):
        integer(accounting[key], 0, limit, 'bounded '+key)
    projected = sum(m['custody']['bytes'] for m in new['members'])
    need(history.witness_bytes+projected <= RETAINED_LIMIT, 'pre-witness cumulative decoded budget')
    need(accounting['fresh_read_bytes'] >= projected and
         accounting['decoded_witness_bytes'] == history.witness_bytes+projected,
         'fresh/read/retained counters cannot undercount or flatten reuse')
    # Includes canonical manifest and exact bodies, never assumes base64 has no cost.
    need(accounting['transport_bytes'] >= len(candidate_wire)+accounting['decoded_witness_bytes'],
         'transport lower bound')
    size, _ = _witnesses([new], fresh_witnesses, history.witness_bytes)
    return CandidateManifest(candidate_wire, digest(candidate_wire), history.witness_bytes, size,
                             accounting['decoded_witness_bytes'], accounting['transport_bytes'])
