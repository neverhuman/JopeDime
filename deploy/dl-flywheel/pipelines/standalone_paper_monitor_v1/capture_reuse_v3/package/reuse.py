"""Closed accepted-custody certificates. No historical body reread is claimed.

ExternalSeal is privileged operator input, never a request field. The source
cannot establish that the operator obtained it independently: actual authority
producer/reader qualification remains a separate mandatory gate.
"""
from dataclasses import dataclass, field
from .vendor.common import (BODY_LIMIT, FRESH_LIMIT, METADATA_LIMIT, RETAINED_LIMIT,
    TRANSPORT_LIMIT, ZERO, absolute, canonical, clock, closed, content_pin, digest,
    document, exact_body, file_pin, integer, need, number, sha)
from .vendor.custody_manifest import _namespace, _no_aliases
from .vendor.authority import CAPS, FRESH_GUARDS
from .vendor.phase_verifier import verify_normal_phase, PhaseBinding
from .framing import strict_lf_lines

_TOKEN = object()
POLICY = {'name':'accepted-historical-custody-reference/v1',
    'historical_freshness':'accepted_at_original_capture_not_reread_now',
    'later_body_use':'exact_bounded_same_fd_rehash_or_hold_no_retry',
    'new_artifacts':'fresh_exact_bytes', 'fresh_read_limit':FRESH_LIMIT,
    'retained_limit':RETAINED_LIMIT, 'transport_limit':TRANSPORT_LIMIT,
    'body_limit':BODY_LIMIT, 'caps':CAPS, 'fresh_guards':list(FRESH_GUARDS)}

@dataclass(frozen=True)
class ExternalSeal:
    physical_pin_json: bytes
    generation: int
    graph_sha256: str
    acceptance_head_sha256: str
    policy_sha256: str

@dataclass(frozen=True)
class ReusedHistory:
    graph_wire: bytes
    seal_wire: bytes
    seal_sha256: str
    historical_bytes_not_read: int
    original_member_count: int
    metadata_bytes_verified: int
    _token: object = field(default=None, repr=False, compare=False)
    def check(self):
        values=(digest(self.graph_wire),digest(self.seal_wire),self.seal_sha256,
                self.historical_bytes_not_read,self.original_member_count,self.metadata_bytes_verified)
        need(self._token == (_TOKEN, values), 'immutable reuse proof binding')
        return document(self.seal_wire),document(self.graph_wire)

def authenticate_history(seal_wire, external, before, after, graph_wire,
                         acceptance_wires, evidence_wires):
    """Only external RootSeal can authorize interpreting prior acceptance as reuse.

    evidence_wires is exact closed metadata custody, not arbitrary callbacks. Raw
    original capture/review bodies remain pinned and verified. Their semantic
    translation to certificate records must be independently reviewed by the
    future producer before the root externally commits this seal.
    """
    need(type(external) is ExternalSeal, 'external root channel required')
    external_pin=file_pin(document(external.physical_pin_json),immutable=True)
    need(before==after==external_pin, 'external root seal identity changed')
    exact_body(seal_wire,external_pin)
    s=document(seal_wire)
    closed(s,('schema','generation','graph','acceptance_head','policy','current',
              'evidence','producer_source','producer_review','root_commitment'), 'root seal fields')
    need(s['schema']=='codex.monitor-capture-reuse-root-seal/v1', 'root seal schema')
    integer(external.generation,1,1000000,'external generation')
    need(type(s['generation']) is int and s['generation']==external.generation,'generation rollback')
    need(canonical(s['policy'])==canonical(POLICY) and
         digest(canonical(s['policy']))==external.policy_sha256,'exact externally committed reuse disposition')
    content_pin(s['graph'], METADATA_LIMIT); exact_body(graph_wire,s['graph'])
    need(s['graph']['sha256']==external.graph_sha256 and s['acceptance_head']==external.acceptance_head_sha256,'external history/head authority')
    for k in ('producer_source','producer_review','root_commitment'):content_pin(s[k],METADATA_LIMIT)
    current=s['current'];closed(current,('namespace','root','source_manifest_sha256','config_sha256','clock','snapshot_path','guards','inventory_sha256'),'current authority')
    absolute(current['root']);absolute(current['snapshot_path']);sha(current['source_manifest_sha256']);sha(current['config_sha256']);sha(current['inventory_sha256']);clock(current['clock'])
    need(type(current['namespace'])is str and current['namespace'],'current namespace')
    need(type(current['guards'])is dict and set(current['guards'])==set(FRESH_GUARDS),'all fresh guard evidence required')
    for r in current['guards'].values():content_pin(r,METADATA_LIMIT)
    g=document(graph_wire);closed(g,('schema','generation_index','namespaces','acceptance_head'),'complete graph')
    need(g['schema']=='codex.monitor-accepted-custody-graph/v1' and type(g['generation_index'])is int and g['generation_index']+1==s['generation'],'exact graph generation')
    rows=g['namespaces'];need(type(rows)is list and 1<=len(rows)<=128,'namespace bound')
    for ns in rows:_namespace(ns,True)
    _no_aliases(rows)
    need(rows[-1]['status']=='normal_closed','active/failed/unknown immediate predecessor cannot be normal closure')
    need(all(ns['id']!=current['namespace'] and current['root']!=ns['root'] and
        not current['root'].startswith(ns['root']+'/') and not ns['root'].startswith(current['root']+'/') for ns in rows),'fresh namespace and no history overlap')
    need(type(acceptance_wires)is tuple and len(acceptance_wires)==len(rows),'all acceptance receipts')
    need(type(evidence_wires)is dict and set(evidence_wires)==set(s['evidence']),'complete exact metadata evidence roster')
    metadata=len(seal_wire)+len(graph_wire)+sum(len(w)for w in acceptance_wires)+sum(len(w)for w in evidence_wires.values())
    need(metadata<=FRESH_LIMIT and metadata<=RETAINED_LIMIT,'metadata pre-consumption aggregate')
    for key,ref in s['evidence'].items():
        sha(key);content_pin(ref,METADATA_LIMIT);need(key==ref['sha256'],'content-addressed evidence identity');exact_body(evidence_wires[key],ref)
    required={s[k]['sha256'] for k in ('producer_source','producer_review','root_commitment')}
    required.update(r['sha256']for r in current['guards'].values())
    for r in [s[k]for k in ('producer_source','producer_review','root_commitment')]+list(current['guards'].values()):need(s['evidence'].get(r['sha256'])==r,'exact root producer/guard provenance pin joins')
    previous=ZERO;count=0;historical=0
    for ns,wire in zip(rows,acceptance_wires):
        exact_body(wire,ns['acceptance']);acc=document(wire)
        closed(acc,('schema','namespace','member_manifest_sha256','source_manifest_sha256','config_sha256','clock_sha256','status','previous_acceptance_sha256','capture','verifier_source','window_state','independent_review','source_inventory_sha256','terminal_capture'),'reuse acceptance fields')
        need(acc['schema']=='codex.monitor-capture-reuse-acceptance/v1' and acc['namespace']==ns['id'],'acceptance namespace schema')
        for k in ('member_manifest_sha256','source_manifest_sha256','config_sha256','status','window_state'):need(acc[k]==ns[k],'acceptance exact member/role/status identity')
        need(acc['clock_sha256']==digest(canonical(ns['clock'])) and acc['previous_acceptance_sha256']==previous,'original clock and acceptance chain')
        source_members=[m for m in ns['members'] if m['role']=='source']
        need(acc['source_inventory_sha256']==digest(canonical(source_members)),'complete source inventory commitment')
        for k in ('capture','verifier_source','independent_review'):
            content_pin(acc[k],METADATA_LIMIT);required.add(acc[k]['sha256'])
            need(s['evidence'].get(acc[k]['sha256'])==acc[k],'capture/review/verifier evidence remains exact')
        if ns['status']=='normal_closed':
            tc=acc['terminal_capture'];closed(tc,('captured_unix','completed_monotonic_ns','boot_id','fsync_receipt','physical_closure_receipt','event_bytes_pin','result_pin'),'normal terminal durability certificate')
            cl=ns['clock'];need(tc['boot_id']==cl['boot_id'],'terminal boot')
            need(number(cl['created_unix'])<=number(tc['captured_unix'])<=number(cl['deadline_unix']),'terminal capture before original deadline')
            integer(tc['completed_monotonic_ns'],cl['created_monotonic_ns'],cl['deadline_monotonic_ns'],'durable terminal paired endpoint')
            for k in ('fsync_receipt','physical_closure_receipt','event_bytes_pin','result_pin'):content_pin(tc[k],METADATA_LIMIT)
            for k in ('fsync_receipt','physical_closure_receipt'):
                required.add(tc[k]['sha256']);need(s['evidence'].get(tc[k]['sha256'])==tc[k],'durability/closure evidence is present')
            need(any(m['role']=='events'and all(m['original'][k]==tc['event_bytes_pin'][k]for k in ('sha256','bytes'))for m in ns['members']),'whole physical terminal event bytes retained')
            need(any(m['role']=='result'and all(m['original'][k]==tc['result_pin'][k]for k in ('sha256','bytes'))for m in ns['members']),'terminal result bytes retained')
        else:need(acc['terminal_capture']is None,'failed/held/unknown never normal certificate')
        previous=digest(wire);count+=len(ns['members']);historical+=sum(m['original']['bytes']for m in ns['members'])
    need(required==set(evidence_wires),'no missing or unexplained metadata provenance')
    need(previous==g['acceptance_head']==s['acceptance_head'],'full anchored acceptance chain')
    need(current['clock']['boot_id']==rows[-1]['clock']['boot_id'] and number(current['clock']['created_unix'])>=number(acc['terminal_capture']['captured_unix']) and current['clock']['created_monotonic_ns']>=acc['terminal_capture']['completed_monotonic_ns'],'fresh generation follows actual same-boot terminal, no old clock reset')
    values=(digest(graph_wire),digest(seal_wire),digest(seal_wire),historical,count,metadata)
    return ReusedHistory(graph_wire,seal_wire,digest(seal_wire),historical,count,metadata,(_TOKEN,values))

def verify_fresh_manifest(request_wire,history):
    """Validate a fresh-only pending capture plan before any body reader is opened."""
    need(type(history)is ReusedHistory,'authenticated history required');s,g=history.check()
    d=document(request_wire);closed(d,('schema','seal_sha256','generation','predecessor_graph_sha256','predecessor_acceptance_head','current','members'),'request closed; no caller-supplied anchor')
    need(d['schema']=='codex.monitor-fresh-capture-plan/v1' and d['seal_sha256']==history.seal_sha256 and type(d['generation'])is int and d['generation']==s['generation'],'fresh request seal/generation')
    need(d['predecessor_graph_sha256']==digest(history.graph_wire) and d['predecessor_acceptance_head']==s['acceptance_head'] and d['current']==s['current'],'exact preserved parent/current authority')
    rows=d['members'];need(type(rows)is list and 1<=len(rows)<=4096,'fresh member bound')
    need(digest(canonical(rows))==s['current']['inventory_sha256'],'complete external current inventory; no omitted or invented fresh member')
    paths=[];nodes=set();total=history.metadata_bytes_verified+len(request_wire)
    for r in rows:
        file_pin(r);need(r['path'].startswith(s['current']['root']+'/')or r['path']==s['current']['snapshot_path'],'only fresh current artifact root or exact singleton')
        need(r['path']not in paths and(r['device'],r['inode'])not in nodes,'fresh physical aliases refused');paths.append(r['path']);nodes.add((r['device'],r['inode']));total+=r['bytes']
    need(paths==sorted(paths),'canonical unique fresh roster')
    need(any(r['sha256']==s['current']['source_manifest_sha256']for r in rows)and any(r['sha256']==s['current']['config_sha256']for r in rows),'fresh actual source and config bodies required')
    need(total<=FRESH_LIMIT and total<=RETAINED_LIMIT,'complete fresh pre-I/O aggregate; history refs not body charges')
    transport=sum(4*((r['bytes']+2)//3)for r in rows)+len(request_wire)+history.metadata_bytes_verified
    need(transport<=TRANSPORT_LIMIT,'pre-serialization base64 transport lower bound')
    return {'state':'PENDING_FRESH_BODY_VERIFICATION','fresh_projected_bytes':total,'transport_payload_lower_bound':transport,
            'historical_bytes_reused_not_read':history.historical_bytes_not_read,'historical_member_count':history.original_member_count,'manifest':d}

def verify_normal_current(config_wire,event_wire,result_wire,snapshot_wire,binding,phase):
    need(type(event_wire)is bytes and 0<len(event_wire)<=512*1024,'bounded whole event bytes before framing')
    strict_lf_lines(event_wire)
    return verify_normal_phase(config_wire,event_wire,result_wire,snapshot_wire,binding,phase)
