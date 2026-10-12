"""Authenticate a root seal supplied through an independently qualified trust boundary.

No file is opened here. ExternalAnchor is an explicit trust input, not a fact this
library discovers. Its bytes/physical identity must be read from the root-owned,
create-only authority channel, independently of the candidate request. The future
reader/operator and its freshness/ownership provenance remain unimplemented.
"""
from dataclasses import dataclass, field
from .common import (BODY_LIMIT, FRESH_LIMIT, RETAINED_LIMIT, TRANSPORT_LIMIT, SNAPSHOT_LIMIT,
                     EVENT_LIMIT, canonical, clock, closed, content_pin, digest, document,
                     exact_body, file_pin, integer, need, sha, identifier, absolute)

_VERIFIED = object()

LIMITS = {'fresh_read_bytes': FRESH_LIMIT, 'decoded_witness_bytes': RETAINED_LIMIT,
          'transport_bytes': TRANSPORT_LIMIT, 'body_bytes': BODY_LIMIT,
          'snapshot_bytes': SNAPSHOT_LIMIT, 'event_bytes': EVENT_LIMIT}
CAPS = {'cpu_quota_percent': 25, 'memory_max': 128*1024**2, 'memory_swap_max': 0,
        'nice': 19, 'tasks_max': 24, 'restart': 'no', 'gpu_access': 'metadata_only'}
FRESH_GUARDS = ('current_source', 'current_units_cgroups', 'current_resource_roster',
                'current_device_policy', 'current_foreign_boundaries',
                'current_activation_chain', 'current_clock')

@dataclass(frozen=True)
class ExternalAnchor:
    """Externally committed pin, not constructed from the candidate authority."""
    seal_pin_json: bytes
    generation_index: int
    predecessor_graph_sha256: str
    predecessor_acceptance_sha256: str
    policy_sha256: str

@dataclass(frozen=True)
class VerifiedAuthority:
    wire: bytes
    sha256: str
    external_anchor_sha256: str
    _token: object = field(default=None, repr=False, compare=False)
    def __post_init__(self):
        expected = (digest(self.wire), self.sha256, self.external_anchor_sha256)
        need(type(self._token) is tuple and self._token == (_VERIFIED, expected)
             and digest(self.wire) == self.sha256, 'authority immutable proof binding')
    # This is pure-data authentication, never runtime admission.
    def data(self):
        self.__post_init__()
        return document(self.wire)

def verify_authority(wire, anchor, observed_before, observed_after):
    need(type(anchor) is ExternalAnchor, 'external authority anchor required')
    pin = file_pin(document(anchor.seal_pin_json), immutable=True)
    file_pin(observed_before, immutable=True); file_pin(observed_after, immutable=True)
    need(observed_before == observed_after == pin, 'external seal physical identity drift')
    exact_body(wire, pin)
    a = document(wire)
    closed(a, ('schema', 'generation_index', 'predecessor_graph_sha256',
               'predecessor_acceptance_sha256', 'engine_source', 'collector_source',
               'policy', 'limits', 'fresh_guards', 'new_namespace', 'clock'), 'authority fields')
    need(a['schema'] == 'codex.monitor-incremental-authority/v1', 'authority schema')
    integer(anchor.generation_index, 1, 1000000, 'external generation')
    integer(a['generation_index'], 1, 1000000, 'authority generation')
    need(a['generation_index'] == anchor.generation_index, 'generation rollback')
    for field in ('predecessor_graph_sha256', 'predecessor_acceptance_sha256'):
        sha(a[field]); sha(getattr(anchor, field))
        need(a[field] == getattr(anchor, field), 'external predecessor mismatch')
    content_pin(a['engine_source']); content_pin(a['collector_source'])
    policy = a['policy']
    closed(policy, ('caps', 'resource_roster_sha256', 'device_policy_sha256',
                    'foreign_boundaries_sha256', 'activation_chain_sha256',
                    'boundary_count'), 'policy fields')
    need(canonical(policy['caps']) == canonical(CAPS) and type(policy['boundary_count']) is int and
         policy['boundary_count'] == 5, 'unchanged caps and five foreign boundaries')
    for key in ('resource_roster_sha256', 'device_policy_sha256',
                'foreign_boundaries_sha256', 'activation_chain_sha256'):
        sha(policy[key])
    sha(anchor.policy_sha256)
    need(digest(canonical(policy)) == anchor.policy_sha256, 'externally committed policy')
    need(a['limits'] == LIMITS and all(type(x) is int for x in a['limits'].values()), 'fixed limits')
    need(a['fresh_guards'] == list(FRESH_GUARDS), 'all current guards must stay fresh')
    closed(a['new_namespace'], ('id', 'root', 'source_manifest_sha256', 'config_sha256'), 'new namespace')
    identifier(a['new_namespace']['id']); absolute(a['new_namespace']['root'])
    sha(a['new_namespace']['source_manifest_sha256']); sha(a['new_namespace']['config_sha256'])
    clock(a['clock'])
    anchor_wire = canonical({'seal_pin': pin, 'generation_index': anchor.generation_index,
                             'predecessor_graph_sha256': anchor.predecessor_graph_sha256,
                             'predecessor_acceptance_sha256': anchor.predecessor_acceptance_sha256,
                             'policy_sha256': anchor.policy_sha256})
    return VerifiedAuthority(wire, digest(wire), digest(anchor_wire),
                             (_VERIFIED, (digest(wire), digest(wire), digest(anchor_wire))))
