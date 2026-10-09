"""Exact selected support, fresh authoritative roles; no partition reassignment."""
import json
import time
from pathlib import Path
from jope_flywheel.program_io import bound, read, need, tick
from jope_flywheel.protocol import digest


class AuthorityLease:
    def __init__(self, config):
        self.config = config

    def check(self, identity=None, parent=None):
        c = self.config
        tick(c['authority_deadline_unix'])
        selected = c['selection']
        need(c['selected_count'] == len(selected) == 128 and
             len({x['identity'] for x in selected}) == 128 and
             len({x['parent_lineage_sha256'] for x in selected}) == 128 and
             digest(selected) == c['selection_sha256'], 'exact selected128 source roster')
        pointer = json.loads(read(c['authority_pointer'], 4096))
        p, root = Path(pointer['path']), Path(c['authority_root'])
        need(p.parent == root/'proofs' and p.name == pointer['sha256']+'.json',
             'content-addressed current authority')
        proof = json.loads(bound(pointer, 2*1024**2))
        need(proof['schema'] == 'jope.fresh-variant-gp-authority/v1' and
             proof['passed'] is True and proof['host'] == 'xbabe2' and
             proof['campaign_sha256'] == c['campaign_sha256'] and
             proof['selection_sha256'] == c['selection_sha256'], 'authority campaign identity')
        need(0 <= time.time()-proof['checked_at_unix'] < 120 and
             proof['deadline_unix'] == c['authority_deadline_unix'], 'stale/future authority')
        need(proof['generator_config_sha256'] == c['generator_config']['sha256'] and
             proof['parent_registry_sha256'] == c['parent_registry_sha256'] and
             proof['authority_pins'] == c['authority_pins'], 'authority source pins')
        support = {x['identity']: x for x in proof['support']}
        need(len(support) == len(proof['support']) == 128 and
             set(support) == {x['identity'] for x in selected}, 'authority exact support')
        for v in selected:
            r = support[v['identity']]
            need(r['parent_lineage_sha256'] == v['parent_lineage_sha256'] and
                 r['partition'] == v['partition'] and r['partition'] in ('train', 'validation') and
                 type(r['protected_overlap']) is int and r['protected_overlap'] == 0,
                 'protected or reassigned development role')
        if identity is not None:
            r = support.get(identity)
            need(r is not None and parent is not None and
                 r['parent_lineage_sha256'] == parent['parent_lineage_sha256'] and
                 r['parent_record_sha256'] == digest(parent) and r['view_id'] == parent['view_id'] and
                 r['partition'] == parent['partition'], 'exact inherited parent role')
        return proof
