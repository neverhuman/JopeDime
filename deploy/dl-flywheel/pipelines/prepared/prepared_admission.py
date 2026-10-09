"""Pinned CPU materialization, consumed only at a reconciled epoch boundary.

This module owns no training process, GPU, network client, ready pointer or
optimizer. The consumer must hold exclusive training ownership during adoption.
"""
import json
from pathlib import Path
from jope_flywheel.program_io import bound, need, sha, read
from jope_flywheel.protocol import canonical
import prepared_snapshot as P
from jope_flywheel import native_variant_snapshot as V


def candidate(descriptor, current, native, *, campaign_sha256,
              preprocessing_sha256, registry_sha256, authority_check):
    need(set(descriptor) == {'schema', 'campaign_sha256', 'approved_for_bounded_research',
                            'preparation_config', 'preparation_status', 'prepared'},
         'closed prepared candidate descriptor')
    need(descriptor['schema'] == 'jope.prepared-epoch-candidate/v1' and
         descriptor['campaign_sha256'] == campaign_sha256 and
         descriptor['approved_for_bounded_research'] is True,
         'explicit matching campaign admission permit')
    config_raw = bound(descriptor['preparation_config'])
    status_raw = bound(descriptor['preparation_status'])
    c = json.loads(config_raw); receipt = json.loads(status_raw)
    root = Path(c['owned_root'])
    need(root.is_absolute() and c['schema'] == 'jope.prepared-mixed-snapshot-config/v1',
         'owned CPU preparation schema')
    for ref, name in ((descriptor['preparation_config'], 'config.json'),
                      (descriptor['preparation_status'], 'status.json'),
                      (descriptor['prepared'], 'prepared.json')):
        need(Path(ref['path']) == root / name, 'one immutable preparation root')
    need(receipt['config_sha256'] == sha(config_raw) and
         receipt['prepared'] == descriptor['prepared'] and
         receipt['extension_sha256'] == c['extension_sha256'],
         'pinned successful receipt and configuration closure')
    need(all(c[k] == receipt[k] == 0 for k in
             ('network_gets', 's3_writes', 'optimizer_updates', 'planner_mutations')),
         'preparation cannot mutate optimizer or admission')
    for rel, pin in c['owned_sources'].items():
        p = Path(rel)
        need(not p.is_absolute() and '..' not in p.parts and
             sha(read(root / p, 32 * 1024**2)) == pin,
             'exact preparation source closure')
    need(receipt['prepared_source_sha256'] == c['owned_sources']['prepared_snapshot.py'] ==
         sha(read(Path(P.__file__).resolve())), 'qualified hydration implementation')
    need(receipt['source_adapter_sha256'] == sha(read(Path(V.__file__).resolve())),
         'qualified measured record adapter')
    extension = read(root / 'extension.json', 32 * 1024**2)
    need(sha(extension) == c['extension_sha256'], 'pinned extension')
    old = json.loads(current.extension_raw); new = json.loads(extension)
    rows = {t['variant_identity']: t for t in new['tasks']}
    need(len(rows) == len(new['tasks']), 'unique variant tasks')
    for task in old['tasks']:
        need(task['variant_identity'] in rows, 'retained variant task')
        row = rows[task['variant_identity']]
        need(all(row.get(k) == task[k] for k in set(task) - {'program_results'}) and
             all(r in row['program_results'] for r in task['program_results']),
             'unchanged prior inputs, roles and exact measured result refs')
    prepared = bound(descriptor['prepared'], 32 * 1024**2)
    result = P.load(prepared, native, receipt=receipt,
                    expected_extension_sha256=c['extension_sha256'],
                    expected_preprocessing_sha256=preprocessing_sha256,
                    expected_parent_registry_sha256=registry_sha256,
                    authority_check=authority_check)
    need(set(current.contexts) <= set(result.contexts) and
         all(current.contexts[k] == result.contexts[k] for k in current.contexts),
         'unchanged existing context descriptors and lineage roles')
    return result


def adopt(planner, old, new, *, checkpoint_sha256, authority_check):
    """No optimizer operation; only the planner's existing append-only boundary API."""
    before = planner.verify()
    need(before['unresolved_batches'] == 0 and planner.pending() is None,
         'no unresolved optimizer reservation')
    state = planner.summary()
    need(state['checkpoint_sha256'] == checkpoint_sha256,
         'physical checkpoint joins planner before admission')
    epoch = planner._get('epoch')
    need(epoch > 0 and planner.db.execute('SELECT state FROM epochs WHERE id=?',
                                         (epoch,)).fetchone()[0] == 'complete',
         'closed epoch required')
    need(set(old.contexts) <= set(new.contexts) and
         all(old.contexts[k] == new.contexts[k] for k in old.contexts),
         'no role/context deletion or drift')
    authority_check(new.extension_raw)
    raw = canonical(new.batch_manifest)
    planner.adopt_manifest(raw, sha(raw), checkpoint_sha256=checkpoint_sha256)
    return {'schema': 'jope.prepared-epoch-adoption/v1',
            'previous_snapshot_sha256': old.identity, 'snapshot_sha256': new.identity,
            'checkpoint_sha256': checkpoint_sha256, 'manifest_sha256': sha(raw),
            'planner_head': planner.verify(), 'optimizer_updates': 0,
            'unique_growth_requires_physical_committed_batches': True}
