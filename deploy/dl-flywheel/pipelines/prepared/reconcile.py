"""Read-only exact journal/physical checkpoint capture; never replays an update.

The caller must freeze its own training cgroup before capture and must thaw it
if capture refuses. GP consumers may keep changing terminal outcome rows.
"""
import json, math, sqlite3
from pathlib import Path
from jope_flywheel.program_io import need, sha, read
from neural_native_stream import BatchStream
from neural_outbox import Outbox


def readonly(path):
    p = Path(path)
    need(p.is_absolute() and all(not x.is_symlink() for x in (p, *p.parents)),
         'absolute nonsymlink owned journal')
    db = sqlite3.connect(p.as_uri() + '?mode=ro', uri=True, isolation_level=None)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA query_only=ON'); db.execute('BEGIN')
    return db


class Snapshot:
    """Frozen verifier transaction control is logical, not physical."""
    def __init__(self, db): self.db = db
    @property
    def in_transaction(self): return False
    def execute(self, sql, *args):
        if sql in ('BEGIN', 'COMMIT'): return self.db.execute('SELECT 1')
        need(sql.lstrip().upper().startswith('SELECT '), 'readonly verifier SQL')
        return self.db.execute(sql, *args)


def capture(training_root, *, campaign_sha256, frozen, max_updates,
            plateau_patience, learning_rate_floor, require_closed_epoch=True, certified_unstarted=None):
    need(frozen is True or (certified_unstarted is not None and certified_unstarted.get('optimizer_started')is False), 'exclusive frozen or certified stopped training owner required')
    root = Path(training_root)
    db = readonly(root / 'batches.sqlite'); out = readonly(root / 'outbox.sqlite')
    try:
        planner = BatchStream.__new__(BatchStream); planner.db = db
        planner.config = planner._get('config')
        head = planner._verify()
        if certified_unstarted is None:
            need(head['unresolved_batches'] == 0, 'unknown optimizer intent refuses transfer')
        else:
            proof=certified_unstarted
            need(proof['schema']=='jope.certified-unstarted-reservation/v1'and proof['optimizer_started']is False and
                 proof['retry_optimizer']is False and proof['reservation_must_remain_in_place']is True and head['unresolved_batches']==1,
                 'only explicitly certified unstarted reservation')
            pending=db.execute("SELECT *FROM batches WHERE state!='committed'").fetchall()
            need(len(pending)==1 and pending[0]['id']==proof['batch_id']and pending[0]['state']=='reserved'and
                 pending[0]['commit_body']is None and out.execute('SELECT count(*)FROM batches WHERE id=?',(proof['batch_id'],)).fetchone()[0]==0,
                 'same untouched reservation with no durable optimizer ACK')
        ancestry = planner._get('ancestry')
        need(ancestry is not None and ancestry['optimizer_steps'] == 429,
             'original physical429 ancestry retained')
        epoch = planner._get('epoch')
        epoch_row = db.execute('SELECT state FROM epochs WHERE id=?', (epoch,)).fetchone()
        need(epoch > 0 and epoch_row is not None and epoch_row[0] in ('open', 'complete'),
             'valid current epoch required')
        need(not require_closed_epoch or epoch_row[0] == 'complete', 'closed current epoch required')
        steps = planner._get('optimizer_step'); absolute = 429 + steps
        need(0 < steps < max_updates and head['committed_batches'] == steps,
             'remaining original update budget')
        q = Outbox.__new__(Outbox); q.db = Snapshot(out); outbox_head = q.verify()
        last = None; ids = set(ancestry['exposed_context_ids'])
        presentations = ancestry['presentations']; fingerprints = set(ancestry['fingerprints'])
        proofs = []
        for row in db.execute("SELECT *FROM batches WHERE state='committed' ORDER BY epoch,batch_index"):
            commit = json.loads(row['commit_body']); intent = json.loads(row['body'])
            step = commit['optimizer_step']; path = root / f'batch-{429 + step:09d}.json'
            raw = read(path, 16 * 1024**2); proof = json.loads(raw)
            need(sha(raw) == commit['evidence']['event_sha256'] and
                 proof['config_sha256'] == campaign_sha256 and
                 proof['absolute_optimizer_step'] == 429 + step and
                 proof['local_optimizer_step'] == step and
                 set(proof['batch']) == set(intent) | {'rows'} and
                 all(proof['batch'][k] == intent[k] for k in intent) and
                 proof['checkpoint']['step'] == 429 + step and
                 proof['checkpoint']['config_sha256'] == campaign_sha256 and
                 proof['checkpoint']['parameter_sha256'] == commit['checkpoint_after'],
                 'every committed physical batch proof joins planner')
            emitted = out.execute('SELECT ack FROM batches WHERE id=?', (row['id'],)).fetchone()
            need(emitted is not None and json.loads(emitted['ack']) == proof['emission_ack'],
                 'durable proposal acknowledgement before every committed update')
            fingerprints.add(row['fingerprint']); ids.update(intent['ordered_data_run_ids'])
            presentations += len(intent['ordered_data_run_ids']); last = proof
            proofs.append({'step': 429 + step, 'sha256': sha(raw)})
        need(last is not None and len(fingerprints) == steps + len(ancestry['fingerprints']),
             'no membership fingerprint lost or repeated')
        slot = root / 'rolling' / f'{absolute:09d}'
        manifest_raw = read(slot / 'manifest.json'); manifest = json.loads(manifest_raw)
        state = read(slot / 'state.safetensors', 32 * 1024**2)
        need(manifest == last['checkpoint'] and manifest['parameter_sha256'] ==
             planner._get('checkpoint') and sha(state) == manifest['state_sha256'] and
             len(state) == manifest['state_bytes'], 'physical latest checkpoint closure')
        control_raw = read(root / 'optimizer-control.json'); control = json.loads(control_raw)
        need(control['schema'] == 'jope.continuous-optimizer-control/v1' and
             control['absolute_optimizer_step'] == absolute and
             control['checkpoint_parameter_sha256'] == manifest['parameter_sha256'] and
             control['checkpoint_state_sha256'] == manifest['state_sha256'],
             'matching optimizer-control state')
        rates = control['group_learning_rates']; plateau = control['plateau']
        need(rates and all(type(x) in (int, float) and math.isfinite(x) and
                           learning_rate_floor <= x <= .0003 for x in rates) and
             plateau['patience'] == plateau_patience and
             plateau['floor'] == learning_rate_floor and
             type(plateau['bad']) is int and 0 <= plateau['bad'] < plateau_patience and
             type(plateau['reductions']) is int and plateau['reductions'] >= 0 and
             plateau['best'] is not None and math.isfinite(plateau['best']),
             'preserve LR and plateau controller, never reset')
        progress = json.loads(read(root / 'progress.json', 32 * 1024**2))
        need(progress['state'] in ('running', 'maximum_updates_reached') and progress['config_sha256'] == campaign_sha256 and
             progress['completed_batches'] == absolute and
             progress['checkpoint_sha256'] == manifest['parameter_sha256'] and
             progress['unique_train_data_runs_seen'] == len(ids) and
             progress['total_train_presentations'] == presentations and
             progress['fixed_panel_sha256'] == control['fixed_panel_sha256'],
             'published progress fully reconciles physical journals')
        cold_batch=None
        if certified_unstarted is not None:
            from neural_native_stream import validate_manifest,fingerprint
            from jope_flywheel.protocol import digest
            from dataclasses import asdict
            from neural_native_stream import Batch
            intent=json.loads(pending[0]['body'])
            need(intent['batch_id']==certified_unstarted['batch_id']==digest({'schema':'jope.flywheel-batch-intent/v1',**{k:v for k,v in intent.items()if k!='batch_id'}})and
                 intent['fingerprint']==fingerprint(intent['ordered_data_run_ids']), 'sealed exact ordered membership digest')
            need(certified_unstarted['absolute_step']==absolute+1 and intent['checkpoint_before']==manifest['parameter_sha256']and
                 intent['manifest_sha256']==planner._get('manifest'), 'certified next step/checkpoint/manifest')
            active=db.execute('SELECT body FROM manifests WHERE sha=?',(intent['manifest_sha256'],)).fetchone()
            _,context_rows=validate_manifest(bytes(active[0]),intent['manifest_sha256'])
            cold_batch=asdict(Batch(**{**intent,'ordered_data_run_ids':tuple(intent['ordered_data_run_ids']),
                'rows':tuple(json.loads(json.dumps(context_rows[c])) for c in intent['ordered_data_run_ids'])}))
        completed = db.execute("SELECT count(*)FROM epochs WHERE state='complete'").fetchone()[0]
        need(progress['completed_epochs'] == ancestry['completed_epochs'] + completed and
             progress['latest_full_validation']['actual_epoch'] == progress['completed_epochs'],
             'closed epoch validation completed before transfer')
        def ref(path, raw): return {'path': str(path), 'sha256': sha(raw), 'bytes': len(raw)}
        return {'schema': 'jope.continuous-transfer-capture/v1',
                'campaign_sha256': campaign_sha256, 'absolute_step': absolute,
                'certified_unstarted': certified_unstarted, 'certified_batch': cold_batch,
                'local_step': steps, 'remaining_updates': max_updates - steps,
                'planner_head': head, 'outbox_head': outbox_head,
                'manifest_sha256': planner._get('manifest'), 'ancestry': ancestry,
                'current_epoch': {'id': epoch, 'state': epoch_row[0],
                    'remaining_context_ids': [r[0] for r in db.execute('SELECT run FROM members WHERE epoch=? AND batch IS NULL ORDER BY run',(epoch,))]},
                'checkpoint_manifest': ref(slot / 'manifest.json', manifest_raw),
                'checkpoint_state': ref(slot / 'state.safetensors', state),
                'control_ref': ref(root / 'optimizer-control.json', control_raw),
                'control': control, 'progress': progress,
                'unique_contexts': sorted(ids), 'presentations': presentations,
                'continuation_seen_ids': [r[0] for r in db.execute(
                    "SELECT DISTINCT m.run FROM members m JOIN batches b ON m.batch=b.id WHERE b.state='committed' ORDER BY m.run")],
                'proof_tail': [{'absolute_step': p['step'], 'proof_sha256': p['sha256'],
                                'parameter_sha256': json.loads(read(
                                    root / f'batch-{p["step"]:09d}.json'))['checkpoint']['parameter_sha256']}
                               for p in proofs[-2:]],
                'fingerprints': sorted(fingerprints), 'batch_proofs': proofs,
                'replays': 0, 'optimizer_mutations': 0, 'journal_mutations': 0}
    finally:
        db.execute('ROLLBACK'); out.execute('ROLLBACK'); db.close(); out.close()
