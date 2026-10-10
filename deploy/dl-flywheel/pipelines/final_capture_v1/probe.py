"""Next-update parity probe and the reference read-only outbox summary (private copies only; never live).

Invariants:
- Every journal the probe writes is a NEW scratch file that is discarded; it never commits anywhere else, never
  opens the original or private-copy planner (the oracle stages a byte copy into scratch first) and never touches
  an outbox. The caller's torch CPU RNG is restored after each update.
- captured_inputs reads only the sealed capture output; fresh_resume_inputs is an independent oracle that derives
  the same successor inputs directly from the private copy with its own SQL and the documented lineage-head
  contract; first_update is the successor's first update (fresh journal + inherit, restore, control LR groups,
  reserve, prepare, one AdamW dropout step, checkpoint) and is identical for both sides, so any difference between
  the two updates is a difference in the captured state itself.
- staged_outbox_reader returns exactly the closed summary capture() requires; it reads byte-staged copies.
"""
import json, shutil, sqlite3, tempfile, time
from pathlib import Path
import torch
from dltrain import model as M
from jope_flywheel.protocol import canonical, digest
from jope_flywheel.program_io import need, read, sha
from continuous_state import normalizer, restore
from neural_native_stream import BatchStream, Waiting
from neural_native_train import prepare_microbatches, run_microbatches
from neural_storage import checkpoint
from neural_outbox import Outbox
from . import capture as C

LINEAGE_HEAD = 'jope.continuous-lineage-planner-head/v1'
MIB = 1024**2


def stage_sqlite(src, dst_dir, name):
    """Byte copy of a private SQLite file (+ -wal) into scratch; SQLite never opens the source."""
    dst_dir.mkdir(parents=True)
    for suffix in ('', '-wal'):
        p = Path(str(src) + suffix)
        if suffix and not p.exists(): continue
        need(p.is_file() and not p.is_symlink(), 'regular private SQLite file: ' + p.name)
        shutil.copyfile(p, dst_dir / (name + suffix), follow_symlinks=False)
    return dst_dir / name


def staged_outbox_reader(v1_path, parent_path, scratch):
    """Reference injected summary: Outbox.verify() + ordered ACK digest for v1, Outbox.summary() for the parent."""
    def reader():
        out = {}; stage = Path(tempfile.mkdtemp(prefix='outbox-stage-', dir=scratch))
        try:
            for role, path in (('v1', v1_path), ('parent', parent_path)):
                db = sqlite3.connect(stage_sqlite(Path(path), stage / role, 'outbox.sqlite'), isolation_level=None)
                db.row_factory = sqlite3.Row; db.execute('PRAGMA query_only=ON'); q = Outbox.__new__(Outbox); q.db = db
                try:
                    if role == 'v1':
                        head = q.verify(); db.execute('BEGIN')
                        acks = sorted((json.loads(r[0]) for r in db.execute('SELECT ack FROM batches')), key=lambda a: a['head']['seq'])
                        db.execute('COMMIT'); out[role] = {**head, 'acks_sha256': digest(acks), 'ack_count': len(acks)}
                    else:
                        db.execute('BEGIN'); out[role] = q.summary(); db.execute('COMMIT')
                finally: db.close()
        finally: shutil.rmtree(stage)
        return out
    return reader


def captured_inputs(capture_dir):
    """Successor inputs from the sealed capture output only, through the verifying reader (all chained parts)."""
    d = Path(capture_dir); body, _, ancestry = C.load_capture(d); manifest_raw = read(d / 'active-manifest.json', 64 * MIB)
    need(sha(manifest_raw) == body['planner']['active_manifest']['sha256'], 'sealed active manifest')
    refs = {k: {'path': str(d / body['checkpoint'][k]['file']), 'sha256': body['checkpoint'][k]['sha256'],
                'bytes': body['checkpoint'][k]['bytes']} for k in ('manifest', 'state')}
    return {'ancestry': ancestry, 'manifest_raw': manifest_raw, 'checkpoint': refs,
            'learning_rates': body['control']['body']['group_learning_rates'], 'seed': body['planner']['config']['seed'],
            'batch_size': body['planner']['config']['batch_size']}


def fresh_resume_inputs(campaign_root, scratch):
    """Independent oracle: what a fresh resume reads straight from the private copy (planner staged into scratch)."""
    root = Path(campaign_root); training = root / 'training'; c = json.loads(read(root / 'config.json'))
    work = Path(tempfile.mkdtemp(prefix='oracle-planner-', dir=scratch))
    try:
        db = sqlite3.connect(stage_sqlite(training / 'batches.sqlite', work / 'p', 'batches.sqlite'), isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            get = lambda k: json.loads(db.execute('SELECT value FROM state WHERE key=?', (k,)).fetchone()[0])
            planner = BatchStream.__new__(BatchStream); planner.db = db; planner.config = get('config'); head = planner.verify()
            db.execute('BEGIN'); anc = get('ancestry'); steps = get('optimizer_step'); ckpt = get('checkpoint'); active = get('manifest')
            own = [r[0] for r in db.execute("SELECT fingerprint FROM batches WHERE state='committed'")]
            need(len(own) == steps and db.execute("SELECT count(*)FROM batches WHERE state!='committed'").fetchone()[0] == 0, 'oracle: resolved planner')
            exposed = set(anc['exposed_context_ids']) | {r[0] for r in db.execute('SELECT DISTINCT run FROM members WHERE batch IS NOT NULL')}
            presentations = anc['presentations'] + db.execute('SELECT count(*)FROM members WHERE batch IS NOT NULL').fetchone()[0]
            completed = anc['completed_epochs'] + db.execute("SELECT count(*)FROM epochs WHERE state='complete'").fetchone()[0]
            manifest_raw = bytes(db.execute('SELECT body FROM manifests WHERE sha=?', (active,)).fetchone()[0]); db.execute('COMMIT')
            config = planner.config
        finally: db.close()
    finally: shutil.rmtree(work)
    prior = anc['head']['generations'] if anc['head'].get('schema') == LINEAGE_HEAD else [{'planner_ref': anc['planner_ref'], 'head': anc['head']}]
    planner_ref = str(Path(c['owned_root']) / 'training' / 'batches.sqlite'); fps = sorted(set(anc['fingerprints']) | set(own))
    ancestry = {'planner_ref': planner_ref,
                'head': {'schema': LINEAGE_HEAD, 'committed_batches': len(fps), 'unresolved_batches': 0,
                         'generations': prior + [{'planner_ref': planner_ref, 'head': head}]},
                'checkpoint_sha256': ckpt, 'optimizer_steps': anc['optimizer_steps'] + steps, 'completed_epochs': completed,
                'fingerprints': fps, 'exposed_context_ids': sorted(exposed), 'presentations': presentations}
    slot = training / 'rolling' / f"{anc['optimizer_steps'] + steps:09d}"
    refs = {}
    for key, name in (('manifest', 'manifest.json'), ('state', 'state.safetensors')):
        raw = read(slot / name, 64 * MIB); refs[key] = {'path': str(slot / name), 'sha256': sha(raw), 'bytes': len(raw)}
    control = json.loads(read(training / 'optimizer-control.json'))
    return {'ancestry': ancestry, 'manifest_raw': manifest_raw, 'checkpoint': refs, 'learning_rates': control['group_learning_rates'],
            'seed': config['seed'], 'batch_size': config['batch_size']}


def first_update(scratch, inputs, *, native, snapshot, preprocessing_raw, max_padded_tokens, campaign_sha256, before_update=None):
    """The successor's first update on a new scratch journal; ``before_update(net, opt)`` exists only so fixtures can
    show the probe is sensitive to optimizer/RNG state. Returns the batch, hashes and checkpoint bytes."""
    work = Path(tempfile.mkdtemp(prefix='next-update-', dir=scratch)); anc = inputs['ancestry']; raw = inputs['manifest_raw']
    need(sha(canonical(snapshot.batch_manifest)) == sha(raw), 'probe snapshot is the active manifest')
    caller = torch.get_rng_state()
    try:
        M.configure_determinism()
        planner = BatchStream(work / 'batches.sqlite', raw, sha(raw), seed=inputs['seed'], batch_size=inputs['batch_size'],
                              initial_checkpoint_sha256=anc['checkpoint_sha256'])
        try:
            planner.inherit(anc)
            net, opt, _ = restore(inputs['checkpoint']['manifest'], inputs['checkpoint']['state'],
                                  expected_parameter_sha256=anc['checkpoint_sha256'], device='cpu')
            need(len(opt.param_groups) == len(inputs['learning_rates']), 'exact optimizer groups')
            for g, r in zip(opt.param_groups, inputs['learning_rates']): g['lr'] = r
            b = planner.reserve(anc['checkpoint_sha256']); need(not isinstance(b, Waiting), 'a unique next batch exists')
            ids = list(b.ordered_data_run_ids); epoch = anc['completed_epochs'] + b.epoch
            batches, total, _ = prepare_microbatches(snapshot, ids, epoch=epoch - 1, batch_index=b.index,
                                                     normalizer=normalizer(preprocessing_raw, native), max_padded_tokens=max_padded_tokens)
            if before_update is not None: before_update(net, opt)
            before = M.parameter_sha256(net); result = run_microbatches(net, batches, total, optimizer=opt, deadline=time.time() + 600)
            after = M.parameter_sha256(net); need(after != before, 'actual parameter update')
            checkpoint(net, opt, work / 'next', step=anc['optimizer_steps'] + 1, config_sha256=campaign_sha256)
        finally: planner.close()
        state = (work / 'next' / 'state.safetensors').read_bytes()
    finally:
        torch.set_rng_state(caller); shutil.rmtree(work)
    return {'batch': {'batch_id': b.batch_id, 'fingerprint': b.fingerprint, 'epoch': b.epoch, 'index': b.index, 'ordered_data_run_ids': ids},
            'lineage_epoch': epoch, 'parameter_before': before, 'parameter_sha256': after, 'state_sha256': sha(state),
            'loss': result['loss'], 'learning_rates': [g['lr'] for g in opt.param_groups], 'state': state}
