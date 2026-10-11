"""v2.0.0 genesis: build the v2 campaign training root from v1's sealed final capture (owner rulings R9, R14, R17).

``build(capture_dir, campaign_root, ...)`` is CPU-only and touches no live v1 file; every input comes from the
capture (W10 r2 ``final_capture_v1``, read back through ``load_capture``) or from explicitly passed bytes.  It creates
a NEW campaign root and refuses an existing one:

  training/preprocessing.json          unchanged train-only normalizer bytes (sha == capture inputs.preprocessing)
  training/rolling/<origin>/           byte copy of v1's final checkpoint (manifest.json, state.safetensors)
  training/best/, training/epochs/     empty recovery classes (runner_v6 retains into them)
  training/batches.sqlite              FRESH planner on v1's ACTIVE manifest, bounded-verify index, inherit(ancestry)
  training/outbox.sqlite               FRESH v2 outbox (model origin = v1 final parameter sha), bounded-verify indexes
  training/v1-final-optimizer-control.json   v1's exact final control (the parity gate runs on this)
  training/control-event-r14.json      jope.optimizer-control-event/v1 (R14: both groups 3.75e-5, plateau reset,
                                       patience 5 / floor 1e-6 unchanged, AdamW moments kept)
  training/optimizer-control.json      the post-R14 control runner_v6 applies
  inbox/                               C10b inbox genesis (genesis snapshot E0 + active manifest, campaign, contract)
  genesis.json                         create-only receipt binding every hash above

The R14 event is applied only after three gates on v1's UNCHANGED control (R14; r2, Codex F1): bitwise CPU and CUDA
restore of the captured checkpoint, and the exact NEXT-update parity receipt (``parity_probe``: v1's own next batch
B* through the frozen v1 step primitives and through runner_v6, bitwise model/AdamW/RNG equality).  ``probe=True``
builds a scratch, pre-R14 root (v1 control unchanged, no R14 event, never a live root) on which that probe runs.

Training objective (r3, owner ruling R23): ``training_objective_sha256`` is the digest of runner_v6's
training-objective identity ``{l6: OBJECTIVE, anchor_kl: ...}`` (anchored_step.training_objective).  It is required,
sealed in genesis.json and returned in every resume.  runner_v6 refuses a config whose identity differs.  The C10
inbox contract keeps pinning the unchanged L6 label objective.

Epoch accounting (r2, Codex F5): v1 stopped inside its open planner epoch 40 (lineage epoch 41; 5,712 consumed,
1,270 remaining).  v2 does NOT continue that partial epoch: its fresh planner opens a full pass over the active
manifest.  genesis.json records v1's open epoch exactly and declares this fresh-schedule choice; v1's partial epoch
is never counted as completed and B* is never scheduled by v2.
"""
import json
import os
import shutil
from pathlib import Path

from jope_flywheel.protocol import canonical, digest
from jope_flywheel.program_io import need, sha
from neural_native_stream import BatchStream
from neural_outbox import Outbox
from pipelines.boundary_admission_v2 import boundary as Bd
from pipelines.final_capture_v1 import capture as FC
from pipelines.final_capture_v1 import probe as FP
from .runner_v6 import (MARKERS, OUTBOX_INDEXES, OUTBOX_MAX_PAYLOAD, OUTBOX_MAX_ROWS, PLANNER_INDEXES, TERMINAL_FILES,
                        ensure_indexes, has_indexes)
from . import custody_v6 as CU

GENESIS_SCHEMA = 'jope.v2-genesis/v1'
EVENT_SCHEMA = 'jope.optimizer-control-event/v1'
R14 = {'group_learning_rate': 3.75e-5, 'plateau': {'best': None, 'bad': 0, 'reductions': 0}}
LIVE_PLATEAU = {'patience': 5, 'floor': 1e-6}     # R14: unchanged from v1; the live CLI asserts v1 carries exactly these
PARITY_SCHEMA = 'jope.v2-next-update-parity/v1'
GATES = {'cpu', 'cuda', 'next_update'}


def check_next_update_parity(receipt, *, capture_sha256, v1_control):
    need(type(receipt) is dict and receipt.get('schema') == PARITY_SCHEMA and receipt.get('passed') is True and
         receipt['capture_sha256'] == capture_sha256 and str(receipt['device']).startswith('cuda') and receipt['v1_control_sha256'] == sha(canonical(v1_control)) and
         receipt['control_applied'] == 'v1 unchanged (pre-R14)' and receipt['bstar']['source'] == 'v1 planner backup (byte copy)' and
         receipt['sides'] and all(v['bitwise_equal'] is True for v in receipt['comparisons'].values()) and
         {'model', 'adamw', 'rng'} <= set(receipt['comparisons']['A_vs_B1']['groups']),
         'exact next-update parity on v1\'s unchanged control on the live device class (CUDA; bitwise model/AdamW/RNG) before R14')


def epoch_accounting(body, ancestry):
    """v1's open epoch, recorded exactly; v2's fresh schedule declared (never the exact next v1 batch)."""
    pl = body['planner']; cur = pl['current_epoch']
    need(pl['completed_epochs_lineage'] == ancestry['completed_epochs'] and cur['state'] in ('open', 'complete') and
         cur['consumed'] + cur['remaining'] > 0 and len(cur['remaining_ids_sha256']) == 64 and
         (cur['state'] == 'open') == (cur['remaining'] > 0), 'captured v1 current epoch')
    offset = pl['completed_epochs_lineage'] - pl['completed_epochs_local']
    return {'v1_lineage_completed_epochs': pl['completed_epochs_lineage'],
            'v1_open_epoch': {**cur, 'lineage_epoch': offset + cur['id'],
                              'disposition': ('not continued; never counted as completed; consumed members count as presentations only'
                                              if cur['state'] == 'open' else 'v1 stopped at a closed epoch boundary; nothing unfinished')},
            'v2_schedule': {'policy': 'fresh full pass over the active manifest (planner.inherit bans every historical fingerprint)',
                            'first_v2_epoch_lineage_number': ancestry['completed_epochs'] + 1,
                            'note': 'the first v2 pass reuses lineage number %d for a different, complete membership' %
                                    (ancestry['completed_epochs'] + 1)},
            'parity_relation': 'the next-update parity probe replays v1\'s own next batch B* only to qualify update equivalence; '
                               'v2 does not schedule B*'}


def _create(path, body):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
    try:
        view = memoryview(body)
        while view: view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)
    return {'path': str(path), 'sha256': sha(body), 'bytes': len(body)}


def r14_event(v1_control, *, r14_receipt_sha256, parity_receipts):
    """The single logged R14 warm-restart event: applied only after the exact-restore parity gate."""
    need(v1_control['schema'] == 'jope.continuous-optimizer-control/v1' and len(v1_control['group_learning_rates']) == 2 and
         set(v1_control['plateau']) == {'best', 'bad', 'reductions', 'patience', 'floor'}, 'closed v1 control with two groups')
    need(type(parity_receipts) is dict and set(parity_receipts) == GATES and
         all(type(v) is dict and v.get('passed') is True and v.get('sha256') for v in parity_receipts.values()),
         'exact-restore (CPU, CUDA) and exact next-update parity gates passed before the R14 event')
    after = {**v1_control, 'group_learning_rates': [R14['group_learning_rate']] * 2,
             'plateau': {**R14['plateau'], 'patience': v1_control['plateau']['patience'], 'floor': v1_control['plateau']['floor']}}
    event = {'schema': EVENT_SCHEMA, 'ruling': {'id': 'R14', 'receipt_sha256': r14_receipt_sha256},
             'unchanged': {'patience': v1_control['plateau']['patience'], 'floor': v1_control['plateau']['floor']},
             'before': v1_control, 'after': after, 'adamw_moments': 'kept',
             'applied_after': 'exact-restore (CPU, CUDA) and exact next-update parity gates on v1\'s unchanged control',
             'parity_gate': parity_receipts}
    return event, after


def build(capture_dir, campaign_root, *, preprocessing_raw, campaign_sha256, contract, r14_receipt_sha256,
          parity_receipts, outbox_max_pending, training_objective_sha256, next_update_parity=None, probe=False, seed=None,
          batch_size=None):
    capture_dir, campaign_root = Path(capture_dir), Path(campaign_root)
    need(campaign_root.is_absolute() and not campaign_root.exists() and campaign_root.parent.is_dir(), 'new campaign root')
    need(type(training_objective_sha256) is str and len(training_objective_sha256) == 64 and
         all(ch in '0123456789abcdef' for ch in training_objective_sha256), 'R23 training-objective identity (sha256)')
    body, joins, ancestry = FC.load_capture(capture_dir)
    inputs = FP.captured_inputs(capture_dir)
    need(inputs['ancestry'] == ancestry, 'captured successor inputs')
    need(sha(preprocessing_raw) == body['inputs']['preprocessing']['sha256'], 'unchanged captured normalizer bytes')
    v1_control = body['control']['body']
    origin = ancestry['optimizer_steps']
    need(v1_control['absolute_optimizer_step'] == origin and v1_control['checkpoint_parameter_sha256'] == ancestry['checkpoint_sha256'],
         'control joins the captured final checkpoint')
    seed = inputs['seed'] if seed is None else seed
    batch_size = inputs['batch_size'] if batch_size is None else batch_size
    need(seed == inputs['seed'] and batch_size == inputs['batch_size'], 'same planner seed and batch size as v1')
    capture_sha = sha((capture_dir / 'final-capture.json').read_bytes())
    accounting = epoch_accounting(body, ancestry)
    if probe:
        need(next_update_parity is None, 'a probe root precedes the parity receipt')
        event, after = None, v1_control
    else:
        check_next_update_parity(next_update_parity, capture_sha256=capture_sha, v1_control=v1_control)
        gates = {**parity_receipts, 'next_update': {'passed': True, 'sha256': sha(canonical(next_update_parity)),
                                                    'what': 'v1 next batch B*: frozen v1 step primitives vs runner_v6, bitwise'}}
        event, after = r14_event(v1_control, r14_receipt_sha256=r14_receipt_sha256, parity_receipts=gates)
    manifest_raw = inputs['manifest_raw']; manifest_sha = sha(manifest_raw)
    genesis_snapshot = json.loads(manifest_raw)['snapshot_sha256']

    campaign_root.mkdir(mode=0o755)
    training = campaign_root / 'training'; training.mkdir()
    refs = {'preprocessing': _create(training / 'preprocessing.json', preprocessing_raw)}
    slot = training / 'rolling' / f'{origin:09d}'; slot.mkdir(parents=True)
    for rel in ('best', 'epochs'): (training / rel).mkdir()
    for key, name in (('manifest', 'manifest.json'), ('state', 'state.safetensors')):
        src = Path(inputs['checkpoint'][key]['path'])
        need(sha(src.read_bytes()) == inputs['checkpoint'][key]['sha256'], 'sealed captured checkpoint bytes')
        shutil.copyfile(src, slot / name, follow_symlinks=False)
        refs['checkpoint_' + key] = {'path': str(slot / name), 'sha256': sha((slot / name).read_bytes())}
        need(refs['checkpoint_' + key]['sha256'] == inputs['checkpoint'][key]['sha256'], 'checkpoint copy exact')

    planner = BatchStream(training / 'batches.sqlite', manifest_raw, manifest_sha, seed=seed, batch_size=batch_size,
                          initial_checkpoint_sha256=ancestry['checkpoint_sha256'])
    try:
        ensure_indexes(planner.db, PLANNER_INDEXES)
        planner.inherit(ancestry)
        planner_head = planner.verify()
        need(has_indexes(planner.db, PLANNER_INDEXES) and planner_head['unresolved_batches'] == 0 and
             planner_head['committed_batches'] == 0 and planner.pending() is None, 'fresh inherited planner')
    finally:
        planner.close()
    need(type(outbox_max_pending) is int and 0 < outbox_max_pending <= 32768, 'bounded own-row emission cap (immutable outbox config)')
    outbox = Outbox(training / 'outbox.sqlite', model_origin_sha256=ancestry['checkpoint_sha256'], max_pending=outbox_max_pending,
                    max_rows=OUTBOX_MAX_ROWS, max_payload_bytes=OUTBOX_MAX_PAYLOAD)
    try:
        ensure_indexes(outbox.db, OUTBOX_INDEXES)
        outbox_head = outbox.verify()
        need(has_indexes(outbox.db, OUTBOX_INDEXES) and outbox_head['observations'] == 0, 'fresh indexed v2 outbox')
    finally:
        outbox.close()

    refs['v1_control'] = _create(training / 'v1-final-optimizer-control.json', canonical(v1_control))
    if event is not None: refs['r14_event'] = _create(training / 'control-event-r14.json', canonical(event))
    refs['control'] = _create(training / 'optimizer-control.json', canonical(after))
    inbox = Bd.Inbox.create(campaign_root / 'inbox', genesis_snapshot_sha256=genesis_snapshot,
                            genesis_manifest_sha256=manifest_sha, campaign_sha256=campaign_sha256, contract=contract)
    genesis = {'schema': GENESIS_SCHEMA, 'generation': 'v2.0.0', 'campaign_sha256': campaign_sha256,
               'training_objective_sha256': training_objective_sha256,
               'probe': bool(probe), 'r14': None if probe else {'applied': True, 'event_sha256': refs['r14_event']['sha256']},
               'epoch_accounting': accounting,
               'capture': {'final_capture_sha256': capture_sha,
                           'capture_receipt_sha256': sha((capture_dir / 'capture-receipt.json').read_bytes()),
                           'class': body['terminal']['class'], 'owner_ruling': body.get('owner_ruling')},
               'origin_absolute_step': origin, 'ancestry_sha256': digest(ancestry),
               'ancestry_summary': {k: ancestry[k] for k in ('checkpoint_sha256', 'optimizer_steps', 'completed_epochs', 'presentations')},
               'active_manifest_sha256': manifest_sha, 'genesis_snapshot_sha256': genesis_snapshot,
               'planner': {'head': planner_head, 'indexes': sorted(PLANNER_INDEXES), 'seed': seed, 'batch_size': batch_size},
               'outbox': {'head': {k: outbox_head[k] for k in ('events', 'head_sha256', 'observations')}, 'indexes': sorted(OUTBOX_INDEXES),
                          'max_pending': outbox_max_pending, 'max_rows': OUTBOX_MAX_ROWS, 'max_payload_bytes': OUTBOX_MAX_PAYLOAD},
               'inherited_backlog': {'v1': {k: body['outbox']['v1'][k] for k in ('observations', 'unresolved', 'states')},
                                     'parent': {k: body['outbox']['parent'][k] for k in ('observations', 'unresolved', 'states')}},
               'inbox': {'root': str(inbox.root), 'genesis': inbox.genesis}, 'refs': refs,
               'resume': {'local_step': 0, 'origin_absolute_step': origin, 'unique_contexts': ancestry['exposed_context_ids'],
                          'presentations': ancestry['presentations'], 'continuation_seen_ids': [], 'proof_tail': []}}
    _create(campaign_root / 'genesis.json', canonical(genesis))
    return genesis


def genesis_resume(campaign_root, *, execution_config_sha256):
    """The runner_v6 ``resume`` for the first v2 start, read back from the sealed genesis receipt only.  Refuses once
    any segment has started (a started-but-unclosed first segment needs review, never a silent genesis restart)."""
    root = Path(campaign_root); raw = (root / 'genesis.json').read_bytes(); g = json.loads(raw)
    need(g['schema'] == GENESIS_SCHEMA, 'genesis receipt')
    t = root / 'training'
    need(not any((t / m).exists() for m in MARKERS + TERMINAL_FILES + ('learning-guard.json', 'kl-guard.json')) and not list(root.glob('fatal*.json')) and
         not (t / 'verified-watermark.json').exists() and not list(t.glob('batch-*.json')), 'no segment has ever started')
    control = json.loads((root / 'training' / 'optimizer-control.json').read_bytes())
    need(sha(canonical(control)) == g['refs']['control']['sha256'], 'post-R14 control is the sealed genesis control')
    return {**g['resume'], 'training_objective_sha256': g['training_objective_sha256'],
            'checkpoint_manifest': g['refs']['checkpoint_manifest'], 'checkpoint_state': g['refs']['checkpoint_state'],
            'control': control, 'execution_config_sha256': execution_config_sha256,
            'watermark': CU.genesis_watermark(g), 'basis': {'kind': 'genesis', 'genesis_sha256': sha(raw)}}
