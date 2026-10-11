"""Restart a stopped v2 campaign from its own VERIFIED closure (no automatic recovery; read-only derivation).

``v2_resume(campaign_root, ...)`` returns the runner_v6 ``resume`` for a training root whose last run segment closed
cleanly AND verifiably (r2, Codex F2/F3).  It refuses unless every one of these holds:

  * no ``failure.json``, ``closure-failure.json``, ``uncertain-physical-state`` or ``closure-errors.jsonl`` in the
    training root and no ``fatal*.json`` in the campaign root (each needs review, never automatic retry);
  * ``segment.json``, ``terminal-intent.json``, ``result-verification.json`` and ``result.json`` all name the SAME
    segment (a later segment that never closed leaves a stale result, which is refused);
  * the verification passed, binds the exact intent bytes and this execution, and ``result.json`` binds the exact
    verification and intent bytes; the state is a clean terminal state;
  * the learning-guard (G32) and KL-guard (T12) sidecars are byte-identical to the states sealed in the verification;
  * the published control is the verified control; the full read-only planner verify is the verified watermark head
    with no reservation; the watermark re-joins the planner, the outbox (whose later resolution events may follow)
    and the proof files with no suffix and an exact proof inventory, every committed proof re-hashed against its
    planner-chain evidence NOW (CX-REV finding 1), and the newest rolling slot at the watermark step.

``genesis_resume`` (in genesis.py) covers the first start; it refuses once any segment has started.
"""
import json
import sqlite3
from pathlib import Path

from jope_flywheel.program_io import need, sha
from jope_flywheel.protocol import canonical
from neural_native_stream import BatchStream
from .genesis import GENESIS_SCHEMA
from . import custody_v6 as CU
from .runner_v6 import MARKERS, TERMINAL_FILES

CLEAN = ('global_deadline_reached', 'maximum_updates_reached', 'memory_floor_exit')


def _ref(path):
    return {'path': str(path), 'sha256': sha(Path(path).read_bytes())}


def _ro(path):
    db = sqlite3.connect('file:' + str(Path(path).resolve()) + '?mode=ro', uri=True, isolation_level=None)
    db.row_factory = sqlite3.Row
    return db


def refuse_markers(root):
    t = root / 'training'
    need(not any((t / m).exists() for m in MARKERS), 'no failure, closure failure, uncertain state or closure error (review required)')
    need(not list(root.glob('fatal*.json')), 'no unreviewed CLI fatal record (review required)')


def v2_resume(campaign_root, *, execution_config_sha256):
    root = Path(campaign_root); t = root / 'training'
    g_raw = (root / 'genesis.json').read_bytes(); g = json.loads(g_raw); need(g['schema'] == GENESIS_SCHEMA, 'genesis receipt')
    refuse_markers(root)
    need(all((t / f).is_file() for f in TERMINAL_FILES), 'a complete terminal closure (segment, intent, verification, result)')
    raw = {f: (t / f).read_bytes() for f in TERMINAL_FILES}
    segment, intent, verification, result = (json.loads(raw[f]) for f in TERMINAL_FILES)
    need(segment['segment_id'] == intent['segment_id'] == verification['segment_id'] == result['segment_id'],
         'intent, verification and result all close the latest started segment')
    need(verification['schema'] == 'jope.continuous-neural-result-verification/v2' and verification['passed'] is True and
         verification['intent_sha256'] == sha(raw['terminal-intent.json']) == result['intent_sha256'] and
         result['verification_sha256'] == sha(raw['result-verification.json']) and result['verified'] is True and
         result['schema'] == 'jope.continuous-neural-result/v3' and
         result['state'] == intent['state'] == verification['state'] and result['state'] in CLEAN,
         'clean terminal state bound to a passed verification of the exact intent')
    need(verification['execution_config_sha256'] == execution_config_sha256 == segment['execution_config_sha256'] and
         verification['campaign_sha256'] == g['campaign_sha256'], 'same execution and campaign')
    control_raw = (t / 'optimizer-control.json').read_bytes(); control = json.loads(control_raw)
    need(sha(control_raw) == verification['control_sha256'], 'published control is the verified control')
    gpath = t / 'learning-guard.json'
    need(type(verification.get('learning_guard')) is dict and gpath.is_file() and not gpath.is_symlink() and
         sha(gpath.read_bytes()) == verification['learning_guard_sha256'] == sha(canonical(verification['learning_guard'])),
         'learning-guard state is exactly the closure-sealed state (absent, altered or reset refused)')
    kpath = t / 'kl-guard.json'
    need(type(verification.get('kl_guard')) is dict and kpath.is_file() and not kpath.is_symlink() and
         sha(kpath.read_bytes()) == verification['kl_guard_sha256'] == sha(canonical(verification['kl_guard'])),
         'KL-guard state is exactly the closure-sealed state (absent, altered or reset refused)')
    wm = verification['watermark']; origin = g['origin_absolute_step']
    p = BatchStream.__new__(BatchStream); p.db = _ro(t / 'batches.sqlite')
    try:
        p.db.execute('BEGIN'); p.config = p._get('config'); head = p._verify(); p.db.execute('COMMIT')
    finally:
        p.db.close()
    need(head['unresolved_batches'] == 0 and head['committed_batches'] == wm['local_step'] and head['events'] == wm['planner']['events'] and
         head['head_sha256'] == wm['planner']['head_sha256'], 'full planner verify is exactly the verified watermark head')
    pdb, odb = _ro(t / 'batches.sqlite'), _ro(t / 'outbox.sqlite')
    try:
        mark, _ = CU.extend(t, pdb, odb, wm, origin=origin, campaign_sha256=g['campaign_sha256'],
                            execution_sha256=execution_config_sha256, expect_no_suffix=True)
    finally:
        pdb.close(); odb.close()
    need(mark['local_step'] == wm['local_step'] and mark['parameter_sha256'] == wm['parameter_sha256'], 'watermark re-joins unchanged')
    pdb = _ro(t / 'batches.sqlite')
    try:
        CU.prefix_integrity(t, pdb, origin, wm['local_step'])
    finally:
        pdb.close()
    slot = sorted((t / 'rolling').iterdir())[-1]
    m = json.loads((slot / 'manifest.json').read_bytes())
    need(m['step'] == origin + wm['local_step'] == control['absolute_optimizer_step'] and
         m['parameter_sha256'] == control['checkpoint_parameter_sha256'] == wm['parameter_sha256'] == result['current_parameter_sha256'] and
         m['state_sha256'] == control['checkpoint_state_sha256'], 'newest rolling slot, control, watermark and result agree')
    need(wm['local_step'] == 0 or sha((t / f'batch-{origin + wm["local_step"]:09d}.json').read_bytes()) == wm['last_proof_sha256'],
         'last committed proof is the verified one')
    pdb = _ro(t / 'batches.sqlite')
    try:
        exposed = sorted(r[0] for r in pdb.execute('SELECT DISTINCT run FROM members WHERE batch IS NOT NULL'))
        presented = pdb.execute('SELECT count(*) FROM members WHERE batch IS NOT NULL').fetchone()[0]
    finally:
        pdb.close()
    return {'local_step': wm['local_step'], 'origin_absolute_step': origin,
            'unique_contexts': sorted(set(g['resume']['unique_contexts']) | set(exposed)),
            'presentations': g['resume']['presentations'] + presented, 'continuation_seen_ids': exposed,
            'proof_tail': result['proof_tail'], 'watermark': wm,
            'checkpoint_manifest': _ref(slot / 'manifest.json'), 'checkpoint_state': _ref(slot / 'state.safetensors'),
            'control': control, 'execution_config_sha256': execution_config_sha256,
            'learning_guard': verification['learning_guard'], 'kl_guard': verification['kl_guard'],
            'training_objective_sha256': g['training_objective_sha256'],
            'basis': {'kind': 'closure', 'segment_id': segment['segment_id'], 'result_sha256': sha(raw['result.json']),
                      'verification_sha256': sha(raw['result-verification.json'])},
            'restart_of': {'result_sha256': sha(canonical(result)), 'planner_head': head}}
