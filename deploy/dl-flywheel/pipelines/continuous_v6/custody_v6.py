"""Verified-watermark custody for runner_v6 (v2build r2; answers Codex F3 on the v2 BUILD review).

A watermark names the last committed local step whose lineage is fully joined:

  * every planner event after the previous watermark is chain-verified FROM that watermark's (seq, sha) head;
  * every ``batch.committed`` event in that suffix is the next local step, continues the checkpoint chain, matches
    the stored batch row (intent and commit body), and its evidence sha256 equals the bytes of ``batch-<abs>.json``;
  * that proof's ``emission_ack`` equals the durable outbox ACK row for the same batch, and the ACK's (seq, sha)
    is the outbox ``batch.emitted`` event for that batch and body (so an older outbox copy cannot satisfy it);
  * every outbox event after the previous watermark is chain-verified from that watermark's outbox head;
  * counts stay continuous: outbox batches and emission rows grow by exactly the joined suffix, the planner cursor
    is the last joined step and checkpoint, and no proof file exists beyond it.

Each pass reads only the suffix (bounded work) while re-joining the prefix by its head sha, so the prefix can never
be silently dropped or replaced.  ``extend`` returns a new watermark only after every join passes; callers persist
it only then.  Both databases are read inside one read transaction each; nothing is written.

A verified watermark is evidence about the bytes seen THEN.  Before any clean closure and before any resume,
``prefix_integrity`` therefore re-hashes EVERY committed proof file against the planner-chain evidence sha (measured
~101 KB/proof; ~1 min for 80k proofs at the measured cold-read rate) inside a work budget, and ``proof_inventory``
requires the proof filename set to be exactly the committed steps (no gap, orphan, future, malformed or non-regular
entry) — r2 answer to CX-REV review_CL_b3e01c0b findings 1 and 2.
"""
import json
import os
import re
import time
from pathlib import Path

from jope_flywheel.protocol import canonical, digest
from jope_flywheel.program_io import need, sha

SCHEMA = 'jope.v2-verified-watermark/v1'
ZERO = '0' * 64
PLANNER_KINDS = {'stream.created', 'stream.ancestry', 'epoch.opened', 'epoch.completed', 'batch.reserved',
                 'batch.committed', 'batch.uncertain', 'manifest.adopted'}
OUTBOX_KINDS = {'batch.emitted', 'emission.leased', 'emission.resolved'}
PROOF_SCHEMA = 'jope.continuous-neural-batch-proof/v2'
PROOF_NAME = re.compile(r'batch-([0-9]{9})\.json')
MAX_PROOF = 16 * 1024**2


class Budget:
    """Monotonic work budget; ``None`` means unbounded (tests only)."""
    def __init__(self, seconds):
        self.end = None if seconds is None else time.monotonic() + seconds

    def check(self, what):
        if self.end is not None and time.monotonic() > self.end:
            raise TimeoutError('custody verification exceeded its bounded budget at ' + what)


def _decode(raw):
    def pairs(items):
        d = {}
        for k, v in items:
            need(k not in d, 'duplicate JSON key'); d[k] = v
        return d
    value = json.loads(raw, object_pairs_hook=pairs); canonical(value); return value


def genesis_watermark(genesis):
    """The immutable anchor: the sealed genesis receipt's planner head, an empty outbox, the captured checkpoint."""
    head = genesis['planner']['head']; out = genesis['outbox']['head']
    need(head['committed_batches'] == 0 and head['unresolved_batches'] == 0 and out['observations'] == 0, 'fresh genesis heads')
    return {'schema': SCHEMA, 'anchor': 'genesis', 'local_step': 0, 'absolute_step': genesis['origin_absolute_step'],
            'parameter_sha256': genesis['ancestry_summary']['checkpoint_sha256'], 'last_proof_sha256': None,
            'proof_chain_sha256': digest({'anchor': 'genesis', 'ancestry_sha256': genesis['ancestry_sha256'],
                                          'campaign_sha256': genesis['campaign_sha256']}),
            'adoption_seq': 0, 'planner': {'events': head['events'], 'head_sha256': head['head_sha256'], 'committed': 0},
            'outbox': {'events': out['events'], 'head_sha256': out['head_sha256'], 'batches': 0, 'observations': 0}}


def _closed(w):
    need(type(w) is dict and w.get('schema') == SCHEMA and type(w['local_step']) is int and w['local_step'] >= 0 and
         set(w['planner']) == {'events', 'head_sha256', 'committed'} and
         set(w['outbox']) == {'events', 'head_sha256', 'batches', 'observations'} and w['planner']['committed'] == w['local_step'],
         'closed verified watermark')


def _rejoin(db, events, head, what):
    if events == 0:
        need(head == ZERO, what + ' empty-chain head'); return
    row = db.execute('SELECT sha FROM events WHERE seq=?', (events,)).fetchone()
    need(row is not None and row[0] == head, what + ' prefix head re-joins the watermark (prefix unchanged)')


def _suffix(db, start_seq, start_head, kinds, decode, what, budget):
    previous = start_head; seq = start_seq; out = []
    for row in db.execute('SELECT seq,kind,body,previous,sha FROM events WHERE seq>? ORDER BY seq', (start_seq,)):
        seq += 1; body = decode(row[2])
        need(row[0] == seq and row[3] == previous and row[1] in kinds and
             digest({'seq': seq, 'kind': row[1], 'body': body, 'previous': previous}) == row[4], what + ' suffix event chain')
        previous = row[4]; out.append((seq, row[1], body, row[4]))
        if seq % 1024 == 0: budget.check(what + ' event chain')
    return out, seq, previous


def extend(training, planner_db, outbox_db, watermark, *, origin, campaign_sha256, execution_sha256, budget_seconds=None,
           expect_no_suffix=False):
    """Join the committed suffix after ``watermark``; return (new watermark, report).  Read-only."""
    _closed(watermark); training = Path(training); budget = Budget(budget_seconds); began = time.monotonic()
    need(not planner_db.in_transaction and not outbox_db.in_transaction, 'independent read transactions')
    planner_db.execute('BEGIN'); outbox_db.execute('BEGIN')
    try:
        wp, wo = watermark['planner'], watermark['outbox']
        _rejoin(planner_db, wp['events'], wp['head_sha256'], 'planner'); _rejoin(outbox_db, wo['events'], wo['head_sha256'], 'outbox')
        p_events, p_seq, p_head = _suffix(planner_db, wp['events'], wp['head_sha256'], PLANNER_KINDS, _decode, 'planner', budget)
        o_events, o_seq, o_head = _suffix(outbox_db, wo['events'], wo['head_sha256'], OUTBOX_KINDS, json.loads, 'outbox', budget)
        emitted = {body['batch_id']: (seq, h, body) for seq, kind, body, h in o_events if kind == 'batch.emitted'}
        step = watermark['local_step']; previous = watermark['parameter_sha256']; chain = watermark['proof_chain_sha256']
        last_ack = None; last_proof = watermark['last_proof_sha256']; adoption = watermark['adoption_seq']; emissions = 0; joined = []
        for seq, kind, body, _ in p_events:
            need(kind not in ('stream.created', 'stream.ancestry', 'batch.uncertain'), 'planner suffix holds no genesis or uncertainty event')
            if kind != 'batch.committed': continue
            step += 1; absolute = origin + step; bid = body['batch_id']
            need(body['optimizer_step'] == step and body['checkpoint_before'] == previous and body['checkpoint_after'] != previous,
                 'contiguous committed step and checkpoint chain')
            row = planner_db.execute('SELECT state,body,commit_body FROM batches WHERE id=?', (bid,)).fetchone()
            need(row is not None and row[0] == 'committed' and _decode(row[2]) == body, 'stored commit body is the committed event')
            intent = _decode(row[1])
            ev = body['evidence']; path = training / f'batch-{absolute:09d}.json'
            need(path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_PROOF, 'committed proof present: ' + path.name)
            raw = path.read_bytes(); need(ev.get('origin') == 'native_optimizer' and sha(raw) == ev.get('event_sha256'),
                                         'commit evidence is exactly the proof bytes: ' + path.name)
            proof = json.loads(raw); cp = proof['checkpoint']
            need(proof['schema'] == PROOF_SCHEMA and proof['config_sha256'] == campaign_sha256 and
                 proof['execution_config_sha256'] == execution_sha256 and proof['absolute_optimizer_step'] == absolute and
                 proof['local_optimizer_step'] == step and set(proof['batch']) == set(intent) | {'rows'} and
                 all(proof['batch'][k] == intent[k] for k in intent) and intent['batch_id'] == bid and
                 intent['ordered_data_run_ids'] == body['ordered_data_run_ids'] and intent['checkpoint_before'] == previous and
                 cp['step'] == absolute and cp['config_sha256'] == campaign_sha256 and cp['parameter_sha256'] == body['checkpoint_after'],
                 'proof joins its planner intent, commit, step and checkpoint: ' + path.name)
            need(type(proof['snapshot_sha256']) is str and len(proof['snapshot_sha256']) == 64 and
                 type(proof['adoption_seq']) is int and proof['adoption_seq'] >= adoption, 'proof snapshot and monotone adoption sequence')
            ack = proof['emission_ack']; arow = outbox_db.execute('SELECT body,ack FROM batches WHERE id=?', (bid,)).fetchone()
            need(arow is not None and json.loads(arow[1]) == ack, 'proof ACK is the durable outbox ACK row: ' + path.name)
            need(ack['batch_id'] == bid and ack['global_step'] == absolute and ack['checkpoint_sha256'] == previous and
                 ack['body_sha256'] == sha(bytes(arow[0])) and (last_ack is None or ack['head']['seq'] > last_ack),
                 'ACK identity, step, pre-update checkpoint and order: ' + path.name)
            e = emitted.get(bid)
            need(e is not None and (e[0], e[1]) == (ack['head']['seq'], ack['head']['sha256']) and
                 e[2]['body_sha256'] == ack['body_sha256'] and e[2]['emission_ids'] == ack['emission_ids'],
                 'ACK head is this batch\'s outbox emission event inside the joined outbox suffix: ' + path.name)
            last_ack = ack['head']['seq']; emissions += len(ack['emission_ids']); adoption = proof['adoption_seq']
            last_proof = ev['event_sha256']; previous = body['checkpoint_after']
            chain = digest({'previous': chain, 'absolute_step': absolute, 'proof_sha256': last_proof, 'ack_seq': last_ack,
                            'parameter_sha256': previous})
            joined.append(absolute)
            if step % 64 == 0: budget.check('proof joins')
        committed_ids = {b['batch_id'] for _, k, b, _ in p_events if k == 'batch.committed'}
        need(set(emitted) <= committed_ids, 'every outbox emission in the suffix belongs to a joined committed update (no ACK without a commit)')
        cursor = _decode(planner_db.execute("SELECT value FROM state WHERE key='optimizer_step'").fetchone()[0])
        ckpt = _decode(planner_db.execute("SELECT value FROM state WHERE key='checkpoint'").fetchone()[0])
        need(cursor == step and ckpt == previous, 'planner cursor is the last joined step and checkpoint')
        proof_inventory(training, origin, step)
        batches = outbox_db.execute('SELECT count(*) FROM batches').fetchone()[0]
        rows = outbox_db.execute('SELECT count(*) FROM emissions').fetchone()[0]
        need(batches == wo['batches'] + len(joined) == step and rows == wo['observations'] + emissions,
             'outbox batch and emission counts continue exactly by the joined suffix')
        need(not expect_no_suffix or (not joined and not p_events), 'no planner change since the verified watermark')
        planner_db.execute('COMMIT'); outbox_db.execute('COMMIT')
    except BaseException:
        for db in (planner_db, outbox_db):
            if db.in_transaction: db.execute('ROLLBACK')
        raise
    new = {'schema': SCHEMA, 'anchor': watermark['anchor'], 'local_step': step, 'absolute_step': origin + step,
           'parameter_sha256': previous, 'last_proof_sha256': last_proof, 'proof_chain_sha256': chain, 'adoption_seq': adoption,
           'planner': {'events': p_seq, 'head_sha256': p_head, 'committed': step},
           'outbox': {'events': o_seq, 'head_sha256': o_head, 'batches': batches, 'observations': rows}}
    report = {'joined_steps': len(joined), 'planner_events': len(p_events), 'outbox_events': len(o_events),
              'seconds': time.monotonic() - began, 'from_step': watermark['local_step'], 'to_step': step,
              'previous_watermark_sha256': digest(watermark)}
    return new, report


def proof_inventory(training, origin, upto):
    """The set of ``batch-*`` entries is exactly the committed steps origin+1 .. origin+upto, each a regular file."""
    steps = []
    with os.scandir(training) as entries:
        for e in entries:
            if not e.name.startswith('batch-'): continue
            m = PROOF_NAME.fullmatch(e.name)
            need(m is not None and e.is_file(follow_symlinks=False), 'malformed or non-regular proof entry: ' + e.name)
            steps.append(int(m.group(1)))
    steps.sort()
    need(steps == list(range(origin + 1, origin + upto + 1)),
         'proof inventory is exactly the committed steps (no gap, orphan or future proof): expected %d..%d, found %d entries%s' %
         (origin + 1, origin + upto, len(steps), '' if not steps else ' %d..%d' % (steps[0], steps[-1])))


def prefix_integrity(training, planner_db, origin, upto, *, budget_seconds=None):
    """Re-hash EVERY committed proof file now against its planner-chain commit evidence (current physical prefix)."""
    training = Path(training); budget = Budget(budget_seconds); began = time.monotonic(); n = 0; total = 0
    proof_inventory(training, origin, upto)
    need(not planner_db.in_transaction, 'independent read transaction'); planner_db.execute('BEGIN')
    try:
        for row in planner_db.execute("SELECT seq, body FROM events WHERE kind='batch.committed' ORDER BY seq"):
            body = _decode(row[1]); n += 1
            need(body['optimizer_step'] == n, 'contiguous committed evidence')
            raw = (training / f'batch-{origin + n:09d}.json').read_bytes(); total += len(raw)
            need(sha(raw) == body['evidence']['event_sha256'], 'current proof bytes equal the committed evidence (step %d)' % n)
            if n % 256 == 0: budget.check('prefix integrity')
        need(n == upto, 'every committed step re-hashed')
        planner_db.execute('COMMIT')
    except BaseException:
        if planner_db.in_transaction: planner_db.execute('ROLLBACK')
        raise
    return {'proofs': n, 'bytes': total, 'seconds': time.monotonic() - began}


def audit(training, planner_db, outbox_db, genesis, *, campaign_sha256, execution_sha256, budget_seconds=None):
    """Full re-join from the genesis anchor (O(total); reviews and DR drills).  Valid for a single-execution campaign."""
    return extend(training, planner_db, outbox_db, genesis_watermark(genesis), origin=genesis['origin_absolute_step'],
                  campaign_sha256=campaign_sha256, execution_sha256=execution_sha256, budget_seconds=budget_seconds)
