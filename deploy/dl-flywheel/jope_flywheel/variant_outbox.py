"""Measured variant proposals, with no optimizer or retry authority.

The caller supplies the frozen model outbox, exact RegisteredInputs, a pinned
ProgramObserver/ProgramWorker queue and a version-verifying S3 library. A fresh
consumer must carry the closed historical failed/unknown evaluation-key census.
"""
from dataclasses import asdict
import json
from .program_gate import object_bytes, request_identity, verified_measurement
from .protocol import Context, canonical, digest, sha256


def need(ok, message):
    if not ok:
        raise ValueError(message)


def claim_variant(outbox, owner, *, lease_seconds, deadline):
    """Same frozen outbox event grammar; native/leased/unknown rows are untouched."""
    now = outbox.now()
    need(type(owner) is str and 0 < len(owner) <= 128 and
         type(lease_seconds) is int and 1 <= lease_seconds <= 600 and
         now + lease_seconds <= deadline, 'bounded variant lease')
    db = outbox.db
    db.execute('BEGIN IMMEDIATE')
    try:
        row = db.execute("SELECT * FROM emissions WHERE state='recorded' AND "
                         "json_extract(body,'$.emission.input_descriptor.source_kind')="
                         "'flywheel_variant' ORDER BY rowid LIMIT 1").fetchone()
        if row is None:
            db.execute('COMMIT')
            return None
        lease = {'emission_id': row['id'], 'owner': owner, 'started_at': now,
                 'expires_at': now + lease_seconds,
                 'token': digest({'id': row['id'], 'owner': owner, 'at': now})}
        db.execute("UPDATE emissions SET state='leased',lease=? WHERE id=?",
                   (canonical(lease), row['id']))
        outbox._event('emission.leased', lease)
        db.execute('COMMIT')
        return {'lease': lease, 'body': json.loads(row['body'])}
    except BaseException:
        if db.in_transaction:
            db.execute('ROLLBACK')
        raise


def census_queue(db):
    """One already-open read-only SQL snapshot; never claims or repairs old work."""
    from .queue import Queue
    checker = Queue.__new__(Queue)
    checker.db = db
    proof = checker.verify()
    jobs = {r['id']: dict(r) for r in db.execute("SELECT * FROM jobs WHERE stage IN ('evaluate','publish')")}
    keys, blocked, completed = set(), set(), set()
    for row in jobs.values():
        key = json.loads(row['payload'])['evaluation_key']
        sha256(key)
        need(row['id'] == 'gp-' + row['stage'] + '-' + key, 'historical job identity')
        keys.add(key)
    for key in keys:
        pair = [jobs.get('gp-' + stage + '-' + key) for stage in ('evaluate','publish')]
        if any(r is None for r in pair):
            blocked.add(key)
            continue
        evaluation = json.loads(pair[0]['payload'])
        need(request_identity(Context(**evaluation['context']), evaluation['program']) == key,
             'historical executable request identity')
        untouched = all(r['state'] == 'pending' and r['attempts'] == 0 for r in pair)
        complete = all(r['state'] == 'completed' for r in pair)
        if complete:
            completed.add(key)
        elif not untouched:
            blocked.add(key)
    return {'queue_proof': proof, 'jobs': len(jobs), 'evaluation_keys': len(keys),
            'blocked_evaluation_keys': sorted(blocked), 'completed_evaluation_keys': sorted(completed)}


class VariantConsumer:
    def __init__(self, outbox, inputs, observer, worker, store, *, descriptors,
                 canonicalize, blocked_keys, failure_custody, owner, deadline, completed_keys=()):
        self.outbox, self.inputs, self.observer = outbox, inputs, observer
        self.worker, self.store = worker, store
        need(observer.inputs is inputs and observer.gate.store is store and
             worker.inputs is inputs and worker.store is store and
             worker.queue is observer.gate.queue and
             worker.queue.db is observer.journal.db, 'one exact queue/input/store authority')
        self.descriptors, self.canonicalize = descriptors, canonicalize
        self.blocked_keys = frozenset(blocked_keys)
        self.completed_keys = frozenset(completed_keys)
        for key in self.blocked_keys | self.completed_keys:
            sha256(key)
        self.failure_custody = failure_custody
        self.owner, self.deadline = owner, deadline

    def bind(self, body):
        need(body['model_origin_sha256'] == self.outbox.config['model_origin_sha256'],
             'original model ancestry')
        e, b = body['emission'], body['batch']
        d = e['input_descriptor']
        need(d.get('source_kind') == 'flywheel_variant' and
             d.get('observation_dispatch') == 'qualified_flywheel_worker_adapter',
             'qualified variant dispatch')
        cid = e['context_id']
        need(cid in self.descriptors and d == self.descriptors[cid],
             'exact admitted descriptor, including dataset/encoder publication')
        context, _, actual = self.inputs.describe(d['variant_identity'])
        need(context.identity == cid == d['context_id'] == e['data_run_id'] and
             asdict(context) == d['context'], 'exact executable context')
        need(actual['parent_lineage_sha256'] == d['parent_lineage_sha256'] ==
             e['parent_lineage_sha256'] and actual['partition'] == d['partition'],
             'inherited current parent role')
        phase = {'train': 'training', 'validation': 'validation'}[actual['partition']]
        members = b['emitted_train_data_run_ids'] if phase == 'training' else b['validation_data_run_ids']
        need(e['phase'] == phase and cid in members, 'phase and declared batch membership')
        need(actual['dataset_s3']['receipt'] == d['dataset_publication'],
             'exact registered dataset publication')
        raw = e['raw_program_utf8']
        import hashlib
        need(hashlib.sha256(raw.encode()).hexdigest() == e['raw_program_sha256'],
             'raw generated program bytes')
        program = self.canonicalize(raw)
        key = request_identity(context, program)
        return context, actual, program, key

    def guard_attempt(self, claim, context, key):
        need(key not in self.blocked_keys, 'historical failed/unknown key is never retried')
        for row in self.observer.journal.db.execute(
                "SELECT id,state,attempts FROM jobs WHERE id IN (?,?)",
                ('gp-evaluate-' + key, 'gp-publish-' + key)):
            need(row['state'] not in ('failed', 'needs_reconcile', 'running') and
                 not (row['state'] == 'pending' and row['attempts']),
                 'prior queue attempt is not repeatable')
        # Bind canonical keys across different batch/phase aliases, not just row IDs.
        rows = self.outbox.db.execute(
            "SELECT id,body FROM emissions WHERE id!=? AND state IN "
            "('leased','unknown_attempt','failure_custody_verified','pending_unsupported') "
            "AND json_extract(body,'$.emission.context_id')=?",
            (claim['lease']['emission_id'], context.identity)).fetchall()
        for row in rows:
            prior = json.loads(row['body'])['emission']
            if prior['input_descriptor'].get('source_kind') != 'flywheel_variant':
                continue
            need(request_identity(context, self.canonicalize(prior['raw_program_utf8'])) != key,
                 'unresolved or failed canonical alias is never retried')

    def guard_queue(self, key):
        allowed = {'gp-evaluate-' + key, 'gp-publish-' + key}
        for row in self.worker.queue.db.execute(
                "SELECT id,state FROM jobs WHERE state!='completed'"):
            need(row['id'] in allowed and row['state'] == 'pending',
                 'unrelated or unresolved queue work forbids execution')

    def verified_success(self, body, outcome):
        context, _, program, key = self.bind(body)
        need(outcome['evaluation_key'] == key, 'outcome evaluation identity')
        entry = self.store.lookup(key)
        need(entry is not None, 'publication must be authoritatively present')
        measured = verified_measurement(self.store, context, program, entry)
        ref = self.store.record_refs[key]
        need(outcome['record_ref'] == ref, 'exact successful wrapper reference')
        raw = object_bytes(self.store, ref)
        need(json.loads(raw) == entry,
             'successful wrapper exact-version closure')
        return measured['evaluation_key'] == key

    def once(self):
        remaining = int(self.deadline - self.outbox.now())
        need(remaining >= 600, 'reserve complete bounded evaluation/publication lease')
        claim = claim_variant(self.outbox, self.owner, lease_seconds=600,
                              deadline=self.deadline)
        if claim is None:
            return {'state': 'empty', 'optimizer_updates': 0}
        body, lease = claim['body'], claim['lease']
        try:
            context, actual, program, key = self.bind(body)
            self.guard_attempt(claim, context, key)
            observed = self.observer.observe({
                'variant_identity': body['emission']['input_descriptor']['variant_identity'],
                'descriptor_sha256': digest(actual), 'context': asdict(context),
                'dsl': program, 'phase': body['emission']['phase'],
                'checkpoint_sha256': body['batch']['checkpoint_sha256'],
                'step': body['batch']['global_step'], 'producer': self.owner})
            cached = observed['state'] == 'ready'
            need(cached or key not in self.completed_keys,
                 'historical completed measurement requires verified cached success; never rerun')
            if not cached:
                # The existing queue must run evaluate, then its dependent publish.
                for stage in ('evaluate', 'publish'):
                    self.guard_queue(key)
                    result = self.worker.once()
                    need(result.get('state') == 'completed' and result.get('stage') == stage and
                         result.get('job') == 'gp-' + stage + '-' + key,
                         'exact dependent measured job completed once')
            self.store.lookup(key)  # Capture the exact wrapper reference.
            outcome = {'state': 'cached_verified' if cached else 'published_verified',
                       'evaluation_key': key, 'record_ref': self.store.record_refs[key],
                       'attempt_custody_ref': None, 'error': None}
            self.outbox.finish(lease, outcome, verify_outcome=self.verified_success)
            return {'state': outcome['state'], 'emission_id': lease['emission_id'],
                    'evaluation_key': key, 'record_ref': outcome['record_ref'],
                    'phase': body['emission']['phase'], 'optimizer_updates': 0,
                    'optimizer_reward_allowed': False}
        except BaseException as error:
            # No numeric failure score and no speculative retry. A successful S3
            # write followed by a timeout remains unknown until separate readback.
            message = type(error).__name__ + ': ' + str(error)[:3500]
            ref = None
            try:
                # Until a separately qualified exact-version custody verifier is
                # supplied, the callback may seal local evidence only. A format
                # shaped reference alone is never a verified S3 failure receipt.
                need(self.failure_custody(claim, message) is None,
                     'unqualified failure custody reference')
            except BaseException:
                pass  # Partial custody publication is itself retained as unknown.
            outcome = {'state': 'unknown_attempt', 'evaluation_key': None,
                       'record_ref': None, 'attempt_custody_ref': ref, 'error': message}
            self.outbox.finish(lease, outcome, verify_outcome=lambda actual, result:
                               actual == body and result == outcome)
            return {'state': 'unknown_attempt', 'emission_id': lease['emission_id'],
                    'attempt_custody_ref': ref, 'error': message,
                    'automatic_retry': False, 'optimizer_updates': 0}
