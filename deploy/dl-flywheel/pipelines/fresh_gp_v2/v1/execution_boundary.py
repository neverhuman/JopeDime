"""Last native launch boundary; injected only into the new worker source copy."""
import json
import time
from jope_flywheel.program_gate import request_identity
from jope_flywheel.program_io import bound, need, sha
from jope_flywheel.protocol import canonical, digest
from global_attempts import ownership


class ExecutionBoundary:
    def __init__(self, config, queue, inputs, attempts, history, owner, ownership_reader):
        self.config, self.queue, self.inputs = config, queue, inputs
        self.attempts, self.history, self.owner, self.ownership_reader = attempts, history, owner, ownership_reader
        self.work = config['work_deadline_unix']; self.global_end = config['global_deadline_unix']
        self.mono_end = time.monotonic()+max(0,min(self.work,self.global_end)-time.time())

    def live_job(self, job):
        row = self.queue.db.execute('SELECT * FROM jobs WHERE id=?',(job['id'],)).fetchone()
        need(row is not None and row['state']=='running' and row['lease_token']==job['lease_token'] and
             row['stage']==job['stage'] and row['plan_sha']==job['plan_sha256'] and
             json.loads(row['payload'])==job['payload'] and row['attempts']==1 and
             row['lease_deadline']==job['lease_deadline'], 'actual live job/lease/body changed')
        now=time.time()
        deadline=min(row['lease_deadline'],self.work,self.global_end,now+self.mono_end-time.monotonic())
        need(deadline > now, 'lease/work/global deadline elapsed during preparation')
        return deadline

    def check(self, job, context, program, raw, descriptor):
        for ref in self.config['source_pins']:
            bound(ref,64*1024**2)
        # Exact source replicas, replayed bytes, fresh current roles, source S3
        # versions, Kumo qualification and preserved validation role rechecked.
        current, actual, latest = self.inputs.describe(descriptor['variant_identity'])
        need(current==context and actual==raw and canonical(latest)==canonical(descriptor), 'final input/descriptor drift')
        need(descriptor['teaching_eligible'] is False and descriptor['partition'] in ('train','validation'),
             'evaluation is not optimizer admission')
        self.inputs.lease.check(descriptor['variant_identity'],self.inputs.parents[descriptor['parent_lineage_sha256']])
        self.attempts.assert_owner(context,descriptor,self.owner)
        proof=self.ownership_reader()
        ownership(proof,self.config,self.config['context_descriptors'],time.time())
        history=self.history.check()
        self.live_job(job)
        return history, proof

    def before_fit(self, job, context, program, raw, descriptor):
        history,proof=self.check(job,context,program,raw,descriptor)
        for ref in self.config['source_pins']:
            bound(ref,64*1024**2)
        # Pin the queue row under its real transaction after expensive replay;
        # permanent intent commits before native execution. Crash or later
        # expiry conservatively consumes this key forever.
        self.queue.db.execute('BEGIN IMMEDIATE')
        try:
            self.live_job(job)
            self.attempts.begin_attempt(context,program,self.owner,job,history,self.history.pin)
            self.live_job(job)
            self.queue.db.execute('COMMIT')
        except BaseException:
            if self.queue.db.in_transaction:self.queue.db.execute('ROLLBACK')
            raise
        # Registry commit/fsync may itself consume the remaining lease.
        self.inputs.lease.check(descriptor['variant_identity'],self.inputs.parents[descriptor['parent_lineage_sha256']])
        proof=self.ownership_reader()
        ownership(proof,self.config,self.config['context_descriptors'],time.time())
        return min(self.live_job(job),proof['lease_expires_unix'])

    def before_cached(self, job, context, program, raw, descriptor, entry):
        self.check(job,context,program,raw,descriptor)
        need(entry['evaluation_key']==request_identity(context,program), 'cached exact context/program')
        # The unchanged worker follows with full verified_measurement. A cache
        # verification failure remains a failure, never falls through to fit.

    def before_publication(self, job, context, program, raw, descriptor):
        self.check(job,context,program,raw,descriptor)
        # Cached paths may have no new local attempt; the immutable evaluation
        # result and unchanged worker verifier decide cache/measurement custody.
        return self.live_job(job)
