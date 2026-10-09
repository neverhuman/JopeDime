"""Durable emission boundary: record every bounded request before validation.

Callers must not expose a generated candidate to scoring/training before this
function returns. A ready research measurement still cannot be an optimizer
reward without a separate frozen-manifest admission decision.
"""
from dataclasses import asdict
import json
import time
import uuid
from .program_gate import ProgramGate
from .program_io import need
from .protocol import canonical, digest


class Observations:
    def __init__(self, queue, *, maximum=100000):
        self.db, self.maximum = queue.db, maximum
        need(type(maximum) is int and 1 <= maximum <= 100000, 'bounded observations')
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS program_emissions(seq INTEGER PRIMARY KEY, id TEXT NOT NULL,
            at REAL NOT NULL, kind TEXT NOT NULL, body TEXT NOT NULL, previous TEXT NOT NULL, sha TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS program_inputs(context_id TEXT PRIMARY KEY,
            variant_identity TEXT NOT NULL, descriptor_sha TEXT NOT NULL, descriptor TEXT NOT NULL);
        ''')

    def append(self, identity, kind, body):
        data = canonical(body); need(len(data) <= 65536, 'observation payload cap')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            tail = self.db.execute('SELECT seq,sha FROM program_emissions ORDER BY seq DESC LIMIT 1').fetchone()
            seq, previous = (tail['seq']+1, tail['sha']) if tail else (1, '0'*64)
            if kind == 'observed':
                need(self.db.execute("SELECT count(*) FROM program_emissions WHERE kind='observed'").fetchone()[0] < self.maximum,
                     'observation capacity reached; producer must stop')
            need(seq <= 2*self.maximum, 'observation journal capacity')
            event = {'seq':seq,'id':identity,'at':time.time(),'kind':kind,'body':body,'previous':previous}
            self.db.execute('INSERT INTO program_emissions VALUES(?,?,?,?,?,?,?)',
                            (seq,identity,event['at'],kind,data.decode(),previous,digest(event)))
            self.db.execute('COMMIT')
            return event
        except BaseException:
            if self.db.in_transaction: self.db.execute('ROLLBACK')
            raise

    def register(self, context, descriptor):
        encoded = canonical(descriptor).decode(); value = digest(descriptor)
        self.db.execute('INSERT OR IGNORE INTO program_inputs VALUES(?,?,?,?)',
                        (context.identity,descriptor['variant_identity'],value,encoded))
        old = self.db.execute('SELECT * FROM program_inputs WHERE context_id=?',(context.identity,)).fetchone()
        need(old['descriptor_sha'] == value and old['descriptor'] == encoded, 'registered context collision')

    def verify(self):
        previous, seq, pending = '0'*64, 0, set()
        for row in self.db.execute('SELECT * FROM program_emissions ORDER BY seq'):
            seq += 1
            event = {k:row[k] for k in ('seq','id','at','kind','previous')}; event['body'] = json.loads(row['body'])
            need(row['seq'] == seq and row['previous'] == previous and digest(event) == row['sha'], 'emission journal integrity')
            if row['kind'] == 'observed':
                need(row['id'] not in pending, 'duplicate observation identity'); pending.add(row['id'])
            elif row['kind'] == 'outcome':
                need(row['id'] in pending, 'outcome without observation'); pending.remove(row['id'])
            else: raise ValueError('unknown observation event')
            previous = row['sha']
        return {'events':seq,'head_sha256':previous,'unresolved_observations':sorted(pending)}


class ProgramObserver:
    def __init__(self, queue, inputs, store, *, plan_sha256, canonicalize):
        self.inputs = inputs; self.journal = Observations(queue)
        self.gate = ProgramGate(queue,store,plan_sha256=plan_sha256,canonicalize=canonicalize)

    def observe(self, request):
        # Only a finite JSON envelope bound precedes durable capture. All semantic
        # errors, unknown contexts, invalid DSL and failed lookups are journaled.
        need(type(request) is dict and len(canonical(request)) <= 60000, 'bounded JSON emission envelope')
        identity = uuid.uuid4().hex
        self.journal.append(identity,'observed',request)
        try:
            need(set(request) == {'variant_identity','descriptor_sha256','context','dsl','phase','checkpoint_sha256','step','producer'},
                 'exact emission envelope fields')
            need(type(request['producer']) is str and 0 < len(request['producer']) <= 200, 'named producer required')
            context, _, descriptor = self.inputs.describe(request['variant_identity'])
            need(request['context'] == asdict(context) and request['descriptor_sha256'] == digest(descriptor), 'producer input/context binding')
            self.journal.register(context,descriptor)
            result = self.gate.ensure(context,request['dsl'],phase=request['phase'],
                checkpoint_sha256=request['checkpoint_sha256'],step=request['step'])
            outcome = {'observation_id':identity,'state':result.state,'evaluation_key':result.evaluation_key,
                       'measurement':result.measurement,'teaching_eligible':False,'optimizer_reward_allowed':False}
            self.journal.append(identity,'outcome',outcome)
            return outcome
        except BaseException as error:
            self.journal.append(identity,'outcome',{'state':'error','error':type(error).__name__+': '+str(error)[:1000],
                                                   'teaching_eligible':False,'optimizer_reward_allowed':False})
            raise


def require_optimizer_admission(observation):
    # This research bridge has no authority to override held-out lineage roles,
    # missing encoder blocks, source licensing or frozen training manifests.
    raise RuntimeError('research program measurement is not optimizer/training admission')
