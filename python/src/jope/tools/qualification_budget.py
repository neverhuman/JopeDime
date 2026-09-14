"""One durable budget domain; output directories never create funding."""
from contextlib import contextmanager
import fcntl
import hashlib
import math
from pathlib import Path
import time

import experiment_runtime as runtime
from promotion_gate import canonical, immutable
from resume_expired_cycle import read, require, sha
from funding_client import NativeFundingClient


# Installation authority location is independent of the loaded source/release
# path. Moving a reviewed release cannot select another funding domain.
WORKSPACE_DOMAIN = Path('/home/ubuntu/jope-dime/parallel/runs/qualification-budget')
CANONICAL_DOMAIN = WORKSPACE_DOMAIN


def require_isolated_domain(directory):
    # Production storage belongs to the native authority. This guard prevents
    # isolated Python journal replay from being mistaken for that authority.
    require(directory.resolve() != WORKSPACE_DOMAIN.resolve(),
            'production budget authorization verifier unavailable')


def amount(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


@contextmanager
def locked(directory):
    with (directory / 'budget.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def initialize_domain(authorization_path, expected_sha256):
    """Explicit migration operation, never called by a task or reservation.

    The integrator supplies the reviewed authorization and complete historical
    ledger inventory. Imported ceilings, including unknown consumption, remain
    charged. This operation does not claim that the inventory is complete.
    """
    authorization_path = Path(authorization_path)
    require(sha(authorization_path) == expected_sha256, 'budget authorization changed')
    authorization = read(authorization_path)
    if (CANONICAL_DOMAIN.resolve() == WORKSPACE_DOMAIN.resolve()
            and isinstance(authorization.get('body'), dict)
            and authorization['body'].get('schema') == 'parallel.funding_authorization.v1'):
        # Explicit original signed authorization import. The independently
        # installed native trust, original history/prerequisite approval and
        # durable authority transaction govern admission; no local ledger write.
        return NativeFundingClient().request('register', authorization=authorization)
    require(authorization.get('schema') == 'parallel.budget_authorization.v2', 'unsupported budget authorization')
    # Full two-host legacy-ledger coverage is not available yet. Do not turn
    # caller-supplied hashes or an empty history list into a production grant.
    require(authorization.get('scope') == 'private_synthetic_qualification'
            and CANONICAL_DOMAIN.resolve() != WORKSPACE_DOMAIN.resolve(),
            'production budget initialization unavailable: complete two-host ledger coverage required')
    require(isinstance(authorization.get('domain_id'), str) and authorization['domain_id'], 'budget domain missing')
    require(isinstance(authorization.get('approved_by'), str) and authorization['approved_by'], 'budget authorizer missing')
    require(amount(authorization.get('expires_at')) and authorization['expires_at'] > time.time(), 'budget authorization expired')
    require(all(amount(authorization.get(key)) for key in ('core_seconds', 'node_seconds')), 'invalid budget ceiling')
    require(isinstance(authorization.get('historical_ledgers'), list), 'historical ledger inventory required')
    require(isinstance(authorization.get('capabilities'), list) and authorization['capabilities'], 'budget capabilities missing')
    history = {}
    for ref in authorization['historical_ledgers']:
        path = Path(ref['path'])
        require(sha(path) == ref['sha256'], 'historical budget ledger changed')
        previous = read(path)
        require(previous.get('schema') == 'parallel.qualification_budget.v1', 'unsupported historical budget ledger')
        for identity, row in previous['reservations'].items():
            require(all(amount(row.get(key)) for key in ('core_seconds', 'node_seconds')), 'invalid historical charge')
            require(type(row.get('searches')) is int and 0 <= row['searches'] <= 16, 'invalid historical allocation')
            # Distinct origins remain charged even when local names coincide.
            key = digest(dict(path=str(path.resolve()), identity=identity))
            history['historical-' + key] = dict(row, imported=True, original_identity=identity,
                original_ledger=dict(path=str(path.resolve()), sha256=ref['sha256']),
                complete=row.get('complete') is True)
    CANONICAL_DOMAIN.mkdir(parents=True, exist_ok=True)
    with locked(CANONICAL_DOMAIN):
        require(not (CANONICAL_DOMAIN / 'domain.json').exists()
                and not (CANONICAL_DOMAIN / 'journal.jsonl').exists(), 'budget domain already exists; reset forbidden')
        immutable(CANONICAL_DOMAIN / 'authorization.json', authorization)
        # Interrupted initialization requires reconciliation, not a new grant.
        runtime.append(CANONICAL_DOMAIN / 'journal.jsonl', dict(sequence=0, parent_sha256=None,
            event='history_imported', reservations=history, authorization_sha256=sha(CANONICAL_DOMAIN / 'authorization.json')))
        immutable(CANONICAL_DOMAIN / 'domain.json', dict(schema='parallel.budget_domain.v2',
            domain_id=authorization['domain_id'], authorization_sha256=sha(CANONICAL_DOMAIN / 'authorization.json')))


class Budget:
    def __init__(self, output):
        # Output is attribution only. It cannot choose the ledger.
        self.output = Path(output).resolve()
        self.directory = CANONICAL_DOMAIN
        self.path = self.directory / 'journal.jsonl'
        self.native = (NativeFundingClient()
                       if self.directory.resolve() == WORKSPACE_DOMAIN.resolve() else None)
        if self.native is not None:
            return
        require_isolated_domain(self.directory)
        require((self.directory / 'domain.json').is_file() and self.path.is_file(),
                'canonical budget domain unavailable; reviewed history and funding initialization required')

    def _state(self):
        require_isolated_domain(self.directory)
        domain = read(self.directory / 'domain.json')
        require(domain.get('schema') == 'parallel.budget_domain.v2', 'unsupported budget domain')
        require(sha(self.directory / 'authorization.json') == domain['authorization_sha256'], 'budget authorization changed')
        authorization = read(self.directory / 'authorization.json')
        require(authorization.get('scope') == 'private_synthetic_qualification',
                'production budget authorization verifier unavailable')
        require(authorization['domain_id'] == domain['domain_id'], 'budget domain changed')
        require(self.path.read_bytes().endswith(b'\n'), 'incomplete budget journal; reconciliation required')
        events = runtime.journal(self.path)
        require(events and events[0].get('event') == 'history_imported', 'budget history missing')
        require(events[0]['authorization_sha256'] == domain['authorization_sha256'], 'budget history authorization changed')
        state, parent = {}, None
        for sequence, event in enumerate(events):
            require(event.get('sequence') == sequence and event.get('parent_sha256') == parent, 'budget journal lineage changed')
            name = event['event']
            if name == 'history_imported':
                require(sequence == 0, 'budget history reset forbidden')
                state.update(event['reservations'])
            elif name == 'reserved':
                require(event['identity'] not in state, 'duplicate budget reservation')
                state[event['identity']] = event['reservation']
            elif name in ('consumed', 'completed'):
                require(event['identity'] in state, 'orphan budget event')
                state[event['identity']].update(event['update'])
            else:
                raise RuntimeError('unknown budget event')
            parent = digest(event)
        return authorization, state, len(events), parent

    def _append(self, sequence, parent, event, **details):
        runtime.append(self.path, dict(sequence=sequence, parent_sha256=parent, event=event, **details))

    def reserve(self, identity, *, core_seconds=0, node_seconds=0, searches=0,
                capability='qualification', task_sha256=None, deadline=None):
        require(self.native is None,
                'production funding requires reserve_task with original signed authorization, reservation and task descriptor')
        require(isinstance(identity, str) and identity, 'reservation identity required')
        require(type(searches) is int and 0 <= searches <= 16, 'at most sixteen concurrent qualification searches')
        require(amount(core_seconds) and amount(node_seconds) and core_seconds + node_seconds > 0, 'invalid reservation')
        with locked(self.directory):
            authorization, state, sequence, parent = self._state()
            require(time.time() < authorization['expires_at'], 'budget authorization expired')
            require(capability in authorization['capabilities'], 'unfunded capability')
            require(identity not in state, 'reservation already consumed; restart requires a new attempt identity')
            rows = state.values()
            require(sum(r['core_seconds'] for r in rows) + core_seconds <= authorization['core_seconds'], 'CPU qualification cap reached')
            require(sum(r['node_seconds'] for r in rows) + node_seconds <= authorization['node_seconds'], 'GPU qualification cap reached')
            require(sum(r['searches'] for r in rows if not r['complete']) + searches <= 16, 'qualification worker allocation exhausted')
            if capability == 'regression_training':
                require(isinstance(task_sha256, str) and len(task_sha256) == 64
                        and all(c in '0123456789abcdef' for c in task_sha256), 'training task identity missing')
                require(amount(deadline) and time.time() < deadline <= authorization['expires_at'], 'training deadline invalid')
                require(node_seconds >= deadline - time.time(), 'training node reservation below deadline ceiling')
            row = dict(core_seconds=core_seconds, node_seconds=node_seconds, searches=searches,
                complete=False, consumed=False, capability=capability, task_sha256=task_sha256,
                deadline=deadline, output=str(self.output), domain_id=authorization['domain_id'])
            self._append(sequence, parent, 'reserved', identity=identity, reservation=row)
            return row

    def verify(self, ref, task_sha256, *, consume=False):
        require(self.native is None,
                'production funding requires preflight_task; numerical slots are consumed only by the native authority and original peer')
        require(isinstance(ref, dict) and set(ref) == {'domain_id', 'identity', 'task_sha256'}, 'valid task reservation required')
        with locked(self.directory):
            authorization, state, sequence, parent = self._state()
            require(ref['domain_id'] == authorization['domain_id'], 'reservation domain changed')
            row = state.get(ref['identity'])
            require(row and not row['complete'] and not row.get('consumed'), 'reservation unavailable or already consumed')
            require(row.get('capability') == 'regression_training'
                    and row.get('task_sha256') == ref['task_sha256'] == task_sha256, 'reservation task identity changed')
            require(time.time() < min(row['deadline'], authorization['expires_at']), 'training reservation expired')
            if consume:
                self._append(sequence, parent, 'consumed', identity=ref['identity'],
                    update=dict(consumed=True, consumed_at=time.time()))
            return dict(row)

    def complete(self, identity, receipt):
        require(self.native is None,
                'production completion requires original native witnessed termination; Python receipts cannot release funding custody')
        with locked(self.directory):
            _, state, sequence, parent = self._state()
            require(identity in state, 'unknown reservation')
            row = state[identity]
            ref = dict(path=str(Path(receipt).resolve()), sha256=sha(receipt))
            if row['complete']:
                require(row['receipt'] == ref, 'completed reservation changed')
                return
            # Lost, completed, and crashed tasks never refund their ceilings.
            self._append(sequence, parent, 'completed', identity=identity, update=dict(complete=True, receipt=ref))

    def reserve_task(self, *, authorization, reservation, descriptor_json):
        """Register an exact independently signed task; never invent a ceiling."""
        require(self.native is not None, 'typed native task reservation requires the installation funding authority')
        require(isinstance(authorization, dict) and isinstance(reservation, dict)
                and isinstance(descriptor_json, dict), 'signed originals and sealed common task descriptor required')
        return self.native.request('reserve', authorization=authorization, reservation=reservation,
                                   descriptor_json=descriptor_json)

    def preflight_task(self, *, authorization, reservation, descriptor_json):
        """Read-only verification; this cannot consume a rank or launch work."""
        require(self.native is not None, 'typed native task preflight requires the installation funding authority')
        require(isinstance(authorization, dict) and isinstance(reservation, dict)
                and isinstance(descriptor_json, dict), 'signed originals and sealed common task descriptor required')
        return self.native.request('preflight', authorization=authorization, reservation=reservation,
                                   descriptor_json=descriptor_json)
