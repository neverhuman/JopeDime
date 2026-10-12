"""C10b revision 7 (every prepared root is consumed only with its operator acceptance envelope -- exact terminal
marker bytes, the sealed original bound, physical in-time success -- pinned in the candidate variant component) --
revision 6 (overlay components: a candidate may append native-result overlay packets to the active chain;
the snapshot is the composite (variant base E, overlay head H); restart and exposures follow the chain) -- revision 3: a first exposure is claimed only against a complete committed-batch
census; with any earlier committed batch unproved it is reported as earliest OBSERVED, partial): sequenced, repeatable closed-epoch adoption of prepared candidates (library API, no executor).

Cold successor of the HELD C10 v1 packet (DELIVERY 3290153b; v1 bytes preserved unchanged).  The sole training
owner (root's executor) calls, at every fully completed epoch boundary and before reserving the next batch:

    state = inbox.reconcile(planner)              # replay and verify every link; finalize a proved CAS or HOLD
    replacement = inbox.adopt_next(planner, current, native, ...)
    if replacement is not None: current = replacement

Revision 2 closes the independent review findings B1-B4 (CODEX_C10_INDEPENDENT_FINDINGS_20261009_v1.md):

* B1  every hydration -- publication, adoption and restart (``load_adopted``) -- goes through one strict read-only
      root validator: descriptor-pinned config/status/prepared bytes read without following symlinks (current files
      are never trusted by name), the owned-source closure, the qualified hydration and adapter implementations,
      the sealed C10a receipt bound by the candidate envelope, and the campaign target contract.
* B2  ``reconcile`` replays every inbox record against the full verified planner event chain and the exact stored
      manifest bodies: each adoption joins its intent and candidate, the ``manifest.adopted`` event immediately after
      the intent's planner head, the completed epoch (the head itself, with no unconsumed member) and the preceding
      committed checkpoint.  A broken existing link refuses restart (``Refused``); an open intent whose joins fail,
      or any unexplained planner state, is a HOLD; a CAS is never attempted again.
* B3  first exposures are derived from the sealed C10a receipts and the exact composed result references, joined
      to verified committed batches in optimizer order (deduplicated, conflicting proofs refused), and count across
      verified descendant adoptions that retain the exact reference.
* B4  one authority proof hydrates; a second fresh proof is taken after hydration, immediately before the intent
      and the CAS, rechecked inside the CAS call, and both digests are bound into the intent and the record.  A
      refusal writes an explicit HOLD; nothing is replayed.

Candidate envelopes carry a component map.  The variant (X1/C10a) component is implemented; the native-result
overlay component is reserved for its reviewed package and refused until then.  Nothing here reserves, trains,
replays an optimizer update, retries an adoption, or touches S3/GPU/network state.
"""
import json
import os
import sqlite3
import time
from pathlib import Path

from jope_flywheel.program_io import bound, need, read, sha
from jope_flywheel.protocol import canonical, digest
from jope_flywheel import native_variant_snapshot as V
from jope_flywheel import training_snapshot_kumos as K
from pipelines.generated_target_admission_v1.common import document
from pipelines.generated_target_admission_v1.transition import validate_extension_transition
from pipelines.generated_target_preparation_v2 import preparation as C
import prepared_admission as PA
import prepared_snapshot as P

GENESIS = 'jope.flywheel-boundary-genesis/v2'
CANDIDATE = 'jope.flywheel-boundary-candidate/v2'
INTENT = 'jope.flywheel-boundary-adoption-intent/v2'
ADOPTION = 'jope.flywheel-boundary-adoption/v2'
HOLD = 'jope.flywheel-boundary-hold/v2'
EXPOSURE = 'jope.flywheel-first-exposure/v3'
COMPOSITE = 'jope.flywheel-composite-snapshot/v1'
DESCRIPTOR_KEYS = frozenset({'schema', 'campaign_sha256', 'approved_for_bounded_research', 'preparation_config',
                             'preparation_status', 'prepared'})
COMPONENTS = frozenset({'variant', 'native_overlay'})
STATE_KEYS = frozenset({'variant_extension_sha256', 'variant_source_seq', 'overlay_head_sha256', 'overlay_packets'})
OVERLAY_KEYS = frozenset({'packets', 'head_sha256', 'predecessor_head_sha256', 'receipt_sha256'})
VARIANT_KEYS = frozenset({'descriptor', 'descriptor_sha256', 'extension_sha256', 'preparation_receipt_sha256',
                          'preparation_acceptance'})
JOINED = ('seq', 'candidate_sha256', 'closed_epoch', 'previous_snapshot_sha256', 'snapshot_sha256', 'manifest_before',
          'manifest_after', 'checkpoint_sha256', 'hydration_authority_sha256', 'cas_authority_sha256',
          'planner_head_before', 'epoch_completed_event', 'last_commit_event', 'contract_sha256')
MAX_DOC = 32 * 1024**2


class Held(ValueError):
    """Adoption is blocked until an operator rules on a recorded HOLD."""


class Refused(ValueError):
    """An existing inbox/planner link is broken; restart must not proceed."""


def _create(path, body):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        view = memoryview(body)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)
    dfd = os.open(Path(path).parent, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def _read(path):
    raw = read(Path(path), MAX_DOC)
    return raw, document(raw, MAX_DOC)


def _name(seq):
    need(type(seq) is int and 0 < seq < 10**8, 'positive bounded sequence number')
    return f'{seq:08d}.json'


def composite_identity(variant_extension_sha256, native_overlay_head_sha256):
    """Snapshot identity: the variant extension alone, or bound with a native-result overlay head."""
    if native_overlay_head_sha256 is None:
        return variant_extension_sha256
    return digest({'schema': COMPOSITE, 'variant_extension_sha256': variant_extension_sha256,
                   'native_overlay_head_sha256': native_overlay_head_sha256})


def validate_root(descriptor, *, campaign_sha256, contract, acceptance, receipt_sha256=None):
    """B1: the one strict read-only validator for a prepared root (publication, adoption and restart).

    Equivalent to ``prepared_admission.candidate``'s descriptor/ref/config/status/owned-source/adapter closure, but
    usable without an old snapshot, and additionally bound to the sealed C10a receipt and the target contract.
    Every byte comes from descriptor-pinned references read without following symlinks.
    """
    need(type(descriptor) is dict and set(descriptor) == DESCRIPTOR_KEYS and
         descriptor['schema'] == 'jope.prepared-epoch-candidate/v1' and
         descriptor['campaign_sha256'] == campaign_sha256 and descriptor['approved_for_bounded_research'] is True,
         'closed prepared candidate descriptor for this campaign')
    config_raw = bound(descriptor['preparation_config'], MAX_DOC)
    status_raw = bound(descriptor['preparation_status'], MAX_DOC)
    prepared_raw = bound(descriptor['prepared'], MAX_DOC)
    config, status = document(config_raw, MAX_DOC), document(status_raw, MAX_DOC)
    root = Path(config['owned_root'])
    need(root.is_absolute() and config['schema'] == C.CONFIG_SCHEMA, 'owned CPU preparation schema')
    for ref, name in ((descriptor['preparation_config'], 'config.json'),
                      (descriptor['preparation_status'], 'status.json'), (descriptor['prepared'], 'prepared.json')):
        need(Path(ref['path']) == root / name, 'one immutable preparation root')
    need(status['schema'] == C.STATUS_SCHEMA and status['state'] == 'passed' and
         status['config_sha256'] == sha(config_raw) and status['prepared'] == descriptor['prepared'] and
         status['extension_sha256'] == config['extension_sha256'], 'pinned successful receipt and configuration closure')
    need(all(config[k] == status[k] == 0 for k in ('network_gets', 's3_writes', 'optimizer_updates', 'planner_mutations')),
         'preparation cannot mutate optimizer or admission')
    for rel, pin in config['owned_sources'].items():
        p = Path(rel)
        need(not p.is_absolute() and len(p.parts) == 1 and '..' not in p.parts and sha(read(root / p, MAX_DOC)) == pin,
             'exact preparation source closure')
    need(status['prepared_source_sha256'] == config['owned_sources']['prepared_snapshot.py'] ==
         sha(read(Path(P.__file__).resolve())), 'qualified hydration implementation')
    need(status['source_adapter_sha256'] == sha(read(Path(V.__file__).resolve())), 'qualified measured record adapter')
    extension_raw = read(root / 'extension.json', MAX_DOC)
    need(sha(extension_raw) == config['extension_sha256'], 'pinned extension')
    receipt_raw = read(root / 'c10a-receipt.json', MAX_DOC)
    need(receipt_sha256 is None or sha(receipt_raw) == receipt_sha256, 'sealed preparation receipt')
    terminal_raw = read(root / 'c10a-terminal.json', MAX_DOC)
    C.check_terminal(terminal_raw, status_raw=status_raw, receipt_raw=receipt_raw)
    C.check_root_acceptance(acceptance, root=root, status_raw=status_raw, receipt_raw=receipt_raw,
                            terminal_raw=terminal_raw)
    receipt = document(receipt_raw, MAX_DOC)
    need(receipt['schema'] == C.RECEIPT_SCHEMA and receipt['root'] == str(root) and
         receipt['config'] == descriptor['preparation_config'] and receipt['status'] == descriptor['preparation_status'] and
         receipt['prepared'] == descriptor['prepared'] and receipt['extension_sha256'] == config['extension_sha256'],
         'preparation receipt binds this exact root')
    need(receipt['contract'] == contract and config['origin']['contract_sha256'] == digest(contract),
         'campaign target contract')
    return {'config_raw': config_raw, 'config': config, 'status_raw': status_raw, 'status': status,
            'prepared_raw': prepared_raw, 'extension_raw': extension_raw, 'receipt_raw': receipt_raw,
            'receipt': receipt, 'acceptance': acceptance}


def hydrate(doc, native, *, preprocessing_sha256, registry_sha256, authority_check):
    """Hydrate exactly the validated bytes (never re-read by name)."""
    return P.load(doc['prepared_raw'], native, receipt=doc['status'],
                  expected_extension_sha256=doc['config']['extension_sha256'],
                  expected_preprocessing_sha256=preprocessing_sha256,
                  expected_parent_registry_sha256=registry_sha256, authority_check=authority_check)


def planner_events(planner):
    """The verified planner event chain as decoded rows (refuses a tampered chain)."""
    head = planner.verify()
    rows = planner.db.execute('SELECT seq,kind,body,previous,sha FROM events ORDER BY seq').fetchall()
    events = [{'seq': r[0], 'kind': r[1], 'body': json.loads(r[2]), 'previous': r[3], 'sha': r[4]} for r in rows]
    need(len(events) == head['events'] and (events[-1]['sha'] if events else '0' * 64) == head['head_sha256'],
         'planner events are the verified chain')
    return head, events


def _manifest_body(planner, manifest_sha256):
    row = planner.db.execute('SELECT body FROM manifests WHERE sha=?', (manifest_sha256,)).fetchone()
    need(row is not None and sha(bytes(row[0])) == manifest_sha256, 'stored manifest body')
    return document(bytes(row[0]), MAX_DOC)


class Inbox:
    """One executor-owned inbox directory; every file is create-only and fsynced."""

    def __init__(self, root, *, fault=None, overlay=None, max_padded_tokens=8192):
        """``overlay`` is the reviewed native-result overlay interface (validate_chain, head_from_sha,
        overlay_snapshot); without it any overlay component refuses."""
        self.overlay, self.max_padded_tokens = overlay, max_padded_tokens
        self.root = Path(root)
        need(self.root.is_absolute() and (self.root / 'GENESIS.json').is_file(), 'initialized absolute inbox')
        _, self.genesis = _read(self.root / 'GENESIS.json')
        need(self.genesis['schema'] == GENESIS and self.genesis['contract_sha256'] == digest(self.genesis['contract']),
             'inbox genesis schema and contract')
        self.fault = fault or (lambda point: None)

    @classmethod
    def create(cls, root, *, genesis_snapshot_sha256, genesis_manifest_sha256, campaign_sha256, contract, **kwargs):
        root = Path(root)
        need(root.is_absolute() and not root.exists(), 'new absolute inbox root')
        need(type(contract) is dict and set(contract) == C.CONTRACT_KEYS and contract['schema'] == C.CONTRACT_SCHEMA,
             'closed campaign target contract')
        root.mkdir(mode=0o755)
        for sub in ('candidates', 'intents', 'adoptions', 'holds'):
            (root / sub).mkdir(mode=0o755)
        _create(root / 'GENESIS.json', canonical({
            'schema': GENESIS, 'snapshot_sha256': genesis_snapshot_sha256, 'manifest_sha256': genesis_manifest_sha256,
            'campaign_sha256': campaign_sha256, 'contract': contract, 'contract_sha256': digest(contract)}))
        return cls(root, **kwargs)

    def _seqs(self, sub):
        seqs = sorted(int(p.name[:8]) for p in (self.root / sub).iterdir() if p.name.endswith('.json'))
        need(seqs == list(range(1, len(seqs) + 1)), 'contiguous ' + sub + ' sequence')
        return seqs

    def candidate(self, seq):
        return _read(self.root / 'candidates' / _name(seq))

    def adoption_files(self):
        return [_read(self.root / 'adoptions' / _name(s)) for s in self._seqs('adoptions')]

    def adoptions(self):
        return [doc for _, doc in self.adoption_files()]

    def holds(self):
        return [document(read(p, MAX_DOC), MAX_DOC) for p in sorted((self.root / 'holds').iterdir())]

    def _hold(self, seq, reason, evidence):
        path = self.root / 'holds' / _name(seq)
        if not path.exists():
            _create(path, canonical({'schema': HOLD, 'seq': seq, 'reason': reason, 'evidence': evidence,
                                     'at_utc_ns': time.time_ns()}))

    def _prior_state(self, seq):
        """Cumulative (variant base, overlay chain) state before candidate ``seq``; genesis has no overlay."""
        if seq == 1:
            return {'variant_extension_sha256': self.genesis['snapshot_sha256'], 'variant_source_seq': 0,
                    'overlay_head_sha256': None, 'overlay_packets': []}
        return self.candidate(seq - 1)[1]['state']

    def _fold_head(self, head, packets):
        need(self.overlay is not None, 'native-result overlay package required for an overlay component')
        for ref in packets:
            head = self.overlay.head_from_sha(ref['sha256'], head)
        return head

    def overlay_view(self, state):
        """The validated cumulative overlay chain view of a state (None without an overlay)."""
        if state['overlay_head_sha256'] is None:
            need(state['overlay_packets'] == [], 'no overlay packets without an overlay head')
            return None
        need(self.overlay is not None, 'native-result overlay package required for an overlay component')
        return self.overlay.validate_chain(state['overlay_packets'], expected_head=state['overlay_head_sha256'])

    def compose(self, base, view, identity):
        """The composite snapshot: base variant snapshot plus the overlay chain view, under its composite identity."""
        if view is None:
            need(base.identity == identity, 'plain candidate identity')
            return base
        s = self.overlay.overlay_snapshot(base, view, max_padded_tokens=self.max_padded_tokens)
        s.identity = identity
        s.batch_manifest = {**base.batch_manifest, 'snapshot_sha256': identity}
        s.base_variant = base
        s.extension_raw = getattr(base, 'extension_raw', None)
        return s

    def validate_candidate(self, envelope):
        """Every component and the cumulative state against the predecessor state; returns (variant, doc, view)."""
        need(envelope['schema'] == CANDIDATE and set(envelope['components']) == COMPONENTS and
             type(envelope.get('state')) is dict and set(envelope['state']) == STATE_KEYS, 'candidate components and state')
        seq, state = envelope['seq'], envelope['state']
        prior = self._prior_state(seq)
        variant, overlay = envelope['components']['variant'], envelope['components']['native_overlay']
        need(variant is not None or overlay is not None, 'a candidate carries at least one new component')
        doc = None
        if variant is not None:
            need(type(variant) is dict and set(variant) == VARIANT_KEYS and
                 digest(variant['descriptor']) == variant['descriptor_sha256'], 'variant component descriptor digest')
            doc = validate_root(variant['descriptor'], campaign_sha256=self.genesis['campaign_sha256'],
                                contract=self.genesis['contract'], acceptance=variant['preparation_acceptance'],
                                receipt_sha256=variant['preparation_receipt_sha256'])
            need(doc['config']['extension_sha256'] == variant['extension_sha256'] == state['variant_extension_sha256']
                 and state['variant_source_seq'] == seq, 'variant component extension and state')
        else:
            need(state['variant_extension_sha256'] == prior['variant_extension_sha256'] and
                 state['variant_source_seq'] == prior['variant_source_seq'], 'unchanged variant state')
        if overlay is not None:
            need(type(overlay) is dict and set(overlay) == OVERLAY_KEYS and type(overlay['packets']) is list and
                 overlay['packets'] and overlay['predecessor_head_sha256'] == prior['overlay_head_sha256'],
                 'overlay component shape and predecessor head')
            need(self._fold_head(prior['overlay_head_sha256'], overlay['packets']) == overlay['head_sha256'] ==
                 state['overlay_head_sha256'] and state['overlay_packets'] == prior['overlay_packets'] + overlay['packets'],
                 'overlay chain head and cumulative packets')
        else:
            need(state['overlay_head_sha256'] == prior['overlay_head_sha256'] and
                 state['overlay_packets'] == prior['overlay_packets'], 'unchanged overlay state')
        view = self.overlay_view(state)
        need(envelope['result_snapshot_sha256'] ==
             composite_identity(state['variant_extension_sha256'], state['overlay_head_sha256']),
             'candidate result snapshot is its composite identity')
        return variant, doc, view

    def publish(self, *, descriptor=None, acceptance=None, predecessor_extension_raw=None, overlay_packets=None,
                overlay_receipt_sha256=None):
        """Append the next immutable candidate: a new variant root and/or new overlay packets on the active chain."""
        seq = len(self._seqs('candidates')) + 1
        prior = self._prior_state(seq)
        predecessor = self.genesis['snapshot_sha256'] if seq == 1 else self.candidate(seq - 1)[1]['result_snapshot_sha256']
        state, components = dict(prior), {'variant': None, 'native_overlay': None}
        if descriptor is not None:
            need(sha(predecessor_extension_raw) == prior['variant_extension_sha256'],
                 'candidate predecessor is the previous published snapshot')
            doc = validate_root(descriptor, campaign_sha256=self.genesis['campaign_sha256'],
                                contract=self.genesis['contract'], acceptance=acceptance)
            validate_extension_transition(document(predecessor_extension_raw, MAX_DOC),
                                          document(doc['extension_raw'], MAX_DOC))
            extension_sha256 = doc['config']['extension_sha256']
            components['variant'] = {'descriptor': descriptor, 'descriptor_sha256': digest(descriptor),
                                     'extension_sha256': extension_sha256,
                                     'preparation_receipt_sha256': sha(doc['receipt_raw']),
                                     'preparation_acceptance': acceptance}
            state.update(variant_extension_sha256=extension_sha256, variant_source_seq=seq)
        if overlay_packets is not None:
            need(type(overlay_packets) is list and overlay_packets and type(overlay_receipt_sha256) is str,
                 'new overlay packets and their build receipt')
            head = self._fold_head(prior['overlay_head_sha256'], overlay_packets)
            components['native_overlay'] = {'packets': overlay_packets, 'head_sha256': head,
                                            'predecessor_head_sha256': prior['overlay_head_sha256'],
                                            'receipt_sha256': overlay_receipt_sha256}
            state.update(overlay_head_sha256=head, overlay_packets=prior['overlay_packets'] + overlay_packets)
        envelope = {'schema': CANDIDATE, 'seq': seq, 'predecessor_snapshot_sha256': predecessor,
                    'result_snapshot_sha256': composite_identity(state['variant_extension_sha256'],
                                                                 state['overlay_head_sha256']),
                    'components': components, 'state': state, 'created_utc_ns': time.time_ns()}
        self.validate_candidate(envelope)
        _create(self.root / 'candidates' / _name(seq), canonical(envelope))
        return envelope

    # ----------------------------------------------------------------------------------------------- B2 reconcile
    def _joins(self, seq, intent, events, manifest_before, snapshot_before, *, record=None):
        """Every join an adoption (or an open intent) must prove; returns a list of failed join names."""
        bad = []
        try:
            candidate_raw, envelope = self.candidate(seq)
        except (OSError, ValueError):
            return ['candidate_missing']
        if sha(candidate_raw) != intent['candidate_sha256']:
            bad.append('candidate_digest')
        if envelope.get('seq') != seq or envelope.get('predecessor_snapshot_sha256') != snapshot_before or \
                envelope.get('result_snapshot_sha256') != intent['snapshot_sha256']:
            bad.append('candidate_chain')
        if intent['seq'] != seq or intent['previous_snapshot_sha256'] != snapshot_before:
            bad.append('previous_snapshot')
        if intent['manifest_before'] != manifest_before:
            bad.append('manifest_before')
        if intent['contract_sha256'] != self.genesis['contract_sha256']:
            bad.append('contract')
        head = intent['planner_head_before']
        n = head.get('events', 0) if type(head) is dict else 0
        if not (type(head) is dict and 0 < n <= len(events) and events[n - 1]['sha'] == head.get('head_sha256') and
                head.get('unresolved_batches') == 0):
            bad.append('planner_head_before')
            return bad
        closed = intent['epoch_completed_event']
        e = events[n - 1]
        if not (closed == {'seq': n, 'sha': e['sha']} and e['kind'] == 'epoch.completed' and
                e['body'].get('epoch') == intent['closed_epoch'] and e['body'].get('manifest_sha256') == manifest_before):
            bad.append('closed_epoch')
        commit = intent['last_commit_event']
        committed = [x for x in events[:n] if x['kind'] == 'batch.committed']
        if not (committed and commit == {'seq': committed[-1]['seq'], 'sha': committed[-1]['sha']} and
                committed[-1]['body']['checkpoint_after'] == intent['checkpoint_sha256']):
            bad.append('checkpoint')
        if record is not None:
            if any(record.get(k) != intent.get(k) for k in JOINED):
                bad.append('record_intent_fields')
        return bad

    def _cas_event(self, intent, events):
        """The exact manifest.adopted event this intent's CAS must have produced, or None."""
        n = intent['planner_head_before']['events']
        if len(events) <= n:
            return None
        e = events[n]
        ok = (e['kind'] == 'manifest.adopted' and e['previous'] == intent['planner_head_before']['head_sha256'] and
              e['body'] == {'old': intent['manifest_before'], 'new': intent['manifest_after'],
                            'unfinished_epoch': intent['closed_epoch'], 'unconsumed_count': 0,
                            'unconsumed_sha256': digest([])})
        return e if ok else False

    def reconcile(self, planner):
        """Replay and verify every link; finalize a proved CAS-before-record crash; HOLD anything unexplained."""
        head, events = planner_events(planner)
        need(events and events[0]['kind'] == 'stream.created' and
             events[0]['body']['manifest_sha256'] == self.genesis['manifest_sha256'],
             'planner journal starts at the inbox genesis manifest')
        adopted, intents = self._seqs('adoptions'), self._seqs('intents')
        if not (adopted == intents[:len(adopted)] and len(intents) <= len(adopted) + 1):
            raise Refused('inbox intents/adoptions are not a single open-intent prefix')
        manifest, snapshot, records, explained = self.genesis['manifest_sha256'], self.genesis['snapshot_sha256'], [], set()
        for (raw, record), seq in zip(self.adoption_files(), adopted):
            try:
                intent_raw, intent = _read(self.root / 'intents' / _name(seq))
            except (OSError, ValueError):
                raise Refused(f'adoption {seq}: intent unreadable') from None
            bad = [] if (record.get('schema') == ADOPTION and record.get('seq') == seq and
                         record.get('intent_sha256') == sha(intent_raw) and intent.get('schema') == INTENT) else ['schema']
            bad += self._joins(seq, intent, events, manifest, snapshot, record=record)
            event = self._cas_event(intent, events)
            if not event or record['planner_event'] != {'seq': event['seq'], 'sha': event['sha']}:
                bad.append('planner_event')
            elif _manifest_body(planner, intent['manifest_after']).get('snapshot_sha256') != intent['snapshot_sha256']:
                bad.append('manifest_body')
            if bad:
                raise Refused(f'adoption {seq} links broken: ' + ','.join(sorted(set(bad))))
            explained.add(event['seq'])
            records.append(record)
            manifest, snapshot = record['manifest_after'], record['snapshot_sha256']
        if len(intents) == len(adopted) + 1:
            seq = intents[-1]
            intent_raw, intent = _read(self.root / 'intents' / _name(seq))
            bad = [] if intent.get('schema') == INTENT else ['schema']
            bad += self._joins(seq, intent, events, manifest, snapshot)
            event = None if bad else self._cas_event(intent, events)
            if bad:
                self._hold(seq, 'INTENT_JOIN_FAILED', {'intent_sha256': sha(intent_raw), 'failed_joins': sorted(set(bad))})
            elif event:
                try:
                    _, envelope = self.candidate(seq)
                    self.validate_candidate(envelope)
                    body_ok = _manifest_body(planner, intent['manifest_after']).get('snapshot_sha256') == intent['snapshot_sha256']
                except ValueError as exc:
                    body_ok, why = False, str(exc)[:300]
                else:
                    why = 'manifest body'
                if body_ok and planner._get('manifest') == intent['manifest_after']:
                    records.append(self._record(seq, intent_raw, intent, event, adopt_receipt=None, reconciled=True))
                    explained.add(event['seq'])
                    manifest, snapshot = intent['manifest_after'], intent['snapshot_sha256']
                else:
                    self._hold(seq, 'INTENT_JOIN_FAILED', {'intent_sha256': sha(intent_raw), 'failed_joins': [why]})
            elif len(events) == intent['planner_head_before']['events'] and planner._get('manifest') == intent['manifest_before']:
                self._hold(seq, 'INTENT_WITHOUT_PLANNER_CAS', {'intent_sha256': sha(intent_raw),
                                                              'planner_manifest': planner._get('manifest')})
            else:
                self._hold(seq, 'PLANNER_STATE_UNEXPLAINED', {'intent_sha256': sha(intent_raw),
                                                             'planner_head': head})
        unexplained = [e['seq'] for e in events if e['kind'] == 'manifest.adopted' and e['seq'] not in explained]
        active = planner._get('manifest')
        if (unexplained or active != manifest) and not self.holds():
            self._hold(len(records) + 1, 'PLANNER_MANIFEST_NOT_FROM_THIS_INBOX',
                       {'planner_manifest': active, 'expected': manifest, 'unexplained_events': unexplained})
        return {'adopted': len(records), 'records': records, 'holds': self.holds(), 'active_manifest_sha256': active,
                'active_snapshot_sha256': snapshot, 'head': head, 'events': events}

    def _record(self, seq, intent_raw, intent, event, *, adopt_receipt, reconciled):
        record = {'schema': ADOPTION, **{k: intent[k] for k in JOINED}, 'intent_sha256': sha(intent_raw),
                  'planner_event': {'seq': event['seq'], 'sha': event['sha']},
                  'adopt_receipt': adopt_receipt, 'reconciled_after_crash': reconciled, 'at_utc_ns': time.time_ns()}
        _create(self.root / 'adoptions' / _name(seq), canonical(record))
        return record

    # ---------------------------------------------------------------------------------------------- adoption
    def adopt_next(self, planner, current, native, *, campaign_sha256, preprocessing_sha256, registry_sha256,
                   checkpoint_sha256, fresh_authority, authority_check):
        """Admit the next candidate at a completed epoch, or return None when none is published.

        ``fresh_authority()`` returns current role-proof bytes; ``authority_check(extension_raw, proof_raw)`` must
        raise on any role/descriptor drift or a stale proof.  Refusals before the intent mutate nothing except an
        explicit HOLD for an authority refusal (B4).
        """
        need(campaign_sha256 == self.genesis['campaign_sha256'], 'campaign permit is the inbox campaign')
        state = self.reconcile(planner)
        if state['holds']:
            raise Held('boundary adoption HELD: ' + ', '.join(h['reason'] for h in state['holds']))
        records, events = state['records'], state['events']
        seq = len(records) + 1
        if not (self.root / 'candidates' / _name(seq)).exists():
            return None
        candidate_raw, envelope = self.candidate(seq)
        need(envelope['schema'] == CANDIDATE and envelope['seq'] == seq, 'next candidate envelope')
        need(current.identity == envelope['predecessor_snapshot_sha256'] == state['active_snapshot_sha256'],
             'candidate predecessor is the active snapshot')
        epoch = planner._get('epoch')
        row = planner.db.execute('SELECT state FROM epochs WHERE id=?', (epoch,)).fetchone()
        need(epoch > 0 and row is not None and row[0] == 'complete', 'closed epoch required')
        last = events[-1]
        need(last['kind'] == 'epoch.completed' and last['body']['epoch'] == epoch and
             last['body']['manifest_sha256'] == state['active_manifest_sha256'],
             'the completed epoch is the planner head (closed boundary)')
        need(all(r['closed_epoch'] != epoch for r in records), 'one adoption per completed epoch')
        need(planner.pending() is None and state['head']['unresolved_batches'] == 0, 'no unresolved optimizer reservation')
        need(planner.summary()['checkpoint_sha256'] == checkpoint_sha256, 'physical checkpoint joins planner')
        committed = [e for e in events if e['kind'] == 'batch.committed']
        need(committed and committed[-1]['body']['checkpoint_after'] == checkpoint_sha256,
             'the last committed batch produced this checkpoint')
        variant, doc, view = self.validate_candidate(envelope)
        base_current = getattr(current, 'base_variant', current)
        hydration_proof = fresh_authority()
        if variant is not None:
            validate_extension_transition(document(base_current.extension_raw, MAX_DOC),
                                          document(doc['extension_raw'], MAX_DOC))
            base = PA.candidate(variant['descriptor'], base_current, native, campaign_sha256=campaign_sha256,
                                preprocessing_sha256=preprocessing_sha256, registry_sha256=registry_sha256,
                                authority_check=lambda raw: authority_check(raw, hydration_proof))
            need(P.sha(doc['prepared_raw']) == variant['descriptor']['prepared']['sha256'], 'hydrated the validated bytes')
        else:
            base = base_current
            authority_check(base.extension_raw, hydration_proof)
        need(base.identity == envelope['state']['variant_extension_sha256'], 'hydrated variant base identity')
        replacement = self.compose(base, view, envelope['result_snapshot_sha256'])
        need(replacement.identity == envelope['result_snapshot_sha256'], 'hydrated candidate identity')
        manifest_after = sha(canonical(replacement.batch_manifest))
        cas_proof = fresh_authority()                       # B4: fresh proof after hydration, right before the CAS
        try:
            authority_check(replacement.extension_raw, cas_proof)
        except ValueError as exc:
            self._hold(seq, 'AUTHORITY_REFUSED_BEFORE_CAS', {'candidate_sha256': sha(candidate_raw),
                                                             'cas_authority_sha256': sha(cas_proof),
                                                             'error': str(exc)[:300]})
            raise Held('boundary adoption HELD: AUTHORITY_REFUSED_BEFORE_CAS') from exc
        intent = {'schema': INTENT, 'seq': seq, 'candidate_sha256': sha(candidate_raw), 'closed_epoch': epoch,
                  'manifest_before': state['active_manifest_sha256'], 'manifest_after': manifest_after,
                  'previous_snapshot_sha256': current.identity, 'snapshot_sha256': replacement.identity,
                  'checkpoint_sha256': checkpoint_sha256, 'hydration_authority_sha256': sha(hydration_proof),
                  'cas_authority_sha256': sha(cas_proof), 'planner_head_before': state['head'],
                  'epoch_completed_event': {'seq': last['seq'], 'sha': last['sha']},
                  'last_commit_event': {'seq': committed[-1]['seq'], 'sha': committed[-1]['sha']},
                  'contract_sha256': self.genesis['contract_sha256'], 'at_utc_ns': time.time_ns()}
        intent_raw = canonical(intent)
        self.fault('before_intent')
        _create(self.root / 'intents' / _name(seq), intent_raw)
        self.fault('after_intent')
        rechecked = []

        def cas_check(raw):
            authority_check(raw, cas_proof)               # freshness is re-evaluated at CAS time
            rechecked.append(True)

        try:
            receipt = PA.adopt(planner, current, replacement, checkpoint_sha256=checkpoint_sha256,
                               authority_check=cas_check)
        except ValueError as exc:
            if not rechecked and planner._get('manifest') == intent['manifest_before']:
                self._hold(seq, 'AUTHORITY_REFUSED_AT_CAS', {'intent_sha256': sha(intent_raw), 'error': str(exc)[:300]})
                raise Held('boundary adoption HELD: AUTHORITY_REFUSED_AT_CAS') from exc
            raise
        self.fault('after_cas')
        _, events = planner_events(planner)
        event = self._cas_event(intent, events)
        need(bool(event), 'exactly the planner manifest.adopted event this intent proves')
        self._record(seq, intent_raw, intent, event, adopt_receipt=receipt, reconciled=False)
        self.fault('after_record')
        return replacement


def active_manifest(db_path):
    """Read-only: the planner's ACTIVE manifest bytes, so a restarted executor can reopen BatchStream on them."""
    con = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    try:
        value = con.execute("SELECT value FROM state WHERE key='manifest'").fetchone()
        need(value is not None, 'planner state present')
        active = json.loads(value[0])
        body = con.execute('SELECT body FROM manifests WHERE sha=?', (active,)).fetchone()
        need(body is not None and sha(bytes(body[0])) == active, 'stored active manifest bytes')
        return bytes(body[0]), active
    finally:
        con.close()


def load_adopted(inbox, record, native, *, genesis, preprocessing_sha256, registry_sha256, authority_check):
    """B1: re-hydrate the snapshot of one VERIFIED adoption record through the strict root validator.

    The variant base comes from the candidate that last set it (or the supplied genesis snapshot), the overlay from the
    record's validated cumulative chain.  ``record`` must be byte-identical to the stored record (after reconcile).
    """
    stored = {d['seq']: (raw, d) for raw, d in inbox.adoption_files()}
    need(record.get('seq') in stored and canonical(record) == stored[record['seq']][0], 'stored adoption record')
    candidate_raw, envelope = inbox.candidate(record['seq'])
    need(sha(candidate_raw) == record['candidate_sha256'], 'adopted candidate bytes')
    _, _, view = inbox.validate_candidate(envelope)
    state = envelope['state']
    if state['variant_source_seq'] == 0:
        need(genesis.identity == state['variant_extension_sha256'], 'genesis variant base')
        base = genesis
    else:
        _, source = inbox.candidate(state['variant_source_seq'])
        _, doc, _ = inbox.validate_candidate(source)
        base = hydrate(doc, native, preprocessing_sha256=preprocessing_sha256, registry_sha256=registry_sha256,
                       authority_check=authority_check)
    need(base.identity == state['variant_extension_sha256'], 'adopted variant base identity')
    snapshot = inbox.compose(base, view, envelope['result_snapshot_sha256'])
    need(snapshot.identity == record['snapshot_sha256'] == envelope['result_snapshot_sha256'] and
         sha(canonical(snapshot.batch_manifest)) == record['manifest_after'], 'adopted snapshot and manifest identity')
    return snapshot


def term_support(records, window_ids, program_sha256):
    """Supported objective terms for one record inside an actually selected program window."""
    window = [r for r in records if r.program_sha256 in set(window_ids)]
    target = next(r for r in window if r.program_sha256 == program_sha256)
    pair = [r for r in window if r.pairwise_eligible]
    terms = []
    if target.ce_eligible:
        terms.append('ce')
    if target.quality_eligible:
        terms.append('q')
    if target.pairwise_eligible and any(r.q_grid != target.q_grid and r.q_opt != target.q_opt
                                        for r in pair if r is not target):
        terms.append('bt')
    if target.pairwise_eligible and len(pair) >= 3:
        terms.append('list')
    if target.feasibility is not None:
        terms.append('feas')
    if target.runtime_eligible:
        terms.append('rt')
    return terms


def _ref_in(snapshot, context_id, program_sha256):
    """The exact normalized result reference a snapshot holds for one composed variant or overlay record (or None)."""
    receipt = getattr(snapshot, 'receipt', None) or {}
    found = [x for x in receipt.get('variants', {}).get(context_id, {}).get('records', [])
             if x['program_sha256'] == program_sha256]
    if not found:
        found = [x for x in (getattr(snapshot, 'overlay_records', None) or {}).get(context_id, [])
                 if x['program_sha256'] == program_sha256]
    need(len(found) <= 1, 'one composed record per program')
    return digest(K.normalized_ref(found[0]['result'])) if found else None


def _terms(snapshot, key, member, selection, overlay_record):
    """Supported terms inside the actually selected window.  Variant records use the snapshot's composed records; an
    overlay record must be one the window receipt SELECTED (exact ref digest), and its window is recomputed from the
    snapshot's overlay loader at the receipt's (epoch, batch_index) and must reproduce the proof's program window."""
    cid, ph = key
    if overlay_record is None:
        return term_support(snapshot.variant_records[cid], member['program_ids'], ph)
    need(type(selection.get('epoch')) is int and type(selection.get('batch_index')) is int,
         'overlay window receipt position')
    again = snapshot.context_loader.get([cid], epoch=selection['epoch'], batch_index=selection['batch_index'])
    records = again.records[cid]
    need([r.program_sha256 for r in records] == list(member['program_ids']), 'recomputed overlay window is the proof window')
    return term_support(records, member['program_ids'], ph)


def first_exposures(planner, inbox, proof_paths, snapshots, *, origin_step=0):
    """B3: join every admitted result to the first physically committed batch that actually selected it.

    ``snapshots`` maps snapshot identity -> hydrated snapshot for the genesis and every adoption a proof uses
    (normally from ``load_adopted``).  Wanted entries come only from the sealed C10a receipts of verified adoptions;
    proofs are joined to the planner reservation and commit, deduplicated by content and ordered by verified
    optimizer step; a retained exact reference in a descendant snapshot still counts.

    Coverage: every committed batch under an adopted manifest is taken from the verified planner event chain.  An
    exposure is ``first`` only if every committed batch under a manifest that could carry the wanted row, with a
    step before it, has a supplied verified proof; otherwise it is ``earliest_observed_partial``.  A wanted row with
    no observed selection is ``pending`` under complete coverage and ``unobserved_partial`` otherwise.
    """
    state = inbox.reconcile(planner)
    records = state['records']
    by_manifest = {inbox.genesis['manifest_sha256']: (0, inbox.genesis['snapshot_sha256'])}
    wanted = {}
    for record in records:
        by_manifest[record['manifest_after']] = (record['seq'], record['snapshot_sha256'])
        _, envelope = inbox.candidate(record['seq'])
        variant, doc, view = inbox.validate_candidate(envelope)
        snapshot = snapshots[record['snapshot_sha256']]
        need(snapshot.identity == record['snapshot_sha256'], 'supplied adopted snapshot')
        admitted = []
        if variant is not None:
            admitted += [(row['context_id'], row['program_sha256'], row['result_ref'], None) for row in doc['receipt']['rows']
                         if row['disposition'] in ('record', 'record_masked_no_q')]
        if envelope['components']['native_overlay'] is not None:
            before = len(inbox._prior_state(record['seq'])['overlay_packets'])
            admitted += [(r.context_id, r.record.program_sha256, r.result_ref, r.record) for r in view.rows
                         if r.packet_seq > before]
        for cid, ph, ref, overlay_record in admitted:
            key = (cid, ph)
            need(key not in wanted, 'one admission per (context, program)')
            ref_key = digest(K.normalized_ref(ref))
            need(_ref_in(snapshot, *key) == ref_key, 'admitted reference is the composed record reference')
            wanted[key] = {'seq': record['seq'], 'ref_key': ref_key, 'overlay_record': overlay_record}
    committed = {}
    for e in state['events']:
        if e['kind'] == 'batch.committed':
            row = planner.db.execute('SELECT manifest FROM batches WHERE id=?', (e['body']['batch_id'],)).fetchone()
            need(row is not None, 'committed batch row')
            if row[0] in by_manifest:
                committed[e['body']['batch_id']] = (e['body']['optimizer_step'], by_manifest[row[0]][0])
    proofs = {}
    for path in proof_paths:
        raw = read(Path(path), MAX_DOC)
        proof = json.loads(raw)
        bid = proof['batch']['batch_id']
        need(bid not in proofs or proofs[bid][0] == sha(raw), 'conflicting proofs for one batch')
        proofs[bid] = (sha(raw), proof)
    ordered = []
    for bid, (digest_raw, proof) in proofs.items():
        row = planner.db.execute('SELECT state,body,commit_body FROM batches WHERE id=?', (bid,)).fetchone()
        need(row is not None and row[0] == 'committed', 'proof joins a committed planner batch')
        reserved, commit, batch = json.loads(row[1]), json.loads(row[2]), proof['batch']
        need(commit['evidence']['event_sha256'] == digest_raw, 'proof is the committed evidence')
        need(all(batch[k] == reserved[k] for k in ('batch_id', 'manifest_sha256', 'epoch', 'fingerprint',
                                                     'checkpoint_before')) and batch['index'] == reserved['index'] and
             list(batch['ordered_data_run_ids']) == reserved['ordered_data_run_ids'] == commit['ordered_data_run_ids'] and
             commit['checkpoint_before'] == reserved['checkpoint_before'], 'proof is the reserved batch')
        need(proof['checkpoint']['parameter_sha256'] == commit['checkpoint_after'] and
             proof['absolute_optimizer_step'] == origin_step + commit['optimizer_step'], 'proof checkpoint and step')
        need(reserved['manifest_sha256'] in by_manifest, 'proof under a manifest of this inbox')
        seq, snapshot_sha256 = by_manifest[reserved['manifest_sha256']]
        need(proof['snapshot_sha256'] == snapshot_sha256, 'proof snapshot is the adopted snapshot')
        members = proof['selection']['selection']['members']
        need(sorted(m['context_id'] for m in members) == sorted(reserved['ordered_data_run_ids']),
             'selected members are the reserved batch')
        ordered.append((commit['optimizer_step'], seq, snapshot_sha256, bid, digest_raw, proof, commit))
    ordered.sort(key=lambda x: x[0])
    need(len({x[0] for x in ordered}) == len(ordered), 'one proof per optimizer step')
    missing = {bid: v for bid, v in committed.items() if bid not in proofs}

    def covered(seq_from, before_step):
        return not any(seq >= seq_from and step < before_step for step, seq in missing.values())

    found, contexts, presentations = {}, set(), 0
    for step, seq, snapshot_sha256, bid, digest_raw, proof, commit in ordered:
        snapshot = snapshots.get(snapshot_sha256)
        selection = proof['selection']['selection']
        for member in selection['members']:
            presentations += 1
            contexts.add(member['context_id'])
            selected = {x['program_sha256']: x for x in member.get('selected', ())}
            for ph in member['program_ids']:
                key = (member['context_id'], ph)
                if key not in wanted or key in found or wanted[key]['seq'] > seq:
                    continue
                if wanted[key]['overlay_record'] is not None:
                    if ph not in selected:
                        continue                              # a base-window copy is not the admitted overlay result
                    need(selected[ph]['result_ref_sha256'] == wanted[key]['ref_key'],
                         'overlay window receipt selects the exact admitted reference')
                need(snapshot is not None and snapshot.identity == snapshot_sha256, 'snapshot for exposing proof')
                need(_ref_in(snapshot, *key) == wanted[key]['ref_key'],
                     'descendant snapshot retains the exact admitted reference')
                found[key] = {'schema': EXPOSURE, 'adoption_seq': wanted[key]['seq'], 'exposed_under_seq': seq,
                              'status': 'first' if covered(wanted[key]['seq'], step) else 'earliest_observed_partial',
                              'snapshot_sha256': snapshot_sha256, 'context_id': key[0], 'program_sha256': ph,
                              'result_ref_sha256': wanted[key]['ref_key'],
                              'terms': _terms(snapshot, key, member, selection, wanted[key]['overlay_record']),
                              'component': 'variant' if wanted[key]['overlay_record'] is None else 'native_overlay',
                              'absolute_optimizer_step': proof['absolute_optimizer_step'], 'batch_id': bid,
                              'proof_sha256': digest_raw, 'checkpoint_before': commit['checkpoint_before'],
                              'checkpoint_after': commit['checkpoint_after'], 'planner_commit_verified': True}
    pending = []
    for k in sorted(wanted):
        if k not in found:
            complete = covered(wanted[k]['seq'], float('inf'))
            pending.append(list(k) + [wanted[k]['seq'], 'pending' if complete else 'unobserved_partial'])
    return {'exposures': [found[k] for k in sorted(found, key=lambda k: found[k]['absolute_optimizer_step'])],
            'pending': pending, 'proofs_verified': len(ordered),
            'coverage': {'committed_under_inbox_manifests': len(committed), 'supplied': len(committed) - len(missing),
                         'missing': len(missing), 'complete': not missing},
            'unique_contexts_presented': len(contexts), 'context_presentations': presentations}
