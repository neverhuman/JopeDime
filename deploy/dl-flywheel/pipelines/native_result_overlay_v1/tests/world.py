"""Private synthetic native world for the W3 fixtures (tests only; CPU; no live data, S3, GP, planner, optimizer, GPU).

REAL frozen code exercised: ``load_snapshot`` / ``NativeContextLoader``, ``label_context``, the native tokenizer and
SYNTAX grammar, ``NativeInputs.describe`` / ``masked_training``, ``request_identity`` / ``verified_measurement`` /
``verify_bodies`` / ``validate_records``, ``fit_preprocessing``, ``prepare_microbatches`` / ``collate`` and the frozen
``Outbox.record_batch`` (emission bodies and identities come from a real fixture-local outbox).

FIXTURE values (nothing here is presented as real): every dataset, teacher, shard, inventory, authority proof,
artifact body and versioned object. The in-memory VersionId store and its canonicalizer (Python canonical JSON) are
NOT S3 custody or native-serde qualification. No GP, native worker or Rust code ever ran: predictions are synthesized,
'rust_source'/'rust_executable' are placeholder bytes, and the parity block is constructed, not measured.

Outer-test cells of every teacher hold a signaling-NaN SENTINEL; ``Tripwire`` raises if any numpy.frombuffer or
struct.unpack(_from) call converts a byte range containing it.
"""
import base64
import gzip
import importlib
import json
import math
import struct
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

import native_evaluate as NE
import native_inputs as NI
import neural_native_batch as NB
import test_training_snapshot_kumos as K
from dltrain import automaton
from dltrain import dsl_tokenizer as T
from jope_flywheel import training_snapshot_native as N
from jope_flywheel.program_worker import compare_predictions, split_scale
from jope_flywheel.protocol import canonical, digest, partition
from jope_flywheel.score_join import build_records
from neural_native_train import fit_preprocessing
from neural_outbox import Outbox

from pipelines.native_result_overlay_v1 import build_packet
from pipelines.native_result_overlay_v1 import common as C

SENTINEL = struct.pack('<Q', 0x7FF4DEADBEEF5EED)   # signaling NaN payload in every outer-test cell
VOCAB = T.Vocab.from_names({})
MAX_SEQ = 128
N_TRAIN, N_TEST, P = 50, 10, 2
CUT = 'w3-fixture-native-cut-v1'
NS = digest({'w3-fixture-split-namespace': 1})
CAMPAIGN = digest({'w3-fixture-campaign': 'allowed'})
OTHER_CAMPAIGN = digest({'w3-fixture-campaign': 'not-allowed'})
BACKUP_SHA = digest({'w3-fixture-backup': 'v1'})
ORIGIN = digest({'w3-fixture-model-origin': 1})
LONG_PROGRAM = '(pipe X ' + '(standardize) ' * 40 + '(ridge alpha=1.0))'
SPECS = [  # name, role, shard rows, shard baseline mode, HWM winner DSL
    ('A', 'train', 12, 'ulp', '(pipe X (ridge alpha=1.0))'),
    ('B', 'train', 0, None, '(pipe X (ridge alpha=900.0))'),
    ('C', 'validation', 6, 'exact', '(pipe X (ridge alpha=1.0))'),
    ('D', 'train', 6, 'disagree', '(pipe X (ridge alpha=1.0))'),
]
_GRAMMAR = {}


def sha(b):
    return C.sha(b)


def canonicalize(raw):
    """Fixture stand-in for the pinned native canonicalizer (whitespace normalization only)."""
    return ' '.join(raw.split())


class OuterTestDecoded(AssertionError):
    pass


class Tripwire:
    """Raise if any numpy.frombuffer / struct.unpack / struct.unpack_from converts an outer-test SENTINEL cell."""

    def __enter__(self):
        self.calls = 0
        self.saved = (np.frombuffer, struct.unpack, struct.unpack_from)
        frombuffer, unpack, unpack_from = self.saved

        def guarded_frombuffer(*args, **kwargs):
            arr = frombuffer(*args, **kwargs)
            self.calls += 1
            if SENTINEL in arr.tobytes():
                raise OuterTestDecoded('outer-test cell converted by numpy.frombuffer')
            return arr

        def guarded_unpack(fmt, buffer):
            self.calls += 1
            if SENTINEL in bytes(buffer):
                raise OuterTestDecoded('outer-test cell converted by struct.unpack')
            return unpack(fmt, buffer)

        def guarded_unpack_from(fmt, buffer, offset=0):
            self.calls += 1
            if SENTINEL in bytes(buffer[offset:offset + struct.calcsize(fmt)]):
                raise OuterTestDecoded('outer-test cell converted by struct.unpack_from')
            return unpack_from(fmt, buffer, offset)
        np.frombuffer, struct.unpack, struct.unpack_from = guarded_frombuffer, guarded_unpack, guarded_unpack_from
        return self

    def __exit__(self, *exc):
        np.frombuffer, struct.unpack, struct.unpack_from = self.saved
        return False


class PathStore:
    """Exact local-ref fixture store ({path, sha256, bytes} -> bytes); stands in for LocalStore and native bound()."""

    def __init__(self):
        self.b, self.calls = {}, []

    def put(self, path, body):
        self.b[path] = body
        return {'path': path, 'sha256': sha(body), 'bytes': len(body)}

    def doc(self, path, x):
        return self.put(path, canonical(x))

    def read(self, ref):
        self.calls.append(ref['path'])
        return self.b[ref['path']]

    def bound(self, ref, cap=NI.MAX_RAW):
        body = self.b[ref['path']]
        C.need(sha(body) == ref['sha256'] and ('bytes' not in ref or len(body) == ref['bytes']), 'fixture source SHA')
        return body


def grammar_doc():
    if not _GRAMMAR:
        sig = automaton.build_signatures([])
        _GRAMMAR['doc'] = {'mode': 'SYNTAX', 'vocab_sha256': VOCAB.sha256, 'signatures': sig,
                           'grammar': automaton.build_grammar(sig, VOCAB, MAX_SEQ)}
    return _GRAMMAR['doc']


def lineage_for(name, role):
    i = 0
    while True:
        lin = digest({'w3-fixture-lineage': name, 'i': i})
        if partition(lin, NS) == role:
            return lin
        i += 1


def reconstruct(y, folds):
    vals = []
    for j in range(2):
        tr, va = y[folds != j], y[folds == j]
        mean = math.fsum(map(float, tr)) / len(tr)
        vals.append(math.sqrt(math.fsum((float(v) - mean) ** 2 for v in va) / len(va)))
    return vals


class World:
    """One synthetic native corpus: A (train, ulp-anchored shard), B (train, no shard), C (validation), D (train,
    shard baseline 1e-6 off its teacher prefix: BASELINE_DISAGREEMENT)."""

    def __init__(self, root, *, protected_context_ids=(), protected_views=(), r6=None):
        self.root = Path(root)
        self.store, self.s3 = PathStore(), K.Store()
        self.protected = (list(protected_context_ids), list(protected_views))
        self.counter = 0
        self.ctx = {}
        for i, (name, role, n_shard, mode, winner) in enumerate(SPECS):
            self.ctx[name] = self._context(i, name, role, n_shard, mode, winner)
        self._snapshot()
        self.outbox = Outbox(self.root / 'fixture-outbox.sqlite', model_origin_sha256=ORIGIN)
        self.sources = [{'outbox_kind': 'v1', 'backup_sha256': BACKUP_SHA,
                         'event_head': {'events': 1, 'head_sha256': digest({'w3-fixture-head': 1})}}]
        self.r6 = canonical(r6 if r6 is not None else self.r6_doc([self.ctx['C']['cid']], [self.ctx['C']['lineage']]))

    # ------------------------------------------------------------------ synthetic contexts, teachers and shards
    def _context(self, i, name, role, n_shard, mode, winner):
        cid = digest({'w3-fixture-context': name})
        dataset = digest({'w3-fixture-dataset': name})[:16]
        view = f'{dataset}__r0__f0__w3{name}'
        lin = lineage_for(name, role)
        rng = np.random.default_rng(1000 + i)
        y = np.round(rng.normal(2.0 + i, 1.5, N_TRAIN), 6).astype('<f8')
        x = np.column_stack([np.round(rng.normal(0., 1., N_TRAIN), 6),
                             np.arange(N_TRAIN, dtype='<f8') / N_TRAIN]).astype('<f8')
        folds = np.array([0, 1] * (N_TRAIN // 2), dtype=np.uint8)
        train = np.column_stack([x, y]).astype('<f8')
        stride = (P + 1) * 8
        raw = (b'JTT1' + struct.pack('<IIIB', N_TRAIN, N_TEST, P, 2) + folds.tobytes() + train.tobytes() +
               SENTINEL * (N_TEST * (P + 1)))
        end = 17 + N_TRAIN + N_TRAIN * stride
        masked = bytearray(raw)
        for t in range(N_TEST):
            off = end + t * stride + P * 8
            masked[off:off + 8] = b'\0' * 8
        masked = bytes(masked)
        gz = gzip.compress(raw, mtime=0)
        teacher = self.store.put(f'/fixture/teachers/{name}.jtt1.gz', gz)
        cj = self.store.doc(f'/fixture/ctx/{name}/context.json',
                            {'context_sha256': cid, 'view_id': view, 'dataset_hash': dataset, 'fold': 0,
                             'files': {'teacher.jtt1.gz': {'sha256': teacher['sha256']}}})
        inp = {'context_sha256': cid, 'view_id': view, 'dataset_hash': dataset, 'fold': 0, 'cut_id': CUT,
               'teacher': {'path': teacher['path'], 'gz_sha256': teacher['sha256'], 'gz_bytes': teacher['bytes'],
                           'raw_bytes': len(raw), 'raw_sha256': sha(raw)},
               'context_dir': f'/fixture/ctx/{name}', 'context_json_sha256': cj['sha256'],
               'shape': {'n_train': N_TRAIN, 'n_test': N_TEST, 'p': P, 'k': 2, 'fold_sizes': [25, 25]},
               's1_input': {'bytes': len(masked), 'sha256': sha(masked)},
               'p3': {'assignment': 'train' if role == 'train' else 'dev'}}
        recon = reconstruct(y, folds)
        score = NI.OriginalScoreContext(cid, dataset, CUT, sha(folds.tobytes()), sha(masked), sha(y.tobytes()))
        c = {'name': name, 'cid': cid, 'dataset': dataset, 'view': view, 'lineage': lin, 'role': role, 'y': y, 'x': x,
             'folds': folds, 'raw': raw, 'masked': masked, 'input_row': inp, 'context_json': cj, 'recon': recon,
             'score': score, 'winner': winner, 'shard': None, 'group': i}
        if n_shard:
            pair = {'exact': list(recon), 'ulp': [math.nextafter(recon[0], math.inf), recon[1]],
                    'disagree': [recon[0] * (1 + 1e-6), recon[1]]}[mode]
            split_role = 'train_fit' if role == 'train' else 'dev'
            rows = []
            for j in range(n_shard):
                dsl = f'(pipe X (ridge alpha={j + 1}.0))'
                tokens = T.tokenize(dsl, VOCAB)
                rows.append({'context_sha256': cid, 'program_sha256': sha(dsl.encode()),
                             'skeleton_sha256': T.skeleton_sha256(dsl), 'tokens': tokens, 'length': len(tokens),
                             'outcome': 'completed', 'rmse_j': [b * (0.5 + j / 40) for b in recon],
                             'baseline_rmse_j': list(pair), 'wall_us': 1000 + j, 'backend': 'local',
                             'shared_seed': False, 'family_id': lin, 'dataset_hash': dataset, 'split_role': split_role})
            doc = {'schema': 'jope.dl-s1-tail-shard/v1',
                   'context': {'context_sha256': cid, 'view_id': view, 'dataset_hash': dataset, 'fold': 0,
                               'cut_id': CUT, 'family_id': lin, 'split_role': split_role, 'y_train': y.tolist(),
                               'n_records': n_shard},
                   'records': rows, 'source': {'protocol': 'jope.s1-two-fold/v1', 'vocab_sha256': VOCAB.sha256}}
            rel = f'shards/{cid[:2]}/{cid}.json'
            c['shard'] = {'doc': doc, 'pair': pair, 'rel': rel, 'ref': self.store.doc('/fixture/tails/' + rel, doc)}
        return c

    def _snapshot(self):
        st, names = self.store, [s[0] for s in SPECS]
        cs = [self.ctx[n] for n in names]
        winners = [c['winner'] for c in cs]
        meta = {'n': len(cs), 'cids': [c['cid'] for c in cs], 'view_id': [c['view'] for c in cs],
                'dataset': [c['dataset'] for c in cs], 'fold': [0] * len(cs),
                'program_sha256': [sha(w.encode()) for w in winners]}
        hwm = {'programs': [{'cid': c['cid'], 'program_sha256': sha(w.encode()), 'dsl': w} for c, w in zip(cs, winners)]}
        vectors = np.stack([np.linspace(-1., 1., N.DIM, dtype=np.float32) * (c['group'] + 1) for c in cs])
        refs = {'meta.json': st.doc('/fixture/bundle/meta.json', meta),
                'hwm_programs.json': st.doc('/fixture/bundle/hwm_programs.json', hwm),
                'gid.npy': st.put('/fixture/bundle/gid.npy', npy(np.arange(len(cs), dtype=np.int32))),
                'blocks_kumo_tabular_s.npy': st.put('/fixture/bundle/blocks.npy', npy(vectors)),
                'encoding_sha256.json': st.doc('/fixture/bundle/encoding.json',
                                               {'kumo_tabular_s': {c['cid']: digest({'kumo': c['cid']}) for c in cs}})}
        refs['MANIFEST.json'] = st.doc('/fixture/bundle/MANIFEST.json', {'files': {k: r['sha256'] for k, r in refs.items()}})
        parents = [{'bundle_manifest_sha256': refs['MANIFEST.json']['sha256'], 'gid_sha256': refs['gid.npy']['sha256'],
                    'parent_lineage_sha256': c['lineage'], 'partition': c['role'], 'split_namespace_sha256': NS,
                    'retained_group': c['group'], 'retained_lineage_context_sha256': [c['cid']]} for c in cs]
        self.registry = b''.join(canonical(p) + b'\n' for p in parents)
        inv = {'rows': [{'context_sha256': c['cid'], 'view_id': c['view'], 'dataset_hash': c['dataset'], 'fold': 0,
                         'cut': CUT, 'gp_index': {'outcome': 'selection_reached', 'main': {'fixture': c['name']}}}
                        for c in cs]}
        self.authpins = {'target': st.doc('/fixture/auth/target.json', {'gp_target': len(cs), 'views': [c['view'] for c in cs]}),
                         'role': st.doc('/fixture/auth/role.json', {'fixture': True}),
                         'layers': st.doc('/fixture/auth/layers.json', {'fixture': True})}
        index = b''.join(canonical({'context_sha256': c['cid'], 'view_id': c['view'],
                                    'n_records': len(c['shard']['doc']['records']), 'shard': c['shard']['rel'],
                                    'sha256': c['shard']['ref']['sha256'], 'bytes': c['shard']['ref']['bytes']}) + b'\n'
                         for c in cs if c['shard'])
        inputs = st.doc('/fixture/inputs/list.json', {'schema': 'jope.dl-s1-prod-input-list/v1',
                                                       'admitted': [c['input_row'] for c in cs]})
        self.spec = {'prior_ancestry': None, 'schema': N.SCHEMA, 'components': {'kumo_tabular_s': N.DIM},
                     'max_seq': MAX_SEQ, 'program_window': 8,
                     'consumer_source_sha256': {m: C.module_sha(importlib.import_module(m)) for m in N.MODULES},
                     'vocab_sha256': VOCAB.sha256, 'grammar': st.doc('/fixture/grammar.json', grammar_doc()),
                     'authority_pins': self.authpins, 'bundle': refs,
                     'parent_registry': st.put('/fixture/parents.jsonl', self.registry),
                     'inventory': st.doc('/fixture/inventory.json', inv), 'input_lists': [inputs],
                     'context_json_bytes': {c['context_json']['sha256']: c['context_json']['bytes'] for c in cs},
                     'tails': [{'root': '/fixture/tails',
                                'header': st.doc('/fixture/tails/header.json',
                                                 {'vocab_sha256': VOCAB.sha256, 'shard_schema': 'jope.dl-s1-tail-shard/v1'}),
                                'index': st.put('/fixture/tails/index.jsonl', index)}]}
        self.spec_bytes = canonical(self.spec)
        self.nat = N.load_snapshot(self.spec_bytes, sha(self.spec_bytes), st, authority_bytes=self.authority(),
                                   now=time.time())
        self.inputs = NI.NativeInputs(self.spec['input_lists'], self.nat.contexts, reader=st.bound,
                                      authority=lambda: json.loads(self.authority()), authority_pins=self.authpins)
        _, pre = fit_preprocessing(self.nat)
        self.pre = canonical(pre)
        self.source_pins = {m: C.module_sha(importlib.import_module(m)) for m in C.DEPENDENCIES}

    def authority(self, *, at=None):
        return canonical({'schema': C.AUTHORITY_SCHEMA, 'passed': True,
                          'checked_at_unix': time.time() if at is None else at, 'source_pins': self.authpins,
                          'protected_context_ids': self.protected[0], 'protected_views': self.protected[1]})

    @staticmethod
    def r6_doc(contexts, lineages):
        return {'schema': C.R6_SCHEMA, 'seed': 'w3-fixture-only', 'method': 'fixture roster (not the real R6 split)',
                'd_tune': {'contexts': list(contexts), 'lineages': list(lineages)},
                'd_confirm': {'contexts': [], 'lineages': []}}

    # ------------------------------------------------------------------ fake verified results (7 artifacts)
    def result(self, name, dsl, rmse=None, *, evaluation_baseline='exact', wall=(1500, 1700), tolerance=None,
               wrapper_key=None):
        """Synthesized measured result for (context, canonical DSL); passes the frozen verified_measurement."""
        c = self.ctx[name]
        score, x, y, folds = c['score'], c['x'], c['y'], c['folds']
        rmse = rmse or [b * 0.6 for b in c['recon']]
        ph = sha(dsl.encode())
        key = NE.request_identity(score, dsl)
        inner, scores = [], []
        for i in (0, 1):
            target = y[folds == i]
            split = split_scale(x, y, folds, i)
            signs = np.where(np.arange(len(target)) % 2 == 0, 1.0, -1.0)
            pred = (target + rmse[i] * signs).tolist()
            tgt = target.tolist()
            observed = math.sqrt(math.fsum((a - b) ** 2 for a, b in zip(pred, tgt)) / len(tgt))
            fold = {'schema': 'jope.teacher-fit/v2', 'index': i, 'kind': 'inner_cv', 'score_rows': len(tgt),
                    'train_rows': N_TRAIN - len(tgt), 'scaler_sha256': split['scaler_sha256'],
                    'serialized_prediction_replay_exact': True, 'predictions': pred, 'targets': tgt,
                    'metrics': {'rmse': observed}, 'wall_us': wall[i]}
            if evaluation_baseline is not None:
                base = c['recon'] if evaluation_baseline == 'exact' else evaluation_baseline
                fold['baseline_metrics'] = {'rmse': base[i]}
            inner.append(fold)
            scores.append(observed)
        ev = {'schema': 'jope.s1-evaluation/v1', 'status': 'completed', 'fold_protocol': 'jope.s1-two-fold/v1',
              'inner_cv_folds': 2, 'input_semantics': 'unscaled_fit_rows_v1',
              'program': {'dsl': dsl, 'program_sha256': ph}, 'inner_cv': inner}
        bodies = {'input': c['masked'], 'program': dsl.encode(), 'evaluation': canonical(ev),
                  'rust_source': b'// FIXTURE placeholder: no Rust was emitted, compiled or executed\n',
                  'rust_executable': b'FIXTURE placeholder bytes; never executed'}
        checks = []
        for i in (0, 1):
            check = compare_predictions(inner[i]['predictions'], inner[i]['predictions'])
            check.update({'fold': i, 'scaler_sha256': inner[i]['scaler_sha256'],
                          'scaled_predictions_sha256': sha(b'fixture')})
            if tolerance is not None:
                check['tolerance'] = tolerance
            checks.append(check)
        provenance = {'schema': 'jope.flywheel-native-execution-provenance/v1', 'evaluator_sha256': score.evaluator_sha256,
                      'input_sha256': sha(c['masked']), 'program_sha256': ph, 'evaluation_sha256': sha(bodies['evaluation']),
                      'source_commit': 'a' * 40, 'execution_timestamp': 1791000000,
                      'input_descriptor': {'context': {'context_id': c['cid']}}, 'fold_checks': checks,
                      'scope': 'FIXTURE: synthesized predictions; no native, GP or Rust execution'}
        parity = {'schema': 'jope.flywheel-rust-parity/v1', 'passed': True, 'executed': True, 'folds_verified': 2,
                  'evaluation_key': key, 'program_sha256': ph, 'evaluator_sha256': score.evaluator_sha256,
                  'native_provenance': provenance}
        parity.update({r + '_sha256': sha(bodies[r]) for r in ('input', 'evaluation', 'rust_source', 'rust_executable')})
        bodies['rust_parity'] = canonical(parity)
        bodies['score_records'] = build_records(score, ph, key, scores, sha(bodies['rust_parity']), 'a' * 40,
                                                1791000000, canonical)
        artifacts = {r: self.s3.put(f'gp/native/{key}/{r}', bodies[r]) for r in C.ARTIFACT_ROLES}
        entry = {'schema': C.RESULT_SCHEMA, 'context': asdict(score), 'evaluation_key': key,
                 'program_sha256': ph, 'status': 'completed', 'measurement': 'executed', 'artifacts': artifacts}
        record = self.s3.put(wrapper_key or f'gp/native/{key}/record.json', canonical(entry))
        return {'key': key, 'record_ref': record, 'artifact_refs': artifacts, 'program': dsl, 'program_sha256': ph,
                'rmse': scores}

    # ------------------------------------------------------------------ real outbox emissions -> uniform rows
    def emit(self, name, raw_program, *, phase='training', campaign=CAMPAIGN, descriptor=None):
        self.counter += 1
        n = self.counter
        c = self.ctx.get(name)
        cid = c['cid'] if c else name
        d = descriptor if descriptor is not None else self.nat.contexts[cid]

        def emission(ph, run, desc, raw):
            return {'phase': ph, 'data_run_id': run, 'context_id': run, 'parent_lineage_sha256': desc['parent_lineage_sha256'],
                    'input_descriptor': desc, 'raw_program_utf8': raw, 'raw_program_sha256': sha(raw.encode()),
                    'decoder_seed': n, 'decoder_protocol_sha256': digest('w3-fixture-decoder'),
                    'decode_artifact_sha256': digest({'w3-fixture-decode': n})}
        if phase == 'training':
            batch_train, emissions = [cid], [emission('training', cid, d, raw_program)]
            dev = []
        else:
            a = self.ctx['A']['cid']
            batch_train, dev = [a], [cid]
            emissions = [emission('training', a, self.nat.contexts[a], '(pipe X (ridge alpha=1.0))'),
                         emission('validation', cid, d, raw_program)]
        batch = {'batch_id': digest({'w3-fixture-batch': n}), 'campaign_sha256': campaign,
                 'checkpoint_sha256': digest({'w3-fixture-checkpoint': n}), 'global_step': n,
                 'manifest_sha256': digest('w3-fixture-manifest'), 'train_data_run_ids': batch_train,
                 'validation_data_run_ids': dev, 'emitted_train_data_run_ids': batch_train}
        ack = self.outbox.record_batch(batch, emissions)
        eid = ack['emission_ids'][-1]
        stored = bytes(self.outbox.db.execute('SELECT body FROM emissions WHERE id=?', (eid,)).fetchone()[0])
        return eid, stored, json.loads(stored)

    def row(self, name, res=None, *, raw=None, phase='training', state='published_verified', campaign=CAMPAIGN,
            program=None, descriptor=None, outbox_kind='v1'):
        """One uniform W4-shaped verified-result row backed by a real frozen-Outbox emission body."""
        program = res['program'] if res else (program or '(pipe X (ridge alpha=77.0))')
        eid, stored, body = self.emit(name, raw if raw is not None else program, phase=phase, campaign=campaign,
                              descriptor=descriptor)
        cid = body['emission']['context_id']
        verified = state in C.VERIFIED_STATES
        obs = {'schema': 'jope.drain-old-observation/v1', 'outbox_path': str(self.root / 'fixture-outbox.sqlite'),
               'config_sha256': sha(canonical(self.outbox.config)), 'backup_sha256': BACKUP_SHA,
               'backup_head': {'events': 1, 'head_sha256': digest({'w3-fixture-head': 1})}, 'emission_id': eid,
               'body_sha256': sha(stored), 'context_id': cid, 'phase': phase, 'campaign_sha256': campaign,
               'canonical_program_sha256': sha(canonicalize(body['emission']['raw_program_utf8']).encode()),
               'guard_key': digest({'context': cid, 'program': sha(program.encode())}),
               'evaluation_key': res['key'] if (res and verified) else None}
        return {'outbox_kind': outbox_kind, 'observation': obs, 'emission_body_b64': base64.b64encode(stored).decode('ascii'),
                'emission_body_sha256': sha(stored), 'state': state,
                'record_ref': res['record_ref'] if (res and verified) else None,
                'artifact_refs': res['artifact_refs'] if (res and verified) else None,
                'evaluation_key': res['key'] if (res and verified) else None, 'context_id': cid, 'phase': phase,
                'campaign': campaign, 'program': program if (res and verified) else None}

    def variant_row(self):
        vid = digest({'w3-fixture-variant': 1})
        desc = {'context_id': vid, 'source_kind': 'flywheel_variant', 'variant_identity': digest('variant'),
                'parent_lineage_sha256': self.ctx['A']['lineage'], 'partition': 'train'}
        return self.row(vid, program='(pipe X (ridge alpha=5.0))', descriptor=desc, state='published_verified')

    # ------------------------------------------------------------------ build helpers
    def args(self, root, **over):
        args = dict(native=self.nat, store=self.s3, authority=self.authority, inputs=self.inputs,
                    canonicalize=canonicalize, native_spec_bytes=self.spec_bytes, preprocessing_bytes=self.pre,
                    preprocessing_sha256=sha(self.pre), registry_bytes=self.registry, objective_sha256=digest(NB.OBJECTIVE), source_pins=self.source_pins,
                    r6_roster_bytes=self.r6, r6_roster_sha256=sha(self.r6), campaigns=[CAMPAIGN], sources=self.sources,
                    max_padded_tokens=8192, root=self.root / root, deadline=time.time() + 600, predecessor_chain=None, predecessor_head=None)
        args.update(over)
        return args

    def build(self, rows, root, **over):
        return build_packet(iter(rows), **self.args(root, **over))


def npy(a):
    import io
    f = io.BytesIO()
    np.save(f, a, allow_pickle=False)
    return f.getvalue()


def new_root():
    return tempfile.TemporaryDirectory()
