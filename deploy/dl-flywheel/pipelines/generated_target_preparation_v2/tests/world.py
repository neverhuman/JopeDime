"""Shared private fixture helpers for the C10 r2 packages (tests only; no live data, S3, GP, planner or GPU).

Production-shaped pins are REAL where they are campaign invariants: the exact production objective bytes
(sha256 bf02f873..., reproduced from the pinned continuous-runtime document), the KumoSmall 2051 roster and weight,
and the actually imported labels/compose source bytes.  Data, normalizer and authority proofs are isolated fixture
values.  Stores are wrapped so every artifact read is counted and attested (C10a revision 2, finding A3).
"""
import json
import math
import time
from pathlib import Path

import numpy as np

from dltrain import labels
from jope_flywheel import native_variant_snapshot as V
from jope_flywheel import training_snapshot_kumos as S
from jope_flywheel.program_gate import request_identity, verified_measurement
from jope_flywheel.protocol import Context, canonical
from jope_flywheel.score_join import build_records
import prepared_snapshot as P
from pipelines.generated_target_admission_v1.common import canonical as xcanonical, digest as xdigest, sha
from pipelines.generated_target_preparation_v2 import preparation as C

BASE = Path(P.__file__).resolve().parents[2]
PRODUCTION_OBJECTIVE_SHA256 = 'bf02f87341bc52aa4b3c329492d1eae5621c3543a9e60c9b3f34aa7b407ece48'
OBJECTIVE_BYTES = canonical(json.loads((BASE / 'docs/continuous-runtime-pins-20261009.json').read_bytes())
                            ['neural']['objective'])
assert sha(OBJECTIVE_BYTES) == PRODUCTION_OBJECTIVE_SHA256, 'production objective bytes reproduce the live pin'
PROVENANCE = {'request_sha256': '1' * 64, 'cutoff_sha256': '2' * 64, 'x1_source_packet_sha256': '9' * 64}


class AuditedStore:
    """Closed exact-version local store wrapper with audited counters (``network=True`` simulates remote GETs)."""

    def __init__(self, inner, *, network=False, on_read=None):
        self.inner, self.network, self.on_read = inner, network, on_read
        self.reads = self.read_bytes = self.network_gets = 0

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def get_version(self, ref):
        body, version = self.inner.get_version(ref)
        self.reads += 1
        self.read_bytes += len(body)
        self.network_gets += 1 if self.network else 0
        if self.on_read is not None:
            self.on_read(ref)
        return body, version

    def audit(self):
        return {'schema': C.STORE_AUDIT, 'closed_exact_version_local': True, 'network_gets': self.network_gets,
                'local_reads': self.reads, 'local_bytes': self.read_bytes}


def contract(pre, *, packet='9' * 64):
    """The campaign-invariant production contract for a fixture normalizer."""
    return {'schema': C.CONTRACT_SCHEMA, 'objective_sha256': sha(OBJECTIVE_BYTES), 'normalizer_sha256': sha(pre),
            'roster': dict(C.ROSTER),
            'construction_sources': {'labels': sha(Path(labels.__file__).resolve().read_bytes()),
                                     'compose': sha(Path(V.__file__).resolve().read_bytes())},
            'x1_source_packet_sha256': packet}


def add_program(f, task, dsl, rmse):
    """Append one measured program to an existing fixture task; mirrors the fixture world's builder bytes."""
    raw = f.raws[task['variant_identity']]
    y, folds, _ = S.decode_input(raw)
    ctx = Context(**task['context'])
    ph = S.byte_sha(dsl.encode())
    key = request_identity(ctx, dsl)
    ev = {'schema': 'jope.s1-evaluation/v1', 'status': 'completed', 'fold_protocol': 'jope.s1-two-fold/v1',
          'inner_cv_folds': 2, 'input_semantics': 'unscaled_fit_rows_v1',
          'program': {'dsl': dsl, 'program_sha256': ph}, 'inner_cv': []}
    for i in (0, 1):
        target = y[folds == i]
        train = y[folds != i]
        base = math.sqrt(math.fsum((float(v) - float(np.mean(train))) ** 2 for v in target) / len(target))
        ev['inner_cv'].append({'schema': 'jope.teacher-fit/v2', 'kind': 'inner_cv', 'index': i,
                               'score_rows': int(len(target)), 'train_rows': int(len(train)),
                               'metrics': {'rmse': rmse}, 'baseline_metrics': {'rmse': base},
                               'predictions': (target - rmse).tolist(), 'targets': target.tolist(),
                               'serialized_prediction_replay_exact': True, 'wall_us': 100})
    bodies = {'input': raw, 'program': dsl.encode(), 'evaluation': canonical(ev),
              'rust_source': b'fixture code', 'rust_executable': b'fixture binary'}
    parity = {'schema': 'jope.flywheel-rust-parity/v1', 'passed': True, 'executed': True, 'folds_verified': 2,
              'evaluation_key': key, 'program_sha256': ph, 'evaluator_sha256': '3' * 64,
              'native_provenance': {'schema': 'jope.flywheel-native-execution-provenance/v1',
                                    'evaluator_sha256': '3' * 64, 'input_sha256': S.byte_sha(raw),
                                    'program_sha256': ph, 'evaluation_sha256': S.byte_sha(bodies['evaluation']),
                                    'source_commit': 'a' * 40, 'execution_timestamp': 123}}
    parity.update({k + '_sha256': S.byte_sha(bodies[k]) for k in ('input', 'evaluation', 'rust_source', 'rust_executable')})
    bodies['rust_parity'] = canonical(parity)
    bodies['score_records'] = build_records(ctx, ph, key, [rmse, rmse], S.byte_sha(bodies['rust_parity']),
                                            'a' * 40, 123, canonical)
    artifacts = {n: f.store.put('fixture/program/' + key + '/' + n, b) for n, b in bodies.items()}
    entry = {'schema': 'jope.flywheel-program-result/v1', 'context': ctx.__dict__, 'program_sha256': ph,
             'evaluation_key': key, 'status': 'completed', 'measurement': 'executed', 'artifacts': artifacts}
    return f.store.put('fixture/program/' + key + '/wrapper', canonical(entry))


def annotate(ref):
    """The established seven-field annotated reference shape for an exact five-field reference."""
    return {**ref, 'state': 'published_verified', 'verification': 'exact VersionId GET + SHA256'}


def x1_manifest(f, old_spec, new_spec, appended, *, pre, rows_for=None):
    """An X1-shaped reference manifest (exact v2 wire keys, production contract fields) for appended refs."""
    reader = V.Reader(f.store, 256 * 1024**2, time.time() + 60)
    rows, census = [], {k: [] for k in C.CENSUS}
    for index, ref in enumerate(appended if rows_for is None else rows_for):
        entry = f.store.doc(ref)
        ctx = Context(**entry['context'])
        program = entry['artifacts']['program']
        dsl = f.store.blobs[(program['key'], program['version_id'])].decode()
        measured = verified_measurement(reader, ctx, dsl, entry)
        task = next(t for t in new_spec['tasks'] if Context(**t['context']).identity == ctx.identity)
        emission = xdigest({'fixture_emission': index, 'evaluation_key': entry['evaluation_key']})
        rows.append({'context_id': ctx.identity, 'parent_lineage_sha256': task['parent_lineage_sha256'],
                     'partition': 'train', 'evaluation_key': entry['evaluation_key'],
                     'program_sha256': entry['program_sha256'], 'result_ref': ref, 'artifact_refs': entry['artifacts'],
                     'measurement': measured, 'ce_eligible': False, 'source_emission_ids': [emission],
                     'source_batch': {'batch_id': 'fixture-batch'}, 'source_optimizer_commit_verified': False})
        census['appended'].append({'emission_id': emission, 'state': 'published_verified', 'phase': 'train',
                                   'context_id': ctx.identity, 'evaluation_key': entry['evaluation_key']})
    rows.sort(key=lambda r: r['evaluation_key'])
    predecessor = xcanonical(old_spec)
    pins = contract(pre)
    roles = {t['parent_lineage_sha256']: 'train' for t in new_spec['tasks']}
    proof = {'objective_sha256': pins['objective_sha256'], 'normalizer_sha256': pins['normalizer_sha256'],
             'roster': pins['roster'], 'parent_roles': roles}
    manifest = {'schema': C.MANIFEST_SCHEMA, 'request_sha256': PROVENANCE['request_sha256'],
                'cutoff_sha256': PROVENANCE['cutoff_sha256'],
                'cutoff_head': {'events': 1, 'head_sha256': '3' * 64,
                                'observations': sum(len(v) for v in census.values()), 'batches': 1},
                'predecessor_sha256': sha(predecessor), 'extension': new_spec, 'extension_sha256': xdigest(new_spec),
                'objective_sha256': pins['objective_sha256'], 'normalizer_sha256': pins['normalizer_sha256'],
                'construction_sources': {'labels': {'path': str(Path(labels.__file__).resolve()),
                                                    'sha256': pins['construction_sources']['labels']},
                                         'compose': {'path': str(Path(V.__file__).resolve()),
                                                     'sha256': pins['construction_sources']['compose']}},
                'roster': pins['roster'], 'authority_proofs': [dict(proof), dict(proof)], 'rows': rows,
                'census': census, 'counts': {k: len(v) for k, v in census.items()},
                'artifact_reads': {'calls': 0, 'attempted_bytes': 0}, **C.MANIFEST_FIXED}
    return xcanonical(manifest), predecessor
