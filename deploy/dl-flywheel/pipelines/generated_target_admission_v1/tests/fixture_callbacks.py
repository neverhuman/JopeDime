"""Synthetic source-pinned callbacks. These are NOT production qualification.

They exercise exporter return/identity contracts with tiny local fake artifacts.
No native evaluator, network client or actual score verifier is represented.
"""
import copy
import time
from ..common import digest, document, need, sha

STORE = {}
ROLES = {}
MODE = None
CALLS = []


def read_version(ref):
    CALLS.append(('get', ref['key']))
    raw, version = STORE[(ref['bucket'], ref['key'], ref['version_id'])]
    if MODE == 'wrong_version':
        return raw, version + '-stale'
    if MODE == 'corrupt':
        return raw + b'x', version
    return raw, version


def canonicalize(raw):
    CALLS.append(('canonicalize', raw))
    return ' '.join(raw.split())


def verify(ctx, dsl, entry, bodies):
    CALLS.append(('verify', entry['evaluation_key']))
    if MODE == 'bare_true':
        return True
    ev = document(bodies['evaluation'], 1024)
    parity = document(bodies['rust_parity'], 1024)
    need(parity['passed'] is True and ev['context'] == ctx and ev['dsl'] == dsl,
         'synthetic parity/context failure')
    result = {'evaluation_key': entry['evaluation_key'], 'program_sha256': sha(dsl.encode()),
              'context_id': digest(ctx), 'measurement': 'executed', 'status': 'completed',
              'teaching_eligible': False, 'inner_fold_rmse': ev['rmse'],
              'evaluation_sha256': sha(bodies['evaluation']),
              'artifact_versions': {k: v['version_id'] for k, v in entry['artifacts'].items()},
              'canonical_score_row_sha256': [sha(b'row0'), sha(b'row1')]}
    if MODE == 'invented_q':
        result['q_opt'] = 42.
    if MODE == 'wrong_measurement_context':
        result['context_id'] = 'f' * 64
    return result


def authority(request):
    CALLS.append(('authority', None))
    result = {'schema': 'jope.generated-target-current-authority/v1', 'state': 'passed',
              'request_sha256': digest(request), 'checked_at_unix': time.time(),
              'source_pins': copy.deepcopy(request['authority_source_pins']),
              'descriptors': copy.deepcopy(request['admitted_descriptors']),
              'parent_roles': copy.deepcopy(ROLES),
              'objective_sha256': request['objective_sha256'],
              'normalizer_sha256': request['normalizer_sha256'],
              'native_train_context_ids_sha256': request['native_train_context_ids_sha256'],
              'roster': copy.deepcopy(request['roster'])}
    if MODE == 'stale_authority':
        result['checked_at_unix'] -= 61
    if MODE == 'wrong_authority_objective':
        result['objective_sha256'] = 'f' * 64
    if MODE == 'role_drift' and sum(c[0] == 'authority' for c in CALLS) > 1:
        result['parent_roles'] = {k: 'validation' for k in ROLES}
    return result
