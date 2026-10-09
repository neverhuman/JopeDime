"""Two native-compatible jope.score/v1 rows for measured S1 inner folds.

Canonical bytes MUST come from a separately pinned native serde_json callback.
Python JSON encoding is never a fallback for row hashes or cut identities.
This fixed subset omits optional cost; it does not publish HWM winner records.
"""
from dataclasses import asdict
import hashlib
import json
import math
import re

PROTOCOL = 'jope.flywheel-s1-inner-rmse/v1'
FIELDS = frozenset(('schema', 'dataset_hash', 'cut_id', 'method_id', 'evaluator_sha256',
    'protocol', 'metric', 'value', 'direction', 'benchmark_valid', 'receipt_sha256',
    'run_id', 'git_sha', 'ts_unix', 'source', 'row_sha256'))


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def native_bytes(value, nativecanonical_json):
    need(callable(nativecanonical_json), 'pinned native canonical JSON callback required')
    result = nativecanonical_json(value)
    need(type(result) is bytes and 0 < len(result) <= 2 * 1024**2, 'bounded native canonical bytes required')
    need(json.loads(result) == value, 'native canonical callback changed the JSON value')
    return result


def inner_cut(context, fold, nativecanonical_json):
    need(type(fold) is int and fold in (0, 1), 'exact inner fold index required')
    descriptor = {'schema': 'jope.flywheel-inner-cut/v1', 'context_id': context.identity,
        'parent_cut_id': context.cut_id, 'split_sha256': context.split_sha256,
        'fold_protocol': 'jope.s1-two-fold/v1', 'validation_fold': fold}
    return sha(native_bytes(descriptor, nativecanonical_json))


def source(context, evaluation_key):
    return {'kind': 'local_lambda', 'source_kind': 'executed_native_s1',
        'source_revision': context.evaluator_sha256, 'source_uri': 'jope:flywheel-evaluation:' + evaluation_key,
        'notes': 'Fresh measured inner-fold RMSE in prepared input target units; benchmark_valid=false; no HWM promotion.'}


def build_records(context, program_sha256, evaluation_key, values, receipt_sha256,
                  source_commit, timestamp, nativecanonical_json):
    need(len(values) == 2 and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values),
         'two finite measured RMSE values required')
    need(type(timestamp) is int and 0 <= timestamp < 2**64, 'uint64 execution timestamp required')
    need(type(source_commit) is str and re.fullmatch('[0-9a-f]{40}', source_commit), 'source commit required')
    rows = []
    for fold, value in enumerate(values):
        row = {'schema': 'jope.score/v1', 'dataset_hash': context.dataset_sha256,
            'cut_id': inner_cut(context, fold, nativecanonical_json), 'method_id': 'gp:' + program_sha256,
            'evaluator_sha256': context.evaluator_sha256, 'protocol': PROTOCOL, 'metric': 'rmse',
            'value': float(value), 'direction': 'lower', 'benchmark_valid': False,
            'receipt_sha256': receipt_sha256, 'run_id': evaluation_key + ':inner:' + str(fold),
            'git_sha': source_commit, 'ts_unix': timestamp, 'source': source(context, evaluation_key), 'row_sha256': ''}
        row['row_sha256'] = sha(native_bytes(row, nativecanonical_json))
        rows.append(row)
    return native_bytes(rows, nativecanonical_json)


def validate_records(body, context, program_sha256, evaluation_key, values, receipt_sha256,
                     source_commit, timestamp, nativecanonical_json):
    need(type(body) is bytes and 0 < len(body) <= 2 * 1024**2, 'bounded canonical score array required')
    rows = json.loads(body)
    need(type(rows) is list and len(rows) == 2, 'exactly two canonical inner-fold score rows required')
    need(body == native_bytes(rows, nativecanonical_json), 'score array is not native canonical JSON')
    expected = json.loads(build_records(context, program_sha256, evaluation_key, values, receipt_sha256,
                                       source_commit, timestamp, nativecanonical_json))
    for index, (row, want) in enumerate(zip(rows, expected)):
        need(type(row) is dict and set(row) == FIELDS, 'canonical score fields differ from the fixed native subset')
        need(type(row['value']) in (int, float) and math.isfinite(row['value']) and row['value'] >= 0,
             'finite observed RMSE required')
        need(type(row['benchmark_valid']) is bool and row['benchmark_valid'] is False, 'pilot cannot become benchmark-valid')
        need(type(row['ts_unix']) is int, 'native timestamp must be an integer')
        blank = dict(row, row_sha256='')
        need(row['row_sha256'] == sha(native_bytes(blank, nativecanonical_json)), 'native canonical row SHA mismatch')
        need(row == want, 'canonical score identity/value/provenance mismatch at inner fold ' + str(index))
    return rows
