"""Bind the actual S1 control receipt to the separately pinned build-plan bytes.

The caller must first read/hash-check native_build_receipt and the executable
descriptors using its bounded reader. The receipt has no source.plan_sha256.
The separate six-field plan is located uniquely in source_pins, read with the
same reader, and independently checked here before any native execution.
Receipt binary paths may be the s1-control copies; receipt SHA/size, not those
paths, establish equivalence to the configured s1-prod binaries.
"""
import hashlib
import json
import re
from pathlib import PurePosixPath

PLAN_FIELDS = {'source_commit', 'native_build_receipt', 'worker', 'emitter',
               'native_canonicalizer', 'rustc'}
PLAN_CAP = 1024 * 1024


def need(ok, message):
    if not ok:
        raise ValueError('exact native build: ' + message)


def hash_string(value, size=64):
    return type(value) is str and re.fullmatch('[0-9a-f]{%d}' % size, value) is not None


def descriptor(value):
    need(type(value) is dict and set(value) in ({'path', 'sha256'}, {'path', 'sha256', 'bytes'}),
         'closed descriptor')
    path = value['path']
    need(type(path) is str and PurePosixPath(path).is_absolute() and
         str(PurePosixPath(path)) == path and '..' not in PurePosixPath(path).parts,
         'absolute descriptor path')
    need(hash_string(value['sha256']), 'descriptor SHA')
    if 'bytes' in value:
        need(type(value['bytes']) is int and value['bytes'] > 0, 'descriptor byte count')


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        need(key not in value, 'duplicate plan field')
        value[key] = item
    return value


def verify_native_build(config, receipt, read):
    """Verify actual plan/receipt joins; ``read(ref)`` returns hash-checked bytes.

    A current descriptor may add a byte count absent from the historical plan.
    Fields actually recorded by the plan must match exactly; an optional count
    can never contradict it. The receipt independently binds source, honest
    worker and canonicalizer identity, including their byte counts when known.
    """
    pin = config['source_build_plan_sha256']
    need(hash_string(pin), 'build-plan SHA')
    need(type(config['source_pins']) is list, 'source pin list')
    matches = [r for r in config['source_pins'] if type(r) is dict and r.get('sha256') == pin]
    need(len(matches) == 1, 'one uniquely pinned build-plan descriptor')
    ref = matches[0]
    descriptor(ref)
    need(type(ref.get('bytes')) is int and 0 < ref['bytes'] <= PLAN_CAP, 'bounded build-plan bytes')
    raw = read(ref)
    need(type(raw) is bytes and len(raw) == ref['bytes'] and
         hashlib.sha256(raw).hexdigest() == pin, 'raw build-plan SHA and length')
    plan = json.loads(raw, object_pairs_hook=unique_object)
    need(type(plan) is dict and set(plan) == PLAN_FIELDS, 'closed six-field plan')
    need(hash_string(config['source_commit'], 40) and plan['source_commit'] == config['source_commit'],
         'plan source commit')
    for key in PLAN_FIELDS - {'source_commit'}:
        descriptor(plan[key]); descriptor(config[key])
        need(all(config[key].get(field) == value for field, value in plan[key].items()),
             'plan descriptor differs: ' + key)
    need(type(receipt) is dict and receipt.get('schema') == 'jope.dl-s1-control-receipt/v1',
         'S1 control receipt schema')
    need(type(receipt.get('source')) is dict and receipt['source'].get('pin') == config['source_commit'],
         'receipt source commit')
    for key, role in (('worker', 'worker_honest'), ('native_canonicalizer', 'proposal')):
        binary = receipt.get('binaries', {}).get(role)
        need(type(binary) is dict and binary.get('sha256') == config[key]['sha256'],
             'receipt binary SHA: ' + role)
        need(type(binary.get('bytes')) is int and binary['bytes'] > 0, 'receipt binary byte count: ' + role)
        if 'bytes' in config[key]:
            need(binary['bytes'] == config[key]['bytes'], 'receipt binary size: ' + role)
    return {'schema': 'jope.native-build-plan-binding/v2', 'source_commit': config['source_commit'],
            'source_build_plan_sha256': pin, 'plan_ref': ref,
            'native_build_receipt_sha256': config['native_build_receipt']['sha256'],
            'worker_sha256': config['worker']['sha256'],
            'native_canonicalizer_sha256': config['native_canonicalizer']['sha256']}
