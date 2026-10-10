"""Closed, bounded data contracts. This module performs no IO."""
import hashlib
import json
import re
from decimal import Decimal
from pathlib import PurePosixPath

BODY_LIMIT = 8 * 1024**2
EVENT_LIMIT = 512 * 1024
SNAPSHOT_LIMIT = 7_500_000
FRESH_LIMIT = 32 * 1024**2
RETAINED_LIMIT = 32 * 1024**2
TRANSPORT_LIMIT = 48 * 1024**2
METADATA_LIMIT = 1024 * 1024
ZERO = '0' * 64

class Refusal(ValueError):
    """Known invalid evidence. Never an authorization to retry."""

def need(condition, why):
    if not condition:
        raise Refusal(why)

def closed(value, keys, why):
    need(type(value) is dict and set(value) == set(keys), why)
    return value

def integer(value, lo, hi, why):
    need(type(value) is int and lo <= value <= hi, why)
    return value

def digest(wire):
    need(type(wire) is bytes, 'exact bytes required')
    return hashlib.sha256(wire).hexdigest()

def sha(value):
    need(type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None, 'SHA256')
    return value

def number(value):
    need(type(value) in (int, float), 'finite number, not boolean')
    d = Decimal(str(value))
    need(d.is_finite(), 'finite number')
    return d

def canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                          allow_nan=False).encode('utf-8')
    except (TypeError, ValueError, OverflowError) as exc:
        raise Refusal('canonical JSON value') from exc

def _pairs(rows):
    result = {}
    for key, value in rows:
        need(key not in result, 'duplicate JSON key')
        result[key] = value
    return result

def document(wire, limit=METADATA_LIMIT, canonical_required=True):
    need(type(wire) is bytes and 0 < len(wire) <= limit, 'bounded JSON body')
    try:
        value = json.loads(wire.decode('utf-8'), object_pairs_hook=_pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(Refusal('nonfinite JSON')))
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise Refusal('valid finite unique-key JSON') from exc
    if canonical_required:
        need(canonical(value) == wire, 'canonical JSON bytes')
    return value

def absolute(path):
    need(type(path) is str and 0 < len(path) <= 4096 and '\x00' not in path, 'bounded path')
    pp = PurePosixPath(path)
    need(path.startswith('/') and path != '/' and str(pp) == path and
         all(x not in ('.', '..', '') for x in path.split('/')[1:]), 'normalized absolute path')
    return path

def relative(path):
    need(type(path) is str and 0 < len(path) <= 1024 and '\x00' not in path, 'bounded member')
    need(not path.startswith('/') and str(PurePosixPath(path)) == path and
         all(x not in ('.', '..', '') for x in path.split('/')), 'safe relative member')
    return path

def identifier(value):
    need(type(value) is str and re.fullmatch('[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}', value) is not None,
         'namespace identifier')
    return value

def content_pin(pin, limit=BODY_LIMIT):
    closed(pin, ('sha256', 'bytes'), 'content pin fields')
    sha(pin['sha256']); integer(pin['bytes'], 1, limit, 'content bytes')
    return pin

def file_pin(pin, immutable=False):
    keys = {'path', 'sha256', 'bytes', 'device', 'inode', 'mtime_ns'}
    if immutable:
        keys |= {'uid', 'mode', 'nlink', 'file_type'}
    closed(pin, keys, 'physical pin fields')
    absolute(pin['path']); sha(pin['sha256']); integer(pin['bytes'], 1, BODY_LIMIT, 'physical bytes')
    integer(pin['device'], 0, 2**63-1, 'device'); integer(pin['inode'], 1, 2**63-1, 'inode')
    integer(pin['mtime_ns'], 0, 2**63-1, 'mtime')
    if immutable:
        integer(pin['uid'], 0, 2**31-1, 'uid')
        need(type(pin['mode']) is int and pin['mode'] in (0o400, 0o444), 'read-only custody mode')
        need(type(pin['nlink']) is int and pin['nlink'] == 1 and pin['file_type'] == 'regular',
             'single-link regular custody')
    return pin

def exact_body(wire, pin):
    need(type(wire) is bytes and len(wire) == pin['bytes'] and digest(wire) == pin['sha256'],
         'exact captured bytes')

def clock(value):
    closed(value, ('boot_id', 'created_unix', 'deadline_unix', 'created_monotonic_ns',
                   'deadline_monotonic_ns'), 'paired clock fields')
    need(type(value['boot_id']) is str and re.fullmatch('[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}',
                                                     value['boot_id']) is not None, 'boot id')
    start, end = number(value['created_unix']), number(value['deadline_unix'])
    mono_start = integer(value['created_monotonic_ns'], 1, 2**63-1, 'monotonic start')
    mono_end = integer(value['deadline_monotonic_ns'], 1, 2**63-1, 'monotonic end')
    need(0 < end-start <= 3600 and 0 < mono_end-mono_start <= 3600*10**9, 'original bounded window')
    need(abs((end-start)*10**9-(mono_end-mono_start)) < 10_000, 'paired duration mismatch')
    return value
