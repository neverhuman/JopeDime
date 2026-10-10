"""Pure budget preflight. No I/O, authority issuance, body verification or admission."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import PurePosixPath
import re

BODY_LIMIT = 8 * 1024**2
METADATA_LIMIT = 1024**2
READ_LIMIT = RETAINED_LIMIT = 32 * 1024**2
TRANSPORT_LIMIT = 48 * 1024**2

class Refusal(ValueError):
    pass

def need(value, why):
    if not value:
        raise Refusal(why)

def closed(value, keys, why):
    need(type(value) is dict and set(value) == set(keys), why)

def integer(value, lo, hi, why):
    need(type(value) is int and lo <= value <= hi, why)
    return value

def sha(value):
    need(type(value) is str and re.fullmatch(r'[0-9a-f]{64}', value) is not None, 'sha256')

def identifier(value):
    need(type(value) is str and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', value) is not None, 'identifier')

def absolute(value):
    need(type(value) is str and 1 < len(value) <= 4096 and '\x00' not in value,
         'absolute path')
    need(value.startswith('/') and str(PurePosixPath(value)) == value and
         all(part not in ('', '.', '..') for part in value.split('/')[1:]), 'absolute path')

def pairs(rows):
    result = {}
    for key, value in rows:
        need(key not in result, 'duplicate JSON key')
        result[key] = value
    return result

@dataclass(frozen=True)
class ExternalPlanPin:
    """Root must independently supply this pin. Type/hash checks do not prove its origin."""
    sha256: str
    bytes: int

@dataclass(frozen=True)
class Projection:
    state: str
    plan_sha256: str
    original_member_count: int
    fresh_read_bytes_reserved: int
    retained_bytes_reserved: int
    transport_bytes_reserved: int
    source_body_reads: int = 0
    admission_authorized: bool = False


def project(plan_wire, external):
    """Reject an over-budget finite roster before any caller opens a payload.

    WITHIN_BUDGET_NOT_ADMISSION is only arithmetic. The caller still needs complete
    external history, identities, policy, semantic source/role/clock and actual
    physical/RSS/durable one-use qualification. This module cannot open files.
    """
    need(type(external) is ExternalPlanPin, 'independently supplied typed plan pin')
    sha(external.sha256)
    integer(external.bytes, 1, METADATA_LIMIT, 'plan metadata bound')
    need(type(plan_wire) is bytes and len(plan_wire) == external.bytes, 'exact plan bytes')
    need(hashlib.sha256(plan_wire).hexdigest() == external.sha256, 'external plan hash')
    try:
        plan = json.loads(plan_wire.decode('utf-8'), object_pairs_hook=pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(Refusal('nonfinite JSON')))
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise Refusal('bounded unique-key JSON') from exc
    closed(plan, ('schema', 'roster', 'metadata', 'output_reservations',
                  'transport_envelope_reservation_bytes'), 'closed plan, no authority echo or caps')
    need(plan['schema'] == 'codex.legacy-producer-budget-preflight/v1', 'plan schema')
    rows, metadata, outputs = plan['roster'], plan['metadata'], plan['output_reservations']
    need(type(rows) is list and 1 <= len(rows) <= 4096, 'original roster bound')
    need(type(metadata) is list and 1 <= len(metadata) <= 4096, 'metadata roster bound')
    need(type(outputs) is list and 1 <= len(outputs) <= 4096, 'output reservation bound')
    read = retained = len(plan_wire)
    transport = len(plan_wire) + integer(plan['transport_envelope_reservation_bytes'],
                                        1, TRANSPORT_LIMIT, 'full serialization envelope reserve')
    paths, ids = set(), set()
    for row in rows:
        closed(row, ('namespace', 'original_path', 'sha256', 'bytes', 'read_count',
                     'retained_copies', 'serialized_copies'), 'original row fields')
        identifier(row['namespace']); absolute(row['original_path']); sha(row['sha256'])
        need(row['original_path'] not in paths, 'original path duplicate or alias')
        paths.add(row['original_path'])
        size = integer(row['bytes'], 1, BODY_LIMIT, 'single body bound')
        read += size * integer(row['read_count'], 1, 8, 'read count')
        retained += size * integer(row['retained_copies'], 1, 8, 'retained copies')
        transport += 4 * ((size + 2) // 3) * integer(row['serialized_copies'], 0, 8, 'serialized copies')
    for row in metadata:
        closed(row, ('id', 'sha256', 'bytes', 'read_count', 'retained_copies',
                     'serialized_copies'), 'metadata row fields')
        identifier(row['id']); sha(row['sha256']); need(row['id'] not in ids, 'metadata ID duplicate')
        ids.add(row['id'])
        size = integer(row['bytes'], 1, METADATA_LIMIT, 'metadata body bound')
        read += size * integer(row['read_count'], 1, 8, 'metadata read count')
        retained += size * integer(row['retained_copies'], 1, 8, 'metadata retention')
        transport += 4 * ((size + 2) // 3) * integer(row['serialized_copies'], 0, 8, 'metadata serialization')
    reservations = set()
    for row in outputs:
        closed(row, ('id', 'maximum_body_bytes', 'maximum_readbacks',
                     'maximum_retained_copies', 'maximum_transport_bytes'),
               'future output capacity only; no fabricated identity')
        identifier(row['id']); need(row['id'] not in reservations, 'output reservation duplicate')
        reservations.add(row['id'])
        size = integer(row['maximum_body_bytes'], 1, BODY_LIMIT, 'output body bound')
        read += size * integer(row['maximum_readbacks'], 1, 8, 'output readbacks')
        retained += size * integer(row['maximum_retained_copies'], 1, 8, 'output retention')
        transport += integer(row['maximum_transport_bytes'], size, TRANSPORT_LIMIT,
                             'output full serialization reserve')
    need(read <= READ_LIMIT, 'full packet read budget exceeded before payload IO')
    need(retained <= RETAINED_LIMIT, 'full packet retention exceeded before payload IO')
    need(transport <= TRANSPORT_LIMIT, 'full packet transport reservation exceeded')
    return Projection('WITHIN_BUDGET_NOT_ADMISSION', external.sha256, len(rows), read,
                      retained, transport)
