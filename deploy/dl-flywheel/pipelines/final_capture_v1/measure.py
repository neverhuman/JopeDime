"""Capacity measurement for final_capture_v1 parts (CPU only; writes a create-only receipt).

profile() serializes SYNTHETIC join/fingerprint/exposed records with live-scale field widths (64-hex identities,
step/epoch/index/ack integers at the stated scale) through the real write_parts/read_parts, so part counts and
bytes are measured, not estimated; they are labelled synthetic. main() also measures the actual join-record bytes
of a real fixture-world capture and the inherit() event headroom (canonical ancestry vs MAX_EVENT).
"""
import argparse, json, os, tempfile, time
from pathlib import Path
from jope_flywheel.protocol import canonical, digest
from neural_native_stream import MAX_EVENT
from . import capture as C

EPOCH_BATCHES = 437          # ceil(6982 live training contexts / batch 16)
ACKS_PER_BATCH = 5           # one batch.emitted + up to two lease/resolve events per emission (two emissions)


def synthetic_join(i, origin):
    h = lambda tag: digest({'synthetic-join': tag, 'i': i})
    return {'absolute_step': origin + i, 'local_step': i, 'batch_id': h('batch'), 'epoch': 1 + (i - 1) // EPOCH_BATCHES,
            'index': (i - 1) % EPOCH_BATCHES, 'fingerprint': h('fingerprint'), 'proof_sha256': h('proof'),
            'checkpoint_before': h('before'), 'parameter_sha256': h('after'), 'ack_seq': ACKS_PER_BATCH * i}


def profile(directory, *, joins, lineage_fingerprints, exposed, origin=429, part_max=C.PART_MAX):
    """Write and re-read the three chained sections at the given scale; return measured sizes."""
    d = Path(directory); out = {}
    sections = {'joins': [synthetic_join(i, origin) for i in range(1, joins + 1)],
                'fingerprints': sorted(digest({'synthetic-fingerprint': i}) for i in range(lineage_fingerprints)),
                'exposed': sorted(digest({'synthetic-context': i}) for i in range(exposed))}
    for kind, records in sections.items():
        chain = C.write_parts(d, kind, records, part_max=part_max)
        assert C.read_parts(d, chain) == records
        sizes = [p['bytes'] for p in chain['parts']]
        out[kind] = {'records': len(records), 'parts': len(sizes), 'max_part_bytes': max(sizes), 'total_bytes': sum(sizes),
                     'bytes_per_record': sum(sizes) / max(1, len(records)), 'part_max': part_max,
                     'chain_entry_bytes': len(canonical(chain['parts'][0]))}
    ancestry = {'planner_ref': '/' + 'x' * 120, 'head': {'schema': C.LINEAGE_HEAD, 'committed_batches': lineage_fingerprints,
                'unresolved_batches': 0, 'generations': [{'planner_ref': '/' + 'x' * 120, 'head': {'events': 10**6, 'committed_batches': 10**5,
                'unresolved_batches': 0, 'head_sha256': '0' * 64}}] * 3}, 'checkpoint_sha256': '0' * 64, 'optimizer_steps': lineage_fingerprints,
                'completed_epochs': 400, 'fingerprints': sections['fingerprints'], 'exposed_context_ids': sections['exposed'], 'presentations': 10**7}
    size = len(canonical(ancestry)); per = 67  # 64 hex + 2 quotes + 1 comma inside the canonical list
    out['inherit_event'] = {'ancestry_canonical_bytes': size, 'max_event_bytes': MAX_EVENT, 'headroom_bytes': MAX_EVENT - size,
                            'max_lineage_fingerprints_at_this_exposure': lineage_fingerprints + (MAX_EVENT - size) // per}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument('--receipt', type=Path, required=True); args = parser.parse_args(argv)
    from .tests.world import world
    started = time.time(); w = world('killed')
    with tempfile.TemporaryDirectory(prefix='w10-measure-') as tmp:
        base = Path(tmp).resolve(); root = w.private_copy(base / 'copy'); body, receipt = w.capture(root, base / 'out')
        part = body['planner']['joins']['parts'][0]
        fixture = {'joins': body['planner']['joins']['count'], 'join_bytes_per_record': part['bytes'] / part['count'],
                   'document_bytes': receipt['capture_bytes'], 'receipt_files': len(receipt['files']),
                   'planner_backup_bytes': receipt['files']['planner-backup.sqlite']['bytes']}
        scales = {}
        for name, joins, fps in (('v1_actual_17773', 17773, 17773 + 429), ('v2_cap_80000', 80000, 17773 + 80000)):
            d = base / name; d.mkdir(); scales[name] = profile(d, joins=joins, lineage_fingerprints=fps, exposed=7248)
            scales[name]['document_bytes_estimate'] = fixture['document_bytes'] + sum(
                (s['parts'] - 1) * s['chain_entry_bytes'] for k, s in scales[name].items() if k in ('joins', 'fingerprints', 'exposed'))
    value = {'schema': 'jope.w10-capacity-measurement/v1', 'fixture_world': fixture, 'synthetic_scales': scales,
             'part_max': C.PART_MAX, 'document_max': C.DOC_MAX, 'started_unix': started, 'finished_unix': time.time(),
             'scope': 'Fixture-world record bytes are measured from a real capture; scale rows serialize synthetic records of live field widths.'}
    fd = os.open(args.receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, 'w') as f: f.write(json.dumps(value, indent=1, sort_keys=True) + '\n'); f.flush(); os.fsync(f.fileno())
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ('joins', 'inherit_event')} for k, v in scales.items()}))


if __name__ == '__main__':
    main()
