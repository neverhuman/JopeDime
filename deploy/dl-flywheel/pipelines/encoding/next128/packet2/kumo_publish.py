"""Publish sealed evergreen encoder vectors; preserve original dataset metadata.

Read-only source verification, create-only S3 writes, exact-VersionId GET proof.
No GP measurements, training admission, or expansion database mutation.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import sys
import time

sys.dont_write_bytecode = True
SCHEMA = 'jope.flywheel-kumos-fresh32-publication/v1'
from campaign import ROOT as CAMPAIGN_ROOT,PACKET_ID,COUNT,GLOBAL_DEADLINE,PIPELINE_SHA
ROOT = Path(CAMPAIGN_ROOT)
BASE_SHA = '63b0a13aa76369c08553e9cfb360a6998492ac43e8c1a152e9e08c841b9bc965'
ENCODER_SHA = '0883dd4085e01cc0e1a3353396fd7004854c4ae2e68868f1eb79a053a6cb2a42'
AUTHORITY_SHA = 'bb633b6b2b46768380df0687deeb9155042953e3ddb3dd922c06f784a9f5b936'
DIMS = {'kumo_tabular_s': 2051}
LIMIT = 16 * 1024**2


def need(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(body):
    return hashlib.sha256(body).hexdigest()


def load_module(path, digest, name):
    p = Path(path)
    need(p.is_absolute() and all(not q.is_symlink() for q in (p, *p.parents)), 'unsafe module path')
    need(p.is_file() and p.stat().st_size <= LIMIT, 'bounded module required')
    need(sha(p.read_bytes()) == digest, 'reviewed module SHA changed')
    spec = importlib.util.spec_from_file_location(name, p)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check_ref(base, ref):
    need(ref['bucket'] == base.BUCKET and ref['key'].startswith(base.PREFIX), 'foreign prior object')
    base.digest(ref['sha256'])
    need(type(ref['bytes']) is int and 0 < ref['bytes'] <= LIMIT, 'prior object cap')
    need(type(ref['version_id']) is str and ref['version_id'] not in ('', 'null'), 'exact prior VersionId required')
    need(ref['verification'] == 'exact VersionId GET + SHA256', 'unverified prior object')


def get_verified(base, client, ref, deadline, *, load=False):
    """One bounded exact-version read, including an EOF deadline check."""
    check_ref(base, ref)
    base.tick(deadline)
    result = client.get_object(Bucket=ref['bucket'], Key=ref['key'], VersionId=ref['version_id'])
    stream = result['Body']
    digest, count, parts = hashlib.sha256(), 0, []
    try:
        need(result.get('VersionId') == ref['version_id'], 'GET returned another version')
        need(result.get('ContentLength') == ref['bytes'], 'GET byte length differs')
        while True:
            base.tick(deadline)
            chunk = stream.read(min(1024**2, ref['bytes'] - count + 1))
            base.tick(deadline)
            if not chunk:
                break
            count += len(chunk)
            need(count <= ref['bytes'], 'GET grew beyond exact object size')
            digest.update(chunk)
            if load:
                parts.append(chunk)
    finally:
        stream.close()
    need(count == ref['bytes'] and digest.hexdigest() == ref['sha256'], 'prior VersionId content mismatch')
    base.tick(deadline)
    return b''.join(parts) if load else None


def prior_binding(base, prior, variant, metadata):
    """Bind three unchanged dataset objects plus their original custody receipt."""
    identity = variant['identity']
    selected = [v for v in prior['variants'] if v['variant_identity'] == identity]
    need(len(selected) == 1 and selected[0]['sequence'] == variant['sequence'], 'prior variant mismatch')
    pfx = base.PREFIX + 'datasets/flywheel/v1/' + identity + '/'
    objects = {}
    for ref in prior['object_receipts']:
        if ref['key'].startswith(pfx):
            name = ref['key'][len(pfx):]
            need(name not in objects, 'duplicate prior object')
            objects[name] = ref
    need(set(objects) == {'encoder.jtt1.gz', 'metadata.json', 'generation.json'}, 'prior dataset closure')
    for name, ref in objects.items():
        check_ref(base, ref)
        source = variant['files'][name]
        need(ref['sha256'] == source['sha256'] and ref['bytes'] == source['bytes'], 'prior/source object mismatch')
    receipt = selected[0]['receipt']
    check_ref(base, receipt)
    need(receipt['key'] == base.PREFIX + 'receipts/flywheel-datasets/v1/' + identity + '/' + receipt['sha256'] + '.json',
         'prior receipt namespace')
    summary = [v for v in prior['metadata'] if v['variant_identity'] == identity]
    need(len(summary) == 1, 'prior metadata summary closure')
    for name in ('content_sha256', 'target_sha256', 'source_target_sha256', 'parent_lineage_sha256', 'partition',
                 'rows', 'features', 'outer_test_rows'):
        need(summary[0][name] == metadata[name], 'prior metadata semantics differ: ' + name)
    need(summary[0].get('sequence')==variant['sequence'] and canonical({k:v for k,v in summary[0].items() if k!='sequence'})==canonical(metadata), 'prior full metadata body differs')
    return {'receipt': receipt, 'objects': objects}


def guard_sql_files(base, directory):
    for name in ('evidence.sqlite', 'evidence.sqlite-wal', 'evidence.sqlite-shm'):
        p = directory / name
        if p.exists() or p.is_symlink():
            base.read(p, 32 * base.MIB)


def source_release(config, manifest):
    host=config['host']
    need(host=='xbabe3' and config['encoder_worker']['sha256']==ENCODER_SHA and
         manifest['packet_id']==PACKET_ID and manifest['total_variants']==COUNT and
         config['encoder_worker']['path']==str(ROOT/'evergreen_kumos_packet.py') and
         config['encoder_manifest']['path']==str(ROOT/'manifest.json') and
         manifest['worker_path']==config['encoder_worker']['path'] and manifest['owned_root']==str(ROOT) and
         manifest['worker_sha256']==ENCODER_SHA and 0<manifest['deadline_unix']-manifest['created_unix']<=1200 and manifest['deadline_unix']<manifest['queue_deadline_unix']<=GLOBAL_DEADLINE and config['deadline_unix']<=manifest['queue_deadline_unix'] and
         set(manifest['hosts'])=={'xbabe3'} and manifest['hosts'][host]['out']==str(ROOT/'runs'/host),
         'fixed reviewed source/manifest/output tuple changed')
    return {'release':'fresh32-xbabe3-full-qualification-sealed','source_sha256':ENCODER_SHA,'out':str(ROOT/'runs'/host)}


def preflight(base, config, deadline):
    base.tick(deadline)
    manifest = json.loads(base.bound(config['encoder_manifest'], cap=LIMIT))
    msha = config['encoder_manifest']['sha256']
    release = source_release(config, manifest)
    worker = load_module(config['encoder_worker']['path'], release['source_sha256'], 'publication_encoder_worker')
    # The generic worker deliberately binds ROOT per manifest; verification does not
    # rerun its expired training/encoding launch guard. Exact tuple is checked above.
    worker.ROOT=ROOT
    need(manifest['schema'] == 'jope.flywheel-evergreen-kumos-packet/v1' and manifest['authority_helper_sha256'] == AUTHORITY_SHA and
         manifest['components'] == DIMS and manifest['approved_for_qualification'] is True and
         manifest['total_variants'] == COUNT and manifest['roster_version'] == 'kumo-s-only-v1' and manifest['teaching_eligible'] is False and
         type(manifest['new_gp_labels']) is int and manifest['new_gp_labels'] == 0, 'encoder manifest closure')
    host = config['host']
    cfg = manifest['hosts'][host]
    need(cfg['out'] == release['out'] and cfg['host'] == host and len(cfg['variants']) == COUNT,
         'source host/root closure')
    for ref in manifest['source_pins']:
        base.bound(ref, cap=LIMIT)
    # Authority is freshly produced on xbabe2 and copied under its content digest;
    # this process does not assume cross-host SSH credentials or renew the proof.
    role = worker.authority(manifest, msha, config['authority'], fresh=True)
    guard_sql_files(base,Path(cfg['out'])/'ofm');guard_sql_files(base,Path(cfg['out']))
    replayed=worker.load_inputs(manifest,cfg);need(len(replayed)==COUNT,'replay support')
    closure=worker.verify_campaign_seal(manifest,cfg,host,msha)
    need(closure['ledger']==config['ledger'] and closure['status_sha256']==config['status_sha256'],'coordinator seal differs')
    phase_anchors={'ofm':worker.stage_seal(Path(cfg['out'])/'ofm',manifest,host,msha,'ofm')}
    need(phase_anchors==config['phase_ledgers'],'phase seal differs')
    rows=worker.join(manifest,cfg,host,msha,Path(cfg['out']))
    need(rows==closure['status']['outputs'],'sealed joins disagree')
    prior = json.loads(base.bound(config['prior_publication'], cap=LIMIT))
    need(config['prior_publication']==manifest['dataset_publication'] and prior['state'] == 'complete' and prior['pipeline_config_sha256']==PIPELINE_SHA and
         prior['schema']=='jope.flywheel-generated-packet-publication-complete/v1' and len(prior['variants'])==COUNT and len(prior['object_receipts'])==4*COUNT, 'reviewed32 dataset custody required')
    by_id = {v['identity']: v for v in cfg['variants']}
    need(len(by_id) == COUNT and len({v['lineage'] for v in by_id.values()}) == COUNT, 'distinct assigned support')
    plans = {}
    for item in replayed:
        identity, metadata = item['identity'], item['metadata']
        original = prior_binding(base, prior, by_id[identity], metadata)
        plans[identity] = {'variant_identity': identity, 'sequence': by_id[identity]['sequence'],
                          'metadata': {'metadata_sha256': by_id[identity]['files']['metadata.json']['sha256'], **{k: metadata[k] for k in ('content_sha256', 'target_sha256',
                          'source_target_sha256', 'parent_lineage_sha256', 'partition', 'rows', 'features',
                          'inner_counts', 'outer_test_rows')}}, 'prior': original, 'objects': {}, 'records': {}}
    for row in rows:
        base.tick(deadline)
        identity, component = row['variant_identity'], row['component']
        need(identity in plans and component in DIMS and component not in plans[identity]['records'],
             'unknown/duplicate vector slot')
        raw = base.bound(row['output'], cap=LIMIT)
        base.numeric_vector(raw, DIMS[component])
        record = json.loads(base.bound(row['receipt'], cap=LIMIT))
        phase = 'ofm'
        need(record['phase'] == phase, 'component qualification phase differs')
        qref = record['qualification']
        qraw = base.bound(qref, Path(cfg['out']) / phase, cap=base.MIB)
        qualification = json.loads(qraw)
        need(qualification['passed'] is True and len(qualification['tests'])==6 and all(v is True for v in qualification['tests'].values()) and canonical(qualification) == qraw,
             'qualified canonical evidence required')
        key = base.PREFIX + 'datasets/flywheel/v1/' + identity + '/encodings/' + component + '.npy'
        need(all(ref['key'] != key for ref in prior['object_receipts']), 'would replace original object')
        plans[identity]['objects'][key] = raw
        plans[identity]['records'][component] = {'record': record, 'record_ref': row['receipt'],
                                                'vector_ref': row['output'],
                                                'qualification': {'ref': {**qref, 'path': str(Path(cfg['out']) / phase / qref['path'])},
                                                                  'body': qualification}}
    for plan in plans.values():
        need(set(plan['records']) == set(DIMS) and len(plan['objects']) == 1, 'KumoS-vector closure')
        need(len(canonical(plan['records'])) <= base.MIB, 'embedded encoder evidence cap')
    need(len(plans) == COUNT and sum(len(b) for p in plans.values() for b in p['objects'].values()) <= LIMIT,
         'bounded32-variant support')
    proof = {'manifest': config['encoder_manifest'], 'source_sha256': release['source_sha256'], 'source_release': release,
             'coordinator_ledger': closure['ledger'], 'coordinator_status_sha256': closure['status_sha256'],
             'phase_ledgers': phase_anchors, 'authority_ref': config['authority'], 'authority': role,
             'prior_publication_sha256': config['prior_publication']['sha256'], 'finalization_custody': None,
             'producer_phase_sealed': True, 'original_coordinator_state': 'complete'}
    base.tick(deadline)
    return sorted(plans.values(), key=lambda p: p['sequence']), proof


def verify_prior_remote(base, client, plan, deadline):
    body = get_verified(base, client, plan['prior']['receipt'], deadline, load=True)
    receipt = json.loads(body)
    need(receipt['variant_identity'] == plan['variant_identity'], 'prior S3 receipt identity')
    want = plan['prior']['objects']
    actual = {ref['key']: ref for ref in receipt['objects']}
    need(len(actual) == len(receipt['objects']) == 3, 'prior S3 receipt objects')
    need(actual == {ref['key']: ref for ref in want.values()}, 'prior S3 receipt bindings differ')
    for ref in want.values():
        get_verified(base, client, ref, deadline)


def progress(out, config, state, done, count, **extra):
    result = {'schema': SCHEMA, 'host': config['host'], 'at': time.time(), 'state': state,
              'expected_variants': COUNT, 'completed_variants': len(done), 'expected_objects': 2*COUNT,
              'verified_objects': count, 'expected_vectors': COUNT, 'new_gp_labels': 0,
              'teaching_eligible': False, 'native_context_admission': 'unqualified',
              'variants': done, **extra}
    p = out / 'status.pending'
    with p.open('xb') as f:
        f.write(canonical(result) + b'\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(p, out / 'status.json')
    print(json.dumps({k: v for k, v in result.items() if k != 'variants'}), flush=True)


def publish(base, config, plans, proof, client, deadline):
    out = Path(config['out'])
    need(not out.exists() and all(not p.is_symlink() for p in (out, *out.parents)), 'fresh nonsymlink publication output')
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    done, count = [], 0
    progress(out, config, 'publishing', done, count)
    try:
        with (out / 'objects.jsonl').open('xb') as journal:
            def record(ref):
                nonlocal count
                journal.write(canonical(ref) + b'\n')
                journal.flush()
                os.fsync(journal.fileno())
                count += 1
            for plan in plans:
                verify_prior_remote(base, client, plan, deadline)
                refs = []
                for key, raw in sorted(plan['objects'].items()):
                    ref = base.put_verified(client, key, raw, deadline)
                    record(ref)
                    refs.append(ref)
                receipt = {'schema': SCHEMA, 'roster_version':'kumo-s-only-v1', 'required_encoders':['kumo_tabular_s'], 'variant_identity': plan['variant_identity'],
                           'sequence': plan['sequence'], 'host': config['host'], **plan['metadata'],
                           'source_proof': proof, 'prior_dataset_publication': plan['prior'],
                           'objects': refs, 'encoder_records': plan['records'],
                           'completed_encoders': list(DIMS), 'missing_encoders': [],
                           'publisher_sha256': config['helper_sha256'], 'rights': 'research',
                           'prepared': True, 'raw_units': False, 'new_gp_labels': 0,
                           'teaching_eligible': False, 'native_context_admission': 'unqualified',
                           'lift_eligible': False,
                           'limits': ['Descendants retain parent lineage and development partition.',
                                      'Prepared source and encoder pretraining caveats remain.',
                                      'Encoders may use task-training y; no GP-inner-fit-only claim.',
                                      'Fresh native GP results and a qualified training snapshot are still required.']}
                raw = canonical(receipt) + b'\n'
                key = base.PREFIX + 'receipts/flywheel-datasets/v1/' + plan['variant_identity'] + '/' + sha(raw) + '.json'
                ref = base.put_verified(client, key, raw, deadline)
                record(ref)
                done.append({'variant_identity': plan['variant_identity'], 'sequence': plan['sequence'],
                             **plan['metadata'], 'receipt': ref, 'completed_encoders': list(DIMS)})
                progress(out, config, 'publishing', done, count)
        need(len(done) == COUNT and count == 2*COUNT, 'publication support incomplete')
        base.tick(deadline)
        progress(out, config, 'complete', done, count, vectors_verified=COUNT, completed_encoders=list(DIMS), missing_encoders=[])
    except BaseException as exc:
        progress(out, config, 'failed', done, count, error=type(exc).__name__ + ': ' + str(exc)[:500])
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-sha', required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    base = load_module(Path(__file__).with_name('pilot_publish_core.py').absolute(), BASE_SHA, 'reviewed_encoder_s3_base')
    config = json.loads(base.bound({'path': args.config, 'sha256': args.config_sha}, cap=LIMIT))
    need(config['schema'] == SCHEMA and config['host']=='xbabe3' and
         socket.gethostname().split('.')[0] == config['host'], 'host/config mismatch')
    need(config['helper_sha256'] == sha(base.read(Path(__file__).absolute(), LIMIT)), 'publisher source drift')
    need(type(config['created_unix']) in (int, float) and type(config['deadline_unix']) in (int, float) and
         math.isfinite(config['created_unix']) and math.isfinite(config['deadline_unix']) and
         0<config['deadline_unix'] - config['created_unix'] <= 600 and config['deadline_unix']<=GLOBAL_DEADLINE and config['created_unix'] <= time.time() + 5,
         'fixed600second publication window')
    need(config['out'] == str(ROOT / 'publication' / config['host']), 'owned output root')
    deadline = config['deadline_unix']
    base.tick(deadline)
    envelope = base.envelope()
    plans, proof = preflight(base, config, deadline)
    print(json.dumps({'state': 'preflight_pass', 'host': config['host'], 'variants': COUNT,
                      'vectors': COUNT, 'objects': 2*COUNT, 'envelope': envelope,
                      'source_proof': proof}, sort_keys=True), flush=True)
    if args.preflight_only:
        return
    import boto3
    from botocore.config import Config
    client = boto3.Session(profile_name='veox-jepson-prod').client('s3', config=Config(
        connect_timeout=5, read_timeout=10, retries={'max_attempts': 0}, max_pool_connections=1))
    publish(base, config, plans, proof, client, deadline)


if __name__ == '__main__':
    main()
