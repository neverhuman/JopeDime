"""Registered prepared research inputs; replay and current role checks on every use."""
from dataclasses import asdict
import gzip
import io
import json
from pathlib import Path
from . import evergreen
from .augmentation import array_sha256
from .program_io import bound, need, sha, tick
from .protocol import Context, canonical, digest, sha256

BUCKET = 'veox-jopedime-learning-use2'
PREFIX = 'experiments/gp-policy-dl/v2-trainonly/'


def current_roles(config):
    docs = {r['path']: json.loads(bound(r)) for r in config['authority_pins']}
    paths = [p for p in docs if '/gp-target-one-holdout-after-layers-' in p]
    need(len(paths) == 1, 'one current target authority')
    path = Path(paths[0]); target = docs[str(path)]
    need(max(path.parent.glob('gp-target-one-holdout-after-layers-v*.json'),
             key=lambda p: p.stat().st_mtime) == path, 'newer target authority exists')
    rp = Path(target['registry']['path']); registry = docs[str(rp)]
    need(max(rp.parent.glob('TRAINING-EXCLUSION-LAYERS-v*.json'),
             key=lambda p: p.stat().st_mtime) == rp, 'newer exclusions exist')
    cids, views = set(), set()
    for ref in registry['layers']:
        item = docs[ref['path']]
        need(sha(canonical(item)) == sha(canonical(json.loads(bound(ref)))), 'layer authority mismatch')
        if item.get('schema') == 'jope.dltrain-training-exclusion-layer/v1':
            for row in item['entries']:
                cids.add(row['context_sha256']); views.update(row.get('view_ids', []))
    return set(target['views']), cids, views


def _dataset_version(client, ref, deadline):
    """First read verifies exact response version, bytes and hash; never absence."""
    tick(deadline)
    result = client.get_object(Bucket=BUCKET, Key=ref['key'], VersionId=ref['version_id'])
    need(result.get('VersionId') == ref['version_id'] and result.get('ContentLength') == ref['bytes'], 'dataset version/length')
    data = bytearray()
    with result['Body'] as stream:
        while True:
            tick(deadline); block = stream.read(min(1024**2, ref['bytes']-len(data)+1))
            if not block: break
            data.extend(block); need(len(data) <= ref['bytes'], 'bounded dataset read')
    tick(deadline)
    need(len(data) == ref['bytes'] and sha(data) == ref['sha256'], 'dataset exact version hash')
    return bytes(data), ref['version_id']


def dataset_object(client, ref, *, variant, name, deadline, version_cache=None):
    tick(deadline); sha256(ref['sha256'])
    expected = PREFIX + ('receipts/flywheel-datasets/v1/' if name == 'receipt' else 'datasets/flywheel/v1/') + variant + '/'
    expected += ref['sha256'] + '.json' if name == 'receipt' else name
    need(ref['bucket'] == BUCKET and ref['key'] == expected, 'dataset object namespace')
    need(type(ref['bytes']) is int and 0 < ref['bytes'] <= 8*1024**2, 'dataset object cap')
    need(type(ref['version_id']) is str and ref['version_id'] not in ('', 'null'), 'exact dataset version')
    # Namespace and expected context path are checked before every cache hit.
    body, version = (_dataset_version(client, ref, deadline) if version_cache is None
                     else version_cache.get_version(ref))
    tick(deadline)
    need(version == ref['version_id'] and len(body) == ref['bytes'] and sha(body) == ref['sha256'],
         'dataset cached exact version hash')
    return body


class RegisteredInputs:
    def __init__(self, config, registry, native, client, deadline):
        self.config, self.registry, self.native = config, registry, native
        self.client, self.deadline = client, deadline
        self.version_cache = None
        self.generator = json.loads(bound(config['generator_config']))
        need(registry['schema'] == 'jope.flywheel-program-inputs/v1' and
             registry['generator_config_sha256'] == config['generator_config']['sha256'], 'registered generator identity')
        need(registry['research_only'] is True and registry['teaching_eligible'] is False, 'research authorization scope')
        rows = registry['inputs']; need(0 < len(rows) <= 128, 'registered input count')
        self.rows = {r['identity']: r for r in rows}
        need(len(self.rows) == len(rows), 'unique registered variants')
        parents = [json.loads(line) for line in bound(self.generator['parent_registry']).splitlines()]
        self.parents = {p['parent_lineage_sha256']: p for p in parents}
        need(len(parents) == len(self.parents) == 1068, 'full qualified parent registry')

    def enable_version_cache(self, max_bytes):
        """Fresh-runtime opt-in; roles, source binding and replay stay uncached."""
        need(self.version_cache is None, 'input cache already configured')
        from pipelines.hpc_verified_v4.verified_cache import VerifiedVersionCache
        self.version_cache = VerifiedVersionCache(
            lambda ref: _dataset_version(self.client, ref, self.deadline),
            check=lambda: tick(self.deadline), max_bytes=max_bytes, max_entries=1024)

    def cache_stats(self):
        return ({'enabled': False} if self.version_cache is None
                else {'enabled': True, **self.version_cache.snapshot()})

    def dataset_read(self, ref, *, variant, name):
        return dataset_object(self.client, ref, variant=variant, name=name,
                              deadline=self.deadline, version_cache=self.version_cache)

    def describe(self, identity):
        tick(self.deadline); need(identity in self.rows, 'unregistered program input')
        row = self.rows[identity]; parent = self.parents[row['parent']]
        # Every use rechecks source closure, role freshness and protected lineage membership.
        for ref in self.generator['source_pins']: bound(ref)
        bound(self.config['generator_config']); bound(self.generator['parent_registry'])
        eligible, cids, views = current_roles(self.generator)
        need(parent['view_id'] in eligible and parent['view_id'] not in views and
             not (set(parent['retained_lineage_context_sha256']) & cids), 'current role excludes parent lineage')
        files = {}
        for name in ('encoder.jtt1.gz', 'metadata.json', 'generation.json'):
            ref = row['files'][name]
            need(Path(ref['path']) == Path(self.generator['out'])/'variants'/identity/name, 'registered input path')
            files[name] = bound(ref, 8*1024**2)
            need(self.dataset_read(row['s3'][name], variant=identity, name=name) == files[name],
                 'published dataset differs from registered local source')
        publication = json.loads(self.dataset_read(row['s3']['receipt'], variant=identity, name='receipt'))
        meta = json.loads(files['metadata.json']); generation = json.loads(files['generation.json'])
        need(canonical(meta) == files['metadata.json'] and meta['parent'] == parent and
             meta['parent_record_sha256'] == digest(parent), 'exact parent/transformation metadata')
        need(generation['schema'] == evergreen.SCHEMA and generation['config_sha256'] == self.config['generator_config']['sha256']
             and generation['sequence'] == row['sequence'] and generation['variant_identity'] == identity
             and generation['parent_lineage_sha256'] == row['parent'], 'generation receipt identity')
        need(generation['encoder_sha256'] == sha(files['encoder.jtt1.gz']) and
             generation['metadata_sha256'] == row['metadata_sha'] == sha(files['metadata.json']), 'generation artifact hashes')
        with gzip.GzipFile(fileobj=io.BytesIO(files['encoder.jtt1.gz'])) as stream: raw = stream.read(evergreen.MAX_RAW+1)
        need(len(raw) <= evergreen.MAX_RAW and sha(raw) == row['content_sha'] == generation['raw_sha256'] == meta['content_sha256'], 'generated raw hash')
        x, y, folds = evergreen.decode(raw)
        # Pure replay includes fit-only transform/noise calibration and exact target identity.
        for name in ('encoder', 'context', 'preparation_receipt'):
            if parent.get(name) is not None: bound(parent[name])
        px, py, pf = evergreen.source(parent)
        variant, replay, expected = evergreen.generate(parent, row['ordinal'], px, py, pf)
        need(variant == identity and replay == raw and canonical(expected) == files['metadata.json'], 'deterministic input replay')
        need(meta['partition'] == parent['partition'] and meta['rights'] == 'research' and meta['prepared'] is True and
             meta['raw_units'] is False and meta['outer_test_rows'] == 0 and meta['teaching_eligible'] is False and
             meta['inherited_gp_rewards'] is False and meta['gp_measurements'] == 0, 'research input is not teaching admission')
        need(publication['schema'] == 'jope.flywheel-evergreen-dataset-publication/v1' and publication['variant_identity'] == identity
             and publication['content_sha256'] == sha(raw) and publication['metadata_sha256'] == sha(files['metadata.json'])
             and publication['parent_lineage_sha256'] == row['parent'] and publication['teaching_eligible'] is False,
             'publication receipt binding')
        for name in files:
            need(any(all(ref.get(k) == row['s3'][name].get(k) for k in ('bucket','key','version_id','sha256','bytes'))
                     for ref in publication['objects']), 'publication input object closure')
        protocol_sha = sha(self.native.canonical_json(self.config['protocol_descriptor']))
        vocabulary_sha = sha(self.native.canonical_json(self.config['vocabulary_descriptor']))
        cut = {'schema':'jope.flywheel-prepared-two-fold-cut/v1', 'variant_identity':identity,
               'metadata_sha256':sha(files['metadata.json']), 'dataset_sha256':sha(raw),
               'target_sha256':array_sha256(y), 'split_sha256':sha(folds.tobytes()), 'protocol_sha256':protocol_sha}
        context = Context('regression',sha(raw),cut['target_sha256'],sha(self.native.canonical_json(cut)),
                          self.config['worker']['sha256'],vocabulary_sha,protocol_sha,cut['split_sha256'],
                          'rmse','prepared_synthetic_target_units','minimize')
        descriptor = {'schema':'jope.flywheel-program-input/v1','variant_identity':identity,'context':asdict(context),
            'parent_lineage_sha256':row['parent'],'partition':parent['partition'],'metadata_sha256':row['metadata_sha'],
            'parent_record_sha256':digest(parent),'preparation_receipt':parent.get('preparation_receipt'),
            'rows':len(y),'features':x.shape[1],'inner_counts':meta['inner_counts'],'outer_test_rows':0,
            'dataset_s3':row['s3'],'teaching_eligible':False,'research_only':True}
        tick(self.deadline)
        return context, raw, descriptor
