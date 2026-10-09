"""Exact original descriptors with explicit authenticated owned replicas.

No path rewriting in metadata/context identities, no monkeypatching the frozen
reader. Current role authority is freshly read on xbabe2, never inferred from a
copied registry on xbabe3. Exact-version dataset bodies cache immutably per shard.
"""
from dataclasses import asdict
import gzip,io,json,time
from pathlib import Path
from jope_flywheel import evergreen
from jope_flywheel.augmentation import array_sha256
from jope_flywheel.program_io import bound,read,create,need,sha,tick
from jope_flywheel.program_inputs import dataset_object
from jope_flywheel.protocol import Context,canonical,digest,partition

class ReplicaMap:
    def __init__(self, mapping):
        need(mapping['schema']=='jope.flywheel-source-replicas/v1','explicit replica map')
        self.root=Path(mapping['replica_root']); need(self.root.is_absolute(),'replica root absolute')
        self.rows={}
        for row in mapping['entries']:
            src,dest=row['source'],row['replica']; path=Path(dest['path'])
            need(path.is_relative_to(self.root) and path!=self.root,'replica inside owned root')
            need(src['path'] not in self.rows and src['sha256']==dest['sha256'],'duplicate source or mismatched replica SHA')
            need('bytes' not in src or src['bytes']==dest['bytes'],'replica size')
            self.rows[src['path']]=row
    def bound(self, ref, cap=32*1024**2):
        need(ref['path'] in self.rows,'source missing from explicit map')
        row=self.rows[ref['path']]
        need(ref['sha256']==row['source']['sha256'] and
             ('bytes' not in ref or ref['bytes']==row['replica']['bytes']),'source descriptor changed')
        return bound(row['replica'],cap)

from authority_lease import AuthorityLease

class VersionCache:
    def __init__(self, root, client, deadline):
        self.root,self.client,self.deadline=Path(root),client,deadline
        self.root.mkdir(mode=0o700,exist_ok=False)
    def get(self, ref, identity, name):
        tick(self.deadline);key=digest({'ref':ref,'variant':identity,'name':name})
        record=self.root/(key+'.json');payload=self.root/(key+'.bytes')
        if record.exists():
            receipt=json.loads(read(record,4096))
            need(receipt['ref']==ref and receipt['variant']==identity and receipt['name']==name,'version cache identity')
            data=bound({'path':str(payload),'sha256':ref['sha256'],'bytes':ref['bytes']},8*1024**2)
            tick(self.deadline);return data
        need(not payload.exists(),'incomplete version cache; no uncertain retry')
        data=dataset_object(self.client,ref,variant=identity,name=name,deadline=self.deadline)
        create(payload,data);create(record,canonical({'ref':ref,'variant':identity,'name':name,'verified_at':time.time()}))
        return data

def staged_parent(parent,pinned):
    need(parent['schema']=='jope.flywheel-prepared-parent/v1' and parent['parent_admission']=='prepared-research-transforms-only','qualified research parent required')
    need(parent['rights']=='research' and parent['raw_units']is False and parent['prepared']is True,'prepared research declaration required')
    need(parent['role_authority']['eligible_target_all_views']is True and parent['role_authority']['excluded_context_or_view']is False,'current eligible role required')
    need(parent['partition']==partition(parent['parent_lineage_sha256'],parent['split_namespace_sha256']),'lineage partition mismatch')
    policy=parent['allowed_transforms']
    need(policy['feature_gaussian_noise']is False and policy['resample_rows']is False,'unsupported source policy')
    need(policy['feature_drop_max_fraction']>=.1 and policy['nuisance_max_fraction']>=.1 and policy['nuisance_max_columns']>=8
         and policy['max_output_features']>=512 and policy['max_train_rows']>=1999 and policy['numeric_target_noise']is True,'parent transform limits too restrictive')
    pinned(parent['context'])
    if parent.get('preparation_receipt') is not None:pinned(parent['preparation_receipt'])
    zipped=pinned(parent['encoder'])
    with gzip.GzipFile(fileobj=io.BytesIO(zipped))as f:raw=f.read(evergreen.MAX_RAW+1)
    need(len(raw)<=evergreen.MAX_RAW and sha(raw)==parent['raw_payload_sha256'],'raw source hash mismatch')
    x,y,folds=evergreen.decode(raw)
    need((len(y),x.shape[1])==(parent['rows'],parent['features']),'source shape mismatch')
    need(array_sha256(x)==parent['feature_matrix_sha256'] and array_sha256(y)==parent['target_sha256'],'source feature/target mismatch')
    need(sha(folds.tobytes())==parent['inner_labels_sha256'],'source folds mismatch')
    need(not any(__import__("numpy").array_equal(x[:,j],y) for j in range(x.shape[1])),'exact target-copy feature refused')
    return x,y,folds


class StagedInputs:
    def __init__(self, config, registry, native, client, deadline, cache=None):
        self.replicas = ReplicaMap(json.loads(bound(config["staging_map"])))
        self.bound = self.replicas.bound
        self.lease = AuthorityLease(config)
        self.cache = cache if cache is not None else VersionCache(Path(config["owned_root"])/"data/version_cache", client, deadline)
        self.config, self.registry, self.native = config, registry, native
        self.client, self.deadline = client, deadline
        self.generator = json.loads(self.bound(config['generator_config']))
        need(registry['schema'] == 'jope.flywheel-program-inputs/v1' and
             registry['generator_config_sha256'] == config['generator_config']['sha256'], 'registered generator identity')
        need(registry['research_only'] is True and registry['teaching_eligible'] is False, 'research authorization scope')
        rows = registry['inputs']; need(len(rows) == 128, 'exact fresh128 registered input count')
        self.rows = {r['identity']: r for r in rows}
        need(len(self.rows) == len(rows) and set(self.rows) == {x['identity'] for x in config['selection']}, 'exact unique selected128 variants')
        parents = [json.loads(line) for line in self.bound(self.generator['parent_registry']).splitlines()]
        self.parents = {p['parent_lineage_sha256']: p for p in parents}
        need(len(parents) == len(self.parents) == 1068, 'full qualified parent registry')

    def describe(self, identity):
        tick(self.deadline); need(identity in self.rows, 'unregistered program input')
        row = self.rows[identity]; parent = self.parents[row['parent']]
        # Every use rechecks source closure, role freshness and protected lineage membership.
        for ref in self.generator['source_pins']: self.bound(ref)
        self.bound(self.config['generator_config']); self.bound(self.generator['parent_registry'])
        self.lease.check(identity, parent)
        files = {}
        for name in ('encoder.jtt1.gz', 'metadata.json', 'generation.json'):
            ref = row['files'][name]
            need(Path(ref['path']) == Path(self.generator['out'])/'variants'/identity/name, 'registered input path')
            files[name] = self.bound(ref, 8*1024**2)
            need(self.cache.get(row['s3'][name], identity, name) == files[name],
                 'published dataset differs from registered local source')
        publication = json.loads(self.cache.get(row['s3']['receipt'], identity, 'receipt'))
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
            if parent.get(name) is not None: self.bound(parent[name])
        px, py, pf = staged_parent(parent, self.bound)
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
        # The old dataset schema is retained, but the current growth authority and
        # exact Kumo-only publication closure are mandatory new provenance gates.
        from receipt_gate import validate_current_receipts
        validate_current_receipts(self.config, row, meta, generation, publication,
                                  self.cache, self.deadline)
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
