"""KumoS-only frozen, read-only flywheel -> PolicyLM adapter (no DB admission or training).

``load_snapshot(spec_bytes, expected_sha256, store, *, registry_bytes,
authority_bytes, authority_sha256, grammar_bytes, now=None)`` reads exact S3 versions through
``store.get_version(ref) -> (bytes, VersionId)``. The same store must expose the
pinned native ``nativecanonical_json`` callback required by program_gate. It
returns LoadedSnapshot(data, receipt); data is the unchanged dltrain.TrainData,
or None when there is no optimizer universe. Only state ``loaded`` confirms an
actual TrainData load; group and feasibility eligibility are reported separately.

The externally reviewed spec pins publisher, encoder, authority, grammar and
consumer sources. Authority bytes must be freshly produced by the pinned live
role checker; this module cannot independently determine remote latest pointers.
Publisher receipts attest execution/seals; vector bytes are independently read
and checked here. This adapter does not rerun encoder qualification, execute GP,
publish objects, release queue capacity, or claim unseen/pretraining-clean data.

Spec keys are deliberately closed. Each task names a full64 variant identity,
an actual protocol.Context, dataset/encoder publication refs, and all program
result refs. Context identity is never a fabricated legacy native-context ID.
Unknown/missing claimed artifacts fail the whole load. Tokenizer refusals are
all retained in the receipt; missing programs are never fabricated as failures.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import gzip
import hashlib
import importlib
import io
import json
import math
from pathlib import Path
import struct
import time

import numpy as np

from .augmentation import array_sha256
from .protocol import Context, canonical, digest, partition, sha256
from .program_gate import ARTIFACTS, object_bytes, verified_measurement

SCHEMA = 'flywheel-policy-kumos-tranche/v1'
RECEIPT_SCHEMA = 'jope.flywheel-policy-kumos-loaded/v1'
COMPONENTS = {'kumo_tabular_s': 2051}
WEIGHT_SHA256 = 'fc672f536501d64356036050ac2acea7e46927f7b9ad38366abe048f88e7d423'
MODEL_REVISION = '3c3e10bbdb590ace29e7026847f92db3c603096d'
ROSTER_AUDIT_SHA256 = '0b5aaba0080c34d7ea5d20b21bff8eda8e364f0e11fbdb304347305f4ed5e564'
CONSUMER_MODULES = ('dltrain', 'dltrain.common', 'dltrain.labels', 'dltrain.sampler',
                    'dltrain.dsl_tokenizer', 'dltrain.automaton', 'dltrain.oracle')
ADAPTER_MODULES = ('jope_flywheel', 'jope_flywheel.protocol', 'jope_flywheel.augmentation',
                   'jope_flywheel.program_gate', 'jope_flywheel.score_join', 'jope_flywheel.batch_stream')
SPEC_KEYS = {'schema', 'tasks', 'components', 'consumer_source_sha256', 'vocab_sha256',
             'grammar_sha256', 'max_seq', 'parent_registry_sha256', 'generator_config_sha256',
             'authority_helper_sha256', 'dataset_publisher_sha256',
             'encoder_publisher_sha256', 'encoder_releases', 'shared_seed_program_sha256', 'evaluation_contract',
             'adapter_dependency_sha256', 'roster_audit_sha256'}
MAX_OBJECT = 32 * 1024**2
MAX_READ = 768 * 1024**2


def need(ok, message):
    if not ok:
        raise ValueError(message)


def byte_sha(body):
    return hashlib.sha256(body).hexdigest()


def document(body, limit=MAX_OBJECT):
    need(type(body) is bytes and len(body) <= limit, 'bounded JSON bytes required')
    def pairs(items):
        out = {}
        for key, value in items:
            need(key not in out, 'duplicate JSON key')
            out[key] = value
        return out
    def invalid(value):
        raise ValueError('nonfinite JSON number: ' + value)
    obj = json.loads(body, object_pairs_hook=pairs, parse_constant=invalid)
    canonical(obj)  # overflow such as 1e999 is also refused
    return obj


def normalized_ref(ref):
    need(type(ref) is dict and set(ref) in (
        {'bucket', 'key', 'version_id', 'sha256', 'bytes'},
        {'bucket', 'key', 'version_id', 'sha256', 'bytes', 'state', 'verification'}),
        'exact versioned object reference required')
    if 'verification' in ref:
        need(ref['verification'] == 'exact VersionId GET + SHA256', 'unverified custody reference')
    return {k: ref[k] for k in ('bucket', 'key', 'version_id', 'sha256', 'bytes')}


class _Reader:
    """Bound reads before dispatch, deduplicate exact versions, forbid alias drift."""
    def __init__(self, store):
        self.store, self.cache, self.bindings, self.total = store, {}, {}, 0
        self.nativecanonical_json = getattr(store, 'nativecanonical_json', None)

    def get_version(self, ref):
        key = (ref['bucket'], ref['key'], ref['version_id'])
        bound = (ref['sha256'], ref['bytes'])
        need(self.bindings.setdefault(key, bound) == bound, 'same object version has conflicting pins')
        if key not in self.cache:
            size = ref['bytes']
            need(type(size) is int and 0 < size <= MAX_OBJECT and self.total + size <= MAX_READ,
                 'snapshot read budget exceeded')
            body, version = self.store.get_version(ref)
            need(type(body) is bytes and len(body) == size and version == key[2]
                 and byte_sha(body) == ref['sha256'], 'object version/bytes/SHA mismatch')
            self.cache[key] = body
            self.total += size
        return self.cache[key], key[2]

    def read(self, ref):
        return object_bytes(self, normalized_ref(ref))


def decode_input(raw):
    """Decode a bounded stripped JTT1 and return y/folds without any outer rows."""
    need(type(raw) is bytes and 17 <= len(raw) <= 8 * 1024**2 and raw[:4] == b'JTT1', 'bounded JTT1 required')
    n, nt, p, k = struct.unpack('<IIIB', raw[4:17])
    need(50 <= n < 2000 and nt == 0 and 1 <= p <= 512 and k == 2
         and len(raw) == 17 + n + 8*n*(p+1), 'stripped regression input shape')
    folds = np.frombuffer(raw[17:17+n], dtype=np.uint8)
    arr = np.frombuffer(raw[17+n:], dtype='<f8').reshape(n, p+1)
    need(np.isfinite(arr).all() and set(folds.tolist()) == {0, 1}
         and min(np.bincount(folds)) >= 25, 'finite input / minimum inner-fold support')
    return arr[:, -1], folds, p


def mean_baselines(raw, evaluation):
    """Verify native mean(opposite train fold) -> this validation fold RMSE."""
    y, folds, _ = decode_input(raw)
    rows = evaluation.get('inner_cv')
    need(type(rows) is list and len(rows) == 2, 'two baseline folds required')
    values = []
    for i, row in enumerate(rows):
        need(row.get('index') == i and row.get('kind') == 'inner_cv', 'ordered baseline folds required')
        train_y, test_y = y[folds != i], y[folds == i]
        mean = math.fsum(float(v) for v in train_y) / len(train_y)
        expected = math.sqrt(math.fsum((float(v)-mean)**2 for v in test_y) / len(test_y))
        actual = row.get('baseline_metrics', {}).get('rmse')
        need(type(actual) in (int, float) and math.isfinite(actual) and actual >= 0
             and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12),
             'native baseline differs from opposite-fold mean reconstruction')
        values.append(float(actual))
    return values


def _consumer(pins, grammar_bytes, spec):
    need(set(pins) == set(CONSUMER_MODULES), 'exact consumer source closure required')
    modules = {}
    for name in CONSUMER_MODULES:
        mod = importlib.import_module(name)
        path = Path(mod.__file__)
        need(path.is_file() and byte_sha(path.read_bytes()) == sha256(pins[name]), 'consumer source drift: ' + name)
        modules[name] = mod
    tokenizer, automaton = modules['dltrain.dsl_tokenizer'], modules['dltrain.automaton']
    vocab = tokenizer.Vocab.from_names({})
    need(vocab.sha256 == spec['vocab_sha256'], 'base vocabulary identity mismatch')
    need(byte_sha(grammar_bytes) == spec['grammar_sha256'], 'grammar hash mismatch')
    gdoc = document(grammar_bytes)
    expected_sig = automaton.build_signatures([])
    expected = {'schema': 'jope.flywheel-policy-grammar/v1', 'mode': 'SYNTAX',
                'vocab_sha256': vocab.sha256, 'max_seq': spec['max_seq'], 'signatures': expected_sig,
                'grammar': automaton.build_grammar(expected_sig, vocab, spec['max_seq'])}
    need(gdoc == expected, 'grammar must be frozen data-independent SYNTAX grammar')
    return modules, vocab, automaton.Grammar(gdoc['grammar'], gdoc['signatures'], vocab)


def _authority(spec, raw, parents, now, expected_authority_sha256):
    need(byte_sha(raw) == sha256(expected_authority_sha256), 'fresh role proof hash mismatch')
    proof = document(raw)
    at = proof.get('checked_at_unix')
    need(type(at) in (int, float) and math.isfinite(at) and 0 <= now-at < 120, 'stale/future role proof')
    need(proof.get('schema') == 'jope.flywheel-evergreen-encoding-authority/v1'
         and proof.get('state') == 'passed' and proof.get('host') == 'xbabe2'
         and proof.get('authority_helper_sha256') == spec['authority_helper_sha256']
         and proof.get('parent_registry_sha256') == spec['parent_registry_sha256']
         and proof.get('generator_config_sha256') == spec['generator_config_sha256'], 'role authority identity')
    by_id = {}
    for row in proof['support']:
        identity = sha256(row['variant_identity'])
        need(identity not in by_id, 'duplicate authority support')
        by_id[identity] = row
    for task in spec['tasks']:
        row = by_id.get(task['variant_identity'])
        need(row is not None, 'missing current-role support')
        parent = parents.get(row['parent_lineage_sha256'])
        need(parent is not None and row['parent_record_sha256'] == digest(parent)
             and row['parent_context_sha256'] == parent['parent_context_sha256']
             and row['view_id'] == parent['view_id'] and row['partition'] == parent['partition']
             and row['retained_contexts_checked'] == len(parent['retained_lineage_context_sha256'])
             and row['eligible_view'] is True and row['excluded_contexts_or_views'] is False,
             'current parent/role/exclusion proof mismatch')
        need(parent.get('protected_role') in (None, 'train', 'validation')
             and parent.get('protected_role', parent['partition']) == parent['partition'], 'protected parent role conflicts')
    return by_id


def _objects_by_suffix(refs, identity, names):
    need(type(refs) is list and len(refs) == len(names), 'exact object closure required')
    result = {}
    for ref in refs:
        nref = normalized_ref(ref)
        candidates = [name for name in names if nref['key'].endswith('/datasets/flywheel/v1/'+identity+'/'+name)]
        need(len(candidates) == 1 and candidates[0] not in result, 'unexpected/duplicate object namespace')
        result[candidates[0]] = ref
    need(set(result) == set(names), 'missing object')
    return result


def _dataset(reader, task, spec, parents, support):
    identity = sha256(task['variant_identity'])
    pub = document(reader.read(task['dataset_publication']))
    need(pub['schema'] == 'jope.flywheel-evergreen-dataset-publication/v1'
         and pub['variant_identity'] == identity and pub['generator_config_sha256'] == spec['generator_config_sha256']
         and pub['publication_helper_sha256'] == spec['dataset_publisher_sha256']
         and pub['parent_registry']['sha256'] == spec['parent_registry_sha256'], 'dataset publication identity')
    refs = _objects_by_suffix(pub['objects'], identity, ('encoder.jtt1.gz', 'metadata.json', 'generation.json'))
    metadata_raw, generation_raw = reader.read(refs['metadata.json']), reader.read(refs['generation.json'])
    meta, gen = document(metadata_raw), document(generation_raw)
    need(gen['schema'] == 'jope.flywheel-evergreen/v1', 'versioned generation receipt required')
    with gzip.GzipFile(fileobj=io.BytesIO(reader.read(refs['encoder.jtt1.gz']))) as stream:
        raw = stream.read(8*1024**2 + 1)
    y, folds, p = decode_input(raw)
    parent = parents[support['parent_lineage_sha256']]
    need(meta['schema'] == 'jope.flywheel-dataset/v1' and meta['variant_identity'] == identity
         and meta['parent'] == parent and meta['parent_record_sha256'] == digest(parent)
         and meta['parent_lineage_sha256'] == parent['parent_lineage_sha256'] == pub['parent_lineage_sha256']
         and meta['partition'] == parent['partition'] == partition(parent['parent_lineage_sha256'], parent['split_namespace_sha256']),
         'dataset parent lineage/partition binding')
    need(meta['content_sha256'] == pub['content_sha256'] == gen['raw_sha256'] == byte_sha(raw)
         and pub['metadata_sha256'] == gen['metadata_sha256'] == byte_sha(metadata_raw)
         and gen['variant_identity'] == identity and gen['encoder_sha256'] == refs['encoder.jtt1.gz']['sha256']
         and gen['config_sha256'] == spec['generator_config_sha256'], 'dataset content/generation binding')
    need(meta['target_sha256'] == array_sha256(y) and meta['split_sha256'] == byte_sha(folds.tobytes())
         and meta['source_target_sha256'] == parent['target_sha256'] and meta['rows'] == len(y)
         and meta['features'] == p and meta['inner_counts'] == np.bincount(folds).tolist()
         and meta['outer_test_rows'] == 0 and meta['prepared'] is True and meta['raw_units'] is False
         and meta['inherited_gp_rewards'] is False and meta['teaching_eligible'] is False,
         'dataset target/fold/research scope')
    need(pub['qualification']['deterministic_replay_exact'] is True, 'dataset replay qualification required')
    ctx = Context(**task['context'])
    contract = spec['evaluation_contract']
    need(set(contract) == {'evaluator_sha256', 'vocabulary_sha256', 'protocol_sha256', 'units'}
         and all(getattr(ctx, k) == v for k, v in contract.items()), 'evaluation contract differs')
    cut = {'schema': 'jope.flywheel-prepared-two-fold-cut/v1', 'variant_identity': identity,
           'metadata_sha256': byte_sha(metadata_raw), 'dataset_sha256': byte_sha(raw),
           'target_sha256': meta['target_sha256'], 'split_sha256': meta['split_sha256'],
           'protocol_sha256': ctx.protocol_sha256}
    need(callable(reader.nativecanonical_json), 'native canonical JSON callback required')
    cut_bytes = reader.nativecanonical_json(cut)
    need(type(cut_bytes) is bytes, 'native canonicalizer must return bytes')
    need(ctx.dataset_sha256 == byte_sha(raw) and ctx.target_sha256 == meta['target_sha256']
         and ctx.split_sha256 == meta['split_sha256'] and ctx.cut_id == byte_sha(cut_bytes),
         'new flywheel context input/target/cut identity')
    need(ctx.task == 'regression' and ctx.metric == 'rmse' and ctx.direction == 'minimize', 'regression RMSE only')
    return ctx, meta, raw, refs


def component_output_binding(record, entry, proof, identity, component):
    """Resolve the published phase-relative output against its exact source root."""
    from pathlib import PurePosixPath
    relative = 'variants/'+identity+'/'+component+'.npy'
    output, vector = record['output'], entry['vector_ref']
    release = proof['source_release']
    root = PurePosixPath(release['out'])
    need(root.is_absolute() and '..' not in root.parts
         and release['source_sha256'] == proof['source_sha256'], 'component source release root')
    need(record['phase'] == 'ofm' and output['path'] == relative,
         'component output must be exact phase-relative variant path')
    need(vector['path'] == str(root/'ofm'/relative)
         and all(output[k] == vector[k] for k in ('sha256', 'bytes')),
         'component resolved output path/hash/bytes mismatch')
    return True


def _encodings(reader, task, spec, meta, dataset_refs):
    pub = document(reader.read(task['encoder_publication']))
    identity = task['variant_identity']
    need(pub['schema'] == 'jope.flywheel-evergreen-encoder-publication/v1'
         and pub['variant_identity'] == identity and pub['publisher_sha256'] == spec['encoder_publisher_sha256']
         and 'kumo_tabular_s' in pub['completed_encoders'] and 'kumo_tabular_s' not in pub['missing_encoders']
         and pub['teaching_eligible'] is False and pub['lift_eligible'] is False, 'KumoS-only publication closure')
    need(normalized_ref(pub['prior_dataset_publication']['receipt']) == normalized_ref(task['dataset_publication']),
         'encoder publication refers to another dataset receipt')
    for name, ref in dataset_refs.items():
        need(normalized_ref(pub['prior_dataset_publication']['objects'][name]) == normalized_ref(ref), 'prior dataset objects differ')
    for key in ('content_sha256', 'target_sha256', 'source_target_sha256', 'parent_lineage_sha256', 'partition',
                'rows', 'features', 'inner_counts', 'outer_test_rows'):
        need(pub[key] == meta[key], 'encoder/dataset identity differs: '+key)
    proof = pub['source_proof']
    release = {'source_sha256': proof['source_sha256'], 'manifest_sha256': proof['manifest']['sha256']}
    need(release in spec['encoder_releases'], 'unreviewed encoder source/manifest')
    need('ofm' in proof['phase_ledgers'], 'qualified OFM phase seal required')
    for anchor in [proof['coordinator_ledger'], proof['phase_ledgers']['ofm']]:
        need(anchor['schema'] == 'jope.flywheel-evidence/v1' and anchor['event_count'] > 0
             and anchor['runs'] and all(r['sealed'] is True for r in anchor['runs'].values()), 'unsealed encoder source')
        sha256(anchor['head_sha256'])
    selected = [r for r in pub['objects'] if r['key'].endswith('/encodings/kumo_tabular_s.npy')]
    refs = _objects_by_suffix(selected, identity, ('encodings/kumo_tabular_s.npy',))
    need('kumo_tabular_s' in pub['encoder_records'], 'KumoS record required')
    vectors, loaded = [], {}
    for name, dim in COMPONENTS.items():
        entry = pub['encoder_records'][name]
        record = entry['record']
        ref = refs['encodings/'+name+'.npy']
        weight = record['producer']['weight']
        need(weight['repo'] == 'nvidia/Kumo-Tabular' and weight['revision'] == MODEL_REVISION
             and weight['file'] == 'small/regressor.pt' and weight['sha256'] == WEIGHT_SHA256,
             'reviewed KumoS weight/revision required')
        record_bytes = canonical(record)
        need(byte_sha(record_bytes) == entry['record_ref']['sha256']
             and len(record_bytes) == entry['record_ref']['bytes'], 'embedded source record hash mismatch')
        need(record['manifest_sha256'] == release['manifest_sha256'] and record['variant_identity'] == identity
             and record['component'] == name and record['dimension'] == dim and record['input_raw_sha256'] == meta['content_sha256']
             and record['target_sha256'] == meta['target_sha256'] and record['partition'] == meta['partition']
             and record['parent_lineage_sha256'] == meta['parent_lineage_sha256']
             and record['source_metadata']['sha256'] == dataset_refs['metadata.json']['sha256']
             and record['source_input']['sha256'] == dataset_refs['encoder.jtt1.gz']['sha256']
             and component_output_binding(record, entry, proof, identity, name) and record['output']['sha256'] == ref['sha256']
             and record['output']['bytes'] == ref['bytes'], 'component source/input/vector identity')
        qualification = entry['qualification']
        qbytes = canonical(qualification['body'])
        need(qualification['body'].get('passed') is True
             and byte_sha(qbytes) == qualification['ref']['sha256'] == record['qualification']['sha256']
             and len(qbytes) == qualification['ref']['bytes'] == record['qualification']['bytes'],
             'encoder qualification body/ref mismatch')
        body = reader.read(ref)
        stream = io.BytesIO(body)
        version = np.lib.format.read_magic(stream)
        need(version in ((1, 0), (2, 0)), 'bounded NPY header version required')
        shape, fortran, dtype = (np.lib.format.read_array_header_1_0(stream) if version == (1, 0)
                                  else np.lib.format.read_array_header_2_0(stream))
        need(shape == (dim,) and dtype == np.dtype('<f4') and len(body)-stream.tell() == 4*dim,
             'NPY header/payload dimension mismatch')
        z = np.frombuffer(body, dtype='<f4', offset=stream.tell())
        need(isinstance(z, np.ndarray) and z.dtype == np.dtype('<f4') and z.shape == (dim,)
             and np.isfinite(z).all(), 'full finite float32 component required: '+name)
        vectors.append(z)
        loaded[name] = {'dimension': dim, 'object': normalized_ref(ref), 'qualification': record['qualification'],
                        'record_sha256': digest(record)}
    return np.concatenate(vectors), loaded


@dataclass(frozen=True)
class LoadedSnapshot:
    data: object | None
    receipt: dict


def load_snapshot(spec_bytes, expected_sha256, store, *, registry_bytes, authority_bytes, authority_sha256, grammar_bytes, now=None):
    """Validate immutable membership, then run the real CPU labels/sampler loader.

    No live CLI integration is implicit. The caller must pin the returned spec,
    source closure, receipt and TrainData hash in its own new trainer version.
    Recheck role authority at execution/admission; a saved receipt is historical.
    """
    need(byte_sha(spec_bytes) == sha256(expected_sha256), 'snapshot spec hash mismatch')
    spec = document(spec_bytes)
    need(set(spec) == SPEC_KEYS and spec['schema'] == SCHEMA, 'closed versioned snapshot schema required')
    need(set(spec['adapter_dependency_sha256']) == set(ADAPTER_MODULES), 'adapter dependency closure required')
    for name in ADAPTER_MODULES:
        module = importlib.import_module(name)
        need(byte_sha(Path(module.__file__).read_bytes()) == sha256(spec['adapter_dependency_sha256'][name]),
             'adapter dependency source drift: '+name)
    need(sha256(spec['roster_audit_sha256']) == ROSTER_AUDIT_SHA256, 'reviewed roster audit required')
    need(spec['components'] == COMPONENTS and type(spec['max_seq']) is int and 2 <= spec['max_seq'] <= 2048,
         'fixed KumoS-only layout / bounded maximum sequence required')
    tasks = spec['tasks']
    need(type(tasks) is list and 0 < len(tasks) <= 128 and len({t['variant_identity'] for t in tasks}) == len(tasks),
         'bounded unique task membership')
    need(sum(len(t['program_results']) for t in tasks) <= 4096, 'program membership cap')
    for t in tasks:
        need(set(t) == {'variant_identity', 'context', 'dataset_publication', 'encoder_publication', 'program_results'}
             and type(t['program_results']) is list and len(t['program_results']) <= 256, 'closed task membership')
    need(byte_sha(registry_bytes) == spec['parent_registry_sha256'] and len(registry_bytes) <= 32*1024**2,
         'parent registry hash/size mismatch')
    parents = {}
    for line in registry_bytes.splitlines():
        p = document(line)
        lineage = sha256(p['parent_lineage_sha256'])
        need(lineage not in parents, 'duplicate parent lineage')
        parents[lineage] = p
    clock = time.time if now is None else now
    support = _authority(spec, authority_bytes, parents, clock(), authority_sha256)
    modules, vocab, grammar = _consumer(spec['consumer_source_sha256'], grammar_bytes, spec)
    sampler, tokenizer, automaton = (modules['dltrain.'+n] for n in ('sampler', 'dsl_tokenizer', 'automaton'))
    config = sampler.SamplerConfig()  # frozen successor4 teachers/group8, never tuned by task count
    reader = _Reader(store)
    contexts, records, vectors, assignments, details, refusals = [], [], {}, {}, {}, []
    seeds = spec['shared_seed_program_sha256']
    need(type(seeds) is list and len(set(seeds)) == len(seeds), 'unique shared seed declarations')
    for seed in seeds:
        sha256(seed)
    seen_contexts = set()
    for task in tasks:
        identity = task['variant_identity']
        ctx, meta, raw, dataset_refs = _dataset(reader, task, spec, parents, support[identity])
        need(ctx.identity not in seen_contexts, 'duplicate context identity')
        seen_contexts.add(ctx.identity)
        z, loaded = _encodings(reader, task, spec, meta, dataset_refs)
        role = {'train': 'train_fit', 'validation': 'dev'}[meta['partition']]
        family = meta['parent_lineage_sha256']
        need(assignments.setdefault(family, role) == role, 'family role conflict')
        y, _, _ = decode_input(raw)
        common = {'context_sha256': ctx.identity, 'context_schema': 'jope.flywheel-context/v1',
                  'dataset_hash': identity, 'family_id': family, 'split_role': role}
        contexts.append({**common, 'y_train': y.tolist(), 'flywheel_context': asdict(ctx)})
        vectors[ctx.identity] = z
        programs, baseline = [], None
        for ref in task['program_results']:
            entry = document(reader.read(ref))
            need(set(entry['artifacts']) == set(ARTIFACTS), 'all seven program artifacts required')
            dsl = reader.read(entry['artifacts']['program']).decode('utf-8')
            measure = verified_measurement(reader, ctx, dsl, entry)
            need(reader.read(entry['artifacts']['input']) == raw, 'program scored different input bytes')
            evaluation = document(reader.read(entry['artifacts']['evaluation']))
            pair = mean_baselines(raw, evaluation)
            need(baseline is None or baseline == pair, 'baseline pair changes between programs')
            baseline = pair
            ph = measure['program_sha256']
            need(ph not in {p['program_sha256'] for p in programs}, 'duplicate program membership')
            program_detail = {**measure, 'wrapper': normalized_ref(ref), 'baseline_rmse_j': pair}
            programs.append(program_detail)
            tokens = tokenizer.tokenize(dsl, vocab)
            reason = None
            if tokens[1] == tokenizer.RAW_BEGIN:
                reason = 'raw_form'
            elif len(tokens) > spec['max_seq']:
                reason = 'over_max_seq'
            elif tokenizer.detokenize(tokens, vocab) != dsl:
                raise ValueError('lossless tokenizer roundtrip failed')
            elif not grammar.accepts(tokens, automaton.Mode.SYNTAX).ok:
                reason = 'syntax_refused'
            if reason:
                refusals.append({'context_sha256': ctx.identity, 'variant_identity': identity,
                                 'program_sha256': ph, 'reason': reason, 'length': len(tokens)})
                continue
            walls = [f.get('wall_us') for f in evaluation['inner_cv']]
            wall = sum(walls) if all(type(w) is int and w >= 0 for w in walls) else None
            records.append({**common, 'program_sha256': ph, 'skeleton_sha256': tokenizer.skeleton_sha256(dsl),
                            'tokens': tokens, 'length': len(tokens), 'outcome': 'completed',
                            'rmse_j': measure['inner_fold_rmse'], 'baseline_rmse_j': pair, 'wall_us': wall,
                            'backend': 'local', 'shared_seed': ph in seeds})
        details[ctx.identity] = {'variant_identity': identity, 'parent_lineage_sha256': family, 'split_role': role,
                                'vectors': loaded, 'programs': programs, 'baseline_rmse_j': baseline, 'context': asdict(ctx), 'partition': meta['partition'],
                                'source_publications': {key: normalized_ref(task[key]) for key in ('dataset_publication', 'encoder_publication')}}
    corpus = sampler.Corpus.build(contexts, records, vectors, assignments)
    plans = {c: sampler.context_plan(corpus, c, config).to_json() for c in sorted(corpus.contexts)}
    data, fit = None, None
    if any(corpus.contexts[c]['split_role'] == 'train_fit' and p['labelled'] for c, p in plans.items()):
        fit = sampler.fit(corpus, config)
        seqs = tuple(dict.fromkeys(tuple(r['tokens']) for r in records))
        table, ids = grammar.compile_masks([list(t) for t in seqs], automaton.Mode.SYNTAX)
        table.setflags(write=False)
        masks = dict(zip(seqs, ids))
        def legal(tokens):
            key = tuple(tokens)
            need(key in masks, 'tokens outside frozen snapshot')
            return table[masks[key][1:]]
        refs = {'corpus': 'flywheel-snapshot:'+expected_sha256, 'encoding': 'flywheel-snapshot:'+expected_sha256,
                'grammar': 'grammar:'+spec['grammar_sha256'], 'tokenizer': 'vocab:'+vocab.sha256}
        data = sampler.TrainData(corpus, fit, config, legal, vocab.size, sum(COMPONENTS.values()), refs)
    # A role change or expired freshness during an expensive load also refuses success.
    _authority(spec, authority_bytes, parents, clock(), authority_sha256)
    receipt = {'schema': RECEIPT_SCHEMA, 'state': 'loaded' if data else 'validated_no_optimizer_universe',
               'snapshot_sha256': expected_sha256, 'loader_sha256': byte_sha(Path(__file__).read_bytes()),
               'consumer_source_sha256': spec['consumer_source_sha256'], 'authority_sha256': authority_sha256,
               'adapter_dependency_sha256': spec['adapter_dependency_sha256'],
               'parent_registry_sha256': spec['parent_registry_sha256'], 'components': COMPONENTS,
               'ctx_dim': sum(COMPONENTS.values()), 'feature_layout': [['pooled_states_mean_std_p10_p90', 0, 2048], ['scalars_query_r2_mean_spread_log_spread', 2048, 2051]], 'weight_sha256': WEIGHT_SHA256, 'roster_audit_sha256': spec['roster_audit_sha256'], 'vocab_sha256': vocab.sha256, 'grammar_sha256': spec['grammar_sha256'],
               'sampler_config': config.to_json(), 'contexts': details, 'plans': plans, 'refusals': refusals,
               'counts': {'contexts_loaded': len(contexts), 'programs_verified': sum(len(t['program_results']) for t in tasks),
                          'records_kept': len(records), 'records_refused': len(refusals), 'missing_components': 0,
                          'fit_contexts': 0 if fit is None else fit.n_contexts(),
                          'fit_group_contexts': 0 if fit is None else sum(p.stream == 'group' for p in fit.plans.values()),
                          'fit_feasibility_contexts': 0 if fit is None else sum(p.stream == 'feasibility' for p in fit.plans.values()),
                          'dev_contexts': sum(c['split_role'] == 'dev' for c in contexts), 'independent_lineages': len(assignments)},
               'train_data_content_sha256': None if data is None else data.content_sha256,
               'sampler_fit_sha256': None if fit is None else fit.sha256(), 'fit_statistics': None if fit is None else fit.statistics(),
               'all_membership_sha256': digest({'contexts': contexts, 'records': records, 'refusals': refusals, 'plans': plans}),
               'bytes_read': reader.total, 'read_budget_bytes': MAX_READ, 'optimizer_updates': 0, 'database_admission_performed': False,
               'limits': ['Development lineage split; prepared-source and unknown pretraining caveats retained.',
                          'Only KumoS2051 is read; no A0, other encoder vectors, L/Z statistics or other-model diagnostics are features.',
                          'Source publisher seals attest encoder execution; no independent execution replay here.',
                          'SYNTAX masks are a data-independent superset, not a POLICY-generation admission.',
                          'TrainData retains native sampler statistics, but this loader executes no sampler or optimizer; use the separately pinned exhaustive batch planner.',
                          'No HWM claim, optimizer exposure, queue admission or generalization improvement is inferred.']}
    canonical(receipt)
    return LoadedSnapshot(data, receipt)



def batch_manifest(loaded, *, pending_inventory_bytes=None, pending_inventory_sha256=None):
    """Build exhaustive planner membership; validation/pending/refused remain visible.

    The data_run binds the complete prepared variant and its exact two-fold
    protocol. It is one context draw, not one program or raw feature-table row.
    Group-eligible rows permit native successor_v1; feasibility-only rows retain
    their explicit restricted stream. A trainer must obey this plan, never turn
    feasibility rows into CE teachers. No optimizer exposure is inferred here.
    """
    receipt = loaded.receipt
    need(receipt['schema'] == RECEIPT_SCHEMA and receipt['components'] == COMPONENTS,
         'KumoS loaded receipt required')
    rows = []
    for cid, details in sorted(receipt['contexts'].items()):
        ctx, plan = details['context'], receipt['plans'][cid]
        need(Context(**ctx).identity == cid, 'batch context identity mismatch')
        data_run = {'schema': 'jope.flywheel-data-run/v1',
                    'variant_identity': details['variant_identity'], 'context_id': cid,
                    **{k: ctx[k] for k in ('dataset_sha256', 'target_sha256', 'split_sha256',
                                           'protocol_sha256', 'cut_id')}}
        refused = [r for r in receipt['refusals'] if r['context_sha256'] == cid]
        if plan['labelled']:
            status, reason = 'eligible', None
        elif details['programs']:
            status, reason = 'refused', 'NO_TOKENIZABLE_VERIFIED_PROGRAMS'
        else:
            status, reason = 'pending', 'NO_VERIFIED_GP_PROGRAMS'
        rows.append({'data_run': data_run, 'data_run_id': digest(data_run),
                     'parent_lineage_sha256': details['parent_lineage_sha256'],
                     'partition': details['partition'], 'status': status, 'reason': reason,
                     'payload': {'context_sha256': cid, 'stream': plan['stream'],
                                 'eligibility_reason': plan['reason'],
                                 'teachers': plan['teachers'], 'labelled_programs': plan['labelled'],
                                 'verified_programs': [r['program_sha256'] for r in details['programs']],
                                 'tokenizer_refusals': refused,
                                 'source_publications': details['source_publications']}})
    inventory_sha = None
    if pending_inventory_bytes is not None:
        inventory_sha = sha256(pending_inventory_sha256)
        need(byte_sha(pending_inventory_bytes) == inventory_sha, 'pending inventory hash mismatch')
        inventory = document(pending_inventory_bytes)
        need(set(inventory) == {'schema', 'source_registry_sha256', 'rows'} and
             inventory['schema'] == 'jope.flywheel-pending-data-runs/v1', 'pending inventory schema')
        sha256(inventory['source_registry_sha256'])
        need(type(inventory['rows']) is list and len(inventory['rows']) <= 100000, 'pending inventory row cap')
        known = {r['data_run_id']: r for r in rows}; seen = set()
        for row in inventory['rows']:
            need(set(row) == {'data_run', 'data_run_id', 'parent_lineage_sha256', 'partition', 'status', 'reason', 'payload'}
                 and row['status'] == 'pending' and row['reason'] == 'AWAITING_KUMOS_AND_VERIFIED_GP_CLOSURE',
                 'inventory rows must remain unadmitted pending support')
            rid = digest(row['data_run'])
            need(rid == row['data_run_id'] and rid not in seen, 'pending data_run identity/duplicate')
            seen.add(rid)
            if rid in known:
                need(all(row[k] == known[rid][k] for k in ('data_run', 'parent_lineage_sha256', 'partition')),
                     'inventory and loaded membership differ')
            else:
                rows.append(document(canonical(row)))
        need(set(known) <= seen, 'pending inventory omits loaded support')
    else:
        need(pending_inventory_sha256 is None, 'pending inventory bytes required')
    manifest = {'schema': 'jope.flywheel-batch-manifest/v1',
                'snapshot_sha256': receipt['snapshot_sha256'],
                'role_authority_sha256': receipt['authority_sha256'],
                'metadata': {'loader_sha256': receipt['loader_sha256'], 'components': COMPONENTS,
                             'train_data_content_sha256': receipt['train_data_content_sha256'],
                             'all_membership_sha256': receipt['all_membership_sha256'],
                             'membership_unit': 'prepared variant plus exact two-fold GP protocol',
                             'pending_inventory_sha256': inventory_sha,
                             'coverage_scope': 'Declared flywheel inventory only; original7120 corpus integration remains separate and outstanding.',
                             'exposure_claim': 'none; pending real native optimizer journal'},
                'rows': rows}
    # Reuse the planner's closed membership/lineage validation, without opening any DB.
    from .batch_stream import validate_manifest
    body = canonical(manifest)
    validate_manifest(body, byte_sha(body))
    return manifest


def save_receipt(loaded, path):
    """Create a receipt once; no overwrite, path traversal or symlink parents."""
    path = Path(path)
    need(path.is_absolute() and '..' not in path.parts and all(not p.is_symlink() for p in (path, *path.parents)),
         'absolute non-symlink receipt path required')
    body = canonical(loaded.receipt) + b'\n'
    with path.open('xb') as stream:
        stream.write(body)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    return {'path': str(path), 'sha256': byte_sha(body), 'bytes': len(body)}
