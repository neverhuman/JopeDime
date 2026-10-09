"""Read-only exact-version raw and KumoS closure, never training admission.

The receipt refs and source_proof bodies are pinned per variant by the cold
qualification manifest. There is intentionally no best-effort legacy fallback.
"""
import io
import json
import numpy as np
from jope_flywheel.program_io import need, sha, tick
from jope_flywheel.protocol import canonical, digest

BUCKET = 'veox-jopedime-learning-use2'
PREFIX = 'experiments/gp-policy-dl/v2-trainonly/'
SCHEMA = 'jope.flywheel-kumos-fresh32-publication/v1'
WORKER = '0883dd4085e01cc0e1a3353396fd7004854c4ae2e68868f1eb79a053a6cb2a42'
WEIGHT = 'fc672f536501d64356036050ac2acea7e46927f7b9ad38366abe048f88e7d423'
GENERATOR_PUBLISHER = 'e11dd3cc829b4379212a33cb69c4ce0ae992777874836253b20b0ae780d10f34'
PIPELINE = 'b4eb3a3aa5e85ff9af0eab29101c21dec74018826e3244ee9b5103caa47d5e78'
REF_KEYS = ('bucket', 'key', 'version_id', 'sha256', 'bytes')


def same_ref(a, b):
    return all(a.get(k) == b.get(k) for k in REF_KEYS)


def validate_current_receipts(config, row, meta, generation, raw_publication, cache, deadline):
    tick(deadline)
    identity = row['identity']
    need(config['required_encoders'] == ['kumo_tabular_s'], 'required KumoS-only roster')
    need(raw_publication['schema'] == 'jope.flywheel-evergreen-dataset-publication/v1' and
         raw_publication['publication_helper_sha256'] == GENERATOR_PUBLISHER and
         raw_publication['pipeline_config_sha256'] == PIPELINE and
         raw_publication['required_encoders'] == ['kumo_tabular_s'], 'current growth publication authority')
    need(raw_publication['variant_identity'] == identity and
         raw_publication['content_sha256'] == row['content_sha'] == meta['content_sha256'] and
         raw_publication['metadata_sha256'] == row['metadata_sha'] and
         raw_publication['parent_lineage_sha256'] == row['parent'] == meta['parent_lineage_sha256'] and
         raw_publication['qualification']['partition'] == meta['partition'] in ('train', 'validation') and
         raw_publication['teaching_eligible'] is False and raw_publication['new_gp_labels'] == 0,
         'raw receipt exact context/role/non-admission')
    need(raw_publication['qualification']['deterministic_replay_exact'] is True and
         raw_publication['qualification']['target_sha256'] == meta['target_sha256'] and
         raw_publication['qualification']['outer_test_rows'] == 0, 'raw replay qualification')
    need(generation['config_sha256'] == config['generator_config']['sha256'], 'unchanged transform generator')
    # Existing VersionCache reads exact VersionId + bounded SHA/bytes on every use.
    body = cache.get(row['kumo']['receipt'], identity, 'receipt')
    receipt = json.loads(body)
    need(canonical(receipt)+b'\n' == body and receipt['schema'] == SCHEMA,
         'canonical current Kumo receipt schema')
    for k in ('content_sha256', 'target_sha256', 'source_target_sha256', 'parent_lineage_sha256',
              'partition', 'rows', 'features', 'inner_counts', 'outer_test_rows'):
        need(receipt[k] == meta[k], 'Kumo input/target/role binding: '+k)
    need(receipt['variant_identity'] == identity and receipt['sequence'] == row['sequence'] and
         receipt['metadata_sha256'] == row['metadata_sha'] and receipt['host'] == 'xbabe3' and
         receipt['required_encoders'] == receipt['completed_encoders'] == ['kumo_tabular_s'] and
         receipt['missing_encoders'] == [] and receipt['roster_version'] == 'kumo-s-only-v1' and
         receipt['teaching_eligible'] is False and receipt['lift_eligible'] is False and
         receipt['new_gp_labels'] == 0 and receipt['native_context_admission'] == 'unqualified' and
         receipt['prepared'] is True and receipt['raw_units'] is False and receipt['rights'] == 'research',
         'Kumo roster and explicit non-admission')
    pin = row['kumo']
    need(receipt['publisher_sha256'] == pin['publisher_sha256'] and
         digest(receipt['source_proof']) == pin['source_proof_sha256'], 'frozen Kumo source closure')
    proof = receipt['source_proof']
    need(proof['source_sha256'] == WORKER and proof['producer_phase_sealed'] is True and
         proof['original_coordinator_state'] == 'complete' and proof['finalization_custody'] is None and
         proof['manifest']['sha256'] == pin['manifest_sha256'] and
         proof['coordinator_ledger']['event_count'] > 0 and set(proof['phase_ledgers']) == {'ofm'} and
         proof['phase_ledgers']['ofm']['event_count'] > 0, 'qualified sealed encoder provenance')
    prior = receipt['prior_dataset_publication']
    need(same_ref(prior['receipt'], row['s3']['receipt']) and
         set(prior['objects']) == {'encoder.jtt1.gz', 'metadata.json', 'generation.json'} and
         all(same_ref(prior['objects'][n], row['s3'][n]) for n in prior['objects']), 'prior dataset VersionId closure')
    need(set(receipt['encoder_records']) == {'kumo_tabular_s'} and len(receipt['objects']) == 1,
         'one required vector record')
    entry = receipt['encoder_records']['kumo_tabular_s']; record = entry['record']
    need(sha(canonical(record)) == entry['record_ref']['sha256'] and
         len(canonical(record)) == entry['record_ref']['bytes'], 'encoder record exact bytes')
    need(record['variant_identity'] == identity and record['component'] == 'kumo_tabular_s' and
         record['dimension'] == 2051 and record['phase'] == 'ofm' and
         record['manifest_sha256'] == pin['manifest_sha256'] and record['host'] == 'xbabe3' and
         record['input_raw_sha256'] == row['content_sha'] and
         record['target_sha256'] == meta['target_sha256'] and record['partition'] == meta['partition'] and
         record['parent_lineage_sha256'] == meta['parent_lineage_sha256'] and
         record['outer_test_rows'] == 0 and record['teaching_eligible'] is False,
         'component context identity')
    for name, source in [('encoder.jtt1.gz', 'source_input'), ('metadata.json', 'source_metadata')]:
        need(record[source]['sha256'] == row['files'][name]['sha256'] and
             record[source]['bytes'] == row['files'][name]['bytes'], 'component original source file')
    weight = record['producer']['weight']
    need(weight['repo'] == 'nvidia/Kumo-Tabular' and
         weight['revision'] == '3c3e10bbdb590ace29e7026847f92db3c603096d' and
         weight['file'] == 'small/regressor.pt' and weight['sha256'] == WEIGHT and
         digest(record['producer']['runtime_inventory']) == pin['runtime_inventory_sha256'],
         'exact allowed weight/runtime provenance')
    q = entry['qualification']; qbody = canonical(q['body'])
    need(sha(qbody) == q['ref']['sha256'] == record['qualification']['sha256'] and
         len(qbody) == q['ref']['bytes'] == record['qualification']['bytes'] and
         q['body']['passed'] is True and len(q['body']['tests']) == 6 and
         all(v is True for v in q['body']['tests'].values()), 'full numerical encoder qualification')
    vector_ref = receipt['objects'][0]
    need(same_ref(vector_ref, pin['vector']) and
         vector_ref['sha256'] == record['output']['sha256'] == entry['vector_ref']['sha256'] and
         vector_ref['bytes'] == record['output']['bytes'] == entry['vector_ref']['bytes'], 'vector exact bytes/version')
    vector_bytes = cache.get(vector_ref, identity, 'encodings/kumo_tabular_s.npy')
    vector = np.load(io.BytesIO(vector_bytes), allow_pickle=False)
    need(vector.shape == (2051,) and vector.dtype == np.dtype('<f4') and np.isfinite(vector).all(),
         'finite2051 float32 required')
    tick(deadline)
    return {'partition': meta['partition'], 'teaching_eligible': False, 'training_admission': False,
            'encoding_receipt_sha256': sha(body), 'vector_sha256': sha(vector_bytes)}
