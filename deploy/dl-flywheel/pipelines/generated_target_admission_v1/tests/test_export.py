"""NEW offline fixtures using the exact frozen outbox schema implementation."""
import copy
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from ..common import CheckedCallback, ExportError, canonical, digest, sha
from ..cutoff import capture_cutoff
from ..export_verified import ARTIFACTS, COEFFICIENTS, KUMO_WEIGHT_SHA256, evaluation_key, export_targets
from ..transition import validate_extension_transition
from . import fixture_callbacks as cb
from .reference_outbox import Outbox


def h(value):
    return sha(str(value).encode())


def binding(fn):
    p = Path(__file__).with_name('fixture_callbacks.py').resolve()
    return {'path': str(p), 'sha256': sha(p.read_bytes()), 'qualname': fn.__qualname__, 'dependencies': []}


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        cb.STORE, cb.ROLES, cb.MODE, cb.CALLS = {}, {}, None, []
        self.tasks, self.descriptors = [], {}
        self.rows = []
        self.counter = 0
        self.authority_pin = {'path': '/explicit/read-only/authority.json', 'sha256': h('roles')}
        self.train = self.task('train', 'train')
        self.dev = self.task('dev', 'validation')
        self.fit_ids = [h('original-native-training-context')]
        self.normalizer = {'schema': 'jope.native-kumos-preprocessing/v1', 'snapshot_sha256': h('native'),
                           'validation_used_for_fit': False, 'clip': [-10., 10.], 'mean': [0.] * 2051,
                           'scale': [1.] * 2051, 'fit_context_ids': self.fit_ids,
                           'method': 'float64 deterministic sorted-context Welford population variance'}
        self.objective = {'schema': 'jope.flywheel-native-support-objective/v1',
                          'name': 'native-support-mask-v1', 'coefficients': COEFFICIENTS}

    def ref(self, raw, label='artifact'):
        self.counter += 1
        ref = {'bucket': 'fixture-only', 'key': label + '/' + str(self.counter),
               'version_id': 'version-' + str(self.counter), 'sha256': sha(raw), 'bytes': len(raw)}
        cb.STORE[(ref['bucket'], ref['key'], ref['version_id'])] = (raw, ref['version_id'])
        return ref

    def task(self, name, role):
        ctx = {'task': 'regression', 'dataset_sha256': h('data-' + name), 'target_sha256': h('y-' + name),
               'cut_id': 'inner-2', 'evaluator_sha256': h('evaluator'), 'vocabulary_sha256': h('vocab'),
               'protocol_sha256': h('protocol'), 'split_sha256': h('split-' + name),
               'metric': 'rmse', 'units': 'prepared-units', 'direction': 'minimize'}
        task = {'variant_identity': h('variant-' + name), 'parent_lineage_sha256': h('parent-' + name),
                'context': ctx, 'dataset_publication': self.ref(b'dataset-proof'),
                'encoder_publication': self.ref(b'encoder-proof'), 'program_results': []}
        cid = digest(ctx)
        self.tasks.append(task)
        self.descriptors[cid] = {'context_id': cid, 'source_kind': 'flywheel_variant',
            'variant_identity': task['variant_identity'], 'parent_lineage_sha256': task['parent_lineage_sha256'],
            'partition': role, 'context': ctx, 'observation_dispatch': 'qualified_flywheel_worker_adapter',
            'dataset_publication': task['dataset_publication'], 'encoder_publication': task['encoder_publication']}
        cb.ROLES[task['parent_lineage_sha256']] = role
        return task

    def result(self, task, program='ridge alpha=0.1', parity=True):
        ctx = task['context']
        bodies = {'input': b'fixture-not-real-JTT1', 'program': program.encode(),
                  'evaluation': canonical({'context': ctx, 'dsl': program, 'rmse': [.5, .7]}),
                  'rust_source': b'fixture-not-real-Rust', 'rust_executable': b'fixture-not-executable',
                  'rust_parity': canonical({'passed': parity}), 'score_records': b'fixture-not-native-scores'}
        artifacts = {k: self.ref(v, k) for k, v in bodies.items()}
        entry = {'schema': 'jope.flywheel-program-result/v1', 'context': copy.deepcopy(ctx),
                 'evaluation_key': evaluation_key(ctx, program), 'program_sha256': sha(program.encode()),
                 'status': 'completed', 'measurement': 'executed', 'artifacts': artifacts}
        return self.ref(canonical(entry), 'wrapper'), entry

    def emission(self, task, *, program='ridge alpha=0.1', state='published_verified', result=None):
        if result is None and state in ('published_verified', 'cached_verified'):
            result = self.result(task, ' '.join(program.split()))[0]
        self.rows.append((task, program, state, result))

    def build(self):
        dbpath = self.root / ('capture-' + str(len(list(self.root.glob('capture-*')))) + '.sqlite')
        outbox = Outbox(dbpath, model_origin_sha256=h('origin'), max_pending=1000, max_rows=1000, now=lambda: 1000.)
        for i, (task, program, state, ref) in enumerate(self.rows):
            cid = digest(task['context']); role = self.descriptors[cid]['partition']
            selected = [cid] if role == 'train' else [digest(self.train['context'])]
            # The real outbox requires at least one training proposal in a batch.
            members = [(self.train, 'recorded', None, 'ridge alpha=0.1')] if role == 'validation' else []
            members.append((task, state, ref, program))
            batch = {'batch_id': h('batch' + str(i)), 'campaign_sha256': h('campaign'),
                     'checkpoint_sha256': h('checkpoint-' + str(i)), 'global_step': i + 1,
                     'manifest_sha256': h('manifest'), 'train_data_run_ids': selected,
                     'validation_data_run_ids': [cid] if role == 'validation' else [],
                     'emitted_train_data_run_ids': selected}
            emissions = []
            for t, st, rr, raw in members:
                c = digest(t['context']); d = self.descriptors[c]
                emissions.append({'phase': 'training' if d['partition'] == 'train' else 'validation',
                    'data_run_id': c, 'context_id': c, 'parent_lineage_sha256': t['parent_lineage_sha256'],
                    'input_descriptor': d, 'raw_program_utf8': raw, 'raw_program_sha256': sha(raw.encode()),
                    'decoder_seed': i, 'decoder_protocol_sha256': h('decoder'), 'decode_artifact_sha256': h('decode')})
            ack = outbox.record_batch(batch, emissions)
            for eid, (t, st, rr, raw) in zip(ack['emission_ids'], members):
                if st == 'recorded':
                    continue
                # Claim the intended row with the same schema, without selecting
                # unrelated intentionally pending fixture rows from earlier batches.
                lease = {'emission_id': eid, 'owner': 'fixture', 'started_at': 1000., 'expires_at': 1100.,
                         'token': digest({'id': eid, 'owner': 'fixture', 'at': 1000.})}
                outbox.db.execute('BEGIN IMMEDIATE')
                outbox.db.execute('UPDATE emissions SET state=?,lease=? WHERE id=?', ('leased', canonical(lease), eid))
                outbox._event('emission.leased', lease)
                outbox.db.execute('COMMIT')
                if st == 'leased':
                    continue
                success = st in ('published_verified', 'cached_verified')
                outcome = {'state': st, 'evaluation_key': evaluation_key(t['context'], ' '.join(raw.split())) if success else None,
                           'record_ref': rr if success else None,
                           'attempt_custody_ref': self.ref(b'failure-proof') if st == 'failure_custody_verified' else None,
                           'error': None if success else 'fixture nonnumeric outcome'}
                outbox.finish(lease, outcome, verify_outcome=lambda _b, _o: True)
        full = outbox.verify()
        config = copy.deepcopy(outbox.config)
        head = {k: full[k] for k in ('events', 'head_sha256', 'observations', 'batches')}
        outbox.close()  # Final WAL checkpoint leaves a standalone immutable image.
        raw = capture_cutoff(dbpath, expected_sha256=sha(dbpath.read_bytes()), expected_config=config, expected_head=head)
        prior = {'schema': 'jope.native-variant-extension/v1', 'native_snapshot_sha256': h('native'),
            'parent_registry_sha256': h('registry'), 'generator_config_sha256': h('generator'),
            'authority_helper_sha256': h('authority-helper'), 'dependency_sha256': {},
            'dataset_publishers': [], 'encoder_releases': [], 'tasks': copy.deepcopy(self.tasks),
            'max_read_bytes': 8*1024**3, 'preprocessing_sha256': digest(self.normalizer),
            'teacher_rule': 'disabled_for_unpromoted_variants/v1'}
        source = Path(cb.__file__).resolve()
        construction = {'path': str(source), 'sha256': sha(source.read_bytes())}
        request = {'schema': 'jope.generated-train-target-export-request/v1', 'cutoff_sha256': sha(raw),
            'predecessor_sha256': digest(prior), 'objective_sha256': digest(self.objective),
            'normalizer_sha256': digest(self.normalizer), 'native_train_context_ids_sha256': digest(self.fit_ids),
            'construction_sources': {'labels': construction, 'compose': construction},
            'roster': {'name': 'kumo-s-only-v1', 'dimension': 2051, 'weight_sha256': KUMO_WEIGHT_SHA256},
            'callbacks': {k: binding(fn) for k, fn in [('read_version', cb.read_version), ('canonicalize', cb.canonicalize),
                                                      ('verify', cb.verify), ('authority', cb.authority)]},
            'authority_source_pins': [self.authority_pin], 'admitted_descriptors': copy.deepcopy(self.descriptors),
            'allowed_campaign_sha256': [h('campaign')],
            'limits': {'max_contexts': 16, 'max_new_rows': 64, 'max_reads': 4096, 'max_read_bytes': 512*1024**2, 'seconds': 120}}
        return {'dbpath': dbpath, 'config': config, 'head': head, 'cutoff': raw, 'predecessor': canonical(prior), 'request': request}

    def run_export(self, fixture):
        request = canonical(fixture['request'])
        return json.loads(export_targets(request, expected_request_sha256=sha(request),
            cutoff_raw=fixture['cutoff'], predecessor_raw=fixture['predecessor'],
            objective_raw=canonical(self.objective), normalizer_raw=canonical(self.normalizer),
            get_version=cb.read_version, canonicalize=cb.canonicalize, verify=cb.verify, authority=cb.authority))

    def test_success_all_seven_and_no_numerical_or_admission_claim(self):
        self.emission(self.train)
        f = self.build(); result = self.run_export(f)
        self.assertEqual(result['counts']['appended'], 1)
        self.assertEqual(result['artifact_reads']['calls'], 8)
        self.assertEqual(set(result['rows'][0]['artifact_refs']), set(ARTIFACTS))
        self.assertIs(result['rows'][0]['ce_eligible'], False)
        self.assertIs(result['numeric_targets_materialized'], False)
        self.assertIs(result['admission_committed'], False)
        self.assertEqual(result['optimizer_updates'], 0)

    def test_complete_pending_refusal_validation_census(self):
        for st in ('recorded', 'leased', 'unknown_attempt', 'pending_unsupported', 'failure_custody_verified'):
            self.emission(self.train, state=st)
        self.emission(self.dev)
        result = self.run_export(self.build())
        self.assertEqual(result['counts'], {'appended': 0, 'duplicates': 0, 'validation': 1, 'pending': 5, 'refused': 1})
        self.assertEqual(result['artifact_reads']['calls'], 0)

    def test_canonical_alias_dedup_after_selection_cap(self):
        ref, _ = self.result(self.train)
        self.emission(self.train, result=ref)
        self.emission(self.train, program=' ridge   alpha=0.1 ', result=ref, state='cached_verified')
        f = self.build(); f['request']['limits']['max_new_rows'] = 1
        result = self.run_export(f)
        self.assertEqual(result['counts']['duplicates'], 1)
        self.assertEqual(len(result['rows'][0]['source_emission_ids']), 2)
        self.assertEqual(len(result['extension']['tasks'][0]['program_results']), 1)
        self.assertEqual(result['artifact_reads']['calls'], 8)

    def test_conflicting_alias_ref_refused_even_after_cap(self):
        self.emission(self.train); self.emission(self.train)
        f = self.build(); f['request']['limits']['max_new_rows'] = 1
        with self.assertRaisesRegex(ExportError, 'conflicting result'):
            self.run_export(f)

    def test_old_ref_retained_and_existing_alias_not_double_weighted(self):
        ref, _ = self.result(self.train); self.train['program_results'].append(ref)
        self.emission(self.train, result=ref)
        result = self.run_export(self.build())
        self.assertEqual(result['counts']['duplicates'], 1)
        self.assertEqual(result['rows'], [])
        self.assertEqual(result['extension']['tasks'][0]['program_results'], [ref])

    def test_old_reference_removal_reordering_and_replacement_rejected(self):
        a, _ = self.result(self.train); b, _ = self.result(self.train, 'ridge alpha=2')
        self.train['program_results'] = [a, b]
        old = json.loads(self.build()['predecessor'])
        for refs in ([b], [b, a], [a, a], [a, {**b, 'version_id': 'changed'}]):
            with self.subTest(refs=refs), self.assertRaises(ExportError):
                new = copy.deepcopy(old); new['tasks'][0]['program_results'] = refs
                validate_extension_transition(old, new)

    def test_duplicate_weight_in_predecessor_refused(self):
        ref, _ = self.result(self.train); self.train['program_results'] = [ref, ref]
        with self.assertRaisesRegex(ExportError, 'duplicate old result'):
            self.run_export(self.build())

    def test_stale_version_and_corrupt_bytes_refused(self):
        self.emission(self.train); f = self.build()
        for mode in ('wrong_version', 'corrupt'):
            with self.subTest(mode=mode), self.assertRaisesRegex(ExportError, 'version/length/SHA'):
                cb.MODE = mode; self.run_export(f)

    def test_missing_version_and_missing_object_refused(self):
        ref, _ = self.result(self.train); self.emission(self.train, result=ref); f = self.build()
        cb.STORE.pop((ref['bucket'], ref['key'], ref['version_id']))
        with self.assertRaises(KeyError): self.run_export(f)
        for version in ('', 'null', None):
            with self.subTest(version=version):
                cutoff = json.loads(f['cutoff']); cutoff['records'][0]['outcome']['record_ref']['version_id'] = version
                f['cutoff'] = canonical(cutoff); f['request']['cutoff_sha256'] = sha(f['cutoff'])
                with self.assertRaises(ExportError): self.run_export(f)

    def test_wrong_wrapper_context_program_and_protocol_refused(self):
        for field in ('context', 'program_sha256', 'evaluation_key'):
            with self.subTest(field=field):
                ref, entry = self.result(self.train)
                if field == 'context': entry[field]['protocol_sha256'] = h('other-protocol')
                else: entry[field] = h('other')
                wrong = self.ref(canonical(entry)); self.rows = []
                self.emission(self.train, result=wrong)
                with self.assertRaisesRegex(ExportError, 'wrapper identity'): self.run_export(self.build())

    def test_wrong_objective_and_normalizer_refused(self):
        self.emission(self.train); f = self.build()
        self.objective['coefficients'] = {**COEFFICIENTS, 'q': .5}
        f['request']['objective_sha256'] = digest(self.objective)
        with self.assertRaisesRegex(ExportError, 'unchanged supervised'): self.run_export(f)
        self.objective['coefficients'] = COEFFICIENTS; f['request']['objective_sha256'] = digest(self.objective)
        self.normalizer['validation_used_for_fit'] = True
        f['request']['normalizer_sha256'] = digest(self.normalizer)
        with self.assertRaisesRegex(ExportError, 'train-only Kumo'): self.run_export(f)

    def test_valid_but_unapproved_kumo_weight_refused(self):
        self.emission(self.train); f = self.build()
        f['request']['roster']['weight_sha256'] = h('different-valid-64-hex-weight')
        with self.assertRaisesRegex(ExportError, 'exact campaign KumoSmall weight'):
            self.run_export(f)
        self.assertEqual(cb.CALLS, [])

    def test_normalizer_roster_drift_refused(self):
        self.emission(self.train); f = self.build()
        f['request']['native_train_context_ids_sha256'] = h('wrong-fit-roster')
        with self.assertRaisesRegex(ExportError, 'normalization roster'): self.run_export(f)

    def test_validation_alias_cannot_claim_training_role(self):
        self.emission(self.train); f = self.build()
        cb.ROLES[self.train['parent_lineage_sha256']] = 'validation'
        with self.assertRaisesRegex(ExportError, 'inherited parent roles'): self.run_export(f)

    def test_stale_role_proof_wrong_objective_and_drift_refused(self):
        self.emission(self.train); f = self.build()
        for mode in ('stale_authority', 'wrong_authority_objective', 'role_drift'):
            with self.subTest(mode=mode), self.assertRaises(ExportError):
                cb.MODE = mode; cb.CALLS.clear(); self.run_export(f)

    def test_bare_verifier_success_and_wrong_identity_refused(self):
        self.emission(self.train); f = self.build()
        for mode in ('bare_true', 'wrong_measurement_context', 'invented_q'):
            with self.subTest(mode=mode), self.assertRaisesRegex(ExportError, 'structured exact verifier'):
                cb.MODE = mode; self.run_export(f)

    def test_parity_rejection_no_target_manifest(self):
        ref, _ = self.result(self.train, parity=False); self.emission(self.train, result=ref)
        with self.assertRaisesRegex(ExportError, 'parity'): self.run_export(self.build())

    def test_new_unique_limit_retains_complete_pending_denominator(self):
        self.emission(self.train)
        self.emission(self.train, program='ridge alpha=2')
        f = self.build(); f['request']['limits']['max_new_rows'] = 1
        result = self.run_export(f)
        self.assertEqual(result['counts']['appended'], 1); self.assertEqual(result['counts']['pending'], 1)

    def test_artifact_budget_exhaustion_and_deadline_refuse_partial_manifest(self):
        self.emission(self.train); f = self.build()
        f['request']['limits']['max_reads'] = 1
        with self.assertRaisesRegex(ExportError, 'read budget'): self.run_export(f)
        f['request']['limits']['max_reads'] = 4096
        with patch('generated_target_admission_v1.common.time.monotonic', side_effect=[0., 121.]):
            with self.assertRaisesRegex(ExportError, 'deadline'): self.run_export(f)

    def test_no_sql_network_subprocess_or_source_mutation_during_export(self):
        self.emission(self.train); f = self.build()
        before = {str(p): sha(p.read_bytes()) for p in self.root.rglob('*') if p.is_file()}
        with patch('sqlite3.connect', side_effect=AssertionError('no DB in export')), \
             patch('subprocess.run', side_effect=AssertionError('no evaluator')), \
             patch('socket.socket', side_effect=AssertionError('no network')):
            self.run_export(f)
        after = {str(p): sha(p.read_bytes()) for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        self.assertTrue(all(k[0] in ('get', 'authority', 'canonicalize', 'verify') for k in cb.CALLS))

    def test_snapshot_no_mutation_and_exact_head(self):
        self.emission(self.train); f = self.build(); p = f['dbpath']; raw = p.read_bytes()
        before = sorted(x.name for x in p.parent.iterdir())
        got = capture_cutoff(p, expected_sha256=sha(raw), expected_config=f['config'], expected_head=f['head'])
        self.assertEqual(got, f['cutoff']); self.assertEqual(raw, p.read_bytes())
        self.assertEqual(before, sorted(x.name for x in p.parent.iterdir()))
        with self.assertRaisesRegex(ExportError, 'event head drift'):
            capture_cutoff(p, expected_sha256=sha(raw), expected_config=f['config'], expected_head={**f['head'], 'head_sha256': h('wrong')})

    def test_snapshot_sidecars_hash_config_and_row_bounds(self):
        self.emission(self.train); f = self.build(); p = f['dbpath']
        for suffix in ('-wal', '-shm', '-journal'):
            q = Path(str(p) + suffix); q.write_bytes(b'x')
            with self.assertRaisesRegex(ExportError, 'sidecars'):
                capture_cutoff(p, expected_sha256=sha(p.read_bytes()), expected_config=f['config'], expected_head=f['head'])
            q.unlink()
        with self.assertRaisesRegex(ExportError, 'hash mismatch'):
            capture_cutoff(p, expected_sha256=h('wrong'), expected_config=f['config'], expected_head=f['head'])
        with self.assertRaisesRegex(ExportError, 'configuration'):
            capture_cutoff(p, expected_sha256=sha(p.read_bytes()), expected_config={**f['config'], 'max_rows': 999}, expected_head=f['head'])
        with self.assertRaisesRegex(ExportError, 'head exceeds'):
            capture_cutoff(p, expected_sha256=sha(p.read_bytes()), expected_config=f['config'], expected_head=f['head'], max_events=1)

    def test_snapshot_event_and_materialized_ack_tampering(self):
        for mutation in ('event', 'state', 'ack'):
            with self.subTest(mutation=mutation):
                self.rows = []; self.emission(self.train); f = self.build(); p = f['dbpath']
                db = sqlite3.connect(p)
                if mutation == 'event': db.execute("UPDATE events SET previous=? WHERE seq=2", (h('wrong'),))
                elif mutation == 'state': db.execute("UPDATE emissions SET state='unknown_attempt'")
                else:
                    row = db.execute('SELECT id,ack FROM batches').fetchone(); ack = json.loads(row[1]); ack['numeric_outcomes_required_for_this_optimizer'] = True
                    db.execute('UPDATE batches SET ack=? WHERE id=?', (canonical(ack), row[0]))
                db.commit(); db.close()
                with self.assertRaises(ExportError):
                    capture_cutoff(p, expected_sha256=sha(p.read_bytes()), expected_config=f['config'], expected_head=f['head'])

    def test_loaded_callback_code_must_match_current_pinned_file(self):
        p = self.root / 'stale.py'; p.write_text('def call(x):\n    return x\n')
        spec = importlib.util.spec_from_file_location('stale_fixture', p); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        old = {'path': str(p), 'sha256': sha(p.read_bytes()), 'qualname': 'call', 'dependencies': []}
        CheckedCallback(mod.call, old)
        p.write_text('def call(x):\n    return True\n'); changed = {**old, 'sha256': sha(p.read_bytes())}
        with self.assertRaisesRegex(ExportError, 'loaded callback bytecode'): CheckedCallback(mod.call, changed)

    def test_callback_source_and_dependency_drift(self):
        self.emission(self.train); f = self.build()
        f['request']['callbacks']['verify']['sha256'] = h('wrong-code')
        with self.assertRaisesRegex(ExportError, 'callback source'): self.run_export(f)
        f['request']['callbacks']['verify'] = binding(cb.verify)
        f['request']['callbacks']['verify']['dependencies'] = [{'path': str(Path(cb.__file__).resolve()), 'sha256': h('wrong-dependency')}]
        with self.assertRaisesRegex(ExportError, 'dependency'): self.run_export(f)


if __name__ == '__main__':
    unittest.main()
