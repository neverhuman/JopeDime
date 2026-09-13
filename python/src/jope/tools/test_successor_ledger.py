"""Successor ledger tests on Codex's REAL seed-continuation-v2 qualification-all64 fixture.

The ledger binds its evaluator to a manifest (bindings[cpu_executable], cross-checked against the binary on disk) and
pins the 2 s resource protocol, so the all64 rows (evaluator 340e5798..., 30 s protocol) are the REFUSED fixture. The
VERIFIED fixture is derived from it by rewriting the evaluator/protocol to the bound identities (new episode ids, new
evaluation identity digests) and supplying the seeds' admitted row-0 RMSE (F5 evidence).
"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import successor_ledger as sl
import successor_targets as st
from successor_ledger import Ledger, verify_episode, encoded, digest, canonical, IDENTITY_KEYS, replicate_rng, selection_key

PARALLEL = Path(__file__).resolve().parents[1]
FIXTURE = PARALLEL / 'recovery/eight-a100-20260912/experiments/seed-continuation-v2/qualification-all64'
PREPARED_REL = 'data/prepared/views-expanded-20/55e9dfc7a86361d7'
PREPARED = PARALLEL / PREPARED_REL
FIXTURE_EVALUATOR = '340e579811cca4ef47c1020f699b8d5ac8bf651a52632e57d73806a633634055'
V2_RESOURCE = 'single_candidate_30s_8gib_v1'   # the v2 driver stamps the deployed 30 s constant (lane defect 6a)
FOREIGN = 'f' * 64
GENERATION = 73
CODEX_RESERVED_CORE_SECONDS = 45.96471887000371   # qualification-all64/economics.json
EVALUATOR_BYTES = b'frozen evaluator fixture bytes'
ONESTEP = 'onestep_forced_mutation_v2'
TIERB = 'bounded_seed_lineage_v3_cap2s'
TIERB4 = 'bounded_seed_lineage_v3_cap4s'


class FixtureCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert FIXTURE.is_dir(), 'real fixture missing: ' + str(FIXTURE)
        assert (PREPARED / 'prepared.json').is_file(), 'prepared dataset missing: ' + str(PREPARED)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir='/dev/shm')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'parallel'
        shutil.copytree(FIXTURE, self.root / 'fixture/all64')
        (self.root / PREPARED_REL / 'cut-0').mkdir(parents=True)
        shutil.copy(PREPARED / 'prepared.json', self.root / PREPARED_REL / 'prepared.json')
        shutil.copy(PREPARED / 'cut-0/preprocessing.json', self.root / PREPARED_REL / 'cut-0/preprocessing.json')
        self.episodes = sorted(d for d in (self.root / 'fixture/all64').iterdir() if d.is_dir())
        self.identity = json.loads((self.episodes[0] / 'identity.json').read_bytes())
        self.assertEqual(self.identity['evaluator_sha256'], FIXTURE_EVALUATOR)
        self.vocabulary = self.identity['vocabulary_sha256']
        (self.root / 'bin').mkdir()
        (self.root / 'bin/gp-native-v5').write_bytes(EVALUATOR_BYTES)
        self.bound = sl.sha(self.root / 'bin/gp-native-v5')
        self.assertNotEqual(self.bound, FIXTURE_EVALUATOR)
        self.manifest(self.bound)

    def manifest(self, evaluator, name='manifest.json'):
        (self.root / name).write_bytes(encoded(dict(schema='parallel.continuous_learning.v1', cpu_executable='bin/gp-native-v5',
                                                     bindings={'bin/gp-native-v5': evaluator}, vocabulary_sha256=self.vocabulary, output='fixture')))
        return name

    def ledger(self, name='successor', **extra):
        return Ledger(self.root, name, 'fixture-run', 'fixture', 'manifest.json', vocabulary=self.vocabulary, **extra)

    def ingest_all(self, ledger, episodes=None, generation=GENERATION, **extra):
        return [ledger.ingest_completed(d, generation=generation, arm='neural', role='candidate', replicate=0, seed_origin='neural_policy', **extra)
                for d in (episodes if episodes is not None else self.episodes)]

    def derive(self, source, generation_dir, rng_seed, mutate=None, identity_updates=None):
        """A byte-consistent copy of a real episode under a new rng (new identity), optionally rewritten or corrupted."""
        identity = json.loads((source / 'identity.json').read_bytes())
        identity['rng_seed'] = rng_seed
        identity.update(identity_updates or {})
        identity_bytes = encoded(identity)
        episode_id = digest(identity_bytes)
        rows = json.loads((source / 'search/observations.json').read_bytes())
        receipt = json.loads((source / 'search/search.json').read_bytes())
        for row in rows:
            row.update(rng_seed=rng_seed, continuation_sha256=episode_id)
        receipt.update(rng_seed=rng_seed, continuation_sha256=episode_id)
        if mutate:
            mutate(rows, receipt)
        for row in rows:
            row['evaluation_identity_sha256'] = digest(canonical({k: row[k] for k in IDENTITY_KEYS}))
        observations = encoded(rows)
        receipt['observations_sha256'] = digest(observations)
        dest = generation_dir / episode_id
        (dest / 'search').mkdir(parents=True)
        (dest / 'identity.json').write_bytes(identity_bytes)
        shutil.copy(source / 'seeds.json', dest / 'seeds.json')
        if (source / 'result.json').is_file():
            shutil.copy(source / 'result.json', dest / 'result.json')
        (dest / 'manifest.json').write_bytes(encoded(dict(schema='parallel.seed_experiment.v2', continuation_identity_sha256=episode_id, final_tests_opened=False)))
        (dest / 'search/observations.json').write_bytes(observations)
        (dest / 'search/search.json').write_bytes(encoded(receipt))
        return dest, identity, rows

    def rebind(self, rows, receipt):
        """Rewrite a v2 episode to the bound evaluator and the pinned 2 s protocol (the only way its rows can verify)."""
        for row in rows:
            row['evaluator_sha256'] = self.bound
            row['resource_protocol'] = sl.RESOURCE_PROTOCOL
        receipt['evaluator_sha256'] = self.bound

    def verified_copy(self, source, generation_dir, rng_seed, mutate=None):
        def rewrite(rows, receipt):
            self.rebind(rows, receipt)
            if mutate:
                mutate(rows, receipt)
        return self.derive(source, generation_dir, rng_seed, rewrite, identity_updates=dict(evaluator_sha256=self.bound))

    def verified_fixture(self, generation_dir, rng_base=1000):
        """All 64 episodes rebound to the manifest evaluator plus their seeds' admitted row-0 RMSE map."""
        dirs, seed_rmse = [], {}
        for index, source in enumerate(self.episodes):
            dest, identity, rows = self.verified_copy(source, generation_dir, rng_base + index)
            dirs.append(dest)
            seed_rmse.update(sl.admitted_rmse(rows[:1]))
        return dirs, seed_rmse


class RealFixtureTests(FixtureCase):
    def test_all64_fixture_reproduces_the_reported_numbers_and_is_refused_by_the_bound_ledger(self):
        ledger = self.ledger()
        results = self.ingest_all(ledger)
        self.assertEqual(len(results), 64)
        # Measurement never depends on admission: the numbers reproduce while every episode stays unverified.
        self.assertFalse(any(r['verified'] for r in results))
        for result in results:
            self.assertIn('evaluator_mismatch: ' + FIXTURE_EVALUATOR, result['reasons'])
            self.assertTrue(any(reason.startswith('row0_resource_protocol: ' + V2_RESOURCE) for reason in result['reasons']), result['reasons'])
            self.assertTrue(any(reason.startswith('row0_label_unverif') for reason in result['reasons']), result['reasons'])
            self.assertIsNone(result['row0_label_verified'])
        self.assertEqual(sum(r['measurement_status'] == 'measured' for r in results), 61)
        self.assertEqual(ledger.db.execute('SELECT count(*) FROM seed_values').fetchone()[0], 64)
        self.assertEqual(ledger.db.execute('SELECT count(*) FROM seed_values WHERE own_rmse IS NOT NULL').fetchone()[0], 61)
        info = ledger.info_report(GENERATION)
        pooled = info['pooled']
        self.assertEqual(info['schema'], 'parallel.successor_info.v1')
        self.assertEqual((pooled['candidates'], pooled['measured'], pooled['finite_children']), (64, 61, 60))
        self.assertEqual((pooled['better'], pooled['tied'], pooled['worse']), (16, 6, 38))
        self.assertEqual(pooled['flips'], 16)
        context = info['contexts'][self.identity['prepared_sha256'] + ':0']
        self.assertAlmostEqual(context['spearman_r0_log_improvement'], 0.703, delta=0.01)
        self.assertEqual((context['p_flip_finite_flips'], context['p_flip_finite_pairs']), (63, 1770))
        self.assertAlmostEqual(context['p_flip_finite'], 63 / 1770)
        self.assertEqual((context['p_flip_flips'], context['p_flip_pairs']), (64, 1830))
        self.assertFalse(context['child_beats_best_parent'])
        self.assertAlmostEqual(context['best_parent'], 0.7387020167237595)
        self.assertAlmostEqual(context['best_child'], 0.7543831431670589)
        bundle = ledger.bundle(GENERATION)
        self.assertEqual(bundle['admitted'], 0)
        sources = json.loads((self.root / 'successor/admitted-sources.json').read_bytes())
        self.assertEqual(sources['sources'], [])
        self.assertEqual(len(sources['excluded']), 1)
        self.assertEqual(sources['excluded'][0]['verified'], 0)
        self.assertIn('evaluator_mismatch', sources['excluded'][0]['reasons'])
        self.assertIn('row0_resource_protocol', sources['excluded'][0]['reasons'])
        self.assertEqual(sources['row0_mismatches'], 0)
        self.assertEqual(sources['evaluator_binding']['evaluator_sha256'], self.bound)
        self.assertTrue(sources['evaluator_binding']['binary_present'])
        export = json.loads(Path(ledger.export(GENERATION)['path']).read_bytes())
        self.assertTrue(all(r['verified'] is False and r['row0_label_verified'] is None for r in export['rows']))
        self.assertTrue(all('evaluator_mismatch: ' + FIXTURE_EVALUATOR in r['verification_reasons'] for r in export['rows']))
        print('\nall64 fixture: episodes=64 measured(r0 finite)=61 finite_children=60 better/tied/worse=16/6/38 '
              'spearman(r0, ln r0/r_child)=%.4f p_flip=%d/%d=%.4f (refused: foreign evaluator, 30 s protocol)' % (
                  context['spearman_r0_log_improvement'], context['p_flip_finite_flips'], context['p_flip_finite_pairs'], context['p_flip_finite']))

    def test_export_is_immutable_and_bundle_hashes_match_on_the_rebound_fixture(self):
        ledger = self.ledger()
        generation = GENERATION + 1
        episodes, seed_rmse = self.verified_fixture(self.root / 'fixture/rebound')
        self.assertEqual(len(seed_rmse), 61)   # three seeds timed out: no admitted row-0 RMSE exists for them
        results = self.ingest_all(ledger, episodes, generation=generation, seed_rmse=seed_rmse)
        self.assertEqual(sum(r['verified'] for r in results), 61)
        self.assertEqual(sum(r['row0_label_verified'] is True for r in results), 61)
        unmeasured = [r for r in results if not r['verified']]
        self.assertEqual([r['reasons'] for r in unmeasured], [['row0_label_unverifiable: seed row unmeasured']] * 3)
        export = ledger.export(generation)
        path = Path(export['path'])
        self.assertEqual(path, self.root / 'successor/seed-values-74.json')
        document = json.loads(path.read_bytes())
        self.assertEqual(document['schema'], 'parallel.seed_value.v1')
        self.assertIs(document['final_tests_opened'], False)
        self.assertEqual(document['evaluator_sha256'], self.bound)
        self.assertEqual(len(document['rows']), 64)
        self.assertEqual(document['counts']['by_status'], dict(measured=61, failed=3))
        self.assertEqual(export['sha256'], ledger.export(generation)['sha256'])
        row = next(r for r in document['rows'] if r['status'] == 'measured')
        for key in ('generation', 'dataset', 'family', 'prepared_sha256', 'cut', 'context_key', 'episode_id', 'identity_sha256', 'protocol', 'role',
                    'arm', 'seed_origin', 'seed_program_sha256', 'seed_genome', 'producing_checkpoint_sha256', 'evaluator_sha256', 'driver_sha256',
                    'r0', 'child', 'flip', 'S_rmse', 'replicate_pair', 'status', 'completed_at', 'row0_label_verified', 'verification_reasons',
                    'admitted_seed_rmse'):
            self.assertIn(key, row)
        self.assertEqual(row['protocol'], 'onestep_forced_mutation_v2')
        self.assertEqual(row['S_rmse'], min(row['r0'], row['child']['rmse']))
        self.assertEqual(digest(row['seed_genome'].encode()), row['seed_program_sha256'])
        self.assertIs(row['row0_label_verified'], True)
        self.assertEqual(row['admitted_seed_rmse'], row['r0'])
        self.assertEqual(row['verification_reasons'], [])
        failed = next(r for r in document['rows'] if r['status'] == 'failed')
        self.assertIsNone(failed['r0'])
        self.assertEqual(failed['failure'], 'candidate_or_view_timeout')
        self.assertIsNone(failed['row0_label_verified'])
        bundle = ledger.bundle(generation)
        self.assertEqual(bundle['schema'], 'parallel.successor_bundles.v1')
        self.assertEqual((bundle['contexts'], bundle['admitted']), (1, 1))
        entry = bundle['bundles'][0]
        self.assertTrue(entry['admitted'])
        self.assertIs(entry['row0_label_verified'], True)
        self.assertEqual((entry['verified'], entry['unverified'], len(entry['failed_unmeasured']), entry['row0_mismatches']), (61, [], 3, 0))
        directory = self.root / 'successor/bundles/generation-74' / entry['context_key']
        observations = (directory / 'observations.json').read_bytes()
        receipt = json.loads((directory / 'search.json').read_bytes())
        self.assertEqual(receipt['schema'], 'parallel.search.v1')
        self.assertEqual(receipt['observations_sha256'], digest(observations))
        self.assertEqual(receipt['observations_sha256'], entry['observations_sha256'])
        expected_rows = []
        for episode in receipt['bundle']['episodes']:
            source = self.root / episode['path']
            self.assertEqual(digest((source / 'observations.json').read_bytes()), episode['observations_sha256'])
            self.assertEqual(digest((source / 'search.json').read_bytes()), episode['search_sha256'])
            expected_rows.extend(json.loads((source / 'observations.json').read_bytes()))
        self.assertEqual(digest(encoded(expected_rows)), receipt['observations_sha256'])
        self.assertEqual(receipt['completed_evaluations'], 122)   # 61 verified episodes x 2 rows; unmeasured seeds never enter
        self.assertTrue(all(r['evaluator_sha256'] == self.bound and r['resource_protocol'] == sl.RESOURCE_PROTOCOL for r in expected_rows))
        self.assertEqual(receipt['bundle']['schema'], 'parallel.successor_bundle.v1')
        self.assertEqual(len(receipt['bundle']['episodes']), 61)
        self.assertEqual(receipt['bundle']['failed_unmeasured_episodes'], 3)
        self.assertEqual(receipt['evaluator_sha256'], self.bound)
        self.assertEqual(receipt['resource_protocol'], sl.RESOURCE_PROTOCOL)
        self.assertIs(receipt['final_tests_opened'], False)
        self.assertAlmostEqual(receipt['best_validation_rmse'], 0.7387020167237595 * expected_rows[0]['target_scale'])
        sources = json.loads((self.root / 'successor/admitted-sources.json').read_bytes())
        self.assertEqual(sources['schema'], 'parallel.successor_sources.v1')
        self.assertEqual(sources['sources'], [entry['path']])
        self.assertEqual(sources['generation'], generation)
        self.assertIs(sources['meta_validation_excluded'], True)
        self.assertIs(sources['row0_label_verified'], True)
        self.assertEqual((sources['row0_mismatches'], sources['row0_unverified_bundles'], sources['excluded']), (0, 0, []))
        self.assertIsNone(sources['resource_protocol_override'])
        economics = ledger.economics(generation, reserved_core_seconds=CODEX_RESERVED_CORE_SECONDS)
        self.assertEqual(economics['schema'], 'parallel.successor_economics.v1')
        self.assertEqual((economics['attempted_fits'], economics['successful_fits'], economics['unique_successful_fits'], economics['timeouts']), (128, 121, 121, 7))
        self.assertAlmostEqual(economics['unique_fits_per_core_second'], 2.6324538249044718, places=9)
        self.assertEqual(economics['superseded'], 0)
        fresh = self.ledger('successor-elapsed')
        self.ingest_all(fresh, episodes, generation=generation, seed_rmse=seed_rmse)
        measured = fresh.economics(generation)
        elapsed = sum(json.loads((d / 'result.json').read_bytes())['elapsed_seconds'] for d in self.episodes)
        self.assertEqual(measured['reserved_source'], 'sum_of_result_elapsed_seconds')
        self.assertAlmostEqual(measured['reserved_core_seconds'], elapsed)

    def test_row0_evidence_is_mandatory_for_admission(self):
        ledger = self.ledger()
        episodes, seed_rmse = self.verified_fixture(self.root / 'fixture/rebound')
        results = self.ingest_all(ledger, episodes[:4], generation=74)   # rebound rows, but no admitted seed RMSE anywhere
        self.assertFalse(any(r['verified'] for r in results))
        parents = [json.loads((d / 'identity.json').read_bytes())['parent_program_sha256'] for d in episodes[:4]]
        self.assertEqual([r['reasons'] for r in results],
                         [['row0_label_unverified: no admitted seed RMSE supplied for parent ' + parent] for parent in parents])
        self.assertTrue(all(r['row0_label_verified'] is None for r in results))
        rejected = ledger.bundle(74)
        self.assertEqual(rejected['admitted'], 0)
        self.assertEqual(rejected['bundles'][0]['reasons'], ['row0_label_unverified'])
        sources = json.loads((self.root / 'successor/admitted-sources.json').read_bytes())
        self.assertEqual(sources['sources'], [])
        self.assertEqual(sources['excluded'][0]['reasons'], ['row0_label_unverified'])
        self.assertIn('row0_label_unverified', sources['excluded'][0]['reason'])
        # The same episodes verify once the evidence is supplied (re-ingest replaces the verification); the bundle receipt of a
        # generation is immutable, so the admitted bundle is produced by a ledger that never wrote the refused one.
        again = self.ingest_all(ledger, episodes[:4], generation=74, seed_rmse=seed_rmse)
        self.assertTrue(all(r['verified'] and r['row0_label_verified'] is True for r in again))
        with self.assertRaisesRegex(ValueError, 'Immutable artifact changed'):
            ledger.bundle(74)
        evidenced = self.ledger('successor-evidence')
        self.ingest_all(evidenced, episodes[:4], generation=74, seed_rmse=seed_rmse)
        self.assertEqual(evidenced.bundle(74)['admitted'], 1)

    def test_admission_excludes_row0_mismatch_and_foreign_evaluator(self):
        ledger = self.ledger()
        derived = self.root / 'fixture/derived'
        source = self.episodes[0]
        rows0 = json.loads((source / 'search/observations.json').read_bytes())
        parent = rows0[0]['program_sha256']
        admitted_rmse = rows0[0]['metrics']['rmse']

        def bump_row0(rows, receipt):
            rows[0]['metrics']['rmse'] = admitted_rmse + 1e-9

        def foreign(rows, receipt):
            for row in rows:
                row['evaluator_sha256'] = FOREIGN
            receipt['evaluator_sha256'] = FOREIGN

        mismatch, _, _ = self.verified_copy(source, derived / 'g74', 1, bump_row0)
        alien, _, alien_rows = self.verified_copy(self.episodes[1], derived / 'g74', 2, foreign)
        alien_parent = json.loads((self.episodes[1] / 'search/observations.json').read_bytes())[0]
        clean, _, _ = self.verified_copy(source, derived / 'g75', 3)
        evidence = {parent: admitted_rmse, alien_parent['program_sha256']: alien_parent['metrics']['rmse']}
        first = ledger.ingest_completed(mismatch, generation=74, arm='neural', role='candidate', replicate=0, seed_origin='neural_policy', seed_rmse=evidence)
        self.assertFalse(first['verified'])
        self.assertIs(first['row0_label_verified'], False)
        self.assertEqual(first['reasons'], ['%s: %r != admitted %r' % (sl.ROW0_MISMATCH, admitted_rmse + 1e-9, admitted_rmse)])
        second = ledger.ingest_completed(alien, generation=74, arm='neural', role='candidate', replicate=0, seed_origin='neural_policy', seed_rmse=evidence)
        self.assertFalse(second['verified'])
        self.assertTrue(any(reason.startswith('evaluator_mismatch: ' + FOREIGN) for reason in second['reasons']), second['reasons'])
        self.assertIs(second['row0_label_verified'], True)   # its row 0 still reproduces the admitted label; the evaluator is what fails
        self.assertEqual(alien_rows[0]['evaluator_sha256'], FOREIGN)
        third = ledger.ingest_completed(clean, generation=75, arm='neural', role='candidate', replicate=0, seed_origin='neural_policy', seed_rmse=evidence)
        self.assertTrue(third['verified'], third['reasons'])
        self.assertIs(third['row0_label_verified'], True)
        self.assertEqual(third['r0'], admitted_rmse)
        rejected = ledger.bundle(74)
        self.assertEqual(rejected['admitted'], 0)
        self.assertEqual(rejected['bundles'][0]['verified'], 0)
        self.assertEqual({u['episode_id'] for u in rejected['bundles'][0]['unverified']}, {mismatch.name, alien.name})
        self.assertEqual(rejected['bundles'][0]['row0_mismatches'], 1)
        self.assertGreaterEqual(set(rejected['bundles'][0]['reasons']), {'evaluator_mismatch', 'row0_evaluator_mismatch', 'row1_evaluator_mismatch', sl.ROW0_MISMATCH})
        self.assertTrue(rejected['bundles'][0]['reason'].startswith('2 unverified episode(s): '))
        self.assertIsNone(rejected['bundles'][0]['path'])
        accepted = ledger.bundle(75)
        self.assertEqual(accepted['admitted'], 1)
        sources = json.loads((self.root / 'successor/admitted-sources.json').read_bytes())
        self.assertEqual(sources['sources'], [accepted['bundles'][0]['path']])
        self.assertIs(sources['row0_label_verified'], True)
        self.assertEqual([e['generation'] for e in sources['excluded']], [74])
        self.assertEqual(sources['excluded'][0]['row0_mismatches'], 1)
        self.assertIn(sl.ROW0_MISMATCH, sources['excluded'][0]['reasons'])
        self.assertTrue(any(reason.startswith(sl.ROW0_MISMATCH) for u in sources['excluded'][0]['unverified'] for reason in u['reasons']))
        self.assertEqual(sources['row0_mismatches'], 1)
        self.assertEqual(sources['generation'], 75)
        self.assertEqual(ledger.export(74)['counts']['episodes'], 2)
        self.assertEqual(ledger.export(74)['counts']['by_role'], dict(candidate=2))
        export = json.loads(Path(ledger.export(74)['path']).read_bytes())
        self.assertEqual(sorted((r['row0_label_verified'], r['verified']) for r in export['rows']), [(False, False), (True, False)])
        self.assertEqual(ledger.export(75)['counts']['by_status'], dict(measured=1))

    def test_verify_episode_fails_closed_without_a_receipt_or_row0_evidence(self):
        empty = self.root / 'fixture/empty'
        empty.mkdir()
        (empty / 'identity.json').write_bytes((self.episodes[0] / 'identity.json').read_bytes())
        expected = dict(evaluator_sha256=FIXTURE_EVALUATOR, resource_protocol=V2_RESOURCE)
        result = verify_episode(empty, expected)
        self.assertFalse(result['verified'])
        self.assertEqual(result['reasons'], ['search_receipt_absent'])
        self.assertEqual(result['episode_id'], self.episodes[0].name)
        rows = json.loads((self.episodes[0] / 'search/observations.json').read_bytes())
        without_evidence = verify_episode(self.episodes[0], expected)
        self.assertFalse(without_evidence['verified'])
        self.assertEqual(without_evidence['reasons'], ['row0_label_unverified: no admitted seed RMSE supplied for parent ' + rows[0]['program_sha256']])
        self.assertIsNone(without_evidence['row0_label_verified'])
        real = verify_episode(self.episodes[0], dict(expected, seed_rmse={rows[0]['program_sha256']: rows[0]['metrics']['rmse']}))
        self.assertTrue(real['verified'], real['reasons'])
        self.assertIs(real['row0_label_verified'], True)
        self.assertEqual(real['observations_sha256'], digest((self.episodes[0] / 'search/observations.json').read_bytes()))
        strict = verify_episode(self.episodes[0], dict(evaluator_sha256='9' * 64, resource_protocol=V2_RESOURCE))
        self.assertIn('evaluator_mismatch: ' + FIXTURE_EVALUATOR, strict['reasons'])

    def test_ledger_identity_is_bound_to_the_manifest_evaluator_and_pinned_protocol(self):
        ledger = self.ledger()
        self.assertEqual(ledger.identity['evaluator_sha256'], self.bound)
        self.assertEqual(ledger.identity['resource_protocol'], sl.RESOURCE_PROTOCOL)
        self.assertEqual(ledger.evaluator, self.root / 'bin/gp-native-v5')
        self.assertIsNone(ledger.resource_protocol_override())
        with self.assertRaises(ValueError):
            self.ledger(allow_resource_protocol=V2_RESOURCE)   # a different protocol is a different ledger identity
        with self.assertRaises(ValueError):
            self.ledger().sources(self.bound)
        # The on-disk binary must reproduce the manifest binding.
        self.manifest(FOREIGN, 'stale-manifest.json')
        with self.assertRaisesRegex(ValueError, 'differs from the manifest binding'):
            Ledger(self.root, 'other', 'fixture-run', 'fixture', 'stale-manifest.json', vocabulary=self.vocabulary)
        (self.root / 'stale-manifest.json').write_bytes(encoded(dict(schema='parallel.continuous_learning.v1', cpu_executable='bin/absent',
                                                                      bindings={'bin/absent': FOREIGN}, vocabulary_sha256=self.vocabulary)))
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            Ledger(self.root, 'successor', 'fixture-run', 'fixture', 'stale-manifest.json', vocabulary=self.vocabulary)
        unbound = Ledger(self.root, 'unbound', 'fixture-run', 'fixture', 'stale-manifest.json', vocabulary=self.vocabulary)
        self.assertFalse(unbound.evaluator_binding['binary_present'])
        self.assertIsNone(unbound.evaluator)
        with self.assertRaisesRegex(ValueError, 'does not bind'):
            (self.root / 'stale-manifest.json').write_bytes(encoded(dict(cpu_executable='bin/absent', bindings={}, vocabulary_sha256=self.vocabulary)))
            Ledger(self.root, 'unbound2', 'fixture-run', 'fixture', 'stale-manifest.json', vocabulary=self.vocabulary)
        # The manifest's vocabulary is the default identity; a mismatch with the episodes' identity is a verification reason.
        defaulted = Ledger(self.root, 'defaulted', 'fixture-run', 'fixture', 'manifest.json')
        self.assertEqual(defaulted.identity['vocabulary_sha256'], self.vocabulary)

    def test_resource_protocol_override_is_explicit_and_journaled(self):
        overridden = self.ledger('override', allow_resource_protocol=V2_RESOURCE)
        journal = overridden.resource_protocol_override()
        self.assertEqual((journal['resource_protocol'], journal['pinned']), (V2_RESOURCE, sl.RESOURCE_PROTOCOL))
        self.assertEqual(overridden.db.execute("SELECT count(*) FROM events WHERE status='resource_protocol_override'").fetchone()[0], 1)
        rows = json.loads((self.episodes[0] / 'search/observations.json').read_bytes())
        # Even under the override the evaluator stays bound to the manifest: the v2 rows are still refused.
        result = overridden.ingest_completed(self.episodes[0], generation=GENERATION, arm='neural', role='candidate', replicate=0,
                                             seed_origin='neural_policy', seed_rmse={rows[0]['program_sha256']: rows[0]['metrics']['rmse']})
        self.assertFalse(result['verified'])
        self.assertFalse(any('resource_protocol' in reason for reason in result['reasons']), result['reasons'])
        self.assertIn('evaluator_mismatch: ' + FIXTURE_EVALUATOR, result['reasons'])
        overridden.bundle(GENERATION)
        sources = json.loads((self.root / 'override/admitted-sources.json').read_bytes())
        self.assertEqual(sources['resource_protocol_override']['resource_protocol'], V2_RESOURCE)

    def test_row0_mismatch_reaches_the_series_and_trips_f5(self):
        """Ledger export/bundle -> flywheel_series.episode_rows -> flywheel_brake.decide: a bumped row 0 trips F5 immediately."""
        import flywheel_brake
        import flywheel_series
        from test_flywheel_series import baseline_digest
        ledger = self.ledger()
        generation = GENERATION + 1
        episodes, seed_rmse = self.verified_fixture(self.root / 'fixture/rebound')
        bumped, _, _ = self.verified_copy(self.episodes[0], self.root / 'fixture/bumped', 5000,
                                          lambda rows, receipt: rows[0]['metrics'].update(rmse=rows[0]['metrics']['rmse'] + 1e-9))
        self.ingest_all(ledger, episodes[:6] + [bumped], generation=generation, seed_rmse=seed_rmse)
        ledger.export(generation)
        bundle = ledger.bundle(generation)
        self.assertEqual(bundle['admitted'], 0)
        manifest = json.loads((self.root / 'manifest.json').read_bytes())
        rows = flywheel_series.episode_rows(self.root, manifest, self.root / 'successor')
        row = next(r for r in rows if r['generation'] == generation)
        self.assertEqual(row['successor']['row0_mismatches'], 1)
        self.assertEqual(row['successor']['row0_verified_rows'], 6)
        self.assertEqual(row['successor']['statuses'], dict(measured=7))
        self.assertEqual(row['evaluator_sha256'], self.bound)
        decision = flywheel_brake.decide(rows, baseline_digest(), None, 'm', 1)
        self.assertTrue(decision['held'])
        self.assertEqual(decision['tripped_by'], ['F5'])
        self.assertEqual(decision['rejects'][0]['row0_mismatches'], 1)
        self.assertTrue(flywheel_brake.pause(None, decision, None)['due'])
        # Without the mismatch the same chain is silent.
        clean = self.ledger('successor-clean')
        self.ingest_all(clean, episodes[:6], generation=generation, seed_rmse=seed_rmse)
        clean.export(generation)
        self.assertEqual(clean.bundle(generation)['admitted'], 1)
        rows = flywheel_series.episode_rows(self.root, manifest, self.root / 'successor-clean')
        self.assertEqual(rows[0]['successor']['row0_mismatches'], 0)
        self.assertFalse(flywheel_brake.decide(rows, baseline_digest(), None, 'm', 1)['held'])

    def test_next_outcome_joins_generation_g_seeds_to_generation_g_plus_one(self):
        ledger = self.ledger()
        self.ingest_all(ledger)
        prepared_sha, cut = self.identity['prepared_sha256'], self.identity['cut']
        report = self.root / 'run/generation-000074/quality-report.json'
        report.parent.mkdir(parents=True)
        rows = [dict(arm=arm, prepared_sha256=prepared_sha, cut=cut, dataset=self.identity['dataset'], partition='meta_train', generation=74,
                     best_initial_neural_seed=(0.7 if arm == 'neural' else None), best_initial_arm=0.7 if arm == 'neural' else 0.9, best_final_gp=0.6)
                for arm in sl.ARMS]
        report.write_bytes(encoded(dict(schema='parallel.continuous_quality.v1', jobs_sha256='j' * 64, rows=rows, final_tests_opened=False)))
        with self.assertRaisesRegex(ValueError, 'requires the generation 74 quality report'):
            ledger.next_outcome(GENERATION)
        produced = ledger.next_outcome(GENERATION, quality_report='run/generation-000074/quality-report.json')
        self.assertEqual(produced['schema'], 'parallel.successor_next_outcome.v1')
        self.assertEqual((produced['generation'], produced['next_generation'], produced['outcome']), (73, 74, 'best_initial'))
        self.assertEqual(produced['counts'], dict(seeds=61, joined=61, unjoined=0, contexts=1, joined_contexts=1))
        self.assertTrue(all(r['outcome'] == -0.7 and r['context_key'] == prepared_sha + ':0' for r in produced['rows']))
        self.assertEqual(len({r['seed_program_sha256'] for r in produced['rows']}), 61)
        self.assertEqual(produced['source']['kind'], 'quality_report')
        self.assertEqual(produced['source']['sha256'], sl.sha(report))
        self.assertIs(produced['final_tests_opened'], False)
        path = Path(produced['path'])
        self.assertEqual(path, self.root / 'successor/next-outcome-73.json')
        self.assertEqual(ledger.next_outcome(GENERATION, quality_report='run/generation-000074/quality-report.json')['sha256'], produced['sha256'])
        loaded = st.load_next_outcome(path)
        self.assertEqual(len(loaded), 61)
        self.assertEqual(set(loaded.values()), {-0.7})
        stale = json.loads(report.read_bytes())
        stale['rows'][0]['generation'] = 75
        (self.root / 'run/stale-report.json').write_bytes(encoded(stale))
        with self.assertRaisesRegex(ValueError, 'stamped generation'):
            ledger.next_outcome(GENERATION, quality_report='run/stale-report.json')
        # child_beats_incumbent: the seed's own generation-74 episode, recorded against an incumbent at ingest.
        episodes, seed_rmse = self.verified_fixture(self.root / 'fixture/rebound')
        self.ingest_all(ledger, episodes[:8], generation=74, seed_rmse=seed_rmse, incumbent_rmse=0.9)
        recursive = ledger.next_outcome(GENERATION, outcome='child_beats_incumbent')
        self.assertEqual(recursive['source']['kind'], 'ledger_seed_values')
        self.assertEqual(recursive['counts']['joined'], sum(1 for d in episodes[:8] if sl.rmse_of(json.loads((d / 'search/observations.json').read_bytes())[1]) is not None))
        self.assertTrue(all(r['outcome'] in (0.0, 1.0) for r in recursive['rows']))
        self.assertEqual(Path(recursive['path']), self.root / 'successor/next-outcome-73-child_beats_incumbent.json')
        self.assertEqual(len(st.load_next_outcome(recursive['path'])), recursive['counts']['joined'])
        with self.assertRaisesRegex(ValueError, 'outcome must be one of'):
            ledger.next_outcome(GENERATION, outcome='other')


class TierBTests(FixtureCase):
    """Tier-B (bounded_seed_lineage_v3) episodes synthesized from a real seed row: 24-fit lineages, violations, incompletes, overruns."""

    def tierb_episode(self, source, generation_dir, rng_seed, *, protocol=TIERB, descendants=23, timeout_at=7, violation_at=None,
                      termination='budget_exhausted', result=True, arm='neural', role='candidate', seed_origin='neural_policy',
                      generation=GENERATION, mutate=None):
        """Row 0 is the real seed row (rebound to the manifest evaluator and the cap's resource protocol); descendants are
        lineage members bred from the top-4 evaluated members (mutations, a crossover every third fit), one timeout."""
        budget = sl.tierb_budget(protocol)
        identity = json.loads((source / 'identity.json').read_bytes())
        identity.update(rng_seed=rng_seed, protocol=protocol, evaluator_sha256=self.bound, seed_arm=arm, role=role, replicate=0,
                        seed_origin=seed_origin, generation=generation, **budget)
        identity_bytes = encoded(identity)
        episode_id = digest(identity_bytes)
        template = json.loads((source / 'search/observations.json').read_bytes())[0]
        self.assertIsNotNone(sl.rmse_of(template))
        seed_rmse = template['metrics']['rmse']
        rows, evaluated = [dict(template)], [(seed_rmse, template['program_sha256'])]
        for index in range(1, descendants + 1):
            genome = '%s ;; lineage-%d' % (template['genome'], index)
            program = digest(genome.encode())
            pool = sorted(evaluated)[:sl.TIERB_PARENT_POOL]
            if index % 3 == 0 and len(pool) >= 2:
                parents, operator = [pool[0][1], pool[1][1]], 'crossover'
            else:
                parents, operator = [pool[index % len(pool)][1]], 'mutation'
            if violation_at == index:
                parents = [FOREIGN]
            row = dict(template, genome=genome, program_sha256=program, genetic_parents=parents, genetic_operator=operator,
                       initial_seed_slot=None, origin='gp_evolution')
            if index == timeout_at:
                row.update(failure='candidate_or_view_timeout', metrics=None)
            else:
                rmse = seed_rmse * (0.8 + 0.03 * ((index * 7) % 11))
                row['metrics'] = dict(template['metrics'], rmse=rmse)
                evaluated.append((rmse, program))
            rows.append(row)
        for index, row in enumerate(rows):
            row.update(rng_seed=rng_seed, continuation_sha256=episode_id, continuation_protocol=protocol, evaluator_sha256=self.bound,
                       resource_protocol=budget['resource_protocol'], evaluation_index=index)
        receipt = json.loads((source / 'search/search.json').read_bytes())
        if mutate:
            mutate(rows, receipt)
        for row in rows:
            row['evaluation_identity_sha256'] = digest(canonical({k: row[k] for k in IDENTITY_KEYS}))
        observations = encoded(rows)
        best = min(((r['metrics']['rmse'] * r['target_scale'], r['genome']) for r in rows if sl.rmse_of(r) is not None), default=(None, None))
        receipt.update(rng_seed=rng_seed, continuation_sha256=episode_id, continuation_protocol=protocol, evaluator_sha256=self.bound,
                       observations_sha256=digest(observations), completed_evaluations=len(rows), unique_programs=len(rows),
                       best_validation_rmse=best[0], best_program=best[1])
        dest = generation_dir / episode_id
        (dest / 'search').mkdir(parents=True)
        (dest / 'identity.json').write_bytes(identity_bytes)
        shutil.copy(source / 'seeds.json', dest / 'seeds.json')
        (dest / 'manifest.json').write_bytes(encoded(dict(schema='parallel.seed_experiment.v2', continuation_identity_sha256=episode_id, final_tests_opened=False)))
        (dest / 'search/observations.json').write_bytes(observations)
        (dest / 'search/search.json').write_bytes(encoded(receipt))
        if result:
            measured = sum(sl.rmse_of(r) is not None for r in rows[1:])
            (dest / 'result.json').write_bytes(encoded(dict(status='measured', attempted_fits=len(rows), attempted_descendants=len(rows) - 1,
                                                            successful_descendants=measured, termination_reason=termination,
                                                            elapsed_seconds=0.75 * len(rows), identity_sha256=episode_id, final_tests_opened=False)))
        return dest, identity, rows

    def seed_evidence(self, source):
        return sl.admitted_rmse(json.loads((source / 'search/observations.json').read_bytes())[:1])

    def test_valid_lineage_episode_is_verified_credited_and_bundled(self):
        ledger = self.ledger()
        source = self.episodes[0]
        rows0 = json.loads((source / 'search/observations.json').read_bytes())
        seed_rmse = self.seed_evidence(source)
        episode, identity, rows = self.tierb_episode(source, self.root / 'fixture/tierb', 7001)
        self.assertEqual((identity['protocol'], identity['evaluations'], identity['candidate_seconds'], identity['seconds'], identity['resource_protocol']),
                         (TIERB, 24, 2, 54, 'single_candidate_2s_8gib_v1'))
        self.assertEqual(len(rows), 24)
        self.assertEqual({r['genetic_operator'] for r in rows[1:]}, {'mutation', 'crossover'})
        self.assertEqual(sum(len(r['genetic_parents']) == 2 for r in rows[1:]), 7)
        self.assertEqual(sum(r['failure'] is not None for r in rows), 1)
        result = ledger.ingest_completed(episode, seed_rmse=seed_rmse)   # bindings come from the v3 identity (seed_arm/role/generation)
        self.assertTrue(result['verified'], result['reasons'])
        self.assertEqual((result['protocol'], result['role'], result['arm'], result['generation'], result['status']), (TIERB, 'candidate', 'neural', GENERATION, 'completed'))
        self.assertEqual((result['attempted_descendants'], result['successful_descendants'], result['termination_reason'], result['lineage_size']),
                         (23, 22, 'budget_exhausted', 23))
        finite = [sl.rmse_of(r) for r in rows if sl.rmse_of(r) is not None]
        self.assertEqual(len(finite), 23)
        self.assertEqual(result['S_rmse'], min(finite))                    # S_lineage over seed + descendants
        self.assertLess(result['S_rmse'], result['r0'])
        self.assertEqual(result['r0'], rows0[0]['metrics']['rmse'])       # F5: row 0 reproduces the admitted label
        self.assertIs(result['row0_label_verified'], True)
        self.assertEqual((result['measurement_status'], result['flip']), ('measured', 1))
        best = min((sl.rmse_of(r), i, r) for i, r in enumerate(rows[1:]) if sl.rmse_of(r) is not None)
        self.assertEqual((result['child']['program_sha256'], result['child']['rmse'], result['child']['failure']), (best[2]['program_sha256'], best[0], None))
        # the same seed's one-step episode of the same generation is a distinct seed value: protocol is part of the key
        onestep, _, _ = self.verified_copy(source, self.root / 'fixture/tierb-onestep', 7002)
        again = ledger.ingest_completed(onestep, generation=GENERATION, arm='neural', role='candidate', replicate=0, seed_origin='neural_policy', seed_rmse=seed_rmse)
        self.assertTrue(again['verified'], again['reasons'])
        self.assertEqual((again['protocol'], again['attempted_descendants'], again['successful_descendants'], again['termination_reason'], again['lineage_size']),
                         (ONESTEP, 1, 1, None, 2))
        self.assertEqual(again['S_rmse'], min(again['r0'], again['child']['rmse']))
        self.assertEqual(ledger.db.execute('SELECT count(*) FROM seed_values').fetchone()[0], 2)
        self.assertEqual(ledger.db.execute("SELECT count(*) FROM events WHERE status='seed_value_duplicate_role'").fetchone()[0], 0)
        self.assertEqual(sorted(r[0] for r in ledger.db.execute('SELECT protocol FROM seed_values')), [TIERB, ONESTEP])
        stored = ledger.db.execute('SELECT attempted_descendants,successful_descendants,termination_reason,lineage_size,S_rmse FROM seed_values WHERE protocol=?', (TIERB,)).fetchone()
        self.assertEqual(tuple(stored), (23, 22, 'budget_exhausted', 23, result['S_rmse']))
        # export rows carry the protocol fields; the one-step row keeps one-step semantics
        export = json.loads(Path(ledger.export(GENERATION)['path']).read_bytes())
        self.assertEqual(export['protocols'], [TIERB, ONESTEP])
        by_protocol = {r['protocol']: r for r in export['rows']}
        for key in ('protocol', 'attempted_descendants', 'successful_descendants', 'termination_reason', 'S_rmse', 'lineage_size'):
            self.assertIn(key, by_protocol[TIERB])
            self.assertIn(key, by_protocol[ONESTEP])
        self.assertEqual((by_protocol[TIERB]['attempted_descendants'], by_protocol[TIERB]['successful_descendants'], by_protocol[TIERB]['termination_reason'],
                          by_protocol[TIERB]['lineage_size'], by_protocol[TIERB]['status'], by_protocol[TIERB]['S_rmse']), (23, 22, 'budget_exhausted', 23, 'measured', result['S_rmse']))
        self.assertEqual((by_protocol[ONESTEP]['attempted_descendants'], by_protocol[ONESTEP]['successful_descendants'], by_protocol[ONESTEP]['termination_reason'],
                          by_protocol[ONESTEP]['lineage_size'], by_protocol[ONESTEP]['status']), (1, 1, None, 2, 'measured'))
        self.assertEqual(export['counts']['by_status'], dict(measured=2))
        self.assertEqual(export['counts']['by_protocol'][TIERB], dict(candidates=1, measured=1, incomplete=0, finite_children=1, better=1, tied=0, worse=0, flips=1))
        self.assertEqual(export['counts']['by_protocol'][ONESTEP], dict(candidates=1, measured=1, incomplete=0, finite_children=1, better=0, tied=0, worse=1, flips=0))
        self.assertEqual((export['counts']['better'], export['counts']['worse']), (0, 1))   # existing keys stay one-step
        info = ledger.info_report(GENERATION)
        self.assertEqual(set(info['by_protocol']), {TIERB, ONESTEP})
        self.assertEqual((info['pooled']['candidates'], info['by_protocol'][TIERB]['pooled']['candidates']), (1, 1))
        self.assertEqual(info['by_protocol'][TIERB]['credited_statuses'], ['measured', 'incomplete'])
        # the lineage rows enter the bundle as their own rows beside the one-step pair
        bundle = ledger.bundle(GENERATION)
        entry = bundle['bundles'][0]
        self.assertEqual((entry['verified'], entry['admitted'], entry['unverified']), (2, True, []))
        receipt = json.loads((self.root / 'successor/bundles/generation-73' / entry['context_key'] / 'search.json').read_bytes())
        self.assertEqual(receipt['completed_evaluations'], 26)
        self.assertEqual(receipt['bundle']['protocols'], [TIERB, ONESTEP])
        self.assertEqual({e['protocol']: e['rows'] for e in receipt['bundle']['episodes']}, {TIERB: 24, ONESTEP: 2})
        sources = json.loads((self.root / 'successor/admitted-sources.json').read_bytes())
        self.assertEqual(sources['sources'], [entry['path']])
        economics = ledger.economics(GENERATION)
        self.assertEqual((economics['attempted_fits'], economics['successful_fits'], economics['timeouts']), (26, 25, 1))

    def test_lineage_violation_is_excluded_from_bundles_and_credit(self):
        ledger = self.ledger()
        source = self.episodes[0]
        episode, _, rows = self.tierb_episode(source, self.root / 'fixture/tierb', 7003, violation_at=5)
        self.assertEqual(rows[5]['genetic_parents'], [FOREIGN])
        result = ledger.ingest_completed(episode, seed_rmse=self.seed_evidence(source))
        self.assertFalse(result['verified'])
        self.assertEqual(len(result['reasons']), 1)
        self.assertTrue(result['reasons'][0].startswith('lineage_violation: row 5 parents'), result['reasons'])
        self.assertIn(FOREIGN, result['reasons'][0])
        self.assertEqual(result['measurement_status'], 'measured')   # measured, never credited: excluded below
        bundle = ledger.bundle(GENERATION)
        entry = bundle['bundles'][0]
        self.assertEqual((entry['verified'], entry['admitted'], entry['reasons'], entry['path']), (0, False, ['lineage_violation'], None))
        self.assertEqual(entry['unverified'][0]['episode_id'], episode.name)
        sources = json.loads((self.root / 'successor/admitted-sources.json').read_bytes())
        self.assertEqual((sources['sources'], sources['excluded'][0]['reasons']), ([], ['lineage_violation']))
        export = json.loads(Path(ledger.export(GENERATION)['path']).read_bytes())
        self.assertIs(export['rows'][0]['verified'], False)
        self.assertEqual(export['rows'][0]['verification_reasons'][0].split(':')[0], 'lineage_violation')
        # an immigrant (no parents) and a seed row that is not the seed are violations too
        def immigrant(rows, receipt):
            rows[9]['genetic_parents'] = []
        stray, _, _ = self.tierb_episode(source, self.root / 'fixture/tierb', 7009, mutate=immigrant)
        self.assertEqual([r.split(':')[0] for r in ledger.ingest_completed(stray, seed_rmse=self.seed_evidence(source))['reasons']], ['lineage_violation'])

        def not_the_seed(rows, receipt):
            rows[0]['initial_seed_slot'] = 3
        wrong, _, _ = self.tierb_episode(source, self.root / 'fixture/tierb', 7010, mutate=not_the_seed)
        self.assertTrue(any(r.startswith('row0_not_the_seed') for r in ledger.ingest_completed(wrong, seed_rmse=self.seed_evidence(source))['reasons']))

    def test_incomplete_episode_is_retained_and_credited_for_what_it_measured(self):
        ledger = self.ledger()
        source = self.episodes[0]
        seed_rmse = self.seed_evidence(source)
        episode, _, rows = self.tierb_episode(source, self.root / 'fixture/tierb', 7004, descendants=8, timeout_at=None, termination='episode_timeout')
        result = ledger.ingest_completed(episode, seed_rmse=seed_rmse)
        self.assertTrue(result['verified'], result['reasons'])
        self.assertEqual((result['attempted_descendants'], result['successful_descendants'], result['termination_reason'], result['measurement_status'], result['status']),
                         (8, 8, 'episode_timeout', 'incomplete', 'completed'))
        self.assertEqual(result['S_rmse'], min(sl.rmse_of(r) for r in rows))
        # without result.json the termination is unknown: still verified, still incomplete (matched shuffled arm, same rule)
        unknown, _, _ = self.tierb_episode(source, self.root / 'fixture/tierb', 7005, descendants=8, timeout_at=None, result=False,
                                           arm='shuffled', role='control_shuffled')
        second = ledger.ingest_completed(unknown, seed_rmse=seed_rmse)
        self.assertTrue(second['verified'], second['reasons'])
        self.assertEqual((second['termination_reason'], second['measurement_status'], second['arm'], second['role']), ('unknown', 'incomplete', 'shuffled', 'control_shuffled'))
        export = json.loads(Path(ledger.export(GENERATION)['path']).read_bytes())
        self.assertEqual(export['counts']['by_status'], dict(incomplete=2))
        self.assertEqual(export['counts']['by_protocol'][TIERB]['incomplete'], 1)   # candidate role only
        self.assertEqual({r['termination_reason'] for r in export['rows']}, {'episode_timeout', 'unknown'})
        self.assertTrue(all(r['status'] == 'incomplete' and r['attempted_descendants'] == 8 for r in export['rows']))
        self.assertEqual(ledger.bundle(GENERATION)['admitted'], 1)   # retained: its measured rows are real fits

    def test_budget_overrun_and_cap_variant_drift_are_refused(self):
        ledger = self.ledger()
        source = self.episodes[0]
        seed_rmse = self.seed_evidence(source)
        overrun, _, _ = self.tierb_episode(source, self.root / 'fixture/tierb', 7006, descendants=24)
        result = ledger.ingest_completed(overrun, seed_rmse=seed_rmse)
        self.assertFalse(result['verified'])
        self.assertEqual(result['reasons'], ['budget_exceeded: 25 attempted fits > 24', 'termination_inconsistent: budget_exhausted with 25 attempted fits'])
        self.assertEqual(result['attempted_descendants'], 24)
        self.assertEqual(ledger.bundle(GENERATION)['admitted'], 0)

        def rows_2s(rows, receipt):
            for row in rows:
                row['resource_protocol'] = sl.RESOURCE_PROTOCOL
        drifted, identity, _ = self.tierb_episode(source, self.root / 'fixture/tierb', 7007, protocol=TIERB4, mutate=rows_2s)
        self.assertEqual((identity['candidate_seconds'], identity['seconds'], identity['resource_protocol']), (4, 102, 'single_candidate_4s_8gib_v1'))
        drift = ledger.ingest_completed(drifted, seed_rmse=seed_rmse)
        self.assertFalse(drift['verified'])
        self.assertEqual(len(drift['reasons']), 24)
        self.assertTrue(all(r.endswith('resource_protocol: ' + sl.RESOURCE_PROTOCOL) for r in drift['reasons']), drift['reasons'][:2])
        clean, _, _ = self.tierb_episode(source, self.root / 'fixture/tierb', 7008, protocol=TIERB4)
        self.assertTrue(ledger.ingest_completed(clean, seed_rmse=seed_rmse)['verified'])
        # an unknown protocol string is refused at the identity
        alien, _, _ = self.tierb_episode(source, self.root / 'fixture/tierb', 7011)
        alien_identity = json.loads((alien / 'identity.json').read_bytes())
        alien_identity['protocol'] = 'bounded_seed_lineage_v3'
        alien_dir = self.root / 'fixture/tierb-alien' / digest(encoded(alien_identity))
        shutil.copytree(alien, alien_dir)
        (alien_dir / 'identity.json').write_bytes(encoded(alien_identity))
        verification = verify_episode(alien_dir, dict(evaluator_sha256=self.bound, resource_protocol=sl.RESOURCE_PROTOCOL))
        self.assertIn('identity_protocol: bounded_seed_lineage_v3', verification['reasons'])


class PlanningTests(FixtureCase):
    def setUp(self):
        super().setUp()
        self.population = json.loads((self.root / 'fixture/all64/population.json').read_bytes())
        self.generic = json.loads((self.root / 'fixture/all64/generic.json').read_bytes())
        self.prepared_sha = digest((self.root / PREPARED_REL / 'prepared.json').read_bytes())
        self.assertEqual(self.population['prepared_sha256'], self.prepared_sha)
        generation = self.root / 'run/generation-000073'
        header = {k: self.population[k] for k in ('schema', 'prepared_sha256', 'cut', 'training_sha256', 'context_sha256', 'checkpoint_sha256')}
        arms = dict(neural=self.population,
                    generic=dict(header, candidates=[dict(genome=g, origin='generic_comparison') for g in self.generic[:80]]),
                    shuffled=dict(header, candidates=[dict(genome=g, origin='neural_policy') for g in self.generic[48:128]]))
        best = dict(neural=self.population['candidates'][0]['genome'], generic=self.generic[5], shuffled=self.generic[100])
        self.seeds, self.searches = {}, {}
        for arm, document in arms.items():
            directory = generation / arm / '55e9dfc7a86361d7/cut-0'
            (directory / 'search').mkdir(parents=True)
            (directory / 'seeds.json').write_bytes(encoded(document))
            (directory / 'search/search.json').write_bytes(encoded(dict(schema='parallel.search.v1', best_program=best[arm], prepared_sha256=self.prepared_sha,
                                                                        cut=0, final_tests_opened=False, evaluator_sha256=self.bound)))
            if arm == 'neural':
                # The arm's search ledger carries every seed's admitted row-0 RMSE (F5 evidence) except the last neural emission.
                neural_genomes = [c['genome'] for c in document['candidates'] if c['origin'] == 'neural_policy']
                self.last_emission = digest(neural_genomes[-1].encode())
                observations = [dict(origin=c['origin'], program_sha256=digest(c['genome'].encode()), genome=c['genome'], initial_seed_slot=i,
                                     failure=None, metrics=dict(rmse=0.5 + 0.001 * i, units='cut_normalized_target'))
                                for i, c in enumerate(document['candidates']) if c['genome'] != neural_genomes[-1]]
                (directory / 'search/observations.json').write_bytes(encoded(observations))
            self.seeds[arm] = str((directory / 'seeds.json').relative_to(self.root))
            self.searches[arm] = str((directory / 'search/search.json').relative_to(self.root))
        self.defaults = dict(excluded_families=self.identity['excluded_families'], rng_seed=20260912, parent_checkpoint='run/checkpoint',
                             parent_checkpoint_sha256=self.identity['producing_checkpoint_sha256'])

    def plan(self, ledger):
        return ledger.plan_generation(GENERATION, self.seeds['neural'], self.seeds['generic'], self.seeds['shuffled'], self.searches,
                                      PREPARED_REL, self.generic, self.defaults)

    def test_v2_plan_reproduces_the_fixture_identities_byte_for_byte(self):
        # The fixture was planned by seed-continuation-v2/continuation.py; Codex's seed-dispatch plan() later added
        # resource_protocol to the identity. With defaults, the extended plan() must match v2 modulo exactly that key.
        import importlib.util
        v2_path = PARALLEL / 'recovery/eight-a100-20260912/experiments/seed-continuation-v2/continuation.py'
        spec = importlib.util.spec_from_file_location('continuation_v2_lane', v2_path)
        v2 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(v2)
        prepared = json.loads((self.root / PREPARED_REL / 'prepared.json').read_bytes())
        kwargs = dict(prepared_sha=self.prepared_sha, population_sha=digest((self.root / 'fixture/all64/population.json').read_bytes()),
                      generic_sha=digest((self.root / 'fixture/all64/generic.json').read_bytes()), evaluator_sha=FIXTURE_EVALUATOR,
                      vocabulary_sha=self.vocabulary, excluded_families=self.identity['excluded_families'], rng_seed=20260912)
        original = v2.plan(self.population, prepared, self.generic, **kwargs)
        self.assertEqual({e['episode_id'] for e in original['episodes']}, {d.name for d in self.episodes})
        extended = sl.plan(self.population, prepared, self.generic, **kwargs)
        self.assertEqual(len(extended['episodes']), 64)
        for before, after in zip(original['episodes'], extended['episodes']):
            self.assertEqual(before['seeds'], after['seeds'])
            self.assertEqual(dict(before['identity'], resource_protocol='single_candidate_2s_8gib_v1'), after['identity'])
            self.assertEqual(after['identity']['protocol'], 'parallel.seed_continuation.qualification.v2')
            self.assertNotIn('seed_arm', after['identity'])
        self.assertEqual([(e['slot'], e['status']) for e in original['emissions']], [(e['slot'], e['status']) for e in extended['emissions']])

    def test_roles_deterministic_controls_and_replicate_panel(self):
        ledger = self.ledger()
        plan = self.plan(ledger)
        self.assertEqual(plan['schema'], 'parallel.successor_plan.v1')
        self.assertEqual(plan['counts'], dict(candidate=64, control_generic=8, control_shuffled=8, gp_final=3, replicate_panel=8))
        self.assertEqual(plan['inserted'], 91)
        self.assertEqual(plan['context_key'], self.prepared_sha + ':0')
        self.assertEqual(plan['evaluator_sha256'], self.bound)
        by_role = {}
        for episode in plan['episodes']:
            by_role.setdefault(episode['role'], []).append(episode)
        candidates = {e['parent_program_sha256'] for e in by_role['candidate']}
        controls = {e['parent_program_sha256'] for role in ('control_generic', 'control_shuffled') for e in by_role[role]}
        self.assertEqual(len(candidates), 64)
        self.assertEqual(candidates & controls, set())
        for role in ('control_generic', 'control_shuffled'):
            self.assertEqual(len({e['parent_program_sha256'] for e in by_role[role]}), 8)
        self.assertTrue({e['parent_program_sha256'] for e in by_role['replicate_panel']} <= candidates)
        self.assertEqual({e['episode_id'] for e in by_role['candidate']} & {d.name for d in self.episodes}, set())   # v3 identity differs from v2
        pool = [g for g in dict.fromkeys(self.generic[:80]) if digest(g.encode()) not in candidates]
        expected = sorted(pool, key=lambda g: selection_key(self.prepared_sha, 0, GENERATION, 'generic', digest(g.encode())))[:8]
        self.assertEqual([digest(g.encode()) for g in expected], plan['selection']['control_generic']['chosen'])
        self.assertEqual({e['parent_program_sha256'] for e in by_role['control_generic']}, set(plan['selection']['control_generic']['chosen']))
        for episode in plan['episodes']:
            identity = json.loads((self.root / 'successor/episodes' / episode['episode_id'] / 'identity.json').read_bytes())
            self.assertEqual(digest(encoded(identity)), episode['episode_id'])
            self.assertEqual(identity['protocol'], 'parallel.seed_continuation.qualification.v3')
            self.assertEqual(identity['evaluator_sha256'], self.bound)
            self.assertEqual((identity['seed_arm'], identity['role'], identity['replicate'], identity['seed_origin'], identity['generation']),
                             (episode['arm'], episode['role'], episode['replicate'], episode['seed_origin'], GENERATION))
            self.assertEqual((identity['evaluations'], identity['candidate_seconds'], identity['seconds'], identity['resource_protocol']),
                             (2, 2, 6, 'single_candidate_2s_8gib_v1'))
            seeds = json.loads((self.root / 'successor/episodes' / episode['episode_id'] / 'seeds.json').read_bytes())
            self.assertEqual(len(seeds['candidates']), 80)
            self.assertEqual(seeds['candidates'][0]['origin'], episode['seed_origin'])
            if episode['role'] == 'replicate_panel':
                self.assertEqual(identity['rng_seed'], replicate_rng(plan['context_key'], GENERATION, episode['parent_program_sha256'], 1))
            else:
                self.assertEqual(identity['rng_seed'], 20260912)
        self.assertEqual({e['seed_origin'] for e in by_role['control_generic']}, {'generic_comparison'})
        self.assertEqual({e['seed_origin'] for e in by_role['control_shuffled']}, {'neural_policy'})
        self.assertEqual({e['arm'] for e in by_role['gp_final']}, set(sl.ARMS))
        neural_final = next(e for e in by_role['gp_final'] if e['arm'] == 'neural')
        self.assertIn(neural_final['parent_program_sha256'], candidates)   # search best == an emission: distinct role, distinct episode
        self.assertNotIn(neural_final['episode_id'], {e['episode_id'] for e in by_role['candidate']})
        # F5 evidence travels with the plan: neural seeds present in the arm's search ledger carry their admitted RMSE;
        # the last emission (absent from the ledger) and the arms without a ledger do not, and the plan says so.
        evidence = plan['row0_evidence']
        self.assertEqual(evidence['sources'], dict(neural='run/generation-000073/neural/55e9dfc7a86361d7/cut-0/search/observations.json'))
        payloads = {r['id']: json.loads(r['payload']) for r in ledger.db.execute('SELECT id, payload FROM episodes')}
        with_evidence = [e for e in plan['episodes'] if payloads[e['episode_id']]['extra'].get('admitted_rmse') is not None]
        self.assertEqual(evidence['episodes_with_evidence'], len(with_evidence))
        self.assertEqual(evidence['episodes_with_evidence'] + evidence['episodes_without_evidence'], 91)
        self.assertTrue(all(e['arm'] == 'neural' for e in with_evidence))
        self.assertTrue(any(e['role'] == 'gp_final' for e in with_evidence))
        self.assertTrue(any(e['role'] == 'replicate_panel' for e in with_evidence))
        self.assertTrue(any(e['parent_program_sha256'] == self.last_emission and e['role'] == 'candidate' for e in plan['episodes']))
        self.assertFalse(any(e['parent_program_sha256'] == self.last_emission for e in with_evidence))
        self.assertEqual([s for s in plan['skipped'] if s['reason'] == 'row0_evidence_absent'][0]['episodes'], evidence['episodes_without_evidence'])
        jobs = ledger.pending_bundle()['jobs']
        self.assertEqual(len(jobs), 91)
        roles = {e['episode_id']: e['priority'] for e in plan['episodes']}
        priorities = [roles[j['continuation']['identity_sha256']] for j in jobs]
        self.assertEqual(priorities, sorted(priorities))
        self.assertEqual(jobs[0]['prepared'], PREPARED_REL)
        self.assertEqual(jobs[0]['output'], 'successor/episodes/' + jobs[0]['continuation']['identity_sha256'] + '/search')
        self.assertEqual(ledger.db.execute('SELECT count(*) FROM emissions').fetchone()[0], 80)
        again = self.plan(ledger)
        self.assertEqual(again['inserted'], 0)
        self.assertEqual(again['receipt_sha256'], plan['receipt_sha256'])
        other = self.plan(self.ledger('successor-b'))
        self.assertEqual([e['episode_id'] for e in other['episodes']], [e['episode_id'] for e in plan['episodes']])
        self.assertEqual(other['selection'], plan['selection'])

    def test_plan_from_jobs_plans_a_generation_from_its_jobs_file(self):
        prepared_doc = json.loads((self.root / PREPARED_REL / 'prepared.json').read_bytes())
        partition = prepared_doc['meta_partition']
        jobs = []
        for arm in sl.ARMS:
            jobs.append(dict(prepared=PREPARED_REL, prepared_sha256=self.prepared_sha, cut=0, output=str(Path(self.searches[arm]).parent),
                             seeds=self.seeds[arm], seeds_sha256=digest((self.root / self.seeds[arm]).read_bytes()), generation=GENERATION,
                             rng_seed=20260912, parent_checkpoint='run/checkpoint', parent_checkpoint_sha256=self.identity['producing_checkpoint_sha256']))
        # a second context whose preparation is not clean: listed as inadmissible, never planned, never in the subset
        unclean = dict(prepared_doc, clean_evaluation=False, dataset='unclean-copy')
        unclean_dir = self.root / 'data/prepared/views-expanded-20/unclean00000000'
        unclean_dir.mkdir(parents=True)
        (unclean_dir / 'prepared.json').write_bytes(encoded(unclean))
        unclean_sha = digest(encoded(unclean))
        jobs.append(dict(jobs[0], prepared='data/prepared/views-expanded-20/unclean00000000', prepared_sha256=unclean_sha,
                         seeds='run/generation-000073/neural/unclean00000000/cut-0/seeds.json', output='run/generation-000073/neural/unclean00000000/cut-0/search'))
        jobs_path = self.root / 'run/generation-000073/jobs.json'
        jobs_path.write_bytes(encoded(dict(schema='parallel.gp_jobs.v1', qualification=False, vocabulary='vocab', jobs=jobs)))
        excluded = self.identity['excluded_families']
        # one-step: byte-identical to the explicit plan_generation call
        ledger = self.ledger()
        summary = ledger.plan_from_jobs(GENERATION, 'run/generation-000073/jobs.json', excluded, partition=partition)
        self.assertEqual(summary['schema'], 'parallel.successor_plan_summary.v1')
        self.assertEqual(summary['contexts'], dict(total=2, admissible=1, visited=1, planned=1, by_partition={partition: 2}))
        self.assertEqual(summary['inadmissible'], [dict(context_key=unclean_sha + ':0', dataset='unclean-copy', family=prepared_doc['family'],
                                                        reasons=['clean_evaluation is not true'])])
        self.assertEqual(summary['counts'], dict(candidate=64, control_generic=8, control_shuffled=8, gp_final=3, replicate_panel=8))
        self.assertEqual((summary['episodes_inserted'], summary['episodes_planned'], summary['protocol'], summary['tierb']), (91, 91, sl.PROTOCOL_V3, None))
        self.assertEqual(summary['inserted_by_context'], {self.prepared_sha + ':0': 91})
        # the background is the generic arm's generic_comparison seeds (Codex's qualify_dispatch rule), not the fixture's wider list
        explicit = self.ledger('successor-explicit').plan_generation(GENERATION, self.seeds['neural'], self.seeds['generic'], self.seeds['shuffled'],
                                                                     self.searches, PREPARED_REL, list(dict.fromkeys(self.generic[:80])), self.defaults)
        self.assertEqual(summary['plans'][0]['receipt_sha256'], explicit['receipt_sha256'])
        self.assertEqual(summary['plans'][0]['searches'], ['generic', 'neural', 'shuffled'])
        self.assertEqual(summary['jobs']['sha256'], digest(jobs_path.read_bytes()))
        summary_path = self.root / 'successor/plans/generation-73/summary.onestep.json'
        self.assertEqual(digest(summary_path.read_bytes()), summary['summary_sha256'])
        again = ledger.plan_from_jobs(GENERATION, 'run/generation-000073/jobs.json', excluded, partition=partition)
        self.assertEqual((again['episodes_inserted'], again['episodes_planned'], again['summary_sha256']), (0, 91, summary['summary_sha256']))   # idempotent
        self.assertEqual(digest(summary_path.read_bytes()), summary['summary_sha256'])
        self.assertNotIn('episodes_inserted', json.loads(summary_path.read_bytes()))   # the immutable summary carries no per-call counter
        with self.assertRaisesRegex(ValueError, 'Immutable artifact changed'):   # one summary per (generation, protocol) and ledger
            ledger.plan_from_jobs(GENERATION, 'run/generation-000073/jobs.json', excluded, partition='no_such_partition')
        empty = self.ledger('successor-empty').plan_from_jobs(GENERATION, 'run/generation-000073/jobs.json', excluded, partition='no_such_partition')
        self.assertEqual((empty['contexts']['admissible'], empty['contexts']['planned'], empty['episodes_inserted']), (0, 0, 0))
        self.assertEqual([i['reasons'] for i in empty['inadmissible']],
                         [["partition 'meta_train' is not no_such_partition"], ["partition 'meta_train' is not no_such_partition", 'clean_evaluation is not true']])
        # an excluded family is inadmissible too, and a refusal inside plan_generation is a recorded skip
        family_excluded = self.ledger('successor-family').plan_from_jobs(GENERATION, 'run/generation-000073/jobs.json', excluded + [prepared_doc['family']], partition=partition)
        self.assertEqual((family_excluded['contexts']['admissible'], family_excluded['contexts']['planned']), (0, 0))
        self.assertIn("family %r is excluded" % prepared_doc['family'], family_excluded['inadmissible'][0]['reasons'])
        with self.assertRaisesRegex(ValueError, 'generation 72 is not 73'):
            bad = dict(jobs[0], generation=72)
            (self.root / 'run/generation-000073/jobs-bad.json').write_bytes(encoded(dict(jobs=[bad])))
            ledger.plan_from_jobs(GENERATION, 'run/generation-000073/jobs-bad.json', excluded, partition=partition)
        with self.assertRaisesRegex(ValueError, 'second neural job'):
            (self.root / 'run/generation-000073/jobs-dup.json').write_bytes(encoded(dict(jobs=[jobs[0], jobs[0]])))
            ledger.plan_from_jobs(GENERATION, 'run/generation-000073/jobs-dup.json', excluded, partition=partition)
        # Tier-B: every arm needs admitted row-0 evidence for its top-3; the single context is inside any subset of <= 32
        for arm in ('generic', 'shuffled'):
            directory = self.root / 'run/generation-000073' / arm / '55e9dfc7a86361d7/cut-0'
            doc = json.loads((directory / 'seeds.json').read_bytes())
            observations = [dict(origin=c['origin'], program_sha256=digest(c['genome'].encode()), genome=c['genome'], initial_seed_slot=i,
                                 failure=None, metrics=dict(rmse=0.9 - 0.002 * i, units='cut_normalized_target')) for i, c in enumerate(doc['candidates'])]
            (directory / 'search/observations.json').write_bytes(encoded(observations))
        tierb = self.ledger('successor-tierb').plan_from_jobs(GENERATION, 'run/generation-000073/jobs.json', excluded, protocol=TIERB, partition=partition, limit=5)
        self.assertEqual(tierb['counts'], dict(candidate=3, control_generic=3, control_shuffled=3))
        self.assertEqual((tierb['episodes_inserted'], tierb['protocol']), (9, TIERB))
        self.assertEqual(tierb['tierb'], dict(subset=[self.prepared_sha + ':0'], subset_size=32, universe=1,
                                              universe_sha256=digest(encoded([self.prepared_sha + ':0'])), limit=5))
        self.assertTrue((self.root / ('successor-tierb/plans/generation-73/summary.%s.json' % TIERB)).is_file())
        self.assertTrue((self.root / ('successor-tierb/plans/generation-73/%s:0.%s.json' % (self.prepared_sha, TIERB))).is_file())
        for episode in self.ledger('successor-tierb').pending_bundle()['jobs']:
            identity = json.loads((self.root / episode['continuation']['identity']).read_bytes())
            self.assertEqual((identity['protocol'], identity['evaluations'], identity['generation']), (TIERB, 24, GENERATION))
            self.assertIn(identity['seed_arm'], sl.ARMS)

    def test_tierb_plan_top3_per_arm_on_the_rotating_subset(self):
        # every arm's search ledger carries admitted row-0 RMSE so each arm's top-3 seeds can be ranked
        for arm in ('generic', 'shuffled'):
            directory = self.root / 'run/generation-000073' / arm / '55e9dfc7a86361d7/cut-0'
            doc = json.loads((directory / 'seeds.json').read_bytes())
            observations = [dict(origin=c['origin'], program_sha256=digest(c['genome'].encode()), genome=c['genome'], initial_seed_slot=i,
                                 failure=None, metrics=dict(rmse=0.9 - 0.002 * i, units='cut_normalized_target')) for i, c in enumerate(doc['candidates'])]
            (directory / 'search/observations.json').write_bytes(encoded(observations))
        context_key = self.prepared_sha + ':0'

        def tierb_plan(ledger, protocol, universe, defaults=None):
            return ledger.plan_generation(GENERATION, self.seeds['neural'], self.seeds['generic'], self.seeds['shuffled'], self.searches,
                                          PREPARED_REL, self.generic, defaults or self.defaults, protocol=protocol, tierb_contexts=universe)
        ledger = self.ledger()
        with self.assertRaisesRegex(ValueError, 'tierb_contexts'):
            tierb_plan(ledger, TIERB, None)
        with self.assertRaisesRegex(ValueError, 'protocol must be'):
            tierb_plan(ledger, 'bounded_seed_lineage_v3', [context_key])
        with self.assertRaisesRegex(ValueError, 'include ' + context_key):
            tierb_plan(ledger, TIERB, ['%064x:0' % 1])
        # rotation rule: the 32 smallest sha256(prepared:cut:g:'tierb'); a universe of 32 lower-ranked contexts excludes this one
        ours = sl.tierb_context_rank(self.prepared_sha, 0, GENERATION)
        below, index = [], 1
        while len(below) < 32:
            key = '%064x:0' % index
            if sl.tierb_context_rank(key[:64], 0, GENERATION) < ours:
                below.append(key)
            index += 1
        subset = sl.tierb_subset([context_key] + below, GENERATION)
        self.assertEqual(len(subset), 32)
        self.assertNotIn(context_key, subset)
        self.assertEqual(subset, sorted(subset, key=lambda k: sl.tierb_context_rank(k[:64], 0, GENERATION)))
        excluded = tierb_plan(self.ledger('successor-excluded'), TIERB, [context_key] + below)
        self.assertFalse(excluded['tierb']['selected'])
        self.assertEqual((excluded['inserted'], excluded['counts'], excluded['episodes'], excluded['tierb']['subset']), (0, {}, [], subset))
        self.assertEqual(excluded['skipped'][0]['reason'], 'tierb_subset_excluded')
        self.assertEqual((excluded['tierb']['universe'], excluded['tierb']['subset_size'], excluded['tierb']['top_seeds']), (33, 32, 3))
        # a universe of 21 contexts: every context is in the subset; the plan holds the top-3 seeds of each arm
        universe = [context_key] + below[:20]
        plan = tierb_plan(ledger, TIERB, universe)
        self.assertEqual(plan['schema'], 'parallel.successor_plan.v1')
        self.assertTrue(plan['tierb']['selected'])
        self.assertEqual(plan['tierb']['subset'], sl.tierb_subset(universe, GENERATION))
        self.assertEqual(plan['tierb']['universe'], 21)
        self.assertIn('smallest sha256(prepared_sha256:cut:generation:', plan['tierb']['rotation_rule'])
        self.assertEqual(plan['tierb']['budget'], dict(evaluations=24, candidate_seconds=2, seconds=54, resource_protocol='single_candidate_2s_8gib_v1'))
        self.assertEqual((plan['tierb']['base'], plan['tierb']['parent_pool'], plan['tierb']['terminations']),
                         ('bounded_seed_lineage_v3', 4, ['budget_exhausted', 'episode_timeout', 'seed_failed', 'lineage_exhausted']))
        self.assertEqual(plan['counts'], dict(candidate=3, control_generic=3, control_shuffled=3))
        self.assertEqual(plan['inserted'], 9)
        self.assertEqual((plan['protocol'], plan['credit_protocol'], plan['resource_protocol']), (TIERB, TIERB, 'single_candidate_2s_8gib_v1'))
        self.assertEqual(plan['row0_evidence']['episodes_without_evidence'], 0)
        by_role = {}
        for episode in plan['episodes']:
            by_role.setdefault(episode['role'], []).append(episode)
        for arm, role, population in (('neural', 'candidate', self.population),
                                      ('generic', 'control_generic', json.loads((self.root / self.seeds['generic']).read_bytes())),
                                      ('shuffled', 'control_shuffled', json.loads((self.root / self.seeds['shuffled']).read_bytes()))):
            observations = self.root / 'run/generation-000073' / arm / '55e9dfc7a86361d7/cut-0/search/observations.json'
            admitted = sl.admitted_rmse(json.loads(observations.read_bytes()))
            pool = sl.seed_genomes(population, sl.SEED_ORIGIN_BY_ARM[arm])
            expected = [sha for _, sha in sorted((admitted[digest(g.encode())], digest(g.encode())) for g in pool if digest(g.encode()) in admitted)[:3]]
            self.assertEqual(plan['selection'][role]['chosen'], expected)                           # rank order
            self.assertEqual(sorted(e['parent_program_sha256'] for e in by_role[role]), sorted(expected))   # episodes follow slot order
            self.assertEqual({e['arm'] for e in by_role[role]}, {arm})
        payloads = {r['id']: json.loads(r['payload']) for r in ledger.db.execute('SELECT id, payload FROM episodes')}
        for episode in plan['episodes']:
            identity = json.loads((self.root / 'successor/episodes' / episode['episode_id'] / 'identity.json').read_bytes())
            self.assertEqual(digest(encoded(identity)), episode['episode_id'])
            self.assertEqual((identity['protocol'], identity['evaluations'], identity['candidate_seconds'], identity['seconds'], identity['resource_protocol']),
                             (TIERB, 24, 2, 54, 'single_candidate_2s_8gib_v1'))
            self.assertEqual((identity['rng_seed'], identity['seed_arm'], identity['role'], identity['replicate'], identity['generation'], identity['evaluator_sha256']),
                             (20260912, episode['arm'], episode['role'], 0, GENERATION, self.bound))
            seeds = json.loads((self.root / 'successor/episodes' / episode['episode_id'] / 'seeds.json').read_bytes())
            self.assertEqual((len(seeds['candidates']), seeds['candidates'][0]['origin']), (80, episode['seed_origin']))
            self.assertIsNotNone(payloads[episode['episode_id']]['extra']['admitted_rmse'])
        self.assertTrue((self.root / 'successor/plans/generation-73' / (context_key + '.' + TIERB + '.json')).is_file())
        again = tierb_plan(ledger, TIERB, universe)
        self.assertEqual((again['inserted'], again['receipt_sha256']), (0, plan['receipt_sha256']))
        # the cap-4s variant rebinds exactly candidate_seconds / seconds / resource_protocol and gets its own identities
        four = tierb_plan(ledger, TIERB4, universe)
        self.assertEqual((four['counts'], four['inserted']), (plan['counts'], 9))
        self.assertEqual({e['episode_id'] for e in four['episodes']} & {e['episode_id'] for e in plan['episodes']}, set())
        self.assertEqual([e['parent_program_sha256'] for e in four['episodes']], [e['parent_program_sha256'] for e in plan['episodes']])
        identity4 = json.loads((self.root / 'successor/episodes' / four['episodes'][0]['episode_id'] / 'identity.json').read_bytes())
        identity2 = json.loads((self.root / 'successor/episodes' / plan['episodes'][0]['episode_id'] / 'identity.json').read_bytes())
        self.assertEqual((identity4['protocol'], identity4['evaluations'], identity4['candidate_seconds'], identity4['seconds'], identity4['resource_protocol']),
                         (TIERB4, 24, 4, 102, 'single_candidate_4s_8gib_v1'))
        self.assertEqual({k: v for k, v in identity4.items() if k not in ('protocol', 'candidate_seconds', 'seconds', 'resource_protocol')},
                         {k: v for k, v in identity2.items() if k not in ('protocol', 'candidate_seconds', 'seconds', 'resource_protocol')})
        # the one-step plan coexists in the same ledger with its own receipt and 91 more pending episodes
        onestep = self.plan(ledger)
        self.assertEqual(onestep['inserted'], 91)
        self.assertEqual(ledger.db.execute("SELECT count(*) FROM episodes WHERE status='pending'").fetchone()[0], 109)
        self.assertEqual(ledger.db.execute("SELECT count(*) FROM episodes WHERE status='pending' AND role='candidate'").fetchone()[0], 70)
        with self.assertRaisesRegex(ValueError, 'contradict'):
            tierb_plan(self.ledger('successor-contradict'), TIERB, universe, dict(self.defaults, evaluations=2))

    def test_supersede_is_terminal(self):
        ledger = self.ledger()
        plan = self.plan(ledger)
        with self.assertRaises(ValueError):
            ledger.supersede(GENERATION, GENERATION + 1)
        receipt = ledger.supersede(GENERATION, GENERATION + 2)
        self.assertEqual(receipt['count'], 91)
        statuses = {r[0] for r in ledger.db.execute('SELECT status FROM episodes')}
        self.assertEqual(statuses, {'superseded'})
        first = plan['episodes'][0]['episode_id']
        ledger.transition(first, 'running')
        self.assertEqual(ledger.db.execute('SELECT status FROM episodes WHERE id=?', (first,)).fetchone()[0], 'superseded')
        self.assertEqual(ledger.db.execute("SELECT count(*) FROM events WHERE status='transition_refused_superseded'").fetchone()[0], 1)
        ledger.reconcile(now=0, process_live=lambda _: True)
        self.assertEqual({r[0] for r in ledger.db.execute('SELECT status FROM episodes')}, {'superseded'})
        self.assertEqual(ledger.pending_bundle()['jobs'], [])
        refused = ledger.ingest_completed('successor/episodes/' + first)
        self.assertFalse(refused['ingested'])
        export = ledger.export(GENERATION)
        self.assertEqual(export['counts']['by_status'], dict(superseded=91))
        self.assertEqual(ledger.supersede(GENERATION, GENERATION + 3)['count'], 0)
        economics = ledger.economics(GENERATION)
        self.assertEqual(economics['superseded'], 91)
        self.assertIsNone(economics['reserved_core_seconds'])
        self.assertIsNone(economics['unique_fits_per_core_second'])

    def test_export_refuses_open_episodes(self):
        ledger = self.ledger()
        self.plan(ledger)
        with self.assertRaisesRegex(ValueError, 'terminal'):
            ledger.export(GENERATION)


class ContinuationEvaluatorTests(FixtureCase):
    """A dual-identity manifest binds the ordinary evaluator (cpu_executable) and a continuation evaluator named by the
    seed driver's admission configuration; continuation episodes verify against the latter, never an operator digest."""

    def dual_manifest(self, name='dual-manifest.json', configuration=None, declared_sha=True):
        (self.root / 'bin/jope-seed-dispatch').write_bytes(b'continuation evaluator fixture bytes')
        continuation = sl.sha(self.root / 'bin/jope-seed-dispatch')
        config = configuration if configuration is not None else dict(evaluator='bin/jope-seed-dispatch', evaluator_sha256=continuation)
        (self.root / 'admission-config.json').write_bytes(encoded(config))
        admission = dict(configuration='admission-config.json')
        if declared_sha:
            admission['configuration_sha256'] = sl.sha(self.root / 'admission-config.json')
        (self.root / name).write_bytes(encoded(dict(
            schema='parallel.continuous_learning.v1', cpu_executable='bin/gp-native-v5', vocabulary_sha256=self.vocabulary, output='fixture',
            bindings={'bin/gp-native-v5': self.bound, 'bin/jope-seed-dispatch': continuation}, continuation_source_admission=admission)))
        return name, continuation

    def continuation_copy(self, source, generation_dir, rng_seed, continuation):
        def rewrite(rows, receipt):
            for row in rows:
                row['evaluator_sha256'] = continuation
                row['resource_protocol'] = sl.RESOURCE_PROTOCOL
            receipt['evaluator_sha256'] = continuation
        return self.derive(source, generation_dir, rng_seed, rewrite, identity_updates=dict(evaluator_sha256=continuation))

    def test_continuation_evaluator_is_derived_from_the_manifest_admission_configuration(self):
        manifest, continuation = self.dual_manifest()
        self.assertNotEqual(continuation, self.bound)
        ledger = Ledger(self.root, 'dual', 'fixture-run', 'fixture', manifest, vocabulary=self.vocabulary)
        self.assertEqual(ledger.identity['evaluator_sha256'], self.bound)
        self.assertEqual(ledger.identity['continuation_evaluator_sha256'], continuation)
        self.assertEqual(ledger.continuation_evaluator_sha256, continuation)
        source = ledger.evaluator_binding['continuation_evaluator_source']
        self.assertEqual((source['kind'], source['evaluator'], source['binary_present']), ('continuation_source_admission', 'bin/jope-seed-dispatch', True))
        # An episode evaluated by the continuation binary verifies (row 0 bit-equal to its admitted RMSE); one carrying the
        # ordinary evaluator is a continuation episode too and is refused against the continuation binding.
        dest, _, rows = self.continuation_copy(self.episodes[0], self.root / 'fixture/continuation', 7000, continuation)
        result = ledger.ingest_completed(dest, generation=GENERATION, arm='neural', role='candidate', replicate=0, seed_origin='neural_policy',
                                         seed_rmse=sl.admitted_rmse(rows[:1]))
        self.assertTrue(result['verified'], result['reasons'])
        self.assertTrue(result['row0_label_verified'])
        ordinary, _, rows = self.verified_copy(self.episodes[1], self.root / 'fixture/ordinary', 7001)
        result = ledger.ingest_completed(ordinary, generation=GENERATION, arm='neural', role='candidate', replicate=0, seed_origin='neural_policy',
                                         seed_rmse=sl.admitted_rmse(rows[:1]))
        self.assertFalse(result['verified'])
        self.assertIn('identity_evaluator_mismatch: ' + self.bound, result['reasons'])
        self.assertIn('evaluator_mismatch: ' + self.bound, result['reasons'])
        export = ledger.export(GENERATION)
        document = json.loads(Path(export['path']).read_bytes())
        self.assertEqual((document['evaluator_sha256'], document['continuation_evaluator_sha256']), (self.bound, continuation))
        self.assertEqual({r['verified'] for r in document['rows']}, {True, False})
        # verify_episode itself: without a continuation binding the ordinary evaluator is expected of continuation rows.
        expected = dict(evaluator_sha256=self.bound, resource_protocol=sl.RESOURCE_PROTOCOL, protocols=sl.PROTOCOLS, partition='meta_train')
        self.assertEqual(verify_episode(ordinary, expected)['expected_evaluator_sha256'], self.bound)
        self.assertEqual(verify_episode(ordinary, dict(expected, continuation_evaluator_sha256=continuation))['expected_evaluator_sha256'], continuation)
        # The explicit digest must equal the derived binding; without a derivable one it must be manifest-bound; a
        # single-evaluator manifest keeps its identity bytes (no continuation key) and its default binding.
        with self.assertRaisesRegex(ValueError, 'contradicts the manifest-derived'):
            Ledger(self.root, 'dual-explicit', 'fixture-run', 'fixture', manifest, vocabulary=self.vocabulary, continuation_evaluator=self.bound)
        self.assertEqual(Ledger(self.root, 'dual-explicit', 'fixture-run', 'fixture', manifest, vocabulary=self.vocabulary,
                                continuation_evaluator=continuation).identity['continuation_evaluator_sha256'], continuation)
        with self.assertRaisesRegex(ValueError, 'not a manifest-bound digest'):
            self.ledger('explicit-foreign', continuation_evaluator=FOREIGN)
        with self.assertRaisesRegex(ValueError, '64-hex'):
            self.ledger('explicit-short', continuation_evaluator='abc')
        explicit = self.ledger('explicit-bound', continuation_evaluator=self.bound)
        self.assertNotIn('continuation_evaluator_sha256', explicit.identity)
        self.assertEqual(explicit.evaluator_binding['continuation_evaluator_source']['kind'], 'explicit')
        plain = self.ledger()
        self.assertNotIn('continuation_evaluator_sha256', plain.identity)
        self.assertEqual(plain.continuation_evaluator_sha256, self.bound)
        self.assertEqual(plain.evaluator_binding['continuation_evaluator_source']['kind'], 'cpu_executable')
        # Fail closed: an admission configuration whose evaluator the manifest does not bind, a configuration whose bytes
        # drifted from the declared digest, and a continuation binary that differs from its binding are all refused.
        unbound, _ = self.dual_manifest('unbound-manifest.json', configuration=dict(evaluator='bin/jope-seed-dispatch', evaluator_sha256=FOREIGN))
        with self.assertRaisesRegex(ValueError, 'not bound by the manifest'):
            Ledger(self.root, 'unbound', 'fixture-run', 'fixture', unbound, vocabulary=self.vocabulary)
        drifted, _ = self.dual_manifest('drifted-manifest.json')
        (self.root / 'admission-config.json').write_bytes(encoded(dict(evaluator='bin/jope-seed-dispatch', evaluator_sha256=continuation, note='drift')))
        with self.assertRaisesRegex(ValueError, 'configuration differs from the manifest binding'):
            Ledger(self.root, 'drifted', 'fixture-run', 'fixture', drifted, vocabulary=self.vocabulary)
        stale, _ = self.dual_manifest('stale-binary-manifest.json')
        (self.root / 'bin/jope-seed-dispatch').write_bytes(b'rebuilt continuation evaluator')
        with self.assertRaisesRegex(ValueError, 'continuation evaluator on disk differs'):
            Ledger(self.root, 'stale-binary', 'fixture-run', 'fixture', stale, vocabulary=self.vocabulary)


if __name__ == '__main__':
    unittest.main()
