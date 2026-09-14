import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import original_query_selection as selection


def h(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


class OriginalQuerySelectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.index = dict(schema='parallel.context_index.v1', dimensions=2048, qualification=True,
                          assembly_sha256=h('assembly'), encodings=[])
        self.inventory = dict(schema='parallel.family_contamination_inventory.v1',
            prepared=[], known_conflicts=[], unresolved_memberships=2,
            final_tests_opened=False, csv_or_target_values_opened=False)
        for number in range(2):
            family, dataset, prepared = f'family-{number}', f'dataset-{number}', f'prepared/{number}'
            cut = dict(index=0, training_sha256=h(f'train-{number}'),
                       validation_sha256=h(f'validation-{number}'), preprocessing_sha256=h(f'pp-{number}'))
            self.inventory['prepared'].append(dict(family=family, dataset=dataset, prepared=prepared,
                prepared_sha256=h(prepared), source_sha256=h(f'source-{number}'), partition='meta_train', cuts=[cut]))
            self.index['encodings'].append(dict(family=family, dataset=dataset, prepared=prepared,
                prepared_sha256=h(prepared), partition='meta_train', cut=0,
                sha256=h(f'context-{number}'), receipt_sha256=h(f'encoding-receipt-{number}')))
        code = Path(selection.__file__).read_bytes()
        files = {selection.PRODUCER_PATH: dict(sha256=hashlib.sha256(code).hexdigest(), bytes=len(code), executable=False)}
        release = dict(schema='parallel.registry_source_freeze.v1', files=files,
                       source_closure_sha256=selection.digest(files))
        self.request = dict(schema=selection.REQUEST_SCHEMA, experiment_id='original-panel',
            scope='training-evaluation', role='training_diagnostic', source_release=self.write('release.json', release),
            context_index=None, family_inventory=None,
            evaluation=dict(evaluator_sha256=h('evaluator'), runtime_sha256=h('runtime'),
                numerical_contract_sha256=h('numerics'), vocabulary_sha256=h('vocab'),
                query_contract=dict(selection.QUERY_CONTRACT)),
            arms=[dict(name=name, parent_checkpoint_sha256=h(name),
                       original_parent_receipt=dict(path=f'parents/{name}.json', sha256=h(f'parent-{name}')))
                  for name in ('challenger', 'recovery')], rng_seeds=[101, 202],
            contexts=[dict(prepared_sha256=row['prepared_sha256'], cut=0) for row in self.index['encodings']],
            programs_per_pool=2, pools=[])
        self.populations = {}
        for number, context in enumerate(self.index['encodings']):
            for arm in self.request['arms']:
                for seed in self.request['rng_seeds']:
                    name = f'pools/{number}-{arm["name"]}-{seed}.json'
                    population = dict(schema='parallel.seed_population.v1', prepared_sha256=context['prepared_sha256'],
                        cut=0, training_sha256=self.inventory['prepared'][number]['cuts'][0]['training_sha256'],
                        context_sha256=context['sha256'], checkpoint_sha256=arm['parent_checkpoint_sha256'],
                        candidates=[dict(genome=f'(pipeline-{i})', origin='neural_policy', rank_log_weight=-i)
                                    for i in range(3)])
                    self.populations[name] = population
                    self.request['pools'].append(dict(prepared_sha256=context['prepared_sha256'], cut=0,
                        arm=arm['name'], rng_seed=seed, population=self.write(name, population),
                        original_proposal_receipt=dict(path=f'receipts/{number}-{arm["name"]}-{seed}.json',
                                                       sha256=h(name + 'receipt'))))
        self.refresh_indices()

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        data = selection.canonical(value) + b'\n'
        path.write_bytes(data)
        return dict(path=name, sha256=hashlib.sha256(data).hexdigest())

    def refresh_indices(self):
        self.request['context_index'] = self.write('contexts.json', self.index)
        self.inventory['context_index_sha256'] = self.request['context_index']['sha256']
        self.request['family_inventory'] = self.write('families.json', self.inventory)

    def freeze(self, request=None, output='selection.json'):
        return selection.freeze(self.root, self.write('request.json', request or self.request), output)

    def first_pool(self):
        return self.request['pools'][0]

    def replace_population(self, value):
        pool = self.first_pool()
        pool['population'] = self.write(pool['population']['path'], value)

    def test_real_immutable_artifact_rederives_complete_grid_without_access_claim(self):
        result = self.freeze()
        self.assertEqual(result['candidates'], 16)
        self.assertFalse(result['query_access_allowed'])
        verified = selection.verify(self.root, result['selection'])
        self.assertEqual(verified['status'], 'metadata_selection_verified')
        self.assertFalse(verified['original_receipt_execution_verified'])
        self.assertFalse(verified['independent_publication_order_verified'])
        body = json.loads((self.root / 'selection.json').read_text())
        self.assertEqual(len({row['selection_key'] for row in body['candidates']}), 16)
        self.assertEqual({row['rng_seed'] for row in body['candidates']}, {101, 202})
        self.assertEqual({row['arm'] for row in body['candidates']}, {'challenger', 'recovery'})
        self.assertTrue(all(row['origin'] == 'neural_policy' for row in body['candidates']))

    def test_public_cli_freeze_and_verify_produce_receipt(self):
        request = self.write('request.json', self.request)
        script = str(Path(selection.__file__))
        command = [sys.executable, '-B', script, 'freeze', '--root', str(self.root),
                   '--input', request['path'], '--sha256', request['sha256'], '--output', 'cli.json']
        made = subprocess.run(command, text=True, capture_output=True, timeout=15)
        self.assertEqual(made.returncode, 0, made.stderr)
        ref = json.loads(made.stdout)['selection']
        checked = subprocess.run([sys.executable, '-B', script, 'verify', '--root', str(self.root),
            '--input', ref['path'], '--sha256', ref['sha256']], text=True, capture_output=True, timeout=15)
        self.assertEqual(checked.returncode, 0, checked.stderr)
        self.assertFalse(json.loads(checked.stdout)['query_access_allowed'])

    def test_receipt_pointers_and_payloads_are_never_opened(self):
        opened = []
        real = selection.Metadata.read
        def observe(store, ref):
            opened.append(ref['path'])
            if ref['path'].startswith(('parents/', 'receipts/')):
                self.fail('original outcome/receipt pointer was opened')
            return real(store, ref)
        with patch.object(selection.Metadata, 'read', observe):
            result = self.freeze()
        self.assertTrue(opened)
        body = json.loads((self.root / result['selection']['path']).read_text())
        self.assertEqual(body['candidates'][0]['original_proposal_receipt']['sha256'],
                         next(pool['original_proposal_receipt']['sha256'] for pool in self.request['pools']
                              if pool['population'] == body['candidates'][0]['population']))

    def test_changed_source_and_self_rehashed_selection_cannot_verify(self):
        result = self.freeze()
        body = json.loads((self.root / 'selection.json').read_text())
        body['candidates'][0]['origin'] = 'retained_cut_teacher'
        changed = self.write('altered.json', body)
        with self.assertRaisesRegex(ValueError, 'differs from original metadata'):
            selection.verify(self.root, changed)
        pool = self.first_pool()
        self.write(pool['population']['path'], dict(self.populations[pool['population']['path']], cut=1))
        with self.assertRaisesRegex(ValueError, 'original metadata bytes changed'):
            selection.verify(self.root, result['selection'])

    def test_caller_verification_flags_and_outcome_fields_rejected_before_output(self):
        for place in ('request', 'candidate'):
            with self.subTest(place=place):
                request = copy.deepcopy(self.request)
                if place == 'request':
                    request['independent_evaluation'] = True
                else:
                    row = copy.deepcopy(self.populations[self.first_pool()['population']['path']])
                    row['candidates'][0]['rmse'] = .01
                    request['pools'][0]['population'] = self.write('bad-pool.json', row)
                with self.assertRaises(ValueError):
                    self.freeze(request)
                self.assertFalse((self.root / 'selection.json').exists())

    def test_wrong_partition_rejected_before_population_reads(self):
        self.inventory['prepared'][0]['partition'] = 'meta_test'
        self.index['encodings'][0]['partition'] = 'meta_test'
        self.refresh_indices()
        real = selection.Metadata.read
        def observe(store, ref):
            self.assertFalse(ref['path'].startswith('pools/'))
            return real(store, ref)
        with patch.object(selection.Metadata, 'read', observe):
            with self.assertRaisesRegex(ValueError, 'sealed or wrong-partition'):
                self.freeze()
        self.assertFalse((self.root / 'selection.json').exists())

    def test_boolean_aliases_and_incomplete_or_duplicate_grid_rejected(self):
        changes = [lambda r: r['rng_seeds'].__setitem__(0, True),
                   lambda r: r['contexts'][0].__setitem__('cut', False),
                   lambda r: r['pools'][0].__setitem__('cut', False),
                   lambda r: r.__setitem__('programs_per_pool', True),
                   lambda r: r['pools'].pop(),
                   lambda r: r['pools'].__setitem__(0, copy.deepcopy(r['pools'][1]))]
        for change in changes:
            request = copy.deepcopy(self.request)
            change(request)
            with self.assertRaises(ValueError):
                self.freeze(request)
            self.assertFalse((self.root / 'selection.json').exists())

    def test_cross_context_parent_and_duplicate_program_origins_rejected(self):
        for field, value in (('prepared_sha256', h('other')), ('checkpoint_sha256', h('other')),
                             ('context_sha256', h('other')), ('cut', False)):
            row = copy.deepcopy(self.populations[self.first_pool()['population']['path']])
            row[field] = value
            self.replace_population(row)
            with self.assertRaises(ValueError):
                self.freeze()
        row = copy.deepcopy(self.populations[self.first_pool()['population']['path']])
        row['candidates'][1] = dict(row['candidates'][0], origin='teacher')
        self.replace_population(row)
        with self.assertRaisesRegex(ValueError, 'duplicate program or conflicting origin'):
            self.freeze()

    def test_rank_weight_alias_is_diagnostic_never_a_selection_propensity(self):
        row = copy.deepcopy(self.populations[self.first_pool()['population']['path']])
        row['candidates'][0]['policy_logprob'] = row['candidates'][0].pop('rank_log_weight')
        self.replace_population(row)
        result = self.freeze()
        body = json.loads((self.root / result['selection']['path']).read_text())
        self.assertTrue(all('policy_logprob' not in r and 'rank_log_weight' not in r for r in body['candidates']))
        row['candidates'][0]['rank_log_weight'] = 0
        self.replace_population(row)
        with self.assertRaisesRegex(ValueError, 'conflicting aliases'):
            self.freeze(output='alias.json')

    def test_family_source_alias_and_fold_conflict_rejected(self):
        self.inventory['prepared'][1]['source_sha256'] = self.inventory['prepared'][0]['source_sha256']
        self.refresh_indices()
        with self.assertRaisesRegex(ValueError, 'family/source crosses'):
            self.freeze()
        self.inventory['prepared'][1]['source_sha256'] = h('source-1')
        self.inventory['prepared'][0]['cuts'].append(copy.deepcopy(self.inventory['prepared'][0]['cuts'][0]))
        self.refresh_indices()
        with self.assertRaisesRegex(ValueError, 'ambiguous original fold'):
            self.freeze()

    def test_source_release_must_bind_actual_producer_bytes(self):
        release = json.loads((self.root / 'release.json').read_text())
        release['files'][selection.PRODUCER_PATH]['sha256'] = h('different producer')
        release['source_closure_sha256'] = selection.digest(release['files'])
        self.request['source_release'] = self.write('release.json', release)
        with self.assertRaisesRegex(ValueError, 'producer differs'):
            self.freeze()

    def test_forbidden_paths_symlinks_and_fifo_fail_without_payload_reads(self):
        for name in ('data/targets.json', 'raw/rows.json', 'dev-sealed/outcomes.json',
                     'development_validation/labels.json', '../outside.json', '/tmp/absolute.json'):
            request = copy.deepcopy(self.request)
            request['pools'][0]['population']['path'] = name
            with self.assertRaises(ValueError):
                self.freeze(request)
        target = self.root / 'fifo.json'
        os.mkfifo(target)
        request = copy.deepcopy(self.request)
        request['pools'][0]['population'] = dict(path='fifo.json', sha256=h('fifo'))
        with self.assertRaisesRegex(ValueError, 'regular metadata'):
            self.freeze(request)
        (self.root / 'link.json').symlink_to(self.root / self.first_pool()['population']['path'])
        request['pools'][0]['population']['path'] = 'link.json'
        with self.assertRaises(OSError):
            self.freeze(request)

    def test_changed_after_capture_fails_before_publication(self):
        request_ref = self.write('request.json', self.request)
        real = selection.Metadata.publish
        def change(store, name, body):
            path = self.root / self.first_pool()['population']['path']
            path.write_bytes(path.read_bytes() + b' ')
            return real(store, name, body)
        with patch.object(selection.Metadata, 'publish', change):
            with self.assertRaisesRegex(ValueError, 'changed before selection publication'):
                selection.freeze(self.root, request_ref, 'selection.json')
        self.assertFalse((self.root / 'selection.json').exists())

    def test_existing_history_cannot_be_replaced_or_claim_witnessed(self):
        result = self.freeze()
        path = self.root / 'selection.json'
        original = path.read_bytes()
        with self.assertRaises(FileExistsError):
            self.freeze()
        self.assertEqual(path.read_bytes(), original)
        body = json.loads(original)
        body['local_publication']['independent_clock_verified'] = True
        with self.assertRaisesRegex(ValueError, 'clock observation'):
            selection.verify(self.root, self.write('clock.json', body))
        body['local_publication']['independent_clock_verified'] = False
        body['local_publication']['observed_unix_ns'] = 2**64 - 1
        with self.assertRaisesRegex(ValueError, 'clock observation'):
            selection.verify(self.root, self.write('future-clock.json', body))
        self.assertFalse(selection.verify(self.root, result['selection'])['independent_publication_order_verified'])

    def test_duplicate_keys_nonfinite_and_size_limits_reject(self):
        for raw in (b'{"schema":1,"schema":2}', b'{"value":NaN}', b'{"value":1e999}'):
            (self.root / 'bad.json').write_bytes(raw)
            ref = dict(path='bad.json', sha256=hashlib.sha256(raw).hexdigest())
            store = selection.Metadata(self.root)
            try:
                with self.assertRaises(ValueError):
                    store.read(ref)
            finally:
                store.close()
        with patch.object(selection, 'MAX_FILE', 16):
            with self.assertRaisesRegex(ValueError, 'bounded regular metadata'):
                self.freeze()


if __name__ == '__main__':
    unittest.main()
