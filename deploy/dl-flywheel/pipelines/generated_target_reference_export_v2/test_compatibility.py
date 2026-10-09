"""New synthetic compatibility boundaries; prior test methods are not run."""
import copy
import json
import unittest
from ..generated_target_admission_v1.common import ExportError, canonical, sha
from ..generated_target_admission_v1.tests import test_export as fixtures
from ..generated_target_admission_v1.tests import fixture_callbacks as cb
from ..generated_target_admission_v1.transition import validate_extension_transition
from .export_verified import export_targets
from .publication_reference import publication_identity


class CompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ExportTests(methodName='runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def annotate(self, task, key):
        task[key] = {**task[key], 'state': 'created',
                     'verification': 'exact VersionId GET + SHA256'}

    def run_export(self, f):
        req = canonical(f['request'])
        return json.loads(export_targets(req, expected_request_sha256=sha(req),
            cutoff_raw=f['cutoff'], predecessor_raw=f['predecessor'],
            objective_raw=canonical(self.fixture.objective),
            normalizer_raw=canonical(self.fixture.normalizer),
            get_version=cb.read_version, canonicalize=cb.canonicalize,
            verify=cb.verify, authority=cb.authority))

    def test_actual_seven_field_receipts_preserve_all_old_fields_and_refs(self):
        x = self.fixture
        old_ref, _ = x.result(x.train, 'ridge alpha=2')
        x.train['program_results'].append(old_ref)
        for task in x.tasks:
            self.annotate(task, 'encoder_publication')
        self.annotate(x.train, 'dataset_publication')
        x.emission(x.train)
        f = x.build(); original = f['predecessor']; before = json.loads(original)
        result = self.run_export(f)
        self.assertEqual(f['predecessor'], original)
        self.assertEqual(result['counts']['appended'], 1)
        for a, b in zip(before['tasks'], result['extension']['tasks']):
            for key in a:
                if key != 'program_results': self.assertEqual(a[key], b[key])
            self.assertEqual(b['program_results'][:len(a['program_results'])], a['program_results'])
        self.assertFalse(result['numeric_targets_materialized'])
        self.assertFalse(result['admission_committed'])
        self.assertEqual(result['optimizer_updates'], 0)
        self.assertTrue(validate_extension_transition(before, result['extension']))

    def test_frozen_v1_still_refuses_actual_annotated_receipt(self):
        x = self.fixture; self.annotate(x.train, 'encoder_publication'); x.emission(x.train)
        f = x.build()
        with self.assertRaises(ExportError): x.run_export(f)
        self.assertEqual(self.run_export(f)['counts']['appended'], 1)

    def test_unknown_or_partial_publication_annotations_refused(self):
        ref = self.fixture.train['encoder_publication']
        for extra in ({'state': 'created'}, {'verification': 'exact VersionId GET + SHA256'},
                      {'state': 'created', 'verification': 'exact VersionId GET + SHA256', 'authority': True},
                      {'extra': 'arbitrary'}):
            with self.subTest(extra=extra), self.assertRaises(ExportError):
                publication_identity({**ref, **extra})

    def test_failed_unknown_or_unverified_annotation_refused(self):
        ref = self.fixture.train['encoder_publication']
        for state, verification in [('failed', 'exact VersionId GET + SHA256'),
                                    ('unknown', 'exact VersionId GET + SHA256'),
                                    ('created', 'HEAD only'), ('created', 'exact VersionId GET + SHA256 '),
                                    (True, 'exact VersionId GET + SHA256')]:
            with self.subTest(state=state, verification=verification), self.assertRaises(ExportError):
                publication_identity({**ref, 'state': state, 'verification': verification})

    def test_normalized_descriptor_exact_identity_mismatch_refused(self):
        x = self.fixture; self.annotate(x.train, 'encoder_publication'); x.emission(x.train)
        for key, value in [('version_id', 'different-version'), ('sha256', '0'*64), ('bytes', 999)]:
            f = x.build(); cid = fixtures.digest(x.train['context'])
            f['request']['admitted_descriptors'][cid]['encoder_publication'][key] = value
            with self.subTest(key=key), self.assertRaises(ExportError): self.run_export(f)

    def test_validation_annotation_never_creates_training_target(self):
        x = self.fixture
        self.annotate(x.dev, 'dataset_publication'); self.annotate(x.dev, 'encoder_publication')
        x.emission(x.dev)
        result = self.run_export(x.build())
        self.assertEqual(result['counts']['validation'], 1)
        self.assertEqual(result['counts']['appended'], 0)
        self.assertEqual(result['artifact_reads']['calls'], 0)
        self.assertEqual(result['extension']['tasks'], x.tasks)

    def test_annotated_program_result_ref_remains_forbidden(self):
        x = self.fixture; ref, _ = x.result(x.train)
        x.train['program_results'].append({**ref, 'state': 'created', 'verification': 'exact VersionId GET + SHA256'})
        with self.assertRaises(ExportError): self.run_export(x.build())

    def test_transition_does_not_normalize_or_replace_old_annotations(self):
        x = self.fixture; self.annotate(x.train, 'encoder_publication')
        old = json.loads(x.build()['predecessor']); new = copy.deepcopy(old)
        new['tasks'][0]['encoder_publication'] = publication_identity(new['tasks'][0]['encoder_publication'])
        with self.assertRaises(ExportError): validate_extension_transition(old, new)
