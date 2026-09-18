import tempfile
import unittest
from pathlib import Path

from qualify_regression_gain import learning_inputs, monitor_reference, verify_trial_inputs
from resume_expired_cycle import atomic, sha


class LearningAdmissionTests(unittest.TestCase):
    def test_monitor_reference_matches_executable_corpus_and_panel_membership(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'corpus').mkdir();(root/'monitoring').mkdir()
            (root/'evaluator').write_bytes(b'qualified');atomic(root/'corpus/index.json',{'corpus':1})
            atomic(root/'panel.json',{'panel_sha256':'frozen-membership'})
            value=dict(executable_sha256=sha(root/'evaluator'),panel_sha256='frozen-membership',
                       corpus_sha256=sha(root/'corpus/index.json'))
            path=root/'monitoring/recovery.json';atomic(path,dict(binding=value))
            m=dict(monitor_executable='evaluator',monitor_corpus='corpus',monitor_panel='panel.json',
                   bindings={'monitoring/recovery.json':sha(path)})
            self.assertEqual(monitor_reference(root,m),path)
            atomic(path,dict(binding=dict(value,executable_sha256='old-executable')))
            m['bindings']['monitoring/recovery.json']=sha(path)
            with self.assertRaisesRegex(RuntimeError,'one bound recovery'):
                monitor_reference(root,m)

    def test_changed_nested_policy_or_executable_rejected_before_dispatch(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);p=root/'policy.json';atomic(p,{'sampling_seed':1})
            trial={'policies':{'fresh':{'path':'policy.json','sha256':sha(p)}}}
            verify_trial_inputs(root,trial)
            atomic(p,{'sampling_seed':2})
            with self.assertRaisesRegex(RuntimeError,'frozen qualification input changed'):
                verify_trial_inputs(root,trial)

    def test_legacy_labels_wrong_evaluator_and_contaminated_families_block_trials(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);corpus=root/'corpus';corpus.mkdir()
            evaluator=root/'evaluator';evaluator.write_bytes(b'frozen evaluator')
            index=dict(schema='parallel.teaching_shards.v1',label_policy='strict_midrank_v4',
                context_index_sha256='contexts',admission_policy=dict(schema='parallel.label_admission.v2',
                    evaluator='evaluator',evaluator_sha256=sha(evaluator),
                    evaluator_protocol='featmath_training_fold_normalized_v1',runtime_sha256='a'*64,
                    numerical_contract_sha256='b'*64,authority_instance='c'*64,
                    authority_root='authority',authority_witness='witness'))
            inventory=dict(schema='parallel.family_contamination_inventory.v1',context_index_sha256='contexts',
                           known_conflicts=[],unresolved_memberships=3,final_tests_opened=False)
            inventory_path=root/'inventory.json';atomic(inventory_path,inventory)
            atomic(corpus/'index.json',dict(index,label_policy='reference_v1'))
            with self.assertRaisesRegex(RuntimeError,'strictly admitted'):learning_inputs(root,corpus,inventory_path)
            atomic(corpus/'index.json',index)
            self.assertFalse(learning_inputs(root,corpus,inventory_path)['independent_evaluation'])
            evaluator.write_bytes(b'changed evaluator')
            with self.assertRaisesRegex(RuntimeError,'evaluator changed'):learning_inputs(root,corpus,inventory_path)
            evaluator.write_bytes(b'frozen evaluator')
            for change,reason in [(dict(context_index_sha256='different cut identity'),'identity changed'),
                (dict(known_conflicts=['overlapping families']),'contamination')]:
                atomic(inventory_path,dict(inventory,**change))
                with self.assertRaisesRegex(RuntimeError,reason):learning_inputs(root,corpus,inventory_path)
