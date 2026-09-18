import copy
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import candidate_intake
import continuous_learning as controller
from promotion_gate import artifact, sha
from publish_replay_candidate import publish


class IntakeTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)

    def put(self,path,value):
        (self.root/path).parent.mkdir(parents=True,exist_ok=True)
        controller.atomic(self.root/path,value)
        return dict(path=path,sha256=sha(self.root/path))

    def fixture(self):
        corpus=self.put('corpus/index.json',dict(label_policy='strict_midrank_v4'))
        state=dict(corpus_sha256=corpus['sha256'],vocabulary_sha256='vocab',step=1000,parameter_sha256='identical')
        for name,key in [('model.ot','model_sha256'),('optimizer.ot','optimizer_sha256'),
                         ('optimizer.json','optimizer_metadata_sha256'),('config.json','config_sha256')]:
            state[key]=self.put('checkpoint/'+name,{'fixture':True})['sha256']
        checkpoint=self.put('checkpoint/state.json',state)
        policy=self.put('policy.json',dict(fresh_fraction=.5))
        executable=dict(path='helper',sha256='helper-sha')
        trial=self.put('trial.json',dict(schema='parallel.regression_gain_trial.v1',final_tests_opened=False,
            corpus_index=corpus,policies={'fresh_reset':policy},steps_per_variant=1000,
            recovery=dict(path='recovery/state.json',sha256='recovery-sha'),
            proposal_executable=executable,training_executable=executable))
        ranks=[self.put(f'rank-{rank}.json',dict(rank=rank,world=8,corpus_sha256=corpus['sha256'],
            steps=1000,local_steps=1000,replay_policy_sha256=policy['sha256'],
            parameter_sha256='identical',actual_fresh_comparisons_this_rank_attempt=4,
            actual_fresh_programs_this_rank_attempt=5)) for rank in range(8)]
        training=self.put('training.json',dict(status='trained',checkpoint='checkpoint',
            checkpoint_sha256=checkpoint['sha256'],rank_receipts=ranks))
        result=self.put('result.json',dict(trial_sha256=trial['sha256'],eligible_challengers=['fresh_reset'],
            results={'fresh_reset':dict(checkpoint_sha256=checkpoint['sha256'],diagnostic_pass=True)}))
        request=self.put('nomination-request.json',dict(trial=trial,training_result=result,arms={'fresh_reset':checkpoint}))
        nomination=self.put('nomination.json',dict(status='nominated',winner='fresh_reset',
            checkpoint_sha256=checkpoint['sha256'],training_only=True,validation_results_used=False,
            final_tests_opened=False,request_sha256=request['sha256']))
        run=self.put('run.json',dict(trial_sha256=trial['sha256'],policy=policy,corpus=corpus,executable=executable,steps=1000))
        return dict(schema='parallel.candidate_submission.v1',trial=trial,training=training,run=run,
            training_result=result,nomination_request=request,nomination=nomination,
            checkpoint=checkpoint,corpus=corpus,arm='fresh_reset')

    def test_cross_trial_missing_exposure_and_tampered_receipts_reject(self):
        submission=self.fixture()
        self.assertEqual(candidate_intake.validate(self.root,submission,'vocab')['checkpoint'],'checkpoint')
        for mutation in ('trial','fresh','tamper','duplicate_rank','wrong_corpus','wrong_world','wrong_parameters'):
            changed=copy.deepcopy(submission)
            if mutation=='trial':
                row=artifact(self.root,submission['nomination_request']);row['trial']['sha256']='other'
                changed['nomination_request']=self.put('different-request.json',row)
            elif mutation in ('fresh','duplicate_rank','wrong_corpus','wrong_world','wrong_parameters'):
                training=artifact(self.root,submission['training'])
                refs=[]
                for i,ref in enumerate(training['rank_receipts']):
                    rank=artifact(self.root,ref)
                    if mutation=='fresh':rank['actual_fresh_comparisons_this_rank_attempt']=0
                    elif mutation=='duplicate_rank':rank['rank']=0
                    elif mutation=='wrong_corpus':rank['corpus_sha256']='another-corpus'
                    elif mutation=='wrong_world':rank['world']=1
                    else:rank['parameter_sha256']='another-model'
                    refs.append(self.put(f'empty-rank-{i}.json',rank))
                training['rank_receipts']=refs;changed['training']=self.put('empty-training.json',training)
            else:changed['training']['sha256']='tampered'
            with self.assertRaises(RuntimeError):candidate_intake.validate(self.root,changed,'vocab')

    def test_intake_recovers_after_publication_and_consumes_corpus_without_promoting(self):
        submission=self.fixture();self.put('candidate-inbox/input.json',submission)
        s=controller.Supervisor.__new__(controller.Supervisor)
        s.root=s.output=self.root;s.manifest_sha='controller-manifest'
        s.state_path=self.root/'state.json';s.journal=self.root/'journal.jsonl'
        s.m=dict(vocabulary_sha256='vocab',contexts_sha256='contexts')
        s.s=dict(candidates=[],candidate_submissions={},trained_snapshots=[],consumed_training_snapshots=[],
            checkpoint='serving',serving_checkpoint='serving',serving_checkpoint_sha256='serving-sha')
        s.save('fixture')
        with patch('continuous_learning.verify_corpus'):
            with patch.object(s,'save',side_effect=OSError('interrupted state commit')):
                with self.assertRaises(OSError):s.intake_candidates()
            s.s=controller.read(s.state_path);s.intake_candidates()
            s.s['candidates'][0]['qualification_status']='rejected';s.save('qualification_rejected')
            s.s=controller.read(s.state_path);s.intake_candidates()
        self.assertEqual(s.s['checkpoint'],'serving')
        self.assertEqual(s.s['serving_checkpoint_sha256'],'serving-sha')
        self.assertEqual(s.s['consumed_training_snapshots'],['corpus'])
        self.assertEqual(len(s.s['candidates']),1)
        self.assertEqual(s.s['candidates'][0]['qualification_status'],'rejected')

    def test_publication_is_idempotent_and_does_not_change_serving_state(self):
        submission=self.fixture();run=self.root/'fresh_reset';run.mkdir()
        for ref,name in [(submission['training'],'qualification-training.json'),(submission['run'],'run.json')]:
            shutil.copyfile(self.root/ref['path'],run/name)
        result=self.put('training-policy-result.json',artifact(self.root,submission['training_result']))
        request=artifact(self.root,submission['nomination_request']);request['training_result']=result
        request_ref=self.put('request.json',request)
        nomination=artifact(self.root,submission['nomination']);nomination['request_sha256']=request_ref['sha256']
        self.put('nomination.json',nomination)
        self.put('controller.json',dict(output='production',vocabulary_sha256='vocab'))
        serving=self.put('production/state.json',dict(serving_checkpoint='recovery',consumed_training_snapshots=[]))
        path=publish(self.root,self.root/'controller.json',self.root/'trial.json',self.root/'nomination.json')
        before=path.read_bytes()
        self.assertEqual(publish(self.root,self.root/'controller.json',self.root/'trial.json',self.root/'nomination.json'),path)
        self.assertEqual(path.read_bytes(),before)
        self.assertEqual(sha(self.root/serving['path']),serving['sha256'])
