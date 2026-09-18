"""Publish a nominated candidate for intake; serving still requires the full gate."""
import argparse
from pathlib import Path

import candidate_intake
from promotion_gate import immutable
from qualify_regression_gain import ref
from resume_expired_cycle import read, require


def publish(root,manifest,trial_path,nomination_path):
    active=read(manifest);trial=read(trial_path);nomination=read(nomination_path)
    require(nomination['status']=='nominated','there is no nominated challenger to publish')
    arm=nomination['winner'];run=trial_path.parent/arm
    training=read(run/'qualification-training.json')
    submission=dict(schema='parallel.candidate_submission.v1',arm=arm,
        trial=ref(root,trial_path),training=ref(root,run/'qualification-training.json'),
        run=ref(root,run/'run.json'),training_result=ref(root,trial_path.parent/'training-policy-result.json'),
        nomination=ref(root,nomination_path),nomination_request=ref(root,nomination_path.parent/'request.json'),
        checkpoint=ref(root,root/training['checkpoint']/'state.json'),corpus=trial['corpus_index'])
    candidate=candidate_intake.validate(root,submission,active['vocabulary_sha256'])
    destination=root/active['output']/'candidate-inbox'/(candidate['checkpoint_sha256']+'.json')
    immutable(destination,submission)
    return destination


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','manifest','trial','nomination'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(publish(a.root.resolve(),a.manifest.resolve(),a.trial.resolve(),a.nomination.resolve()))
