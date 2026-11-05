"""The owner requires archived native FFBN zero before any fit or CPU probe."""
import json
from pathlib import Path

def evaluate(values,count):
 assert count==972 and values and all(isinstance(v,(int,float)) for v in values)
 unique=sorted(set(float(x) for x in values))
 ok=unique==[0.0]
 return dict(status='PASS' if ok else 'INCOMPLETE_SPEC_MISMATCH',parameter='feature_fraction_bynode',requested_archived_value=0,observed_archived_values=unique,rankers_verified=count,requested_treatment_value=1,proposed_native_effective_delta=None if not ok else [0,1],treatment_fits=0,capacity_probes=0,reason='Archived native value must be zero under the owner nudge; nonzero closes before fits. Value1 is already the full node-feature fraction, so numeric1 would not change it; this is a numeric fraction rather than a Boolean switch.' if not ok else 'Owner precondition satisfied; all other gates still mandatory.')
def require_spec_pass(path):
 proof=json.loads(Path(path).read_text())
 if proof.get('status')!='PASS':raise RuntimeError('INCOMPLETE_SPEC_MISMATCH; no fits, scores or CPU probe authorized')
 return proof
