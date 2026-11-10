"""After both regressions, report the fresh archive native precondition; zero fits."""
from pathlib import Path
import json,datetime
from spec_guard import evaluate
P=Path(__file__).resolve().parent.parent;D=Path('/home/ubuntu/jope-data/h384-dsc01')
def main():
 assert json.loads((D/'regression.json').read_text())['status']=='PASS'
 assert json.loads((P/'nl15-regression.json').read_text())['status']=='PASS'
 x=json.loads((P/'nl15-fetch-receipt.json').read_text());assert x['status']=='PASS'
 r=evaluate(x['native_feature_fraction_bynode_values'],x['native_feature_fraction_bynode_count'])
 r.update(utc=datetime.datetime.now(datetime.UTC).isoformat(),source_archive=x['archive_manifest'],archive_parts=len(x['parts']),archive_members_verified=x['members_verified'],BASE_and_NL15_regressions='EXACT_PASS',feature_fraction=.8,min_split_gain=0,literal_boolean_fit_tested=False,documentation='https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html#feature_fraction_bynode')
 for path in [P/'SPEC-CHECK.json',D/'SPEC-CHECK.json']:path.write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps(r),flush=True)
 return 0 if r['status']=='PASS' else 3
if __name__=='__main__':raise SystemExit(main())
