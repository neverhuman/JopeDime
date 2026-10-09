"""One-use local clock freeze; no staging, services, network or upload."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
ROOT=Path('/home/ubuntu/JopeDime/.agent/proposals/dl-flywheel-trace-recovery-20261009-v3')
GLOBAL=1791592146.380285
INPUTS=('source','base','previous_source','prior1_capture','prior2_capture','terminal_proof')

def need(value,message):
    if not value:raise ValueError(message)

def raw_file(path,limit=16*1024**2):
    path=Path(path)
    need(path.is_file()and not any(p.is_symlink()for p in(path,*path.parents))and path.stat().st_size<=limit,'bounded physical input')
    raw=path.read_bytes();need(len(raw)<=limit,'bounded input grew');return raw

def freeze(directory,template_sha,review_path,review_sha,*,now=None):
    directory=Path(directory);raw=raw_file(directory/'config.template.json');need(hashlib.sha256(raw).hexdigest()==template_sha,'template pin')
    review_raw=raw_file(review_path);need(hashlib.sha256(review_raw).hexdigest()==review_sha,'independent review pin')
    review=json.loads(review_raw);need(review['state']=='passed'and review['template_sha256']==template_sha,'reviewed exact template')
    c=json.loads(raw);need(c['schema']=='jope.trace-archive-recovery/v3'and c['out']==str(ROOT)and c['created_unix']is None and c['deadline_unix']is None,'unfrozen new namespace')
    need(c['maximum_segments']==2500 and c['maximum_global_put_attempts']==5000 and c['maximum_put_attempts']==3787 and c['global_deadline_unix']==GLOBAL,'conserved original limits')
    for name in INPUTS:
        ref=c[name];need(Path(ref['path']).parent==ROOT,'new namespace input reference')
        body=raw_file(directory/Path(ref['path']).name)
        need(len(body)==ref['bytes']and hashlib.sha256(body).hexdigest()==ref['sha256'],'exact source/capture input pin')
    now=time.time()if now is None else now;need(0<now and now+3600<=GLOBAL,'full one-hour window must fit original global deadline')
    c.update(created_unix=now,deadline_unix=now+3600);body=json.dumps(c,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    target=directory/'config.frozen.json'
    with target.open('xb')as stream:stream.write(body);stream.flush();os.fsync(stream.fileno())
    return {'path':str(target),'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest(),'created_unix':now,'deadline_unix':now+3600}

def main():
    p=argparse.ArgumentParser();p.add_argument('--template-sha',required=True);p.add_argument('--review',required=True);p.add_argument('--review-sha',required=True);args=p.parse_args()
    print(json.dumps(freeze(HERE,args.template_sha,args.review,args.review_sha)))

if __name__=='__main__':main()
