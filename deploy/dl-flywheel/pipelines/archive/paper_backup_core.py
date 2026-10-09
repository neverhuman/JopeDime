#!/usr/bin/env python3
"""Bounded paper-evidence snapshot and create-only version-verified S3 custody."""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import socket
import sqlite3
import stat
import sys
import tarfile
import time

sys.dont_write_bytecode = True
W = Path('/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final')
SOURCE = W / 'paper_observer_addfba0176a7/jope_flywheel/evidence.py'
SOURCE_SHA = 'e4bd1d1e92455bd9b4a210d4e46cddf392cc565d7bcc7c1b6bea3e48a6a070a9'
MANIFESTS = {'full7120': 'c942197bf28e292f407abd7acae61395b8cf88b6ada8cf3a35ffa5cec5d6f584',
             'original7087': '5e2f842188b92371b38d057eb5c02b4b241f9fd6c5f377547091f50babae48d1'}
CAMPAIGNS = {'full7120':'full7120-ea0da768ff','original7087':'d910a26e9c'}
BUCKET = 'veox-jopedime-learning-use2'
PREFIX = 'experiments/gp-policy-dl/v2-trainonly/paper-evidence/20261008/'
CAP = 512 * 1024**2
MIB = 1024**2


def require(ok, text):
    if not ok:
        raise ValueError(text)


def digest(x):
    require(type(x) is str and re.fullmatch('[a-f0-9]{64}', x), 'invalid SHA256')
    return x


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for b in iter(lambda: stream.read(MIB), b''):
            h.update(b)
    return h.hexdigest()


def time_check(deadline):
    require(time.time() < deadline, 'fixed host backup deadline expired')


def open_dir(path):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts, 'absolute confined path required')
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd); fd = nxt
        return fd
    except BaseException:
        os.close(fd)
        raise


def safe_open(path, cap):
    path = Path(path); parent = open_dir(path.parent)
    try:
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    finally:
        os.close(parent)
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > cap:
        os.close(fd); raise ValueError('file is not bounded regular single-link evidence')
    return os.fdopen(fd, 'rb'), info


def read_small(path, cap=4*MIB):
    stream, before = safe_open(path, cap)
    with stream:
        body = stream.read(cap + 1); after = os.fstat(stream.fileno())
    require(len(body) <= cap and (before.st_ino,before.st_size,before.st_mtime_ns) == (after.st_ino,after.st_size,after.st_mtime_ns), 'source changed during read')
    return body


def used(root):
    total = 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs:
            require(not (Path(directory)/name).is_symlink(), 'symlink in owned package root')
        for name in files:
            p = Path(directory)/name; info = p.lstat()
            require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, 'nonregular owned package file')
            total += info.st_size
    return total


def write_new(path, raw, budget_root):
    require(used(budget_root) + len(raw) <= CAP, 'host package budget exceeded')
    with Path(path).open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def cgroup_check():
    rel = next(x.split(':',2)[2] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
    root = Path('/sys/fs/cgroup') / rel.lstrip('/')
    cpu = (root/'cpu.max').read_text().split()
    mem = (root/'memory.max').read_text().strip()
    pids = (root/'pids.max').read_text().strip()
    require(cpu[0] != 'max' and int(cpu[0])/int(cpu[1]) <= 1, 'CPU cap missing')
    require(mem != 'max' and int(mem) <= CAP, 'memory cap missing')
    require(pids != 'max' and int(pids) <= 32, 'task cap missing')
    require(os.getpriority(os.PRIO_PROCESS,0) >= 19, 'nice bound missing')
    return {'cgroup':str(root),'cpu_max':cpu,'memory_max':mem,'pids_max':pids}


def read_config(path, expected):
    raw = read_small(path, 128*1024);require(hashlib.sha256(raw).hexdigest() == digest(expected), 'config SHA mismatch')
    c = json.loads(raw)
    require(c['schema']=='jope.paper-backup/v1' and c['host'] in ('xbabe2','xbabe3'), 'config identity')
    require(socket.gethostname()==c['host'], 'wrong host')
    require(c['deadline_unix']-c['created_unix']==90 and c['created_unix']<=time.time()+5, 'deadline bounds')
    time_check(c['deadline_unix'])
    require(c['helper_sha256']==sha_file(__file__), 'helper source changed')
    require(re.fullmatch('[a-f0-9]{12}',c['snapshot_id']), 'unsafe snapshot id')
    require(c['out']==str(W/('paper_backup_snapshot_'+c['snapshot_id']+'_'+c['host'])), 'output identity')
    labels = ['original7087'] if c['host']=='xbabe2' else ['full7120','original7087']
    require([a['label']for a in c['archives']]==labels, 'archive closure')
    for a in c['archives']:
        label=a['label']
        require(a['archive']==str(W/('paper_archive_'+label+'_'+c['host'])), 'archive path mismatch')
        require(a['manifest_sha256']==MANIFESTS[label], 'campaign SHA mismatch')
        require(a['manifest']==str(W/('paper_observer_addfba0176a7/'+label+'.manifest.json')), 'campaign path mismatch')
        digest(a['prior_anchor']['head_sha256'])
        require(type(a['prior_anchor']['event_count']) is int and a['prior_anchor']['event_count']>0,'invalid prior count')
    return c


def copy_sqlite(source, destination, deadline, after_anchor=None):
    # Never open the live database read-write; its WAL is part of this transaction.
    for suffix in ('','-wal','-shm'):
        path=Path(str(source)+suffix)
        if path.exists():
            stream,_=safe_open(path,32*MIB);stream.close()
    src=sqlite3.connect(Path(source).as_uri()+'?mode=ro',uri=True,timeout=.5,isolation_level=None)
    dst=sqlite3.connect(str(destination),isolation_level=None)
    try:
        src.execute('PRAGMA query_only=ON');src.execute('BEGIN')
        count,head=src.execute("SELECT COUNT(*),COALESCE((SELECT event_hash FROM events ORDER BY seq DESC LIMIT 1),'') FROM events").fetchone()
        require(0<count<=4000,'event count bound')
        ledger_id=src.execute("SELECT value FROM metadata WHERE key='ledger_id'").fetchone()[0]
        if after_anchor:after_anchor()
        page=src.execute('PRAGMA page_size').fetchone()[0]
        def progress(status,remaining,total):
            time_check(deadline);require(total*page<=16*MIB,'SQLite snapshot exceeds16MiB')
        src.backup(dst,pages=128,progress=progress,sleep=.01)
        src.rollback()
        dst.execute('PRAGMA journal_mode=DELETE')
    finally:
        src.close();dst.close()
    return {'event_count':count,'head_sha256':head,'ledger_id':ledger_id}


def allowlist(m, host, manifest_path):
    root=Path(m['hosts'][host]['root']);runtime=Path(m['hosts'][host]['runtime'])
    a={'campaign/manifest.json':(str(manifest_path),'snapshot')}
    for name in ('manifest.json','arm_V1.json','preflight.json','status.json'):
        a['runtime/'+name]=(str(runtime/name),'snapshot')
    a['runtime/gpu_qualification.json']=(m['gpu_qualification']['path'],'snapshot')
    a['runtime/events.jsonl']=(str(runtime/'events.jsonl'),'jsonl_chunk')
    for job in m['jobs']:
        if job['host']!=host:continue
        rid=job['run_id'];run=root/'runs'/rid
        a[rid+'/identity.json']=(str(runtime/(rid+'.identity.json')),'snapshot')
        for name in ['summary.json']+[f'hist_r{r}f{k}.json'for r in range(3)for k in range(5)]+[f'{stem}r{r}.npy'for stem in ('oof_','oofraw_')for r in range(3)]:
            a[rid+'/'+name]=(str(run/name),'snapshot')
        for name in ('batch_metrics.jsonl','exposure.jsonl'):
            a[rid+'/'+name]=(str(run/name),'jsonl_chunk')
            a[rid+'/'+name+':final_bytes']=(str(run/name),'snapshot')
    return a


def verify_snapshot(db, evidence, m, host, manifest_path, prior, captured):
    blobs={};records=[];cursors={}
    with evidence.Ledger(db) as ledger:  # Only the owned backup; never the live DB.
        anchor=ledger.verify(expected_head=captured['head_sha256'],expected_count=captured['event_count'])
        require(anchor['ledger_id']==captured['ledger_id']==prior['ledger_id'],'ledger identity')
        require(anchor['event_count']>=prior['event_count'],'archive truncated before external anchor')
        old=ledger._connection.execute('SELECT event_hash FROM events WHERE seq=?',(prior['event_count'],)).fetchone()
        require(old and old[0]==prior['head_sha256'],'external head is not retained in chain')
        runid='archive-'+host+'-'+m['campaign_id']
        expected={'source_sha256':m['source_archive_sha256'],'schema':'jope.wave4-paper-archive/v1','campaign_manifest_sha256':MANIFESTS['full7120' if m['campaign_id'].startswith('full7120-')else'original7087'],'host':host,'source_git_sha':m['source_sha'],'bundle_sha256':m['bundle_sha'],'claims':'imported source observations; no independent scientific validation'}
        rows=list(ledger._connection.execute('SELECT * FROM runs'))
        require(len(rows)==1 and rows[0]['run_id']==runid and json.loads(rows[0]['manifest_json'])==expected,'archive run binding')
        allowed=allowlist(m,host,manifest_path)
        for row in ledger._connection.execute('SELECT * FROM events ORDER BY seq'):
            e=ledger._decode(row)
            if e['event_type']=='ledger.run_registered':continue
            require(e['event_type']=='archive.source_observed' and e['evidence_kind']=='imported_aggregate','nonarchive event')
            d=e['payload']['aggregate'];h=digest(d['sha256'])
            require(d['blob']=='blobs/'+h and d['label']in allowed,'unapproved CAS/label')
            require((d['source_path'],d['kind'])==allowed[d['label']],'source path/kind is outside archive allowlist')
            require(e['payload']['source_receipt_sha256']==h,'import receipt mismatch')
            require(e['event_id']=='source-'+hashlib.sha256(canonical(d)).hexdigest(),'descriptor identity')
            require(type(d['bytes'])is int and 0<=d['bytes']<=64*MIB,'blob bound')
            require(h not in blobs or blobs[h]['bytes']==d['bytes'],'conflicting CAS length')
            blobs[h]={'sha256':h,'bytes':d['bytes'],'path':'blobs/'+h}
            if d['kind']=='jsonl_chunk':
                previous=cursors.get(d['label']);require(d['start']==(previous['end']if previous else 0) and d['end']-d['start']==d['bytes'],'noncontiguous imported log')
                cursors[d['label']]=d
            records.append(d)
        campaign=[d for d in records if d['label']=='campaign/manifest.json']
        require(campaign and all(d['sha256']==expected['campaign_manifest_sha256']for d in campaign),'campaign manifest CAS binding')
        ledger.export_jsonl('events.jsonl');ledger.export_csv('events.csv')
    # Make the portable backup self-contained, without a WAL dependency.
    conn=sqlite3.connect(str(db));conn.execute('PRAGMA journal_mode=DELETE');conn.close()
    return anchor,records,blobs


class CheckedReader:
    def __init__(self, stream, deadline):
        self.stream=stream;self.deadline=deadline;self.hash=hashlib.sha256();self.n=0
    def read(self,n=-1):
        time_check(self.deadline);require(0<=n<=MIB,'unbounded tar read')
        data=self.stream.read(n);self.hash.update(data);self.n+=len(data);return data


class CappedWriter:
    def __init__(self,stream,limit,deadline):self.stream=stream;self.limit=limit;self.deadline=deadline;self.n=0
    def write(self,data):
        time_check(self.deadline);require(self.n+len(data)<=self.limit,'host package total exceeds512MiB')
        self.n+=len(data);return self.stream.write(data)
    def tell(self):return self.n
    def flush(self):return self.stream.flush()


def package_tar(path, entries, budget_root, deadline):
    pins=[];limit=CAP-used(budget_root)-8*MIB
    with path.open('xb')as dest:
        writer=CappedWriter(dest,limit,deadline)
        with tarfile.open(fileobj=writer,mode='w|',format=tarfile.USTAR_FORMAT)as tar:
            for name,source,expected,nbytes in entries:
                stream,before=safe_open(source,64*MIB)
                with stream:
                    require(before.st_size==nbytes,'entry size changed')
                    reader=CheckedReader(stream,deadline)
                    info=tarfile.TarInfo(name);info.size=nbytes;info.mode=0o600;info.uid=info.gid=0;info.mtime=0
                    tar.addfile(info,reader)
                    after=os.fstat(stream.fileno())
                    require(reader.n==nbytes and reader.hash.hexdigest()==expected,'CAS/file hash mismatch')
                    require((before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_size,after.st_mtime_ns,after.st_ctime_ns),'CAS/file modified during read')
                pins.append({'path':name,'sha256':expected,'bytes':nbytes})
        dest.flush();os.fsync(dest.fileno())
    return pins


def verify_tar(path,pins,deadline):
    expected={p['path']:p for p in pins};seen=set()
    require(len(expected)==len(pins),'duplicate package entry')
    with tarfile.open(path,mode='r|')as tar:
        for member in tar:
            time_check(deadline)
            require(member.isfile() and member.name in expected and member.name not in seen,'unexpected/nonregular package member')
            p=expected[member.name];require(member.size==p['bytes'],'package member size mismatch')
            stream=tar.extractfile(member);h=hashlib.sha256();size=0
            while True:
                block=stream.read(MIB)
                if not block:break
                time_check(deadline);h.update(block);size+=len(block)
            require(size==p['bytes'] and h.hexdigest()==p['sha256'],'package readback hash mismatch')
            seen.add(member.name)
    require(seen==set(expected),'missing package members')


def snapshot(c,config_sha):
    root=Path(c['out']);parent=open_dir(root.parent)
    try:os.mkdir(root.name,0o700,dir_fd=parent)
    finally:os.close(parent)
    raw=read_small(SOURCE);require(hashlib.sha256(raw).hexdigest()==SOURCE_SHA,'verifier source pin mismatch')
    spec=importlib.util.spec_from_file_location('pinned_evidence',SOURCE);evidence=importlib.util.module_from_spec(spec);spec.loader.exec_module(evidence)
    results=[]
    for a in c['archives']:
        time_check(c['deadline_unix']);archive=Path(a['archive'])
        marker=json.loads(read_small(archive/'archive.json'))
        require(marker=={'schema':'jope.wave4-paper-archive/v1','manifest_sha256':a['manifest_sha256'],'host':c['host']},'archive marker mismatch')
        mr=read_small(a['manifest']);require(hashlib.sha256(mr).hexdigest()==a['manifest_sha256'],'manifest bytes changed')
        m=json.loads(mr);campaign=m['campaign_id'];require(re.fullmatch('[A-Za-z0-9_.-]{1,128}',campaign),'unsafe campaign')
        out=root/campaign;out.mkdir(mode=0o700)
        captured=copy_sqlite(archive/'evidence.sqlite',out/'evidence.sqlite',c['deadline_unix'])
        anchor,records,blobs=verify_snapshot(out/'evidence.sqlite',evidence,m,c['host'],a['manifest'],a['prior_anchor'],captured)
        require(sum(b['bytes']for b in blobs.values())+used(root)+8*MIB<=CAP,'referenced evidence exceeds host package capacity')
        entries=[]
        for name in ('evidence.sqlite','events.jsonl','events.csv'):
            p=out/name;entries.append((name,p,sha_file(p),p.stat().st_size))
        metadata={'schema':'jope.paper-evidence-snapshot/v1','host':c['host'],'campaign_id':campaign,'archive_root':str(archive),'campaign_manifest_sha256':a['manifest_sha256'],'snapshot_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'ledger':anchor,'prior_external_anchor':a['prior_anchor'],'prior_anchor_included':True,'config_sha256':config_sha,'helper_sha256':c['helper_sha256'],'verifier_sha256':SOURCE_SHA,'metadata_files':[{'path':n,'sha256':h,'bytes':s}for n,p,h,s in entries],'referenced_blobs':list(blobs.values()),'artifact_observations':records,'unreferenced_blobs_copied':0,'raw_dataset_bytes_read':0,'checkpoint_bytes_copied':0,'scope':'Consistent committed evidence snapshot; source observations retain original limitations and do not prove native batch order or independent scientific correctness.'}
        write_new(out/'snapshot_manifest.json',canonical(metadata)+b'\n',root)
        p=out/'snapshot_manifest.json';entries.append((p.name,p,sha_file(p),p.stat().st_size))
        entries.extend((b['path'],archive/b['path'],h,b['bytes'])for h,b in sorted(blobs.items()))
        package=out/'archive.tar';pins=package_tar(package,entries,root,c['deadline_unix'])
        verify_tar(package,pins,c['deadline_unix'])
        record={'host':c['host'],'campaign_id':campaign,'campaign_manifest_sha256':a['manifest_sha256'],'archive_root':str(archive),'ledger':anchor,'prior_anchor_included':True,'package_path':str(package),'package_sha256':sha_file(package),'package_bytes':package.stat().st_size,'package_entries':pins,'snapshot_manifest_sha256':sha_file(out/'snapshot_manifest.json'),'unique_referenced_blobs':len(blobs),'artifact_observations':len(records),'verified':True}
        write_new(out/'snapshot_result.json',canonical(record)+b'\n',root);results.append(record)
    result={'schema':'jope.paper-backup-snapshots/v1','config_sha256':config_sha,'host':c['host'],'helper_sha256':c['helper_sha256'],'captured_at':time.time(),'deadline_unix':c['deadline_unix'],'archives':results,'total_disk_bytes_before_receipt':used(root),'package_bytes_total':sum(r['package_bytes']for r in results),'raw_dataset_bytes_read':0,'checkpoint_bytes_copied':0,'observers_stopped':False}
    write_new(root/'snapshot_result.json',canonical(result)+b'\n',root)
    require(used(root)<=CAP,'final host package capacity');return result


def put_verified(client,key,source,expected,size,deadline):
    time_check(deadline)
    require(key.startswith(PREFIX),'S3 namespace escape')
    body=source.open('rb')if isinstance(source,Path)else io.BytesIO(source)
    try:
        try:
            r=client.put_object(Bucket=BUCKET,Key=key,Body=body,ContentLength=size,IfNoneMatch='*',Metadata={'sha256':expected})
            version=r.get('VersionId');status='created'
        except Exception as error:
            code=getattr(error,'response',{}).get('Error',{}).get('Code')
            if code not in ('PreconditionFailed','ConditionalRequestConflict','412'):raise
            r=client.head_object(Bucket=BUCKET,Key=key);version=r.get('VersionId');status='existing_verified'
    finally:body.close()
    require(type(version)is str and version not in ('','null'),'versioned destination required')
    time_check(deadline);r=client.get_object(Bucket=BUCKET,Key=key,VersionId=version)
    require(r.get('VersionId')==version and r.get('ContentLength')==size,'GET version/size metadata mismatch')
    h=hashlib.sha256();n=0
    with r['Body']as stream:
        while True:
            time_check(deadline);part=stream.read(MIB)
            if not part:break
            n+=len(part);require(n<=size,'GET object exceeds expected bytes');h.update(part)
    require(n==size and h.hexdigest()==expected,'version-specific S3 GET hash mismatch')
    return {'bucket':BUCKET,'key':key,'version_id':version,'sha256':expected,'bytes':size,'status':status,'verified_at':time.time(),'verification':'exact VersionId GET + SHA256'}


def upload(c,config_sha):
    from botocore.config import Config
    import boto3
    root=Path(c['out']);result=json.loads(read_small(root/'snapshot_result.json',16*MIB))
    require(result['config_sha256']==config_sha and result['host']==c['host'] and result['helper_sha256']==c['helper_sha256'],'snapshot result identity')
    expected={CAMPAIGNS[a['label']]:a['manifest_sha256']for a in c['archives']}
    require(len(result['archives'])==len(expected) and {r['campaign_id']for r in result['archives']}==set(expected),'snapshot campaign closure')
    require(used(root)<=CAP,'host package capacity')
    client=boto3.Session(profile_name='veox-jepson-prod').client('s3',config=Config(connect_timeout=5,read_timeout=10,retries={'max_attempts':0},max_pool_connections=1))
    uploads=[]
    for r in result['archives']:
        time_check(c['deadline_unix']);package=Path(r['package_path'])
        require(r['host']==c['host'] and r['campaign_manifest_sha256']==expected[r['campaign_id']],'snapshot campaign binding')
        require(package==root/r['campaign_id']/'archive.tar','package path escape')
        stream,_=safe_open(package,CAP);stream.close()
        require(sha_file(package)==r['package_sha256'] and package.stat().st_size==r['package_bytes'],'package changed before upload')
        verify_tar(package,r['package_entries'],c['deadline_unix'])
        prefix=PREFIX+c['host']+'/'+r['campaign_id']+'/'+r['package_sha256']+'/'
        obj=put_verified(client,prefix+'archive.tar',package,r['package_sha256'],r['package_bytes'],c['deadline_unix'])
        proof={'schema':'jope.paper-evidence-s3-custody/v1','config_sha256':config_sha,'host':c['host'],'campaign_id':r['campaign_id'],'campaign_manifest_sha256':r['campaign_manifest_sha256'],'ledger':r['ledger'],'snapshot_manifest_sha256':r['snapshot_manifest_sha256'],'package':obj,'package_entries_verified':len(r['package_entries']),'scope':'Versioned custody of verified evidence bytes, not independent scientific validation.'}
        raw=canonical(proof)+b'\n';receipt=put_verified(client,prefix+'verification.json',raw,hashlib.sha256(raw).hexdigest(),len(raw),c['deadline_unix'])
        uploads.append({'package':obj,'verification_receipt':receipt,'ledger':r['ledger'],'host':c['host'],'campaign_id':r['campaign_id']})
    final={'schema':'jope.paper-evidence-upload-result/v1','config_sha256':config_sha,'at':time.time(),'host':c['host'],'archives':uploads,'status':'verified'}
    write_new(root/'s3_upload_result.json',canonical(final)+b'\n',root);return final


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--config-sha',required=True);ap.add_argument('--mode',choices=('snapshot','upload'),required=True);a=ap.parse_args()
    c=read_config(a.config,a.config_sha);limits=cgroup_check()
    result=snapshot(c,a.config_sha)if a.mode=='snapshot'else upload(c,a.config_sha)
    print(json.dumps({'result':result,'resource_limits':limits},allow_nan=False),flush=True)


if __name__=='__main__':main()
