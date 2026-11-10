"""Version-bound S3 custody for H325's own generated model files only."""
import hashlib,json,os,tarfile
from pathlib import Path
import boto3
BUCKET='veox-jopedime-learning-use2'
PREFIX='experiments/rd_breakthrough_20261030/H325/solhwv-20261104T152733Z/'
_CLIENT=None

def client():
    global _CLIENT
    if _CLIENT is None:_CLIENT=boto3.Session(profile_name='veox-jepson-prod').client('s3')
    return _CLIENT

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def upload_verify(path,key):
    path=Path(path);digest=sha(path);s=client()
    with path.open('rb') as f:r=s.put_object(Bucket=BUCKET,Key=PREFIX+key,Body=f,IfNoneMatch='*',Metadata={'sha256':digest})
    version=r.get('VersionId');assert version and version!='null','versioning required'
    r=s.get_object(Bucket=BUCKET,Key=PREFIX+key,VersionId=version)
    h=hashlib.sha256();size=0
    for b in r['Body'].iter_chunks(chunk_size=1024*1024):h.update(b);size+=len(b)
    r['Body'].close()
    assert h.hexdigest()==digest and size==path.stat().st_size
    return dict(bucket=BUCKET,key=PREFIX+key,version_id=version,sha256=digest,bytes=size,exact_version_get_sha256_verified=True)

def compress_inner(tdir,durable):
    """After a XT lock: retain compressed inner members and remote verified recovery before removing raw copies."""
    tdir=Path(tdir);receipt=tdir/'inner_archive.json';archive=tdir/'inner_members.tar.gz'
    if receipt.exists():return
    files=sorted(p for p in tdir.glob('inner_tau*/*') if p.is_file())
    assert files and all(p.suffix in ('.txt','.json') for p in files)
    inventory={str(p.relative_to(tdir)):sha(p) for p in files}
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
        for p in files:tar.add(p,arcname=str(p.relative_to(tdir)),recursive=False)
    record=upload_verify(archive,'xt_members/'+'-'.join(tdir.parts[-3:])+'.tar.gz')
    record['members']=inventory
    durable(receipt,record)
    # Each removed file has both a byte pin and independently GET-verified S3 recovery.
    for p in files:
        assert sha(p)==inventory[str(p.relative_to(tdir))]
        p.unlink()


def restore_inner(tdir):
    tdir=Path(tdir);r=json.loads((tdir/'inner_archive.json').read_text())
    assert r['exact_version_get_sha256_verified'] and sha(tdir/'inner_members.tar.gz')==r['sha256']
    with tarfile.open(tdir/'inner_members.tar.gz','r:gz') as tar:
        for member in tar.getmembers():
            name=member.name
            assert member.isfile() and name in r['members'] and '..' not in Path(name).parts and not Path(name).is_absolute()
            p=tdir/name;p.parent.mkdir(exist_ok=True)
            data=tar.extractfile(member).read();assert hashlib.sha256(data).hexdigest()==r['members'][name]
            p.write_bytes(data)


def remove_restored_inner(tdir):
    tdir=Path(tdir);r=json.loads((tdir/'inner_archive.json').read_text())
    assert r['exact_version_get_sha256_verified']
    for name,digest in r['members'].items():
        p=tdir/name;assert sha(p)==digest;p.unlink()
