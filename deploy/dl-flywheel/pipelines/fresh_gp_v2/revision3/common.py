"""Bounded cold artifacts and lifecycle primitives; no fitting or S3 mutation."""
import hashlib,json,os,socket,time
from pathlib import Path
from jope_flywheel.program_io import read,bound,create,need
from jope_flywheel.protocol import canonical,digest

GLOBAL=1791592146.380285
MIB=1024**2
BUCKET='veox-jopedime-learning-use2'
PREFIX='experiments/gp-policy-dl/v2-trainonly/'
ROOT_PREFIX='/home/ubuntu/JopeDime/.agent/proposals/'

def ref(path):
    raw=read(path,256*MIB);return {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
def obj(reference,cap=256*MIB):return json.loads(bound(reference,cap))
def put(root,name,value):return create(Path(root)/name,canonical(value))
def fresh(path):
    p=Path(path);need(p.is_absolute()and'..'not in p.parts and all(not q.is_symlink()for q in(p,*p.parents)),'owned absolute nonsymlink output')
    p.mkdir(mode=0o700,parents=False,exist_ok=False);return p
def host(expected):need(socket.gethostname().split('.')[0]==expected,'exact host required')
def atom(path,body):
    p=Path(path);tmp=p.with_name(p.name+'.'+str(os.getpid())+'.pending')
    create(tmp,body);os.replace(tmp,p)
    fd=os.open(p.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)

class Clock:
    def __init__(self,deadline,maximum=None):
        self.deadline=min(deadline,GLOBAL);duration=self.deadline-time.time()
        need(0<duration and(maximum is None or duration<=maximum),'fixed nonexpired bounded window')
        self.monotonic=time.monotonic()+duration
    def remaining(self):return min(self.deadline-time.time(),self.monotonic-time.monotonic())
    def tick(self):need(self.remaining()>0,'fixed shared clock expired')

def source_closure(source_manifest):
    manifest=obj(source_manifest,4*MIB)
    need(manifest['schema']=='jope.fresh128-source/v2'and manifest['files'],'source manifest')
    for r in manifest['files']:bound(r,64*MIB)
    return manifest

def timing(root,event,**fields):
    p=Path(root)/'timing.jsonl';raw=canonical({'event':event,'utc_ns':time.time_ns(),'monotonic_ns':time.monotonic_ns(),
        'pid':os.getpid(),'host':socket.gethostname(),**fields})+b'\n'
    fd=os.open(p,os.O_WRONLY|os.O_APPEND|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:os.write(fd,raw);os.fsync(fd)
    finally:os.close(fd)
