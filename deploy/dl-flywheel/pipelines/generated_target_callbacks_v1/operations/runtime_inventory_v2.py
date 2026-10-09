"""Fresh bounded metadata-only Python3.12 inventory; no cloud/DB/GP API."""
import sys
sys.dont_write_bytecode=True
import hashlib, importlib, importlib.metadata, json, os, platform, resource, sysconfig, time
from pathlib import Path

start=time.time_ns();mono=time.monotonic_ns();cpu0=resource.getrusage(resource.RUSAGE_SELF)
assert platform.python_version()=='3.12.3'
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
row=next(x.split(':',2)[2] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
cg=Path('/sys/fs/cgroup')/row.lstrip('/')
quota,period=(cg/'cpu.max').read_text().split()
assert quota!='max' and int(quota)<=int(period)
assert int((cg/'memory.max').read_text())<=512*1024**2 and int((cg/'memory.swap.max').read_text())==0

def pin(path):
    path=Path(path).resolve();assert path.is_file() and path.stat().st_size<=64*1024**2
    data=path.read_bytes();return {'path':str(path),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}

modules={}
for name in ('numpy','boto3','botocore'):
    m=importlib.import_module(name)
    modules[name]={'version':m.__version__,'file':pin(m.__file__)}
paths=set()
for distribution in ('numpy','boto3','botocore','s3transfer','jmespath','python-dateutil','urllib3','six'):
    d=importlib.metadata.distribution(distribution)
    for f in d.files or ():
        q=Path(d.locate_file(f)).resolve()
        if q.suffix in ('.py','.so') and q.is_file():paths.add(str(q))
# Distribution metadata can select a parallel installation while Python imports
# a system package. Pin actual package origins, including known lazy SDK imports.
actual_roots={}
for name in ('numpy','boto3','botocore','s3transfer','jmespath','dateutil','urllib3'):
    module=importlib.import_module(name)
    origin=Path(module.__file__).resolve();root=origin.parent
    assert root.is_dir() and root!=Path('/') and str(root).startswith(('/home/ubuntu/','/usr/lib/python3/'))
    actual_roots[name]={'origin':pin(origin),'root':str(root)}
    for q in root.rglob('*'):
        if q.suffix in ('.py','.so') and q.is_file():paths.add(str(q.resolve()))
    assert len(paths)<=4096
importlib.import_module('s3transfer.compat')
stdlib=Path(sysconfig.get_path('stdlib')).resolve()
for m in list(sys.modules.values()):
    f=getattr(m,'__file__',None)
    if not f or f.startswith('<'):continue
    q=Path(f).resolve()
    if q==Path(__file__).resolve():continue
    if q.is_relative_to(stdlib) and 'site-packages'not in q.parts and 'dist-packages'not in q.parts:continue
    paths.add(str(q))
assert 3<=len(paths)<=4096
files=[pin(q)for q in sorted(paths)];assert sum(x['bytes']for x in files)<=256*1024**2
cpu1=resource.getrusage(resource.RUSAGE_SELF)
print(json.dumps({'schema':'jope.generated-target-runtime-inventory/v1','state':'passed',
 'python_version':platform.python_version(),'interpreter':pin(sys.executable),'modules':modules,'import_files':files,'actual_package_origins':actual_roots,
 'started_at_unix_ns':start,'ended_at_unix_ns':time.time_ns(),'monotonic_elapsed_ns':time.monotonic_ns()-mono,
 'cpu_user_seconds':cpu1.ru_utime-cpu0.ru_utime,'cpu_system_seconds':cpu1.ru_stime-cpu0.ru_stime,
 'gp_executions':0,'s3_requests':0,'database_writes':0,'optimizer_updates':0},sort_keys=True))
