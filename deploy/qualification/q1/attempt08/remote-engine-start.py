import datetime,hashlib,json,os,subprocess,time
from pathlib import Path
E=Path('/home/ubuntu/jope_plan/evidence/QLT-04/qlt02-t0240'); H=Path('/mnt/jope-build-native/jope-artifacts/sol_v1_gov03/qlt04-t0115/engine-home'); B=Path('/mnt/jope-build-native/jope-artifacts/sol_v1_api01/qlt04-rc4-linux/jopedime-q1-linux'); bundle=Path('/home/ubuntu/jope_plan/evidence/QLT-04/api01-linux-rc4-t1955/authenticated-r782')
sha=lambda p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
assert sha(B)=='4e9239a1126589a42c3d80b1742d463d1b2331def59e5a08796169b4f3cec177'
assert not (H/'daemon.json').exists()
assert not (E/'RESUME-ENGINE-STARTED.json').exists()
assert not (E/'resume-engine.stdout').exists()
assert os.getloadavg()[0] < 85 and __import__('shutil').disk_usage('/').free > 100*1024**3
for key in ('AWS_','GH_','GITHUB_','JOPE_AWS_'):
    pass
env={k:v for k,v in os.environ.items() if not k.startswith(('AWS_','GH_','GITHUB_','JOPE_AWS_'))}
env.update(JOPEDIME_HOME=str(H),RAYON_NUM_THREADS='8',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',TMPDIR=str(H/'scratch'),PYTHONDONTWRITEBYTECODE='1')
args=['nice','-n','19',str(B),'research','serve-q1','--bundle-dir',str(bundle)]
with (E/'resume-engine.stdout').open('xb') as out,(E/'resume-engine.stderr').open('xb') as err:
    p=subprocess.Popen(args,cwd=E,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
rec={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':p.pid,'argv':args,'engine_home':str(H),'evaluator_sha256':sha(B),'activity':'same pinned evaluator and existing journal; no request outcome copied or opened'}
(E/'RESUME-ENGINE-STARTED.json').write_text(json.dumps(rec,indent=2)+'\n')
for _ in range(120):
    if p.poll() is not None: raise RuntimeError('engine exited before readiness; inspect private stderr')
    d=H/'daemon.json'
    if d.is_file():
        data=json.loads(d.read_text()); assert data['pid']==p.pid and d.stat().st_mode & 0o077 == 0
        assert sha(Path('/proc')/str(p.pid)/'exe')==rec['evaluator_sha256']
        rec.update(status='ENGINE_READY',endpoint=data['url'],daemon_mode='0600')
        (E/'RESUME-ENGINE-STARTED.json').write_text(json.dumps(rec,indent=2)+'\n')
        print('ENGINE_READY',p.pid,data['url'],flush=True);break
    time.sleep(.25)
else: raise RuntimeError('daemon readiness timeout')
