from pathlib import Path
import subprocess,json,hashlib,shlex
P=Path(__file__).parent;D=Path('/home/ubuntu/jope-data/h368-x1');remote=str(D);control=remote+'/control';allow=['source','vendor','x1_prepare.py','x1_launch.py','runtime-source-manifest.json','prereg-published.json','staged-input-inventory.json','base-fetch-receipt.json','nl15-fetch-receipt.json','science-equivalence.json']
assert json.loads((P/'prereg-published.json').read_text())['published'];assert json.loads((P/'inputs-unit-exit.json').read_text())['EXIT_STATUS']=='0';assert json.loads((P/'reference-unit-exit.json').read_text())['EXIT_STATUS']=='0'
subprocess.run(['ssh','-T','-o','BatchMode=yes','xbabe1','test ! -e '+remote+' && mkdir -p '+control+' '+remote+'/transfer/requests '+remote+'/transfer/responses'],check=True)
subprocess.run(['rsync','-a','--exclude=__pycache__','-e','ssh -T -o BatchMode=yes',str(D)+'/', 'xbabe1:'+remote+'/'],check=True)
for name in allow:subprocess.run(['rsync','-a','--exclude=__pycache__','-e','ssh -T -o BatchMode=yes',str(P/name)+('/' if (P/name).is_dir() else ''),'xbabe1:'+control+'/'+name+('/' if (P/name).is_dir() else '')],check=True)
code='''import platform,json,hashlib,lightgbm,numpy,sklearn,os
from pathlib import Path
p=Path(lightgbm.basic._LIB._name);r=dict(host=os.uname().nodename,kernel=platform.release(),python=platform.python_version(),numpy=numpy.__version__,lightgbm=lightgbm.__version__,sklearn=sklearn.__version__,lightgbm_library_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),GPU_used=False,credential_transfer=False);assert r['lightgbm_library_sha256']=='6a12e1a7b13824d14c252d2a4d2e36af467383331a114eb0e7448eecbf02d7b1';print(json.dumps(r))'''
env=subprocess.check_output(['ssh','-T','-o','BatchMode=yes','xbabe1','OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 19 taskset -c 35 python3 -c '+shlex.quote(code)]);(P/'environment.json').write_bytes(env)
print('H368 input/code/proof allowlist copied; no credentials/client control transferred')
