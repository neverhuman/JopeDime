"""One-use bounded selected GET-only qualification controller; no foreign actions."""
from pathlib import Path
import argparse,hashlib,json,math,os,subprocess,time
ROOT=Path('/home/ubuntu/JopeDime/.agent/proposals/dl-flywheel-generated-target-qualification-20261009-v1')
UNIT='jopedime-generated-target-callback-qualification-20261009-v1'
GLOBAL=1791592146.380285

def sha(raw):return hashlib.sha256(raw).hexdigest()
def main():
    a=argparse.ArgumentParser();a.add_argument('--config-sha',required=True);args=a.parse_args()
    raw=(ROOT/'config.json').read_bytes();assert sha(raw)==args.config_sha
    c=json.loads(raw);now=time.time();assert c['created_unix']<=now<c['deadline_unix']<=GLOBAL==c['global_deadline_unix']
    remaining=math.floor(c['deadline_unix']-now);assert 30<=remaining<=180
    assert c['mode']=='selected-qualification' and len(c['selected'])==1 and c['max_gets']==16 and c['max_read_bytes']==64*1024**2
    assert c['output_root']==str(ROOT/'output') and not (ROOT/'output').exists()
    assert c['runtime']['python_version']=='3.12.3'
    for rel,digest in c['package_files'].items():
        assert '..'not in Path(rel).parts and not Path(rel).is_absolute()
        assert sha((ROOT/'runtime'/rel).read_bytes())==digest
    p=subprocess.run(['systemctl','--user','show',UNIT,'-p','LoadState','-p','MainPID','-p','ControlGroup'],capture_output=True,text=True,check=False)
    prior=dict(x.split('=',1)for x in p.stdout.splitlines()if '='in x)
    assert prior['LoadState']=='not-found' and prior['MainPID']=='0' and prior['ControlGroup']==''
    mem={x.split(':')[0]:x.split(':')[1].strip()for x in Path('/proc/meminfo').read_text().splitlines()}
    assert int(mem['MemAvailable'].split()[0])>26*1024**2
    intent={'schema':'jope.generated-target-selected-launch-intent/v1','at_unix':time.time(),'config_sha256':args.config_sha,
      'controller_sha256':sha(Path(__file__).read_bytes()),'unit':UNIT,'deadline_unix':c['deadline_unix'],'runtime_max_seconds':remaining,
      'cpu_quota_percent':100,'memory_max_bytes':512*1024**2,'memory_swap_max_bytes':0,'prior_unit':prior,
      'available_kib':int(mem['MemAvailable'].split()[0]),'scope':'existing bounded source-support qualification; not held scoring expansion'}
    with (ROOT/'launch-intent.json').open('x')as f:json.dump(intent,f,sort_keys=True);f.flush();os.fsync(f.fileno())
    command=['systemd-run','--user','--quiet','--collect','--wait','--pipe','--unit='+UNIT,
      '-p','RuntimeMaxSec='+str(remaining),'-p','CPUQuota=100%','-p','MemoryMax=512M','-p','MemorySwapMax=0',
      '-p','Nice=19','-p','TasksMax=24','-p','Restart=no','-p','WorkingDirectory='+str(ROOT),
      '--setenv=CUDA_VISIBLE_DEVICES=','--setenv=OMP_NUM_THREADS=1','--setenv=OPENBLAS_NUM_THREADS=1','--setenv=MKL_NUM_THREADS=1',
      '/home/ubuntu/jopedime_probe_exp/ofm-20261008/venv/bin/python','-B','-m','runtime.cli','--config',str(ROOT/'config.json'),'--config-sha',args.config_sha]
    with (ROOT/'controller-stdout.json').open('xb')as out,(ROOT/'controller-stderr.txt').open('xb')as err:
        process=subprocess.Popen(command,stdout=out,stderr=err)
        while process.poll()is None:
            if time.time()>c['deadline_unix']+10 or (ROOT/'controller-stdout.json').stat().st_size>8*1024**2 or (ROOT/'controller-stderr.txt').stat().st_size>65536:
                process.terminate()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                break
            time.sleep(.2)
    reply={'schema':'jope.generated-target-controller-result/v1','at_unix':time.time(),'returncode':process.returncode,
      'config_sha256':args.config_sha,'stdout_sha256':sha((ROOT/'controller-stdout.json').read_bytes()),
      'stdout_bytes':(ROOT/'controller-stdout.json').stat().st_size,'stderr_bytes':(ROOT/'controller-stderr.txt').stat().st_size,
      'no_restart_or_retry':True}
    with (ROOT/'controller-result.json').open('x')as f:json.dump(reply,f,sort_keys=True);f.flush();os.fsync(f.fileno())
    print(json.dumps(reply,sort_keys=True))
if __name__=='__main__':main()
