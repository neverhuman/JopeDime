#!/usr/bin/env python3
"""Build the frozen source on the target ABI within two existing CPU slots."""
import fcntl
import shutil
import sys
import time
from pathlib import Path


def main(root,build):
    source=build/'source/parallel'
    sys.path.insert(0,str(source/'tools'))
    import experiment_runtime as runtime
    from promotion_gate import immutable
    from qualification_budget import Budget
    from resume_expired_cycle import read, require, sha
    request=read(build/'build.json')
    for name,digest in request['source_bindings'].items():
        require(sha(build/'source'/name)==digest,'frozen build source changed: '+name)
    budget=Budget(root/'runs/regression-gain-20260909/trial')
    identity='flywheel-'+build.name+'-target-build'
    budget.reserve(identity,core_seconds=1800)
    start=time.monotonic();receipts=[];result=dict(status='inconclusive')
    try:
        with (root/'runs/experiment-spare-cpus.lock').open('a') as lease:
            while True:
                try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:
                    require(time.monotonic()-start<900,'build allocation timeout');time.sleep(1)
            runtime._lease_fds=(*runtime._lease_fds,lease.fileno())
            env=runtime.child_env(PATH='/home/ubuntu/.cargo/bin:/usr/bin:/bin',LIBTORCH_USE_PYTORCH='1',CXX='g++',
                CXXFLAGS='-g0',CARGO_BUILD_JOBS='2',CARGO_TARGET_DIR=str(root/'target'),OMP_NUM_THREADS='2',
                RAYON_NUM_THREADS='2',LD_LIBRARY_PATH='/usr/lib/python3/dist-packages/torch/lib')
            commands=[['clean','--release','-p','jope-parallel'],
                      ['build','--release','--features','neural','--bin','jope-parallel','--example','replay_training','--example','qualification_proposals'],
                      ['test','--release','--features','neural','--test','corpus_incremental','--test','replay_training','--test','gp_contracts','--test','gp_admission','--test','service_continuous','--test','sealed_storage','--test','resident_worker'],
                      ['test','--release','--features','neural','--lib']]
            if request.get('cpu_only'):
                commands=[['clean','--release','-p','jope-parallel'],
                    ['build','--release','--bin','jope-parallel'],
                    ['test','--release','--test','corpus_incremental','--test','gp_contracts','--test','gp_admission','--test','service_continuous','--test','sealed_storage'],
                    ['test','--release','--lib']]
            for number,args in enumerate(commands):
                command=['/home/ubuntu/.cargo/bin/cargo','+1.90.0',*args,'--offline','--locked','--manifest-path',str(source/'Cargo.toml'),*(['-j','2'] if args[0]!='clean' else [])]
                if '--lib' in args:command.extend(['--','--test-threads=1'])
                t=time.monotonic()
                with (build/f'target-build-{number}.log').open('xb') as log:
                    children=[]
                    with runtime.owned_ranks(children):
                        child=runtime.popen(command,cwd=source,env=env,stdout=log,stderr=log);children.append(child)
                        while child.poll() is None:
                            require(time.monotonic()-start<900,'native build ceiling exhausted');time.sleep(.5)
                        receipts.append(dict(command=command,exit_code=child.returncode,seconds=time.monotonic()-t))
                        require(child.returncode==0,'target native build/test failed')
            destination=build/'target-bin';destination.mkdir()
            binaries=[('jope-parallel-native',root/'target/release/jope-parallel'),
                ('replay_training',root/'target/release/examples/replay_training'),
                ('qualification_proposals',root/'target/release/examples/qualification_proposals')]
            if request.get('cpu_only'):binaries=binaries[:1]
            for name,path in binaries:
                shutil.copy2(path,destination/name);(destination/name).chmod(0o555)
            result.update(status='passed',bindings={p.name:sha(p) for p in destination.iterdir()})
    except (OSError,RuntimeError,ValueError) as error:result['reason']=str(error)
    finally:
        result.update(receipts=receipts,seconds=time.monotonic()-start,source_sha256=sha(build/'source.tar.gz'),driver_sha256=sha(Path(__file__)),final_tests_opened=False)
        immutable(build/'target-build.json',result);budget.complete(identity,build/'target-build.json')
    require(result['status']=='passed',result.get('reason','build inconclusive'))


if __name__=='__main__':main(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
