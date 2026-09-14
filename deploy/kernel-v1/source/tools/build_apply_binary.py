#!/usr/bin/env python3
"""Build one immutable apply-only binary and bind actual source bytes before/after."""
import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
root=Path('/home/ubuntu/jope-dime/parallel')
out=root/'bin/jope-parallel-neural-v43-apply'
receipt=root/'runs/parity/neural-v43-apply-build.json'
assert not out.exists() and not receipt.exists()
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def sources():
    files=[root/x for x in ['Cargo.toml','Cargo.lock','build.rs','rust-toolchain.toml'] if (root/x).exists()]
    for name in ['src','native','assets/source/jope-dime/vendor/feat-math']:
        files.extend(p for p in (root/name).rglob('*') if p.is_file() and p.suffix in ['.rs','.cpp','.c','.h','.toml','.lock','.cu','.cuh'])
    return {str(p.relative_to(root)):digest(p) for p in sorted(set(files))}
before=sources();started=time.time()
(root/'runs/parity/neural-v43-apply-sources-before.json').write_text(json.dumps(before,indent=2)+'\n')
env=dict(os.environ,LIBTORCH_USE_PYTORCH='1',CXX='g++',CARGO_BUILD_JOBS='2',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PATH='/home/ubuntu/.cargo/bin:'+os.environ['PATH'])
cmd=['/home/ubuntu/.cargo/bin/cargo','build','--release','--locked','--features','neural','--bin','jope-parallel']
with (root/'runs/parity/neural-v43-apply-build.log').open('x') as log:
    result=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
after=sources();assert before==after,'Source mutated during build; binary not published'
assert result.returncode==0,'Build failed; inspect preserved log'
with (root/'target/release/jope-parallel').open('rb') as src,out.open('xb') as dst:
    shutil.copyfileobj(src,dst);dst.flush();os.fsync(dst.fileno())
out.chmod(0o755)
row=dict(schema='parallel.immutable_apply_build.v1',passed=True,timestamp=int(time.time()),elapsed_seconds=time.time()-started,
    binary=str(out.relative_to(root)),binary_sha256=digest(out),source_before=before,source_after=after,
    command=cmd,scope='v41 remote base plus verified corpus cache and contexts delta; apply/union/enqueue only',
    source_changed_during_build=False)
receipt.write_text(json.dumps(row,indent=2)+'\n');print(json.dumps({k:v for k,v in row.items() if not k.startswith('source_')}))
