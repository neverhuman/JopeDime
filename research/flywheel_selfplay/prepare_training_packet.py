import subprocess
import tarfile
from pathlib import Path

root = Path("/home/ubuntu/jope-data/flywheel-gp-fused/input")
script = r"""
import io,json,tarfile,gzip,struct,sys,hashlib
from pathlib import Path
base=Path('/mnt/jope-archive/flywheel-20261023')
m=json.loads((base/'corpus/corpus_manifest.json').read_text())
w=json.loads((base/'corpus_v2_train_winner_integrated/corpus_v2_manifest.json').read_text())['all_90_qualified_novel_winner_records']
cids={r['context_sha256'] for r in w}; families={r['family_id'] for r in w}
train=[r for r in m['records'] if r['context_sha256'] in cids]
dev=[]
for r in m['records']:
 if r['source_split']=='train' and r['dl_split']=='DL-Train' and r['family_id'] not in families and r['n_candidates']>=192:
  dev.append(r); families.add(r['family_id'])
  if len(dev)==8: break
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as tf:
 def emit(name,data):
  ti=tarfile.TarInfo(name);ti.size=len(data);tf.addfile(ti,io.BytesIO(data))
 for r in train+dev:
  cid=r['context_sha256']; p=Path(r['pool_path']); raw=p.read_bytes()
  emit('pools/'+cid+'.npz',raw); r['pool_sha256']=hashlib.sha256(raw).hexdigest()
  if True:
   raw=gzip.decompress(Path(r['teacher_jtt1_path']).read_bytes());nt,nq,p=struct.unpack_from('<III',raw,4)
   # Deliver training rows only: no outer features or labels in local closure.
   end=17+nt+nt*(p+1)*8; out=bytearray(raw[:end]);struct.pack_into('<I',out,8,0)
   emit('tables/'+cid+'.jtt1',out);r['train_only_jtt1_sha256']=hashlib.sha256(out).hexdigest()
 emit('selected.json',json.dumps({'train':train,'dev':dev}).encode())
"""
p = subprocess.Popen(
    ["ssh", "xbabe2", "python3", "-c", __import__("shlex").quote(script)],
    stdout=subprocess.PIPE,
)
with tarfile.open(fileobj=p.stdout, mode="r|") as tf:
    for item in tf:
        dest = root / item.name
        if not dest.resolve().is_relative_to(root.resolve()):
            raise ValueError(item.name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(tf.extractfile(item).read())
if p.wait():
    raise SystemExit(p.returncode)
print("training-only transfer complete")
