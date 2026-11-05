"""Exact identity of final veto models against pinned archived BASE bytes."""
from pathlib import Path
import hashlib,json
D=Path('/home/ubuntu/jope-data/h367-dsc01');HERE=Path(__file__).resolve().parent
def verify(tdir,inventory=None,root=None):
 tdir=Path(tdir);root=Path(root) if root is not None else D
 inv=inventory if inventory is not None else json.loads((HERE/'base_veto_sha256.json').read_text())['members']
 models=[]
 for p in sorted((tdir/'refit').glob('*-veto.txt')):
  key='run/'+str(p.relative_to(root));h=hashlib.sha256(p.read_bytes()).hexdigest();expected=inv[key]['sha256'];assert h==expected,(key,h,expected)
  models.append(dict(file=str(p.relative_to(tdir)),archive_member=key,sha256=h,BASE_sha256=expected))
 assert len(models)==3
 return dict(status='PASS',matched=3,models=models)
