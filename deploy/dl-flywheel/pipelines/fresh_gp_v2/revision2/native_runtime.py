"""Stream-verify the unchanged complete native toolchain before a cold worker."""
import hashlib,os
from pathlib import Path
from common import *

def verify(reference,root,deadline):
    doc=obj(reference);base=Path(root);clock=Clock(deadline)
    need(base.is_absolute()and doc['files'],'native runtime closure')
    rows=[]
    for r in doc['files']:
        rel=Path(r['relative']);need(not rel.is_absolute()and'..'not in rel.parts,'native relative path')
        p=base/rel;need(p.is_file()and all(not x.is_symlink()for x in(p,*p.parents)),'native nofollow payload')
        st=p.stat();need(st.st_size==r['bytes'],'native closure bytes');h=hashlib.sha256()
        with p.open('rb')as f:
            while True:
                clock.tick();b=f.read(1024*1024)
                if not b:break
                h.update(b)
        need(h.hexdigest()==r['sha256']and(p.stat().st_ino,p.stat().st_size,p.stat().st_mtime_ns)==(st.st_ino,st.st_size,st.st_mtime_ns),'native runtime pin changed')
        if r.get('executable'):need(os.access(p,os.X_OK),'native executable bit')
        rows.append({'relative':str(rel),'sha256':h.hexdigest(),'bytes':st.st_size})
    clock.tick();return {'manifest_sha256':reference['sha256'],'files':len(rows),'bytes':sum(r['bytes']for r in rows),'closure_sha256':digest(rows)}
