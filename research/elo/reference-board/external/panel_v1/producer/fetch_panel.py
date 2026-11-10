import json,subprocess,os,hashlib,sys
from concurrent.futures import ThreadPoolExecutor
E=dict(os.environ,AWS_PROFILE="veox-jepson"); B="s3://veox-jopedime-datasets-use2/"
P=json.load(open("/home/ubuntu/jope_plan/elo/panel_v1.json"))["datasets"]
def get(d):
    o=os.path.join(os.environ.get("JOPE_ELO_PANEL_DIR",os.path.join(os.environ["JOPE_S3_CACHE_DIR"],"elo-panel") if os.environ.get("JOPE_S3_CACHE_DIR") else "/home/ubuntu/jope_plan/elo/panel"),d["id"]); os.makedirs(o,exist_ok=True)
    mf=o+"/manifest.json"
    if not os.path.exists(mf): subprocess.run(["aws","s3","cp","--quiet",B+"datasets/%s/manifest.json"%d["id"],mf],env=E)
    m=json.load(open(mf)); st="OK"
    for k in ("train","test"):
        f=m["files"][k]; p=o+"/"+k+".csv"
        if not os.path.exists(p): subprocess.run(["aws","s3","cp","--quiet",B+"blobs/sha256/"+f["sha256"],p],env=E)
        if not os.path.exists(p) or hashlib.sha256(open(p,"rb").read()).hexdigest()!=f["sha256"]: st="BAD"
    json.dump(dict(has_header=m["files"]["train"].get("has_header",False),task=d["task"],family=d["family"],fold=d["fold"]),open(o+"/meta.json","w"))
    return d["id"],st
with ThreadPoolExecutor(16) as ex:
    for r in ex.map(get,P): print(*r,flush=True)
