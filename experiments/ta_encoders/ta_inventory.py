#!/usr/bin/env python3
"""Inventory every jtt1 input object (encoder-s0 / teacher-s0 / encoder) listed in the MAIN v2-trainonly manifests, plus the G3
context list (manifests whose name contains g3). Read-only (list + get on manifests/). Usage: ta_inventory.py OUTDIR"""
import boto3, botocore, json, concurrent.futures as cf, collections, os, sys
B="veox-jopedime-learning-use2"; P="experiments/gp-policy-dl/v2-trainonly/"
s3=boto3.Session(profile_name="veox-jepson-prod").client("s3",config=botocore.config.Config(max_pool_connections=32))
out=sys.argv[1]; os.makedirs(out,exist_ok=True)
keys=[]
for page in s3.get_paginator("list_objects_v2").paginate(Bucket=B,Prefix=P+"manifests/"):
    for o in page.get("Contents",[]):
        if o["Key"].endswith(".json"): keys.append(o["Key"])
print("manifests",len(keys))
def rd(k):
    try: return k,json.loads(s3.get_object(Bucket=B,Key=k)["Body"].read())
    except Exception as e: return k,None
inp={}; ctxinfo=collections.defaultdict(dict); mans=collections.defaultdict(set)
with cf.ThreadPoolExecutor(16) as ex:
    for k,d in ex.map(rd,keys):
        if not isinstance(d,dict): continue
        cl=d.get("contexts"); cl=list(cl.values()) if isinstance(cl,dict) else cl
        if not isinstance(cl,list): continue
        mname=k[len(P+"manifests/"):].split("/")[0]
        for c in cl:
            if not isinstance(c,dict) or "context_sha256" not in c: continue
            cs=c["context_sha256"]; mans[cs].add(mname)
            for f in ("dataset_hash","fold","rights","cut_id","view_id","values_mode","class"):
                if c.get(f) is not None: ctxinfo[cs][f]=c[f]
            for o in c.get("objects") or []:
                if o.get("component") in ("jtt1-encoder-s0","jtt1-teacher-s0","jtt1-encoder"):
                    inp.setdefault(cs,{})[o["component"]]={kk:o.get(kk) for kk in ("key","sha256","version_id","bytes","research_only")}
rows=[]
for cs,d in inp.items():
    rows.append(dict(context_sha256=cs, **ctxinfo[cs], manifests=sorted(mans[cs]), inputs=d))
json.dump(rows,open(os.path.join(out,"jtt1_inventory.json"),"w"))
s0=[r for r in rows if "jtt1-encoder-s0" in r["inputs"]]
print("contexts with any jtt1",len(rows),"with encoder-s0",len(s0))
print("rights",collections.Counter(r.get("rights") for r in s0))
print("research_only obj",collections.Counter(r["inputs"]["jtt1-encoder-s0"].get("research_only") for r in s0))
print("g3 manifests",collections.Counter(m for r in rows for m in r["manifests"] if "g3" in m.lower()))
print("s0 w/ g3 manifest",sum(any("g3" in m.lower() for m in r["manifests"]) for r in s0))
print("datasets s0",len({r.get("dataset_hash") for r in s0}))
print("cut prefixes",collections.Counter((r.get("cut_id") or "").split(":")[0] for r in s0))
# G3 context list
g3=set()
for cs,ms in mans.items():
    if any("g3" in m.lower() for m in ms): g3.add(cs)
json.dump(sorted(g3),open(os.path.join(out,"g3_contexts.json"),"w")); print("g3 contexts in manifests",len(g3))
