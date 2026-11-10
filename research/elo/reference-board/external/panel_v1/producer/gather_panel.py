import json,glob,os,collections,sys
sys.path.insert(0,os.path.expanduser("~/jope_plan/elo"))
from elo import fit
R=os.path.expanduser("~/jope_plan/elo/panel_results")
panel=json.load(open(os.path.expanduser("~/jope_plan/elo/panel_v1.json")))
meta={d["id"]:d for d in panel["datasets"]}
recs={}
for f in sorted(glob.glob(R+"/*/*.jsonl")):
    for l in open(f):
        l=l.strip()
        if not l: continue
        r=json.loads(l)
        if r.get("status")!="ok" or r.get("value") is None or "tabpfn" in r["method"].lower(): continue
        recs[(r["dataset"],r["method"])]=r
out={}
for task,minimize in (("regression",True),("binary",False)):
    rows=[(d,m,r["value"]) for (d,m),r in recs.items() if meta.get(d,{}).get("task")==task]
    table,nd=fit(rows,minimize=minimize)
    out[task]={"n_datasets":nd,"n_families":len({meta[d]["family"] for d,_,_ in rows}),"elo":[{"method":m,"elo":round(e,1),"coverage":c} for m,e,c in table]}
    print(f"== {task}: {nd} datasets, {out[task]['n_families']} families")
    for i,(m,e,c) in enumerate(table,1): print(f"{i:2d} {m:22s} {e:7.1f} cov={c}")
json.dump(out,open(R+"/elo_panel_v1.json","w"),indent=1)
with open(R+"/all_results.jsonl","w") as f:
    for r in recs.values(): f.write(json.dumps(r)+"\n")
