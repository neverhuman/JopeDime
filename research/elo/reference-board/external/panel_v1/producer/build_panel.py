import json,glob,re,os,hashlib,random,collections
inv=json.load(open("/home/ubuntu/jope_plan/ns/s3_manifest_inventory.json"))["items"]
excl={}
def add(ids,why):
    for i in ids: excl.setdefault(i,why)
m=json.load(open("/mnt/jope-archive/flywheel-20261023/corpus/corpus_manifest.json"))["records"]
add([r["dataset_id"] for r in m],"flywheel-corpus")
a=json.load(open("/home/ubuntu/gpu_bench/fused/bridge_provenance_audit.json"))["selected_datasets"]
add([r["dataset_id"] for r in a],"fused-bridge-train/val")
for p in glob.glob("/home/ubuntu/gpu_bench/runs/infer_v1_cohort915/*.json")+glob.glob("/home/ubuntu/gpu_bench/runs/infer_v1_cohort915/*manifest*"):
    try: add(re.findall(r'"dataset_id"\s*:\s*"([0-9a-f]{16})"',open(p).read()),"cohort915")
    except Exception: pass
g=json.load(open("/home/ubuntu/JopeDime/assets/regression/sotarena-tabular-regression-v2/registry/datasets.tabular-regression-v2.json"))["datasets"]
add([d["id"] for d in g],"golden45")
excl_fam=set(it["family_key"] for it in inv if it["id"] in excl and it.get("family_key"))
rng=random.Random(20261029); panel=[]; stats=collections.Counter()
for task,cap in (("regression",240),("binary",240)):
    by=collections.defaultdict(list)
    for it in inv:
        if it["task"]!=task or it.get("origin")!="real" or it.get("has_meta_test"): continue
        if it["id"] in excl: stats[task+":excl_id"]+=1; continue
        fam=it.get("family_key") or ("ds:"+it["id"])
        if fam in excl_fam: stats[task+":excl_family"]+=1; continue
        if not (50<=it.get("train_rows",0)<=50000 and it.get("test_rows",0)>=20): stats[task+":size"]+=1; continue
        by[fam].append(it)
    fams=sorted(by); rng.shuffle(fams); fams=fams[:cap]
    for f in sorted(fams):
        it=sorted(by[f],key=lambda x:x["id"])[0]
        h=int(hashlib.sha256(f.encode()).hexdigest(),16)%5
        panel.append(dict(id=it["id"],task=task,family=f,fold=h,train_rows=it["train_rows"],test_rows=it["test_rows"],collections=it["collections"],platform=it.get("platform"),manifest_sha256=it["manifest_sha256"],manifest_version_id=it["version_id"]))
    stats[task+":families_available"]=len(by); stats[task+":selected"]=len(fams)
out=dict(schema="jope.elo-panel/v1",created="2026-10-29",seed=20261029,excluded_dataset_count=len(excl),excluded_family_count=len(excl_fam),exclusion_reasons=collections.Counter(excl.values()),stats=stats,datasets=panel)
json.dump(out,open("/home/ubuntu/jope_plan/elo/panel_v1.json","w"),indent=1)
print(json.dumps({k:v for k,v in out.items() if k!="datasets"},indent=1))
print("sha256",hashlib.sha256(open("/home/ubuntu/jope_plan/elo/panel_v1.json","rb").read()).hexdigest())
