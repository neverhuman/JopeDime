"""Read-only bounded metadata census. No source rows, component vectors or observations."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib,json,time
r=Path("/home/ubuntu/jope-dime/parallel")
started=time.time(); bindings=[]; errors=[]
def read(path):
    path=Path(path); data=path.read_bytes()
    bindings.append(dict(path=str(path.relative_to(r)),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
    return json.loads(data)
def maybe(path):
    try:return read(path)
    except Exception as e:errors.append(dict(path=str(path),error=str(e)));return None
def counts(rows):return dict(Counter(rows))
active_path=r/"runs/dataset-inclusion-20260909/correction-final/current-contexts.json"
active=read(active_path); rows=active["encodings"]
index_paths=sorted(set(list((r/"data/encodings").glob("*/index.json"))+list((r/"runs").glob("expansion-encode-*/contexts/index.json"))+list((r/"runs/dataset-inclusion-20260909").glob("**/*contexts.json"))+list((r/"manifests/generated").glob("*/contexts.json"))))
retained=set(); indexes=[]
for p in index_paths:
    v=maybe(p)
    if not v or v.get("schema")!="parallel.context_index.v1":continue
    keys={(x.get("dataset"),x.get("prepared_sha256"),x.get("cut")) for x in v["encodings"]}
    retained.update(keys);indexes.append(dict(path=str(p.relative_to(r)),rows=len(v["encodings"]),identities=len(keys),partitions=counts(x.get("partition") for x in v["encodings"])))
regs=[];admitted=defaultdict(list);sealed=set()
for p in sorted((r/"manifests/generated/families").glob("*.json")):
    v=maybe(p)
    if not v or v.get("schema")!="parallel.family_registry.v1":continue
    regs.append(dict(path=str(p.relative_to(r)),families=len(v["families"]),partitions=counts(f["partition"] for f in v["families"].values())))
    for name,f in v["families"].items():
        if f["partition"]=="meta_test":sealed.add(name)
        for d in f["datasets"]:
            admitted[d["dataset"]].append(dict(family=name,partition=f["partition"],eligible=bool(d.get("preparation_eligible")) and not f.get("holds"),registry=str(p.relative_to(r))))
prepared={}; active_prepares={x["prepared"] for x in rows}
for p in sorted({r/x/"prepared.json" for x in active_prepares}|set((r/"data/prepared").glob("expansion*/*/prepared.json"))):
    v=maybe(p)
    if v:prepared[str(p.parent.relative_to(r))]=(v,bindings[-1]["sha256"])
component_roots=sorted([p for p in (r/"data/encodings").iterdir() if p.is_dir() and p.name.startswith(("hyperion","foundation","target"))]+list((r/"runs").glob("expansion-encode-*/components/*")))
components=defaultdict(lambda:defaultdict(list))
for root in component_roots:
    for p in sorted(root.glob("*/cut-*/*.json")):
        if p.name not in ("hyperion.json","foundation.json","target.json"):continue
        v=maybe(p)
        if not v:continue
        key=(v.get("dataset"),v.get("prepared_sha256"),v.get("cut"),v.get("training_sha256"),v.get("preprocessing_sha256"))
        components[key][p.stem].append(dict(path=str(p.relative_to(r)),spec_sha256=v.get("spec_sha256"),signature_sha256=v.get("signature_sha256"),shape=v.get("shape"),family=v.get("family"),final_test_targets_used=v.get("final_test_targets_used")))
expansion=[]
for path,(p,sha) in prepared.items():
    if not path.split("/")[-2].startswith("expansion"):continue
    for c in p["cuts"]:
        key=(p["dataset"],sha,c["index"],c["training_sha256"],c["preprocessing_sha256"])
        cc=components.get(key,{})
        valid={kind:any(v["shape"]==[dim] and v["family"]==p["family"] and v["final_test_targets_used"] is False and isinstance(v["spec_sha256"],str) and len(v["spec_sha256"])==64 and isinstance(v["signature_sha256"],str) and len(v["signature_sha256"])==64 for v in cc.get(kind,[])) for kind,dim in [("hyperion",512),("foundation",2048),("target",512)]}
        historical=admitted.get(p["dataset"],[])
        eligible=[d for d in historical if d["eligible"] and d["family"]==p["family"] and d["partition"]==p["meta_partition"] and d["partition"] in ("meta_train","meta_validation","development_validation")]
        indexed=key[:3] in retained
        expansion.append(dict(dataset=p["dataset"],family=p["family"],task=p["task"],partition=p["meta_partition"],cut=c["index"],prepared=path,prepared_sha256=sha,training_sha256=c["training_sha256"],preprocessing_sha256=c["preprocessing_sha256"],retained_index=indexed,component_metadata_complete=all(valid.values()),missing_components=[k for k,v in valid.items() if not v],registered_eligible=bool(eligible),historical_registry_dispositions=historical,metadata_eligible_addition=bool(eligible) and p["task"]=="regression" and all(valid.values()) and not indexed))
active_key={(x["prepared"],x["cut"]):x for x in rows}
coverage=Counter();receipts=Counter();gens=[];changed=[];outside=[]
base=r/"runs/continuous-learning-v2-20260909"
state=read(base/"state.json")
for gen in sorted(base.glob("generation-*")):
    arms=[]
    for arm in ("neural","generic","shuffled"):
        path=gen/arm/"jobs.json"
        if not path.exists():continue
        jobs=read(path)["jobs"];completed=0
        for j in jobs:
            key=(j["prepared"],j["cut"]);row=active_key.get(key)
            if row is None or row["partition"] not in ("meta_train","meta_validation"):
                outside.append(dict(generation=gen.name,arm=arm,prepared=j["prepared"],cut=j["cut"]));continue
            coverage[(row["dataset"],row["family"],row["partition"],j["cut"])]+=1
            # Existence of a terminal receipt is evidence of publication, not strict admission.
            receipt=r/j["output"]/"search.json"
            if receipt.exists():completed+=1;receipts[(row["dataset"],j["cut"])]+=1
        arms.append(dict(arm=arm,jobs=len(jobs),terminal_receipts_present=completed,cuts=counts(j["cut"] for j in jobs)))
    gens.append(dict(generation=gen.name,arms=arms))
geometry=[]
for path,(p,sha) in prepared.items():
    if path not in active_prepares:continue
    width=len(p.get("storage",{}).get("feature_columns",[]))
    for c in p["cuts"]:
        if (path,c["index"]) in active_key:geometry.append(dict(dataset=p["dataset"],family=p["family"],task=p["task"],partition=p["meta_partition"],cut=c["index"],train_rows=len(c["train_indices"]),validation_rows=len(c["validation_indices"]),width=width,entity_columns=p.get("entity_columns"),clean_status=p.get("clean_status"),frozen=p.get("family_partition_frozen"),train_validation_index_overlap=len(set(c["train_indices"])&set(c["validation_indices"]))))
report=dict(schema="parallel.redteam_metadata_audit.v1",started_at=started,completed_at=time.time(),scope="metadata-only named roots, index glob scopes and expansion prepared directories; not a full-project or numerical verification",protected_values_opened=False,index_scope=["data/encodings/*/index.json","runs/expansion-encode-*/contexts/index.json","runs/dataset-inclusion-20260909/**/*contexts.json","manifests/generated/*/contexts.json"],component_roots=[str(p.relative_to(r)) for p in component_roots],active=dict(rows=len(rows),datasets=len({x["dataset"] for x in rows}),families=len({x["family"] for x in rows}),partitions=counts(x["partition"] for x in rows),eligible_rows=sum(x["partition"] in ("meta_train","meta_validation") for x in rows)),indexes=indexes,registries=regs,sealed_family_union=sorted(sealed),component_identity_groups=len(components),component_complete_groups=sum(all(k in v for k in ("hyperion","foundation","target")) for v in components.values()),component_groups_outside_retained_indexes=sum(all(k in v for k in ("hyperion","foundation","target")) and key[:3] not in retained for key,v in components.items()),expansion=dict(prepares=sum(p.split("/")[-2].startswith("expansion") for p in prepared),cuts=len(expansion),tasks=counts(x["task"] for x in expansion),component_metadata_complete=sum(x["component_metadata_complete"] for x in expansion),outside_retained_indexes=sum(not x["retained_index"] for x in expansion),registered_eligible=sum(x["registered_eligible"] for x in expansion),metadata_eligible_additions=sum(x["metadata_eligible_addition"] for x in expansion)),expansion_rows=expansion,coverage=dict(generations=len(gens),declared_jobs=sum(coverage.values()),distinct_cuts=len(coverage),distinct_datasets=len({k[0] for k in coverage}),terminal_receipts_present=sum(receipts.values()),minimum_attempts_per_cut=min(coverage.values()),maximum_attempts_per_cut=max(coverage.values()),cuts=counts(k[3] for k in coverage),rows=[dict(dataset=k[0],family=k[1],partition=k[2],cut=k[3],attempts=v,terminal_receipts_present=receipts[(k[0],k[3])]) for k,v in sorted(coverage.items())],outside_active=outside),generations=gens,geometry=geometry,configured_generation_cuts=counts(str((g.get("cuts"),g.get("cut_start"))) for g in state["generations"]),strict_admitted_fresh_observations="unavailable: authority witness/completion/label join not performed",bindings=bindings,errors=errors)
print(json.dumps(report,sort_keys=True))
