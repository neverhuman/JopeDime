import json, hashlib, glob, re, sys
import numpy as np
from collections import defaultdict
B="/home/ubuntu/jope-data"
sel=json.load(open(f"{B}/hwave-h448/input/selected.json"))
train=[r["context_sha256"] for r in sel["train"]]; fam={r["context_sha256"]:r["family_id"] for r in sel["train"]+sel["dev"]}
an=json.load(open(f"{B}/hwave-h447/analysis_inc_train.json")); assert an["train"]==train
u=np.array(an["u_inc_train"]); orc=np.array(an["oracle"]); reg=orc-u
tr_fams=sorted({fam[c] for c in train}, key=lambda f: hashlib.sha256(f"h444/{f}".encode()).hexdigest())
fs={f:sum(1 for c in train if fam[c]==f) for f in tr_fams}; fof={}; load=[0]*5
for f in sorted(tr_fams,key=lambda f:-fs[f]):
    j=int(np.argmin(load)); fof[f]=j; load[j]+=fs[f]
fold=np.array([fof[fam[c]] for c in train])
out={"fold_sizes":load,"per_fold":{}}
for k in range(5):
    m=fold==k; fk=[f for f in tr_fams if fof[f]==k]; sz=sorted([fs[f] for f in fk],reverse=True)
    out["per_fold"][k]={"n":int(m.sum()),"n_fam":len(fk),"top3_fam_sizes":sz[:3],"max_fam_share":sz[0]/m.sum(),
      "inc_util_mean":float(u[m].mean()),"inc_gatefail":int((u[m]<=-2.0+1e-9).sum()),"regret_mean":float(reg[m].mean()),
      "regret_median":float(np.median(reg[m])),"frac_regret0":float((reg[m]<=1e-12).mean()),
      "regret_mean_excl_fail":float(reg[m][u[m]>-2+1e-9].mean())}
# cross-wave fold gains and per-fold W/L from fold_sig
fg=defaultdict(dict)
for w in ["h444","h445","h446","h447","h448"]:
    for p in sorted(set(glob.glob(f"{B}/hwave-{w}/results/fold_sig/*_cv*_seed*.json"))):
        mm=re.search(r"/([a-z_]+)_cv(\d)_seed(\d)\.json",p); a,k,s=mm.group(1),int(mm.group(2)),int(mm.group(3))
        o=json.load(open(p)); x=np.array(o["oof"])
        fg[f"{w}:{a}"].setdefault(s,{})[k]={"gain":round(float(x.mean()),4),"W":int((x>0).sum()),"L":int((x<0).sum())}
out["cross_wave"]=fg
# per-fold gain excluding incumbent gate-fail contexts (tail-free), H448 arms seed1
tf={}
for a in ["trust_rank","trust_regret","trust_pair"]:
    for s in [1,2,3]:
        for k in range(5):
            o=json.load(open(f"{B}/hwave-h448/results/fold_sig/{a}_cv{k}_seed{s}.json")); x=np.array(o["oof"]); m=fold==k
            uk=u[m]; nf=uk>-2+1e-9
            tf.setdefault(a,{}).setdefault(s,{})[k]={"gain_all":round(float(x.mean()),4),"gain_passing":round(float(x[nf].mean()),4),"fail_recov_contrib":round(float(x[~nf].sum()/len(x)),4),"W_pass":int((x[nf]>0).sum()),"L_pass":int((x[nf]<0).sum())}
out["h448_tail_split"]=tf
# fold-level correlation: gain vs regret
json.dump(out,open(sys.argv[1],"w"),indent=1)
print(json.dumps(out["per_fold"],indent=0))
for key,v in fg.items():
    print(key, {s:[v[s][k]["gain"] for k in sorted(v[s])] for s in sorted(v)})
for a,v in tf.items():
    print(a, "seed1", [(v[1][k]["gain_passing"],v[1][k]["fail_recov_contrib"],v[1][k]["W_pass"],v[1][k]["L_pass"]) for k in range(5)])
