# [OPUS] Evidence transcript (docs-only; probe sources inlined below)

Host xbabe2, CPU only (`nice -n 15`, OMP_NUM_THREADS=4), Python 3 `-I`, scikit-learn 1.4.1.post1.
Input packet: `~/jope-data/h50/input` (`source_manifest.json`: DL-Train 1219 ctx / 205 fam, DL-Val
91 / 19, holdout excluded, outer rows 0). **No DL-Val label was read by any OPUS probe**; DL-Val
numbers quoted in FINDINGS come from H56/H58 receipts. Sealed/holdout: 0 opened. GPU: none.

## E1 train_baselines.py — exit 0
```
$ cd ~/jope-data/h50/input && nice -n 15 python3 -I .../opus/train_baselines.py .
dev contexts 91 dev families 19 max ctx/family 5 sizes [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 1]
pool keys ['program', 'fitness', 'logratio', 'gate', 'program_sha256', 'birth_seq_min', 'birth_seq_max', 'generation_mask', 'delivery_count']
DL-Train n=1219  best-of-revealed4=1.483107  oracle=1.905853  mean=0.827791  slot0=0.877697  pool len min/max 1151/1152
```
(The "dev" line reads only family IDs from selected.json, no labels.)

## E2 admissions.csv null distribution — exit 0
```
$ gawk -F, 'NR>1{...} END{...}' docs/whitepaper/figures/data/admissions.csv
rounds 695 distinct candidate values 39
post-r882 rounds 587 mean 1.133918 sd 0.002103  >=r882: 0
217 1.1352490 | 158 1.1344341 | 124 1.1304714 | 17 1.1377131 | 14 1.1329746 | 11 1.1354388 | 10 1.1364931 ...
p50 1.1344340853025057 p95 1.136493097959418 p99 1.1377131278566008 max 1.1377131278566008
$ gawk (rounds.csv) -> n=25 corr(train,val)=0.734 train mean 1.42225 sd 0.00223 val mean 1.13413 sd 0.00141
```

## E3 H52 family-gain noise scale — exit 0
Per-arm/seed mean, SD, SE and nonzero counts of `family_gains_vs_r882` over 205 families
(`~/jope-data/h52/runs/*/seed*/*-family-gains.json`). Examples: CONTROL-k2-lr1e-4 SD 0.0248,
31/205 nonzero; CONTROL-k16-lr1e-4 SD ≈0.196, 138–141 nonzero; ANCHOR/k16-lr3e-4 SD 0.29–0.31.

## E4 alias_ceiling.py (200 ctx) — exit 0
```
oracle 1.9913 mean 0.8078
round to 3/2/1 dp: ceiling 1.9913 (oracle loss 0.0000); distinct buckets 1152/1152/1150
best candidate's feature-NN: median dist/median-dist 0.1877 ; frac NN dist<1% of median 0.035 ; median u gap to NN 0.0277 ; frac gap>0.1 0.290
```

## E5 loss_anatomy.py (400 ctx) — exit 0
```
target var total 2.4747 | between-context 1.1147 (45%) | within 1.3600
SmoothL1 share from gate-fail targets (median-predictor proxy): 23%  (fail frac of slots 0.032)
KL target softmax(-t/0.2): mass on revealed slots 0..3 median 0.044 mean 0.099 ; on slot0 0.037 ; effective #cands 120.4
top1..top5 cost spread median 0.0057  p10 0.0005
```

## E6 listwise_temp.py (400 ctx) — exit 0
```
median pass-std 0.305  MAD 0.234
H58-D: top1-top5 gap in std units 0.0318 ; weight ratio top1/top5 at tau .10 = 1.374, at .25 = 1.136 ; eff #cands 51.2 / 76.6
H59 NORM (MAD-scaled, KL tau .2): top1-top5 gap 0.0374 MAD units ; weight ratio 1.206
frac contexts where H58-D tau.10 ratio top1/top5 > 2: 0.270
```

## E7 learnability_probe.py (96 unrevealed samples/ctx; folds = sha256(family) mod 5) — exit 0, ~3 min
```
contexts 1219 fold sizes [285 175 265 274 220]
families 205
U0 1.4582 | U1 1.5787 | U4 1.6568 | U16 1.7358 | R4 1.5746 | R16 1.6740 | TOP1_unverified 1.1933 | ORACLE 1.8525 | spearman 0.6761
U1 - U0: mean +0.1205  95% CI [+0.0704, +0.1867]
U4 - U0: mean +0.1985  95% CI [+0.1221, +0.2977]
U4 - R4: mean +0.0822  95% CI [+0.0313, +0.1520]
U16 - R16: mean +0.0618  95% CI [+0.0278, +0.1054]
TOP1_unverified - U0: mean -0.2649  95% CI [-0.3625, -0.1664]
```

## E8 learnability_ablation.py estimator.* only (5 cols) — exit 0
```
U0 1.4582 | U1 1.5305 | U4 1.6198 | U16 1.6864 | R4 1.5746 | R16 1.6740 | TOP1_unverified 0.7612 | spearman 0.2051
U4 - R4: +0.0452 [+0.0041, +0.1022] ; U16 - R16: +0.0124 [-0.0191, +0.0492]
```

## E9 H58 receipts (read-only) and h58_prior_check.py — exit 0
```
$ jq ... ~/jope-data/h58/runs/cell_receipts.json (per-arm means over 18 cells)
A update_l2 0.257 grad 0.774 loss 3.414->3.387 rho 0.126 trainable 29211
B update_l2 0.305 grad 0.921 loss 6.425->6.342 rho 0.178 trainable 23546
C update_l2 0.306 grad 0.904 loss 6.604->6.486 rho 0.179 gp_admitted 720 harvested 116076
D update_l2 0.295 grad 7.012 loss 38.50->38.18 rho 0.180 gp_admitted 720 harvested 116509
$ jq t537_gates_native: B_vs_INIT_G0 +0.00028 tied 169 ; C_vs_INIT_G0 +0.00053 tied 172 ; D_vs_INIT_G0 +0.00198 tied 178 ; D_vs_C +0.00145 tied 187
$ python3 -I h58_prior_check.py ~/jope-data/h50/input ~/jope-data/h58/cache
base_prior_cost: rho 0.1840 | U0 1.4582 | U1 1.4799 | U4 1.5115 | R4 1.5843 | U4-R4 -0.0728 CI [-0.1189, -0.0402]
sos_prior[:,0]:  rho 0.2614 | U4 1.5630 | U4-R4 -0.0213 CI [-0.0656, +0.0282]
sos_prior[:,1]:  rho 0.0014 | U4 1.5737 | U4-R4 -0.0106 CI [-0.0309, +0.0097]
```
(R4 differs slightly between probes because the random draws differ.)

## E10 generation_confound.py (96/ctx) — exit 0
```
U0 1.4582 | U4 1.6423 | R4 1.5756 | B4 (birth-order) 1.4911 | Brho 0.1151 | W4 (within-generation) 1.5445 | WR4 1.4861 | Wrho 0.5993 | spearman 0.6734
U4 - R4: +0.0666 [+0.0264, +0.1201]
B4 - R4: -0.0845 [-0.1162, -0.0573]
U4 - B4: +0.1511 [+0.0925, +0.2290]
W4 - WR4: +0.0584 [+0.0336, +0.0930]
```

## E11 crossview_pairs.py — exit 0
```
all pairs: n=140418  SD((x-y)/sqrt2)=0.6371  corr(x,y)=0.919
   |x| in [0.000,0.005): n= 31701  sign agree 0.678
   |x| in [0.005,0.020): n= 16404  sign agree 0.734
   |x| in [0.020,0.050): n= 14921  sign agree 0.783
   |x| in [0.050,0.200): n= 26785  sign agree 0.859
   |x| in [0.200,9.000): n= 46425  sign agree 0.914
both in top25% of both views: n=35097  SD((x-y)/sqrt2)=0.4014  corr(x,y)=0.826
   |x| in [0.000,0.005): n= 14687  sign agree 0.728
   |x| in [0.005,0.020): n=  5172  sign agree 0.779
   |x| in [0.020,0.050): n=  3870  sign agree 0.835
   |x| in [0.050,0.200): n=  4211  sign agree 0.881
   |x| in [0.200,9.000): n=  6686  sign agree 0.920
```
(mean y/x columns omitted: unstable for x near 0.)

## E12 learnability_probe_rep.py (192/ctx, rng 11, model seeds 100+k) — exit 0
```
U0 1.4582 | U1 1.5797 | U4 1.6533 | U16 1.7329 | R4 1.5832 | R16 1.6902 | TOP1_unverified 1.1540 | spearman 0.6818
U1 - U0: +0.1215 [+0.0695, +0.1930] ; U4 - U0: +0.1950 [+0.1247, +0.2839]
U4 - R4: +0.0700 [+0.0292, +0.1189] ; U16 - R16: +0.0427 [+0.0203, +0.0717]
TOP1_unverified - U0: -0.3042 [-0.3878, -0.2275]
```

## E13 context_probe.py prog vs ctx (96/ctx, rng 11, identical sampled rows) — exit 0 each
```
prog: U1 1.5581 | U4 1.6497 | R4 1.5736 | rho 0.6735 | U4-R4 +0.0762 [+0.0306, +0.1352]
ctx : U1 1.6110 | U4 1.6835 | R4 1.5736 | rho 0.6708 | U4-R4 +0.1099 [+0.0561, +0.1770]
paired ctx - prog (205 families, 10k family bootstrap):
U1  +0.0529 [+0.0231, +0.0942] W/T/L 115/34/56 median +0.0011
U4  +0.0337 [+0.0116, +0.0589] W/T/L 134/7/64 median +0.0031
rho -0.0027 [-0.0165, +0.0103]
U4 family score 0.671 -> Elo +123.6
```

## E14 context_probe_rep.py (rng 23, model seeds 200+k) — exit 0
```
prog: rho 0.6750 | R4 1.5653 | U4-R4 +0.0827 [+0.0357, +0.1441] | U4 1.6480
ctx : rho 0.6726 | R4 1.5653 | U4-R4 +0.1122 [+0.0548, +0.1850] | U4 1.6774
paired ctx - prog: U1 +0.0526 [+0.0257, +0.0895] W/T/L 135/23/47 ; U4 +0.0294 [+0.0088, +0.0554] W/T/L 133/6/66
```

## E15 crossview_noise.py — exit 0
```
shared passing program pairs 27967
all      |du|: n=27967 median 0.0512 p25 0.0163 p75 0.1918 frac>0.0043 0.928
top10% either: n=4336 median 0.0730 p25 0.0210 p75 0.3027 frac>0.0043 0.940
top10% both  : n=1813 median 0.0615 p25 0.0193 p75 0.1873 frac>0.0043 0.928
```
(Raw cross-view gaps include view difficulty offsets; E11 is the offset-free version used in GT-O14.)

## E16 promotions.jsonl margins — exit 0
```
$ jq ... results/fused-20261029/promotions.jsonl
95 lines: 85 historical_admission, 10 promotion; promotion margin (new-old utility) median 0.00428 p25 0.00116 p75 0.01687 min 4.86e-04 max 0.1812
```

## Corrections and withdrawals (kept for the record)
- GT-O5 "KL target concentrated on revealed slots": withdrawn (E5: ~120 effective candidates).
- "H58 arm C underpowered (<=40 rows)": withdrawn (E9: 116k harvested negatives).
- GT-O8 "P(accept) < 0.005": IID-only bound; the fact is 0/587 descriptive.
- D-O2 multiplicative threshold: wrong sign for negative costs; replaced by additive, verified U_k.
- D-O4 0.09 overfit number: illustrative worst-case scaling, not measured.
- I-O5 prior weakened by GT-O14 (73% sign replication at small margins).
- Feature aliasing hypothesis: refuted (E4).
- Generation-detector concern about GT-O10: refuted (E10).

## Probe sources (verbatim) and SHA-256
Scratch outputs live outside Git (session scratchpad); these sources reproduce every number above.

### train_baselines.py (sha256 5aaeb6e5f0ae6537e40894724c90751a7ebf4677a0aafb2c1c706a818225eea2)
```python
"""DL-Train only: revealed-4 baseline vs oracle vs mean; family counts. No DL-Val labels read."""
import json, sys
from collections import defaultdict
import numpy as np
root = sys.argv[1]
sel = json.load(open(f"{root}/selected.json"))
fam_dev = defaultdict(int)
for r in sel["dev"]:
    fam_dev[r["family_id"]] += 1
print("dev contexts", len(sel["dev"]), "dev families", len(fam_dev), "max ctx/family", max(fam_dev.values()),
      "sizes", sorted(fam_dev.values(), reverse=True))
rows = []
for r in sel["train"]:
    p = np.load(f"{root}/pools/{r['context_sha256']}.npz", allow_pickle=False)
    f, g = p["fitness"].astype(float), p["gate"].astype(bool)
    u = np.where(g, -f, -2.0)
    rows.append((r["family_id"], u[:4].max(), u.max(), u.mean(), u[0], len(u)))
if rows:
    k = list(np.load(f"{root}/pools/{sel['train'][0]['context_sha256']}.npz").keys()); print("pool keys", k)
a = np.array([x[1:] for x in rows])
print("DL-Train n=%d  best-of-revealed4=%.6f  oracle=%.6f  mean=%.6f  slot0=%.6f  pool len min/max %d/%d" % (
    len(a), a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean(), a[:, 3].mean(), a[:, 4].min(), a[:, 4].max()))
```

### alias_ceiling.py (sha256 949ddfec35786ee065c2d42ee0af5283b30b34c6406772605d79ddd5884857d6)
```python
"""DL-Train only. Feature-aliasing ceiling: if candidates with (near-)identical 412-d program features
have different utilities, no function of those features can separate them. Ceiling(eps) =
mean_c max_cluster mean(u in cluster), clusters = connected groups at standardized distance < eps
(greedy leader clustering on utility-sorted order is avoided: we use exact rounding buckets)."""
import json, sys
import numpy as np
root, nctx = sys.argv[1], int(sys.argv[2])
sel = json.load(open(f"{root}/selected.json"))
rng = np.random.default_rng(0)
ctx = [sel["train"][i] for i in rng.choice(len(sel["train"]), nctx, replace=False)]
P = [np.load(f"{root}/pools/{r['context_sha256']}.npz", allow_pickle=False) for r in ctx]
X = np.concatenate([p["program"][:1152] for p in P]).astype(np.float64)
mu, sd = X.mean(0), X.std(0); keep = sd > 1e-9
print("feature dims", X.shape[1], "non-constant", int(keep.sum()))
res = {d: [] for d in (None, 3, 2, 1)}
nn_stats = []
for p in P:
    u = np.where(p["gate"].astype(bool), -p["fitness"].astype(float), -2.0)[:1152]
    Z = ((p["program"][:1152].astype(np.float64) - mu) / np.where(keep, sd, 1))[:, keep]
    for dec in res:
        if dec is None:
            res[dec].append((u.max(), u.max(), u.mean())); continue
        keys = np.round(Z, dec)
        _, inv = np.unique(keys, axis=0, return_inverse=True)
        inv = inv.ravel(); s = np.bincount(inv, u); n = np.bincount(inv)
        res[dec].append((u.max(), (s / n).max(), len(n)))
    # nearest neighbour aliasing: top-1 candidate vs its feature NN
    i = int(np.argmax(u)); d = np.linalg.norm(Z - Z[i], axis=1); d[i] = np.inf; j = int(np.argmin(d))
    med = np.median(np.linalg.norm(Z - Z[i], axis=1))
    nn_stats.append((d[j] / med, u[i] - u[j]))
for dec, rows in res.items():
    a = np.array(rows)
    if dec is None:
        print("oracle %.4f mean %.4f" % (a[:, 0].mean(), a[:, 2].mean()))
    else:
        print("round to %d dp (std units): ceiling %.4f  (oracle loss %.4f)  mean #distinct feature buckets %.0f/1152" % (
            dec, a[:, 1].mean(), a[:, 0].mean() - a[:, 1].mean(), a[:, 2].mean()))
nn = np.array(nn_stats)
print("best candidate's feature-NN: median dist/median-dist %.4f ; frac NN dist<1%% of median %.3f ; median u gap to NN %.4f ; frac gap>0.1 %.3f" % (
    np.median(nn[:, 0]), (nn[:, 0] < 0.01).mean(), np.median(nn[:, 1]), (nn[:, 1] > 0.1).mean()))
```

### loss_anatomy.py (sha256 396c8ef0d00b0143dd0430473b7dcad10ff41224d2aded829d5e9bb4370bc9e9)
```python
"""DL-Train only. Mimic training_indices (top-48 by fitness + 144 random rest) and the fused loss targets
(target = fitness, gate-fail -> +2.0). Report: variance decomposition (between-context offsets vs within),
gate-fail share of a SmoothL1-at-constant-prediction proxy, KL target softmax(-t/0.2) mass on revealed slots 0..3,
and the spread between best revealed and best unrevealed (the decision-relevant quantity)."""
import json, sys
import numpy as np
root, nctx = sys.argv[1], int(sys.argv[2])
sel = json.load(open(f"{root}/selected.json"))
rng = np.random.default_rng(1)
idx = rng.choice(len(sel["train"]), nctx, replace=False)
T, rev_mass, top1_mass, eff_n, fail_frac, gap_rev_unrev, spread_top5 = [], [], [], [], [], [], []
for i in idx:
    p = np.load(f"{root}/pools/{sel['train'][i]['context_sha256']}.npz", allow_pickle=False)
    f, g = p["fitness"].astype(float), p["gate"].astype(bool)
    order = np.argsort(f)
    rest = rng.choice(order[48:], 144, replace=False)
    s = np.concatenate([order[:48], rest])
    t = np.where(g[s], f[s], 2.0)
    T.append(t)
    w = np.exp(-(t - t.min()) / 0.2); w /= w.sum()
    rev_mass.append(w[:4].sum()); top1_mass.append(w[0]); eff_n.append(1 / np.sum(w ** 2))
    fail_frac.append((~g[s]).mean())
    gap_rev_unrev.append(t[4:].min() - t[:4].min())  # >=0 always under sorted reveal
    tt = np.sort(t); spread_top5.append(tt[4] - tt[0])
T = np.array(T)
cm = T.mean(1, keepdims=True)
tot = T.var(); between = cm.var(); within = (T - cm).var()
med = np.median(T, axis=1, keepdims=True)
# SmoothL1 (beta=1) if model predicted each context's median perfectly: share of loss from fail targets
r = np.abs(T - med); sl1 = np.where(r < 1, 0.5 * r ** 2, r - 0.5)
print("contexts", nctx, "slate 192")
print("target var total %.4f | between-context %.4f (%.0f%%) | within %.4f" % (tot, between, 100 * between / tot, within))
print("SmoothL1 share from gate-fail targets (median-predictor proxy): %.0f%%  (fail frac of slots %.3f)" % (
    100 * sl1[T == 2.0].sum() / sl1.sum(), np.mean(fail_frac)))
print("KL target softmax(-t/0.2): mass on revealed slots 0..3 median %.3f mean %.3f ; on slot0 %.3f ; effective #cands %.1f" % (
    np.median(rev_mass), np.mean(rev_mass), np.mean(top1_mass), np.median(eff_n)))
print("best unrevealed - best revealed cost (train slates, >=0 by construction): median %.4f ; frac exactly 0 %.3f" % (
    np.median(gap_rev_unrev), np.mean(np.array(gap_rev_unrev) == 0)))
print("top1..top5 cost spread median %.4f  p10 %.4f" % (np.median(spread_top5), np.percentile(spread_top5, 10)))
```

### listwise_temp.py (sha256 65fde1bab4a2f35f872e70d275a2f8ef0f0ebf2d9d5b32c4c68c59f5161d191f)
```python
"""DL-Train only. For H58-D (r~ = (c - min_pass c)/max(std_pass c, .05), tau .10/.25) and H59 NORM (MAD scale),
how distinguishable are the top candidates in the softmax target? Reports top1/top5 weight ratio and the
effective number of candidates. Also a rank-based alternative (PL on ranks) for contrast."""
import json, sys
import numpy as np
root, nctx = sys.argv[1], int(sys.argv[2])
sel = json.load(open(f"{root}/selected.json"))
rng = np.random.default_rng(2)
out = {k: [] for k in ("std", "mad", "r_std", "r_mad", "neff10", "neff25", "ratio10", "ratio25", "ratio_mad02")}
for i in rng.choice(len(sel["train"]), nctx, replace=False):
    p = np.load(f"{root}/pools/{sel['train'][i]['context_sha256']}.npz", allow_pickle=False)
    f, g = p["fitness"].astype(float), p["gate"].astype(bool)
    order = np.argsort(f); s = np.concatenate([order[:48], rng.choice(order[48:], 144, replace=False)])
    c, gs = f[s], g[s]
    cp = c[gs]; sd = max(cp.std(), 0.05)
    t = np.where(gs, c, 2.0); mad = max(1.4826 * np.median(np.abs(t - np.median(t))), 0.05)
    rt = np.where(gs, np.clip((c - cp.min()) / sd, 0, 6), 8.0)
    srt = np.sort(rt)
    for tau, nk, rk in ((0.10, "neff10", "ratio10"), (0.25, "neff25", "ratio25")):
        w = np.exp(-rt / tau); w /= w.sum(); out[nk].append(1 / np.sum(w ** 2)); out[rk].append(np.exp((srt[4] - srt[0]) / tau))
    tt = np.sort(t); out["ratio_mad02"].append(np.exp((tt[4] - tt[0]) / mad / 0.2))
    out["std"].append(sd); out["mad"].append(mad); out["r_std"].append(srt[4] - srt[0]); out["r_mad"].append((tt[4] - tt[0]) / mad)
m = {k: np.median(v) for k, v in out.items()}
print("median pass-std %.3f  MAD %.3f" % (m["std"], m["mad"]))
print("H58-D: top1-top5 gap in std units %.4f ; weight ratio top1/top5 at tau .10 = %.3f, at .25 = %.3f ; eff #cands %.1f / %.1f" % (
    m["r_std"], m["ratio10"], m["ratio25"], m["neff10"], m["neff25"]))
print("H59 NORM (MAD-scaled, KL tau .2): top1-top5 gap %.4f MAD units ; weight ratio %.3f" % (m["r_mad"], m["ratio_mad02"]))
print("frac contexts where H58-D tau.10 ratio top1/top5 > 2: %.3f" % np.mean(np.array(out["ratio10"]) > 2))
```

### learnability_probe.py (sha256 20aef7f7052c78a65ab7a24ae0e17ea7138583f476e9724968d8058fc10c6ba2)
```python
"""DL-Train ONLY, family-grouped 5-fold CV (assignment: sha256(family_id) mod 5; not seed-4900 — diagnostic only).
Question: is there held-out-family signal in program features (412-d) for WITHIN-CONTEXT ranking, enough to beat
the zero-parameter best-of-revealed (storage slots 0..3) floor under a budget-matched verified rule?
Model: HistGradientBoostingRegressor on target = u_ci - mean_c(u) (offset-free), sampled candidates per context.
Metrics on held-out families (context-mean, then family-mean):
  U0   = best revealed
  Uk   = max(best revealed, max true u over model top-k among unrevealed)  k=1,4  (verified, budget k)
  A1   = unverified argmin rule as in live utility(): model top-1 among unrevealed vs revealed by *predicted* value
  ORACLE, RANDOMk (k random unrevealed, verified)
Prints paired family-bootstrap LCB95 for Uk - U0 and Uk - RANDOMk."""
import hashlib, json, sys
from collections import defaultdict
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
root = sys.argv[1]; per_ctx = int(sys.argv[2])
sel = json.load(open(f"{root}/selected.json"))
rng = np.random.default_rng(7)
fold = lambda fam: int(hashlib.sha256(fam.encode()).hexdigest(), 16) % 5
ctx = []
for r in sel["train"]:
    p = np.load(f"{root}/pools/{r['context_sha256']}.npz", allow_pickle=False)
    u = np.where(p["gate"].astype(bool), -p["fitness"].astype(float), -2.0)
    ctx.append(dict(fam=r["family_id"], fold=fold(r["family_id"]), X=p["program"].astype(np.float32), u=u))
print("contexts", len(ctx), "fold sizes", np.bincount([c["fold"] for c in ctx]))
rows = defaultdict(lambda: defaultdict(list))
for k in range(5):
    Xs, ys = [], []
    for c in ctx:
        if c["fold"] == k:
            continue
        idx = 4 + rng.choice(len(c["u"]) - 4, per_ctx, replace=False)
        Xs.append(c["X"][idx]); ys.append(c["u"][idx] - c["u"].mean())
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_leaf_nodes=63, l2_regularization=1.0,
                                      random_state=k).fit(np.concatenate(Xs), np.concatenate(ys))
    for c in ctx:
        if c["fold"] != k:
            continue
        u = c["u"]; un = np.arange(4, len(u)); pred = m.predict(c["X"][4:])
        o = un[np.argsort(-pred)]; u0 = u[:4].max()
        r_ = rows[c["fam"]]
        r_["U0"].append(u0); r_["U1"].append(max(u0, u[o[0]])); r_["U4"].append(max(u0, u[o[:4]].max()))
        r_["U16"].append(max(u0, u[o[:16]].max()))
        r_["R4"].append(max(u0, u[rng.choice(un, 4, replace=False)].max()))
        r_["R16"].append(max(u0, u[rng.choice(un, 16, replace=False)].max()))
        r_["TOP1_unverified"].append(u[o[0]]); r_["ORACLE"].append(u.max())
        r_["spearman"].append(np.corrcoef(np.argsort(np.argsort(pred)), np.argsort(np.argsort(u[4:])))[0, 1])
    print("fold", k, "done", flush=True)
fams = sorted(rows)
F = {key: np.array([np.mean(rows[f][key]) for f in fams]) for key in rows[fams[0]]}
print("families", len(fams))
for key in ("U0", "U1", "U4", "U16", "R4", "R16", "TOP1_unverified", "ORACLE", "spearman"):
    print("%-16s family-mean %.4f" % (key, F[key].mean()))
def lcb(d, B=10000):
    b = np.random.default_rng(0).choice(len(d), (B, len(d))); return np.percentile(d[b].mean(1), [2.5, 97.5])
for a, b in (("U1", "U0"), ("U4", "U0"), ("U4", "R4"), ("U16", "R16"), ("TOP1_unverified", "U0")):
    d = F[a] - F[b]; lo, hi = lcb(d)
    print("%s - %s: mean %+.4f  95%% CI [%+.4f, %+.4f]" % (a, b, d.mean(), lo, hi))
```

### learnability_ablation.py (sha256 1c858994ef8ccf3ad3b713e918f667b3209f70f3638af7fce776258bfb2192cd)
```python
"""DL-Train ONLY, family-grouped 5-fold CV (assignment: sha256(family_id) mod 5; not seed-4900 — diagnostic only).
Question: is there held-out-family signal in program features (412-d) for WITHIN-CONTEXT ranking, enough to beat
the zero-parameter best-of-revealed (storage slots 0..3) floor under a budget-matched verified rule?
Model: HistGradientBoostingRegressor on target = u_ci - mean_c(u) (offset-free), sampled candidates per context.
Metrics on held-out families (context-mean, then family-mean):
  U0   = best revealed
  Uk   = max(best revealed, max true u over model top-k among unrevealed)  k=1,4  (verified, budget k)
  A1   = unverified argmin rule as in live utility(): model top-1 among unrevealed vs revealed by *predicted* value
  ORACLE, RANDOMk (k random unrevealed, verified)
Prints paired family-bootstrap LCB95 for Uk - U0 and Uk - RANDOMk."""
import hashlib, json, sys
from collections import defaultdict
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
root = sys.argv[1]; per_ctx = int(sys.argv[2]); cols = sys.argv[3]
sel = json.load(open(f"{root}/selected.json"))
rng = np.random.default_rng(7)
names = json.load(open("/home/ubuntu/JopeDimeMicro/assets/projections/program-feature-names.json"))
COLS = [i for i, n in enumerate(names) if cols == "all" or n.startswith(cols)]
print("columns", len(COLS))
fold = lambda fam: int(hashlib.sha256(fam.encode()).hexdigest(), 16) % 5
ctx = []
for r in sel["train"]:
    p = np.load(f"{root}/pools/{r['context_sha256']}.npz", allow_pickle=False)
    u = np.where(p["gate"].astype(bool), -p["fitness"].astype(float), -2.0)
    ctx.append(dict(fam=r["family_id"], fold=fold(r["family_id"]), X=p["program"].astype(np.float32)[:, COLS], u=u))
print("contexts", len(ctx), "fold sizes", np.bincount([c["fold"] for c in ctx]))
rows = defaultdict(lambda: defaultdict(list))
for k in range(5):
    Xs, ys = [], []
    for c in ctx:
        if c["fold"] == k:
            continue
        idx = 4 + rng.choice(len(c["u"]) - 4, per_ctx, replace=False)
        Xs.append(c["X"][idx]); ys.append(c["u"][idx] - c["u"].mean())
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_leaf_nodes=63, l2_regularization=1.0,
                                      random_state=k).fit(np.concatenate(Xs), np.concatenate(ys))
    for c in ctx:
        if c["fold"] != k:
            continue
        u = c["u"]; un = np.arange(4, len(u)); pred = m.predict(c["X"][4:])
        o = un[np.argsort(-pred)]; u0 = u[:4].max()
        r_ = rows[c["fam"]]
        r_["U0"].append(u0); r_["U1"].append(max(u0, u[o[0]])); r_["U4"].append(max(u0, u[o[:4]].max()))
        r_["U16"].append(max(u0, u[o[:16]].max()))
        r_["R4"].append(max(u0, u[rng.choice(un, 4, replace=False)].max()))
        r_["R16"].append(max(u0, u[rng.choice(un, 16, replace=False)].max()))
        r_["TOP1_unverified"].append(u[o[0]]); r_["ORACLE"].append(u.max())
        r_["spearman"].append(np.corrcoef(np.argsort(np.argsort(pred)), np.argsort(np.argsort(u[4:])))[0, 1])
    print("fold", k, "done", flush=True)
fams = sorted(rows)
F = {key: np.array([np.mean(rows[f][key]) for f in fams]) for key in rows[fams[0]]}
print("families", len(fams))
for key in ("U0", "U1", "U4", "U16", "R4", "R16", "TOP1_unverified", "ORACLE", "spearman"):
    print("%-16s family-mean %.4f" % (key, F[key].mean()))
def lcb(d, B=10000):
    b = np.random.default_rng(0).choice(len(d), (B, len(d))); return np.percentile(d[b].mean(1), [2.5, 97.5])
for a, b in (("U1", "U0"), ("U4", "U0"), ("U4", "R4"), ("U16", "R16"), ("TOP1_unverified", "U0")):
    d = F[a] - F[b]; lo, hi = lcb(d)
    print("%s - %s: mean %+.4f  95%% CI [%+.4f, %+.4f]" % (a, b, d.mean(), lo, hi))
```

### learnability_probe_rep.py (sha256 e783046926ffec8749a82a33910e0099ce922c3dd94787419d7da3daf9ca03de)
```python
"""DL-Train ONLY, family-grouped 5-fold CV (assignment: sha256(family_id) mod 5; not seed-4900 — diagnostic only).
Question: is there held-out-family signal in program features (412-d) for WITHIN-CONTEXT ranking, enough to beat
the zero-parameter best-of-revealed (storage slots 0..3) floor under a budget-matched verified rule?
Model: HistGradientBoostingRegressor on target = u_ci - mean_c(u) (offset-free), sampled candidates per context.
Metrics on held-out families (context-mean, then family-mean):
  U0   = best revealed
  Uk   = max(best revealed, max true u over model top-k among unrevealed)  k=1,4  (verified, budget k)
  A1   = unverified argmin rule as in live utility(): model top-1 among unrevealed vs revealed by *predicted* value
  ORACLE, RANDOMk (k random unrevealed, verified)
Prints paired family-bootstrap LCB95 for Uk - U0 and Uk - RANDOMk."""
import hashlib, json, sys
from collections import defaultdict
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
root = sys.argv[1]; per_ctx = int(sys.argv[2])
sel = json.load(open(f"{root}/selected.json"))
rng = np.random.default_rng(11)
fold = lambda fam: int(hashlib.sha256(fam.encode()).hexdigest(), 16) % 5
ctx = []
for r in sel["train"]:
    p = np.load(f"{root}/pools/{r['context_sha256']}.npz", allow_pickle=False)
    u = np.where(p["gate"].astype(bool), -p["fitness"].astype(float), -2.0)
    ctx.append(dict(fam=r["family_id"], fold=fold(r["family_id"]), X=p["program"].astype(np.float32), u=u))
print("contexts", len(ctx), "fold sizes", np.bincount([c["fold"] for c in ctx]))
rows = defaultdict(lambda: defaultdict(list))
for k in range(5):
    Xs, ys = [], []
    for c in ctx:
        if c["fold"] == k:
            continue
        idx = 4 + rng.choice(len(c["u"]) - 4, per_ctx, replace=False)
        Xs.append(c["X"][idx]); ys.append(c["u"][idx] - c["u"].mean())
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_leaf_nodes=63, l2_regularization=1.0,
                                      random_state=100 + k).fit(np.concatenate(Xs), np.concatenate(ys))
    for c in ctx:
        if c["fold"] != k:
            continue
        u = c["u"]; un = np.arange(4, len(u)); pred = m.predict(c["X"][4:])
        o = un[np.argsort(-pred)]; u0 = u[:4].max()
        r_ = rows[c["fam"]]
        r_["U0"].append(u0); r_["U1"].append(max(u0, u[o[0]])); r_["U4"].append(max(u0, u[o[:4]].max()))
        r_["U16"].append(max(u0, u[o[:16]].max()))
        r_["R4"].append(max(u0, u[rng.choice(un, 4, replace=False)].max()))
        r_["R16"].append(max(u0, u[rng.choice(un, 16, replace=False)].max()))
        r_["TOP1_unverified"].append(u[o[0]]); r_["ORACLE"].append(u.max())
        r_["spearman"].append(np.corrcoef(np.argsort(np.argsort(pred)), np.argsort(np.argsort(u[4:])))[0, 1])
    print("fold", k, "done", flush=True)
fams = sorted(rows)
F = {key: np.array([np.mean(rows[f][key]) for f in fams]) for key in rows[fams[0]]}
print("families", len(fams))
for key in ("U0", "U1", "U4", "U16", "R4", "R16", "TOP1_unverified", "ORACLE", "spearman"):
    print("%-16s family-mean %.4f" % (key, F[key].mean()))
def lcb(d, B=10000):
    b = np.random.default_rng(0).choice(len(d), (B, len(d))); return np.percentile(d[b].mean(1), [2.5, 97.5])
for a, b in (("U1", "U0"), ("U4", "U0"), ("U4", "R4"), ("U16", "R16"), ("TOP1_unverified", "U0")):
    d = F[a] - F[b]; lo, hi = lcb(d)
    print("%s - %s: mean %+.4f  95%% CI [%+.4f, %+.4f]" % (a, b, d.mean(), lo, hi))
```

### generation_confound.py (sha256 d9445a292fc716ce2e2209b2f6a945202e6604fc946c24ad6b156619e4b30a15)
```python
"""DL-Train ONLY, family-grouped 5-fold CV (assignment: sha256(family_id) mod 5; not seed-4900 — diagnostic only).
Question: is there held-out-family signal in program features (412-d) for WITHIN-CONTEXT ranking, enough to beat
the zero-parameter best-of-revealed (storage slots 0..3) floor under a budget-matched verified rule?
Model: HistGradientBoostingRegressor on target = u_ci - mean_c(u) (offset-free), sampled candidates per context.
Metrics on held-out families (context-mean, then family-mean):
  U0   = best revealed
  Uk   = max(best revealed, max true u over model top-k among unrevealed)  k=1,4  (verified, budget k)
  A1   = unverified argmin rule as in live utility(): model top-1 among unrevealed vs revealed by *predicted* value
  ORACLE, RANDOMk (k random unrevealed, verified)
Prints paired family-bootstrap LCB95 for Uk - U0 and Uk - RANDOMk."""
import hashlib, json, sys
from collections import defaultdict
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
root = sys.argv[1]; per_ctx = int(sys.argv[2])
sel = json.load(open(f"{root}/selected.json"))
rng = np.random.default_rng(7)
fold = lambda fam: int(hashlib.sha256(fam.encode()).hexdigest(), 16) % 5
ctx = []
for r in sel["train"]:
    p = np.load(f"{root}/pools/{r['context_sha256']}.npz", allow_pickle=False)
    u = np.where(p["gate"].astype(bool), -p["fitness"].astype(float), -2.0)
    gen = np.log2(p["generation_mask"].astype(np.float64).clip(1)).astype(int) if "generation_mask" in p else None
    ctx.append(dict(fam=r["family_id"], fold=fold(r["family_id"]), X=p["program"].astype(np.float32), u=u, gen=gen, birth=p["birth_seq_min"].astype(float)))
print("contexts", len(ctx), "fold sizes", np.bincount([c["fold"] for c in ctx]))
rows = defaultdict(lambda: defaultdict(list))
for k in range(5):
    Xs, ys = [], []
    for c in ctx:
        if c["fold"] == k:
            continue
        idx = 4 + rng.choice(len(c["u"]) - 4, per_ctx, replace=False)
        Xs.append(c["X"][idx]); ys.append(c["u"][idx] - c["u"].mean())
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_leaf_nodes=63, l2_regularization=1.0,
                                      random_state=k).fit(np.concatenate(Xs), np.concatenate(ys))
    for c in ctx:
        if c["fold"] != k:
            continue
        u = c["u"]; un = np.arange(4, len(u)); pred = m.predict(c["X"][4:])
        o = un[np.argsort(-pred)]; u0 = u[:4].max()
        r_ = rows[c["fam"]]
        r_["U0"].append(u0); r_["U1"].append(max(u0, u[o[0]])); r_["U4"].append(max(u0, u[o[:4]].max()))
        r_["U16"].append(max(u0, u[o[:16]].max()))
        r_["R4"].append(max(u0, u[rng.choice(un, 4, replace=False)].max()))
        r_["R16"].append(max(u0, u[rng.choice(un, 16, replace=False)].max()))
        r_["TOP1_unverified"].append(u[o[0]]); r_["ORACLE"].append(u.max())
        g = c["gen"][4:]; gm = np.bincount(g).argmax()  # most common generation among unrevealed
        sub = un[g == gm]
        if len(sub) >= 8:
            ps = m.predict(c["X"][sub]); os_ = sub[np.argsort(-ps)]
            r_["W4"].append(max(u0, u[os_[:4]].max())); r_["WR4"].append(max(u0, u[rng.choice(sub, 4, replace=False)].max()))
            r_["Wrho"].append(np.corrcoef(np.argsort(np.argsort(ps)), np.argsort(np.argsort(u[sub])))[0, 1])
        ob = un[np.argsort(-c["birth"][4:])]
        r_["B4"].append(max(u0, u[ob[:4]].max()))
        r_["Brho"].append(np.corrcoef(np.argsort(np.argsort(c["birth"][4:])), np.argsort(np.argsort(u[4:])))[0, 1])
        r_["spearman"].append(np.corrcoef(np.argsort(np.argsort(pred)), np.argsort(np.argsort(u[4:])))[0, 1])
    print("fold", k, "done", flush=True)
fams = sorted(rows)
keys = set().union(*[rows[f].keys() for f in fams])
fams = [f for f in fams if all(len(rows[f][k]) for k in keys)]
F = {key: np.array([np.mean(rows[f][key]) for f in fams]) for key in keys}
print("families", len(fams))
for key in ("U0", "U4", "R4", "B4", "Brho", "W4", "WR4", "Wrho", "spearman"):
    print("%-16s family-mean %.4f" % (key, F[key].mean()))
def lcb(d, B=10000):
    b = np.random.default_rng(0).choice(len(d), (B, len(d))); return np.percentile(d[b].mean(1), [2.5, 97.5])
for a, b in (("U4", "R4"), ("B4", "R4"), ("U4", "B4"), ("W4", "WR4")):
    d = F[a] - F[b]; lo, hi = lcb(d)
    print("%s - %s: mean %+.4f  95%% CI [%+.4f, %+.4f]" % (a, b, d.mean(), lo, hi))
```

### h58_prior_check.py (sha256 f67749012c97143803db7d521ff849f36261110d75d5231eae2707e4c6dbe53e)
```python
"""DL-Train only. Quality of H58's cached base_prior_cost (pool_priors.npz) as a ranker, same metrics as
learnability_probe (family means): Spearman on unrevealed, U4 vs R4. Tests whether H58 B/C/D ~= INIT_G0 because
argmin is dominated by the prior."""
import json, sys
from collections import defaultdict
import numpy as np
root, cache = sys.argv[1], sys.argv[2]
sel = json.load(open(f"{root}/selected.json"))
pp = np.load(f"{cache}/pool_priors.npz", allow_pickle=False)
cids = [c.decode() if isinstance(c, bytes) else str(c) for c in pp["cids"]]
pos = {c: i for i, c in enumerate(cids)}; off = pp["offsets"]; prior = pp["base_prior_cost"]
rng = np.random.default_rng(3); rows = defaultdict(lambda: defaultdict(list)); mism = 0
rank = lambda x: np.argsort(np.argsort(x))
for r in sel["train"]:
    c = r["context_sha256"]; p = np.load(f"{root}/pools/{c}.npz", allow_pickle=False)
    u = np.where(p["gate"].astype(bool), -p["fitness"].astype(float), -2.0)
    i = pos[c]; pr = prior[off[i]:off[i + 1]]
    if len(pr) != len(u):
        mism += 1; n = min(len(pr), len(u)); pr, u = pr[:n], u[:n]
    un = np.arange(4, len(u)); o = un[np.argsort(pr[4:])]; u0 = u[:4].max(); f = rows[r["family_id"]]
    f["rho"].append(np.corrcoef(rank(-pr[4:]), rank(u[4:]))[0, 1])
    f["U4"].append(max(u0, u[o[:4]].max())); f["R4"].append(max(u0, u[rng.choice(un, 4, replace=False)].max()))
    f["U1"].append(max(u0, u[o[0]])); f["U0"].append(u0)
print("length mismatches", mism)
F = {k: np.array([np.mean(rows[x][k]) for x in rows]) for k in ("rho", "U0", "U1", "U4", "R4")}
for k, v in F.items():
    print("%-4s family-mean %.4f" % (k, np.nanmean(v)))
d = F["U4"] - F["R4"]; b = np.random.default_rng(0).choice(len(d), (10000, len(d)))
print("U4-R4 %+.4f CI [%+.4f, %+.4f]" % (d.mean(), *np.percentile(d[b].mean(1), [2.5, 97.5])))
```

### crossview_noise.py (sha256 66048e184c276f4115275b7f76fda191d6ae7fff258e710c030385a8bfbe90cf)
```python
"""DL-Train only. For programs (same program_sha256) present in >=2 sibling views of a dataset (same family),
measure |u_viewA - u_viewB| overall and for programs in the top decile of their view. Compares against GP
promotion margins (median 0.0043 in results/fused-20261029/promotions.jsonl)."""
import json, sys, itertools
from collections import defaultdict
import numpy as np
root = sys.argv[1]
sel = json.load(open(f"{root}/selected.json"))
fam = defaultdict(list)
for r in sel["train"]:
    fam[r["family_id"]].append(r["context_sha256"])
d_all, d_top, d_top_both, pairs = [], [], [], 0
for f, cs in fam.items():
    if len(cs) < 2:
        continue
    P = []
    for c in cs:
        p = np.load(f"{root}/pools/{c}.npz", allow_pickle=False)
        u = np.where(p["gate"].astype(bool), -p["fitness"].astype(float), -2.0)
        q = np.quantile(u, 0.9)
        P.append({s: (x, x >= q) for s, x in zip(p["program_sha256"].tolist(), u)})
    for a, b in itertools.combinations(P, 2):
        for s in set(a) & set(b):
            (ua, ta), (ub, tb) = a[s], b[s]
            if ua == -2.0 or ub == -2.0:
                continue
            pairs += 1; d = abs(ua - ub); d_all.append(d)
            if ta or tb: d_top.append(d)
            if ta and tb: d_top_both.append(d)
q = lambda x: "n=%d median %.4f p25 %.4f p75 %.4f frac>0.0043 %.3f" % (len(x), np.median(x), np.percentile(x, 25), np.percentile(x, 75), np.mean(np.array(x) > 0.0043)) if x else "n=0"
print("shared passing program pairs", pairs)
print("all      |du|:", q(d_all))
print("top10%% either:", q(d_top))
print("top10%% both  :", q(d_top_both))
```

### crossview_pairs.py (sha256 68978845f6f6c3eb459d494ac56ec3d6505ed969d84b75dd521d55b5b9f90722)
```python
"""DL-Train only. Offset-free replication of pairwise wins across sibling views: for programs i,j shared by
views A and B (same dataset family), x = u_iA - u_jA, y = u_iB - u_jB. Report sign agreement P(sign y = sign x)
by |x| bin and SD of (x - y)/sqrt2 (noise of one view's pairwise difference). Restrict to pairs where both
programs are in their view's top 25% (the promotion regime) and also all pairs."""
import json, sys, itertools
from collections import defaultdict
import numpy as np
root = sys.argv[1]
sel = json.load(open(f"{root}/selected.json"))
fam = defaultdict(list)
for r in sel["train"]:
    fam[r["family_id"]].append(r["context_sha256"])
X, Y, TOP = [], [], []
for f, cs in fam.items():
    if len(cs) < 2:
        continue
    P = []
    for c in cs:
        p = np.load(f"{root}/pools/{c}.npz", allow_pickle=False)
        g = p["gate"].astype(bool); u = np.where(g, -p["fitness"].astype(float), -2.0)
        q = np.quantile(u, 0.75)
        P.append({s: (x, x >= q) for s, x, gg in zip(p["program_sha256"].tolist(), u, g) if gg})
    for a, b in itertools.combinations(P, 2):
        sh = sorted(set(a) & set(b))
        for s, t in itertools.combinations(sh, 2):
            X.append(a[s][0] - a[t][0]); Y.append(b[s][0] - b[t][0])
            TOP.append(a[s][1] and a[t][1] and b[s][1] and b[t][1])
X, Y, TOP = map(np.array, (X, Y, TOP))
for name, m in (("all pairs", np.ones_like(TOP, bool)), ("both in top25% of both views", TOP)):
    x, y = X[m], Y[m]
    print("%s: n=%d  SD((x-y)/sqrt2)=%.4f  corr(x,y)=%.3f" % (name, len(x), np.std((x - y) / np.sqrt(2)), np.corrcoef(x, y)[0, 1]))
    for lo, hi in ((0, 0.005), (0.005, 0.02), (0.02, 0.05), (0.05, 0.2), (0.2, 9)):
        k = (np.abs(x) >= lo) & (np.abs(x) < hi) & (x != 0)
        if k.sum():
            print("   |x| in [%.3f,%.3f): n=%6d  sign agree %.3f  mean y/x %.3f" % (lo, hi, k.sum(), np.mean(np.sign(y[k]) == np.sign(x[k])), np.mean(y[k] / x[k])))
```

### context_probe.py (sha256 4ce9517a883ba2423d4787575c27f74b6f8e0431b651311df7867f4fdad4d8d0)
```python
"""DL-Train ONLY, same protocol as learnability_probe.py (family folds = sha256(family) mod 5, verified U4 vs
random R4). Question: does adding frozen per-context features (H58 cache: lm48_z 48-d + dope629) to the 412
syntactic program features improve held-out within-context selection? mode = prog | ctx.
Per-context features are constant within a context, so they can only help through interactions."""
import hashlib, json, sys
from collections import defaultdict
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

root, per_ctx, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sel = json.load(open(f"{root}/selected.json"))
CF = np.load("/home/ubuntu/jope-data/h58/cache/context_features.npz", allow_pickle=False)
cpos = {(c.decode() if isinstance(c, bytes) else str(c)): i for i, c in enumerate(CF["cids"])}
ctxf = np.concatenate([CF["lm48_z"], CF["dope629"]], 1).astype(np.float32)
rng = np.random.default_rng(11)
fold = lambda fam: int(hashlib.sha256(fam.encode()).hexdigest(), 16) % 5
ctx = []
for r in sel["train"]:
    p = np.load(f"{root}/pools/{r['context_sha256']}.npz", allow_pickle=False)
    u = np.where(p["gate"].astype(bool), -p["fitness"].astype(float), -2.0)
    ctx.append(dict(fam=r["family_id"], fold=fold(r["family_id"]), X=p["program"].astype(np.float32), u=u,
                    c=ctxf[cpos[r["context_sha256"]]]))
print("mode", mode, "contexts", len(ctx), "ctx dims", ctxf.shape[1])


def feats(c, idx):
    if mode == "prog":
        return c["X"][idx]
    return np.concatenate([c["X"][idx], np.repeat(c["c"][None], len(idx), 0)], 1)


rows = defaultdict(lambda: defaultdict(list))
for k in range(5):
    Xs, ys = [], []
    for c in ctx:
        if c["fold"] == k:
            continue
        idx = 4 + rng.choice(len(c["u"]) - 4, per_ctx, replace=False)
        Xs.append(feats(c, idx)); ys.append(c["u"][idx] - c["u"].mean())
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_leaf_nodes=63, l2_regularization=1.0,
                                      random_state=100 + k).fit(np.concatenate(Xs), np.concatenate(ys))
    for c in ctx:
        if c["fold"] != k:
            continue
        u = c["u"]; un = np.arange(4, len(u)); pred = m.predict(feats(c, un))
        o = un[np.argsort(-pred)]; u0 = u[:4].max(); r_ = rows[c["fam"]]
        r_["U4"].append(max(u0, u[o[:4]].max())); r_["U1"].append(max(u0, u[o[0]]))
        r_["R4"].append(max(u0, u[rng.choice(un, 4, replace=False)].max()))
        r_["rho"].append(np.corrcoef(np.argsort(np.argsort(pred)), np.argsort(np.argsort(u[4:])))[0, 1])
    print("fold", k, "done", flush=True)
fams = sorted(rows)
F = {key: np.array([np.mean(rows[f][key]) for f in fams]) for key in rows[fams[0]]}
np.savez(f"/tmp/claude-1000/-home-ubuntu-JopeDime-rd/4bbd7c7c-e74a-46f7-811c-1c554809116c/scratchpad/ctxprobe_{mode}.npz",
         fams=np.array(fams), **F)
for key in ("U1", "U4", "R4", "rho"):
    print("%-4s family-mean %.4f" % (key, F[key].mean()))
d = F["U4"] - F["R4"]; b = np.random.default_rng(0).choice(len(d), (10000, len(d)))
print("U4 - R4: %+.4f [%+.4f, %+.4f]" % (d.mean(), *np.percentile(d[b].mean(1), [2.5, 97.5])))
```

### context_probe_rep.py (sha256 20a5084d07d5c789557917bf0a8860b36ac5ff86f557a58b4d999367d59ccd7b)
```python
"""DL-Train ONLY, same protocol as learnability_probe.py (family folds = sha256(family) mod 5, verified U4 vs
random R4). Question: does adding frozen per-context features (H58 cache: lm48_z 48-d + dope629) to the 412
syntactic program features improve held-out within-context selection? mode = prog | ctx.
Per-context features are constant within a context, so they can only help through interactions."""
import hashlib, json, sys
from collections import defaultdict
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

root, per_ctx, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sel = json.load(open(f"{root}/selected.json"))
CF = np.load("/home/ubuntu/jope-data/h58/cache/context_features.npz", allow_pickle=False)
cpos = {(c.decode() if isinstance(c, bytes) else str(c)): i for i, c in enumerate(CF["cids"])}
ctxf = np.concatenate([CF["lm48_z"], CF["dope629"]], 1).astype(np.float32)
rng = np.random.default_rng(23)
fold = lambda fam: int(hashlib.sha256(fam.encode()).hexdigest(), 16) % 5
ctx = []
for r in sel["train"]:
    p = np.load(f"{root}/pools/{r['context_sha256']}.npz", allow_pickle=False)
    u = np.where(p["gate"].astype(bool), -p["fitness"].astype(float), -2.0)
    ctx.append(dict(fam=r["family_id"], fold=fold(r["family_id"]), X=p["program"].astype(np.float32), u=u,
                    c=ctxf[cpos[r["context_sha256"]]]))
print("mode", mode, "contexts", len(ctx), "ctx dims", ctxf.shape[1])


def feats(c, idx):
    if mode == "prog":
        return c["X"][idx]
    return np.concatenate([c["X"][idx], np.repeat(c["c"][None], len(idx), 0)], 1)


rows = defaultdict(lambda: defaultdict(list))
for k in range(5):
    Xs, ys = [], []
    for c in ctx:
        if c["fold"] == k:
            continue
        idx = 4 + rng.choice(len(c["u"]) - 4, per_ctx, replace=False)
        Xs.append(feats(c, idx)); ys.append(c["u"][idx] - c["u"].mean())
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_leaf_nodes=63, l2_regularization=1.0,
                                      random_state=200 + k).fit(np.concatenate(Xs), np.concatenate(ys))
    for c in ctx:
        if c["fold"] != k:
            continue
        u = c["u"]; un = np.arange(4, len(u)); pred = m.predict(feats(c, un))
        o = un[np.argsort(-pred)]; u0 = u[:4].max(); r_ = rows[c["fam"]]
        r_["U4"].append(max(u0, u[o[:4]].max())); r_["U1"].append(max(u0, u[o[0]]))
        r_["R4"].append(max(u0, u[rng.choice(un, 4, replace=False)].max()))
        r_["rho"].append(np.corrcoef(np.argsort(np.argsort(pred)), np.argsort(np.argsort(u[4:])))[0, 1])
    print("fold", k, "done", flush=True)
fams = sorted(rows)
F = {key: np.array([np.mean(rows[f][key]) for f in fams]) for key in rows[fams[0]]}
np.savez(f"/tmp/claude-1000/-home-ubuntu-JopeDime-rd/4bbd7c7c-e74a-46f7-811c-1c554809116c/scratchpad/ctxprobe_{mode}_rep.npz",
         fams=np.array(fams), **F)
for key in ("U1", "U4", "R4", "rho"):
    print("%-4s family-mean %.4f" % (key, F[key].mean()))
d = F["U4"] - F["R4"]; b = np.random.default_rng(0).choice(len(d), (10000, len(d)))
print("U4 - R4: %+.4f [%+.4f, %+.4f]" % (d.mean(), *np.percentile(d[b].mean(1), [2.5, 97.5])))
```

### insert.sh (sha256 f0135e38a6cc30c264242ad829284680828add9df8231c8d9e03b7a7f0488c94)
```bash
#!/usr/bin/env bash
# usage: insert.sh before|after "<exact anchor line (first match)>" <block-file>
# Inserts block before/after the first exact anchor line, under flock. Never edits others' lines.
set -euo pipefail
cd "$(dirname "$0")/.."
tmp=/tmp/claude-1000/-home-ubuntu-JopeDime-rd/4bbd7c7c-e74a-46f7-811c-1c554809116c/scratchpad/FINDINGS.opus.tmp
flock FINDINGS.md.lock bash -c '
  grep -qxF -- "$1" FINDINGS.md || { echo "anchor not found: $1" >&2; exit 3; }
  gawk -v mode="$0" -v hdr="$1" -v blk="$2" "BEGIN{while((getline l < blk)>0) b=b l \"\n\"}
    \$0==hdr && !done && mode==\"before\" {printf \"%s\", b; done=1; print; next}
    \$0==hdr && !done && mode==\"after\" {print; printf \"%s\", b; done=1; next}
    {print}" FINDINGS.md > "$3"
  n0=$(wc -l < FINDINGS.md); n1=$(wc -l < "$3"); nb=$(wc -l < "$2")
  [ $((n0+nb)) -eq "$n1" ] || { echo "line count mismatch $n0+$nb!=$n1" >&2; exit 4; }
  cat "$3" > FINDINGS.md
' "$1" "$2" "$3" "$tmp"
```

## [ASTRA] Capsule review limits

The E1–E16 displays include condensed transcripts and ellipses; shell snippets such as the E2 `gawk` and E9 `jq` summaries are not complete replay commands. Individual start times, exact elapsed times and raw process receipts were not retained here for every probe; do not infer them from the section-level exit0 labels. ASTRA reviewed the embedded sources and their hashes but did not rerun model fits. A new admitted study must preserve full command/environment/input/version/timing receipts.

Double-argsort correlations arbitrarily order ties; call them rank-correlation diagnostics. The E13 +123.6 figure is a pairwise score transform, not the fixed multi-policy BT rating required by PROPOSALS. Conditional family-bootstrap intervals and two replicates do not supply independent confirmation after adaptive recipe development. Within-generation evidence weakens a pure generation-detector explanation but does not prove transfer to fresh proposals. E4 only checks the sampled412D input buckets; it does not rule out losses in the neural projection or all representation bottlenecks. The printed rounded bucket count is1150 at1decimal, so the earlier 'all1152 distinct even at1decimal' shorthand is too strong.

The prior-check source truncates to the shorter length on mismatch and its summary omits that printed mismatch count. Future admission must reject identity/length mismatches rather than silently truncate; cached-prior and final-model rankings are distinct objects. The context and prior probes materialize mixed cache arrays, although their fitting/scoring loops select DL-Train contexts. The inherited feature/prior provenance and selected-source hashes remain audit requirements.

GT-O14 conditions on shared, passing programs, and its top-quarter subset is selected using both views. It does not identify a selected promotion's replication probability or guarantee a rate below73%. No sibling data transfer is authorized by that analysis. See [ASTRA source review](../astra/CODE_TRUTH.md) and [PROPOSALS](../PROPOSALS.md) for the final interpretation.
