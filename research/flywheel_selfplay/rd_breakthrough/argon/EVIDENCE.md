# `[ARGON]` Durable Evidence & Probe Transcripts (`rd_breakthrough`)

- **Agent**: `rd_argon` (`[ARGON]`)
- **Date (UTC)**: `2026-10-31 02:01–02:23 UTC`
- **Investigated Git Commit**: `ca4c64b36c38b9c3e3f450b09c93896c50d98ee0` (branch `rd/breakthrough-20261030`)
- **Host**: `xbabe2` (CPU-only read-only probes run under `nice -n 15`, $\le 4$ cores; zero GPU usage; zero reads of `DL-Val` pool outcomes, `DL-Holdout`, `sealed_meta_test`, or live `xbabe1` state).

---

## 1. Input Artifact & Manifest SHA-256 Pins

| Artifact / File | Path / S3 URI | Size (B) | SHA-256 |
|---|---|---:|---|
| `selected.json` | `/home/ubuntu/jope-data/res-01/input/selected.json` | 380,683 | `981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1` |
| `source_manifest.json` | `/home/ubuntu/jope-data/res-01/input/source_manifest.json` | 436 | `99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4` |
| `corpus_v2_manifest.json` | `s3://veox-jopedime-learning-use2/experiments/flywheel-selfplay-20261023/corpus_v2_train_winner_integrated/corpus_v2_manifest.json` (and `/mnt/jope-archive/flywheel-20261023/corpus_v2_train_winner_integrated/corpus_v2_manifest.json`) | 244,173 | `05af9e5930092ab41587b96ce9029f10bf84557ff45b9ceac94348e8e8e510c8` |
| `corpus_v3_manifest.json` | `s3://veox-jopedime-learning-use2/experiments/flywheel-selfplay-20261023/corpus_v3_wider_strict_train_256/corpus_v3_manifest.json` (and `/mnt/jope-archive/flywheel-20261023/corpus_v3_wider_strict_train_256/corpus_v3_manifest.json`) | 718,858 | `701affc61e14950bc5c890b6399a9d765f10e7f7f720a2ba8f1a16d93e1a7640` |

---

## 2. `GT-AR1`: `storage[0..3]` Is a `program_sha256`-Sorted Draw Across All 18 GP Generations (`18 × 64 = 1,152`), Not Chronological `Gen 0` Seeds

### Source Receipt
- `research/flywheel_selfplay/dataset_builder.py:216`:
  ```python
  items = sorted(by_dsl.values(), key=lambda x: hashlib.sha256(x["dsl"].encode()).hexdigest())
  ```

### Probe 2A — `program_sha256` Sorting Verification (`probe_gate_identity.py`, SHA-256 `0b72cfa30c9edacae4b226d420fc33926ae32d9928090eed7dc1a0f2c849bc88`)
- **Command**: `nice -n 15 python3 probe_gate_identity.py`
- **Exit status**: `0` (elapsed `2.8s`)
- **Source**:
  ```python
  import json
  from pathlib import Path
  import numpy as np

  res01_dir = Path("/home/ubuntu/jope-data/res-01/input")
  selected = json.loads((res01_dir / "selected.json").read_text())
  train_entries = selected["train"]

  p0 = np.load(res01_dir / "pools" / f"{train_entries[0]['context_sha256']}.npz")
  print("program_sha256 sample:", p0["program_sha256"][:5], p0["program_sha256"].dtype)
  print("birth_seq_min sample:", p0["birth_seq_min"][:10])
  print("fitness sample:", p0["fitness"][:10])
  print("logratio sample:", p0["logratio"][:10])
  print("gate sample:", p0["gate"][:10])

  bmin = p0["birth_seq_min"]
  shas = [str(x) for x in p0["program_sha256"]]
  print("is sha sorted:", shas == sorted(shas))
  print("is bmin sorted:", list(bmin) == sorted(list(bmin)))

  sha_sorted_cnt = 0
  for x in train_entries:
      p = np.load(res01_dir / "pools" / f"{x['context_sha256']}.npz")
      sh = [str(s) for s in p["program_sha256"]]
      if sh == sorted(sh):
          sha_sorted_cnt += 1
  print(f"Total DL-Train pools sha-sorted: {sha_sorted_cnt} / {len(train_entries)}")
  ```
- **Stdout**:
  ```text
  program_sha256 sample: [b'00091c1b1524beef83606e471e15fbfa27f4669e59dde83ed5d379da15fc0e78'
   b'00227e350cffe3593d1b8484964df81742294ae78b664453eb87d97708ca9f46'
   b'005bbc9eef0427b554ca32f23b3d380b2eb1fee9ac8e7b91ca3179b2c44fc3e8'
   b'00a882ca049dba43fe9342cc8be8bede7b5968b40646c88e93950ddbafca9ff7'
   b'00bf487263b8f04d97f753b17b476368daddb64b750104ce3399bde6190daae3'] |S64
  birth_seq_min sample: [1113  440 1166  711  422  785 1208  640  836  987]
  fitness sample: [-0.45353731 -0.75081999 -0.68313807 -0.77905811 -0.76409707 -0.73348364
   -0.772082   -0.7789155  -0.73632164 -0.18700085]
  logratio sample: [-0.5081547  -0.84344986 -0.76901287 -0.87411297 -0.85735255 -0.82687945
   -0.86947468 -0.87348612 -0.82725249 -0.21133763]
  gate sample: [ True  True  True  True  True  True  True  True  True  True]
  is sha sorted: True
  is bmin sorted: False
  Total DL-Train pools sha-sorted: 1219 / 1219
  ```

### Probe 2B — Chronological `birth_seq_min` vs `storage[0:4]` Utility Across All 1,219 `DL-Train` Pools (`probe_birth_seq.py`, SHA-256 `5c32a36ed749733dc337c84a2a082f7d0de9858ee91948cd020566c638467a01`)
- **Command**: `nice -n 15 python3 probe_birth_seq.py`
- **Exit status**: `0` (elapsed `11.4s`)
- **Source**:
  ```python
  import json
  from pathlib import Path
  import numpy as np
  from scipy.stats import spearmanr

  res01_dir = Path("/home/ubuntu/jope-data/res-01/input")
  selected = json.loads((res01_dir / "selected.json").read_text())
  train_entries = selected["train"]

  corrs_bmin_all, corrs_bmin_pass = [], []
  u_sha4, u_earliest4, u_latest4, u_oracle = [], [], [], []

  for x in train_entries:
      p = np.load(res01_dir / "pools" / f"{x['context_sha256']}.npz")
      f = p["fitness"]
      g = p["gate"] == 1
      bmin = p["birth_seq_min"]
      u = np.where(g, -f, -2.0)
      if np.std(bmin) > 0 and np.std(u) > 0:
          corrs_bmin_all.append(spearmanr(bmin, u).statistic)
      if np.sum(g) > 10 and np.std(bmin[g]) > 0 and np.std(u[g]) > 0:
          corrs_bmin_pass.append(spearmanr(bmin[g], u[g]).statistic)
      u_sha4.append(np.max(u[:4]))
      ear4 = np.argsort(bmin)[:4]
      lat4 = np.argsort(bmin)[-4:]
      u_earliest4.append(np.max(u[ear4]))
      u_latest4.append(np.max(u[lat4]))
      u_oracle.append(np.max(u))

  print(f"Across all {len(train_entries)} DL-Train pools:")
  print(f"  Spearman(birth_seq_min, u_all): mean={np.mean(corrs_bmin_all):.4f}, median={np.median(corrs_bmin_all):.4f}")
  print(f"  Spearman(birth_seq_min, u_pass): mean={np.mean(corrs_bmin_pass):.4f}, median={np.median(corrs_bmin_pass):.4f}")
  print(f"  Mean utility of best-of-4 SHA-sorted (storage[0:4]): {np.mean(u_sha4):.6f}")
  print(f"  Mean utility of best-of-4 earliest-born (argsort(birth_seq_min)[:4]): {np.mean(u_earliest4):.6f}")
  print(f"  Mean utility of best-of-4 latest-born (argsort(birth_seq_min)[-4:]): {np.mean(u_latest4):.6f}")
  print(f"  Mean utility of pool oracle (max u): {np.mean(u_oracle):.6f}")
  ```
- **Stdout**:
  ```text
  Across all 1219 DL-Train pools:
    Spearman(birth_seq_min, u_all): mean=0.1158, median=0.1115
    Spearman(birth_seq_min, u_pass): mean=0.1302, median=0.1289
    Mean utility of best-of-4 SHA-sorted (storage[0:4]): 1.483107
    Mean utility of best-of-4 earliest-born (argsort(birth_seq_min)[:4]): 1.416365
    Mean utility of best-of-4 latest-born (argsort(birth_seq_min)[-4:]): 0.922787
    Mean utility of pool oracle (max u): 1.905853
  ```

### Probe 2C — Exact 18-Generation Structure (`generation_mask = 1 << g`, $18 \times 64 = 1,152$) Across All 1,219 `DL-Train` Pools
- **Command**:
  ```bash
  nice -n 15 python3 -c '
  import json
  from pathlib import Path
  import numpy as np
  from scipy.stats import spearmanr

  res01_dir = Path("/home/ubuntu/jope-data/res-01/input")
  selected = json.loads((res01_dir / "selected.json").read_text())
  train_entries = selected["train"]

  gen_u = {g: [] for g in range(25)}
  corrs_gen = []
  gen0_counts = []
  for x in train_entries:
      cid = x["context_sha256"]
      p = np.load(res01_dir / "pools" / f"{cid}.npz")
      gm = p["generation_mask"]
      g_idx = np.array([int(v).bit_length() - 1 for v in gm])
      u = np.where(p["gate"] == 1, -p["fitness"], -2.0)
      gen0_counts.append(int(np.sum(g_idx == 0)))
      if np.std(g_idx) > 0 and np.std(u) > 0:
          corrs_gen.append(spearmanr(g_idx, u).statistic)
      for g in np.unique(g_idx):
          if 0 <= g < 25:
              gen_u[int(g)].append(float(np.mean(u[g_idx == g])))

  print(f"Spearman(generation_idx, u) across 1219 pools: mean={np.mean(corrs_gen):.4f}, median={np.median(corrs_gen):.4f}")
  print(f"Gen 0 candidate count per pool: min={min(gen0_counts)}, median={np.median(gen0_counts)}, max={max(gen0_counts)}")
  for g in sorted(gen_u):
      if len(gen_u[g]) > 100:
          print(f"  Gen {g:2d} (n_pools={len(gen_u[g]):4d}): mean_u={np.mean(gen_u[g]):.4f}, median_u={np.median(gen_u[g]):.4f}")
  '
  ```
- **Exit status**: `0` (elapsed `4.1s`)
- **Stdout**:
  ```text
  Spearman(generation_idx, u) across 1219 pools: mean=0.1405, median=0.1356
  Gen 0 candidate count per pool: min=64, median=64.0, max=64
    Gen  0 (n_pools=1219): mean_u=0.6451, median_u=0.4909
    Gen  1 (n_pools=1219): mean_u=0.6784, median_u=0.5128
    Gen  2 (n_pools=1219): mean_u=0.6986, median_u=0.5246
    Gen  3 (n_pools=1219): mean_u=0.7195, median_u=0.5361
    Gen  4 (n_pools=1219): mean_u=0.7720, median_u=0.5620
    Gen  5 (n_pools=1219): mean_u=0.7827, median_u=0.5573
    Gen  6 (n_pools=1219): mean_u=0.7982, median_u=0.5572
    Gen  7 (n_pools=1219): mean_u=0.8341, median_u=0.5887
    Gen  8 (n_pools=1219): mean_u=0.8437, median_u=0.5765
    Gen  9 (n_pools=1219): mean_u=0.8578, median_u=0.5927
    Gen 10 (n_pools=1219): mean_u=0.8845, median_u=0.6009
    Gen 11 (n_pools=1219): mean_u=0.8880, median_u=0.5968
    Gen 12 (n_pools=1219): mean_u=0.9001, median_u=0.5932
    Gen 13 (n_pools=1219): mean_u=0.9048, median_u=0.6008
    Gen 14 (n_pools=1219): mean_u=0.9137, median_u=0.6125
    Gen 15 (n_pools=1219): mean_u=0.9226, median_u=0.6059
    Gen 16 (n_pools=1219): mean_u=0.9277, median_u=0.6009
    Gen 17 (n_pools=1219): mean_u=0.9286, median_u=0.6077
  ```
- **Interpretation Note**: Aggregate generation means (`mean_u`) rise across `Gen 0..17` (`0.6451 -> 0.9286`), whereas across-pool medians (`median_u`) are not strictly monotone at every step (e.g., `0.6009` at Gen 10 vs `0.5968` at Gen 11) and individual contexts do not improve at every generation.

---

## 3. `GT-AR2` & `GT-AR2b`: Exact S1 Outcome Decomposition (`fitness`, `logratio`, `gate`) and `gate == 0` Anatomy

### Probe 3A — Passing Identity $F - 0.9\,\text{logratio} = 0.2s + 0.0005c \ge 0$ (`probe_outcomes_and_dl03.py`, SHA-256 `6aeee8ac02d739df3bad19ee61c3fa6c6e8d581a20089b5d609250eb7246eee1`)
- **Command**: `nice -n 15 python3 probe_outcomes_and_dl03.py`
- **Exit status**: `0` (elapsed `1.9s`)
- **Note on `dl03_sha_sorted=0` in stdout**: `probe_outcomes_and_dl03.py` checked an optional `/home/ubuntu/jope-data/dl03_pools/pools` path that does not exist (`p_dl03_path.exists() == False`), so `dl03_*` counters remained `0` (unchecked) and the loop fell back to `res-01/input/pools`. Thus the `1219/1219` hash-sorting proof in Probe 2A applies to `/home/ubuntu/jope-data/res-01/input/pools/*.npz`.
- **Stdout**:
  ```text
  DL-Val metadata: 91 contexts, 19 datasets, 19 families
  DL-Val views per dataset distribution: Counter({5: 18, 1: 1})
  Checked 200 DL-Train pools: diff_orders(res01 vs dl03)=0, dl03_sha_sorted=0, dl03_seq_sorted=0
  dl03 non-constant pools (of 200): gen_mask=0, delivery_count=0
  Gate==0 candidates across 200 DL-Train pools: total=9663, finite_logratio=9663 (100.0%), logratio<0 (beat baseline on mean but failed fold/90% gate)=6860 (71.0%)
  Passing candidates F - 0.9*logratio (= 0.2*s + 0.0005*c): min=0.001502, p10=0.003378, median=0.007591, p90=0.027064, max=2.105169
  ```

### Probe 3B — Exploratory `gate == 0` Formula Check on 200 `DL-Train` Pools (`9,663` failing rows) — **INTERPRETATION WITHDRAWN FOR `W3`**
- **PROMINENT WITHDRAWAL & METHODOLOGICAL CORRECTION (Phase C / `W3`)**:
  1. **Formula mismatch & failure-subtype percentages (`63.7% / 3.3% / 32.2%`) WITHDRAWN**: The current vendored Rust evaluator (`JopeDimeMicro/python/rust_evaluator.py:960-999`, `custody/SURROGATE.md:32-39`) defines $d = F_{\text{fail}} - 100 = \max(m, 0) + \max(m+s, 0) + s \ge 0$. Under that piecewise-clipped formula, $F_{\text{fail}} - 100 - 2m = 2s$ holds only when $m \ge 0$, not for $m < 0$. Without a generating-source receipt for the historical static `.npz` pool evaluator, the formula mismatch between static pools and `rust_evaluator.py:994` remains unresolved. Therefore, both the `s0 = 0.5 * (f0 - 100.0 - 2.0 * m0)` derivation and the resulting `63.7% / 3.3% / 32.2%` failure-subtype breakdown printed below are **formally withdrawn**.
  2. **`neither` residual category (`81 / 9,663 = 0.84%`) WITHDRAWN as a sentinel count**: In the script below, `neither = (~fail_90) & (~fail_t)` tests $m_0 \le \log(0.90) \land m_0 + s_0 \le 0$, which is a residual logical branch of the invalid `s0` formula rather than an explicit `(f0 == 100.0) & (m0 == 0.0)` test (note that $m_0 = 0 > \log(0.90)$ actually falls into `fail_90`, not `neither`). The interpretation of `neither` as a `fitness == 100.0 & logratio == 0.0` sentinel count is **formally withdrawn**.
  3. **`W3` resolution (`PROPOSALS.md`)**: In `W3`, we **do not reconstruct failing-row $s$ or failure subtypes from $F_{\text{fail}} - 100$**. `W3` supervises directly measured `logratio` ($m$) on completed non-sentinel rows, passing residual $F - 0.9\,\text{logratio} = 0.2s + 0.0005c \ge 0$ on `gate == 1` rows only, and the binary `gate` label, while conservatively masking explicit `fitness == 100.0 & logratio == 0.0` sentinel-shaped rows and any unreconstructable components. Raw command and stdout are preserved below for audit completeness only.
- **Command**:
  ```bash
  nice -n 15 python3 -c '
  import json, numpy as np
  from pathlib import Path
  res01_dir = Path("/home/ubuntu/jope-data/res-01/input")
  selected = json.loads((res01_dir / "selected.json").read_text())
  diffs = []
  only_90 = only_spill = both = neither = 0
  f_smooth_list = []
  for x in selected["train"][:200]:
      cid = x["context_sha256"]
      p = np.load(res01_dir / "pools" / f"{cid}.npz")
      g0 = p["gate"] == 0
      f0 = p["fitness"][g0]
      m0 = p["logratio"][g0]
      d = f0 - 100.0 - 2.0 * m0
      diffs.extend(d.tolist())
      s0 = 0.5 * d
      t0 = m0 + s0
      f_smooth_list.extend((0.9 * m0 + 0.2 * s0).tolist())
      fail_90 = m0 > np.log(0.90)
      fail_t = t0 > 0.0
      only_90 += int(np.sum(fail_90 & (~fail_t)))
      only_spill += int(np.sum((~fail_90) & fail_t))
      both += int(np.sum(fail_90 & fail_t))
      neither += int(np.sum((~fail_90) & (~fail_t)))
  diffs = np.array(diffs)
  tot = len(diffs)
  f_smooth_arr = np.array(f_smooth_list)
  print(f"Across {tot} gate==0 rows: min(f - 100 - 2*m)={diffs.min():.8f}, p10={np.percentile(diffs,10):.6f}, median={np.median(diffs):.6f}, p90={np.percentile(diffs,90):.6f}, max={diffs.max():.6f}")
  print(f"  Failed ONLY 0.90 margin (both folds beat baseline, t<=0!): {only_90} ({100*only_90/tot:.1f}%)")
  print(f"  Failed ONLY fold spill (exp(m)<=0.90, but one fold t>0!): {only_spill} ({100*only_spill/tot:.1f}%)")
  print(f"  Failed BOTH (m > log(0.90) and t > 0): {both} ({100*both/tot:.1f}%)")
  print(f"  Sentinels (fitness==100.0 & logratio==0.0): {neither} ({100*neither/tot:.2f}%)")
  print(f"  Counterfactual F_smooth = 0.9*m + 0.2*s on gate==0 rows: p10={np.percentile(f_smooth_arr,10):.4f}, median={np.median(f_smooth_arr):.4f}, p90={np.percentile(f_smooth_arr,90):.4f}")
  '
  ```
- **Exit status**: `0` (elapsed `1.8s`)
- **Stdout**:
  ```text
  Across 9663 gate==0 rows: min(f - 100 - 2*m)=-0.00000000, p10=0.001571, median=0.016869, p90=0.294624, max=9.963064
    Failed ONLY 0.90 margin (both folds beat baseline, t<=0!): 6153 (63.7%)
    Failed ONLY fold spill (exp(m)<=0.90, but one fold t>0!): 315 (3.3%)
    Failed BOTH (m > log(0.90) and t > 0): 3114 (32.2%)
    Sentinels (fitness==100.0 & logratio==0.0): 81 (0.84%)
    Counterfactual F_smooth = 0.9*m + 0.2*s on gate==0 rows: p10=-0.0864, median=-0.0289, p90=0.0693
  ```

---

## 4. `GT-AR3` & `GT-AR4`: 5-View Dataset Structure and Cross-Family Recurring AST Backbone (`probe_splits_and_pools.py`, SHA-256 `8e1eed900a0b2e538318d854144a942025745c1274402cdcbd7de0718ed54903`)

- **Command**: `nice -n 15 python3 probe_splits_and_pools.py`
- **Exit status**: `0` (elapsed `17.2s`)
- **Stdout**:
  ```text
  Total DL-Train: 1219, DL-Val (dev): 91
  Out of 100 DL-Train pools: sha_sorted=100, seq_sorted=0, fitness_sorted=0
  Best rank in storage[0:4] median percentile: 0.1463, mean: 0.1977, best_in_0..3=0/100
  generation_mask bit_count top-5: [(1, 115200)]
  delivery_count top-5: [(1, 115200)]
    birth_seq_min     : mean=0.1328, median=0.1300
  Datasets in DL-Train with 5 views (r0__f0..f4): 229 / 257
  Sibling 5-view program_sha256 Jaccard overlap: mean=0.0033, median=0.0022
  Sibling 5-view shared-program fitness Spearman: mean=0.8381, median=0.9146 (n_pairs=159)
  Global DL-Train pool occurrences: 1404287, unique AST hashes: 1316986
  Programs appearing in >=2 contexts: 14905 (1.13%)
  Programs appearing in >=2 disjoint families: 14804 (1.12%), covering 102003/1404287 (7.26%) of all pool slots!
  Top 10 most frequent programs across DL-Train (count, num_families): [('613655a9c150', 767, 193), ('10cfc1aa6374', 664, 197), ('596318acd15b', 648, 185), ('7af2a0bd58dd', 499, 177), ('f3926044678c', 460, 176), ('fc3c1f8a0a40', 457, 181), ('f51736d56a64', 453, 185), ('f84ad8bb9255', 445, 179), ('8efca9399f61', 439, 177), ('5374c1d534b9', 430, 186)]
  ```
- **Leave-One-Family-Out (LOFO) Cross-Family Coverage (`probe_cross_family_eb.py`, SHA-256 `90db35d7c49c60b36778fdc1f8f9135fced0b3d4986e8339718e60691ffb1b99`)**:
  ```text
  Contexts with >=1 candidate seen in >=3 disjoint OTHER families: 1219 / 1219
  Zero-reveal LOFO Empirical-Bayes top-1 utility: 1.127014 (vs r882=1.422548, best-of-4-SHA=1.483107)
  ```

---

## 5. `GT-AR5`, `GT-AR6` & `S-AR1`: Roster-Audited Verification of `corpus_v3_wider_strict_train_256` against `selected.json`

- **Command**:
  ```bash
  nice -n 15 python3 -c '
  import json
  v3 = json.load(open("/mnt/jope-archive/flywheel-20261023/corpus_v3_wider_strict_train_256/corpus_v3_manifest.json"))
  sel = json.load(open("/home/ubuntu/jope-data/res-01/input/selected.json"))
  train_cids = {x["context_sha256"] for x in sel["train"]}
  val_cids = {x["context_sha256"] for x in sel["dev"]}
  val_fams = {x["family_id"] for x in sel["dev"]}
  mps = v3["revisions"]["v3b_strict_train_median"]["materialized_pools"]
  in_tr_cid = sum(1 for m in mps if m["context_sha256"] in train_cids)
  in_val_cid = sum(1 for m in mps if m["context_sha256"] in val_cids)
  in_val_fam = sum(1 for m in mps if m["family_id"] in val_fams)
  print(f"corpus_v3b 254 pools vs selected.json: in_train_cid={in_tr_cid}/254, in_val_cid={in_val_cid}/254, in_val_19_fams={in_val_fam}/254")
  '
  ```
- **Exit status**: `0` (elapsed `0.4s`)
- **Stdout**:
  ```text
  corpus_v3b 254 pools vs selected.json: in_train_cid=254/254, in_val_cid=0/254, in_val_19_fams=0/254
  ```

## [ASTRA] Cross-review correction ledger

The raw command/output transcripts above are preserved as historical observations. Their following interpretations are **not accepted as verified facts** in the consolidated proposal:

- The exact `dataset_builder.py:216` source is `items = sorted(by_dsl.values(), key=lambda x: hashlib.sha256(x["dsl"].encode()).hexdigest())`; the earlier `records.sort(...)` display is a paraphrase, not a verbatim quote.
- Probe3B computes `s=(F-100-2m)/2` and labels the residual `neither` category as sentinels without directly testing `F==100,m==0`. Its 63.7% failure-subtype and81-sentinel interpretations are withdrawn. Pinned `custody/SURROGATE.md:32-45` and vendored `rust_evaluator.py:994` give `F_fail-100=max(m,0)+max(m+s,0)+s`. A different historical generating formula has not been independently source-pinned here. W3 uses measured completed-row logratio, passing-only residuals and original-status masks; it does not reconstruct failing spread from the unsupported inverse.
- Roster membership for254 contexts does not establish complete table/evaluator/cut/byte/alias lineage, original status, or novelty relative to H58 replay. It is an inventory prerequisite, not numeric admission.
- The approved S3 v3 listing contained the manifest only. Local materialized pools and S3 object payload availability are distinct observations. The '12.6× H58' comparison mixes a per-cell row cap with different replay doses; H58 harvested116,076 historical hard negatives, per OPUS's closeout receipt.
- The earliest-born-history floor1.416365 cannot be subtracted from r882's different-history1.422548 to estimate neural lift. Rescore all policies with identical history and count its acquisition cost. Aggregate monotone generation means do not prove per-context monotonicity.
- Historical511/512 screening confirmation is selected precision, not recall. Parent-head count is5,665 including LayerNorm, not5,569; unused AdamW parameters with `grad is None` do not necessarily receive weight decay.

See [PROPOSALS](../PROPOSALS.md) and the tagged cross-critique in [FINDINGS](../FINDINGS.md) for the corrected decision surface. This ledger is ASTRA's source/transcript review, not a rerun of ARGON's probes.
