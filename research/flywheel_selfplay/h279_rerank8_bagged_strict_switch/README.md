# H279: Bagged Strict-Margin Re-Ranker Switching (`h279_rerank8_bagged_strict_switch`)

**Status: COMPLETE. Verdict: `ADVANCE` (`BAG_SWITCH` and `BAG_SWITCH_NESTED` both pass all `9/9` strict gates; research diagnostic, `0` neural optimizer updates, no serving promotion).**

The study was preregistered in [PREREG.md](PREREG.md) and published on `origin/v1/next` (`1e023912e42d37208f0f06ac1f3eb4f86ccc3250` at `21:59Z` and ancestry-check update `aecf7f5a133ca53195bc5d6e2bf6f1d68804348d` at `22:01Z`, `outcomes_read = 0`, same-agent review receipt `sha256 = 29f5be6b…`) before any outcome was read. Before primary compute, `smoke-001` reproduced the `H272` regression cell `F272140-S272101-K0` exactly (`2` invalid, `131` changed, `84` deep picks), `Native91` (`91/91` bit-exact, mean `1.1377941898164525`), and the 4 outcome-blind `H276` clusterings. The single primary compute attempt (`attempt-001`) ran on `xbabe2` CPUs `57–60` (`nice 19`) from `22:01:57Z` to `22:50:53Z` (`2,936.1 s`, peak RSS `0.91 GiB`, `3` threads, `0` pauses) with `0` neural optimizer updates. Checkpoint `round-882.pt` (`sha256 = 0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`) and `DOPE` composite (`sha256 = c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`) were identical before and after.

---

## 1. Executive Summary

- **`BAG_SWITCH` ($B = 8$ family-bootstrapped rankers, $q = 6$ fixed) cuts harmed families (`loss > 0.01` vs `r882`) by `44.0%` (`28.0` vs `50.0` median over seeds, `ratio = 0.560 <= 0.75`) while retaining `97.6%` (`+0.0585` vs `+0.0599`, `ratio = 0.976 >= 0.80`) of `RERANK8_VETO`'s leave-cluster-out lift.**
  - **PRIMARY out-of-cluster (`DL-Train`, `205` families, `1,219` contexts, `60` cells):** `+0.0585 [+0.0385, +0.0735]`, Holm-level (`m = 2`) CI `[+0.0368, +0.0763]`, Holm $p = 0.0002$, **`3/3` seeds positive** (`+0.0585, +0.0545, +0.0594`), and **all `12` seed $\times$ partition values positive** (`+0.0508` to `+0.0714`).
  - **Elo (`r882 = 1500`):** `BAG_SWITCH = 1553.0` (`RERANK8_VETO = 1556.8`, `BAG_SWITCH_NESTED = 1551.6`, `PLACEBO = 1495.6`).
  - **Win/Tie/Loss (`|d| > 1e-4`):** Wins increase from `143, 135, 140` (`RERANK8_VETO`) to **`150, 150, 151` (`BAG_SWITCH`)**, while total losing families fall from `60, 70, 62` to **`39, 43, 47`**.
- **`BAG_SWITCH_NESTED` ($q \in \{5, 6, 7\}$ selected per cell by 4-fold inner family-grouped CV) also passes all `9/9` gates:**
  - **PRIMARY out-of-cluster (`DL-Train`):** `+0.0568 [+0.0405, +0.0724]`, Holm-level CI `[+0.0382, +0.0747]`, Holm $p = 0.0002$, **`3/3` seeds positive** (`+0.0595, +0.0568, +0.0555`), retaining **`94.8%`** of `RERANK8_VETO`'s gain while reducing harmed families (`loss > 0.01`) from `50.0` to **`30.0` (`60.0%` of `REF`, `-40.0%`)**.
- **Cluster consistency:** Both `BAG_SWITCH` and `BAG_SWITCH_NESTED` achieve `5/5` non-negative clusters (`>= 2/3` seeds) in `KM276140`, `KM276142`, and `WARD` (including `WARD` cluster 1, which is positive in `3/3` seeds: `+0.0172, +0.0515, +0.0331`), and `4/5` in `KM276141`.
- **Invalid `-2.0` picks:** `r882 = 204`, `RERANK8_VETO = 178`, `BAG_SWITCH = 178` (`<= REF`), `BAG_SWITCH_NESTED = 177` (`< REF`) on `DL-Train`; `0` for all arms on `DL-Val91`.
- **Placebo (`BAG_SWITCH_SHUF_PLACEBO`):** `-0.0003 [-0.0036, +0.0001]` (`1/3` seeds, Elo `1495.6`), well within `<= +0.005`.
- **Secondary `DL-Val91` (supporting only):** `BAG_SWITCH = +0.0180 [+0.0101, +0.0285]` (`3/3` seeds, Elo `1554.1`, median `0` harmed families with `loss > 0.01`); `BAG_SWITCH_NESTED = +0.0210 [+0.0122, +0.0321]` (`3/3` seeds, Elo `1571.7`, `0` harmed families with `loss > 0.01` on all `3/3` seeds). On watch family `fam-769243ee…`, median loss shrinks by `77.2%` under `BAG_SWITCH` (`-0.00095` vs `-0.00415`) and by `95.2%` under `BAG_SWITCH_NESTED` (`-0.00020` vs `-0.00415`).

---

## 2. Hypothesis & Setup

- **Hypothesis:** In `H276`, `RERANK8_VETO` generalized across held-out family clusters but still harmed `61–63 / 205` families per seed (`40–53 / 205` with `loss > 0.01` under seeds `279101–279103`), switching away from `r882` on `51.8%` (`7,582 / 14,628`) of context evaluations. `H279` tests whether requiring consensus (`>= q` of $B = 8$ family-level bootstrap-bagged rankers exceeding `r882`'s pick by the cell's strict `H265` margin) filters out fragile switches on dissimilar families, cutting harmed families (`loss > 0.01`) to `<= 0.75 x REF` while keeping `>= 0.80 x REF` of the utility gain.
- **Frozen Parent Science:** [topk.py](topk.py) (`fcd2d100c30d97df…`), [h265_core.py](h265_core.py) (`5c17ddad3993b5ce…`), [veto.py](veto.py) (`bf9c06ddd97f793b…`), and [fold_manifest.json](fold_manifest.json) (`e89f9aa130cf1529…`) are byte-identical to `H276`.
- **Arms (`Holm m = 2` over `BAG_SWITCH` and `BAG_SWITCH_NESTED`):**
  - `r882` (`BASE`): Frozen `r882` slot `0`.
  - `RERANK8_VETO` (`REF`): Byte-identical `veto.run_cell_veto` (`tau = 0.10`, `K = 8`).
  - `BAG_SWITCH` (`A`, gated): Fits $B = 8$ `LGBMRanker` models ([bagged.py](bagged.py)) on family-level bootstrap resamples of the cell's `TRAIN` families at `(lr*, n_estimators*)`; switches from slot `0` to `RERANK8_VETO`'s pick `j_ref` iff at least $q = 6$ of $8$ bagged rankers satisfy `bag_score[b, j_ref] - bag_score[b, 0] > strict_margin`.
  - `BAG_SWITCH_NESTED` (`B`, gated): Same outer $B = 8$ bagged voting rule, with $q \in \{5, 6, 7\}$ chosen per cell by 4-fold inner family-grouped CV on `TRAIN` families only (`q = 5` selected in `33/60` cells, `q = 6` in `13/60`, `q = 7` in `14/60`).
  - `PLACEBO` (`BAG_SWITCH_SHUF_PLACEBO`): `BAG_SWITCH` ($q = 6$, $B = 8$) where both the candidate proposer and all $B = 8$ bagged voters are trained on within-family shuffled relevance labels (`topk.shuffle_within_family` on `TRAIN` families only).
- **Evaluation:** `4` outcome-blind partitions (`KM276140`, `KM276141`, `KM276142`, `WARD`) $\times$ `5` held-out clusters $\times$ `3` model seeds (`279101, 279102, 279103`) = `60` cells (`selection.json` `sha256 = 1fc341103fbbee4fc84a5fe5a4df68cdbe021943f9cdd0ea9d8950edc4fec794` fsync-locked before any `DL-Val91` read). Two-level cluster bootstrap `10,000` draws at `boot_seed = 279150`.

---

## 3. Outcomes

### 3.1 PRIMARY: Leave-Family-Cluster-Out on `DL-Train` (`1,219` contexts, `205` families; Elo with `r882 = 1500`)

| Contrast | Per-seed gain (`279101, 279102, 279103`) | Median `[CI95]` | Holm-level CI (`m=2`) | Holm $p$ | Seeds `> 0` | Harmed (`loss > 0.01`) by seed (median) | `W/T/L` (`\|d\| > 1e-4`) by seed | Elo |
|---|---|---|---|---:|---:|---|---|---:|
| **`BAG_SWITCH` − `r882`** | `+0.0585, +0.0545, +0.0594` | **`+0.0585 [+0.0385, +0.0735]`** | **`[+0.0368, +0.0763]`** | `0.0002` | **`3/3`** | **`24, 28, 31` (`28.0`, `0.56x REF`)** | `150/16/39, 150/12/43, 151/7/47` | **`1553.0`** |
| **`BAG_SWITCH_NESTED` − `r882`** | `+0.0595, +0.0568, +0.0555` | **`+0.0568 [+0.0405, +0.0724]`** | **`[+0.0382, +0.0747]`** | `0.0002` | **`3/3`** | **`28, 30, 33` (`30.0`, `0.60x REF`)** | `146/13/46, 148/10/47, 152/7/46` | **`1551.6`** |
| `RERANK8_VETO` (`REF`) − `r882` | `+0.0676, +0.0595, +0.0599` | `+0.0599 [+0.0381, +0.0820]` | `[+0.0360, +0.0857]` | — | `3/3` | `40, 50, 53` (`50.0`, `1.00x REF`) | `143/2/60, 135/0/70, 140/3/62` | `1556.8` |
| `BAG_SWITCH` − `RERANK8_VETO` | `-0.0114, -0.0055, +0.0090` | `-0.0055 [-0.0133, +0.0050]` | — | — | `1/3` | — | `89/4/112, 90/7/108, 92/11/102` | — |
| `BAG_SWITCH_NESTED` − `RERANK8_VETO` | `-0.0065, -0.0021, -0.0059` | `-0.0059 [-0.0142, +0.0051]` | — | — | `0/3` | — | `87/10/108, 88/10/107, 90/8/107` | — |
| `BAG_SWITCH_NESTED` − `BAG_SWITCH` | `+0.0015, +0.0028, -0.0049` | `+0.0015 [-0.0034, +0.0034]` | — | — | `2/3` | — | `73/77/55, 63/93/49, 57/72/76` | — |
| `PLACEBO` − `r882` | `+0.0001, -0.0003, -0.0050` | **`-0.0003 [-0.0036, +0.0001]`** | — | — | `1/3` | `4, 24, 22` (`22.0`) | `12/178/15, 32/125/48, 32/128/45` | `1495.6` |

#### Per Seed $\times$ Partition Equal-Family Gain vs `r882`

| Arm | Seed | `KM276140` | `KM276141` | `KM276142` | `WARD` | Seed Median |
|---|---|---|---|---|---|---|
| **`BAG_SWITCH` ($q=6$)** | `279101` | `+0.0508` | `+0.0601` | `+0.0569` | `+0.0617` | **`+0.0585`** |
| **`BAG_SWITCH` ($q=6$)** | `279102` | `+0.0525` | `+0.0565` | `+0.0568` | `+0.0512` | **`+0.0545`** |
| **`BAG_SWITCH` ($q=6$)** | `279103` | `+0.0511` | `+0.0665` | `+0.0523` | `+0.0714` | **`+0.0594`** |
| **`BAG_SWITCH_NESTED`** | `279101` | `+0.0599` | `+0.0558` | `+0.0592` | `+0.0624` | **`+0.0595`** |
| **`BAG_SWITCH_NESTED`** | `279102` | `+0.0467` | `+0.0655` | `+0.0604` | `+0.0531` | **`+0.0568`** |
| **`BAG_SWITCH_NESTED`** | `279103` | `+0.0671` | `+0.0530` | `+0.0560` | `+0.0550` | **`+0.0555`** |
| `RERANK8_VETO` (`REF`) | `279101` | `+0.0648` | `+0.0804` | `+0.0492` | `+0.0705` | `+0.0676` |
| `RERANK8_VETO` (`REF`) | `279102` | `+0.0346` | `+0.0686` | `+0.0616` | `+0.0573` | `+0.0595` |
| `RERANK8_VETO` (`REF`) | `279103` | `+0.0405` | `+0.0567` | `+0.0730` | `+0.0632` | `+0.0599` |

- **Cluster consistency:** Under both `BAG_SWITCH` and `BAG_SWITCH_NESTED`, `5/5` clusters are non-negative (`>= 2/3` seeds) in `KM276140`, `KM276142`, and `WARD`, and `4/5` in `KM276141` (where only `KM276141` cluster 3 is slightly negative at `-0.0053` and `-0.0013` on seeds 2–3, vs `-0.0136` and `-0.0143` under `RERANK8_VETO`). Notably, `KM276140` cluster 2 on seed `279102` improved from `-0.0793` (`RERANK8_VETO`) to **`+0.0448` (`BAG_SWITCH`)**, and `WARD` cluster 1 (`39` families) is positive in `3/3` seeds (`+0.0172, +0.0515, +0.0331`).
- **Invalid and changed picks (`DL-Train`, `14,628` context evaluations across `60` cells):**
  - Invalid `-2.0` picks: `r882 = 204`, `RERANK8_VETO = 178`, **`BAG_SWITCH = 178`**, **`BAG_SWITCH_NESTED = 177`**, `PLACEBO = 208`.
  - Changed picks (switches away from `r882` slot `0`): `RERANK8_VETO = 7,582` (`51.8%`), **`BAG_SWITCH = 4,117` (`28.1%`)**, **`BAG_SWITCH_NESTED = 4,164` (`28.5%`)**, `PLACEBO = 279` (`1.9%`).

### 3.2 SECONDARY: `DL-Val91` (Supporting Only; `KM276140` 5-Model Mixture per Seed)

| Contrast | Per-seed gain (`279101, 279102, 279103`) | Median `[CI95]` | Seeds `> 0` | Harmed (`loss > 0.01`) by seed (median) | Elo |
|---|---|---|---:|---|---:|
| **`BAG_SWITCH` − `r882`** | `+0.0180, +0.0166, +0.0246` | **`+0.0180 [+0.0101, +0.0285]`** | **`3/3`** | `0, 1, 0` (**`0.0`**) | **`1554.1`** |
| **`BAG_SWITCH_NESTED` − `r882`** | `+0.0210, +0.0181, +0.0273` | **`+0.0210 [+0.0122, +0.0321]`** | **`3/3`** | **`0, 0, 0` (`0.0`)** | **`1571.7`** |
| `RERANK8_VETO` (`REF`) − `r882` | `+0.0346, +0.0303, +0.0392` | `+0.0346 [+0.0180, +0.0532]` | `3/3` | `0, 1, 0` (`0.0`) | `1583.9` |
| `PLACEBO` − `r882` | `-0.0005, -0.0007, +0.0034` | `-0.0005 [-0.0014, +0.0042]` | `1/3` | `0, 3, 0` (`0.0`) | `1497.0` |

- **Watch family `fam-769243ee920953f082bd6784c94795f919abe67c243fae912657abc080863c0c` (`DL-Val91`):**
  - `RERANK8_VETO` − `r882`: `[-0.003292, -0.004148, -0.004586]` (median `-0.004148`)
  - `BAG_SWITCH` − `r882`: `[-0.000139, -0.000946, -0.001070]` (median `-0.000946`, **`-77.2%` smaller loss**)
  - `BAG_SWITCH_NESTED` − `r882`: `[-0.000199, -0.000122, -0.001130]` (median `-0.000199`, **`-95.2%` smaller loss**)
  - Neither `BAG_SWITCH` nor `BAG_SWITCH_NESTED` harms `fam-769243ee…` at the `loss > 0.01` threshold on any seed, though its raw delta remains slightly negative (`< -1e-4`) because 1 of the 5 `KM276140` cell models in the `DL-Val91` mixture still switches on one context of that family.

---

## 4. Gate-by-Gate Decision (`Holm m = 2` over `BAG_SWITCH` and `BAG_SWITCH_NESTED`)

| Gate | Requirement | `BAG_SWITCH` ($q=6$) | `BAG_SWITCH_NESTED` ($q \in \{5,6,7\}$) | Status |
|---|---|---|---|---|
| 1. `median_gt_0_010` | Median gain vs `r882 > +0.010` | `+0.0585 > +0.010` | `+0.0568 > +0.010` | **PASS** |
| 2. `three_seeds` | Positive on `3/3` seeds | `3/3` (`+0.0585, +0.0545, +0.0594`) | `3/3` (`+0.0595, +0.0568, +0.0555`) | **PASS** |
| 3. `holm_lcb_gt_0` | Holm (`m=2`) LCB `> 0` | `+0.0368 > 0` ($p = 0.0002$) | `+0.0382 > 0` ($p = 0.0002$) | **PASS** |
| 4. `harmed_le_075_ref` | Harmed (`loss > 0.01`) `<= 0.75 x REF` (`<= 37.5`) | **`28.0 <= 37.5` (`0.560x REF`)** | **`30.0 <= 37.5` (`0.600x REF`)** | **PASS** |
| 5. `gain_ge_080_ref` | Median gain `>= 0.80 x REF` (`>= +0.0479`) | **`+0.0585 >= +0.0479` (`0.976x REF`)** | **`+0.0568 >= +0.0479` (`0.948x REF`)** | **PASS** |
| 6. `invalid_le_ref` | `DL-Train` `-2.0` picks `<= REF` (`<= 178`) | **`178 <= 178`** | **`177 <= 178`** | **PASS** |
| 7. `clusters_4_of_5_every_partition` | `>= 4/5` clusters non-neg per partition | `5, 4, 5, 5` (`>= 4` in all 4) | `5, 4, 5, 5` (`>= 4` in all 4) | **PASS** |
| 8. `placebo_le_0_005` | `PLACEBO <= +0.005` & does not pass | `-0.0003 <= +0.005` (`False`) | `-0.0003 <= +0.005` (`False`) | **PASS** |
| 9. `dlval_gain_gt_0` | `DL-Val91` median gain vs `r882 > 0` | `+0.0180 > 0` (`3/3`) | `+0.0210 > 0` (`3/3`) | **PASS** |
| **Overall Arm Verdict** | All `9/9` gates required | **`ADVANCE` (`9/9`)** | **`ADVANCE` (`9/9`)** | **`ADVANCE`** |

---

## 5. Self Red-Team, Deviations & Next Step

- **Deviations:** None in science, thresholds, grids, or data. After the initial preregistration push (`1e023912e42d`), a concurrent whitepaper commit (`ea8aa652a1`) advanced `origin/v1/next` 15 seconds before `attempt-001` started, causing `validate_launch` to exit at line 76 before opening any input file (`outcomes_read = 0`; preserved in `prelaunch-refusal-001.*`). We updated `runner.py` (`aecf7f5a133c`, `outcomes_read = 0`, 24 minutes before the `22:25Z` prereg deadline) to verify `git merge-base --is-ancestor launch["source_commit"] refs/remotes/origin/v1/next` so concurrent `v1/next` commits do not cause false launch refusals.
- **Why fixed $q = 6$ (`BAG_SWITCH`) slightly edges `BAG_SWITCH_NESTED` on `DL-Train` (`+0.0585` vs `+0.0568`, `28` vs `30` harmed):** In `33/60` cells, inner CV selected $q = 5$ because $q = 5$ already met the `<= 0.75 x REF` harm gate on inner OOF; on out-of-cluster test families, fixed $q = 6$ provides slightly stronger protection against distribution shift (`28` vs `30` harmed families) while losing almost no lift (`97.6%` of `REF`). Conversely, on `DL-Val91` (`KM276140` 5-model mixture), `BAG_SWITCH_NESTED` edges `BAG_SWITCH` (`+0.0210` vs `+0.0180`, Elo `1571.7` vs `1554.1`, `0` harmed families with `loss > 0.01` on all 3 seeds).
- **Next Step:** `BAG_SWITCH` ($B = 8$ family-bootstrapped rankers, $q = 6$ fixed on top of `RERANK8_VETO`) is the simpler, hyperparameter-free winner on primary leave-cluster-out (`+0.0585`, `97.6%` of `REF` lift, `-44.0%` harmed families) and is ready for clean-room frozen export/serving integration once `P2` export unblocks.

---

## 6. Durable S3 Artifacts (`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H279/attempt-001/`)

| Artifact | S3 Key | `VersionId` | Bytes | `sha256` |
|---|---|---|---:|---|
| `s3_manifest.json` | `.../attempt-001/s3_manifest.json` | `PcULtHIHOtB6BiqWU5gFSOdj7CCUW1Pa` | `3,489` | — |
| `metrics_full.json` | `.../attempt-001/metrics_full.json` | `B.7MWfebzE7nyZ7Rw47vJsNPe2PQpLsP` | `380,487` | `dc43963513f919be…` |
| `selection.json` | `.../attempt-001/selection.json` | `.ecDEnASoaFZZyVwgHwWbCNFpscXo9ZY` | `214,607` | `1fc341103fbbee4f…` |
| `primary_panel.npz` | `.../attempt-001/primary_panel.npz` | `M6Juzo24t7Ta9JVUeBL_idgqmeCxJDbN` | `193,857` | `26167283937b6df3…` |
| `secondary_panel.npz` | `.../attempt-001/secondary_panel.npz` | `_dIixiZL2b0ajfSeOa4W1y6oUIghZiRR` | `5,058` | `761eab9ec148e6e9…` |
| `attempt-001.stdout` | `.../attempt-001/attempt-001.stdout` | `eMPFALk7TOoehMcby_yHnnXCL0eBK7Rg` | `116,713` | `91bdf75793c95bd0…` |
| `attempt-001.resources.jsonl` | `.../attempt-001/attempt-001.resources.jsonl` | `XgLJ3MSiIfhV93fyma9Ap6jHsuJJ.YA7` | `1,220,770` | `2799b8ff4a7b1238…` |
| `attempt-001.resource_summary.json` | `.../attempt-001/attempt-001.resource_summary.json` | `EMPfHCT2wQlEyR_OB23SxDnSkOKpmhMN` | `856` | `72c0a04aef25ce7f…` |
