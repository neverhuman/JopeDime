# H284 — `h284_bagswitch_frozen_fresh` (`NOT_ADVANCED_FRESH`)

- **Wave**: `H284` (`h284_bagswitch_frozen_fresh`)
- **Owner**: `rd_argon` (`[ARGON]`)
- **Authority**: `/home/ubuntu/nudges/hw_t2305_argon_h284.md`
- **Verdict**: **`NOT_ADVANCED_FRESH`** (`3/4` primary gates passed on the `62`-family fresh union; passes `4/4` gates on `H282`'s distinct `r782/r785` GPU `DL-Val91` pools)
- **Parent lineage**: `H279` (`h279_rerank8_bagged_strict_switch`, commit `0b733a590c9cda3a67fa66ede4ded883ca83c5c2`, `ADVANCE`) and `H278` (`h278_rerank8_veto_frozen_export_transfer`, S3 export manifest `VersionId = S_PoMNCZ0jiAYru1vEligisesVuKDuPf`)
- **Host / runtime**: `xbabe2` CPU only (`taskset -c 57-60 nice -n 19`, numeric threads `1`, LightGBM `n_jobs = 1`, peak RSS `< 1.0 GiB`, zero neural optimizer updates; `DOPE` composite `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` verified before and after)

---

## 1. Step 1 — Pre-Pool Regression & Full-`DL-Train` Freeze / Export (`fresh_pools_read = 0`)

Before any fresh pool existed or was read (`fresh_pools_read = 0`, `dl_val_reads_before_lock = 0`, preregistration published on `origin/v1/next` at `972d3011e9f9655c9180e075012477084420ce85`):

1. **Bit-Exact Regressions**:
   - `Native91`: `1.1377941898164525` (`91/91` exact context matches), `TOP4_PICK@r882`: `1.2403601682250796` (ctx) / `1.2519997168684853` (fam).
   - `H279` cell `KM276140-S279101-C0`: reproduced bit-for-bit (`receipts`, `assess_picks`, `outer_votes`, `shuf_votes`, `invalid`, `changed`).
2. **Full-`DL-Train` (`205` families, `1,219` contexts) Export (`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H284/export/`)**:
   - `export_manifest.json`: `VersionId = AHFLVzBwd_fOfBAFGwHd.ABmkbgIbUu3`, `sha256 = b570d3af860dfa69ec4b779315bb2b9924515c40a9e58e2a24e9b92f98a8febc`
   - `fit_lock.json`: `VersionId = 4DfJg9U5rybBD2b4UFNTyLnnnmpUb._b`, `sha256 = 7a65a6c41f04a42510773a57f35e5d01fed7ddaeee78b76926884aa0c9d2257f`
   - All `30` exported LightGBM models (`ranker-seed{s}.txt`, `veto-seed{s}.txt`, `bag0..7-seed{s}.txt` across seeds `279101–279103`) verified reload bit-exact on `64` `DL-Train` contexts (`num_feature = 568`):
     - **Seed `279101`**: `lr = 0.03`, `n_estimators = 50`, `strict_margin = 0.02`, `strict_admitted = true`, `q_fixed = 6`, `q_selected = 6` (`tier1_both`), `ranker` `VersionId = WEcQw3NB2dUpFCUtSsWLNMYqQqTugky0`, `veto` `VersionId = y_FrwdBuPTWLHfeU6zn1Nli1eucdSkNq`
     - **Seed `279102`**: `lr = 0.05`, `n_estimators = 50`, `strict_margin = 0.10`, `strict_admitted = true`, `q_fixed = 6`, `q_selected = 5` (`tier1_both`), `ranker` `VersionId = FH4iAWCFfRNdmwh2SbKWiiKkLCAIfQyB`, `veto` `VersionId = 0jVsW1uuwFBf7W_x8mkULfEWpBYQYkVd`
     - **Seed `279103`**: `lr = 0.05`, `n_estimators = 100`, `strict_margin = 0.10`, `strict_admitted = true`, `q_fixed = 6`, `q_selected = 6` (`tier1_both`), `ranker` `VersionId = Usj49zDv7ETn9nuNHZDgAoIFXQbi9tJQ`, `veto` `VersionId = .OW6QH5dIC_7kjm_BZ1DzvUKuRF7gKyq`

---

## 2. Step 2 — Primary Evaluation on Unseen Fresh Pools (`H283` EVEN $\cup$ `H285` ODD = `62` Families)

- **Shared fresh cohort (`fresh_cohort_v1.json`)**: `sha256 = 12d2b1f37bf0e3c48ee61a24949b4f504251fe56b9b26b3081398623982f323c`
- **`H283` EVEN fresh packets**: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H283/fresh_pools/fresh_top8_packets.npz` (`VersionId = 8_ZHsSqRsNFjOKLg0_1Itd9m8qMZLNFu`, `29` unseen regression families/contexts) + `fresh-build.tar.gz` (`VersionId = 1C.WHs4nOifut6lWLwKOBkN8JYCDHJwL`)
- **`H285` ODD fresh packets**: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H285/astra-t2305/fresh/fresh-top8-no-picks.npz` (`VersionId = 73pe6R.AKcET0qRgr__KEFBj.WDJLZju`, `33` unseen regression families/contexts) + `full-outputs.tar.gz` (`VersionId = x7Hj8CQ6ZtURHZXxmUNlXwp2.am44NXD`)

### Table 1 — Primary Union (`62` Unseen Fresh Families: `29` EVEN + `33` ODD; `slot 0` Mean = `1.3186`, `slot 0` Invalid = `[2, 2, 2]`)

| Arm | Per-Seed Gain vs `slot 0` (`279101 / 279102 / 279103`) | Median Gain vs `slot 0` `[CI95]` | Per-Seed `LCB95 > 0` | Median Gain vs `RERANK8_VETO` `[CI95]` | `W/T/L` vs `slot 0` by Seed (`|d| <= 1e-4` tie) | Harmed (`loss > 0.01`) by Seed (Med / Ratio vs `VETO`) | Invalid `-2.0` Picks by Seed | Changed Picks by Seed | Median Elo (`slot 0 = 1500`) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **`BAG_SWITCH` ($q=6$)** | `+0.1380 / +0.0306 / +0.0222` | **`+0.0306 [+0.0039, +0.0710]`** | `2/3` (`[+0.0414, +0.2622]`, `[+0.0032, +0.0684]`, `[-0.0183, +0.0654]`) | `-0.0872 [-0.1650, -0.0042]` (`-0.0247` vs `H278`) | `22/37/3`, `7/53/2`, `12/45/5` | `[3, 1, 5]` (**`3.0` = `0.300x` paired / `0.333x` `H278`**) | **`[0, 2, 2]`** (`<= slot 0`) | `[25, 10, 18]` (med `18.0`) | **`1549.8`** |
| **`BAG_SWITCH_NESTED` ($q^*=[6,5,6]$)** | `+0.1380 / +0.0342 / +0.0222` | **`+0.0342 [+0.0067, +0.0730]`** | `2/3` (`[+0.0424, +0.2664]`, `[+0.0063, +0.0711]`, `[-0.0185, +0.0660]`) | `-0.0872 [-0.1642, -0.0039]` (`-0.0247` vs `H278`) | `22/37/3`, `10/49/3`, `12/45/5` | `[3, 2, 5]` (**`3.0` = `0.300x` paired / `0.333x` `H278`**) | **`[0, 2, 2]`** (`<= slot 0`) | `[25, 14, 18]` (med `18.0`) | **`1553.2`** |
| `RERANK8_VETO` (`279101–279103`) | `+0.1426 / +0.1448 / +0.1093` | `+0.1426 [+0.0430, +0.2314]` | `2/3` | `0.0000` | `33/16/13`, `25/32/5`, `23/26/13` | `[12, 3, 10]` (med **`10.0`**) | `[0, 2, 1]` | `[48, 32, 37]` (med `37.0`) | `1644.6` |
| `H278_RERANK8_VETO` (`278101–278103`) | `+0.1224 / +0.1088 / +0.0469` | `+0.1088 [-0.0187, +0.2539]` | `1/3` | — | `28/23/11`, `22/34/6`, `27/25/10` | `[10, 5, 9]` (med **`9.0`**) | `[0, 0, 1]` | `[39, 29, 38]` (med `38.0`) | `1615.2` |

### Primary Gate Disposition on Fresh Union (`62` Families)

1. **`BAG_SWITCH - slot0` family-bootstrap `LCB95 > 0` on `3/3` seeds**: **FAIL (`2/3` seeds)** — seeds `279101` (`LCB95 = +0.0414`) and `279102` (`LCB95 = +0.0032`) pass, while seed `279103` has positive gain (`+0.0222`, `12W/45T/5L`) but its 62-family individual `LCB95` is `-0.0183` (the across-seed median bootstrap `CI95` is strictly positive: `[+0.0039, +0.0710]` for `BAG_SWITCH` and `[+0.0067, +0.0730]` for `BAG_SWITCH_NESTED`).
2. **Median per-family gain across seeds `> +0.010`**: **PASS** (`+0.0306` for `BAG_SWITCH`, `+0.0342` for `BAG_SWITCH_NESTED`).
3. **Harmed fresh families (`loss > 0.01` vs `slot 0`) `<= 0.75 *` frozen `VETO`'s harmed count**: **PASS** (`3.0` vs `10.0` = **`0.300x`** against paired `RERANK8_VETO`, and `3.0` vs `9.0` = **`0.333x`** against frozen `H278` `RERANK8_VETO` — a **66.7%–70.0% harm reduction**).
4. **Invalid `-2.0` picks not above `slot 0`**: **PASS** (`[0, 2, 2]` vs `[2, 2, 2]` on `3/3` seeds; seed `279101` rescues both `slot 0` invalid picks).

### Table 2 — Separate Fresh Pool-Sets (`H283` EVEN `29` Families vs `H285` ODD `33` Families)

| Pool-Set | Arm | Per-Seed Gain vs `slot 0` | Median Gain `[CI95]` | `LCB95 > 0` Seeds | `W/T/L` by Seed | Harmed (`> 0.01`) by Seed (Med) | Invalid by Seed (`slot 0`) | Changed by Seed | Median Elo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **`H283` EVEN (`29` fams, GPU)** | `BAG_SWITCH` ($q=6$) | `+0.1133 / +0.0636 / +0.0391` | **`+0.0636 [+0.0058, +0.1419]`** | `2/3` | `10/17/2`, `5/23/1`, `8/18/3` | `[2, 1, 3]` (**`2.0` = `0.40x VETO`**) | `[0, 0, 0]` (`[0,0,0]`) | `[12, 6, 11]` | `1576.6` |
| **`H283` EVEN (`29` fams, GPU)** | `BAG_SWITCH_NESTED` | `+0.1133 / +0.0679 / +0.0391` | **`+0.0679 [+0.0099, +0.1456]`** | `2/3` | `10/17/2`, `7/20/2`, `8/18/3` | `[2, 2, 3]` (**`2.0` = `0.40x VETO`**) | `[0, 0, 0]` (`[0,0,0]`) | `[12, 9, 11]` | `1583.1` |
| `H283` EVEN (`29` fams, GPU) | `RERANK8_VETO` (`H284`) | `+0.0767 / +0.1067 / +0.1054` | `+0.1054 [+0.0099, +0.2056]` | `1/3` | `15/7/7`, `11/15/3`, `13/10/6` | `[7, 3, 5]` (`5.0`) | `[0, 0, 0]` | `[23, 14, 19]` | `1625.7` |
| `H283` EVEN (`29` fams, GPU) | `H278_RERANK8_VETO` | `+0.0172 / +0.0129 / +0.0004` | `+0.0129 [-0.1449, +0.1415]` | `0/3` | `15/8/6`, `12/15/2`, `14/10/5` | `[6, 2, 5]` (`5.0`) | `[0, 0, 0]` | `[21, 15, 19]` | `1634.5` |
| **`H285` ODD (`33` fams, CPU)** | `BAG_SWITCH` ($q=6$) | `+0.1597 / +0.0017 / +0.0073` | **`+0.0073 [+0.0005, +0.0234]`** | `1/3` | `12/20/1`, `2/30/1`, `4/27/2` | `[1, 0, 2]` (**`1.0` = `0.20x VETO`**) | `[0, 2, 2]` (`[2,2,2]`) | `[13, 4, 7]` | `1526.6` |
| **`H285` ODD (`33` fams, CPU)** | `BAG_SWITCH_NESTED` | `+0.1597 / +0.0046 / +0.0073` | **`+0.0073 [+0.0009, +0.0234]`** | `1/3` | `12/20/1`, `3/29/1`, `4/27/2` | `[1, 0, 2]` (**`1.0` = `0.20x VETO`**) | `[0, 2, 2]` (`[2,2,2]`) | `[13, 5, 7]` | `1527.1` |
| `H285` ODD (`33` fams, CPU) | `RERANK8_VETO` (`H284`) | `+0.2005 / +0.1784 / +0.1128` | `+0.1784 [+0.0271, +0.3243]` | `3/3` | `18/9/6`, `14/17/2`, `10/16/7` | `[5, 0, 5]` (`5.0`) | `[0, 2, 1]` | `[25, 18, 18]` | `1653.2` |
| `H285` ODD (`33` fams, CPU) | `H278_RERANK8_VETO` | `+0.2149 / +0.1930 / +0.0878` | `+0.1930 [+0.0145, +0.4577]` | `3/3` | `13/15/5`, `10/19/4`, `13/15/5` | `[4, 3, 4]` (`4.0`) | `[0, 0, 1]` | `[18, 14, 19]` | `1598.6` |

---

## 3. Descriptive Evaluation on `H282`'s Genuinely Distinct `r782` & `r785` GPU `DL-Val91` Pools (`36` Families, `91` Contexts)

On `H282`'s genuinely distinct stored GPU candidate pools (`r782`, `r785`, and their combined mean `r782_r785_distinct_mean_dlval91`), **both `BAG_SWITCH` ($q=6$) and `BAG_SWITCH_NESTED` ($q^*=[6,5,6]$) satisfy all `4/4` gates** (`3/3` seeds `LCB95 > 0`, median gain `+0.0326 > +0.010`, `0` harmed families on `r782/r785` mean, and `0` invalid picks across all `3/3` seeds):

| Pool | Arm | Per-Seed Gain vs `slot 0` | Median Gain `[CI95]` | `LCB95 > 0` Seeds | Harmed (`> 0.01`) by Seed (Med) | Invalid by Seed (`slot 0`) | Changed by Seed | Median Elo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **`r782` + `r785` Mean** | **`BAG_SWITCH` ($q=6$)** | **`+0.0326 / +0.0295 / +0.0422`** | **`+0.0326 [+0.0093, +0.0693]`** | **`3/3`** | **`[0, 0, 0]` (`0.0`)** | **`[0, 0, 0]` (`[1,1,1]`)** | `[71, 38, 58]` | **`1738.8`** |
| **`r782` + `r785` Mean** | **`BAG_SWITCH_NESTED`** | **`+0.0326 / +0.0327 / +0.0422`** | **`+0.0327 [+0.0091, +0.0781]`** | **`3/3`** | **`[0, 1, 0]` (`0.0`)** | **`[0, 0, 0]` (`[1,1,1]`)** | `[71, 52, 58]` | **`1694.0`** |
| `r782` + `r785` Mean | `RERANK8_VETO` (`H284`) | `+0.0543 / -0.0003 / +0.0603` | `+0.0543 [+0.0170, +0.1090]` | `2/3` | `[2, 2, 0]` (`2.0`) | `[0, 2, 0]` (`[1,1,1]`) | `[121, 68, 106]` | `1812.4` |
| `r782` + `r785` Mean | `H278_RERANK8_VETO` | `+0.0796 / +0.0419 / +0.0503` | `+0.0503 [+0.0138, +0.1087]` | `3/3` | `[1, 1, 4]` (`1.0`) | `[0, 0, 0]` (`[1,1,1]`) | `[107, 72, 109]` | `1820.9` |
| `r782` GPU `DL-Val91` | `BAG_SWITCH` ($q=6$) | `+0.0445 / +0.0442 / +0.0584` | `+0.0445 [+0.0063, +0.1181]` | `3/3` | `[2, 1, 0]` (`1.0`) | `[0, 0, 0]` (`[1,1,1]`) | `[35, 19, 29]` | `1669.6` |
| `r782` GPU `DL-Val91` | `BAG_SWITCH_NESTED` | `+0.0445 / +0.0500 / +0.0584` | `+0.0500 [+0.0055, +0.1373]` | `3/3` | `[2, 3, 0]` (`2.0`) | `[0, 0, 0]` (`[1,1,1]`) | `[35, 27, 29]` | `1635.3` |
| `r785` GPU `DL-Val91` | `BAG_SWITCH` ($q=6$) | `+0.0208 / +0.0149 / +0.0261` | `+0.0208 [+0.0080, +0.0333]` | `3/3` | `[1, 0, 0]` (`0.0`) | `[0, 0, 0]` (`[0,0,0]`) | `[36, 19, 29]` | `1727.1` |
| `r785` GPU `DL-Val91` | `BAG_SWITCH_NESTED` | `+0.0208 / +0.0154 / +0.0261` | `+0.0208 [+0.0080, +0.0333]` | `3/3` | `[1, 1, 0]` (`1.0`) | `[0, 0, 0]` (`[0,0,0]`) | `[36, 25, 29]` | `1702.2` |
| `r3175` GPU `DL-Val91` | `BAG_SWITCH` ($q=6$) | `+0.0232 / +0.0169 / +0.0273` | `+0.0232 [+0.0093, +0.0352]` | `3/3` | `[1, 0, 1]` (`1.0`) | `[0, 0, 0]` (`[0,0,0]`) | `[34, 18, 30]` | `1700.8` |
| `r882` CPU `DL-Val91` | `BAG_SWITCH` ($q=6$) | `+0.0212 / +0.0150 / +0.0282` | `+0.0212 [+0.0088, +0.0340]` | `3/3` | `[1, 0, 0]` (`0.0`) | `[0, 0, 0]` (`[0,0,0]`) | `[36, 19, 31]` | `1709.4` |

---

## 4. S3 Artifacts & Closeout Receipts (`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H284/`)

- **Frozen `H284` Export (`export/`)**:
  - `export_manifest.json`: `VersionId = AHFLVzBwd_fOfBAFGwHd.ABmkbgIbUu3`, `sha256 = b570d3af860dfa69ec4b779315bb2b9924515c40a9e58e2a24e9b92f98a8febc`
  - `fit_lock.json`: `VersionId = 4DfJg9U5rybBD2b4UFNTyLnnnmpUb._b`, `sha256 = 7a65a6c41f04a42510773a57f35e5d01fed7ddaeee78b76926884aa0c9d2257f`
- **Transfer Evaluation (`attempt-001/`)**:
  - `s3_manifest.json`: `VersionId = gMYcsVvRVRC1v1MgFjGweZJl2GavdBzc`, `sha256 = e43b716b6c9a8df6eca9ae3ee40ac666c78430b5d12bc6a50fa694191fb40967`
  - `metrics_full.json`: `VersionId = cXFV4GJ5olfIGKTKYwPO0nUvk40ZGJD_`, `sha256 = 3a4a0b248a6d1d93007f5d967754e54f2592c3fb7a59fa80d4cc38b473a7bd5f`
  - `fresh_union_panel.npz`: `VersionId = 76JnDgTaBybN_byiN7I83Tf28tH2ySFr`, `sha256 = ec986b093808c1714f223828be4c3c9f15bf1bcba59fb1b06c473eb0eb41bc9f`
  - `fresh_union_picks.json`: `VersionId = j5DCkTINmMWIw1.No.0WrPUSOzVIlZV2`, `sha256 = db8392af06b5b2b913f77a66a8f4dcfa98275b3f8211be0a5036efcf52490ff6`
  - `export_summary.json`: `VersionId = ttEN3rLTaJQp8ejp7JEbCVHEHzGElble`, `sha256 = eb48c15be8afcfa057c85c510a2fd63ec775ec20be3a7f56333f0d31ab0728f9`
  - `baseline.json`: `VersionId = HzJgq.terXhMPgRVTeLRPJ_6pcqU5Wov`, `sha256 = 2b8bb4994c5674999b31618de3ba25aa547281a69b53d88e732a3609e69ec25d`

