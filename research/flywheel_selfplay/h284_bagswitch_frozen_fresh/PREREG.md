# H284 Preregistration — `h284_bagswitch_frozen_fresh`

- **Wave**: `H284` (`h284_bagswitch_frozen_fresh`)
- **Owner**: `rd_argon` (`[ARGON]`)
- **Authority**: `/home/ubuntu/nudges/hw_t2305_argon_h284.md`
- **Parent lineage**: `H279` (`h279_rerank8_bagged_strict_switch`, commit `0b733a590c9cda3a67fa66ede4ded883ca83c5c2`, `ADVANCE`) and `H278` (`h278_rerank8_veto_frozen_export_transfer`, S3 export manifest `VersionId = S_PoMNCZ0jiAYru1vEligisesVuKDuPf`)
- **Host / compute budget**: `xbabe2` CPU only, `taskset -c 57-60 nice -n 19`, numeric threads `1`, LightGBM `n_jobs = 1`, peak RSS `<= 4 GiB`, zero neural optimizer updates (`DOPE` composite `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` verified before and after)
- **Deadlines**: prereg + frozen full-`DL-Train` export before `00:15Z Nov 4` (`fresh_pools_read = 0`); compute stop `02:30Z Nov 4`; closeout `03:15Z Nov 4`.

## 1. Hypothesis

Frozen full-`DL-Train` exports of `H279`'s `BAG_SWITCH` ($q = 6$) and `BAG_SWITCH_NESTED` ($q^* \in \{5, 6, 7\}$ selected by 4-fold inner family-grouped CV on `DL-Train` only) beat `slot 0` on the fresh unseen regression pools (`H283` EVEN half and/or `H285` ODD half, and their union) AND harm fewer fresh families (`loss > 0.01` vs `slot 0`) than frozen `RERANK8_VETO`.

## 2. Pinned Inputs & Outcome-Blind Fresh Cohort (`fresh_cohort_v1.json`)

- **Shared outcome-blind fresh cohort (`fresh_cohort_v1.json`)**:
  - Path: `/home/ubuntu/jope_plan/ns/receipts/fresh_cohort_v1.json`
  - `sha256`: `12d2b1f37bf0e3c48ee61a24949b4f504251fe56b9b26b3081398623982f323c`
  - Counts: `2,729` eligible unseen regression families (`1,972` buildable: `964` EVEN half, `1,008` ODD half); first `60` buildable EVEN families assigned to `H283` (`rd_opus`, GPU path) and first `60` buildable ODD families assigned to `H285` (`rd_astra`, CPU path).
- **DL-Train / DL-Val91 inputs (for full-`DL-Train` export and descriptive evaluation only)**:
  - `round-882.pt`: `0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64` (`DOPE` `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`)
  - `r882-train-costs.npz`: `57004a91a315b372cde62b126be20edd4eb0f337942e08b19bd85438b520f69f`
  - `r882-secondary-costs.npz`: `0e812aa7ea4ce46a85cbe17cb7eba3a58678cf13ec3b215d48a334dd5626f11f`
  - `selected.json`: `981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1` (`1,219` `DL-Train` contexts across `205` families; `91` `DL-Val` contexts across `36` families)
  - `H279` regression reference (`KM276140-S279101-C0.json`): `ee0d0a1a2ef2d4ce9b3ab748b2ef995fdcfdd016689e0c1cf2f4cbde393cb049`
- **Frozen `H278` `RERANK8_VETO` export (`278101–278103`)**:
  - S3 manifest: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H278/export/export_manifest.json` (`VersionId = S_PoMNCZ0jiAYru1vEligisesVuKDuPf`, `sha256 = 2fce784a959179e0522dce7221c9e1bc73c9d994672af91cd7637b9ad12e9643`)
  - `ranker-seed278101.txt`: `VersionId = yZXnwTfHBK3.U1bZDl1fB35GWifsev8l`, `sha256 = 7a690df843d8ac78f1b3bbbb40c4bfccfa44783c863ae9be26aa66c3abea797b`
  - `veto-seed278101.txt`: `VersionId = z.DQomoEdHXaz96vLF0o37jux3l2KpH.`, `sha256 = 1d099ac06298b460308cd4d55f444b082ee5c68673dd6ea0b0ee26299d5b508d`
  - `ranker-seed278102.txt`: `VersionId = bQcNpCmstPNy42QQHcO5F8fcZk_n3aZ1`, `sha256 = 2a2a8b3b5dd767f6c375bccac0f889291452f40c681d05199ec5014c500476cc`
  - `veto-seed278102.txt`: `VersionId = 4b9vDz.S3com.E6j7dGIukcfJ70kb4IF`, `sha256 = 9df7a85a622b51ae3e52651ed0f00236d2513caf9bdf3a35b7b15a7a39cfb246`
  - `ranker-seed278103.txt`: `VersionId = CnrMHiKmlTjj2_05GxmVK9MI3qkeprvq`, `sha256 = ae406a1c416cd780482b6ca6ff45db8fb054ab0a3527ec41ead6ec8008c89b2d`
  - `veto-seed278103.txt`: `VersionId = uuDUbHpR0CBuTMhf0.e3JSM.xpyc4q4I`, `sha256 = db7283f3c8da2112ab9295844b2d4bcad2f7e0440a9ec61a1795b8bd80f4aaf4`
- **Descriptive `H282` distinct GPU pools (`r782`, `r785`)**:
  - `r782/r3175-gpu-costs.npz`: `VersionId = YDJTrccbAuSDy5S.RG1btOUIgI88KQ9c`
  - `r785/r3175-gpu-costs.npz`: `VersionId = 3zn.w5YHpywRBrZTPdcfolGOnIWBCt3K`

## 3. Step 1 — Regression & Full-`DL-Train` Freeze / Export (`fresh_pools_read = 0`)

1. **Regression first**:
   - Verify `Native91` (`91/91`, `1.1377941898164525`) and `TOP4_PICK@r882` on `DL-Val91`.
   - Re-run `H279` cell `KM276140-S279101-C0` using byte-identical `topk.py`, `h265_core.py`, `veto.py`, and `bagged.py` and verify bit-exact equality of `receipts`, `invalid`, `changed`, `assess_picks`, `outer_votes`, and `shuf_votes`.
2. **Full-`DL-Train` Export (seeds `279101, 279102, 279103`)**:
   - Partition all `205` `DL-Train` families (`1,219` contexts) into 4 family-grouped folds (`freeze_bagged.family_folds(train, fam_by_cid, seed, n=4)`, identical stream to `H278/freeze.py`).
   - Run `bagged.run_cell_bagged` on all `1,219` `DL-Train` contexts (`fit_folds = [1, 2, 3, 4]`):
     - Inner 4-fold family-grouped CV selects `(lr*, n_estimators*, raw_margin)` and `strict_margin` for the `RERANK8_VETO` proposer ($\tau = 0.10$, $K = 8$), plus inner-CV $q^* \in \{5, 6, 7\}$ (`select_q_inner`) for `BAG_SWITCH_NESTED`.
     - Outer fit trains `ranker-seed{s}.txt`, `veto-seed{s}.txt`, and `bag0..7-seed{s}.txt` ($B = 8$ family-level bootstrap-bagged rankers resampling the `205` `DL-Train` families with replacement) on all `1,219` `DL-Train` contexts.
   - Verify reload bit-exactness of all `30` exported LightGBM models (`3` seeds $\times$ `10` models) on `64` `DL-Train` contexts (`num_feature() == 568`, `np.array_equal(mem, disk)`).
   - Upload all `30` model files, `fit_lock.json`, and `export_manifest.json` to `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H284/export/` with `VersionId` BEFORE any fresh pool is read (`fresh_pools_read = 0`).
3. **Locked `H284` Full-`DL-Train` Export S3 Receipts (`fresh_pools_read = 0`, `dl_val_reads_before_lock = 0`)**:
   - `export_manifest.json`: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H284/export/export_manifest.json` (`VersionId = AHFLVzBwd_fOfBAFGwHd.ABmkbgIbUu3`, `sha256 = b570d3af860dfa69ec4b779315bb2b9924515c40a9e58e2a24e9b92f98a8febc`)
   - `fit_lock.json`: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H284/export/fit_lock.json` (`VersionId = 4DfJg9U5rybBD2b4UFNTyLnnnmpUb._b`, `sha256 = 7a65a6c41f04a42510773a57f35e5d01fed7ddaeee78b76926884aa0c9d2257f`)
   - Seed `279101`: `lr = 0.03`, `n_estimators = 50`, `strict_margin = 0.02`, `q_fixed = 6`, `q_selected = 6` (`tier1_both`), `ranker-seed279101.txt` (`VersionId = WEcQw3NB2dUpFCUtSsWLNMYqQqTugky0`), `veto-seed279101.txt` (`VersionId = y_FrwdBuPTWLHfeU6zn1Nli1eucdSkNq`), plus `bag0..7-seed279101.txt` pinned in `export_manifest.json`.
   - Seed `279102`: `lr = 0.05`, `n_estimators = 50`, `strict_margin = 0.10`, `q_fixed = 6`, `q_selected = 5` (`tier1_both`), `ranker-seed279102.txt` (`VersionId = FH4iAWCFfRNdmwh2SbKWiiKkLCAIfQyB`), `veto-seed279102.txt` (`VersionId = 0jVsW1uuwFBf7W_x8mkULfEWpBYQYkVd`), plus `bag0..7-seed279102.txt` pinned in `export_manifest.json`.
   - Seed `279103`: `lr = 0.05`, `n_estimators = 100`, `strict_margin = 0.10`, `q_fixed = 6`, `q_selected = 6` (`tier1_both`), `ranker-seed279103.txt` (`VersionId = Usj49zDv7ETn9nuNHZDgAoIFXQbi9tJQ`), `veto-seed279103.txt` (`VersionId = .OW6QH5dIC_7kjm_BZ1DzvUKuRF7gKyq`), plus `bag0..7-seed279103.txt` pinned in `export_manifest.json`.

## 4. Step 2 — Primary Fresh-Pool Decision Rule & Descriptive Evaluation

When `/home/ubuntu/jope_plan/ns/receipts/fresh_pools_H283.json` and/or `/home/ubuntu/jope_plan/ns/receipts/fresh_pools_H285.json` appear (polled every 5 minutes until `02:15Z Nov 4`), fetch the fresh top-8 packets by S3 `VersionId` and apply unchanged:
- `slot0` (`r3175` / baseline `slot 0`),
- `RERANK8_VETO` (paired `279101–279103` full-`DL-Train` proposer),
- `H278_RERANK8_VETO` (frozen `H278` `278101–278103` full-`DL-Train` export),
- `BAG_SWITCH` ($q = 6$, `279101–279103`),
- `BAG_SWITCH_NESTED` ($q^* \in \{5, 6, 7\}$ per seed from `DL-Train` inner CV, `279101–279103`).

**Primary Decision Rule (on the union of available fresh pools; also reported separately for `H283` EVEN and `H285` ODD)**:
All 4 conditions must hold for `BAG_SWITCH` (and evaluated identically for `BAG_SWITCH_NESTED`):
1. `BAG_SWITCH - slot0` family-bootstrap `LCB95 > 0` on **`3/3` seeds** (`10,000` draws);
2. Median per-family gain across seeds **`> +0.010`**;
3. Harmed fresh families (`loss > 0.01` vs `slot 0`) **`<= 0.75 *` frozen `VETO`'s harmed count** (median over seeds);
4. Invalid `-2.0` picks **not above `slot 0`** (`invalid <= slot0` on `3/3` seeds).

- **Verdict**:
  - `FRESH_ADVANCE` if all 4 primary conditions hold on the fresh union;
  - `NOT_ADVANCED_FRESH` if fresh pools are evaluated and any primary condition fails;
  - `DESCRIPTIVE_ONLY` if no fresh pools appear by `02:15Z Nov 4` (reporting `H282`'s distinct `r782/r785` GPU `DL-Val91` pools and `r882/r3175` `DL-Val91` pools).
