# H417 PREREG — APEX corpus and baseline: the stored SXB recipe on fresh families

**Status.** Published on `origin/v1/next` before any H417 GP arm or deploy runs on the `BASE` panel and before
any APEX AUDIT value is read. §8 lists everything that existed before this PREREG. The gate G0 code (`g0_apex.py`,
§6.2) is published with it, before G0 runs.

- **Owner:** `cl_apex` (Claude Fable 5, xbabe2 session 947340e7). Claim: `ns/receipts/H417_claim.txt`
  (2026-11-10T06:27Z).
- **Authority:** owner order, 2026-11-10: study every H wave, then run the most aggressive waves to raise Elo on
  xbabe0–3 with H numbering; dev91 data and splits stay out of training. Owner answers the same day:
  - every learner must be ours (no external pretrained model);
  - DOPE `c51a56d0` stays frozen;
  - the fresh catalog families may join training, with 20% frozen as a never-trained confirmation cohort;
  - one dev91 once-score for the campaign's final champion only.
- **Hard constraints:**
  - H417 reads **no** dev91, DL-Val, DL-Holdout or sealed byte.
  - The `CONFIRM` cohort is neither deployed nor read. H417 opens no `CONFIRM` table.
  - No stored entrant is rerun or rescored. Nothing is written to the reference boards.
  - Commits go to `v1/next` only.

## 1. Purpose
H417 is the first wave of the APEX campaign (waves H417 onward). It tests no hypothesis. It delivers:
1. a frozen cohort manifest over 1,153 fresh families (§2);
2. the values of our strongest stored recipe, H416 `SXB`, and of its reference builders on fresh families (§3, §4).
   These are the baselines every later APEX wave is measured against;
3. per-context stack matrices on the `FIT-SEL` cohort, the training data of the learned stacker wave;
4. two diagnostics with reading rules fixed here (§6).

## 2. Cohorts
- **Source.** The four fresh-cohort rosters built by H286, H288, H292 and H294 (`fresh-cohort-v1/H<N>`): 200 + 300 +
  295 + 358 = 1,153 regression families, one train-only JTT1 context each (test rows stripped at build time).
- **Disjointness.**
  - H299's exclusion proof covers every family here: metadata-disjoint from the flywheel rosters DL-Train (1,219
    contexts), DL-Val = dev91 (91) and DL-Holdout (93), from prior sealed closures and from known sealed catalog ids.
    Its limit stands: no authoritative sealed-bucket family roster exists in the catalog (hash null in H299).
  - `cohorts_h417.sh` re-checks that no `dataset_id` occurs in the pinned TRAIN roster view (its 1,219 DL-Train rows
    and the 91 DL-Val ids it lists; ids only). Catalog family keys and the roster's coarse `fam-` keys are different
    namespaces, so the family comparison in that script cannot fail; family disjointness rests on H299.
  - `apex_core.selfcheck` asserts that no APEX context id, dataset id or family key matches the read guard's DL-Val
    identifier matcher.
- **Earlier exposure.** H283–H294 scored selector choices on these families (native 2-fold utility of candidates
  from stored pools). No model was ever trained on them, and no GP search or AUDIT split has been run on them.
  The H317 guard lists `fresh-cohort` among its sealed path tokens because these cohorts were then reserved as test
  populations; the owner released them for training on 2026-11-10. No APEX path contains that token.
- **Split rule (frozen).** `d = sha256("APEX-split-v1" ‖ NUL ‖ family_key)`; `k = int(d[0:8], 16) mod 5`.

  | k | Cohort | Families | Core tier | Role in the campaign |
  | --- | --- | --- | --- | --- |
  | 0 | `CONFIRM` | 241 | 142 | never trained on; read once by the champion wave |
  | 1 | `DEV` | 239 | 115 | each later wave's pre-registered test; champion composition |
  | 2 | `FIT-SEL` | 233 | 106 | deploys that train the learned stacker; within-wave design |
  | 3, 4 | `FIT-ICL` | 440 | 225 | real tables for training the in-context model |

- **Core tier (fixed from table shape only, before any outcome):** rows ≥ 800 and 3 ≤ width ≤ 400. This is the size
  regime of DL-Train and dev91. The fresh rosters also hold small tables (10% have ≤ 145 rows); a 30% AUDIT slice of
  those is too small to score. Smaller tables stay in the manifest for training use and are not deployed here.
- **Manifest.** `$H317_DATA/apex/APEX_COHORTS.json`, sha256 `11a57d80…bc4eb`, create-once; context-id list sha256
  `0b338662…7d9b`. Every table's sha256 equals its roster `train_only_jtt1_sha256`, and every table has 0 query rows.
- **AUDIT split.** `common_h317.audit_mask` unchanged: 30% of each context's rows, target-stratified, seeded by the
  context id.

## 3. Recipe: H416 at its published bytes
- **Code.** The published H416 code root (`~/h416-stage/pub-pub3`, verified against its `CODE.sha256`) plus the H417
  files. `deploy_h416.h416_unit` runs unchanged in mode `confirm` with the published design `DESIGN.json`
  (`BCAR.mix`, sha256 `0f0c96ad…f8c4`).
- **Panel `BASE`.** The core tier of `FIT-SEL` and `DEV`: 221 contexts in 221 families, one run, read out per cohort.
- **GP arms.** `GS` (`H0.1-A`), `GE` (`H381-E2-S3fz`), `GL` (`H415-L`): seed F + Evo A (K = 6 live / P = 48
  screened), 50 generations, superset worker `2a848c8f`, 60 s per-evaluation limit, 1 thread per worker. The unit is
  `gp_h381.unit_lf` with `lf_rows` 0, as in H415 and H416.
- **Deploy.** Per context the three arm memos are concatenated in the order GS, GE, GL (H416's order) and the deploy
  unit runs with a 300 s limit: replays, pools, library `LIB3`, H410's builders, the candidate set H, its 5-fold
  out-of-fold matrix Q, KFB, HFR, the fold-model and refit AUDIT columns, and `P3+LIB3.SXB`.
- **Differences from H416, all forced by new families** (`apex_core.py`):
  1. **Screener fold.** Every APEX target uses the fold-0 out-of-fold screener pair of each committee screener. No
     fold model saw an APEX family. This is `h335_lab.build_rt`'s Phase V construction, the one used on dev91.
  2. **Bank.** TRAIN champions and winners only; APEX family and dataset ids are added to the exclusion maps.
  3. **Table path and view recipe.** `C.table_path` and the view recipe's canonical-sha lookup are wrapped for APEX
     context ids. Every other id goes to the frozen function. No frozen file is changed.
  4. **All three arms are new runs** (H416 replayed the stored Phase T run as `GS`). Search seed 417101.
- **Memo conflicts.** H416 amendment 1 showed that a program timing out in one arm and completing in another makes a
  replay diverge from its arm's record. Here a divergent replay is not an error: the deploy unit's own replays define
  the pools. The number of contexts with a conflicting memo key is reported.

## 4. Readout (descriptive; `train_apex.py readout`)
- **Entrants per context:** `X0.GS` (the single-run stock `AGG.g50.CV3@12`, our H335 recipe), `GE.stock`, `GL.stock`,
  `BAG2` (`P2+LIB2.BAG_CAR`), `BAG3` (`P3+LIB3.BAG_CAR`), `KFB`, `HFR` and `SXB`.
- **Invalid outputs.** An invalid `X0.GS` scores 0 (the SEARCH-mean predictor). Any other invalid or missing output
  scores that context's `X0.GS` value and is counted (the H415 / H416 rule).
- **Reported for `DEV`, `FIT-SEL` and both together:**
  - family-mean AUDIT u per entrant;
  - each entrant against `X0.GS` by `h335_stats.primary_test` (clip ±0.5, one-sided sign-flip, leave-one-family-out
    minimum, family median) with the unclipped gain, % RMSE and families won / tied / lost;
  - `SXB` against `KFB`: the same, plus ΔElo with a family bootstrap (10,000 draws, seed 417150);
  - an internal Elo (anchor `X0.GS` = 1500);
  - worker-seconds per component.
- **No verdict depends on these numbers.** H417 closes `BASELINE_RECORDED` when the harness gates of §7 pass, whatever
  the values are. It closes `INCOMPLETE` otherwise.

## 5. What later waves take from H417
- `DEV`: the per-context values of `SXB` and `X0.GS` are the paired baselines of the later waves' tests.
- `FIT-SEL`: the stored matrices (Q, SEARCH y, fold labels, fold-model and refit AUDIT columns, AUDIT y) train the
  learned stacker. AUDIT rows of `FIT-SEL` contexts are training labels for that meta-learner; AUDIT rows of `DEV`
  are used for scoring only.

## 6. Diagnostics
### 6.1 Selection headroom (`train_apex.py headroom`; `FIT-SEL` contexts only)
Per context, from the stored matrices:
- `oracle_best_single_refit`: the highest AUDIT u among the usable programs' full-refit columns;
- `oracle_audit_fitted_caruana`: 100-step Caruana weights fitted on the AUDIT rows themselves.

Both are in-sample upper bounds and never deployable outputs. Reported: their family means and their paired
difference to `SXB`. Reading: the gap from `SXB` to the AUDIT-fitted stack bounds what any selector can gain on this
population; what remains beyond it needs new learners.

### 6.2 Screener gate G0 (`g0_apex.py`; decides whether the search-surrogate wave is built)
- **Panel `G24`:** the first 24 core `FIT-SEL` contexts in the order `stream_seed(417201, "G0", cid)`.
- **Arms**, each the unchanged locked search (seed F + Evo A, 50 generations, seed 417101) with one screener:
  - `SCR`: the stock `H0.1-A` fold-0 pair;
  - `RND`: seeded standard-normal scores, so the pick is driven by the residual corrector and the pick heuristics;
  - `ORC`: every one of the 48 proposals of a generation is evaluated and its true SEARCH utility is the score.
    Only the 6 picked programs enter the population, as in every arm.
- **Metric:** AUDIT u of each arm's stock `CV3@12` output.
- **Rule (fixed here):** the surrogate wave is built in this cycle only if `ORC − SCR` has a clipped family mean
  ≥ +0.02 **and** `ORC` beats `SCR` in at least 60% of the 24 contexts (ties count half). Otherwise it is dropped for
  this cycle. `RND − SCR` is reported: it measures what the neural screener adds over the corrector.

## 7. Integrity and harness gates
- **Identity check (`ID2B`, before this PREREG; §8).** The APEX runtime replays the stored H416 confirm deploy of the
  first 2 P201b contexts from a copy of its memo. Required: `SXB`, `KFB` and `GS.stock` equal to the stored records
  (|Δu| ≤ 1e-9), identical SXB members, 0 physical worker runs.
- **Gates on the `BASE` run** (a failure writes the checks alone and no value is read):
  1. the three arms each hold exactly one stage record per context; every deploy unit completed;
  2. 0 same-family bank hits; AUDIT ledgers consistent; 0 read-guard refusals;
  3. `KCAR100.bag` recomputed from each context's matrices equals the unit's own KFB value (|Δ| ≤ 1e-9);
  4. DOPE `c51a56d0` unchanged before and after the view encoding.
- **Sandboxes.** GP and deploy run in the H335 docker sandbox on xbabe0; the view encoding runs in the H335 bwrap
  sandbox on xbabe2. Readouts run from the pulled, sha-compared copy.
- **Resources.** xbabe0 ≤ 108 cores, nice 19, under the engine's host-CPU ≤ 85% guard; it yields to the `cl_fwriter`
  SXB dev91 run on request. Compute deadline 2026-11-12T06:00Z.

## 8. What existed before this PREREG
All of it is data preparation or harness checking with a dev stage; no APEX outcome was read.
- the claim receipt and the cohort manifest (§2);
- `SMK2`, the 2 smallest core `FIT-SEL` tables: their tables and view sets, the three GP arms and one deploy, run to
  prove the harness end to end. Only run status was displayed; their values are not read before the `BASE` readout,
  where they are 2 of the 221 contexts and are recomputed from scratch by the published stage;
- the `ID2B` identity check of §7 (TRAIN contexts; stored H416 values only). It passed: |Δu| = 0 for `SXB`, `KFB`
  and `GS.stock` on both contexts, identical SXB members, 0 physical worker runs;
- a one-context run of the gate G0 code (status only).

## 9. Files
| File | Role |
| --- | --- |
| `cohorts_h417.sh` | builds the cohort manifest (create-once) |
| `apex_core_meta.py` | manifest, cohorts, tiers, panels (stdlib only) |
| `apex_core.py` | APEX contexts through the frozen stack: table path, view-sha lookup, guard self-check, runtime |
| `gp_apex.py` | view encoding; the GP arms |
| `train_apex.py` | deploy; identity check; readout; headroom |
| `g0_apex.py` | the screener gate G0 (§6.2) |
| `stage_apex.sh`, `run_apex.sh`, `run_apex_xbabe0.sh` | staging and orchestration |
