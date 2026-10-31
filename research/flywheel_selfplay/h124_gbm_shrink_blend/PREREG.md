# H124 — fixed-lambda shrink of the GBM context prior toward r882

**Owner:** `luna_v1_hwv03`. CPU-only. This preregistration must be committed before H124 opens any DL-Train pool, checkpoint, or feature-cache payload for scoring. H119 motivated the question: its fresh-fold GBM context prior improved the primary U4 endpoint but harmed 40/205 families (19.5%), above the 10% cap. H119 values are motivation only; none of its folds, seeds, fit predictions, utilities, adapters, or outcomes enter H124.

## Question and fixed protocol

Does shrinking a fresh, fold-specific `GBM_PROG412_CTX` utility score halfway toward the matching fresh r882 utility score reduce harm while retaining a positive equal-family U4 gain?

Use only the pinned real-S3 H50 DL-Train packet (1,219 regression contexts in 205 families), the pinned H58 context-feature cache, and a byte-verified offline copy of `~/jope-data/h51/ckpt/round-882.pt`. No DL-Val, holdout, sealed, meta-test, outer-test, or live-loop data or payloads. DOPE stays frozen at SHA-256 `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`; no neural parameter is trained. r882 has previously seen these DL-Train families, so the new split is conditional development evidence, not unseen-family or serving qualification.

Fresh family-grouped fivefold split: sort the 205 family IDs; permute with NumPy `default_rng(12400)`; partition into five 41-family folds. Each context is evaluated exactly once out-of-fold. Use fresh model seeds **64, 65, 66**. For each seed, fit one outer HistGradientBoostingRegressor per held-out fold and ten fit-only inner-pair models for calibration (15 fits per seed, 45 total). The fixed H119/H85 tree recipe is `max_iter=300`, `learning_rate=0.08`, `max_leaf_nodes=63`, `l2_regularization=1.0`, with all other installed scikit-learn defaults recorded. Draw 192 unrevealed training programs/context without replacement using a new, fixed H124 sampling RNG `12400`; center targets using the full training-pool utility mean as H119 does. No parameter, feature, seed, fold, calibration, or endpoint selection follows outcomes.

For each outer fold, reproduce H119's positive-slope fit-only calibration: inner predictions on the four non-held folds, on unrevealed slots 4..P-1 and gate-passing targets only. Apply that calibration to the outer model's predictions. Recompute matching r882 costs with the H119 CPU reference recipe, pinned H40b fused adapter, H58 projected features, and four revealed anchors. Convert both cost vectors to higher-is-better utility scores by negating costs; do not fit a post-hoc score transformation. For candidate `i` and fixed `lambda=0.5`,

```text
blend_score[i] = r882_utility_score[i] + 0.5 * (GBM_prior_utility_score[i] - r882_utility_score[i])
```

The only primary comparator is this fixed lambda 0.5 blend versus matching r882. Evaluate Track-2 U4 exactly as H119: max utility among revealed slots 0..3 and the four highest-score unrevealed candidates (slots 4..P-1). Per context, average its context utility into its family; each of the 205 families has equal weight. Pair within seed and family. Admission requires all three: median of the three seed-level mean family deltas **strictly greater than zero**; one-sided family-cluster bootstrap LCB95 **strictly greater than +0.001**; and no more than 10% harmed families. Bootstrap 10,000 family-cluster resamples (RNG 124000), retaining all three paired seed values per sampled family. Harm is a family whose three-seed mean blend-minus-r882 delta is `< -0.01`; exact threshold ties fail.

Secondary, report-only: lambda 1.0 (pure GBM prior), computed from the same fold predictions, calibration, and r882 references. Report its paired family delta, harm fraction, and Elo descriptively to show whether shrinkage changes harm; it has no admission gate and cannot change the primary. For both arms, report per-seed and median Bradley–Terry Elo with 95% family-bootstrap intervals versus r882 anchored at 1500 (Elo bootstrap RNG 124001). Report utility deltas and Elo, never AUC. No outcome summary is emitted unless the complete 3-seed × 5-fold identity checks pass.

## Identities and resource limits

Provenance source read-only during preparation: local H119 source commit `db2862d677a6f402cfc4d075911d70d22c26d666`, runner SHA-256 `8ed175fa240c3769f1fcef81e23c6128736bf539f7e771358da84f4dc7278ab7`. Its code is a recipe reference only; H124 will implement its own split, seeds, samples, and readout. Pin H50 source manifest `99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4`, selected roster `981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1`, fused source `b9f1468a61aa4dc888c8f2931f9c759f7d4f057793a77ccb05eed6300fb4f9ab`; H58 cache manifest `66b1df1374242580d482b7d6f9b72b6fe8f65830f90f658ca662a698aa229897`, context features `75704e8279c19fee8a7843ccbea801e3cd4b67e0ee08d282b070ddae7ab82faf`, and pool priors `6db567ef2c10d2c67c375f7500305ec693a700c4e71eb23003b4cd312ee51cbe`; r882 checkpoint `0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`.

CPU only, `nice 15`, OMP/MKL/OpenBLAS at most 4 threads, load at most 85/102, `MemAvailable >= 16 GiB`, own worker RSS `< 6 GiB`, scratch `< 2 GB`, and no qualification lock or ticket. Recheck load, memory, RSS, and scratch before and during fitting; pause or abort on a limit breach without touching other jobs. Use a unique H124 scratch root. Preserve only the compact metrics/receipt outside scratch, verify it, then remove H124 scratch. No GPU, CUDA, live xbabe1 path, or write to S3.

## Closeout

README records hypothesis, setup, outcomes, metrics, verdict, and next step. Keep `metrics.json` below 100 KB; include identities, complete fit receipts, compact primary/secondary readouts, versions, resource/CPU evidence, and explicit zero forbidden reads/GPU. Add the H124 index row and test coverage/inventory registration. Open one ready PR; do not self-merge. If a required H119 code/feature/data prerequisite is missing or fails its pinned identity checks, stop before scoring and record a blocker receipt instead of substituting inputs.

## Amendment 1 — operational only (rd_opus, 2026-10-31, committed before any scoring)

**Owner transfer.** The hwave overseer reassigned H124 from `luna_v1_hwv03` to `rd_opus` at 15:05Z
(`~/nudges/rd_opus_t0840.md`, receipt `~/jope_plan/ns/receipts/H124_claim.txt`).

**The statistical protocol is unchanged:**
- fold seed 12400, sampling RNG 12400 and model seeds 64–66;
- the H119/H85 tree recipe;
- fit-only positive-slope calibration;
- fixed lambda 0.5, with lambda 1.0 report-only;
- the H119 Track-2 U4 endpoint and equal-family weighting;
- the admission gates, harm rule, bootstrap draws and RNGs (124000 and 124001);
- every input pin.

`protocol_sha256` is unchanged. No outcome from the stopped attempt was produced or used.

The amendment responds to the stopped attempt (7,052,092 KiB peak RSS over the 6 GiB gate) and to audit
`AUDIT_t0855.md` findings A1–A3.

1. **Train-row-only cache reads (A1).**
   - Both H58 caches are pinned, uncompressed (`ZIP_STORED`) archives holding 1,219 train and 91 DL-Val contexts.
   - The runner reads only the train contexts' rows, by byte range:
     - `tokens_10x32`, `mask_10`, `lm48_z` and `dope629` from `context_features.npz`;
     - `offsets` and `proj16` from `pool_priors.npz`.
   - DL-Val rows are never decoded or copied into the process.
   - The `cids` column is read in full as identity metadata. It is the same identity list `selected.json` already
     carries.
   - The receipt records rows and bytes read per member and proves zero overlap with DL-Val row ranges.
   - The proj16 spans must equal each train pool's length and be pairwise disjoint.
   - The pinned whole-file SHA-256 identity checks still stream every byte through the hash without decoding it.
   - The other members (`ctx_feat_584`, `base_prior_cost`, `sos_prior`, `scale`) are not read; the r882 pass already
     used zeros for them.
   - The input files and their pins are unchanged.
2. **Memory-bounded execution (A3).**
   - Pool program matrices are loaded lazily per context, with their pinned SHA-256 rechecked, instead of holding
     all 1,219 (2.3 GB) in memory.
   - The r882 pass streams one context at a time, frees its tensors and trims the heap every 50 contexts.
   - The r882 network is released before the GBM fits.
   - Each fit fills one float64 design matrix in place. scikit-learn converts float32 input to float64, so the
     values it receives are bit-identical to the original float32 concatenation (unit-tested).
   - The 6 GiB ceiling is enforced continuously:
     - a watchdog thread samples current and peak RSS every 0.25 s and stops the worker with a receipt on breach;
     - every context of the r882 pass and every fit checks peak RSS (`ru_maxrss`) and current RSS.
   - A synthetic probe at the largest fit shape (191,808 × 1,089) peaked at 3.86 GB RSS.
3. **Fallback ceiling.** If the 6 GiB run stops on RSS, one relaunch in a fresh run root may use a 10 GiB worker
   ceiling (`--rss-limit-gib 10`).
   - It requires MemAvailable ≥ 24 GiB at launch.
   - The existing MemAvailable ≥ 16 GiB check still runs before and between fits, and the run aborts cleanly below
     it.
   - Any relaunch is disclosed in the closeout.
4. **Registered Elo estimator (A2).**
   - The headline Elo is the preregistered **median of the three per-seed family-grouped Elo ratings** versus
     r882 = 1500.
   - Its 95% interval comes from 1,000 family resamples (RNG 124001) drawn jointly across seeds, so family pairing
     is kept.
   - Per-seed intervals are reported too.
   - The closed-form per-seed rating is checked against the canonical helper.
   - The earlier pooled-score Elo (context U4 averaged across seeds first) is kept, labelled as an additional
     estimate.

Admission still requires all three preregistered gates; Elo is report-only. Resource limits are unchanged: CPU only,
nice 15, at most 4 threads, load ≤ 85/102, scratch < 2 GB, no GPU and no S3 write.
