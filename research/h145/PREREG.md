# H145 — larger family-grouped TRAIN assessment and frozen-adapter re-ranking

Owner: `luna_v1_res08`. Authority: `~/nudges/h145_t1940.md`. CPU only; zero optimizer
updates. This document must be committed, reviewed and merged before construction
of the H145 folds, decoding outcome arrays, or computing a H145 ranking. Metadata
inventory and the immutable latest-checkpoint snapshot precede registration.

## Question and scope

Can a larger family-balanced development panel give more precise and stable
adapter comparisons than a 91-context-sized panel, and what should a prospective
promotion-gate trial measure?

Target: **1,219 DL-Train regression contexts in 205 families**, approximately 13.4
times the current 91-context panel. Use only the exact TRAIN roster below and the
approved real-S3 dataset catalog. Do not add synthetic datasets, sealed families,
DL-Holdout, outer-test rows or new DL-Val payloads. Report exclusions and stop as
INCOMPLETE if this complete panel cannot be validated; do not silently shrink it.

All saved adapters were trained on DL-Train. The cross-fit here is **adapter
selection on four family folds, assessment on the fifth**. No family occurs in
both sets within a split. This prevents selection-stage leakage; it cannot undo
historical model-training exposure. The result is conditional development
evidence, not independently held-out model validation. Fold seeds are not model
replicates. No adapter is retrained or calibrated in this study.

## Frozen identities and inputs

TRAIN roster: the `train` member of the existing H50 `selected.json`, file SHA-256
`981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1`.
Only TRAIN identity metadata is decoded. Every context must have a resolved family,
task regression, source split train/inner, and a dataset ID in
`~/jope_plan/master/source/S3_DATASET_CATALOG.json`. Freeze that catalog's hash,
the ordered context/family roster and its fold membership before reading scores.

Fixed arms, chosen by round rather than by H145 outcomes:

| Arm | Immutable checkpoint SHA-256 |
| --- | --- |
| R782 | `427b0d4608e94f60f5f97d79b5c1c4ce9a7efd59d17e7853f714e901f3a4af0c` |
| R785 | `ef6696f6369122a162d58b9b522fd83cae110a5576ef3760e3b43ba85d129e0a` |
| R859 | `5802245f447b1852610576cde08b8a299a9e1e99eb82991aac63c958284fa715` |
| R882, reference | `0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64` |
| LATEST = R2555 | `f566a2f08ef215f5afc328c0d94374498a06d4f9a8c35b07ae5f4d4ebd6249a6` |

LATEST is the largest retained round number observed in the pre-outcome metadata
inventory at registration, not the later moving latest. An immutable copy is at
`xbabe1:~/jope-data/h145/inputs/LATEST-R2555.pt`; its source was read-only. Never
replace an unavailable arm or advance LATEST after seeing outcomes.

DOPE remains `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`.
Before ranking, verify checkpoint bytes, the DOPE identity, and byte equality of
every non-adapter tensor to R882, including tensor names, shapes and dtypes. Record
the trainable-adapter parameter digests. Exact adapter duplicates remain visible
as aliases, but are counted once in ranking, selection and Elo fitting; the alias
inherits its representative's score, interval and rank probabilities. Prefer R882
as representative for a duplicate of the reference.

### Reuse of authenticated frozen outputs

Existing completed CPU forwards may be reused after identity checks. This study
does not require a new evaluation of an unchanged fixed function. Source custody:

- Bucket `veox-jopedime-learning-use2`, prefix
  `experiments/rd_breakthrough_20261030/H139/opus-h139-20261101a/`:
  `per_context.json` SHA-256
  `86c6a5623ed5aa1636be0cc8f69f5befb29f1b6262b5e3dfeb5cf5f5a2b9c194`.
  Decode only `train_cids` and `native_train`; skip `native_val`, `val_cids` and
  other fields lexically, without decoding their values.
- Same bucket, prefix
  `experiments/rd_breakthrough_20261030/H143/opus-h143-20261101a/`:
  `per_incumbent.json` SHA-256
  `c444a67d6fcd8a06d5f02f46be988cf84a7b847586a1cd6dd8a8698b19c8eec1`.
  Use R785's TRAIN vector only, bound to the original source's ordered roster.
- Retain version IDs, full manifests and per-arm checkpoint-load/DOPE receipts.
  Verify shared R882 TRAIN values agree exactly across both sources before joining.
  Missing/nonfinite values, a different order, protocol, checkpoint or DOPE fail
  closed; neither fill zeros nor label mismatched caches as fresh forwards.

H139 source `78f9d9c1de25bbce49978edff8b2a97eade786ef` and H143 source
`77446bbcef958bacc6c907d0265f48c31b2a8dd9` use the pinned H124 forward module
SHA-256 `9ebce9652a7caadc7e50896a9c1b667de5aea53f1e3cf4f8797f815d8a16285a`.
For an exact duplicate latest adapter, reuse the representative's verified outputs.
If it differs, run that arm once with this same frozen CPU forward, on TRAIN only,
with the pinned context/feature/table/pool identities and DOPE checks before/after.
If this cannot be done within the budget, report the arm and study INCOMPLETE.

## Set construction and selection cross-fit

1. Sort unique family IDs; permute with NumPy `default_rng(145191)` and split into
   five folds with `array_split`. All contexts in a family follow that family.
2. Persist all five fit/evaluation rosters and their hashes. Check family and
   context disjointness in every split, no duplicates, full coverage exactly once
   as an assessment context, and no sealed/holdout roster entries.
3. Within each split, rank fixed adapters by the equal-family mean utility on the
   four fit folds. Choose the highest, breaking exact ties by earliest round.
   Evaluate that chosen adapter minus R882 only on the held-out fold.
4. Pool those out-of-fold observations with equal family weighting. Also report
   the frozen per-adapter full-panel ranking. Cross-fitting does not produce five
   independent estimates of a fixed adapter's utility.

Endpoint: `fixed-dev-cost-argmin-negative-fitness/v1`, higher is better. Pick the
first argmin model cost over the complete original pool; score minus fitness if
its gate passes and **-2 otherwise**. Do not filter proposals by unrevealed fitness
or gate before selection. This is the historical flywheel endpoint, not U4 or AUC.
Report invalid-pick counts separately; exclude no cliff events from the primary.

## Noise, uncertainty, ranking and Elo

- Average contexts within family, then weight all 205 families equally. Report
  each adapter's mean native utility and paired gain against R882.
- Use **10,000 paired family-cluster resamples**, RNG 145000, keeping all adapters
  together. Bootstrap SE is the sample standard deviation of bootstrap means
  (`ddof=1`); the 95% percentile CI uses the 2.5th and 97.5th percentiles with
  linear interpolation. Report SE and interval for utility and paired gain.
- Rank stability: each unique function's bootstrap probability of first place
  (split probability equally among exact ties), rank distribution with average
  tied ranks, and pairwise probability of positive gain. Do not treat the samples
  as independent datasets or tune the threshold after observing them.
- Noise-size control: choose exactly 91 TRAIN contexts before outcomes by sorting
  SHA-256 of `145091:<context_sha256>` and taking the first 91. Preserve its family
  membership and repeat the family bootstrap. This is a **TRAIN small-panel
  control**, not the historical 91-context DL-Val cohort. Report its family count
  and full/small SE ratios; ratios with zero denominator are null, not infinity
  or evidence of improvement. This control does not enter cross-fit selection.
- Elo: canonical family-grouped Bradley–Terry comparisons, R882 fixed at **1500**,
  tie margin `1e-4`, prior `0.01`, 400 MM iterations. Use complete paired panels,
  with equal total weight per family. Report 95% CIs from **1,000 family
  resamples**, RNG 145001. Repeated bootstrap families retain their multiplicity;
  they are not collapsed by the family normalizer. A vectorized equivalent must
  agree with the canonical tracker on the full panel and a repeated-family
  fixture before use. No AUC headline.
- Compare against the **published aggregate** 91-context ranking in H139/H143,
  preserving their original context-weighted estimand and provenance. Use average
  ranks for ties and report Spearman descriptively. No new DL-Val payload/feature
  or per-context outcome read is authorized. If LATEST is not a duplicate and no
  recorded matching aggregate exists, leave its old-panel comparison unknown.

## Registered decision: proposal only

Recommend a **larger-panel shadow trial** only if all are true:

1. Complete 1,219-context/205-family panel, all arms available, source/DOPE and
   fold-isolation checks pass.
2. Median full/small paired-gain SE ratio over distinct non-reference arms with
   nonzero small-panel SE is **at most 0.75**, with at least two such arms.
3. A unique-function top adapter has first-place bootstrap probability **at least
   0.80** and is selected on **at least four of five** fit folds.
4. Its pooled cross-fit selected-minus-R882 equal-family gain is **strictly above
   +0.001**, and its paired family-bootstrap 95% lower bound is **strictly positive**.

Otherwise recommend **keep as research assessment; no gate switch yet**. Exact
ties fail superiority/margin gates. Missing required coverage means INCOMPLETE,
not a negative measurement. Broad intervals or failure to reject do not prove
equivalence or a causal selection artefact.

Even a pass recommends a prospective shadow development-gate trial, **not a live
promotion change**. Independently held-out model validation requires future models
whose fitting excludes assessment families; this retrospective study cannot supply
that evidence. Preserve the existing live loop, Q1 boundary and guidance defaults.

## Execution, receipts and closeout

CPU only, nice at least 10 (prefer 19), at most four math threads; host CPU load
must remain below 85% of available cores. No GPU/qualification lock, keys/signing,
S3 writes, live-directory writes, dataset duplication, or new working tree.
Use existing admitted TRAIN data and bounded needed checkpoint/custody recovery.
One run; maximum wall **7,200 seconds** from first outcome access, hard stop
**2026-11-01T03:45:00Z**; at most **8 CPU-hours**, RSS **6,000,000,000 bytes** and
new scratch **500,000,000 bytes**. Hold these limits across retries/resume; a guard
abort is INCOMPLETE and does not reset the ledger. Begin with an exclusive start
marker that records source/prereg/input hashes and the deadline.

Run focused tests off-host on xbabe4 (or an explicitly permitted available host),
with no local Cargo build and no shared-target cleanup. Pure statistical work may
use a low-load CPU host with the existing data. Record actual host, versions,
threads, memory, wall/process CPU, checkpoints, input hashes and commands. Retain
the roster/folds, per-context/family arrays and bootstrap/rank receipts outside Git
under `~/jope-data/h145/` and `~/jope_plan/evidence/H145/`. Keep Git metrics below
100 KB. Closeout PR includes results, receipt, limitations and the doc-only proposal;
append H145 status to `~/jope_plan/AGENT_CHAT.md`. Do not publish a result from a
partial run or overwrite original evidence.
