# H300 draft — ENS_VOTE2 confirmation across training seeds

**Frozen draft only. H299 returned NO_COHORT: v6A and v6B are empty. H300 is NOT RUN and cannot launch on this manifest.** A later owner-authorized wave must supply a new qualifying, disjoint cohort and publish its immutable manifest before building or scoring it. No previously used, skipped, or unbuildable family may be recycled to meet the floor.

The H299 authority is `~/nudges/hw_t0400_astra_h299.md`. This draft specifies the requested training-seed confirmation; it does not authorize compute, extend the H299 deadline, or alter the H294 historical verdict. H294's three bootstrap seeds describe uncertainty for one deterministic policy. H300's three replicates are separately fitted ensembles.

## Inputs and reservation

Use [cohort_v6_manifest.json](cohort_v6_manifest.json), with its catalog VersionId/ETag, dataset/context identities, exclusion-set hashes, and immutable assignment. H299's split rule was declared before scouting: SHA256 of UTF-8 `H299-v6-split-v1`, one NUL byte, then UTF-8 family key; last-byte parity 0 assigns v6A, 1 assigns v6B. Never rebalance after observing outcomes. H299 requires at least 120 qualifying families in total. A future manifest must also contain a nonempty v6A and v6B; report their actual sizes rather than inventing a confirmation power guarantee from the total floor.

Only v6A may be built/scored for H300. v6B stays untouched beyond H299's metadata qualification: no table, pool, feature, prediction, label, or outcome reads. No cross-seed or per-family dropping after scores. Any failure to build the frozen v6A panel is an incomplete run, not a silent smaller cohort. No replacement from v6B. Any revised cohort needs a new preregistration before outcomes.

Exclude the flywheel 1,219 TRAIN / 91 VAL / 93 HOLDOUT context families, every H283/H285/H286/H288/H292/H294 scanned family including skipped/unbuildable cases, all cohort_v1–v5 families, H287/H291/H295 fitting families, P3 heldout/dev/sealed_excluded closures, catalog sealed-matching IDs, all BeyondArena/sealed families, and DL-Test/holdout. Complete forbidden-roster certification is required before a positive future cohort admission; H299's empty-set exhaustion proof does not supply an unavailable authoritative sealed roster. Never read sealed-bucket objects or holdout data.

## TRAIN fitting: three independently refitted ensembles

The root training seeds are **30001, 30002, 30003**. Each root seed produces one complete ENS_VOTE2 policy containing three independently refitted BAG_SWITCH members. There are nine fitted members in total, not one three-member policy relabelled as three replicates.

Member seed is `core.stream_seed(root_seed, "H300_ENS_VOTE2_MEMBER", member_index)` for member index 0, 1, 2, using H287's pinned `h265_core.py`: SHA256 of colon-joined string arguments, first eight digest bytes interpreted little-endian, masked with `0x7fffffff`.

| Root training seed | Member 0 | Member 1 | Member 2 |
|---|---:|---:|---:|
| 30001 | 685781424 | 1430007429 | 1330439319 |
| 30002 | 1869595753 | 786733481 | 1272701215 |
| 30003 | 1510579523 | 89540748 | 1415857569 |

Follow the fitted-component procedure underlying H287 export `213b30ad52e1ce48927bba960ed609bd98ab4dfe`, with source pins in [source_pins.json](source_pins.json): `ensemble.family_folds`, `veto.run_cell_veto`, `bagged.run_cell_bagged`, and `ensemble.vote_consensus_pick`. The H287 export function copies the original H284 member models; copying those same models would not implement H300 training-seed variation. Refit the components here.

- Use the original H287 TRAIN-only top-eight packet/features and unclipped utility/relevance targets: 1,219 contexts in 205 coarse families, 568 feature columns. The `selected.json` SHA is `981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1`. H287's historical TRAIN proposal features use r882; preserve that fitting input and report it. The fresh comparison uses r3175 slot 0. Do not silently rebuild the fitting dataset with a different checkpoint.
- For each member seed, derive four family-grouped inner folds with `ensemble.family_folds`; use that seed for all inherited ranker, veto, and bag streams. Select LR from 0.03/0.05/0.10, trees from 50/100/200, and margin from the unchanged H287 grid. Preserve tie-breaking, strict TRAIN fit admission, and the infinite-margin fallback. Select exclusively within TRAIN; no v6A/B or reused VAL input to selection.
- Refit each full-TRAIN proposer ranker and invalid classifier, plus eight family-bootstrap rankers. Preserve the H287 invalid threshold 0.10, fixed bag vote requirement 6 of 8, model hyperparameters, and family bootstrap logic. The fold-0 plumbing check may use the first context of TRAIN fold 1 as H287 does; it is not an assessment fold and contributes no selection evidence.
- ENS_VOTE2 switches from slot 0 only when at least two of its three member BAG_SWITCH policies agree on the same nonzero slot. Otherwise it keeps slot 0. No CLIP, HARMVETO, VALGUARD, threshold changes, or best-seed selection in the primary.
- Freeze all nine members, selected configurations, training identity, seed streams, source and model SHAs before any v6A outcomes. Check serialized predictions on the same fixed 64 TRAIN contexts, with no fresh input. Record parameter/model and prediction hashes for each root seed. Coincident predictions are reported honestly; three fits are not three independent cohorts.

No neural optimizer updates. DOPE composite `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` stays frozen. Training inputs contain no outer-test rows or sealed-family data.

## Qualification and terminal scoring

The future H300 launch requires its own approved resource envelope and GPU `flock` on `~/qualification.lock`; this draft grants neither. Use copies, never the live writer directory, PID, or screen. Before new fresh scores, reproduce the published H287 serialization/scorer regression on TRAIN, and in the GPU scoring process replay r3175 on dev91 at exactly **1.1389572403748425**, with all 91 choices and evaluation identity `e79a7715c1531ccb8c4055ab96971101cd4034ec3a71d83bca6d25b0e604ca83`. Checkpoint SHA: `1d6ea900ebed8d2a05c3b3f37576adc15aadd2b53c585bfd5cb71a103e9ff97b`.

Build the frozen v6A contexts under unchanged H286/H294 pool rules, use the pinned H294 GPU scorer and feature construction, and compare the three frozen ENS_VOTE2 policies against r3175 slot 0 on the same panel. The baseline slot is never vetoed. No intermediate model/threshold selection or post-outcome retuning. Failure of a mandatory regression stops the launch before fresh scoring.

## Estimand, uncertainty, and decision

Let `d[s,f]` be a seed's mean utility difference versus slot 0 within family f. Utility is negative passing fitness and −2 for failure, as in the pinned H294 evaluator. Each family has equal weight. `gain[s] = mean_f d[s,f]`; primary statistic is `median_s gain[s]`.

Use 10,000 whole-family bootstrap draws with fixed RNG seed **300150**, shared across all three fitted policies. For each draw, sample the same family indices with replacement for every seed, recompute each seed's mean gain, then take their median. Report its percentile 95% interval (2.5%/97.5%). This is the pooled family-bootstrap interval; do not flatten the three copies into 3N independent families or replace training-seed variation with bootstrap-seed variation. Per-seed one-sided 95% LCB is the 5% quantile, matching H294; report two-sided intervals alongside it.

**CONFIRMED_FRESH_TRAINING_SEEDS iff all hold:**

1. Median gain across training seeds is strictly **> +0.001**.
2. Family-bootstrap one-sided LCB95 is strictly **> 0 on 3/3 training seeds**.
3. On every training seed, harmed-family share (family loss > 0.01) is **≤ 35%**, preserving H294's numerical boundary implementation.
4. On every training seed, invalid-pick excess over slot 0 is **≤ max(2, ceil(0.01 × N_families))**, and the exact one-sided McNemar test on discordant invalid picks gives **p ≥ 0.05**, as in H294. Report both discordant counts and the strict zero-excess diagnostic.

Otherwise **NOT_CONFIRMED_FRESH_TRAINING_SEEDS**. Missing results remain incomplete, not failures filled with zero. Family ties (`|d| ≤ 1e−4`) never count as wins; ties never satisfy a strict improvement clause. The requested +0.001 threshold supersedes H294's historical +0.010 threshold for this draft only. H294's sign-test and null diagnostics may be reported descriptively; they do not silently change the four owner-specified H300 conditions.

Report per-seed and median gain, pooled and per-seed intervals, family W/T/L, changed picks, harm/invalid counts, and **MM Bradley–Terry Elo anchored to r3175=1500**, with family-bootstrap intervals using the same shared family draws and H294's tie/pseudocount convention. Report family-median gain per seed and its median as a secondary; it cannot rescue a failed primary. Also report whether the three fitted policies make identical picks. Never report AUC or a speed verdict.

Even a passing H300 result is research confirmation requiring a separate adoption decision. This draft performs no serving change. **Current disposition: BLOCKED_NO_COHORT; H300 not launched.**
