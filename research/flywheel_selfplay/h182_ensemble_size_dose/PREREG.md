# H182 — fresh-adapter ensemble-size dose

Owner: `luna_ns_hbaseline`; assignment `~/nudges/hwave_t0605_h182_hbaseline.md`. Diagnostic research only. This experiment cannot alter the live flywheel, adopt an arm, promote a model, or authorize serving an ensemble.

## Question and design

Does increasing the number of independently initialized fresh adapters from 5 to 10 to 20 improve held-out family utility and reduce member-selection variance, while preserving the strict `ENS20` confirmation rule?

Reuse the H179 runner, data closure, frozen features, H156 candidate recipe, H176 fresh-adapter loss/optimizer, and score aggregation. Use only the pinned real-S3 TRAIN cohort: 1,219 contexts / 205 families. `DL-Val` is used only for the exact 91-context `r882` baseline and may be shown descriptively. Do not open sealed, holdout, or meta-test inputs. The live flywheel remains read-only.

Use five family groups from `family_folds(..., seed=182140)`, each 41 families. For outer fold `j`, A is group `j`, B is `(j+1) mod 5`, and FIT is the other 123 families. B is unused. Across five folds each of the 205 TRAIN families is assessed once for each outer seed. The fold seed is distinct from H179 (`182140`, not `179140`).

The arms are `R882` (zero-update reference), `ENS5`, `ENS10`, and `ENS20`. Each outer seed/fold trains 20 independent fresh members from distinct constructor-produced initial adapter states, with independent batch streams and exactly 40 H179-recipe updates per member. `ENS5` uses members 1–5, `ENS10` members 1–10, and `ENS20` members 1–20; these prefixes are strictly nested. Each member trains on the same FIT families and changes only the private 29,211-parameter adapter copy. Frozen DOPE and all non-adapter state are checked throughout. Candidate scores are averaged before `argmin`; neither votes nor model weights are averaged.

Seeds are `182101`, `182102`, `182103`. Member seeds are `182201–182220`, `182301–182320`, and `182401–182420`, respectively. The full five-fold design has 300 fresh-member trajectories and 12,000 optimizer updates. No `CONT` trajectory is included. Holm correction is over the three prespecified fresh-arm comparisons (`ENS5`, `ENS10`, `ENS20`), m=3.

Persist per-member raw candidate scores and selected choices for every A context, with context/family IDs, seed/member mappings, candidate counts, and exact score dtype, to the attempt output. These member-level A scores are custody-only and are not committed to Git. Persisted data must be sufficient to independently reconstruct each member choice and every K-member ensemble choice.

## Timing-only plan and guards

First verify the exact native `r882` 91-context DL-Val baseline bit-for-bit, mean `1.1377941898164525`, before opening TRAIN outcome pools. Then run the first four updates of outer fold 0 / seed 182101 / member 1 solely as the timing pilot. Use the median elapsed time from pilot updates 2–4 to project 5 folds × 3 seeds × 20 members × 40 updates, plus 900 seconds reserve, against a 14,400-second cap. If that estimate exceeds four hours, lock outer groups 0, 1, and 2 only and disclose the 123 assessed-family coverage. Keep all three seeds and all 40 updates/member. If the 3-fold projection still exceeds four hours, stop INCOMPLETE before A outcomes; do not cut the member count or update budget. Fsync the final fold/update plan before scoring any A outcome. The pilot's losses, gradients, predictions, and utilities cannot influence the plan.

Run only on `xbabe2`, CPU-only, `nice -n 19`, at most four aggregate process-tree threads, RSS strictly below 6 GB. Stop on host CPU >85%, MemAvailable <16 GiB while swap is exhausted, disk use >=93.8%, free disk <40 GB, or four elapsed hours. Verify frozen DOPE composite SHA-256 `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` before and after. No GPU, `xbabe0`, `xbabe3`, or live-flywheel writes. No AUC or speed metrics.

## Outcomes and verdict

Report per arm median equal-family utility over seeds; paired-family bootstrap gain versus R882 and percentile 95% interval from 10,000 shared family resamples using axis 0 across seeds; one-sided centered-bootstrap p and Holm m=3 adjusted p; Elo versus R882 with CI; each seed gain; exact `-2.0` counts; per-K pairwise member-choice disagreement; and the gain-vs-K curve. If the 3-fold timing branch is used, state that only 123 families were assessed.

`ADVANCE_CANDIDATE` requires every condition: `ENS20` median OOF gain strictly greater than `+0.001`, paired-family bootstrap LCB95 strictly greater than zero, all three seed gains strictly positive, and Holm-adjusted `ENS20` p strictly below `0.05`. Ties reject. Otherwise verdict is `NO_ADVANCE`. Label a null underpowered versus the H175 MDE80 planning heuristic when the absolute `ENS20` median gain is below `0.042341`; state that this is not measured H182 power. An `ADVANCE_CANDIDATE` is not adopted by this study: report it in `AGENT_CHAT.md` for the flywheel owner. H177/H178 calibration still applies before any promotion, and serving an ensemble requires a separate owner ruling.

## Provenance and lifecycle

Reuse H179's immutable model/source/frozen-feature/TRAIN-pool pins and verify the real-S3 TRAIN catalog closure before execution. Regenerate and pin the fold manifest for seed 182140. Data-free tests cover nested member prefixes, all 60 distinct seed IDs and 20 distinct synthetic initialization hashes per outer seed, refusal of duplicate hashes, FIT/A/B isolation, score-averaging-before-argmin, `axis=0` 10,000-draw bootstrap, Holm/verdict boundaries, and 4-hour fold projection. The runner must construct all 20 actual member initial states and reject duplicate hashes before any member trains.

The preregistration PR includes this document, runner/analysis, manifest/provenance/fold pins, and tests. Obtain an exact-head `rd_argon` review (15-minute limit), run targeted local checks including all five `jope-records` verifiers, then squash-merge. Bind exactly one guarded CPU attempt to the merged source and review. Close out separately with README, `metrics.json` <100 KB, one canonical overnight-index row, and versioned S3 custody for raw panels, per-member A scores, and receipts. Keep large outputs out of Git. After closeout merge and custody verification, remove the owned lane using the repository preservation procedure.
