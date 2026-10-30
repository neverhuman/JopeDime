# H57 — DEEP-ES direct-utility search

Status: preregistered design only. No H57 runner, optimizer generation, or GPU cell may run until H55 has closed. H55 itself is gated on the H53b closeout PR being merged. Preserve this preregistration and its commit identity before any H57 code or training is added.

## Question and scope

Test whether direct-utility antithetic evolution strategies (ES) can improve the 29,211-parameter adapter when initialized from a copy of the r882 adapter. DOPE remains frozen at `c51a56d0`; it must be checksummed before and after every ES generation and remain byte-identical. The candidate always starts from a fresh r882 adapter copy; do not transfer weights or optimizer state from another experiment. Search the adapter parameters only.

The prior H50 ES3 result is motivation, not evidence of admission: it produced small generation candidate-mean-delta medians (+0.00157/+0.00103, maximum +0.0084) over only two generations. This design uses eight generations by default and does not select a best generation or perturbation.

## Arms and fixed search protocol

Run three independent seeds: 15, 16, and 17. For each seed, run two independent sigma arms, `sigma=0.001` and `sigma=0.003`. Each arm starts with the same r882 adapter values and a seed-specific, recorded ES random stream. Use 16 antithetic pairs (population 32): evaluate `theta + sigma*epsilon` and `theta - sigma*epsilon` for each pair. Compute centered-rank fitness shaping over the 32 fitness values, form the antithetic ES gradient, and update the mean parameter vector with Adam at learning rate 0.01. Record the ES gradient norm and mean-theta update L2 norm for every generation. No candidate ranking or best-candidate replacement is allowed.

Fitness is direct DL-Train utility, aggregated by family over the four folds in fold set A. Fold assignment is family-grouped and pinned to assignment seed 4900; fold A contains four of five folds. Use the same family and context identities for every perturbation in a generation. The final candidate for each seed/sigma is the mean theta after the last completed generation. Do not score generations for selection.

Default horizon is eight completed generations per seed/sigma. After H55 closes, measure one preregistered timing-only generation on the lower-sigma arm using seed 15. It is excluded from outcome data. Use that measurement only to choose the common fixed horizon for every candidate arm: eight generations by default, or twelve only if a written pre-training resource amendment demonstrates that all six candidate arms plus the null arm fit within a 90-minute cumulative GPU-cell-wall cap, including terminal fold-B scoring. Do not change arm order or horizon based on outcome metrics. If setup/timing cannot support a horizon safely, stop before candidate training and issue a no-outcome resource receipt.

## Selection split, controls, and admission

Score each final mean theta exactly once on untouched inner fold set B, the fifth family-grouped fold from seed 4900. This is the only admission set. Pair every candidate context score with the r882 adapter copy scored on the same fold B contexts. Also score the matched r882 copy on fold B as a control. No fold-B value may enter ES fitness or updates.

Run a drift-only null arm with seed 15, sigma 0.001, eight generations, and the same population, initialization, optimizer, fold-A data, and evaluation budget, but permute/randomize the rank fitness assignments independently per generation before computing the ES update. The null is descriptive and cannot be selected or admitted. It measures movement caused by the optimizer and random rank signal.

For each sigma arm, combine the three seeds' fold-B paired family gains and compute a one-sided family-cluster bootstrap LCB95 with 10,000 resamples. Resample family IDs as clusters, retaining their seed observations together. Admission requires both (a) pooled paired family mean LCB95 strictly greater than 0.001 and (b) no-harm: the median family gain, after taking each family's median across seeds, is nonnegative and no more than 10% of families have median gain below -0.01. Ties fail. Report each seed separately and the pooled median with a two-sided 95% family-cluster bootstrap interval. Do not choose between sigma arms by the fold-B score; report both and apply the stated admission rule independently.

Only an admitted sigma arm may receive one terminal DL-Val read per final seed candidate, after the fold-B decision is frozen. No repeated DL-Val reads, sealed/meta-test families, holdout labels, or outer folds are permitted. DL-Val and Elo versus r882 are diagnostics, not selection criteria.

## Required reporting

Report per sigma and seed: completed generations, evaluated perturbations, per-generation timing, ES gradient/update L2, fold-B family mean gain and LCB95, median/no-harm fraction, terminal status, and source/checkpoint/DOPE hashes. Include the r882 control and drift-only null. Report Elo versus r882, W/T/L, and worst five family gains as diagnostics. Include pooled median and CI across the three seeds. State explicitly that H57 tests the final ES mean theta, not the best generation or sampled candidate.

## Resource and integrity controls

After H55 closes and before any H57 GPU request, record the measured one-generation time, projected total, and final fixed generation cap in a dated amendment. The total candidate plus null cell-wall budget is at most 90 minutes; qualification-lock waits do not authorize CPU fallback or running another GPU job. Each cell must acquire the exclusive `~/jope_plan/qualification.lock`, require at least 6 GiB free on cuda:0, release between cells, and record lock wait/free memory/device. If the GPU lock or memory guard cannot be satisfied within the separately approved wait bound, stop with an infra-void receipt. Do not interrupt another lock holder. Keep CPU utilization at or below 85% before launch and confirm adequate `MemAvailable`; do not start if system pressure makes the run unsafe.

Pin the exact r882 checkpoint, source manifest, fold assignment and all source hashes in the runner receipt before scoring. Verify the DOPE checksum before and after every generation and every terminal scoring pass. Any checksum change, data-identity mismatch, or OOM ends that cell cleanly and marks it infra-void; do not repair or resume partial optimizer state.

## Artifacts and gates

Implement the runner, independent audit, and focused tests under `research/flywheel_selfplay/h57/` only after H55 closes. Save compact immutable per-cell receipts, an audit, `README.md`, and `metrics.json` under 50 KB, plus one research index row. Run the targeted local checks specified by the eventual owner execution packet. Prepare one small PR, perform exact-head review and checks, then use the authorized squash-admin merge flow. Preserve evidence and remove only the owned worktree after the PR closes.
