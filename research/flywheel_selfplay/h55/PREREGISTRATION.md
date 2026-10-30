# H55 preregistration — r882 adapter averaging

**Status: preregistered design; no H55 outcome cell has run.** This is the preregistration-only change. Runner implementation and GPU work begin only after this PR is merged. H53b closeout PR #648 is merged.

## Question and scope

Test whether averaging small, independent adapter updates from the r882 adapter can reduce update noise, and whether a 25-round exponential moving average (EMA) improves over its matched raw final iterate. This is a hypothesis, not an assumed gain. Only the 29,211-parameter adapter may change. Each candidate starts from an independent copy of the pinned r882 adapter; no adapter or optimizer state transfers from another H-wave. DOPE remains frozen at c51a56d0 and is checksummed before and after every cell. H55 uses the fixed offline DL-Train cohort only; no live loop is read or modified.

The source/checkpoint identity is pinned to the H53b preregistered input set: source-manifest SHA-256 `99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4`, manifest identity `3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3`, and r882 comparator/checkpoint SHA-256 `0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`. Frozen DOPE SHA-256 is `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`. Before any training, the implementation must verify these pins and record the exact source and fold-assignment hashes in receipts. If a pin does not match, stop before training and issue a documented preregistration amendment; do not silently substitute inputs.

## Leakage-safe split

Use five folds grouped by family, with deterministic assignment seed 4900, matching H57's split convention. Freeze and hash the assignment before training. Define fold set A as four folds and fold set B as the fifth fold. Candidate optimization may read only fold A. Fold B is untouched until final candidate scoring and is the sole H55 admission set. No fold-B value may influence optimization, weight averaging, EMA selection, stopping, or arm choice.

The r882 adapter copy is scored on fold B as the paired control using exactly the same contexts. Caveat: r882 itself saw fold B during the live loop, which biases this comparison toward the control and makes the test conservative. H55 does not claim an independent evaluation against a fold-unseen r882 model.

## Arms and fixed budgets

Run seeds 12, 13, and 14 for both arms. Seeds govern the deterministic sampler/update streams; the family-grouped split remains fixed across seeds.

### SWA-8

For each seed, create eight independent candidates from fresh r882 adapter copies. Each candidate performs one live-equivalent update unit (`k=2` AdamW optimizer steps, `lr=1e-4`, `weight_decay=1e-4`) using distinct sampler seeds and the pinned live sampler restricted to fold A. The model architecture, live loss, slate size 192, and sampler implementation remain pinned; no fold-B context is presented to an optimizer. Uniformly average the eight resulting adapter weight vectors. Score that single averaged adapter on fold B for admission. Also score each of the eight members on fold B once and report their mean score as a descriptive matched single-candidate control; this mean is not an additional candidate arm and cannot be admitted.

### EMA-25

For each seed, start one fresh r882 adapter copy and continue one trajectory for exactly 25 live-equivalent rounds on fold A (50 optimizer steps at the same pinned `k`, `lr`, and weight decay). Maintain an adapter-only EMA with decay 0.9. Score the final EMA on fold B for admission and the raw round-25 final iterate on fold B as a descriptive matched control. Do not select an intermediate or best round. No early stopping on DL-Val or fold B is allowed.

The budgets are round/update-capped, not efficacy or validation-wall-time capped. The fixed planned totals are three SWA-8 seed cells (24 independent candidate update units) and three EMA-25 seed cells (75 rounds). A separate timing-only EMA-25 seed-12 cell is run before outcome cells, excluded from outcome data, and used only to record throughput/resource use and preregister the execution cap in a dated pre-training amendment. It cannot supply weights, optimizer state, scores, or a control to the outcome cells. Outcome cells start fresh after that amendment.

## Admission and reporting

For each arm independently, pool the three seeds' paired fold-B family gains over r882. Admission requires both:

1. A one-sided family-cluster bootstrap mean-gain LCB95 over 10,000 resamples strictly greater than `0.001`.
2. No harm: the median of per-family seed-medians is at least zero, and no more than 10% of families have a seed-median gain below `-0.01`.

Resample family IDs as clusters, keeping all seed observations for a family together. Ties fail; a tie is not an improvement, and equality does not satisfy the strict LCB threshold. Apply the admission rule separately to SWA-8 and EMA-25; do not choose a winner by comparing fold-B scores.

Report per arm and seed: final candidate definition, update/round count, fold-B family mean gain and LCB95, median gain, family no-harm fraction, W/T/L, worst five family gains, and diagnostic Elo versus r882 (Elo is not AUC). Include pooled family median gain and two-sided 95% family-cluster bootstrap CI, the r882 fold-B control, SWA member-mean and EMA raw-final matched controls, and DOPE/source/checkpoint/fold hashes. Clearly separate the per-arm admission result from the descriptive matched controls.

Only an arm that passes fold-B admission may receive one terminal DL-Val read per seed, after its admission decision is frozen. Compare that read with the r882 **harness** family utility `1.1536152769912449`; do not use the live-loop target `1.1377942`, which is cross-harness and not comparable. DL-Val is diagnostic only and cannot change the admission, select a candidate, or trigger another read. If neither arm is admitted, perform no terminal DL-Val reads.

No sealed/holdout payloads, outer rows, meta-test families, or live-loop state may be opened or delivered. Every receipt must report `holdout_payloads_opened=0` and `outer_rows_delivered=0`.

## Resource, integrity, and stop rules

After implementation and a two-context dry run that produces no outcome, acquire `~/jope_plan/qualification.lock` exclusively for the timing-only cell. Do not interrupt a lock holder. Require at least 6 GiB free on CUDA device 0 while holding the lock, record lock wait/free memory/device, and release the lock after the cell. Before outcome cells, the timing-only measurement must be incorporated into a dated amendment that fixes the execution cap. Keep planned outcome budgets unchanged; do not add wall time based on outcomes. If lock contention lasts more than three hours, issue an infra-void receipt and stop. CPU fallback training is forbidden.

Before any training cell, require host CPU utilization at or below 85%, `MemAvailable` of at least 20 GiB, and a safe resource state; record swap occupancy. The current owner clearance reports near-full swap, so do not start parallel CPU jobs. For every cell, verify the adapter-only trainable parameter inventory, r882 copy identity, fold-A-only sampler inputs, and DOPE checksum before and after. Any identity/checksum mismatch, OOM, or unplanned fold-B access terminates the cell as infra-void; do not repair or resume partial optimizer state.

## Closeout

Keep implementation, audit, and focused tests under `research/flywheel_selfplay/h55/`. Closeout requires a README, `metrics.json` under 50 KB, an index row, immutable per-cell receipts, and an independent audit. Report the no-admission result as such; do not present a diagnostic or matched control as a win. Run focused local checks on touched files, open one small closeout PR, and use the authorized squash-admin flow. Append results to `~/jope_plan/ns/H55_STATUS.md` and `~/jope_plan/comms.md`. H57 remains gated by its merged preregistration until H55 closes.
