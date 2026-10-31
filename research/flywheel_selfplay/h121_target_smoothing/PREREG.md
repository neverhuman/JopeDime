# H121 — target smoothing on a fresh family fold

Registered before outcomes. This is explicitly a fresh-fold replication of H90's epsilon 0.1 utility-mean smoothing plus a new epsilon 0.3 dose; H90 already tested epsilon 0.1 and did not admit it. H51 also explored softened-rank recipes. H121 does not claim smoothing is a new mechanism.

## Arms and training

CONTROL is the H82 recipe unchanged. TS-0.1 and TS-0.3 alter only the sampled targets after H82's existing target construction: `y'=(1-epsilon)*y+epsilon*mean_ctx(y)`, with epsilon 0.1 or 0.3; mean is over candidate utilities within each context. All start from independent fresh copies of `~/jope-data/h51/ckpt/round-882.pt`, pair seeds 45/46/47, and perform exactly 40 adapter updates. Only the 29,211 adapter parameters train; DOPE remains frozen at SHA256 `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`, verified before and after each future cell.

A deterministic five-way whole-family partition uses H50 real DL-Train rows only and seed 12100, not a reused historical fold. First four folds are training; fifth fold is scoring. No DL-Val or sealed/holdout/meta-test rows enter training or the admission fold. DL-Val's 91 contexts are descriptive only if explicitly authorized. Every physical retry uses a new `scratch/{arm}-seed{seed}-attemptN/`; no resume over prior attempt scratch.

## Outcomes and gates

Primary outcome is paired per-family candidate-minus-r882 delta on official argmin utility, using the official gate-failure penalty and candidate selection rules. Summarize the three paired-seed family estimates by their median. Admission requires median strictly greater than zero, one-sided family-cluster bootstrap LCB95 strictly greater than +0.001, and no more than 10% of scoring families harmed by delta < -0.01. Holm correction applies across the two smoothing-vs-r882 treatment comparisons; ties are not wins. Report family-grouped Bradley-Terry Elo and 95% CI against r882=1500; do not report AUC.

Co-readout following H114(a): continuous mean regret over all candidate predictions, reported beside official utility. For context c and all candidate indices i, `regret_ci = max_i(y_ci) - predicted_utility_ci`; average over all candidates within context, then equally across families. This is descriptive and cannot replace or alter the primary gate. H90 outcomes are historical context only, never combined with H121 observations.

## Preparation boundary

This change authorizes CPU protocol preparation and synthetic-only tests/dry run. It does not authorize GPU work, FIFO tickets/waiters, real checkpoint or S3 payload access, timing, fitting, scoring, or outcomes. Future execution remains separately gated after H99, H100, H105, and H109, with owner authorization and the queue pattern in the H121 nudge. No sealed/holdout/meta-test payloads and never the live xbabe1 loop.

## Executor and FIFO implementation

The executor adapts the reviewed H97 training cell path and H61 FIFO lease mechanism. It applies the registered smoothing transform to H82 batch targets and keeps H82's optimizer, candidate sampler, adapter-only trainable parameter inventory, frozen DOPE guard, 40 updates and official terminal argmin utility evaluation. It forms the fresh five-fold whole-family split from the H50 real DL-Train roster; the first four folds supply fit rows and only the fifth fold is loaded for primary scoring. H50 DL-Val payloads are not loaded by H121.

Each arm/seed/attempt has distinct H97-compatible scratch. Coordinator accounting is cumulative under a 2,700 second GPU cell-wall cap, reserves shutdown time, retries only a confirmed pre-scoring host-load abort at a new FIFO tail, and refuses unknown/in-progress budget resumes. It checks merged CPU source dependencies and completed H99/H100 closeout records before starting the FIFO lease. Manual invocation requires `--authorized-execution` plus a durable owner receipt stored under `~/jope_plan/ns/receipts/`, approved for H121 and bound to the exact merged source SHA; CPU dry-run is data-free and creates no run root, ticket or waiter. These implementation guards do not grant execution authorization.
