# H264: zero-update kNN residual memory on frozen r882 embeddings, xbabe2 CPU

**Owner:** rd_opus [OPUS]. **Assignment:** `~/nudges/hw_t1658_opus_h264.md` (watcher / Grok Bot for Jepson,
2026-11-03 16:58Z, decided). **Claim:** `~/jope_plan/ns/receipts/H264_claim.txt`.

This is a diagnostic only. It cannot advance a model, a serving release or a live flywheel checkpoint. DOPE stays
`c51a56d0` (frozen fp32): it is checked at load and before and after the study. **No parameter is updated**: every
adapter tensor has `requires_grad` off, there is no optimizer, and the runner asserts that the adapter is bit-identical
at the end.

r882 (`0084479a…`) is a read-only COPY. The live flywheel is never read, written, paused or preempted.

**Publication.** Under the PR #2071 split (`~/nudges/CI_FREEZE.md`) this prereg is pushed to `origin/v1/next`
(non-force) before anything runs; nothing merges to main. The pre-publication smoke makes 0 updates, scores 0 TRAIN
contexts and runs only the Native91 reproduction. The launch is bound to the published commit, its verified ancestry in
`origin/v1/next` and the sha256 of a durable same-agent review receipt.

## Hypothesis

**What is known.** Every gradient continuation of r882 (H144–H263) was flat or negative, and H176 reported the
parametric adapter as DATA_SATURATED. No wave has tried non-parametric memory.

**H264 asks:** does a kNN-LM-style residual memory over DL-Train FIT (context, candidate) pairs, keyed by frozen r882
embeddings, lift DL-Val91 single-pick utility over r882 by more than +0.002, with no test-time reveals and no weight
change? The result would be a fixed checkpoint artifact (r882 plus a memory bank).

## The policy (constants frozen here)

- **Score.** score(c, p) = r882(c, p) + λ · kNN_residual(c, p). The policy picks argmin score (first index on ties) and
  earns utility −fitness if gate else −2.0.
- **Keys.** The L2-normalised concat of the frozen r882 adapter context embedding of c (`ctx_pooled`, 32-d) and the
  candidate query embedding of p (`cand_q`, 32-d). Both come from `encode_multimodal_context_and_candidates` on the
  context's full candidate pool, as in the Native91 forward, in float32 with no gradients. The embedding pass must
  reproduce the Native91 costs bit for bit.
- **Values (FIT memory only).** The true cost minus r882's predicted cost. The true cost is the H146/H170 CTRL target:
  clip(fitness, −4, 2) if gate else 2.0. Gate-failed (−2.0-utility) candidates participate, but their residual is
  clipped to [−1, 1], the H161 Huber scale.
- **Neighbours.** The exact k nearest memory pairs by Euclidean distance between normalised keys, **excluding every pair
  from the query's own family**. The residual is their softmax(−d/τ)-weighted mean.
- **Shortlist (compute bound, frozen before any outcome).** Both the memory bank and the scored queries are restricted
  to each context's **64 lowest-r882-cost candidates**; the other candidates keep their r882 cost. Pools hold 1,151–1,152
  candidates, so a full-pool bank would have about 840k pairs per fold. A measured exact search on xbabe2 cores 44–47
  took 44.8 s per 1,024 queries against 840k keys, about 8 hours per fold. The shortlist is outcome-free (r882 costs
  only), keeps the search exact within the bank, and covers the candidates most likely to become the pick.
- **Grid.** k ∈ {8, 32, 128}, τ ∈ {0.05, 0.2}, λ ∈ {0.25, 0.5, 1.0}, plus λ = 0 (= r882): 19 configs, with λ = 0 first.

## Arms

- **KNN (primary).** Per cell, the config chosen by the strict inner selector on the SELECT fold, with memory from that
  cell's FIT folds only.
- **KNN_SHUF (secondary, placebo).** The same selected config and neighbours, but with the FIT residual values permuted
  within each family (fixed stream per seed and outer fold).
- **CONTROL.** r882 unchanged.

## Design

- **Data.** Real S3 DL-Train only: 1,219 contexts in 205 families (`S3_DATASET_CATALOG.md`). has_meta_test rows are
  excluded, sealed, holdout and meta-test data are never touched, and **DL-Val91 is never in the memory bank**.
- **Folds.** One fresh fold seed, **264140**, gives a family partition (205 families → 5 × 41) shared by seeds
  264101–264103. Each cell has FIT (3 folds), SELECT (1) and ASSESS (1): 15 cells.
  - Because the fold seed is shared, the memory, the search and KNN's scores are identical across seeds. Seeds differ
    only in the selector's bootstrap stream and the KNN_SHUF permutation, which is disclosed in the metrics.
- **Selector (strict, inner).** On SELECT, the earliest config with the best family-mean single-pick utility (λ = 0 wins
  ties), accepted only if its gain over λ = 0 is > .001 **and** the family-bootstrap LCB95 is > 0; otherwise λ = 0.
- **Lock.** All 15 cells' selections are computed from SELECT alone and written to one fsync-locked `selection.json`
  **before** any ASSESS or DL-Val91 utility of a kNN policy is computed. Each policy is then scored once.
- **Native91 first.** Single-pick r882 = 1.1377941898164525 on all 91 contexts, bit-exact, before any TRAIN outcome
  opens.

## Statistics and decision

- **Primary panel.** DL-Val91 single-pick, each seed's equal mixture of its 5 cells' policies, family-equal.
- **Statistic.** The median over 3 seeds of the per-seed equal-family gain against r882. The CI95 is the
  family-bootstrap median (10,000 draws, boot seed 264150), with a one-sided p and **Holm over KNN and KNN_SHUF**
  (m = 2).
- **ADVANCE** (KNN only) requires all five of:
  - median family gain > **+0.002**;
  - family LCB95 > 0;
  - Holm p < .05;
  - **3/3 seeds positive**;
  - **KNN beats KNN_SHUF** (median KNN − KNN_SHUF gain > 0).

  Ties, or λ = 0 everywhere, mean NO_ADVANCE ("fallback identity, not equivalence"). KNN_SHUF cannot advance.
- **Also reported:**
  - the selected and best (k, τ, λ) per cell, with the selector gain and CI;
  - changed-choice counts against r882, and invalid (−2.0) pick counts;
  - harm families (loss > 1e-4) and context #27 (`fam-9f67f144…`), reported separately and never excluded;
  - the 1,219-context TRAIN out-of-fold panel (descriptive);
  - MM Bradley–Terry Elo with r882 = 1500;
  - the gap to the plateau best 1.1389572.
- **Not used.** No AUC, and no speed or tok/s verdicts.

## Resources and the A1 memory guard (H239 A1)

- xbabe2, CPU only: `taskset` 44–47, nice 19, one math thread, tree RSS < 4 GiB (float32 bank, chunked exact search,
  no faiss-gpu), ≤ 4 threads, disk free ≥ 300 GB.
- No GPU, xbabe0, xbabe1 writes, xbabe3 or live flywheel. No foreign process control. PR #2071, main and v1/fastlane
  are not touched.
- **Admission** at load5 ≤ 100 and MemAvailable ≥ 30 GiB. **Cooperative pause** at load5 > 108 or MemAvailable <
  20 GiB; xbabe2 swap is full, so the guard pauses rather than aborts. **Resume** at load5 ≤ 100 and MemAvailable ≥
  25 GiB, sustained for 30 s. **Hard stop** at MemAvailable < 12 GiB sustained for 30 s or < 6 GiB on any sample, plus
  the RSS, disk, thread and deadline stops. Only its own worker group is signalled.
- **Deadlines.** Prereg pushed to v1/next by **17:40Z**; compute hard stop **19:45Z**; closeout on v1/next by
  **20:30Z**. If the study is incomplete, it closes as CLOSED INCOMPLETE with honest receipts.

## Custody

- **Repo.** pytest, Ruff, JSON parse and `git diff --check` precede the push. The private, sha-verified `jope-records`
  gates are run and recorded. Under the freeze this wave does not edit CI config, records pins, schemas or the root
  README, so its coverage, test-inventory and source-inventory registrations are deferred. A durable same-agent review
  receipt names the exact head.
- **S3.** Full outputs go to create-only S3 `…/rd_breakthrough_20261030/H264/<run_id>/` with VersionId readback.
- **Closeout.** It publishes only the README, a compact metrics JSON and the index rows to v1/next. No `.pt`, `.npz` or
  memory bank goes in Git.
