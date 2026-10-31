# H99 — ESH harm-lambda follow-up to H80

**Status: CPU preparation only.** This preregistration and its CPU-only contract are frozen before any H99 outcome, timing cell, GPU ticket, or data read. It follows the owner assignment in `~/nudges/h99_esh_harm_t0335.md`. The H80 implementation at `research/flywheel_selfplay/h80_long_es/` is the algorithm source: H99 reuses its centered-rank ES, antithetic population, gradient, adapter/checkpoint pins, and family-harm fitness calculation. H99 does not copy or alter H80 source.

## Question and fixed arms

Can raising the H80 family-harm penalty from lambda 1.0 to lambda 2.0 or 4.0 reduce the harmful-family tail enough to meet the admission gate without losing the utility improvement? H80's Fold B was already observed and is not eligible as H99 admission data. Its published results may be cited descriptively only.

The fixed arms are:

- `NULL`: permuted-fitness drift diagnostic, seed 36 only; never eligible for admission.
- `ESH-l2`: H80 ESH fitness with lambda **2.0**, seeds **36, 37, 38**.
- `ESH-l4`: H80 ESH fitness with lambda **4.0**, seeds **36, 37, 38**.

The six candidate cells are paired by seed across both ESH arms. The NULL diagnostic shares the seed-36 perturbation stream and is a single diagnostic cell, preserving H80's seven-cell outcome budget. ESH-hard is excluded before outcomes to keep the fixed budget at H80's seven outcome cells. No arm may be dropped or selected after results.

Each cell uses H80's fixed `sigma=0.001`, 16 generations, population 32 (16 antithetic pairs), Adam learning rate 0.01, and final mean-theta only. The only trainable parameters are the 29,211 `adapter_10tok` parameters. Every cell starts from its own byte-verified r882 copy in a unique `scratch/<cell>-attemptN/` directory. DOPE remains frozen at SHA `c51a56d0`; the r882 checkpoint SHA is `0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`.

## Split and leakage controls

Use only the approved real-S3 H50 `DL-Train` roster and its pinned source manifest (`99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4`). Assign whole training families to five folds with fresh seed **9900**, rather than H80's seed 4900. Fold A is the first four family folds and drives ES fitness; the fifth fold is the new admission Fold B and is scored once per final candidate. The split is formed only from the approved `DL-Train` rows; no sealed/meta-test family identifiers or payloads are opened or supplied to the runner. The pinned H50 manifest/roster boundary must continue to exclude non-training and `has_meta_test` rows. If that boundary or the fresh assignment cannot be verified, admission remains unknown and no scoring is authorized. H80 Fold B may appear only as a clearly labeled descriptive historical reference; it is not an input, tuning set, or admission set.

No DL-Val, outer-fold, sealed, holdout, meta-test, or live-loop payload may be opened. This CPU-preparation PR reads no packet, checkpoint, model output, or protected data.

## Outcomes and admission

For each candidate arm, compute paired family-utility gains against r882 on the new Fold B. Report family median gain and two-sided 95% family-cluster bootstrap CI, pooled mean gain and one-sided 95% family-cluster bootstrap LCB, harm fraction (families whose median across the three seeds is below -0.01), and Elo versus r882 (median and 95% family-cluster CI per seed, plus the median across seeds). Report Elo, never AUC. Use 10,000 family-cluster resamples, retaining the three paired seed values for each sampled family; preregistered RNG seeds are 99000/99001 for ESH-l2/ESH-l4 mean LCBs and 99010/99011 for family-median intervals.

The candidate hypotheses are tested at family-wise alpha 0.05 with Holm step-down across ESH-l2 and ESH-l4, against the strict mean-gain margin 0.001. Report raw LCB95 and Holm-adjusted decision for each arm. An arm passes only if its Holm-adjusted one-sided test supports mean gain **strictly above 0.001**, its family median is **>= 0**, and harm is **<= 10%**. Exact ties fail. All six candidate cells and the single NULL cell must be complete; a missing, void, or duplicate cell is unknown and blocks admission. The NULL is descriptive only.

## Budget and execution boundary

The later wave may use no more than H80's cumulative **9,000 GPU cell-wall seconds**, including timing, setup, scoring, retries, and voids. It must use the FIFO qualification queue, only after the owner authorizes the next wave and all predecessors have closed; this CPU-preparation PR does not enqueue, acquire a ticket, time a cell, run an outcome, or score Fold B. Later launch rules are CPU load <=85/102, MemAvailable >=16 GiB, disk usage <95.5%, child `nice 19`, OMP/MKL 4 and BLAS/NumExpr 1; never pause or kill another job. Never use xbabe0, xbabe3, or xbabe1's live run directory.

## CPU preparation artifact contract

`run_wave.py --cpu-dry-run` is a data-free synthetic check. It exercises H80's shared ES math on an in-memory 29,211-element zero vector and synthetic family utilities, and prints a JSON plan. It must report no input/checkpoint/outcome reads, no GPU, no timing, no ticket, and zero optimizer updates. This is not experiment evidence or authorization to run the wave.
