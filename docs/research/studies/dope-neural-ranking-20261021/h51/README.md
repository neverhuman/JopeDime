# H51 preregistration: seed diversity, step budget, data variety, and robust objectives

Status: **preregistered; execution not started**. This file is the first H51 commit and records the protocol before any H51 training. H50 PR #537 was squash-merged at `22c83d184e2516f090c915401440512518a6ed17` on 2026-10-30.

## Question and hypotheses

H50 found that 79.7% of random adapter perturbations at radius 0.01 produced exactly zero utility change and that no short-search candidate passed the strict family-bootstrap margin. H49 also showed that distinct control adapters can have identical DL-Val utility. H51 asks whether seed variation, optimizer step budget, more family variety, clipped context losses, or a smoothed rank surrogate can produce reproducible inner-family gains that survive the strict admission rule. The overseer-supplied xbabe1 round summary and the two-AdamW-step root-cause account are hypotheses only; the live loop and its files remain out of scope.

## Frozen inputs and data boundary

- DOPE is frozen at SHA256 `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` (29,211 trainable adapter parameters; no DOPE parameter updates).
- Primary baseline checkpoint is the offline copy `~/jope-data/h50/ckpt/round-785.pt`, SHA256 `ef6696f6369122a162d58b9b522fd83cae110a5576ef3760e3b43ba85d129e0a`.
- Input source manifest is `~/jope-data/h50/input/source_manifest.json`, pinned manifest identity `3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3`, file SHA256 `99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4`: 1,219 DL-Train contexts / 205 families and 91 DL-Val contexts / 19 families. The source manifest records 93 excluded holdout contexts and zero excluded-payload reads. Use eligible real S3 regression contexts only.
- No sealed, holdout, outer-test, or live-loop files or payloads may be opened. The only terminal DL-Val read allowed is one read for a candidate already admitted by inner CV.
- Before and after each arm, verify and record the DOPE SHA. All outputs, adapters, and checkpoints stay outside Git; Git stores only compact JSON summaries and code/docs.

## Fixed evaluation and admission

Use five fixed whole-family folds over DL-Train, with the H49 fold definition/seed (`FOLD_SEED=4900`) and family-balanced training batches. Keep each context and its family wholly in one fold. For every candidate, aggregate its held-out per-context utilities into per-family mean gains against the same-fold r785 baseline. Persist the family gains only in the external run evidence; the compact candidate JSON records:

```json
{
  "schema": "jope.h51.candidate-summary/v1",
  "candidate_id": "...",
  "arm": "...",
  "seed": 9,
  "adapter_sha256": "...",
  "family_median_gain": 0.0,
  "family_bootstrap_lb95": 0.0,
  "family_bootstrap_ub95": 0.0,
  "n_families": 0
}
```

Use 10,000 deterministic family-bootstrap resamples, sampling families with replacement and recomputing the median family gain. A candidate is admitted only if its point median is strictly greater than `0.001` **and** its 95% bootstrap lower bound is strictly greater than `0.001`. Equality fails. No admitted candidate means no terminal DL-Val read and Elo is N/A. For each admitted selected candidate, allow exactly one terminal DL-Val score; report Elo versus the pinned r785 control only for that terminal result. A positive inner-CV score alone is not a win.

For seeds 9, 10, and 11, record the adapter SHA and whether the per-family inner utilities differ. Require distinct adapter SHAs; identical utilities must be described as step-like utility, not seed agreement. Do not open DL-Val to manufacture a seed comparison when a seed has no admitted candidate. Where terminal reads are admitted, report whether their utilities differ separately from the inner-CV comparison.

## Arm order and protocols

Run serially in this order; all arms use fresh seeds 9–11 and the same five-fold family grouping unless stated otherwise.

1. **CONTROL** — fresh H49-0-style control training from r785: persistent AdamW at the established baseline settings (16 rounds × 2 updates per round), family-balanced batches with seed-varied sampling. This is the seed-diversity control; it is not a repeated evaluation of the identical r785 weights.
2. **STEPS** — root-cause factorial, from the verified offline r882 checkpoint: persistent AdamW with `k ∈ {2, 16, 64}` updates and `lr ∈ {1e-4, 3e-4}`, family-balanced minibatches. Run `(k=16, lr=3e-4)` and `(k=64, lr=3e-4)` first, then the remaining four combinations. Record update L2 per candidate/arm against the H50-S radius map. The verified checkpoint is `~/jope-data/h51/ckpt/round-882.pt`, SHA256 `0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`; its provenance file says it was copied read-only from xbabe1 without touching the live loop. Before first use, the file hash, round number 882, DOPE SHA `c51a56d0…`, and exactly 29,211 trainable adapter parameters were verified. Never substitute r785.
3. **DATA** — nested, seeded subsets of 25%, 50%, and 100% of DL-Train families, evaluated on the same fixed inner held-out family fold; all subset lists are written before training. If inner utility rises from 50% to 100%, the next question is whether additional eligible non-sealed real S3 regression contexts help; if flat, prioritize capacity/objective. Do not fetch unlisted data or open excluded payloads in this wave.
4. **CLIP + SMOOTH** — run both preregistered objective variants as separately identified subarms. CLIP caps each per-context objective delta to `[-0.5, +0.5]` while preserving the failure floor of `-2.0`, with genuinely family-balanced minibatches. SMOOTH uses temperature-softened ranks at `tau ∈ {0.1, 0.3}`, minibatch AdamW, and an SWA tail. Select only by the fixed family-CV rule above.
5. **NULL** — best-of-N random adapter directions at compute matched to the trained candidate arm, using identical folds, candidate counting, family aggregation, bootstrap, and selection. Report N, direction radii, fold-level results, and selection procedure; do not compare a one-fold maximum with a five-fold trained-candidate aggregate.

DEEP-ES is omitted from this H51 preregistration. The earlier conditional suggestion to add it if eight generations fit is superseded by the later instruction to add no H51 arms; no DEEP-ES arm is planned in this wave.

## Compute and safety guards

- Serialize all arm work. GPU computation requires holding `~/jope_plan/qualification.lock`; never bypass the lock.
- Launch only when MemAvailable is at least 20 GiB, CPU utilization is at most 85%, and GPU free memory is at least 20 GiB. Use `nice -n 15`; cap OMP and MKL threads at 8 (OpenBLAS at 1). Record device, guards, lock wait, and wall time per arm/seed.
- If the GPU lock remains busy for more than 20 minutes before any H51 arm starts, use CPU fallback with at most 8 Torch threads, nice 15, and the same RAM/CPU guards. Keep device strata separate; do not pool CPU/GPU comparisons as if they were matched.
- Stop launching new seeds when a guard fails; retain all completed evidence. Never interrupt another owner's lock holder or process.

## Reporting

Record per candidate the compact summary fields above and retain full per-family vectors, fold assignments, optimizer/update metadata, and provenance externally under `~/jope-data/h51/`. For every arm and seed report completion status, distinct adapter SHA, inner utility differences, candidate count, point median and bootstrap interval, `n_families`, selected-candidate admission, terminal-read status, and Elo status. Explain the overseer-supplied 40-round/7-utility live-loop summary as unverified; no live-loop inspection is allowed. Report unfinished arms and any device change explicitly. Keep the conclusion bounded to the tested seeds, steps, data subsets, objectives, and compute budget.

## Preregistration amendment — 2026-10-30 14:30Z

- The r882 offline copy is verified and available for STEPS: checkpoint SHA256 `0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`, metadata round 882, DOPE SHA256 `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`, 29,211 trainable adapter parameters. Provenance: `~/jope-data/h51/ckpt/round-882.provenance.json`.
- Run the two larger/faster-leaving-basin STEPS settings first: `k=16, lr=3e-4`, then `k=64, lr=3e-4`; continue with the other factorial settings only after those complete.
- A red-team comparison reports that H50-ESEMA rows are byte-identical to H50-ES3 (the same 64 candidates). H50 therefore supplies two independent search arms, not three; ESEMA is not independent corroborating evidence. This comparison is overseer-supplied and is not re-described as independently verified here.
- Emit `CONTROL` as the canonical H51 arm label even where the reused H48 implementation internally calls that baseline arm `0`. Record selected adapter update L2 from r785 and its adapter-weight SHA. If all fold/seed utilities are exactly zero, interpret that as step-like utility consistent with H50-S, not “no change by design.”
- Due to xbabe2 swap being full (19/19 GiB), keep each run's RSS below approximately 6 GiB in addition to the existing RAM/CPU/GPU guards; continue to cap OMP/MKL at 8 and run at nice 15. Use targeted checks only; do not run `just required` for H51.

## Findings (interim, 2026-10-30)

### H50 red-team carryover and H51 protocol reconciliation

H50's bounded conclusion is “no candidate cleared family-bootstrap LB > 0.001 within two generations,” not “ES/CMA cannot work.” Its runner's `elapsed + 150 < 282s` condition ended each seed after two 32-candidate generations; with `SIGMA0=0.001` (nominal adapter L2 about 0.171) and ES displacement `0.05*sigma`, this was a short search near r785, with little time for CMA covariance adaptation and only two iterates in ESEMA. The H50 README and compact metrics retain each generation's candidate mean-delta min/median/max beside the H50-S best-of-64 null at radii 0.1 and 0.3. CMA seed6 generation1 had median +0.001189 and maximum +0.012021 but zero admitted candidates. The inner-CV reference 1.386373 and historical, unrescored DL-Val reference 1.137595 use different splits and are not a direct comparison. All nine H50 candidates were rejected: terminal DL-Val reads were zero and Elo is N/A (no candidate).

An earlier H50 red-team note proposed CLIP/SMOOTH before other H51 arms and conditional DEEP-ES. H51 is already running under the later controlling preregistration: finish STEPS lr=1e-4, then DATA, CLIP + SMOOTH, and NULL; add no H51 arms. H50's earlier proposal is superseded on this sequencing point. Compare seeds using persisted inner-CV results and adapter identities; do not demand distinct terminal validation utilities when admission fails. Terminal reads remain admission-gated, and Elo is N/A for rejected candidates.

### CONTROL: training moves weights; utility is step-like

CONTROL produced three distinct adapter weight hashes. Their update L2 distances from r785 were 0.102599, 0.101629, and 0.101990 for seeds 9, 10, and 11. Mean inner-family gains were +0.005890, +0.005825, and +0.005950, while each seed's family median and bootstrap interval were exactly 0.0 ([0, 0]). This is step-like argmin utility, not “no change”: 176/205, 178/205, and 175/205 families, respectively, had exactly zero gain. A family changes utility only when its selected argmin changes, so weight movement and positive mean gain can coexist with a zero median.

### STEPS: larger r882 regression steps have hurt so far

In the completed lr=3e-4 cells, increasing the step budget from k=2 to k=16 and k=64 produced increasingly negative family medians from the r882 baseline. At k=2, all three seeds had median and interval zero, mean gain about -0.010, and update L2 about 0.105. At k=16, medians were -0.026836, -0.025059, and -0.019407, all with bootstrap upper bounds below zero; means ranged from -0.081818 to -0.093824 and update L2 from 0.927 to 0.972. At k=64, medians were -0.036789, -0.033753, and -0.037672, again with upper bounds below zero; update L2 was 3.112–3.160 and means ranged approximately -0.068 to -0.110. These results are consistent with regression-loss gradient steps moving away from argmin utility, rather than a simple under-optimization plateau. The lr=1e-4 cells are still running, so this interpretation remains provisional until that factorial branch completes.

### Seed dependence and live-loop winner's curse

The overseer reports that k2 seed 9 and seed 10 utilities match to four decimals, but the persisted fold summaries do not exactly match at that precision: for example, fold 2 family deltas are -0.002979 and -0.003380, and the corresponding candidate family utilities differ by about 0.0004. The saved evidence therefore supports similar, step-like fold behavior, not an exact four-decimal match. The three seeds should not be treated as fully independent confirmation; seed changes may mainly alter minibatch order, and across-seed intervals can understate run-to-run variation.

Separately, the overseer-supplied live-loop figures say that r882 utility 1.1377942 was the maximum across roughly 1,280 reads of the same 91 validation contexts, while later live candidates were 1.1344–1.1352. These figures have not been independently verified and the live loop was not inspected. If accurate, selection of the incumbent at that maximum creates winner's-curse bias and regression to the mean, which could contribute to the apparent plateau. This is a caveat, not a change to the offline H51 baseline or protocol.

The preregistered remaining order is STEPS lr=1e-4, then DATA, CLIP + SMOOTH, and NULL. No additional H51 arms are being added. These findings are interim; the final verdict and compact metrics are reserved for closeout after all arms finish.
