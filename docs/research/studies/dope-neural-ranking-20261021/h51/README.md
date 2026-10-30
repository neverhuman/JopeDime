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

Do not add DEEP-ES in this preregistration: it is omitted unless the available per-seed wall budget can first be shown to fit at least eight generations (about 900 seconds/seed) without weakening the guards or other arm protocols.

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
