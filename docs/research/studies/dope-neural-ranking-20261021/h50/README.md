# H50: utility sensitivity and short-horizon search

Status: **complete — no candidate admitted; no terminal DL-Val reads** (2026-10-30, `cuda:0`). H50 measured adapter utility changes under random perturbations, then evaluated antithetic ES, separable CMA-ES, and EMA-of-ES for fresh seeds 6–8. All 9 arm/seed runs completed. None produced a candidate whose family-bootstrap 95% lower bound of median inner-CV family gain exceeded 0.001, so no candidate was selected and the terminal-validation budget was unused. Elo is **N/A (no selected candidate)**; no win or validation claim is made.

## Protocol and cohort

The r785 checkpoint and DOPE weights were frozen. Only the 29,211-parameter adapter was perturbed or optimized. H50-S evaluated 64 seeded random unit directions at each L2 radius `[0.01, 0.03, 0.1, 0.3, 1.0]` on fixed, whole-family held-out DL-Train folds. Each direction was assigned to one fold (13/13/13/13/12 directions per radius across five folds); it was not fit on that fold. The search arms used five-family-fold inner CV, population 32, two generations per seed, sigma 0.001, and the strict admission rule `family-bootstrap 95% lower bound of median family-mean utility delta > 0.001`. Each ESEMA seed also scored its EMA center once per generation.

The copied cohort contains 1,219 DL-Train contexts / 205 families and 91 historical DL-Val contexts / 19 families, with no overlap. Ninety-three DL-Holdout contexts were excluded; no holdout payloads or outer rows were opened or delivered. H50 reused the historical r785 DL-Val reference (mean context utility 1.137595; not rescored). Its inner-CV baseline was 1.386373. These use different splits and are not directly comparable; the gap is a warning about train/validation transfer, not an effect estimate.

## Sensitivity map and random-direction null

| L2 radius | Exactly zero | Minimum Δ | Median Δ | Maximum Δ (best of 64) |
|---:|---:|---:|---:|---:|
| 0.01 | 51/64 (79.7%) | -0.001264 | 0.000000 | +0.001769 |
| 0.03 | 26/64 (40.6%) | -0.001885 | 0.000000 | +0.001769 |
| 0.1 | 2/64 (3.1%) | -0.010392 | +0.000134 | +0.023762 |
| 0.3 | 0/64 (0%) | -0.036698 | -0.000024 | +0.032877 |
| 1.0 | 0/64 (0%) | -0.026387 | +0.000507 | +0.052206 |

The smallest tested radius with any movement was 0.01, so the preregistered stop rule did not fire. At radius 0.01, nearly four in five directions changed no utility. This is consistent with a piecewise-constant, rank-like objective: many small parameter changes do not change ordering or rank. It provides a concrete reason that local gradient smoothing can fail to move the argmin utility in H46–H49; it does not prove smoothing can never help at larger scales.

For the optimizer search, a 0.001 standard deviation over 29,211 parameters has a nominal L2 scale near 0.171, so radius 0.1 is the nearest tested sensitivity radius (0.3 is the next). The H50-S best-of-64 one-fold held-out null at those radii was +0.023762 / +0.032877. Each ES3/CMA seed evaluated 64 random candidates across two generations, so N=64 matches; ESEMA had those same 64 random candidates plus two EMA-center evaluations. **No seed had an admitted/selected candidate**, so a selected inner-CV gain per seed is N/A. The table below instead reports only descriptive, unadmitted generation summaries. This is not a like-for-like winner’s-curse test: the H50-S maxima are single-fold deltas, while search values aggregate family gains over five folds; later-generation candidates are centered on adapted ES/CMA states, and CMA changes its scale. The radius/N match is approximate and is disclosed rather than treated as validation evidence.

| Arm | Seed | Gen 1 min / median / max candidate mean Δ | Gen 2 min / median / max candidate mean Δ | Strictly admitted (of 64) | Best-of-64 H50-S null at r=.1 / r=.3 |
|---|---:|---:|---:|---:|---:|
| ES3 | 6 | -0.003139 / +0.001570 / +0.008416 | -0.003674 / +0.001027 / +0.008169 | 0 | +0.023762 / +0.032877 |
| ES3 | 7 | -0.003675 / +0.001197 / +0.008536 | -0.004055 / +0.002083 / +0.006774 | 0 | +0.023762 / +0.032877 |
| ES3 | 8 | -0.003332 / +0.001371 / +0.010748 | -0.003511 / +0.000477 / +0.008410 | 0 | +0.023762 / +0.032877 |
| CMA | 6 | -0.002728 / +0.001189 / +0.012021 | -0.003604 / +0.004195 / +0.011396 | 0 | +0.023762 / +0.032877 |
| CMA | 7 | -0.002508 / +0.000801 / +0.010304 | -0.002180 / +0.004052 / +0.010128 | 0 | +0.023762 / +0.032877 |
| CMA | 8 | -0.002668 / +0.001223 / +0.008830 | -0.002496 / +0.003046 / +0.008874 | 0 | +0.023762 / +0.032877 |
| ESEMA | 6 | -0.003139 / +0.001570 / +0.008416 | -0.003674 / +0.001027 / +0.008169 | 0 | +0.023762 / +0.032877 |
| ESEMA | 7 | -0.003675 / +0.001197 / +0.008536 | -0.004055 / +0.002083 / +0.006774 | 0 | +0.023762 / +0.032877 |
| ESEMA | 8 | -0.003332 / +0.001371 / +0.010748 | -0.003511 / +0.000477 / +0.008410 | 0 | +0.023762 / +0.032877 |

These generation medians are medians across candidates of each candidate's **mean** family delta; they are not the selected candidate's median family delta. Positive generation medians therefore do not satisfy the preregistered admission rule. Every generation recorded zero strictly admitted candidates. The search output did not persist each rejected candidate's family median and bootstrap interval, so the available aggregates cannot determine candidate by candidate whether its point family median was at or below 0.001 or its bootstrap lower bound alone failed. The defensible conclusion is that no candidate cleared the family-bootstrap lower-bound threshold—not that each candidate's point median failed.

Search depth was limited by the preregistered 282-second wall budget and the runner's `elapsed + 150 < 282` loop condition: two generations, 32 candidates per generation, about 64 random candidate evaluations per seed. At sigma 0.001, ES3 is a short best-of-64 search near r785; CMA has only two updates to adapt its diagonal covariance; ESEMA averages only two ES iterates. Thus **“no win” means no candidate cleared the strict inner-CV admission rule within this two-generation budget**. It does not establish that ES or CMA cannot work with a longer budget or a different objective.

CMA seed 6 generation 1 illustrates the distinction: candidate mean-delta median +0.001189 and maximum +0.012021, but zero of 32 candidates passed strict admission. All ES3 and CMA generations likewise had zero admissions; all six ESEMA EMA-center evaluations were rejected (EMA mean/median delta 0.0 in each saved generation record). The report does not claim to know which rejected candidates failed on their point family median versus only on the bootstrap lower bound, because per-candidate intervals were not retained.

## Terminal evidence and interpretation

All nine `train-result.json` records say `selected_candidate_accepted=false`, `validation_scored=false`, `validation_payloads_opened=false`, and `outer_rows_delivered=0`. There were zero terminal DL-Val reads and no H50 Elo values. The reused historical r785 DL-Val reference is descriptive only and was not rescored. The H50-S null and search-fold gains are exploratory inner-CV evidence, not a win claim.

The random-direction maxima are a winner's-curse control: at the nearest tested radius, a best-of-64 random search already reached +0.023762 on one held-out fold. Search gains in the table are five-fold family means and cannot be directly subtracted from that one-fold maximum. A future matched-compute null must use the same candidate count, fold aggregation, center/update rule, and selection process before claiming the optimizer beat random search.

The live-loop root-cause claim supplied with the H51 nudge (two AdamW steps per round and a ~0.036 update norm) was not independently inspected here; live-loop state remained out of scope. Treat “step budget traps the model in a flat basin” as an H51 hypothesis, not an H50 finding.

## H51 follow-up (not launched by this closeout)

The post-H50 H51 order is CONTROL, STEPS, DATA, CLIP+SMOOTH, then matched-compute NULL. STEPS is a persistent-AdamW factorial over steps `{2, 16, 64}` and learning rates `{1e-4, 3e-4}`, with family-balanced batches and update-L2 reporting against this sensitivity map. DATA is a seeded nested family learning curve at 25/50/100% with a fixed held-out inner fold. If utility continues rising from 50% to 100%, consider additional eligible real, non-sealed S3 contexts; if it is flat, investigate capacity or objective. An overseer-supplied read-only summary says xbabe1 rounds 1190–1229 had 40 distinct candidate SHAs but 7 distinct utilities (1.13047–1.13771), 0/40 accepted, and the best remains 1.137794 from r882. This supports the step-like-utility hypothesis but was not independently inspected by H50; the live loop stays untouched. Use fresh seeds 9–11 and verify both distinct adapter SHAs and whether their validation/inner utilities differ; identical utilities are step-like behavior, not seed agreement. Persist each candidate's family median, bootstrap lower/upper bounds, and family count as compact JSON (no checkpoints in Git). Add DEEP-ES only if the budget permits at least eight generations (about 900 seconds per seed), otherwise omit it. Keep DOPE frozen, regression-only scope, and all sealed/holdout data closed. H51 is a proposal for after this H50 closeout, not an H50 result.

## Integrity and artifacts

- DOPE SHA256: `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`; trainable adapter parameters: 29,211.
- r785 checkpoint SHA256: `ef6696f6369122a162d58b9b522fd83cae110a5576ef3760e3b43ba85d129e0a`.
- Source-manifest file SHA256: `99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4`; pinned manifest identity: `3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3`.
- Full sensitivity and search outputs remain outside Git under `~/jope-data/h50/runs/`; `metrics.json` contains a compact, reproducible summary and input/output hashes.
- H50 source and targeted test harness: `research/flywheel_selfplay/h50/`.
