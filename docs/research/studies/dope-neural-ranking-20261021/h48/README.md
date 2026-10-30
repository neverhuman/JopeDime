# H48: inner-family-CV utility alignment

Status: **complete — no strict win** (2026-10-30, `cuda:0`). H48 compared a restore-on-reject control (0), persistent AdamW/EMA (PF), and antithetic evolution strategy (ES) on 29,211 adapter parameters. H48-G tested whether a normalized surrogate-gradient descent direction improved held-out DL-Train utility. Baseline r785 was reused from H47; it was not rescored on DL-Val.

## Protocol and decision rule

The approved cohort contained 1,219 DL-Train contexts across 205 families and 91 DL-Val contexts across 19 families. The arms used five fixed whole-family train folds. The 0/PF controls selected the best of r785 and 16 checkpoints on each held-out fold, then selected the best fold. That creates five-fold × 16-look selection, and the reported inner-CV maxima are nonnegative by construction. Each selected fold adapter was fit on 4/5 (164/205) train families. ES selected from antithetic candidates on the same inner family-CV objective; its search maxima are exploratory as well. **Inner-CV gains are winner’s-curse biased and must not be read as generalization gains.**

Each arm/seed’s selected adapter received exactly one terminal DL-Val score (nine total). The strict win rule is median family-mean final Δ > 0 and the 10,000-resample family-bootstrap 95% CI lower bound > 0; ties fail. We report context-weighted deltas separately.

## Terminal results

| Arm | Median family-mean Δ [family 95% CI] | Median context Δ | Elo vs r785 | Min LOCO Δ | Effective seeds | Strict win |
|---|---:|---:|---:|---:|---:|---|
| H48-0 | -0.0063 [-0.0153, +0.0011] | -0.0066 | 1492.68 | -0.0075 | 3 | No |
| H48-PF | -0.0213 [-0.0919, +0.0214] | -0.0222 | 1503.66 | -0.0262 | 3 | No |
| H48-ES | +0.0027 [-0.0014, +0.0094] | +0.0028 | 1492.68 | -0.0367 | 3 | No |

No arm meets the strict rule. ES is the only lead: family-mean median Δ +0.0027, but its family CI lower bound is −0.0014, Elo is 1492.68 (below r785’s 1500), and minimum leave-one-family-out Δ is −0.0367. Its paired family delta vs H48-0 is +0.0072 [−0.00002,+0.0154]. PF’s paired family delta vs H48-0 is −0.0150 [−0.0877,+0.0297].

Every H48-0 and H48-PF seed selected fold 3. This repeated fold selection is the winner’s-curse signature and is disclosed explicitly.

### Per-seed terminal values

| Arm | Seed | Selected fold | Inner-CV selected Δ (exploratory) | DL-Val utility | Context Δ | Family-mean Δ | Selected inner-CV train−Val gap (optimistic) |
|---|---:|---:|---:|---:|---:|---:|---:|
| H48-0 | 0 | 3 | +0.0308 | 1.1309 | -0.0066 | -0.0064 | 0.2465 |
| H48-0 | 1 | 3 | +0.0312 | 1.1310 | -0.0066 | -0.0063 | 0.2465 |
| H48-0 | 2 | 3 | +0.0312 | 1.1335 | -0.0041 | -0.0039 | 0.2439 |
| H48-PF | 0 | 3 | +0.0547 | 1.1153 | -0.0223 | -0.0213 | 0.2751 |
| H48-PF | 1 | 3 | +0.0481 | 1.1509 | +0.0133 | +0.0127 | 0.2403 |
| H48-PF | 2 | 3 | +0.0549 | 1.1154 | -0.0222 | -0.0213 | 0.2767 |
| H48-ES | 0 | all | +0.0066 | 1.1042 | -0.0334 | -0.0320 | 0.2716 |
| H48-ES | 1 | all | +0.0066 | 1.1404 | +0.0028 | +0.0027 | 0.2369 |
| H48-ES | 2 | all | +0.0072 | 1.1411 | +0.0035 | +0.0033 | 0.2369 |

Full-train utility minus DL-Val is also retained as a descriptive gap in `metrics.json`; full-train utility includes fit families. The selected inner-CV train−Val gap above is optimistic because the same inner-CV results were used for selection.

## H48-G diagnostic

Across five whole-family DL-Train folds and five unit-L2 negative-gradient step sizes (1e−6 through 1e−4), all 25 tested actual-utility deltas were exactly 0.0. This supports only the narrow conclusion that the tested normalized surrogate-descent direction did not improve those folds over this range; it does not establish global impossibility.

## Integrity and artifacts

The source manifest records 93 excluded DL-Holdout contexts; zero holdout payloads were opened and zero outer rows were delivered. Train and validation cohorts have zero context and family overlap. Checkpoint SHA256: `ef6696f6369122a162d58b9b522fd83cae110a5576ef3760e3b43ba85d129e0a`; DOPE SHA256: `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`; source-manifest SHA256: `3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3`. Per-family results, bootstrap output, arm trajectories, nine terminal records and the gradient diagnostic are in [`metrics.json`](metrics.json).

H49 remains queued until this closeout PR merges. Its preregistered plan retains 0/SAM/SWA/SEL and adds ES2 on fresh seeds 3–5, with twice H48-PF’s median wall budget (~141 seconds per seed), identical inner-CV selection, one terminal DL-Val read per selected seed, and a strict win only when median family-mean Δ > 0, family-bootstrap lower bound > 0, and Elo > 1500. If GPU/lock time is constrained, drop SWA before ES2.
