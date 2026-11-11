# H422 — learned stacker: `SEL_NULL` on DEV (negative result; closeout)

**Verdict: `SEL_NULL`.** On the 115 fresh `DEV` families the frozen stacker (a winsorised combiner, constant k = 6)
on the stored candidates does not meet the pre-registered success rule against the stored stack: clipped
family-mean gain +0.0045 u (needs ≥ +0.010), sign-flip p 0.239, ΔElo +33.4 [−10.4, +78.9]. **The stack stays as it
is.** The TRAIN-side cross-fit had shown +0.0197 u; `DEV` shows about a quarter of that.
**One reported arm stands out and is not a verdict:** the stacker on a wider admitted candidate set (`SA`) beats
the stored stack by +0.019 u (p 0.012, ΔElo +119 [+53, +192]). It needs its own pre-registered test.
TRAIN-side cohorts only: no dev91, DL-Val, DL-Holdout, sealed or `CONFIRM` data was read. No worker evaluation ran.

Plan: [`PREREG.md`](PREREG.md). Numbers: [`metrics.json`](metrics.json). Frozen model: [`FROZEN.json`](FROZEN.json).
TRAIN-side cross-fit: [`CROSSFIT.json`](CROSSFIT.json). Gates: [`gates_h422.sh`](gates_h422.sh).

## Result (DEV, 115 families, value = AUDIT u, paired by family)
| Comparison | Mean gain (u) | Clipped mean | Sign-flip p | Leave-one-family-out min | Median | Win / tie / loss | ΔElo [95% interval] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **`S` vs `B` (primary): stacker, stored candidates** | +0.0148 | **+0.0045** | **0.239** | +0.0002 | 0.0000 | 46 / 35 / 34 | +33.4 [−10.4, +78.9] |
| `A` vs `B`: wider admission, stored combiner | +0.0065 | +0.0065 | 0.104 | +0.0029 | +0.0002 | 64 / 11 / 40 | +70.2 [+17.1, +124.3] |
| `SA` vs `B`: wider admission + stacker | +0.0446 | +0.0191 | 0.012 | +0.0148 | +0.0011 | 71 / 8 / 36 | +119.4 [+52.8, +191.7] |
| `SA` vs `S`: admission, given the stacker | +0.0298 | +0.0136 | 0.032 | +0.0093 | +0.0004 | 66 / 12 / 37 | +86.0 [+29.2, +145.9] |
| `SA` vs `A`: stacker, given the admission | +0.0381 | +0.0155 | 0.034 | +0.0112 | 0.0000 | 50 / 32 / 33 | +49.2 [+6.9, +93.7] |

Mean u: `B` 1.9483, `S` 1.9631, `A` 1.9548, `SA` 1.9929. The frozen model is the constant-k model, so `S1` = `S`.

**Success rule (PREREG §5), primary arm:** gain ≥ +0.010: no (+0.0045). Test passes at one-sided 0.025: no
(p 0.239; also no at 0.05). Elo interval above 0: no. All three fail.

## What the result does and does not say
- **Combining the stored 64 candidates more robustly is not enough.** The estimate is positive, but 46 wins against
  34 losses and +0.0045 u cannot be told from zero. Only 5 of 115 families move by more than 0.05 u (all upward);
  in the design cross-fit 20 of 291 moved up by that much and 8 down.
- **The TRAIN-side estimate again overstated the test** (+0.0197 u out-of-fold on 291 design families, +0.0045 u on
  `DEV`), as in H421. Here the stacker's form was also chosen on the design families, which the cross-fit does not
  cover (PREREG §9).
- **The two levers look complementary, in one reported arm.** With the wider candidate set (median 116 columns
  instead of 64) the same frozen combiner gains +0.0155 u over the wider set alone, and the pair gains +0.0191 u
  over the stored stack with an Elo interval above 0. On `FIT-SEL` (in-sample smoke before the run) the same arm
  showed +0.028 u.
- **Why `SA` is a lead and not a finding:** it was a pre-registered ablation without a success rule, one of three
  non-baseline arms read here, and `DEV` has now been read for it. Its p-value does not survive a correction for
  three arms at the 0.025 level (0.012 × 3 = 0.037).
- **Wider admission alone** wins more families than it loses (64 to 40) by small margins (+0.0065 u, p 0.10).
- **Cost of the wider set, if deployed:** it pools the candidates admitted by four variants of the recipe, so it
  needs the fourth search run and about 1.8 times the 5-fold evaluations of the stack stage.
- **Design-side negatives worth keeping (PREREG §8):** no reweighting of the stored members gains more than
  +0.006 u; removing weak or near-duplicate candidates does not help (24 of 64 slots are near-duplicates and
  harmless).

## TRAIN-side cross-fit (291 design families, family-grouped, out-of-fold; published before DEV was opened)
Clipped gain +0.0197 u, p 0.00009, 136 wins / 73 ties / 82 losses; P201e +0.0188, `FIT-SEL` +0.0215. The inner
choice kept the constant model in 3 of 5 folds and on all contexts (k = 6): the context gate added nothing reliable.

## Integrity gates (`gates_h422.sh`, run after the apply step and before the readout; 8 of 8 pass)
| Gate | Evidence |
| --- | --- |
| Inputs | all four input copies match their manifests |
| Design only | 185 + 106 design contexts; recomputed stack weights equal the stored ones on all 106 records that store them |
| TRAIN gate | passed; k = ∞ reproduces the stored design within 1e-6 u |
| Frozen before DEV | `FROZEN.json` of commit `3b30a9051e` (sha256 `a43739b6469e2964…`) is the file the apply step used; the commit is older than the first `DEV` cache file |
| DEV replay | 115 `DEV` contexts; weights equal the stored ones (difference 0); baseline value within 2.9e-10 u |
| No AUDIT target in predictions | predictions unchanged with the AUDIT targets set to zero |
| Published stage | code hash verified; staged PREREG equals the published one |
| Budget | 651 CPU-seconds in total (cap 21,600) |

## Deviations and weaknesses
1. The stacker's form (winsorising, the features, the objective weight) was chosen after six looks at the design
   contexts (PREREG §8). `DEV` was not involved.
2. The PREREG's p-value rule is one-sided 0.025, stricter than the relayed "p < 0.05"; the result fails both.
3. The P201e input copy has no manifest from its source host (PREREG §9).
4. `SA` applies parameters fitted on 64-candidate stacks to stacks of about 116 candidates.
5. Two development stages are marked superseded (`dev-s1` wrong record name, `dev-s6` staged an unchanged file).
6. The combiner exists only as offline code on stored matrices.

## Queued follow-ups (prompts, not started)
1. **Confirm `SA` as a primary.** "Pre-register wider admission + winsorised combiner (frozen k = 6, `FROZEN.json`
   sha256 `a43739b6…`) as the primary arm against the stored stack on families no wave has read: deploy the four
   stored variants (`BASE`, `TGX`, `TLIB`, `TBOTH`) on a fresh, hash-chosen slice of fresh families outside
   `CONFIRM` (about 100 contexts), seeded from nothing. Estimate the worker-hours from H420's records first (H420
   `TBOTH` on 115 `DEV` contexts: 988,622 worker-seconds) and get the budget approved before launch."
2. **Replicate `S` at no worker cost.** "Apply the frozen combiner to the stored P201b matrices of H416 (201
   DL-Train contexts on xbabe0; same families as P201e, different contexts) as a reported replication: one copy of
   the stored files to xbabe2, `cache_h422.py` + `sel_h422.py apply`-style scoring, no fit. It tells whether
   +0.0197 (design) or +0.0045 (`DEV`) is closer to the truth for the stacker alone."
3. **Make the two levers deployable.** "Add the winsorised combiner (`k` as a design parameter) and a pooled
   admission option to the offline deploy path (`meta_h416.deploy_pred`, `deploy_h415.hf_candidates`) behind new
   design names, with a replay test that k = ∞ and the stored admission reproduce `h416.SXB` to 1e-9 on two stored
   contexts. Code only; no scored run."

## Timeline (UTC, 2026-11-11)
| Time | Event |
| --- | --- |
| 06:40 | go-ahead relayed by the owner's watcher |
| 06:45 | H422 claimed |
| 06:46–07:06 | development on design contexts (six looks, three cross-fits, one code smoke on `FIT-SEL`) |
| 07:07 | PREREG and code published (`30314d9e58`); design caches and cross-fit from the published stage; TRAIN gate passes |
| 07:07 | `FROZEN.json` and cross-fit published (`3b30a9051e`) |
| 07:08 | `DEV` cache and apply; gates (8 of 8); readout: `SEL_NULL` |

Evidence (xbabe2): `~/jope_plan/evidence/H422/` (`results/H422_DEV.{apply,gates,readout}.json`, `results/FROZEN.json`,
`results/CROSSFIT.json`, `logs/`), caches and step outputs in `~/jope-data/h422/pub/`.
