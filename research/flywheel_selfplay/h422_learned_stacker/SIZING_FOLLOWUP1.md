# Sizing of follow-up 1 (wider admission + combiner on fresh families) — planning arithmetic, not a finding

Written 2026-11-11 after the relayed request for a 40 worker-hour sizing pilot (H424). **The pilot was not started
and H424 was not claimed**: the values already stored answer the sizing question more precisely than a pilot of that
size can, and the pilot needs prerequisites that do not exist yet. This note records the arithmetic and the reasons.
No new matrix was opened; the inputs are H422's stored arm values and stored evaluation memos.

## Inputs (all already read by H422)
| Cohort | Families | `SA − B`, clipped mean (u) | Standard deviation | Wins / ties / losses | At the ±0.5 clip |
| --- | --- | --- | --- | --- | --- |
| `DEV` (H422 reported arm) | 115 | +0.0191 | 0.0932 | 71 / 8 / 36 | 3 |
| `FIT-SEL` (code smoke; in-sample for the combiner's form) | 106 | +0.0279 | 0.1000 | 64 / 7 / 35 | 3 |

- Pooled standard deviation 0.0965; pooled mean +0.0233.
- **With the clipped contexts excluded** (the secondary the request asked for): `DEV` +0.0062 u, sd 0.050, 68 wins /
  36 losses of 112; `FIT-SEL` +0.0142 u, sd 0.060, 61 / 35 of 103. The win share of decided families stays at
  64–65% in both cohorts; the mean gain falls to a third (`DEV`) or a half (`FIT-SEL`).
- **Cost per fresh family** for the four-variant pipeline (four search runs, both libraries, 5-fold and AUDIT
  evaluations of about 116 candidates), from the evaluation memos of the 106 `FIT-SEL` contexts: mean 3.52
  worker-hours, median 2.58, 90th percentile 7.5, largest 21.0. Search is 46% of it.

## Families needed for 80% power at one-sided 0.025 (normal approximation)
| If the true effect is | Clipped-mean test: families | Worker-hours | Power at 100 | Power at 225 |
| --- | --- | --- | --- | --- |
| +0.0279 u (`FIT-SEL`, in-sample) | 94 | 331 | 0.82 | 0.99 |
| +0.0233 u (pooled) | 135 | 476 | 0.68 | 0.95 |
| +0.0191 u (`DEV`) | 202 | 712 | 0.51 | 0.84 |
| +0.010 u | 732 | 2,579 | 0.18 | 0.34 |

| If the win share of decided families is | Win-rate (sign) test: families | Worker-hours |
| --- | --- | --- |
| 0.66 (`DEV`) | 80 | 282 |
| 0.65 (`FIT-SEL`) | 91 | 321 |
| 0.60 | 208 | 733 |

- **Recommendation.** Under the campaign's clipped-mean test, 100 families (the size first proposed; about 350
  worker-hours at the measured cost) give only about 50% power if the `DEV` estimate is the truth. About 200 families (about 700
  worker-hours) are needed. Only 225 core fresh families remain outside `CONFIRM` (the `FIT-ICL` cohort), so the
  test would use nearly all of them, once.
- **If the true effect is +0.010 u** (plausible: `SA` was the best of three arms read on `DEV`), no affordable size
  works; there are not that many fresh families.
- **A win-rate primary would be far cheaper** (80–90 families) because the gain is carried by few large movers
  while the win share is steady. That would be a different pre-registered rule from the campaign's standard test;
  it is the owner's choice, not a recommendation to switch.

## Why a 40 worker-hour pilot would not improve this
| Pilot size (40 worker-hours ≈ 11 families; 8 to 15 given the cost spread) | Standard error of its mean | 95% half-width | Relative error of its sd |
| --- | --- | --- | --- |
| 8 families | 0.034 | ±0.067 | 27% |
| 12 families | 0.029 | ±0.056 | 22% |
| 15 families | 0.025 | ±0.049 | 19% |

- The effect in question is about +0.02 u. A pilot mean with a half-width of ±0.056 cannot tell +0.010 from +0.028,
  which is the whole range of the table above.
- The standard deviation is already known to about ±5% from 221 stored families; the pilot would know it to ±22%.
- Its win rate (11 families) would have a standard error of 0.15.
- **Prerequisites that do not exist:** the 225 `FIT-ICL` families have no SEARCH view sets on xbabe2 (0 of 225;
  H417 encoded `DEV` and `FIT-SEL` only), so the pilot would first run the frozen encoder on CPU; the run code
  accepts only the panels `DEV`, `FIT-SEL`, `BASE` and `SMK2`, so a new panel means new published code for the
  search and deploy steps; and no deploy unit pools the four variants (H423's code is not wired). These are the
  same prerequisites follow-up 1 itself needs.
- What a pilot would still give: a rehearsal of the fresh-family pipeline and a measured cost per family. Both come
  for free as the first block of follow-up 1 if it is approved.

## Limits of this arithmetic
- The `DEV` effect is the best of three arms read there and is likely too high; the `FIT-SEL` value is in-sample for
  the combiner's form. The truth may be below both.
- Normal approximation; the campaign's sign-flip test on clipped gains has about the same power for these sizes.
- The cost figures come from memos that include every evaluation of four variants on `FIT-SEL`; a few repeated
  evaluations after the host move are counted twice. Costs are heavy-tailed, so a budget cap by worker-hours makes
  the number of families uncertain.

Files (xbabe2, `~/jope_plan/evidence/H422/sizing/`): `sizing.jq`, `dev_SA_minus_B.json`, `fitsel_SA_minus_B.json`,
`memo_cost.sh`, `fitsel_memo_cost.tsv`, `power.awk`, `power.txt`.
