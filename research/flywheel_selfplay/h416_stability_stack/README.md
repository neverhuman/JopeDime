# H416: stability-selected cross-fit stacking over the pooled population (TRAIN P201e → P201b)

**Verdict: `SXB_CONFIRMED`; also `KFB_BETTER`.** This is a TRAIN-only result. dev91 was not read.

- **Primary passes.** `h416.SXB` (design `BCAR.mix`) against `h415.KFB` on the P201b TRAIN board:
  - ΔElo **+45.1** [+25.2, +65.4] (family bootstrap, seed 416150, 10,000 draws);
  - families W/T/L 133/6/62 (win fraction 0.677);
  - clipped mean +0.0114, sign-flip p 0.022.
  - The mean gain is −0.0022 (−0.2% RMSE). SXB wins more families than KFB, but on mean RMSE the two are level.
- **The two concepts are now distinguishable without the holdout** (Holm over 6 pairs):

  | Pair | ΔElo [95% CI] | W/T/L | Holm |
  | --- | --- | --- | --- |
  | SXB − KFB | +45.1 [+25.2, +65.4] | 133/6/62 | rejected |
  | SXB − H410-style `BAG3` | +134.5 [+101.0, +168.4] | 138/1/62 | rejected |
  | SXB − `BAG2` | +154.5 [+118.3, +191.9] | 142/4/55 | rejected |
  | KFB − `BAG3` | +89.4 [+55.1, +123.6] | 126/3/72 | rejected |
  | KFB − `BAG2` | +109.4 [+74.3, +145.0] | 135/3/63 | rejected |
  | `BAG3` − `BAG2` | +20.0 [−3.4, +44.1] | 98/38/65 | not distinguished (p 0.098) |

- **Against the locked recipe X0b** (stored, replayed exactly):

  | Entrant | RMSE cut | Sign-flip p | LOFO | Passes |
  | --- | --- | --- | --- | --- |
  | SXB | 11.37% (gain +0.121) | < 1e-6 | +0.068 | yes |
  | KFB | 11.56% (gain +0.123) | 9.5e-7 | +0.059 | yes |
  | `BAG3` | 6.91% | 1.4e-4 | +0.042 | yes |
  | `BAG2` | 1.79% | 0.034 | +0.016 | no |

- **TRAIN board Elo** (`p201b-train-audit-v1`, 19 stored entrants + 4 new; anchor `h335.H257_REPLICA` = 1500):

  | Entrant | Elo [95% CI] |
  | --- | --- |
  | `h416.SXB` | **1816.2** [1772.0, 1863.4] |
  | `h415.KFB` | 1771.1 [1729.7, 1815.4] |
  | `h410s.BAG3` | 1681.7 |
  | `h335.H335_AGG` | 1665.4 (the best stored entrant) |
  | `h410s.BAG2` | 1661.7 |

  The stored board ladder is reproduced exactly (max |Δ| 0).

**What this means.** The pooled population from H410/H415 is unchanged. What changes is how the ensemble is chosen:
1. Caruana weights averaged over 32 bootstrap resamples of the SEARCH rows (stability selection).
2. Deployed as the 50/50 mean of the fold-model bag and the full refit.

The result wins more TRAIN families than KFB's single cross-fit selection, and many more than the H410-style builders.
Elo rewards exactly that. The mean RMSE gain over X0b is the same as KFB's. KFB itself also separates cleanly from the
H410 builders on this fresh panel.

## Recommendation for the one dev91 score (owner decision; not spent)
**`h416.SXB`**. It is the top TRAIN-board entrant and is distinguished from KFB, BAG3 and BAG2 (Holm). It beats X0b by the
primary test (pre-registered rule §8: `SCORE`).

The alternative is KFB. It is level on mean RMSE and simpler, but it loses 62 of 201 families to SXB against 133 wins.
A dev91 once-score of SXB needs its own PREREG, LOCK and receipt (H369 pattern).

## Pre-registration and amendment
- [`PREREG.md`](PREREG.md):
  - part A at `23aee0953b`;
  - part B, the design chosen on P201e ([`DESIGN.json`](DESIGN.json), [`DESIGN_select.txt`](DESIGN_select.txt)), at
    `25d8c2d329`;
  - amendment 1 at `23474c1c55`.
  All were published before any P201b entrant value was read.
- **Design selection on P201e:** a grid of 17 designs (14 of 16 candidates eligible, per `DESIGN_select.txt`). `BCAR.mix` was chosen by
  S = min(win fraction vs KFB, vs `BAG3`) = 0.630. Its P201e Elo was 1702.9, against KFB's 1620.9.
- **Amendment 1 (§10).**
  - The P201b `gs_replay` gate failed on one context (`efeb5b1a…`). The deploy memo is the concatenation GS + GE + GL,
    and the engine's memo loader keeps the last entry per key. Two GS Phase T timeouts had completed in GE, so the
    GS replay diverged (Δu 8.0e-4).
  - The amendment admits a GS replay failure only on a context with a GS-involving memo conflict, and scores X0b there
    for all four new entrants. That gives up that context's information and cannot favour any new entrant.
  - The published conflict scan found exactly that one context among 201.
  - The sensitivity readout with the deploy's own values agrees: P1 +46.1 [+26.2, +66.1], SXB 1817.2.

## Integrity
- **Training data only.** No dev91, DL-Val, DL-Test or sealed byte was read, and no holdout model was rescored.
- **No stored entrant was rerun or rescored.** X0b is the stored locked recipe.
- **DOPE** `c51a56d0` is unchanged.
- **Gates, all passed:**
  - GE and GL replays exact (max |Δ| 0, members identical on 201/201);
  - GS exact on 200/201, with 1 amendment-1 exemption;
  - the audit envelope matches the board on 201/201;
  - reference predictions max |Δ| 4.0e-15;
  - `KCAR100.bag` equals KFB exactly;
  - the offline SXB re-derivation from the stored matrices max |Δ| 8.4e-10 (tolerance 1e-9, close to the limit);
  - no invalid entrant values.
- **Runs:**
  - the confirm deploy had 178,333 worker runs, 49 failures (H415: 48) and 0 timeouts;
  - the GE and GL searches had 14 and 39 search timeouts in about 67k runs each;
  - 4 contexts used the bag fallback.
- **Machine-readable results:** [`metrics.json`](metrics.json), projected from `results/H416_CONFIRM_P201B.a1.json`
  (in the closeout bundle) without re-estimation.
- **Archive:** [`ARCHIVE.json`](ARCHIVE.json). It is a create-only S3 bundle (2,383 members), verified by an
  exact-version GET and every member's hash. It holds:
  - the run records;
  - the pulled deploy matrices;
  - the result files;
  - the evidence directory.
- **Owner:** `cl_fwriter`, Claude Opus 5.5. Claim: `ns/receipts/H416_claim.txt`.
