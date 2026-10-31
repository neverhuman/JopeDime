# H58 flywheel self-play defect isolation and ranking-loss ablation (t550)

## Hypothesis

Fixing the four structural defects from the H56 audit would let the `adapter_10tok` ranker learn positive family-grouped utility and Elo from a fold-clean initialization. The fixes are masking the revealed anchors (B), admitting all live GP evaluations as hard negatives (C), and a listwise Plackett-Luce plus margin loss (D). All three are tested against the control (A) and the untouched init (`INIT_G0`). The preregistration is in [`research/flywheel_selfplay/h58/PREREG.md`](../../../../../research/flywheel_selfplay/h58/PREREG.md).

## Setup

The wave ran 4 arms × 3 seeds (12, 13, 14). It used 5-fold family-grouped inner CV on DL-Train (205 families, 1,219 contexts, fold seed 4900), which gave 60 OOF cells. It also ran 12 terminal cells that were fit on all of DL-Train and read DL-Val (19 families, 91 contexts) exactly once per arm and seed, as preregistered. Each cell got 20 self-play rounds × 2 steps (40 optimizer steps). There were 2,880 live Rust two-fold CV evaluations with 0 failures. The wave ran on `cuda:0` under `qualification.lock` and finished at 02:12:48Z (8:12 PM MDT) after 5,530 s of wall time. DOPE `c51a56d0` was verified before and after all 72 cells. There were 0 excluded payloads and 0 outer rows. Peak host CPU was 57.7%.

## Outcome

On leak-free top-1 (`unrev`) utility, the family-cluster bootstrap used 10,000 draws. The t537 gate requires mean LCB95 > +0.001 and family median ≥ 0.

| Comparison | Split | Mean gain | Mean 95% CI | Family median | Gate |
|---|---|---:|---|---:|---|
| D vs A | DL-Train OOF (205 fam) | −0.3832 | [−0.4345, −0.3321] | −0.3595 | FAIL |
| D vs B | DL-Train OOF | +0.0017 | [−0.0010, +0.0051] | 0.0 | FAIL (tie) |
| D vs INIT_G0 | DL-Train OOF | +0.0020 | [−0.0021, +0.0067] | 0.0 | FAIL (tie) |
| D vs A | DL-Val (19 fam) | −0.3431 | [−0.4644, −0.2268] | −0.3858 | FAIL |
| C/D vs B | DL-Val | +0.0019 | [0.0, +0.0058] | 0.0 | FAIL (1/0/18 pos/neg/tie) |

Bradley-Terry Elo uses A = 1500 as the anchor, with family-grouped 95% CIs. On DL-Train OOF unrev, the Elo values were B 1305.5 [1277, 1334], C 1306.0, D 1305.7, and INIT_G0 1305.0, so every fixed arm sits about −194 Elo below A. On DL-Val unrev, they were B 1269.1, C/D 1272.3 [1151, 1359], and INIT_G0 1269.1.

Spearman ρ on the unrevealed pool rose from A 0.127 to D 0.180 on OOF, and from 0.120 to 0.181 on DL-Val. That is ranking-quality evidence, not admission evidence.

## Red-team notes (overseer, 2026-10-30 ~8:30 PM MDT)

- **Not a plateau test.** Every arm starts from a fold-clean init, not r882, and none is compared to r882. H58 cannot show progress against the 1.134–1.138 live validation plateau.
- **Headline metric.** The admission metric is top-1 utility, and on it B/C/D are far worse than A (about −194 Elo). The Spearman ρ gain is secondary and must not be read as a win.
- **Ties and pseudoreplicates.** C/D DL-Val unrev utility is 0.296669 for all three seeds. The +0.0019 C/D-vs-B gain comes from one family and has LCB95 = 0, so it is a tie and is correctly not admitted.
- **Post-hoc DL-Val use.** The `prior10` attenuation sweep in SUMMARY §7 (α ∈ {0, 0.02, 0.05, 0.1, 0.25, 1}) was computed on DL-Val after the terminal read and was not preregistered. It is exploratory only. Any follow-up must choose α by DL-Train inner CV and spend a single fresh terminal read.
- **Arm-A "native leak".** Picking among revealed anchors 0..3 inflates A's native utility by +0.072 OOF. Before reusing any native-mode number, check whether the live loop's validation metric counts revealed anchors the same way.
- **Loss scale.** Arm D's loss (~38) and mean gradient norm (7.0, clipped to 1.0) are about 8× arm C's. The arms are not matched on effective step size.
- **Integrity.** DOPE was frozen, there was no sealed, holdout or outer access, there were no OOMs, and CPU stayed under its cap. No violations were found.

## Verdict

No promotion. No H58 arm passes t537 against A, INIT_G0, or each other. The fixes are leak and hygiene corrections, not a utility gain. The useful mechanistic finding is that the unscaled `prior10` (ptp ≈ 18.2) swamps the W1 anchor interpolation term (ptp ≈ 1.78) in `gamenet_anchor`.

## Next

Keep fixes B/C/D in research code. A follow-up wave (H60 candidate) should treat `prior10` scaling α ∈ [0, 0.05], or `prior10` as an adapter feature only, as a preregistered arm. It should start from an r882 copy, be selected by DL-Train inner CV, take one terminal read, and be judged by paired family bootstrap vs r882 with a strict margin. H57 (ES) and H59 (robust loss) stay queued on `qualification.lock`.

Compact metrics are in [`metrics.json`](metrics.json). Full receipts are in `~/jope-data/h58/runs/` on xbabe2.
