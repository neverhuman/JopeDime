# H53b: extended learn-from-zero wave

**Status: complete; no arm admitted.** Six fresh CUDA cells ran in preregistered order with 20-minute wall caps. The timing-only H53 pilot and the failed CPU attempt were excluded from the six-cell outcomes. No adapter or optimizer state carried between cells.

## Question and protocol

H53b tested whether a freshly initialized 29,211-parameter adapter could learn useful ranking behavior from frozen DOPE without inheriting the r882 adapter. Cells used seeds 12, 13, and 14 at initialization scales 1.0 and 0.1. Each cell used the preregistered live-equivalent `k=2` AdamW update unit, `lr=1e-4`, batch size 1,219 and slate size 192, for up to 300 rounds or 20 minutes. The six-cell order was scale 1.0 seeds 12/13/14, then scale 0.1 seeds 12/13/14.

The 10-minute seed12/scale1.0 pilot (93 rounds) was timing-only, not a cell. The early CPU attempt was terminated as infra-void after one slow step and produced no endpoint. After the owner ruling, all six outcome cells required the exclusive qualification lock, at least 6 GiB free on CUDA, and no CPU fallback. The six cells used 105.6 minutes of cell wall time; cumulative lock wait was 47.7 minutes, excluded from the 150-minute compute budget.

The pinned cohort was 1,219 DL-Train contexts / 205 families and 91 DL-Val contexts / 19 families. The r882 harness family utility was 1.1536153. Frozen DOPE c51a56d0 was checksummed before and after each cell. H53b did not transfer the r882 adapter. No sealed/holdout payloads were opened and no outer rows were delivered.

## Results

Final-round family utility is the primary reported endpoint; higher is better. Mean gain and one-sided family LCB95 versus r882 are shown alongside Elo and W/T/L. Elo is a family-grouped pairwise diagnostic, **not AUC**. Ties fail. Each per-cell mean LCB95 is below zero, so none passes the owner admission rule.

| Scale / seed | Rounds / optimizer updates | Final family utility | Mean gain vs r882 | Mean LCB95 | Median gain [95% CI] | Elo delta | W/T/L | Peak read (round) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 / 12 | 121 / 242 | 1.098315 | -0.0553 | -0.0931 | -0.0098 [-0.0944, 0.0000] | -74.25 | 18/35/38 | 1.132931 (50) |
| 1 / 13 | 185 / 370 | 1.045452 | -0.1082 | -0.2228 | -0.0093 [-0.0970, 0.0032] | -36.71 | 28/21/42 | 1.168560 (75) |
| 1 / 14 | 153 / 306 | 1.083345 | -0.0703 | -0.1375 | -0.0145 [-0.0747, 0.0176] | -36.71 | 29/19/43 | 1.132379 (0) |
| 0.1 / 12 | 211 / 422 | 1.124595 | -0.0290 | -0.0674 | 0.0000 [-0.0199, 0.0132] | -21.97 | 20/45/26 | 1.141452 (175) |
| 0.1 / 13 | 300 / 600 | 1.095493 | -0.0581 | -0.1074 | -0.0098 [-0.0622, 0.0089] | -62.85 | 19/36/36 | 1.154228 (50) |
| 0.1 / 14 | 300 / 600 | 1.096619 | -0.0570 | -0.1090 | 0.0000 [-0.0115, 0.0072] | -21.97 | 24/37/30 | 1.134605 (125) |

The scale-level terminal utility and seed-gain medians are descriptive only; H53b preregistered per-cell outcomes, not a post-hoc scale winner. Both scale medians remain below r882: scale 1.0 median final utility 1.083345 (median cell mean gain −0.0703); scale 0.1 median final utility 1.096619 (median cell mean gain −0.0570). All six mean LCB95 values are negative.

### Matched horizon and unequal rounds

Wall-capped cells completed different numbers of rounds under changing GPU contention. Across all six cells, the median was 198 rounds and the minimum was 121 (61.1% of the median); no cell fell below the preregistered 60% infra-confounding threshold. R*=121, so the largest saved 25-round read at or below R* is round 100. This is diagnostic only and does not replace the final-round primary endpoint.

| Scale / seed | Rounds | DL-Val family utility at round 100 |
|---|---:|---:|
| 1 / 12 | 121 | 1.095957 |
| 1 / 13 | 185 | 1.154854 |
| 1 / 14 | 153 | 1.126308 |
| 0.1 / 12 | 211 | 1.127163 |
| 0.1 / 13 | 300 | 1.138697 |
| 0.1 / 14 | 300 | 1.130740 |

### Per-cell DL-Val trajectories

Every trajectory point below is a repeated DL-Val diagnostic. No best round was selected. The per-cell peak is reported for transparency only; a peak selected on DL-Val is inadmissible because it uses the validation read for selection. Full family-level mean/LCB and worst-five-family gains for each cell are in [`metrics.json`](metrics.json).

| Scale / seed | Round: family utility |
|---|---|
| 1 / 12 | 0: 1.091686; 25: 1.092344; 50: 1.132931; 75: 1.106626; 100: 1.095957; 121: 1.098315 |
| 1 / 13 | 0: 1.135149; 25: 1.106897; 50: 1.111677; 75: 1.168560; 100: 1.154854; 125: 1.081035; 150: 1.088821; 175: 1.079701; 185: 1.045452 |
| 1 / 14 | 0: 1.132379; 25: 1.131602; 50: 1.127858; 75: 1.095459; 100: 1.126308; 125: 1.049778; 150: 1.039737; 153: 1.083345 |
| 0.1 / 12 | 0: 1.107892; 25: 1.106269; 50: 1.126585; 75: 1.122250; 100: 1.127163; 125: 1.134398; 150: 1.102481; 175: 1.141452; 200: 1.127040; 211: 1.124595 |
| 0.1 / 13 | 0: 1.108001; 25: 1.107438; 50: 1.154228; 75: 1.145208; 100: 1.138697; 125: 1.126641; 150: 1.126406; 175: 1.129802; 200: 1.107694; 225: 1.110742; 250: 1.110689; 275: 1.106993; 300: 1.095493 |
| 0.1 / 14 | 0: 1.109198; 25: 1.131330; 50: 1.122373; 75: 1.103251; 100: 1.130740; 125: 1.134605; 150: 1.130262; 175: 1.093815; 200: 1.124543; 225: 1.129354; 250: 1.122434; 275: 1.124135; 300: 1.096619 |

The highest observed diagnostic was 1.168560 at round 75 (scale 1.0, seed 13). It is DL-Val-selected and **not admissible**. The valid comparator is the r882 harness utility 1.1536153. Any crossing of the live-loop target 1.1377942 is cross-harness and not comparable to this r882 harness; that field is omitted from admission and conclusions.

The separate timing-only pilot reached 1.136392 at round 93 and the fresh seed12/scale1.0 wave cell ended at 1.098315 at round 121. These are different runs; the pilot is excluded from wave statistics. Fresh adapter round-0 reads were 1.091686 / 1.135149 / 1.132379 at scale 1.0 and 1.107892 / 1.108001 / 1.109198 at scale 0.1. Taken descriptively, these support the cautious interpretation “adapter marginal value small; learn-from-zero overfits DL-Train” as utility trends down after roughly rounds 75–100 in these trajectories. This is not a win claim or an admissible selection signal.

### Lowest family gains

The five lowest family-gain observations across the six cells were:

| Scale / seed | Family ID | Gain vs r882 |
|---|---|---:|
| 1 / 13 | `fam-538e64066794582b82e32501f5ec76663b2f7945f24a5d985c9a0e204bac7426` | -1.034256 |
| 1 / 13 | `fam-9f67f1449b77152cbf20a6c82926f90496dacd7f05a222376edf0fdfb45f667f` | -0.689003 |
| 1 / 14 | `fam-7d4e2dc9f16cc0145caf3ff803564a61818542d6ab30dfe68270c4fe24b542e8` | -0.477500 |
| 1 / 14 | `fam-3a6cda83e147f955254920279f65461f01a0d30b2018d2bfe9b0d75aae0beae4` | -0.475717 |
| 0.1 / 13 | `fam-538e64066794582b82e32501f5ec76663b2f7945f24a5d985c9a0e204bac7426` | -0.458451 |

Per-cell integrity and access evidence:

| Scale / seed | DOPE SHA-256 before | DOPE SHA-256 after | Holdout payloads opened | Outer rows delivered |
|---|---|---|---:|---:|
| 1 / 12 | `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` | `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` | 0 | 0 |
| 1 / 13 | `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` | `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` | 0 | 0 |
| 1 / 14 | `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` | `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` | 0 | 0 |
| 0.1 / 12 | `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` | `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` | 0 | 0 |
| 0.1 / 13 | `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` | `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` | 0 | 0 |
| 0.1 / 14 | `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` | `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` | 0 | 0 |

## Interpretation and next work

No cell is admitted: all mean LCB95 values are negative, and ties fail. Positive Elo cannot override the negative family-utility means; final utility is the primary endpoint. The pilot and all round-by-round DL-Val reads are descriptive only.

For future H55/H57 designs, use round/generation-capped budgets and select any early stopping point using family-grouped inner cross-validation on DL-Train only. Never choose a stopping point from DL-Val. H55 remains gated on the H53b closeout PR being merged; H57 remains gated on H55 closing per its merged preregistration. H57 uses an r882 checkpoint copy and does not use the live loop.

## Evidence

The external raw cell summaries and wave receipt are under `~/jope-data/h53b/go-1935-gpu/`; the audit is `~/jope-data/h53b/go-1935-gpu/audit.json`. Compact metrics are [`metrics.json`](metrics.json) (23,469 bytes). Rebuild them with `python3 research/flywheel_selfplay/h53b/build_metrics.py ~/jope-data/h53b/go-1935-gpu`; validate the raw receipts with `python3 research/flywheel_selfplay/h53b/audit.py ~/jope-data/h53b/go-1935-gpu`.
