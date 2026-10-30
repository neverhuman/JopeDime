# H53 pilot summary — learn-from-zero

## Hypothesis

A freshly initialized 29,211-parameter outer adapter can learn useful ranking behavior from the frozen DOPE reference program without inheriting r882 adapter parameters. The frozen architecture anchor remains part of the scaffold; only the outer trainable adapter was reset and trained. r882 was scored as a separate comparator, and its outer trainable adapter was discarded before candidate training.

## Arms and protocol

Six preregistered cells: seeds 12, 13, and 14 at initialization scales 1.0 and 0.1. Each ran three rounds with one AdamW update per round on family-balanced DL-Train batches. DL-Val utility was scored at rounds 0–3 for the requested curve and was diagnostic-only; no seed, scale, or checkpoint was selected. The pinned split was 1,219 DL-Train contexts / 205 families and 91 DL-Val contexts / 19 disjoint families. No outer or sealed/holdout payload was opened.

The pilot finished in 6m50s within its 30-minute cap. A separate 6m34s deterministic integrity replay reproduced all six initialization hashes, per-round utilities (within 1e-9), and Elo scores, and verified three optimizer updates changed each trainable adapter. Both runs fit before the original 18:03:21Z deadline. DOPE c51a56d0 checksum matched before and after every cell. Peak sampled worker RSS stayed under the 6 GiB cap; CPU, memory, and GPU preflights passed.

## Per-seed metrics

All utility and Elo values below are DL-Val diagnostics. The target is utility 1.1377942; the r882 baseline DL-Val utility was 1.153615.

| Init scale | Seed | Best DL-Val utility | Best round | Round to 1.1377942 | Gap to target | Gap vs r882 | Elo vs r882 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 12 | 1.091686 | 0 | not reached | -0.046108 | -0.061929 | 1507.32 |
| 1.0 | 13 | 1.135562 | 2 | not reached | -0.002232 | -0.018053 | 1510.98 |
| 1.0 | 14 | 1.132379 | 0 | not reached | -0.005415 | -0.021236 | 1500.00 |
| 0.1 | 12 | 1.109267 | 3 | not reached | -0.028527 | -0.044348 | 1540.41 |
| 0.1 | 13 | 1.108316 | 2 | not reached | -0.029478 | -0.045300 | 1536.71 |
| 0.1 | 14 | 1.110789 | 3 | not reached | -0.027005 | -0.042826 | 1533.01 |

| Init scale | Median best utility [seed-bootstrap 95% CI] | Median Elo [seed-bootstrap 95% CI] | Median gap vs r882 [seed-bootstrap 95% CI] | Diagnostic family mean [one-sided LCB95] | Diagnostic family median [95% CI] |
|---:|---:|---:|---:|---:|---:|
| 1.0 | 1.132379 [1.091686, 1.135562] | 1507.32 [1500.00, 1510.98] | -0.021236 [-0.061929, -0.018053] | -0.033740 [-0.096558] | -0.003986 [-0.013365, 0.017116] |
| 0.1 | 1.109267 [1.108316, 1.110789] | 1536.71 [1533.01, 1540.41] | -0.044348 [-0.045300, -0.042826] | -0.044158 [-0.117715] | 0.000000 [-0.004580, 0.029818] |

Seed bootstrap intervals resample three seed-level observations; they are descriptive and have very low replication. The family estimates pool 19 DL-Val families across the three seeds and are also diagnostic only. Elo is anchored at r882=1500 using the family-grouped Bradley-Terry tracker. Positive Elo does not override the negative utility gaps.

## Verdict

No pilot cell reached 1.1377942, and every seed's best utility was below the r882 DL-Val baseline. The 0.1x scale had a higher median Elo, but both scales had negative mean family-gain diagnostics versus r882. Since these are repeated DL-Val reads and are explicitly diagnostic-only, neither the t537 mean-LCB/no-harm rule nor the prior median-LB rule is evaluated by this pilot. **No admission.**

Do not extend these H53 settings to a longer run based on this pilot. H54 fold-clean baseline remains proposal-only. No model checkpoint or adapter weights are committed.

## Evidence

Compact metrics are in `docs/research/studies/dope-neural-ranking-20261021/h53/metrics.json` (17,325 bytes). Full audit and six cell summaries are external at `~/jope-data/h53/runs/audit.json` and `~/jope-data/h53/runs/scale-{1,0.1}/seed{12,13,14}/summary.json`. Integrity replay receipts are external under `~/jope-data/h53/replay/`. The compact metrics include pinned identities, all per-seed outcomes, seed-bootstrap intervals, admission disposition, and replay parameter-change proof.

## Per-round DL-Val curves (diagnostic-only)

| Init scale | Seed | Round 0 | Round 1 | Round 2 | Round 3 |
|---:|---:|---:|---:|---:|---:|
| 1.0 | 12 | 1.091686 | 1.091686 | 1.091686 | 1.091686 |
| 1.0 | 13 | 1.135149 | 1.135149 | 1.135562 | 1.135562 |
| 1.0 | 14 | 1.132379 | 1.132379 | 1.132379 | 1.132379 |
| 0.1 | 12 | 1.107892 | 1.108417 | 1.108158 | 1.109267 |
| 0.1 | 13 | 1.108001 | 1.108291 | 1.108316 | 1.108316 |
| 0.1 | 14 | 1.109198 | 1.109366 | 1.108505 | 1.110789 |

These curves were repeated diagnostic reads, not a checkpoint-selection signal. Round 0 is the fresh random adapter before optimizer updates.
