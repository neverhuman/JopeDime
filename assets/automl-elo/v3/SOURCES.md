# assets/automl-elo/v3 — six-entrant provenance

Same 64-family panel as [v1](../v1/SOURCES.md). `kimik3p1_max` is new. No earlier entrant is
refitted: the three v1 historical entrants, `Gemini4_argon_high` and `grok4p7_xhigh` keep their
stored cells. Bradley-Terry is fit again on all six, so these ratings are not earlier numbers with
one row appended.

| Rank | Entrant | Elo | 95% CI |
| --- | --- | ---: | --- |
| 1 | Opus5p5_Extra | 1690.8 | 1642.4–1747.1 |
| 2 | Gemini4_argon_high | 1603.3 | 1561.6–1649.4 |
| 3 | **kimik3p1_max** | **1510.7** | 1468.7–1554.3 |
| 4 | GPT6p1_Sol_Max | 1497.5 | 1455.9–1541.8 |
| 5 | GPT6_Astra_Max | 1389.2 | 1339.4–1433.6 |
| 6 | grok4p7_xhigh | 1308.4 | 1240.3–1366.3 |

- Numbers: [`RESULTS.json`](RESULTS.json). Pairwise family outcomes: [`match-matrix.txt`](match-matrix.txt)
  (64 rows; 15 columns in `itertools.combinations` order of the sorted entrant names).
- New cells: [`kimik3p1-max-cells.csv`](kimik3p1-max-cells.csv) and [`KIMI_COSTS.json`](KIMI_COSTS.json).
- `grok4p7_xhigh` and the five-entrant v2 result come from the Grok lane's pull request
  (branch `feat/grok4p7-xhigh-automl-elo`), which was still open when this was computed. Its stored
  cells were reused unchanged (cell manifest SHA-256
  `ab6b0e0258676d6b02c54e99e2c5295f8e0f1e09102abafd4e9faceebf8fd998`).

## Panel and method

- 64 families × 3 folds × 3 seeds (41, 73, 109) = 576 cells per entrant. Every cell of every
  entrant completed. No cell was dropped.
- Family score: RMSE pooled over the 3 folds. Family match: mean of the three seed outcomes.
  Tie within relative 1e-12.
- Rating: historical `elo_core.rs` (SHA-256 `cc937890…3f16d9c`), centre 1500, the same 10,000
  family-bootstrap index file as v1 and v2 (SHA-256 `440d4c56…38f3633`).
- Analyzer: `analyze_six.py` extends the v2 `analyze_five.py` (SHA-256 `43237f33…b889c3e`) with no
  change to the cell reader, outcome rule or rating code. **It first recomputes the five-entrant
  v2 result from the stored cells and refuses to continue unless the text equals the published v2
  `RESULTS.json` exactly, and its five-entrant match matrix is byte-identical to v2's.** Both
  checks passed.
- The cells CSV and cost summary come from `export_cells.py`, which first regenerated the v2
  `grok4p7-xhigh-cells.csv` and `GROK_COSTS.json` byte-identically from Grok's stored cells.
- Inputs: the v1 panel inputs, SHA-pinned by the cell manifest (1,728 input files re-verified).
  Archive `s3://veox-jopedime-learning-use2/staging/regression-flywheel-20261021/w000003/sha256/48065da046d63dcf8202299a6953731c994f35c11b94e96ca14c779dc5bc162e.tar.gz`.

## kimik3p1_max

- Source: `crates/jope-automl/src/kimik3p1_max.rs`, supplied by the owner on 2026-11-07.
- SHA-256: `0c211694ec66605b994595fd8f4cf2b8894df6f156caa3b80f1fcc75f5d6ceb3` (47,156 bytes, 1,788 lines).
- Standalone `rustc --edition=2021 -C opt-level=3 -C codegen-units=1 -D unsafe-code` succeeded on
  the first attempt with no warnings and no source edit.
- Worker SHA-256: `46f0966ea15d4e8fc5698a09f752356cde0b590829e4032e6bd37693320b42f4` (the v2 worker
  with only the module, method and type names changed; a rebuild was byte-identical).
- Score binary SHA-256: `4a42ab6599e5145eaf46e4b4481a0dff9386f7bb339c7787bf9085fc020a2b6d` (the v1 scorer, unchanged).
- Cell manifest SHA-256: `c27900ce70d566164266078db27e402008543f9566e2358729685bcd3b245b2f`.
- Envelope: one CPU, `CPUQuota=100%`, 8 GiB, no swap, no time cutoff, four cells at a time. A
  repeated first cell was byte-identical.
- Cost: 576/576 completed. Fit wall 92.4 s total, max single fit 1.11 s, peak memory 14.2 MiB.
  As with v2, wall time on these small tables is not comparable across hosts beyond "finished".
- Archive (create-only; verified by an exact-version GET and every member's SHA-256):
  `s3://veox-jopedime-learning-use2/experiments/reference-board/external/automl-elo-v3/kimik3p1-max-20261107/kimik3p1_max_automl_v3_evidence.tar.zst`,
  version `8ogcrMeCZtjajBmeDRK3n3dVZo694DSL`, SHA-256
  `d8022fa8fb84baf466ead0c6d63b321e95e29190bc323d2571bd4b326d0a21ce`, 4,087 members.
  It holds the 576 Kimi cells (predictions, scores and logs), the worker and scorer builds, the
  analysis and export scripts, the edge-case probe, the source receipt and the files in this
  directory.

## How Kimi compares

Geometric-mean RMSE ratio of the named entrant over `kimik3p1_max` (below 1 means that entrant has
lower error), and the share of the 64 families Kimi wins (seed-averaged). Holm adjustment is across
Kimi's five comparisons.

| Opponent | Ratio | 95% CI | Holm p | Kimi family win share |
| --- | ---: | --- | ---: | ---: |
| Opus5p5_Extra | 0.793 | 0.626–0.933 | 0.084 | 0.208 |
| Gemini4_argon_high | 0.902 | 0.830–0.967 | 0.035 | 0.359 |
| GPT6p1_Sol_Max | 0.764 | 0.546–0.976 | 0.118 | 0.547 |
| GPT6_Astra_Max | 1.028 | 0.871–1.194 | 0.730 | 0.672 |
| grok4p7_xhigh | 1.195 | 1.067–1.343 | 0.010 | 0.797 |

- Kimi places third. Gemini has significantly lower pooled RMSE after Holm adjustment; Kimi has
  significantly lower RMSE than Grok.
- **Kimi and Sol are effectively level.** Kimi wins 55% of families against Sol, which puts its
  rating 13 points above Sol's, inside both intervals. Sol's geometric-mean RMSE is lower (ratio
  0.764) because its wins are larger; that difference is not significant after Holm adjustment.
- Opus's advantage is not significant at 0.05 after Holm adjustment. Neither is Kimi's against
  Astra.
