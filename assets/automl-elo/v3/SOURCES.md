# assets/automl-elo/v3 — six-entrant provenance

Same 64-family panel as [v1](../v1/SOURCES.md) and [v2](../v2/SOURCES.md). Opus, Astra, Sol,
Gemini, and Grok are not refitted. `kimik3p1_max` is new. Bradley-Terry is fit again on all six,
so these ratings are not the v2 numbers with one row appended.

| Rank | Entrant | Elo | 95% CI | Lexical source tokens |
| --- | --- | ---: | --- | ---: |
| 1 | Opus5p5_Extra | 1690.8 | 1642.4–1747.1 | 14,706 |
| 2 | Gemini4_argon_high | 1603.3 | 1561.6–1649.4 | 17,417 |
| 3 | kimik3p1_max | 1510.7 | 1468.7–1554.3 | 9,528 |
| 4 | GPT6p1_Sol_Max | 1497.5 | 1455.9–1541.8 | 5,442 |
| 5 | GPT6_Astra_Max | 1389.2 | 1339.4–1433.6 | 5,924 |
| 6 | grok4p7_xhigh | 1308.4 | 1240.3–1366.3 | 9,401 |

- Numbers: [`RESULTS.json`](RESULTS.json). Pairwise family outcomes: [`match-matrix.txt`](match-matrix.txt).
- New cells: [`kimik3p1-max-cells.csv`](kimik3p1-max-cells.csv) and [`KIMIK_COSTS.json`](KIMIK_COSTS.json).
- Plot rows that keep this pool separate from the H-wave ladder: [`../../comparisons/protocol-elo/v2/`](../../comparisons/protocol-elo/v2/README.md).

## Panel

- 64 families × 3 folds × 3 seeds (41, 73, 109) = 576 cells. Every Kimik cell completed. No cell was dropped.
- Family score: RMSE pooled over the 3 folds. Family match: mean of the three seed outcomes. Tie within relative 1e-12.
- Rating: historical `elo_core.rs`, centre 1500, 250 iterations, regularization 1e-4, the same 10,000 family-bootstrap index file.
- Inputs: `/home/ubuntu/jope-evidence/regression-flywheel-20261021/newonly-run/inputs`. Archive `s3://veox-jopedime-learning-use2/staging/regression-flywheel-20261021/w000003/sha256/48065da046d63dcf8202299a6953731c994f35c11b94e96ca14c779dc5bc162e.tar.gz`.

## kimik3p1_max

- Source: `crates/jope-automl/src/kimik3p1_max.rs`, copied byte-for-byte from `/home/ubuntu/JopeDime/kimik3p1_max.rs`.
- SHA-256: `0c211694ec66605b994595fd8f4cf2b8894df6f156caa3b80f1fcc75f5d6ceb3`
- Standalone `rustc --edition=2021 -C opt-level=3 -C codegen-units=1 -D unsafe-code` succeeded with no source edit.
- Lexical counter `fe0c0771…`: 9,528 code tokens, 0 doc tokens, 47,156 bytes, 1,788 lines.
- Worker SHA-256: `14f6a99b5bf34063639d574c5163b824eaf973c4e1cc898a3c9a5938c281c867`
- Score binary SHA-256: `4a42ab6599e5145eaf46e4b4481a0dff9386f7bb339c7787bf9085fc020a2b6d` (the v1 scorer, unchanged)
- Envelope: one CPU, `CPUQuota=100%`, 8 GiB, no swap, no time cutoff, four cells at a time. The first cell, repeated, was byte-identical.
- Cost: 576/576 completed at `2026-11-08T01:14:11Z`. Fit wall 90.32 s total, max single fit 0.946 s, peak memory 16.7 MiB.
- Provider token fields are null. No authorship meter was attached to the supplied file.

## How Kimik compares

Geometric-mean RMSE ratio of the named entrant over `kimik3p1_max`. Below 1 means that entrant has lower error. Holm adjustment is across Kimik's five comparisons.

| Opponent | Ratio | 95% CI | Holm p |
| --- | ---: | --- | ---: |
| Opus5p5_Extra | 0.794 | 0.626–0.933 | 0.084 |
| Gemini4_argon_high | 0.902 | 0.830–0.967 | 0.035 |
| GPT6p1_Sol_Max | 0.764 | 0.546–0.976 | 0.118 |
| GPT6_Astra_Max | 1.028 | 0.871–1.194 | 0.730 |
| grok4p7_xhigh | 1.195 | 1.067–1.343 | 0.010 |

Kimik ranks third on family wins. Gemini has significantly lower pooled RMSE. Kimik has significantly lower pooled RMSE than Grok. Opus and Sol have lower pooled-RMSE point estimates that are not significant at 0.05 after Holm adjustment. Astra is not distinguishable from Kimik on pooled RMSE.
