# assets/automl-elo/v2 — five-entrant provenance

Same 64-family panel as [v1](../v1/SOURCES.md). The four earlier entrants are not refitted.
`grok4p7_xhigh` is new. Bradley-Terry is fit again on all five, so these ratings are not the v1
numbers with one row appended.

| Rank | Entrant | Elo | 95% CI |
| --- | --- | ---: | --- |
| 1 | Opus5p5_Extra | 1681.3 | 1632.4–1736.7 |
| 2 | Gemini4_argon_high | 1602.7 | 1562.1–1647.9 |
| 3 | GPT6p1_Sol_Max | 1504.0 | 1464.3–1545.8 |
| 4 | GPT6_Astra_Max | 1393.2 | 1345.0–1436.6 |
| 5 | grok4p7_xhigh | 1318.8 | 1254.9–1374.6 |

- Numbers: [`RESULTS.json`](RESULTS.json). Pairwise family outcomes: [`match-matrix.txt`](match-matrix.txt).
- New cells: [`grok4p7-xhigh-cells.csv`](grok4p7-xhigh-cells.csv) and [`GROK_COSTS.json`](GROK_COSTS.json).

## Panel

- 64 families × 3 folds × 3 seeds (41, 73, 109) = 576 cells. Every cell completed. No cell was dropped.
- Family score: RMSE pooled over the 3 folds. Family match: mean of the three seed outcomes. Tie within relative 1e-12.
- Rating: historical `elo_core.rs`, centre 1500, 250 iterations, regularization 1e-4, the same 10,000 family-bootstrap index file as v1.
- Inputs: `/home/ubuntu/jope-evidence/regression-flywheel-20261021/newonly-run/inputs`, SHA-pinned by the v1 cell manifest. Archive `s3://veox-jopedime-learning-use2/staging/regression-flywheel-20261021/w000003/sha256/48065da046d63dcf8202299a6953731c994f35c11b94e96ca14c779dc5bc162e.tar.gz`.

## grok4p7_xhigh

- Source: `crates/jope-automl/src/grok4p7_xhigh.rs`
- SHA-256: `8e648cef93186aca9d10868ed1ed2a62407950169082aa1f4fab87af379664f8`
- Standalone `rustc --edition=2021 -C opt-level=3 -C codegen-units=1 -D unsafe-code` succeeded with no source edit.
- Worker SHA-256: `0ab9d7a38ea5f251d6ef2eca35ef71842f36c96b3b0714320d2fb708e7dc597b`
- Score binary SHA-256: `4a42ab6599e5145eaf46e4b4481a0dff9386f7bb339c7787bf9085fc020a2b6d` (the v1 scorer, unchanged)
- Envelope: one CPU, `CPUQuota=100%`, 8 GiB, no swap, no time cutoff, four cells at a time. A repeated first cell was byte-identical.
- Cost: 576/576 completed. Fit wall 8.19 s total, max single fit 0.199 s, peak memory 7.7 MiB. The panel tables are small (largest training file 1.8 MB). This timing is not comparable to Gemini's 1,986 s on xbabe2 beyond "both finished."

## How Grok compares

Geometric-mean RMSE ratio of the named entrant over `grok4p7_xhigh` (below 1 means that entrant has lower error). Holm adjustment is across Grok's four comparisons.

| Opponent | Ratio | 95% CI | Holm p |
| --- | ---: | --- | ---: |
| Opus5p5_Extra | 0.664 | 0.491–0.835 | 0.017 |
| Gemini4_argon_high | 0.755 | 0.665–0.849 | 0.0004 |
| GPT6p1_Sol_Max | 0.639 | 0.451–0.837 | 0.017 |
| GPT6_Astra_Max | 0.860 | 0.721–1.004 | 0.071 |

Grok is last. Opus, Gemini, and Sol have significantly lower pooled RMSE. Astra's advantage is not significant at 0.05 after Holm adjustment.
