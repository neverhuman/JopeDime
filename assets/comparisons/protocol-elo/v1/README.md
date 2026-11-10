# Protocol Elo points for later plots

Two Elo numbers in this directory answer two different questions. They share a conventional centre of 1500 and nothing else. Do not draw them on one axis, and do not subtract them.

| `protocol_id` | Question | Anchor |
| --- | --- | --- |
| `automl-64-family-rmse-bt-v2` | Which frozen regressor wins more of the 64 tabular families on RMSE? | The five-player mean, shifted to 1500 |
| `hwave-median-seed-vs-r882` | Did an H129 arm beat incumbent `r882`? | `r882` fixed at 1500 |

`points.csv` and `points.jsonl` contain the same entities and the same Elo values. The JSONL adds provenance. Blank `tokens` and `source_sha256` cells in the CSV are null. `protocol_id` is the facet column.

## Tokens

`tokens.json` is the full inventory for `grok4p7_xhigh`.

- `lexical_total_tokens` is 9401. That is 9380 code tokens plus 21 doc tokens from the pinned v1 counter on `grok4p7_xhigh.rs` only.
- Companion files are counted beside it and are not added in.
- Provider input, output, reasoning, cached, and billed totals are null. The session has no usage meter.

The same counter reproduces the v1 totals for Opus (14706), Astra (5924), and Gemini (17417) on the bytes in this commit. Sol's published 5442 belongs to blob `fe5bb21e…`, which is not in this clone. The Sol row keeps that published total and that blob id. The different file now at `crates/jope-automl/src/GPT6p1_Sol_Max.rs` was not scored and is recorded only in `tokens.json`.

## H-wave row

The strongest admitted arm under the registered `median_seed_elo` versus `r882` is `h129/BLEND_025` at 1523.48 (95% CI 1512.92–1532.76). `h129/PURE_GBM` has a higher point (1528.02) and harmed 38 of 205 families (18.5%), above the 10% cap, so it is not the admitted comparison. Ratings that beat a weak control, including H100 `ASAM10` at 1655.99, are a different contrast and are not in this table.
