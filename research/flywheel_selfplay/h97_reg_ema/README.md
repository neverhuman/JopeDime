# H97 — Adapter regularization and final EMA candidate

**Verdict: NO ADOPTION.** H97 completed all 12 preregistered outcome cells. EMA improved the equal-family dev utility over same-seed CONTROL in all three seeds, but its lower confidence bound did not reach the strict preregistered admission margin. Every arm's median Elo point estimate versus fresh r882 remained below 1500. These results do not break the r882 plateau or qualify the live learner.

## Hypothesis and setup

H97 tested whether stronger AdamW weight decay, train-only dropout on adapter hidden activations, or a final parameter EMA could improve the r882 adapter under the unchanged 40-update budget. Every cell started from its own verified r882 copy. Arms were CONTROL, WD (`weight_decay=1e-2`), DROP (`p=0.1`), and EMA (`decay=0.9`), with seeds 33–35.

The outcome worker used the preregistered H50 packet: 1,219 DL-Train contexts and the 91-context / 19-family dev packet. The dev packet was read for H97's registered scoring. This is not a zero-dev-read study. The worker made no separate terminal validation/admission read; receipts report zero holdout payloads opened and zero outer rows delivered. The pinned protocol excludes sealed, holdout, meta-test, outer-test, live-loop payloads and prior H76/H82/H93 outcomes.

All 12 cells completed 40 updates, with 29,211 trainable values across 47 adapter-scoped parameter names. Each cell's initial/final adapter digest differed. All timing/outcome receipts recorded the frozen DOPE SHA-256 `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` before and after. The fixed optimizer and scoring protocol is in [PREREG.md](PREREG.md).

The timing cell projected 929.86 seconds for the full panel. Total charged GPU cell wall was 825.93 / 2,700 seconds. DROP seed 33 attempt 1 stopped after a pre-scoring scratch collision and remained charged; the corrected physical attempt 2 completed. The failed log, scratch, phase, and ledger entries remain preserved. CONTROL and WD seed 33 and the timing cell were not rerun.

## Outcomes

The primary endpoint is median per-seed equal-family `val_reg` gain versus same-seed CONTROL. Intervals use the preregistered 10,000 common bootstrap draws over 19 paired dev families. Seeds are not independent replicates; they are kept as complete seed vectors, while the interval unit is the 19 dev families.

| Arm | Median gain vs CONTROL | 95% family CI | LCB95 | Elo vs CONTROL (median [95% CI]) | Elo vs fresh r882 (median [95% CI]) | Median adapter L2 from r882 | Admitted |
|---|---:|---:|---:|---:|---:|---:|---|
| CONTROL | 0.00000 | [0.00000, 0.00000] | 0.00000 | 1500.0 [1500.0, 1500.0] | 1321.1 [1128.2, 1444.7] | 0.44668 | No |
| WD | 0.00000 | [0.00000, 0.00000] | 0.00000 | 1500.0 [1500.0, 1500.0] | 1321.1 [1128.2, 1444.7] | 0.44651 | No |
| DROP | -0.00050 | [-0.00374, 0.00084] | -0.00318 | 1500.0 [1444.7, 1536.7] | 1344.0 [1172.2, 1463.3] | 0.43105 | No |
| EMA | **+0.02619** | [+0.00015, +0.07505] | **+0.00045** | **1536.7 [1481.7, 1634.3]** | **1425.7 [1296.7, 1536.7]** | **0.34641** | **No** |

EMA's three seed gains were +0.02713, +0.02555 and +0.02619. Its Bonferroni-adjusted LCB was `7.8565e-6`, also far below the `0.001546` admission margin. The positive EMA Elo versus CONTROL does not override the family-utility gate. EMA's median Elo versus r882 is 1425.7; its interval includes 1500, so this is a lower point estimate, not decisive evidence of harm versus r882.

WD was numerically indistinguishable from CONTROL on the primary endpoint and Elo. Their median adapter L2 values differed by only 0.000174. The short 40-update run and observed near-zero separation do not provide an informative test of the intended stronger-weight-decay effect; do not interpret it as a validated WD recipe.

The full compact metrics are in [metrics_summary.json](metrics_summary.json). Per-cell score receipts, family vectors, adapter snapshots and the charged ledger remain in external H97 custody, not Git.

## Next step

Do not adopt H97 weights or settings. The next stated hypothesis is shrinkage/interpolation toward r882, preregistered as H105 by `sol_v1_dsc08`. H97 provides descriptive evidence that EMA moved the adapter closer to r882 while improving relative to CONTROL, but it did not pass admission and all arms' median Elo point estimates stayed below r882.
