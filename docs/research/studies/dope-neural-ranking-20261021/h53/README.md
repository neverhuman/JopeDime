# H53 learn-from-zero pilot

## Result

The six-cell pilot ran seeds 12/13/14 at fresh outer-adapter initialization scales 1.0 and 0.1, three rounds per cell. The candidate trainable adapter was independently initialized; no r882 outer-adapter parameters were transferred. The frozen architecture anchor and DOPE reference program remained fixed. All cells used the pinned family-grouped 1,219 DL-Train / 91 DL-Val cohort; the sealed holdout remained unopened.

| Init scale | Median best DL-Val utility [seed-bootstrap 95% CI] | Median Elo vs r882 [95% CI] | Median utility gap vs r882 [95% CI] | Cells reaching 1.1377942 |
|---:|---:|---:|---:|---:|
| 1.0 | 1.132379 [1.091686, 1.135562] | 1507.32 [1500.00, 1510.98] | -0.021236 [-0.061929, -0.018053] | 0/3 |
| 0.1 | 1.109267 [1.108316, 1.110789] | 1536.71 [1533.01, 1540.41] | -0.044348 [-0.045300, -0.042826] | 0/3 |

All per-round DL-Val values, Elo, and family gains are diagnostic-only. Every cell's best utility was below both the 1.1377942 target and r882's 1.153615 DL-Val baseline. Diagnostic family mean-gain LCBs were negative for both scales. The t537 mean-LCB/no-harm rule and former median-LB rule are **not evaluated** because this bounded pilot's repeated DL-Val endpoint is not an admission endpoint. **Verdict: no admission; do not extend these settings to a longer run.**

The 6m50s pilot and 6m34s integrity replay completed within the original 30-minute budget. Replay reproduced all six initializations, utility curves, and Elo scores, and verified that all six outer adapters changed after three AdamW updates. Frozen DOPE c51a56d0 matched before and after every cell; no holdout payload was opened.

See the experiment protocol and per-seed table in [`H53 SUMMARY`](../../../../../research/flywheel_selfplay/h53/SUMMARY.md). Compact metrics are 17,325 bytes. Full audit and cell/replay receipts remain external; no checkpoint or adapter weights are in Git.
