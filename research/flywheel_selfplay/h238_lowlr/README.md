# H238: lower-LR / longer-horizon AdamW — CLOSED_INCOMPLETE / CLOSED_INCOMPLETE

Hypothesis: smaller AdamW steps over the same or longer horizon yield adapters that the strict selector admits, improving DL-Val91 utility over r882.

Execution source `8c75d0d0dc54d4d89cf4b3f8153f7b74366da252`, prereg PR #2079. Offline r882 COPY; CPU xbabe2 cores 40–43, nice 19, one math thread. Only 47 adapter tensors / 29,211 parameters train. Seeds 238101–238103, fresh family fold seeds 238141–238143.

| Arm | LR | Updates | Selector steps |
|---|---:|---:|---|
| CONTROL | 1e-4 | 40 | 0/10/20/30/40 |
| LR3E5 (primary) | 3e-5 | 40 | 0/10/20/30/40 |
| LR1E5_120 (secondary) | 1e-5 | 120 | 0/30/60/90/120 |

All arms retain exact H170 CTRL AdamW betas, weight decay, loss, sampler and clip 1.0; the long arm shares the exact first 40 minibatches.

Native91 PASS: all91 values exact, mean **1.1377941898164525**, cached/native forward bit-identical; TRAIN smoke PASS with3 disposable updates and exact r882 restoration. The single primary was paused below20GiB MemAvailable, then the assigned <12GiB hard guard ended it (exit-15). Final observed available RAM12778967040bytes; peak own RSS3392147456bytes/3threads, disk789492039680bytes. This was an assigned memory-guard termination. No scientific retry.

Completed **3/15 cells**, with **600 updates in durable complete-cell traces**. Valid saved snapshots establish a lower bound of **640 primary updates**; the exact interrupted total remains **null**. All40 saved snapshots were readable/finite, with47 tensors/29211coordinates each. The final fourth-cell trajectory had no complete cell receipt.

Primary and secondary final utility effects, CI95, Holm verdict and Elo remain **null** because the wave is incomplete. No NO_ADVANCE/ADVANCE conclusion is drawn from the partial cells. The final in-memory DOPE observation is also null; the earlier Native91/TRAIN-smoke frozen-DOPE observation and raw source/audits are preserved.

Next step: retain the incumbent. This single attempt is consumed; any new computation needs a separately authorized wave.

Full output, source, starting COPY/reference and raw receipts: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261103/H238/res08-t0205/terminal-8c75d0d0dc54-0732a177296ab32d/H238-execution-output.zip` VersionId `42qhj0ABLcgcvhW.ZgRAEK8FmOVwBfEI`, SHA256 `0732a177296ab32d0bab9694ad0fec502142d52c8cf149fd9c7478bc096f6fc1`. Exact-version bytes and all 93 members replay PASS. Manifest `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261103/H238/res08-t0205/terminal-8c75d0d0dc54-0732a177296ab32d/H238-terminal-manifest.json` VersionId `MDw8.q2MS0DEphKo4.f9jMubMeorFg2v`. Raw roots retained: `/home/ubuntu/jope_plan/evidence/H238/res08-t0205` and `/home/ubuntu/jope-data/h238`. No scientific retry or serving promotion.

Protocol deviations: none. Load admission <=100, pause >108/resume <=100; RAM pause <20 GiB/resume >=22 GiB/hard stop <12 GiB, RSS <4 GiB, disk >=300 GB. Compute stop 07:00Z / closeout 07:30Z. Local targeted 14 data-free tests/Ruff repeated at unchanged source bytes, JSON/schema/links/hash/diff checks and head-bound same-agent review. Static coverage validated at preregistration. Hosted CI NOT_RUN/NOT_A_GATE and admin squash explicitly authorized by `hw_t0205_res08_h238_lowlr.md`.

Frozen protocol: [PREREG.md](PREREG.md).
