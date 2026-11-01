# H189 — SWA and EMA on one r882 continuation

**Status: complete. Verdict: `NO_ADVANCE`.** H189 is diagnostic research only; it did not change the live flywheel or promote a checkpoint.

## Question and design

H189 asked whether plain weight averaging (SWA80) or an adapter-only EMA (EMA80) can recover held-out family utility from one 80-update r882 continuation. CTRL80 is the step-80 state. The study used 1,219 TRAIN contexts from 205 families, five outer folds, seeds 189101–189103, and 15 trajectories / 1,200 optimizer updates. The exact native r882 baseline passed bit-for-bit on 91/91 DL-Val contexts (mean utility 1.1377941898164525) before training.

**All three seeds share one family-fold assignment generated with seed 189140.** Seed-to-seed intervals reflect training noise and the family bootstrap, not fold-assignment uncertainty; these are not independent fold replicates. **CTRL80, SWA80, and EMA80 are paired transformations of one trajectory per fold/seed**, not independent training runs.

## Result

The pooled B selector admitted neither candidate. On pooled A versus R882, both candidates had negative family-utility lift, negative 95% lower bounds, and Holm-adjusted one-sided bootstrap p-values of 1.0 under the red-team closeout rule (`LCB95 > 0` and Holm `p < 0.05`, multiplicity 2). A tie is `NO_ADVANCE`.

| Candidate | Median family-utility lift vs R882 | 95% family-bootstrap CI | Red-team Holm p (m=2) | Elo lift vs R882 (95% CI) | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| SWA80 | -0.043815 | [-0.077340, -0.015170] | 1.0000 | -24.09 [-36.17, -11.64] | NO_ADVANCE |
| EMA80 | -0.043253 | [-0.076989, -0.014008] | 1.0000 | -23.99 [-35.74, -11.02] | NO_ADVANCE |

The supplemental A p-values are computed from the same preregistered 10,000-draw family-bootstrap distributions; they do not change the preregistered test or outcome. The pooled B admission gate also failed for both arms. The selected policy therefore remains R882 at 1500 Elo. No AUC or speed metric is reported.

## Integrity and resume record

The final guarded run completed with monitor `PASS`: 1,200 optimizer updates, B lock fsynced before A scoring, 0 protected reads, and no live-flywheel write. The final cumulative guarded elapsed time was 6,305.35 seconds. Two disk-guard stops are preserved in the resource log; the worker resumed only after `/` fell below 93.8%. An earlier attempt-2 restart failed on an incomplete six-file resume binding after reusing 240 saved FIT updates; it opened no B/A data. All 15 FIT cell states were subsequently pinned and reused successfully.

Five resume events are recorded in the archived `H189_resume_ledger.json`. Each launch receipt hash and checkpoint-binding size is listed here; the ledger carries every individual checkpoint SHA-256 and is itself bound by the digest in [`custody.json`](custody.json).

| Event | Launch receipt SHA-256 | Bound checkpoint hashes | Outcome |
| --- | --- | ---: | --- |
| resume1 baseline-boundary attempt | `e46ed7b1a90e8fae2ceb2b0e4dc8f2d162b6ede0dc55ca69ea53289e79659dc3` | 6 | Failed before new FIT updates |
| resume2 initial | `72466d3158480dafa4277e5c7e25abeba0f5e0607e44dbba9e8f781196ad24fe` | 6 | Continued FIT/B; disk guard stopped it |
| resume2 stale restart | `72466d3158480dafa4277e5c7e25abeba0f5e0607e44dbba9e8f781196ad24fe` | 6 | Failed after reusing 240 updates; no B/A opened |
| resume2 recheckpoint | `9798004f384bd86cd719e9abce9554c0ce40fee4f14666f3c252b761f63ffc0a` | 30 | Reused FIT; disk guard stopped B recomputation |
| resume2 recheckpoint final | `9798004f384bd86cd719e9abce9554c0ce40fee4f14666f3c252b761f63ffc0a` | 30 | B lock and A completed; monitor PASS |

The monitor log concatenates JSON objects without newline separators, so it counted the recheckpoints as generation 2; this bookkeeping is disclosed in the ledger. `H189_attempt2_resume4_launch.json` was only a draft and was never executed.

## Evidence and custody

The complete run, source snapshot, launch/review receipts, five-event resume ledger, B lock, raw B cost vectors, A/B panels, and audit are in the versioned S3 archive. Exact-version readback matched the archive and manifest SHA-256 values, and all 126 archive members passed hash verification.

- Archive: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H189/H189_attempt2_20261101T1743Z/H189-evidence.zip`, VersionId `7RF8XZyEi3VCQNSbYAvTRWgA1QH4uPK4`, SHA-256 `8fc229fea1062251165f770328125ed301571e84760cf6006bc2acd2325798c`.
- Manifest: same prefix, `H189-manifest.json`, VersionId `sFyKpzaVuw4rYqtVJeg6PV.IycfOUX2O`, SHA-256 `c014a777e1b310e399e894fca64ed6e94fa514ecea4e37629b33e94c398a4cbc`.

The compact committed result is [`metrics.json`](metrics.json); full panels and raw scoring evidence remain outside Git. No serving change is authorized.
