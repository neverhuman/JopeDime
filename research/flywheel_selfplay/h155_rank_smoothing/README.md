# H155 — pairwise smoothing and TRAIN-fold margin calibration

**Verdict: NO_ADVANCE.** Exploratory conditional continuation; no serving or sealed-test qualification.

Native r882 baseline: **91/91 exact**, context mean **1.1377941898164525**; full H145 TRAIN context mean **1.423073434552** (1219 contexts/205 families).

Five fixed arms, seeds155101–155103, 40 AdamW adapter updates/cell: **15/15 cells,600 adapter updates**. Two TRAIN-only scalar temperature fits:200 updates each. Frozen DOPE **c51a56d0** and all non-adapter state unchanged;29211 trainable adapter parameters/47 tensors. CPU only, no speed gate.

## 91-context exposed development panel and full TRAIN panel

Elo is equal-family win/tie/loss versus r882=1500, with Jeffreys0.5 correction. Values below are medians across3 seeds. Full TRAIN scores are exposed/descriptive.

| Arm | 91ctx mean utility | 91ctx Elo | 91ctx family gain vs r882 [95% CI] | 1219ctx mean utility | 1219ctx Elo |
|---|---:|---:|---|---:|---:|
| CTRL | 1.092180013 | 1352.809 | -0.043585036 [-0.110879015,-0.003267062] | 1.388444936 | 1445.589 |
| LS05 | 1.092180013 | 1352.809 | -0.043585036 [-0.110879015,-0.003267062] | 1.388290417 | 1443.859 |
| LS10 | 1.092180013 | 1352.809 | -0.043585036 [-0.110879015,-0.003267062] | 1.388009294 | 1443.859 |
| LS05_TEMP | 1.090052518 | 1352.809 | -0.045622951 [-0.112191456,-0.003267062] | 1.388762360 | 1443.859 |
| LS10_TEMP | 1.090052518 | 1352.809 | -0.045622951 [-0.112191456,-0.003267062] | 1.388762360 | 1443.859 |

## Prespecified inner TRAIN evaluation

241 contexts/41 families; calibration247/41; adapter fit731/123. These family sets are disjoint, but r882 previously trained on all1219 contexts. This is not an untouched out-of-family generalization estimate.

| Arm | Elo vs r882 | Family gain vs r882 [95% CI] | Family gain vs CTRL [95% CI] | Holm p vs CTRL | Promising |
|---|---:|---|---|---:|---|
| CTRL | 1516.557 | +0.018998273 [-0.007872378,+0.050001719] | +0.000000000 [+0.000000000,+0.000000000] | n/a | control |
| LS05 | 1516.557 | +0.018998273 [-0.007872378,+0.050001719] | +0.000000000 [+0.000000000,+0.000000000] | 1 | False |
| LS10 | 1516.557 | +0.018998273 [-0.007872378,+0.050001719] | +0.000000000 [+0.000000000,+0.000000000] | 1 | False |
| LS05_TEMP | 1508.274 | +0.018619575 [-0.007793572,+0.049903717] | -0.000035296 [-0.000105889,+0.000002266] | 1 | False |
| LS10_TEMP | 1508.274 | +0.018619575 [-0.007793572,+0.049903717] | -0.000035296 [-0.000105889,+0.000002266] | 1 | False |

## Per-seed comparison

| Arm | Seed | 91ctx utility | 91ctx Elo | TRAIN utility | TRAIN Elo |
|---|---:|---:|---:|---:|---:|
| CTRL | 155101 | 1.089149590 | 1352.809 | 1.388753724 | 1445.589 |
| CTRL | 155102 | 1.112406144 | 1352.809 | 1.386740729 | 1436.912 |
| CTRL | 155103 | 1.092180013 | 1411.261 | 1.388444936 | 1445.589 |
| LS05 | 155101 | 1.089149590 | 1352.809 | 1.388700789 | 1445.589 |
| LS05 | 155102 | 1.112406144 | 1352.809 | 1.386740729 | 1436.912 |
| LS05 | 155103 | 1.092180013 | 1411.261 | 1.388290417 | 1443.859 |
| LS10 | 155101 | 1.089149590 | 1352.809 | 1.388714590 | 1445.589 |
| LS10 | 155102 | 1.112406144 | 1352.809 | 1.386789492 | 1436.912 |
| LS10 | 155103 | 1.092180013 | 1411.261 | 1.388009294 | 1443.859 |
| LS05_TEMP | 155101 | 1.089149590 | 1352.809 | 1.391061618 | 1443.859 |
| LS05_TEMP | 155102 | 1.112320667 | 1352.809 | 1.388762360 | 1436.912 |
| LS05_TEMP | 155103 | 1.090052518 | 1392.462 | 1.388383682 | 1447.316 |
| LS10_TEMP | 155101 | 1.089149590 | 1352.809 | 1.391061618 | 1443.859 |
| LS10_TEMP | 155102 | 1.112320667 | 1352.809 | 1.388762360 | 1436.912 |
| LS10_TEMP | 155103 | 1.090052518 | 1392.462 | 1.388383682 | 1447.316 |

## Calibration, provenance and verification

Learned temperatures: eps0.05:T=20.000000000, eps0.1:T=20.000000000. A positive scalar temperature cannot change an inference argmin; the treatment changes training gradients.

All276 dataset IDs independently join versioned S3 catalog v246 with origin=real;0 synthetic training datasets. All1319 numerical inputs/pools/model assets rehashed against the immutable H144 archive manifest.12 runtime source files excluded by that historical archive are separately pinned. Original H50/H58 derivative object versions remain unestablished; the exact numerical closure is preserved by the pinned H144 S3 archive.

Independent audit PASS:19650 utility recounts from actual pool labels/choices,30 family/Elo/bootstrap contrasts,15 adapter hash/inventory checks,30 native argmin model spot replays. Protected reads0, CUDA initialization0, input/checkpoint hashes unchanged. All fixed cells retained; no adaptive selection, retries or foreign outcome reuse.

Preregistration #1309; executor #1310; reserved-block disk-guard repair #1313. The red-team guard defect was repaired and independently cleared before any scoring/calibration/adapter update. Nine CPU tests, Ruff check/format and corrected-source zero-evaluation smoke passed. Exact merged executor: dc1faae7bc16f7c60287eb50a79aeca7e7c7c6ea.

Source/input/fold manifest and complete native cell evidence are bound in the custody archive. See [PREREG.md](PREREG.md), [metrics.json](metrics.json), [independent-audit.json](independent-audit.json) and [S3_CUSTODY.json](S3_CUSTODY.json). No live flywheel/pinned checkpoint change or serving admission.
