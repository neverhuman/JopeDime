# H239 — parameter-subset continuation of an r882 copy

**Status: COMPLETE. Primary BIAS_ONLY verdict: NO_ADVANCE.**

## Hypothesis

Restricting which adapter parameters move may reduce harmful family-level drift and yield a continuation admitted by the strict selector. This tests parameter membership, complementing learning-rate interventions; a drift/utility association alone does not establish causality.

## Setup

Preregistration [#2081](https://github.com/neverhuman/JopeDime/pull/2081), execution `1d31df4c235541c275b23ae7b9543ac3ec878ba5`. [Fixed protocol](PREREG.md), [tensor census](TENSORS.md), [exact membership](trainable_tensors.json).

CONTROL: 47 tensors/29,211 parameters; primary BIAS_ONLY: 23/755; secondary HEAD_ONLY: 2/25; secondary LLRD: 47/29,211 with fixed per-tensor depth multipliers. H170 CTRL AdamW/loss/batch/clip1.0 retained; 40 updates per arm/cell, candidate steps0/10/20/30/40, fresh seeds239101–239103/folds239141–239143. DOPE frozen. DL-Val91 is the primary panel; TRAIN OOF1219/205 is secondary.

## Outcome

| Arm versus r882 | Seed gains239101 /239102 /239103 | Median gain | Family CI95 | Positive seeds | Holm p | Verdict |
|---|---|---:|---|---:|---:|---|
| CONTROL | +0.0000000 / +0.0000000 / -0.0012229 | +0.0000000 | [+0.0000000, +0.0000000] | 0/3 | — | reference control |
| BIAS_ONLY | +0.0000000 / +0.0000000 / +0.0000000 | +0.0000000 | [+0.0000000, +0.0000000] | 0/3 | 1 | NO_ADVANCE |
| HEAD_ONLY | +0.0000000 / +0.0000000 / +0.0001097 | +0.0000000 | [+0.0000000, +0.0000000] | 1/3 | 1 | NO_ADVANCE |
| LLRD | +0.0001097 / +0.0000000 / -0.0005823 | +0.0000000 | [-0.0001392, +0.0003122] | 1/3 | 1 | NO_ADVANCE |

| Arm | DL-Val Elo (r8821500) [CI95] | Elo lift vs CONTROL [CI95] | Utility gain vs CONTROL [CI95] | Harm families | Median step40 L2 drift |
|---|---|---|---|---:|---:|
| CONTROL | 1500.000 [1500.000, 1500.000] | +0.000 [+0.000, +0.000] | 0 [0,0] | 0/19 | 0.3730565 |
| BIAS_ONLY | 1500.000 [1500.000, 1500.000] | +0.000 [+0.000, +0.000] | +0.0000000 [0.0, 0.0] | 0/19 | 0.0683994 |
| HEAD_ONLY | 1500.000 [1500.000, 1500.000] | +0.000 [+0.000, +0.000] | +0.0000000 [0.0, 0.0] | 0/19 | 0.0180233 |
| LLRD | 1500.000 [1489.036, 1507.308] | +0.000 [-3.655, +10.964] | +0.0001097 [-4.6415532165982276e-05, 0.00046835313233083663] | 0/19 | 0.1332109 |

The strict selector retained r882 in all 15 BIAS_ONLY cells. It admitted one CONTROL cell (step 20), one HEAD_ONLY cell (step 40), and two LLRD cells (steps 10 and 20). Thus the BIAS_ONLY zero gain reflects fallback to r882, not proof that its trained candidates are equivalent. The other arms also mostly fell back to r882.

TRAIN OOF (secondary; family-grouped):

| Arm versus r882 | Seed gains | Median [CI95] | Elo [CI95] | Harm families |
|---|---|---|---|---:|
| CONTROL | [0.0, 0.0, -0.0018939110577610857] | +0.0000000 [0.0, 0.0] | 1500.000 [1500.0, 1500.0] | 4/205 |
| BIAS_ONLY | [0.0, 0.0, 0.0] | +0.0000000 [0.0, 0.0] | 1500.000 [1500.0, 1500.0] | 0/205 |
| HEAD_ONLY | [0.0, 0.0, -0.0024408945101234668] | +0.0000000 [0.0, 0.0] | 1500.000 [1500.0, 1500.0] | 1/205 |
| LLRD | [-0.0001759683505588228, 0.0, -0.002446064601832566] | -0.0001760 [-0.00044113081764367567, 0.0] | 1499.369 [1498.5687251737145, 1500.0] | 2/205 |

DL-Val context-mean utility (descriptive comparison with historical plateau best1.1389572):

| Arm | Seed239101 | Seed239102 | Seed239103 | Median context mean minus plateau |
|---|---:|---:|---:|---:|
| CONTROL | 1.137794190 | 1.137794190 | 1.136517573 | -0.001163010 |
| BIAS_ONLY | 1.137794190 | 1.137794190 | 1.137794190 | -0.001163010 |
| HEAD_ONLY | 1.137794190 | 1.137794190 | 1.137908714 | -0.001163010 |
| LLRD | 1.137908714 | 1.137794190 | 1.137186338 | -0.001163010 |

The plateau comparison is descriptive: the decision uses family-weighted paired utility against r882 under the registered protocol. Elo is family-weighted Bradley–Terry with r882 anchored at1500. Harm means a three-seed mean family loss below−0.01. All three seeds are reported; identical fallback predictions are disclosed in [metrics.json](metrics.json) and do not constitute independent replication.

All 15 cells completed: 2,400 logical optimizer updates. Recorded physical study updates across invocations: 2400; smoke updates are counted separately per invocation. Frozen-subset snapshot hashes and all saved selector/utility/Elo panels replayed successfully after termination.

## Verdict and next step

Primary BIAS_ONLY and both secondary treatments are **NO_ADVANCE** under the registered rule. All treatment arms reduced median step-40 drift relative to CONTROL, but this did not produce a qualifying utility gain. No live promotion was performed.

Do not promote or repeat this fixed continuation. Any follow-up needs a separate preregistration and distinct hypothesis; these results are conditional on this checkpoint, corpus and strict selector.

## Resources, boundaries and custody

Native91: {'status': 'PASS', 'matches': 91, 'mean': 1.1377941898164525}. CPU only on xbabe2, cores48–51, nice19, one math thread. A1 was merged before training: admission30GiB, cooperative pause below20GiB, recovery at25GiB for30s; hard stop below12GiB for30s or below6GiB immediately. RSS4GiB, disk300GB and absolute07:30Z limits retained.

The primary process completed in 4,892.8 seconds, with four cooperative pause/resume pairs and no process restarts. Pause requests lasted 1,284.0 seconds in total. Peak sampled process-tree RSS was 3,351,130,112 bytes (3.12 GiB); peak threads were 3. The supervisor exited 0 with status PASS and no stop reasons. The complete resource timeline and summary remain in the archive; no speed inference is drawn from elapsed time.

No sealed/holdout/meta-test or live-loop access. FIT/SELECT/ASSESS families and TRAIN/DL-Val disjointness were audited. These exclusions govern H239 continuation, not the historical training or selection of r882; DL-Val is the exposed development panel. Excluded adapter coordinates are bit-identical in completed snapshots; DOPE and non-adapter checks ran after each update. Four disposable TRAIN-smoke updates precede each process invocation and restore r882 exactly. Resume costs and every pause/stop remain in the full receipts. No AUC or speed claim.

Full runtime archive: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261103/H239/astra-t0235/closeout/runtime.tar.gz`, VersionId `OMus8Jyv45.ClC4oCoL8vGsqn0h3AZrN`, SHA256 `43725a839bcd17e8326db7713a19681e85bb42f617b11c0f70a5ae1f74feebe5`.

Archive manifest: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261103/H239/astra-t0235/closeout/manifest.json`, VersionId `S2wtI9VmumcqiWz976BnQhaDIbJeJ5Lz`, SHA256 `9bd01c59a1936d29c3f8b174696ddd391a904b469dda019283c3afcf18618ae5`. All archive members were recovered and hashed. Small metrics and source/custody manifest are committed; full data remains in S3.

Local targeted tests, Ruff, records, source/index pins, JSON/schema and diff checks accompany the exact-head review. Full remote CI is NOT_RUN under the explicit owner local-check exception, not reported PASS. PR#2071 remains untouched.
