# H138 — retained-trajectory SWA/EMA

**COMPLETE / NO_ADVANCE.** No serving promotion or automatic retry.

Hypothesis: averaging the flywheel's retained adapters might improve r882 without
further gradient steps. [PREREG.md](PREREG.md) fixed SWA4_UNIFORM (782/785/859/882),
SWA2_LATE (859/882) and EMA_0.5 before execution. Average only 47 named adapter
parameter tensors / 29,211 parameters; all other state and buffers come from r882.
[manifest.json](manifest.json) preserves four full checkpoint SHA-256 pins.

Source: `198053105abacf632837414252bb9d2d49a02bee`, centrally merged PR #1243. No optimizer updates,
GPU, qualification lock, live-flywheel writes or protected-data reads.
The prelaunch cache-complement fix and its reproducing synthetic test were merged
before the scored attempt; original pre-fix commit 094697670 remains in Git history.
Fourteen data-free tests and five records gates passed before launch. Full CI was
deferred under the owner targeted-local policy; no hosted-CI pass is claimed.

Baseline r882 reproduced all 91 context utilities exactly, mean 1.1377941898164525,
evaluation identity `e79a7715c1531ccb8c4055ab96971101cd4034ec3a71d83bca6d25b0e604ca83`.
Primary used 1,219 DL-Train contexts / 205 families. Higher native utility is better.

| Arm | Median partition gain | Family 95% CI | Holm adjusted p | Harmed/205 | Elo (r882 = 1500) | Pass |
|---|---:|---|---:|---:|---:|---|
| SWA4_UNIFORM | +0.003595854 | [-0.000393965, +0.010193993] | 0.383962 | 12 | 1503.389 | False |
| SWA2_LATE | -0.000226203 | [-0.000684720, +0.000026053] | 0.815318 | 2 | 1500.000 | False |
| EMA_0.5 | +0.000958756 | [-0.000685542, +0.003483062] | 0.563944 | 9 | 1500.000 | False |

All estimates and individual gate results, family median gains, Elo intervals,
win/tie/loss counts and failure counts are in [metrics.json](metrics.json).

No arm passed primary; no candidate DL-Val panel was run. The mandatory r882
regression baseline is the only validation prediction panel.

Red-team interpretation and checks:

- SWA4_UNIFORM: top 10 absolute family-gain share 0.9259068747874217; signed share 1.0288776033926281.
- SWA2_LATE: top 10 absolute family-gain share 1.0; signed share 1.0.
- EMA_0.5: top 10 absolute family-gain share 0.9518382446522127; signed share 1.0213181841136758.
- The three fold seeds partition the same fixed predictions: identical family
  vectors, not independent fits. Statistics resample 205 unique families, never 615
  replicated observations. No unseen-family generalization claim.
- Checkpoints already saw DL-Train and validation selection. The optional secondary
  remains selector-exposed; sealed/holdout/meta-test data were not accessed.
- All 67 DOPE parameter/normalizer tensors were bitwise equal across inputs and
  composite `c51a56d0` stayed fixed before/after inference. All 47 saved adapter
  tensors replay exactly from the preregistered averages; frozen-state hashes
  match across every arm. Buffers remain r882. Zero optimizer updates.
- Saved raw costs replay every first-argmin choice and native utility exactly;
  primary statistics replay exactly; the candidate secondary was correctly skipped. No AUC or promotion.

Resources: 20.70 minutes, 4,138 external samples, maximum one observed worker
thread, peak RSS 949,440,512 bytes, minimum available filesystem space
143,551,389,696 bytes; no pressure pauses.
CPU xbabe1/nice19, hard RSS<6GB/threads<=4/scratch<2GB/free>=40GB; two-hour
wall budget included pressure waits. Numerical workers used one configured thread.

Numerical custody: [s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H138/astra-t1742/manifest.json](s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H138/astra-t1742/manifest.json),
VersionId `pTaY7_LeCiGFPve6eqovHcicLu2t0ayk`, SHA-256 `379d29fb8616fa67f9dcc4fb22ed5b1771b024a07feac33a290296e0a213d4af`.
Exact-version GET and all 52 archive members were hash-verified. Source remains in Git. Original derived-input S3 versions were not
established; hash-verified originals provide the input-copy recovery route.
After recovery verification, removed 32 remote run files (28,117,423 bytes),
1,319 owned input copies (328,879,874 bytes), and the clean remote execution
worktree. The corresponding local run files were removed too. Original input
files, live checkpoints, Micro assets and foreign lanes/processes remain intact.
Cleanup receipt: [s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H138/astra-t1742/cleanup.json](s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H138/astra-t1742/cleanup.json),
VersionId `AcQ7C6Rh4oFvLgPmCWYQOa8pg2M7Kl_6`, SHA-256 `1702f6eebc217fc04a30401c4dcff05df11a3ea3f2a2c5e5c5e26d6610b0cf51`;
exact-version readback verified. Two local bulk archive copies were also removed
after checking their S3 recovery identity. The source commit remains an ancestor
of verified remote main.

Next: retain r882 and report this fixed trajectory comparison with its dependence
and selection limits. Any new weighting, model or retry needs a new prospective
protocol. This result does not authorize a live admission-policy change.
