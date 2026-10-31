# H127 — step-count selection including zero

**NO_ADVANCE**. Execution status: **COMPLETE**. Preregistered in [PR #1108](https://github.com/neverhuman/JopeDime/pull/1108) before scoring; the frozen [PREREG.md](PREREG.md) is unchanged. No serving promotion.

Hypothesis: H125 showed harmful extra CONTROL updates. Selecting the update count, including zero, inside development-family folds may avoid that harm or find a qualifying positive step count. A finite unsuccessful grid cannot establish that r882 is a local optimum.

Setup: seeds 73–75; a fresh family permutation with seed 127073. Four inner folds rotate 41 validation families against 123 fitting families within the 164-family development set. The final 41-family assessment remains closed until a complete inner selection is frozen and exact-version S3-verified. CONTROL uses 40 updates; ES selects from 0/5/10/20/40/80/120; ES_FREEZE selects the same way but trains only the two cost-output projections (50 parameters). Other arms expose 29,211 adapter parameters. DOPE stays c51a56d0; every trajectory/refit uses a unique copied r882.

Actual contexts: 932 development / 287 assessment, all DL-Train. No DL-Val91 or protected payload is admitted. Prior r882 exposure to DL-Train limits this to additional fine-tuning, not unseen-family generalization.

| Arm | Selected updates | Selection |
|---|---:|---|
| ES | 0 | No positive k qualified; zero fallback |
| ES_FREEZE | 0 | No positive k qualified; zero fallback |

Inner selection uses median seed family-mean gains, 10,000 paired family bootstrap draws and a Bonferroni-six lower bound over the six positive counts. Full checkpoint statistics: [selection.json](selection.json). Missing rotations never count as a zero selection.

| Arm | Median gain vs r882 | Family 95% CI | Median Elo, 95% CI (r882=1500) | Harm | Verdict |
|---|---:|---|---|---:|---|
| CONTROL | -0.029169639 | [-0.077391, 0.007188] | 1483.0, [1395.2, 1577.5] | 14.63% | DIAGNOSTIC |
| ES | 0.000000000 | [0.000000, 0.000000] | 1500.0, [1500.0, 1500.0] | 0.00% | NO_ADVANCE |
| ES_FREEZE | 0.000000000 | [0.000000, 0.000000] | 1500.0, [1500.0, 1500.0] | 0.00% | NO_ADVANCE |

Fresh r882 assessment utility: 1.094988495002. Total actual optimizer updates: 3000, including 2,880 inner-CV updates. All 24 inner trajectories, 168 registered checkpoints and nine final cells are included. The maximum was 3,720 updates; zero-step refits have genuinely zero updates and unchanged adapter hashes. Full precision, per-seed outcomes and repeated-vector flags: [metrics.json](metrics.json).

WIN requires strict positive gain and CI lower bound, Holm over ES/ES_FREEZE, and harm no greater than 10% of contexts losing more than 0.01. Both arms have bootstrap lower-tail area 1.0 and fail Holm (ordered thresholds 0.025/0.05). Zero is a tie, never an improvement win; the three zero-step seed results are identical copies, not three independent confirmations. Comparisons versus trained CONTROL are descriptive and do not replace r882. Bootstrap lower-tail areas are exploratory statistics, not separately calibrated null-test proof.

Resources: CPU only; external monitor status `WORKER_EXITED`, exit `0`; peak tree RSS 4.049 GB and wall time 3800.0 seconds. Limits: six worker threads, 6 GB RSS, 30 GB MemAvailable floor, load1 86.7/102, disk 94%, nice at least15. No GPU or qualification.lock. No original checkpoint/input write or checkpoint/dataset/receipt deletion.

Validation: 15 data-free tests cover exact H125 first-40-step parity, no repeated schedule blocks, zero fallback, complete-grid requirements, strict confidence gating, resource boundaries and state/input-bound frozen-anchor caching. Before updates, unchanged-r882 regression compares independent copies and verifies cache hits bitwise against uncached computations. Full-pool anchor outputs can then be reused; variable training slates are never cached. Source and full checkpoint hashes bind this optimization. Actual guards and any stop are retained in the archive.

Completed-run artifact verification reconstructs family means, checks every saved adapter/checkpoint hash and loss-curve update, replays inner selection and final statistics, and verifies that selection custody preceded assessment admission. A separate comparison verifies unchanged non-output parameters in all 72 positive ES_FREEZE snapshots. These are same-agent artifact checks, not an independent human review.

Next step: Keep r882; neither selection rule admitted positive updates. Any new search needs a fresh protocol; this grid does not establish a local optimum.

Evidence: [custody.json](custody.json), [manifest.json](manifest.json), [REPRO_BUNDLE.md](REPRO_BUNDLE.md). Create-only `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H127/astra-t1105/manifest.json`, VersionId `eAFAm_AFDAxMeujJdwEujz4KZoDDoxG4`, SHA256 `9ebb1313671f485745c131d8548651dfecafbdf44f8f64ac0c7abc451344a428`. 475 evidence members were restored and hash-verified; all bulk output remains outside Git.
