# H194: attempt2 incomplete; one final fresh attempt authorized

**INCOMPLETE.** Neither run reached pooled-B selection or candidate A/Val evaluation.
No utility gain, confidence interval, Holm p-value or Elo result is available, and no
partial scientific effect is claimed. The result does not establish whether IMH helps.
No model or live selector was promoted.

The hypothesis is that a FIT-calibrated invalid-candidate hinge improves r882 continuation
while retaining its active revealed-history kernel. [Preregistration](PREREG.md) fixes
CTRL, primary IMH (Holm1) and the true-valid count-matched shuffled placebo,40updates each,
three independently partitioned seeds and five cyclic folds. FIT/B/A contain123/41/41families;
the real S3 regression TRAIN corpus contains1219contexts/205families. DL-Val91/19 is descriptive.
Only47adapter tensors/29,211parameters train; DOPE and the remaining parameters stay frozen.

| Execution accounting | Attempt1 | Owner-authorized fresh attempt2 |
|---|---:|---:|
| Planned optimizer updates | 1800 | 1800 |
| Optimizer calls started/completed | 1080/1080 | 1246/1246 |
| Backward calls started/completed | 1080/1080 | 1247/1246 |
| Complete40-update trajectories | 27/45 | 31/45 |
| Complete seed/fold cells | 9/15 | 10/15 |
| Additional completed steps in an unfinished trajectory | 0 | 6 IMH steps |
| Candidate A/Val evaluations | 0/0 | 0/0 |
| Pooled-B selection | Not reached | Not reached |
| Guard stop | Shared disk93.8016% >=93.8% | Load131.8555 >108.8 across3independent5s windows |
| Elapsed | 5321.27s (88.69min) | 6407.76s (106.80min) |

Attempt1 stopped during original-r882 B-reference inference. Its [closeout#1627](https://github.com/neverhuman/JopeDime/pull/1627)
merged before the owner-authorized fresh attempt2. Attempt2 used the exact original
[#1601](https://github.com/neverhuman/JopeDime/pull/1601) scientific source, seeds, folds, losses
and1800-update budget, with no partial trajectory or B-selection reuse. The external
supervisor tightened the thread cap from four to three. A pre-model launch-binding error
exited before model construction/scoring/updates; its complete startup record was retained
and corrected before the fresh model run. [Reattempt authority](REATTEMPT.md) records the scope.

Attempt2 completed both first replicates and the third replicate's fold0 CTRL40, then6IMH
updates. Its next backward was interrupted by the sustained-load guard before another
optimizer call. There were2326completed optimizer calls across the two runs and no third
scientific attempt. Both supervisors terminated only their own workers; no foreign job was
paused or killed. CPU nice19/math1, RSS<6GB and the applicable thread limits were enforced.

Native91 utility and every float32 candidate-cost byte replayed exactly before TRAIN pools
opened in both attempts. Per-update frozen checks passed, and protected reads were0.
A post-stop, zero-forward/zero-backward/zero-optimizer check reconstituted all31saved attempt2
adapters, matched their recorded tensor hashes and verified DOPE composite`c51a56d0...`
before/after with the whole reference restored. This validates stored-state integrity;
it does not supply the missing registered scientific result.

[Execution-only metrics](metrics.json) retain exact ledgers and null scientific endpoints.
[Versioned S3 custody](custody.json) records actual exact-version readback and every payload
hash: attempt1 has185payloads/47NPZ/29Torch files; attempt2 has194payloads/54NPZ/33Torch files.
Both scientific Git bundles restored the original commit and36source/parent file hashes
in an independent bare repository with its pinned prerequisite declared. Targeted22data-free
controls, Ruff, five jope-records gates and exact-head author review accompany publication;
full hosted CI and independent native GitHub approval are not claimed.

The owner has authorized **one final fresh attempt3 after this closeout merges**,
with the exact original#1601 source/seeds/folds/budgets and no A1/A2 trajectory or B-selection
reuse. Neither earlier attempt supplied a pooled-B lock or candidate A/Val outcome, so this
infrastructure rerun does not choose among observed scientific results. Prefer xbabe1 when
every input can be staged hash-identically without large copies; the protected flywheel,
H199, H200 and H201 jobs remain untouched. [Reattempt authority](REATTEMPT.md) specifies
host/resource rules and the4h wall cap. No fourth attempt is authorized.
