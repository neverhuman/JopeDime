# H220 — within-family mixup

**NO_ADVANCE**. Completed 30 independent fits and 1,200 full optimizer updates, plus 4 non-scored smoke updates. MIX median gain -0.037514; primary Holm-adjusted LCB -0.064105.

Hypothesis: convex mixing within FIT families smooths adapter behavior and
reduces the harm seen after training from r882. MIX uses Beta(0.4,0.4), mixing
probability 0.5 and distinct same-family partners; singleton families remain
identity rows. CONTROL shares the primary samples and 40-update/four-microbatch
recipe. The optional cross-family arm was omitted before outcomes.

| Arm / reference | Median gain | 95% CI | LCB95 | Holm p | Harm | Positive seeds |
|---|---:|---|---:|---:|---:|---:|
| CONTROL / R882 | -0.039604 | [-0.070338, -0.009210] | -0.065031 | diagnostic | 78/205 | 0/3 |
| MIX / R882 | -0.037514 | [-0.069373, -0.010981] | -0.064105 | 0.996700 | 78/205 | 0/3 |
| MIX_vs_CONTROL / CONTROL | +0.001400 | [-0.001303, +0.003155] | -0.000786 | diagnostic | 5/205 | 2/3 |

Elo (r882=1500): CONTROL 1482.154, MIX 1483.211.

MIX applied distinct same-family partners on 19,173/38,400 primary rows; 118 singleton draws stayed unchanged.

[PREREG.md](PREREG.md) fixes the exact input/target mixing and all decision gates.
Each of 3 fresh seeds rotates 5 whole-family folds over 205 families/1,219 TRAIN
contexts. [fold_jaccard.json](fold_jaccard.json) reports all overlap comparisons
against H210/H217/H218/H219 (maximum 0.242424). This reused development cohort
does not establish sealed generalization. MIX vs CONTROL is secondary.

Each cell resets to a SHA-verified r882 copy with 29,211 trainable adapter
parameters. DOPE stays frozen; proofs record gradients, optimizer steps,
changed parameters and changed FIT forwards. All terminal models were
locked and S3-recovered before held-out utility reads. The original dataset
labels stay intact; convex targets exist only in transient training batches.

Unchanged-r882 native91 regression passed before training: all 91 utilities
matched bit-exact, with mean 1.1377941898164525. Cached/native parity also passed
on all 1,219 TRAIN contexts.
See [metrics.json](metrics.json) for the observed receipts. No trained arm is
scored on DL-Val. No sealed/meta-test/outer-test/holdout reads, U4 measurement,
speed verdict or serving promotion.

Preregistration [#1843](https://github.com/neverhuman/JopeDime/pull/1843) merged
before launch. Executed source `98f5ff06746d38c95690932b42c44276bb1389f4`. Attempt 2026-11-02T12:12:09Z through
2026-11-02T12:17:08.207587+00:00; supervisor accounting 298.724s
includes setup/native91/smoke and worker finalization. One fixed 90-minute attempt,
GPU0 FIFO, nice 19/numeric 1, RAM 20 GiB/free 200 GB and host-load guards; no retry.

[manifest.json](manifest.json) retains execution-source hashes and adds custody;
[provenance.json](provenance.json) records real-S3 inputs. S3 manifest
`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261102/H220/astra-t0500/closeout/manifest.json`; VersionId `jVUv47WHJrdkms9HA3H_8AVwf_KEJkc.`;
SHA256 `3380ba762638a4798563144d045d8a208cfc769ac54acb7cc48d93f997481a89`. Every worker object and runtime archive member
was recovered by exact version and SHA256. Bulk evidence stays outside Git.
No training, inference or statistical replay occurred during closeout.

Next step: Retain r882; this fixed mixup recipe did not meet the preregistered improvement gates.
