# H52: objective alignment and anchored updates

Status: **complete — no paired configuration passed preregistered admission; no terminal DL-Val reads; Elo N/A** (2026-10-30).

## Hypothesis and setup

H52 asked whether cost-focused ranking, an explicit gate penalty, anchor regularization, or a within-family target permutation could improve argmin utility over regression controls. It ran 27 cells on seeds 12–14: nine matched CONTROL cells (`k=2/lr=1e-4`, `k=16/lr=1e-4`, and `k=16/lr=3e-4`), six FOCUS cells (`tau=0.1,0.5`), three GATE cells, six ANCHOR cells (`lambda=1,10`), and three NULL cells. Training was serialized on CUDA with the qualification lock released between cells. The matched-slate proof showed exact equality for CONTROL `k=16/lr=1e-4` and FOCUS `tau=0.1`, seed 12, across five folds × 16 steps.

The pinned offline source was r882 (SHA256 `0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`) with frozen DOPE SHA256 `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` and 29,211 trainable adapter parameters. The cohort was 1,219 DL-Train contexts / 205 families; five family-grouped folds used seed 4900. The 93 holdout contexts remained excluded. The Step 0 CPU receipt is [`step0-diagnostic.json`](step0-diagnostic.json). Per-cell source checks confirmed the checkpoint, manifest, and DOPE identities before/after training.

Admission was preregistered on the 615 paired family rows per configuration: pooled family median strictly above `0.001` and family-cluster bootstrap 95% lower bound strictly above `0.001`; ties fail. Diagnostic-only metrics include nonzero and signed family counts, exact sign-test p-values, and family-cluster bootstrap intervals for the paired mean. Those diagnostics do not alter admission.

## Results

All six objective configurations had pooled family median `0` and cluster-bootstrap LB95 `0`; none passed admission. Per-seed medians and lower bounds were also `0`. There were zero terminal DL-Val reads, so Elo is **N/A**. All seeds produced distinct adapter identities and distinct per-family utility vectors within each objective configuration. The full compact candidate record is [`metrics.json`](metrics.json) (46,732 bytes, below the 50 KB limit). Full run artifacts and the larger audit remain outside Git under `~/jope-data/h52/runs/`.

| Configuration | Paired family mean | Cluster-bootstrap 95% CI for mean, diagnostic only | Median / median LB95 | Sign-test p | Nonzero families | Candidate mean vs r882 |
|---|---:|---:|---:|---:|---:|---:|
| FOCUS `tau=0.1` | +0.0206 | [+0.0022, +0.0427] | 0 / 0 | 0.0353 | 120/205 | −0.0092 |
| FOCUS `tau=0.5` | +0.0248 | [+0.0080, +0.0451] | 0 / 0 | 0.0197 | 125/205 | −0.0051 |
| GATE | +0.0251 | [+0.0098, +0.0430] | 0 / 0 | 0.0709 | 123/205 | −0.0048 |
| NULL placebo | +0.0220 | [+0.0029, +0.0446] | 0 / 0 | 0.0232 | 122/205 | −0.0079 |
| ANCHOR `lambda=1` | +0.00015 | [−0.00010, +0.00054] | 0 / 0 | 1.0000 | 2/205 | −0.0796 |
| ANCHOR `lambda=10` | +0.00006 | [−0.00031, +0.00054] | 0 / 0 | 1.0000 | 4/205 | −0.0797 |

The FOCUS and GATE mean gains against their matched `k=16/lr=1e-4` CONTROL are also present in the NULL placebo. That placebo match means the paired mean gain does not identify an objective-specific effect. The matched CONTROL itself averaged about `−0.030` vs the live-sized `k=2/lr=1e-4` CONTROL; `k=16/lr=3e-4` averaged about `−0.079`. The apparent paired advantage is therefore consistent with those higher-step controls being harmful, rather than the changed objective helping.

Continuity means versus r882 were negative for every configuration: FOCUS `tau=0.1/0.5` −0.0092/−0.0051, GATE −0.0048, NULL −0.0079, and ANCHOR `lambda=1/10` −0.0796/−0.0797. ANCHOR had only two or four nonzero paired families out of 205 across its pooled seed rows, consistent with the penalty holding the adapter near its control. None of these results beat r882 on the held-fold continuity measure.

The later owner ruling proposed a mean-LCB/no-harm gate for future H-wave decisions. It arrived after H52's full results had been audited, and the H52 closeout instruction explicitly preserved the preregistered median+LB95 rule. H52 therefore keeps its original zero-admission verdict; the later rule is recorded prospectively in the H53 preregistration, with no retroactive reclassification.

## Verdict and next step

**No arm beats r882; the FOCUS/GATE paired mean gains are explained by the NULL placebo. H52 is closed.** The diagnostic mean intervals are descriptive and do not overturn the preregistered admission decision. No validation data was opened for terminal scoring and no Elo was assigned.

Next, start H53, the preregistered learn-from-zero pilot with fresh adapter initializations and DL-Val curves used only as diagnostics. H54 is the proposed fold-clean baseline (retrain from pre-r882 initialization using only each fold's out-of-fold families); H54 is proposed only and was not run in H52.
