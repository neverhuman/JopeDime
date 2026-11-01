# H165: annealed gradient noise and adapter-input dropout

**NO_ADVANCE — completed 15 cross-fit cells / 60 trajectories / 2,400 optimizer updates across three seeds.** Retain r882; no serving promotion. [Small metrics](metrics.json), [preregistration](PREREG.md), [A1 setup disclosure](PREREG_AMENDMENT_A1.md), [versioned evidence custody](custody.json).

Hypothesis: gradient noise or input dropout could regularize r882 adapter continuation. CTRL uses AdamW with 40 updates, GN01/GN10 use eta .01/.1 and the prespecified median-CTRL-gradient-norm schedule, and DROP applies p=.1 to adapter inputs during fitting only. All start from a read-only r882 copy with 47 trainable tensors / 29,211 parameters. Frozen DOPE c51a56d0 and all non-adapter bytes are checked after every update/evaluation. The native 91-context baseline is exact (mean 1.1377941898164525), and every step-zero dropout-disabled cached/native cost is bit-identical.

Setup uses retained approved real-S3-derived 1219-context/205-family TRAIN inputs and 91-context/19-family secondary DL-Val. All 276 dataset IDs join origin=real in the actual versioned catalog GET; 1,319 numeric input members were rehashed against the versioned recovery closure. Within each outer fold, 123 families fit, 41 select and 41 assess. Three seeds 165101–165103, family split 165140, bootstrap 165141. Assessment and DL-Val never fit or calibrate noise. H156's 8-context/24-candidate batches are paired across arms. Strict inner selection requires best-step gain > .001 with 1e-12 tie rejection and bootstrap LCB > 0; otherwise step 0. Decisions are durable before trained-policy A/Val assessment.

## Primary: equal-family out-of-fold utility

| Policy | Median family gain vs r882 | 95% family CI | Median Elo |
| --- | ---: | --- | ---: |
| CTRL | +0.000000000 | [+0.000000000, +0.000000000] | 1500.000 |
| GN01 | +0.000000000 | [+0.000000000, +0.000000000] | 1500.000 |
| GN10 | +0.000000000 | [+0.000000000, +0.000000000] | 1500.000 |
| DROP | +0.000000000 | [+0.000000000, +0.000000000] | 1500.000 |

Points are medians of the three per-seed equal-family mean gains. Intervals use 10,000 shared family-bootstrap draws and median seed effects in every draw; the three treatments' r882 contrasts receive Holm adjustment. Elo uses the canonical 400-step Bradley–Terry MM with family-weighted context-pairwise wins, prior .01, tie margin 1e-4, r882 = 1500 and 1,000 family-bootstrap Elo draws. Full per-seed gains, adjusted p-values and Elo intervals are in metrics.json. Advancement requires median gain > .001, LCB > 0 and Holm p < .05; CTRL is a descriptive reference. All 60 strict arm-cell decisions chose step 0, so the three evaluated seed panels are identical r882 predictions; these are not independent prediction replications.

## Secondary: historical 91-context DL-Val

| Policy | Median family gain vs r882 | 95% family CI | Median Elo |
| --- | ---: | --- | ---: |
| CTRL | +0.000000000 | [+0.000000000, +0.000000000] | 1500.000 |
| GN01 | +0.000000000 | [+0.000000000, +0.000000000] | 1500.000 |
| GN10 | +0.000000000 | [+0.000000000, +0.000000000] | 1500.000 |
| DROP | +0.000000000 | [+0.000000000, +0.000000000] | 1500.000 |

Each seed's secondary panel averages its five selected outer-fold policies' context utilities before 19-family inference; it does not treat 455 policy-context outcomes as independent observations. This panel was not used to tune the treatments or select checkpoints.

## Descriptive unconstrained selector

| Policy | Median family gain vs r882 | 95% family CI | Median Elo |
| --- | ---: | --- | ---: |
| CTRL | -0.016092436 | [-0.028730314, -0.002110392] | 1494.731 |
| GN01 | -0.004176428 | [-0.012388264, +0.002436201] | 1497.190 |
| GN10 | -0.001754570 | [-0.010230206, +0.002346186] | 1498.912 |
| DROP | -0.016092436 | [-0.028730314, -0.002108630] | 1494.731 |

This uses the same earliest best inner-B step including 0, without strict eligibility gates, locked before A/Val scoring. It is not an assessment/Val oracle and cannot advance. Every arm has 0/15 nonzero strict picks. Unconstrained nonzero picks: CTRL, GN01 and DROP each 9/15; GN10 6/15. Negative or flat selector panels and raw decisions remain in the archive. The paired median GN01-minus-CTRL gain is +0.008970822 (95% CI [-0.000047068, +0.025209755]); GN10-minus-CTRL is +0.005885413 ([-0.000828121, +0.015736610]). Both ordinary intervals include zero. Paired median contrasts need not equal differences between arm medians.

## Validation, failure disclosure and scope

Original setup source e247e6bdd stopped INCOMPLETE after 234.29 s, with 0 optimizer updates/trained snapshots/selection locks, when a diagnostic logger dereferenced grad=None. Native baseline/raw-bit gates had passed. A1 corrected only inactive-gradient norm logging before any trained-policy outcome; original protocol bytes/seeds/folds/2,400-update budget and original 12-hour absolute deadline were preserved. REDTEAM review 5726971026 plus supplemental /root/rank11_redteam source review PASS; A1 source merged in #1417 as f7a4728de4499aca4941069fb84c5e3fbdd18201. The original failed setup and both source versions/logs are retained in S3. This is a disclosed setup continuation, not two completed scientific runs. Nine data-free tests, zero-forward smoke and five static record gates passed; full hosted/native workspace qualification is not claimed.

Independent recount passes 108,179 saved choice/label/gate checks, all 240 adapter snapshot hashes and 2,400 update traces, 28 family contrasts and canonical Elo points/CIs. Fresh native replay matches all 91 baseline cost vectors bit-for-bit and 36 trained-adapter choices/utilities. Observed dropout fraction 0.100156; realized noise-norm ratios [0.987256, 1.011981]. No protected/refused input reads, GPU use, speed measurements, live-loop or serving writes.

CPU xbabe2, nice 19, one math thread. Measured supervisor/worker peak threads 3, RSS 3.383 GB, maximum host CPU 44.6%, raw load 75.708; minimum available RAM 62.44 GiB, free disk 639.01 GB. Hard guards stayed within registered bounds. Only owned H165 processes moved from contended core 9 to online core 10; no source, numerical rule or foreign process changed.

Interpretation is conditional continuation on a historically exposed r882/selection cohort, not unseen-family confirmation. The bootstrap conditions on trained policies/selection and does not retrain inside each draw. An all-fallback assessment panel does not prove all nonzero trajectories are harmful or that the selector has calibrated power; the planted +.005 data-free control establishes mechanical reachability only. Original H50/H58 derivative S3 versions are not established; exact numeric recovery is bound to the archived H144 version and runtime sources are separately pinned.

Evidence: s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H165/codex-t0058/8358749a38650727/manifest.json (VersionId LkqfRFJZmbLMbAqVmGDJd4IMdWKv15S7, SHA256 eed9e83f809375fd04273fd244a7d4f803c11733358b420cabb6d6021327d848); archive and every member verified, with create-only PUT and actual exact-version byte-readback. Large arrays/model weights remain outside Git. Next step: retain r882 and record this fixed screen; any further scientific experiment needs its own prospective assignment and registration.
