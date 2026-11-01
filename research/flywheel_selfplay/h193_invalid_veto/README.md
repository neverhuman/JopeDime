# H193 — revealed-history invalid veto

**NO_ADVANCE.** One zero-update CPU attempt completed. The r882 adapter, DOPE and kernel constants remained fixed; no serving adoption. [metrics.json](metrics.json) preserves the exact result.

The hypothesis was that a hard veto based on the kernel-weighted share of revealed neighbours with original utility exactly −2 would remove damaging choices. [PREREG.md](PREREG.md) fixed FIT-only positive-score quantiles {inf,q90,q75,q50}, three literal family reshuffles, and 15 equal-family B-only threshold choices. Infinity preserves r882; all-veto falls back to r882. Every choice and each descriptive DL-Val median threshold was locked before thresholded A/Val. VETO is the sole primary under imported H191/H177/H178 (Holm1 plus median-margin/LCB gate). Fixed quantiles and DL-Val cannot rescue the primary.

Seeds 193101/193102/193103 use distinct literal fold seeds 193141/193142/193143. Each cyclic five-fold cell has 123 FIT, 41 B and 41 A families; every family appears exactly once in pooled A per seed. Bootstrap seed 193150 uses 10,000 paired family draws (Elo: 1,000 draws at 193151). One attempt completed without a resume.

The selected VETO policy retained all 17 exact −2 picks in every reshuffle. Fixed Q50 reduced that count to 16, but its utility interval crossed zero; that descriptive result does not rescue the primary.

| Policy | Role | A per-seed gains | A median gain [95% CI] | A Elo [95% CI] | DL-Val median gain [95% CI] | DL-Val Elo [95% CI] |
|---|---|---|---|---|---|---|
| VETO | primary | +0.00228660, -0.00000991, -0.00154885 | -0.00000991 [-0.00250264, +0.00084250] | 1500.61 [1499.12, 1501.96] | -0.00019566 [-0.00058698, +0.00000000] | 1497.08 [1491.23, 1500.00] |
| Q90 | descriptive | +0.00009931, +0.00029257, +0.00009931 | +0.00009931 [+0.00000000, +0.00028345] | 1500.44 [1500.00, 1501.22] | +0.00000000 [+0.00000000, +0.00000000] | 1500.00 [1500.00, 1500.00] |
| Q75 | descriptive | +0.00121009, +0.00121009, +0.00121009 | +0.00121009 [-0.00573754, +0.00910377] | 1501.15 [1499.52, 1502.91] | -0.00019566 [-0.00058698, +0.00000000] | 1497.08 [1491.23, 1500.00] |
| Q50 | descriptive | +0.00248858, +0.00251108, +0.00248858 | +0.00248858 [-0.00396207, +0.01058668] | 1500.75 [1498.08, 1503.43] | -0.01396111 [-0.03553225, +0.00023845] | 1494.89 [1479.52, 1507.31] |

Imported primary admission `False`, Holm1 p `0.43795620437956206`; combined median confirmation `False`. The unchanged H178 receipt disposition is in metrics. R882 anchors Elo 1500 in the five-policy panel; win frequency and utility magnitude measure different effects.

| Reshuffle seed | B-selected thresholds, folds0–4 | Locked DL-Val median |
|---|---|---|
| 193101 | inf, 0.301672161, 0.242362723, 0.241902202, 0.405629605 | 0.301672161 |
| 193102 | 0.24583824, 0.298617683, inf, 0.241345853, inf | 0.298617683 |
| 193103 | 0.294478171, inf, inf, 0.243263073, 0.241600074 | 0.294478171 |

Selection counts across 15 cells: `{'INF': 5, 'Q90': 1, 'Q75': 3, 'Q50': 6}`. Exact grids and numeric thresholds are retained in metrics and the immutable selection lock.

| Policy | A context utilities by seed | DL-Val context utilities by seed (val_reg) | A exact −2 counts | DL-Val exact −2 counts |
|---|---|---|---|---|
| R882 | 1.423073435, 1.423073435, 1.423073435 | 1.137794190, 1.137794190, 1.137794190 | [17, 17, 17] | [0, 0, 0] |
| VETO | 1.424780419, 1.423118504, 1.421981046 | 1.137589930, 1.137589930, 1.137589930 | [17, 17, 17] | [0, 0, 0] |
| Q90 | 1.423186162, 1.423319439, 1.423186162 | 1.137794190, 1.137794190, 1.137794190 | [17, 17, 17] | [0, 0, 0] |
| Q75 | 1.424144342, 1.424144342, 1.424144342 | 1.137589930, 1.137589930, 1.137589930 | [17, 17, 17] | [0, 0, 0] |
| Q50 | 1.425240529, 1.425259452, 1.425240529 | 1.123219402, 1.123219402, 1.123219402 | [16, 16, 16] | [0, 0, 0] |

The exact −2 count is the registered mechanism check; it includes both gate failures and passing original fitness exactly +2. Context means are descriptive; the primary uses equal family weights.

| Stage / seed | Policy | Candidate veto fraction | All-veto fallback fraction |
|---|---|---|---|
| A / 193101 | R882 | 0/1404287 (0.000000%) | 0/1219 (0.000000%) |
| A / 193101 | VETO | 62084/1404287 (4.421034%) | 0/1219 (0.000000%) |
| A / 193101 | Q90 | 24822/1404287 (1.767587%) | 0/1219 (0.000000%) |
| A / 193101 | Q75 | 60758/1404287 (4.326608%) | 0/1219 (0.000000%) |
| A / 193101 | Q50 | 121553/1404287 (8.655852%) | 0/1219 (0.000000%) |
| A / 193102 | R882 | 0/1404287 (0.000000%) | 0/1219 (0.000000%) |
| A / 193102 | VETO | 49713/1404287 (3.540088%) | 0/1219 (0.000000%) |
| A / 193102 | Q90 | 24461/1404287 (1.741880%) | 0/1219 (0.000000%) |
| A / 193102 | Q75 | 61093/1404287 (4.350464%) | 0/1219 (0.000000%) |
| A / 193102 | Q50 | 121672/1404287 (8.664326%) | 0/1219 (0.000000%) |
| A / 193103 | R882 | 0/1404287 (0.000000%) | 0/1219 (0.000000%) |
| A / 193103 | VETO | 65043/1404287 (4.631746%) | 0/1219 (0.000000%) |
| A / 193103 | Q90 | 24655/1404287 (1.755695%) | 0/1219 (0.000000%) |
| A / 193103 | Q75 | 60623/1404287 (4.316995%) | 0/1219 (0.000000%) |
| A / 193103 | Q50 | 121431/1404287 (8.647164%) | 0/1219 (0.000000%) |
| DL-Val / 193101 | R882 | 0/104832 (0.000000%) | 0/91 (0.000000%) |
| DL-Val / 193101 | VETO | 5753/104832 (5.487828%) | 0/91 (0.000000%) |
| DL-Val / 193101 | Q90 | 3032/104832 (2.892247%) | 0/91 (0.000000%) |
| DL-Val / 193101 | Q75 | 6035/104832 (5.756830%) | 0/91 (0.000000%) |
| DL-Val / 193101 | Q50 | 11826/104832 (11.280907%) | 0/91 (0.000000%) |
| DL-Val / 193102 | R882 | 0/104832 (0.000000%) | 0/91 (0.000000%) |
| DL-Val / 193102 | VETO | 5975/104832 (5.699596%) | 0/91 (0.000000%) |
| DL-Val / 193102 | Q90 | 2819/104832 (2.689064%) | 0/91 (0.000000%) |
| DL-Val / 193102 | Q75 | 5989/104832 (5.712950%) | 0/91 (0.000000%) |
| DL-Val / 193102 | Q50 | 11893/104832 (11.344818%) | 0/91 (0.000000%) |
| DL-Val / 193103 | R882 | 0/104832 (0.000000%) | 0/91 (0.000000%) |
| DL-Val / 193103 | VETO | 6244/104832 (5.956197%) | 0/91 (0.000000%) |
| DL-Val / 193103 | Q90 | 3036/104832 (2.896062%) | 0/91 (0.000000%) |
| DL-Val / 193103 | Q75 | 5981/104832 (5.705319%) | 0/91 (0.000000%) |
| DL-Val / 193103 | Q50 | 11865/104832 (11.318109%) | 0/91 (0.000000%) |

The B lock also preserves per-cell/per-grid veto and fallback counts and rates. Native costs and invalid scores are seed-invariant; FIT quantiles and selected utilities can vary. These are family reshuffles, not independent model replications. Each cell excludes its own A outcomes; families recur in other FIT/B roles and were historically exposed. Paired family intervals condition on fitted thresholds and locked policies, without reselection in each bootstrap draw. H178 calibrated original m4, not nested H193 m1; no new empirical FWER guarantee or sealed qualification is claimed. Invalid share is a kernel score, not calibrated probability.

Native 91 baseline passed first, full vector and mean 1.1377941898164525 bit-exact; infinity matched r882 on every 1,219 TRAIN context. Adapter SHA before=after `9c3ba940d9eb60aa2a39dd57c229c0952927ea1c2f22c372ae9764f43baa8669`; DOPE `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`. Saved-cost/share/label reconstruction verified 34278 choices, all 15 FIT grids/B decisions, both treatment panels, rates, full baseline and resource receipts. The registered statistical implementation was re-executed for one selector/eight contrasts/Elo, not independently reimplemented; probability vectors were source-pinned inputs to replay, not a second model inference. Replay performed zero model forwards/updates.

Next: retain r882; this result supplies no eligible serving promotion or alternative-primary rescue.

Run source `b5a1c73de8e3dd074eff1d5c81992056511f7161`, prereg [PR #1592](https://github.com/neverhuman/JopeDime/pull/1592). Targeted 18 tests, Ruff and five records/source pins and audited zero-score constructor smoke passed. Author review and peer review/timeout disposition retained in custody: author COMMENT `5729210434`; peer IDs `[]`; bounded timeout `False`. CPU xbabe2 nice 19, aggregate<=4threads/RSS<6GB, CPU85%, sustained load>85% online cores (three5s samples), Mem16GiB/disk93.8%/free40GB terminal guards; one 14,400s cap/output root<500MB. No hosted full-CI/product qualification claim, GPU, protected data or live-loop writes. Application audit is not an OS sandbox.

Versioned custody: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H193/astra-t0930/manifest.json`, VersionId `_6yS4XXNnEvnAcaK1k0T6VGH__zIgRvJ`, SHA256 `3cdc7c49d365d0096c7d08a7b5d80145358b915d4eb584e0ab0aaca65abb02bf`. [custody.json](custody.json) binds exact-version full payload recovery and 1460 archive-member hashes. Archives were constructed/recovered in memory. Inputs, raw native costs/invalid shares, lock, panels, manifests and receipts preserved; source remains Git-pinned. Only verified owned bulk is eligible for reclamation after closeout merge.

The owner launch nudge `hwave_t1002_h193_launch.md` explicitly removed further review wait after ENG merged the preregistration following independent targeted tests. No rd_argon review or elapsed-time timeout is claimed.

Metadata: H191 internally calls VETO `KXSEL`; this alias does not change the intervention. The summarizer emits primary-role/p-value fields inside DL_Val_descriptive; they remain descriptive and never enter admission. H177 uses seed-averaged family gains, whereas the separate median-bootstrap result reduces the seed axis by median. Both are retained without favorable substitution.
