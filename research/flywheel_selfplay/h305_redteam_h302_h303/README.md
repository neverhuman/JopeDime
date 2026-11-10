# H305: independent red-team of H302 and H303

**Both frozen-model regressions pass. Both waves receive `RT_FAIL` under the preregistered audit contract.** H303's numerical `NESTED_CONFIRMED` result reproduces, including its added concentration criterion. The audit failures concern historical design exposure, data-read assertions and, for H303, incorrect baseline attribution; they do not mean the observed TRAIN gain disappears.

Preregistration `4844618b6a1b049e14d5a85bcc05adab20a66ea6` reached `origin/v1/next` before any H305 scoring. Owner: [ASTRA]. Authority: `~/nudges/hw_t0600_astra_h305.md`. H302 source: `aa224daa9f5bee4dc3d8e093e7ef81d0ef0653f8`; H303 closed at `7bef8870f7a9ffdb6565c5277c59f4af6c27c93d` before its H305 replay.

## Hypothesis and setup

The reported ENS_VOTE2 gains should reproduce from frozen models and survive checks of family separation, chronology, ties, direction, seed dependence, concentration, a conditional permutation null, prohibited reads and DOPE identity. H305 independently rebuilt the 568-column features and implemented model scoring and statistics. It imported no peer research runners or evaluators.

Inputs are the original 1,219 TRAIN contexts / 205 coarse families, root training seeds 30001–30003, and the exact-version H302/H303 S3 archives. H305 replayed the 360 final LightGBM models per wave; it did not refit them or replay H303's 4,320 inner models. All model hashes, all four H302 locks, all twelve H303 locks, and every primary pick hash match. H303's 18-row inner tables also reproduce every chosen configuration and conservative tie-break.

The actual reference is **r882 slot 0**, selected by `r882-train-costs.npz`. **Requested r3175-anchored Elo is unmeasured.** No new dev91 or val_reg outcome was read. The Elo below is MM Bradley–Terry with symmetric prior .01, r882=1500, and family ties excluded.

## Results

| Quantity | H302 | H303 nested |
|---|---:|---:|
| Per-seed family-equal gain | +.026823 / +.060358 / +.058892 | +.041289 / +.064316 / +.072258 |
| Median gain [family-bootstrap 95% CI] | +.058892 [.013275, .126473] | +.064316 [.022289, .133592] |
| One-sided LCB95 positive | 3/3 | 3/3 |
| Family-median gain | 0 / .001588 / 0 | 0 / .001588 / .006984 |
| Actual r882-anchored Elo | 1762.34 / 1690.82 / 1755.88 | 1725.66 / 1722.48 / 1727.10 |
| Gain after dropping top 1 family | .022418 / .033123 / .031650 | .034482 / .037100 / .045081 |
| Gain after dropping top 5 families | .013592 / .015995 / .014567 | .022905 / .018970 / .027514 |
| Gain after dropping top 10 families | .007948 / .006323 / .005566 | .013174 / .008633 / .016621 |
| Top-10 share of net gain | 71.8% / 90.0% / 91.0% | 69.7% / 87.2% / 78.1% |
| Drop-top-5 LCB95 | +.006062 / −.002748 / −.003722 | +.012384 / −.000304 / +.007092 |
| Registered permutation p | 1/1001 | 1/1001 |
| Spectral effective seed count, descriptive | 1.796 | 1.710 |

Shared family bootstrap: RNG 302150, 10,000 draws. Permutations: RNG 305150, 1,000 draws, exchanging the three-seed pick vectors jointly between contexts within each family. Both observed statistics exceed every draw. This conditional context-exchange null is **not** a retraining/label-shuffle null and cannot eliminate design-selection bias.

The seeds produce distinct picks. H302 pairwise agreement is 78.7–84.4%; H303 is 77.9–80.1%. Agreement on the union of switched contexts is only 25.1–40.5% and 29.2–39.5%, respectively. Nevertheless, seeds 30002/30003 have family-gain correlations .9842 and .9804. Three training seeds on shared families are not three independent data samples.

## Audit (a)–(i)

| Item | H302 | H303 |
|---|---|---|
| **(a) Fit/selection family overlap** | **FAIL for unbiased confirmation:** inherited tau=.10, q=6 and agreement2 were evaluated/retained on all 205 TRAIN families. Direct per-fold fitting and lr/n/margin selection are isolated. | **PASS within the declared grid; FAIL for an unqualified independent-family claim.** All12 nested choices exclude outer-test families. Architecture, grid, member recipe and r882 proposer still reuse the historically exposed TRAIN population, as H303 acknowledges. |
| **(b) Prereg precedes scoring** | Supported with caveats: git05:03:14 precedes four lock mtimes05:10:38–05:19:11; source scores afterward. | Supported with caveats: git05:38:11 precedes all 12 locks; the archived pre-score inventory matches every lock hash. |
| **(c) Ties** | PASS: W/T/L 77/111/17,108/61/36,96/87/22. | PASS: 88/93/24,108/67/30,122/50/33. Absolute delta≤1e−4 never wins. |
| **(d) Direction/reference** | Utility and Elo sign PASS; requested r3175 comparison unavailable. | **FAIL baseline label:** README/prereg call the reference r3175, but the code/data use r882. Sign and numeric Elo otherwise reproduce. |
| **(e) Seed noise** | Distinct picks; dependence caveat above. | Distinct picks; dependence caveat above. |
| **(f) Concentration** | Measured; drop-top-5 LCB positive 1/3. | Measured; drop-top-5 LCB positive 2/3, satisfying its new criterion. |
| **(g) Conditional null** | p=1/1001, scope caveat above. | p=1/1001, scope caveat above. |
| **(h) Prohibited reads** | **FAIL:** mixed NPZ arrays decode 91 DL-Val feature rows before TRAIN filtering. | **FAIL:** actual runtime source does the same. |
| **(i) DOPE** | PASS: independent tensor hash equals c51a56d0. | PASS: same frozen cached-input lineage; no neural update. |

For (a), [structure_audit.json](structure_audit.json) independently reconstructs SHA256(`H302-outer-v1` NUL family), yielding 57/45/50/53 test families. Every direct model fit and inner selection partition excludes its corresponding outer test in reviewed source. H303's threshold grid and tie choices also match the frozen locks. This is a conditional selector result; the audit does not assert that nesting failed or that historical exposure caused the measured gain.

For (h), the cache metadata contains 1,310 rows; its final 91 context IDs exactly equal the dev91 roster. `npz[key][i]` first materializes the whole NPY array. H302 `run_h302.py:255,264–265` and H303's final `run_h303.py:194–206` therefore read validation features. The resulting fit/selection dictionaries contain TRAIN rows only: **no validation-label fitting or validation-outcome use was found in the called wave code.** No sealed, holdout or fresh-cohort payload read was found. An exhaustive historical syscall trace is unavailable, so these are source/artifact conclusions, not a complete runtime absence proof. H305 reads only verified TRAIN NPY prefixes; ZIP decompression may prefetch physical compressed bytes, but no trailing feature rows are requested, decoded or used.

Both wave loaders also build all outer-test TRAIN features and labels before writing fit locks. Their literal `outer_test_rows_read_before_lock=0` counters are therefore false. Prediction ordering and fit/selection exclusion are separate facts. Archive mtimes are mutable, and per-score absolute receipts are incomplete; (b) cannot establish the absence of undocumented earlier scores.

DOPE was independently hashed from the non-live r882 copy: 56 parameter tensors and 11 normalizers yield `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`. Whole-checkpoint SHA `0084479a…` is identical before/after. This does not inspect or certify unrelated live state.

Two reporting corrections also matter. H303's “most families are neutral” prose contradicts its tie shares 45.4%,32.7%,24.4%; two seeds improve a majority of families by the registered threshold, although gains remain concentrated. Its unused “design-leak-dependent” branch compares H302's four criteria with H303's five. That branch was not taken here, but would not establish a causal effect of leakage if only the added concentration criterion failed.

## Verification, deviations and custody

Twelve focused tests pass, including a preserved failing fixture for a permutation roundoff edge case. Advanced indexing changed C/F memory layout and reduction order, incorrectly making an unchanged null look smaller. A shared statistic function with explicit contiguous layout fixes it, with no tolerance or scientific rule change. H302 p remains 1/1001. This correction occurred after the H302 result and before any H303 outcome was read; [numerical_correction.json](numerical_correction.json) preserves the chronology.

H303's observed running source matches its closeout source. Its declared pre-outcome cache-load optimization changes only materialization frequency; it preserves both selected TRAIN values and the mixed-cache read issue. Model reproduction, lock verification, source review, lint/format and exact-head diff checks support this report. Hosted CI is **NOT_RUN** under the owner freeze.

Monitored work used xbabe2, nice 19, cores 60–63, one numeric thread, RSS ≤4 GiB, MemAvailable ≥20 GiB, host CPU ≤85% and scratch ≤3 GB. One supplemental tar metadata/hash extraction inadvertently ran outside the supervisor for a few seconds; it is recorded as a scheduling-control deviation, not certified compliant. It performed no numerical scoring or foreign-process action. No GPU, other-host/live-job action, main/fastlane change or CI dispatch occurred. Absolute compute stop 08:30Z and close 08:45Z were retained.

[metrics.json](metrics.json), the separate per-wave statistics/regression receipts, [source_pins.json](source_pins.json), [environment.json](environment.json) and [PREREG.md](PREREG.md) provide compact evidence. Evidence is at `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H305/astra-20261104T055813Z/h305_evidence.tar.gz`, VersionId `msfob314CaZPinXSrZwkIeQEy7k8wj9.`, SHA256 `8fd13d825b08e97d075bf820911b939d4df7f94f8c04b46b9ab8925cb04920a4`. Exact-version GET and all 162 archive member hashes verified. [manifest.json](manifest.json) binds execution code `7919046451cef265b635745c433970fc47c17a4c`, source/input identities and resource limits; [s3_manifest.json](s3_manifest.json) pins its separate S3 version. [cleanup_receipt.json](cleanup_receipt.json) records deletion of all 747 own scratch files (433,398,988 bytes) after recovery mapping verification and a clean privileged consumer check. Scratch is absent; compact source/receipts remain. Local compact custody: `/home/ubuntu/jope-rescue-20260921/h305-astra`.

## Next step

Retain H303 as a **conditional TRAIN selector result**, correct the r882 reference and read-boundary claims, and isolate TRAIN-only inputs before another wave. Keep historical design exposure explicit. Fresh confirmation requires a genuinely new authorized population; this audit authorizes no serving promotion or new experiment.
