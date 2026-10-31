# H135 prospective Amendment A3 — cap the GBM launcher

Authorization: owner-directed `~/nudges/rd_astra_t1605_h135a3.md`, 2026-10-31.
This later direction explicitly supersedes A2's no-third-attempt disposition for
**one** further continuation after this amendment merges. It does not waive any
resource or scientific gate. The incomplete closeout [PR #1204](https://github.com/neverhuman/JopeDime/pull/1204)
merged first at `40f4dfff971e90bcca0a95d83981bd2c6d5aa150`.

Both earlier attempts remain preserved. Attempt 1 stopped during baseline setup.
Attempt 2 reproduced all 91 native r882 utilities exactly (mean
1.1377941898164525), then stopped when seed 135 reached eight OS threads. There
was no completed seed prediction archive or fit receipt and no blend outcome.
Internal fit progress and the unflushed seed read counts remain unknown. They
must not be relabeled zero. Its immutable numerical manifest is
`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H135/astra-a2/manifest.json`,
VersionId `unhuGh1cUZNOWybJazyaZQEC5.lYyZeh`, SHA256
`b0e17eee25e915855035f01fca3a44a761583d81b9c2e7746e5913f543928313`.

## Diagnosis and correction

Installed sklearn's `_hist_gradient_boosting/gradient_boosting.py` obtains its
thread count from `_openmp_effective_n_threads`; `binning.py:235` passes it to
`Parallel(n_jobs=self.n_threads, backend="threading")`. The previous scoped
OpenMP limit of four therefore selected four joblib workers. Python's
`multiprocessing/pool.py:227-248` adds three management threads, alongside the
main thread. A synthetic reproduction captured `n_jobs=4` immediately before
an external `/proc` sampler observed eight threads and stopped the child.
The A1 before/after-only check missed this transient binning pool; its result
was not proof that peak total threads stayed within four.

Only execution control changes. `monitor.py` sets OMP, MKL, OPENBLAS, NUMEXPR and
RAYON to one **before interpreter start**, retaining the A1 Arrow allocator cap,
CPU-only environment and nice 19. Numerical seed stages enter `launcher.py`.
The launcher loads sklearn's native libraries, applies
`threadpoolctl.threadpool_limits(limits=1)`, and scopes nested requests to one
with exception-safe restoration. This is necessary because the unchanged H135
runner requests OpenMP four inside its fit context. With that request capped,
sklearn binning chooses joblib `n_jobs=1` without changing estimator parameters.
Baseline and summary continue to use the unchanged runner directly.

`runner.py`, `science.py`, original PREREG, A1, A2 and the pinned H124/H129 modules
remain byte-identical. No estimator hyperparameter, sampling, calibration,
weight, metric, family roster, seed, bootstrap, harm or admission gate changes.
No estimator code is copied into the launcher. The synthetic preflight is a
separate diagnostic; it never opens study datasets, checkpoints or metadata.

## Prospective execution and budget

After central merger review and actual merged-source verification, repeat the
external synthetic preflight, refresh the seed-use audit, and start exactly one
new custody directory. The prior seed 135 partial/unscored attempt is explicitly
retained; this is an authorized continuation, not an assertion of virgin seeds.
Report external reservations/executions separately and do not silently replace
seeds. The fixed seeds remain 135, 136, 137; samples use 135000+seed. The owner
explicitly retains the three-fit continuation budget. Earlier unknown internal
work remains disclosed outside those three required completed fits.

First reproduce the ordered 91-context r882 vector and mean within 1e-6 from
an authorized offline checkpoint copy. Any parity failure stops before fitting
or blend reads. Fit only DL-Train after excluding every validation family;
calibration remains the preregistered unsampled-training-candidate adaptation.
Read no sealed, holdout or meta-test metadata or payload. DOPE stays
`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`.

The primary is fixed BLEND_025; PURE_GBM and BLEND_050 remain report-only.
All original primary gates apply: median gain >0.001, one-sided family-cluster
bootstrap LCB95 >0, 3/3 positive seeds, harm <=10%, positive top-10-trimmed gain,
and distinct seed utility vectors. No outcome-driven changes or selection.
A pass is CANDIDATE_REVIEW_ONLY, never a serving promotion. The selector-exposed
validation and r882 winner-selection caveats remain unchanged.

CPU nice 19, at most four total worker threads, RSS below 8 GB (the imported
helper also retains its stricter 6 GiB check), scratch below 2 GB, root usage
below 93.8%, load at most 85% of available CPUs and at least 16 GiB available
memory. No GPU, qualification lock or live-loop writes. Any execution failure
stops this continuation; no automatic fourth attempt or budget reset.

Closeout preserves source hashes, baseline/fit/read/resource receipts and all
fixed arms through numerical-only, create-only S3 custody under
`experiments/rd_breakthrough_20261030/H135/astra-a3/`, with exact-version GET and
member hashes checked before owned scratch cleanup. Publish README, metrics
below 100 KB and the single index row through central review. Report native
utility and family-cluster uncertainty plus Elo anchored at r882=1500, not AUC.

## Data-free evidence pinned before scientific continuation

Synthetic matrix: 2,048 rows × 1,089 features, RNG0, estimator random_state135. Original wrapper: external guard stopped at8 threads, binning n_jobs4. Amended wrapper: 300 iterations and prediction completed, maximum1 thread across1,172 external samples, peak RSS328900608 bytes, wall88.49s, binning n_jobs1. No study metadata or payload reads. Finite sampling is a regression proof, not full-study resource qualification.

- `/home/ubuntu/jope-rescue-20260921/h135-gbm-thread-a3/diagnosis.json` SHA256 `524d0e3e338014bf3d136a2176a8d505ec164d3351bdd3743701777bf335b523`.
- `/home/ubuntu/jope-rescue-20260921/h135-gbm-thread-a3/original-preflight/receipt.json` SHA256 `dbfebacbb3a617c6b0714ec7102a1af3de2a50362cbd45f209538b10ab44b6c1`.
- `/home/ubuntu/jope-rescue-20260921/h135-gbm-thread-a3/amended-preflight/receipt.json` SHA256 `cbdb067327d33cdbbef2940715f8f29ac19f50fb3838f2ef04bd2ed8479ef443`.
- Launcher SHA256 `3e101d24ba691812551784b1dada528ba4c63d3a262d69e036e7520e5a75bd82`; preflight SHA256 `d6bacacb91f5bb05ecbd8e8cc2bf474ab22d1c1683f2f51f290241a72dbc31e6`.
