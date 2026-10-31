# H135 — prospective A3 GBM thread continuation

**A3 preregistered; not executed.** The explicit owner directive `rd_astra_t1605_h135a3.md` authorizes one continuation after [PREREG_AMENDMENT_A3.md](PREREG_AMENDMENT_A3.md) merges. The earlier incomplete results below remain unchanged.

The eight-thread synthetic failure is reproduced in HistGB binning (joblib n_jobs4). A launcher now sets thread environments before interpreter startup and caps native pools and nested limit requests at one. The same 300-iteration HistGB recipe on 2,048 × 1,089 synthetic inputs completed with maximum one thread under external sampling. This corrects the earlier pre/post-only fixture, which missed transient threads. Scientific runner, imported recipes, sampling, seeds, baseline parity and statistical gates stay byte-identical.

Next: central review/merge, verify actual source, repeat synthetic preflight, refresh seed-use receipts, then one continuation starting with the 91-context r882 parity gate. No outcome has been used to change the recipe. Report original unknown partial fit progress honestly. A further execution failure stops; no automatic fourth attempt. CPU and all original resource/data guards remain in force.

---

# H135 — fixed BLEND_025 on flywheel validation

**INCOMPLETE; blend outcome unmeasured.** This is the single prospective continuation authorized by [Amendment A2](AMENDMENT_A2.md), merged in [PR #1200](https://github.com/neverhuman/JopeDime/pull/1200) before execution. The original incomplete attempt and its custody remain in [metrics.json](metrics.json) and the historical record below.

Hypothesis: the fixed H129 BLEND_025 recipe improves the flywheel’s own 91-context regression validation utility over r882. The primary weight is 0.25; PURE_GBM and BLEND_050 are report-only. Native protocol is `fixed-dev-cost-argmin-negative-fitness/v1`: full-pool first argmin, utility negative fitness when valid and −2 otherwise, with no U4 clipping.

Setup: 1,219 DL-Train contexts / 205 families; 91 DL-Val contexts / 19 families; zero family overlap and zero validation training rows. The fixed plan samples, for each of seeds 135–137, 192 candidates per training context using sampling seed 135000+seed and an independent HistGB random_state. The unchanged imported H124/H129 model recipe is used. Calibration is the explicitly preregistered full-train within-context adaptation, not H129 nested family cross-fitting. There is no weight search or best-seed selection.

Baseline gate **PASS** before any fit or blend: mean **1.1377941898164525**, maximum per-context error **0.0**, mean error **0.0** against the pinned 91-context reference.

The continuation stopped during seed 135 after the external guard observed 8 worker threads (limit 4). No seed prediction archive or completed fit receipt was produced. Internal fit progress is unknown; zero completed receipts does not prove zero fitting work. Seeds 136/137 never started, and no blend utility was computed. The baseline audit and DOPE pre/post check completed; the seed 135 read audit did not flush. Missing gains, Elo, intervals and admission remain unknown. No third automatic attempt is permitted.

Resources: 398 external samples, peak RSS 6,073,786,368 bytes, maximum 8 worker threads, peak scratch 8,514,991 bytes, maximum root usage 92.8688%. CPU only, nice 19; unchanged limits are four total threads, RSS below 8 GB, scratch below 2 GB, root below 93.8%, load at most 85% of available CPUs and at least 16 GiB available memory. No GPU, qualification lock or live-loop writes.

Interpretation and limitations:

- r882 was selected as the maximum over approximately 2,385 rounds on this same validation metric. That selection favors r882, while the selector-exposed validation set cannot establish sealed-test generalization. A pass is only CANDIDATE_REVIEW_ONLY; it never authorizes serving promotion.
- The primary requires every preregistered gate: median gain strictly above 0.001, positive one-sided 95% family-bootstrap LCB, all three seeds positive, at most 10% of families harmed below −0.01, positive top-10-trimmed gain and distinct utility vectors. Report-only arm success cannot rescue a failed primary.
- Seed audit was refreshed immediately before launch. H133 reserved 135, but its latest observed execution used 133 only; no prior 135–137 scientific execution was observed in bounded local receipts/refs. This is not an exhaustive external or S3 history. Synthetic fixture fits were disclosed separately.
- Training excludes validation families. The selected train+validation packet and explicit validation pools are authorized; protected sealed, holdout and meta-test inputs are not. Missing runtime audits from the first aborted attempt remain a documented limitation, not a zero-read proof.

Validation: A2 passed 12 data-free tests, native one-thread constructor preflight and five local records gates. Closeout validation checks JSON, links, source/prereg pins and the complete documentation diff. These are same-agent checks; full CI remains deferred under the owner targeted-local policy.

Next step: Stop after this continuation; retain evidence without another automatic retry or budget reset.

Verified numerical custody: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H135/astra-a2/manifest.json`, VersionId `unhuGh1cUZNOWybJazyaZQEC5.lYyZeh`, SHA256 `b0e17eee25e915855035f01fca3a44a761583d81b9c2e7746e5913f543928313`. Archive `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H135/astra-a2/numerical-evidence.npz`, VersionId `gDFRqKSTbKZI1oa5MFtWIL5tDTIPtpJi`, SHA256 `38836aef288e4b1ccde5473f4b0f6db4a5af9e80bae94b7166ccd4a6e5e8b07b`; 17 members restored and hash-checked. Code stays in Git at `b6d4c6b232a741c2328535fd856b89e1871e6ad3`.

Recovery: GET the exact archive version, verify SHA256, load NPZ with `allow_pickle=False`, then verify each uint8 member against numerical-inventory.json. Original H50/H58 derived inputs and r882 remain pinned local inputs; their exact S3 recovery versions were not established by H135.


Cleanup: deleted only the nine owned run-scratch files (8514991 bytes) after exact archive/member recovery checks; the duplicate r882 checkpoint was independently hash-matched to its retained authorized input. The immutable S3 evidence and local lifecycle receipts remain.

---

# H135 — fixed BLEND_025 on native flywheel validation

**Attempt1 INCOMPLETE; no scientific verdict.** Baseline setup exceeded the four-total-thread cap before the checkpoint-load receipt and prediction loop. No baseline prediction, GBM fit or blend result was produced. [metrics.json](metrics.json) preserves null metrics, the failure and exact-version S3 custody; no gain/Elo/admission is inferred from the stop.

Hypothesis/setup remain fixed: H129 quarter blend versus r882 on91 native argmin validation contexts, three full DL-Train fits, independent192-candidate samples and training-only calibration. PURE_GBM/BLEND_050 are report-only. The original [protocol](PREREG.md) and [A1](AMENDMENT_A1.md) remain intact.

[Prospective A2](AMENDMENT_A2.md) caps baseline Torch at1 while retaining the imported baseline calculation and four-total-thread external ceiling. An unscored model-construction diagnostic completed at1 thread without datasets, r882 loading or prediction. A2 permits one reviewed continuation after merge, fresh seed-use audit and constructor preflight, using new custody. It does not relax the mandatory r882 vector gate or any statistical gate. No third automatic attempt.

Outcome metrics are unmeasured. Next step: merge/verify A2, then execute that continuation once. A failure remains INCOMPLETE; a statistical pass is only CANDIDATE for adoption review, never serving promotion. Ten prior controls plus two cap/restoration tests and the five records gates validate the implementation; full CI is deferred under owner policy.

Historical preregistration README follows; its original NOT EXECUTED wording describes the pre-attempt state.

---

# H135 — fixed BLEND_025 on flywheel validation

**PREREGISTERED / NOT EXECUTED.** [Protocol](PREREG.md) and [source/input pins](manifest.json) freeze one three-seed CPU attempt. Launch waits for preregistration merge and a fresh receipt audit for prior seed execution; H133 also reserves135.

Hypothesis: the fixed H129 quarter blend improves r882 on the native 91-context full-pool argmin metric. Setup: three full DL-Train HistGB fits, independent192-candidate samples, training-family exclusion, mandatory r882 vector reproduction before fitting, BLEND_025 primary; PURE_GBM/BLEND_050 report-only. The existing fit/blend/calibration functions are imported; full-train within-context calibration replaces nested cross-fitting and is explicitly preregistered.

Outcome, gains, Elo, confidence intervals and failure counts are **unmeasured**. No scientific verdict or promotion. Targeted data-free tests cover native scoring, context weighting, sanity refusal, harm/trim/seed gates and resource ceilings; full CI is deferred under the owner targeted-local policy. Next step: merge exact reviewed source and refresh the prior-use audit, then launch once with external guards. Closeout must publish all outcomes and verified numerical S3 custody.

[Amendment A1](AMENDMENT_A1.md) disables PyArrow allocator background threads before imports, after a synthetic probe exposed five total threads in the original HistGB environment. No scientific run has started. Original protocol/runner/science are preserved; ten data-free controls include the real native-library thread regression. Execute only after the amendment merges.
