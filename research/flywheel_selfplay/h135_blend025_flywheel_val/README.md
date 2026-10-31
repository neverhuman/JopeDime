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
