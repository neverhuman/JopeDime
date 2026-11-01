# H186 — zero-update kernel dose and shape sweep

Owner rd_astra [ASTRA]; authority `/home/ubuntu/nudges/hwave_t0730_h186_astra.md`. One CPU attempt after reviewed preregistration merge; no retries or grid extension after outcomes. H184 (#1544, c6f678236633689607eec23f82bba8e34dde4f41) motivates testing the revealed-history residual kernel. This is an intervention on r882 constants, not fresh random weights.

## Fixed treatments

R882 is unchanged. Seven arms change exactly one constant in the H184 component scorer; every learned tensor stays byte-identical to the private r882 checkpoint copy. All optimizer constructions and backward calls are forbidden, every parameter has gradients disabled, and final adapter/frozen/DOPE hashes must match initial values.

| Arm | Only changed constant |
|---|---|
| KX0.5 | kernel coefficient 0.45 × 0.5 |
| KX1.5 | kernel coefficient 0.45 × 1.5 |
| KX2.0 | kernel coefficient 0.45 × 2.0 |
| TOPK4 | nearest-history cap 8 → 4 |
| TOPK16 | nearest-history cap 8 → 16 |
| W0X0.5 | complete null weight `(softplus(raw) + 0.15)` × 0.5 |
| W0X2 | complete null weight `(softplus(raw) + 0.15)` × 2 |

Distance scales, learned blend, attention, all learned weights and reveal protocol stay fixed. A changed null weight also propagates through `sem_adj` into the existing attention head: that is the native downstream effect of this single constant, not an output-only ablation. H184's PRIOR+KERNEL+ATTN arithmetic order is preserved. Data-free AST comparison permits only the three registered configurable expressions. Every scored batch proves default components equal H184 and sum bit-exactly to the native adapter delta.

The inherited full-pool evaluator reveals exactly the first four stored candidates. Since `min(topk, K_rev)` has `K_rev=4`, TOPK4/TOPK16 are structural no-ops and must produce exact R882 costs. They remain in the registered seven-arm multiplicity; no deeper reveals or replacement arms are authorized. Full candidate pools, original fitness/gate labels, first argmin tie behavior, and utility `-fitness` for gate-valid choices / `-2.0` otherwise remain intact. Exact utility `-2.0` counts and gate-invalid counts are reported separately.

## Inputs and folds

Exact 91-context native r882 baseline must pass **first**, bit-exact per context and mean `1.1377941898164525`, before TRAIN scoring or any treatment. This required reference-only DL-Val gate is the sole pre-B-lock validation exception. Candidate DL-Val scoring occurs only after the lock. DOPE composite must remain `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`.

Only the existing real-S3-derived 1,219 TRAIN contexts / 205 families and explicitly authorized 91 DL-Val contexts / 19 families are used. The pinned 276-dataset catalog join, context-feature cache, pool hashes, private read-only r882/reference copies and external model/runtime source hashes are in [manifest.json](manifest.json) and [provenance.json](provenance.json). No sealed/holdout/meta-test payloads or live flywheel state are inputs. Labels stay intact. Historical TRAIN exposure and repeated DL-Val use remain limitations.

Seeds 186101–186103 independently reshuffle the 205 families using `H177.stream_seed(186140, seed, 'fold-reshuffle')`. Each has five cyclic 123 FIT / 41 B / 41 A family folds. FIT is unused (zero updates), next cyclic fold is B, outer fold is A. Each family/context appears exactly once in pooled B and once in pooled A per seed. The committed [fold_manifest.json](fold_manifest.json) binds all three distinct reshuffles. These seeds are **not independent fitted models**; identical context policies/outcomes across folds or seeds are explicitly reported, not treated as independent replications.

## Locked two-stage analysis

1. Score all B folds for all three reshuffles. Per-candidate raw costs, choices, utility and invalid flags are retained, alongside context/family role receipts. No A or treatment DL-Val scoring is allowed yet.
2. Import `pooled_b_select`, `H177_DECISION_RULE`, `StageOrderGuard`, `verify_pooled_b_coverage`, and `resolve_h177_verdict` from pinned H177 source. Rebind only the private imported module's arm/fold/candidate-version configuration, never its function bodies. Seven candidate panels enter its actual `holm()` calculation, so **m=7**. H177's literal output metadata `holm_multiplicity=4` is retained as a legacy metadata field and corrected to actual m=7; the original decision object is not represented as a newly calibrated seven-arm object. Positive candidate step slot 1 means **candidate version 1**, never an optimizer update. The lock retains `wave=H177` solely for imported guard schema compatibility, with `study=H186` and zero-update semantics explicit.
3. H177 pools equal-context-within-family utility, averages seeds within each family, then uses 10,000 shared paired family resamples (seed 186141), ordinary 95% CI, and seven-arm Holm. Admission requires mean gain `> 0.001 + 1e-12`, LCB > 0 and Holm p < 0.05. Ties reject. The companion median-seed gain/CI is also retained. Write and fsync `pooled_b_selection_lock.json` with all seven decisions and B panel/adapter hashes before proceeding; pin its SHA and reject mutation.
4. Only then score all A folds and DL-Val. Pooled A reuses the same 205 families under rotated roles and the same fixed policies: it is a **correlated readout, not independent confirmation**. Every A/DL-Val call checks the imported guard and immutable lock hash. No selection, constant changes, best-of-k, refitting, or adaptive retries follow.
5. For B, A and descriptive DL-Val report each arm's per-seed gain, median seed gain, 10,000-draw paired-family CI reducing over seeds on **axis=0**, positive-seed count, seven-arm Holm p, Elo vs R882=1500 and CI, exact -2.0 counts and gate-invalid counts. Elo uses imported canonical H177/H148 400-iteration BT MM with prior .01 and context tie tolerance 1e-4, then equal-family wins. Elo CI is median across the three seed ratings in each of 1,000 shared-family bootstrap draws (seed 186142).

ADVANCE requires B admission **and** pooled A median gain > 0, 3/3 seeds positive and median gain LCB > 0; `resolve_h177_verdict` must also accept the committed H178 receipt at the exact H177 analysis hash. H178 empirically calibrated the original four-arm rule; this seven-arm configuration uses stricter Holm multiplicity but is **not separately empirically calibrated**. This limit, correlated A, identical seeds if observed, and prior exposure accompany any ADVANCE. DL-Val never selects or gates promotion. Diagnostic dose-response uses ordered KX0.5 / R882 (×1) / KX1.5 / KX2.0 A median gains and both nondecreasing/nonincreasing flags; no causal monotonicity inference from a noisy ordering.

## Resource, custody and disposition

CPU xbabe2 only, nice 19, math libraries one thread, supervisor plus worker aggregate ≤4 OS threads and RSS <6 GB. Terminal stops: host CPU >85%, load >85% of online cores, MemAvailable <16 GiB, user-visible disk usage ≥93.8% or free <40 GB. One 4-hour scientific wall cap; no reset on resume. Application audit records protected-read refusals; it is not an OS sandbox. No GPU, xbabe0/xbabe3, live loop edits, qualification lock, serving changes, speed metrics or AUC.

Preregistration → rd_argon cross-review bounded to 15 minutes (silence is not approval) → targeted pytest/Ruff/static records gates and exact-head author review → squash merge → single guarded attempt → independent saved-score/statistical replay → create-only versioned S3 custody → small closeout PR and H186 index row → merge → verify cleanup and remove the owned lane. Failed/incomplete execution stays INCOMPLETE, not NO_ADVANCE. No second scientific attempt.

All long-lived evidence goes under `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H186/astra-t0730/`; exact VersionId/SHA256 readback and member verification precede local bulk deletion. Metrics JSON stays below 100 KB. Any ADVANCE is a research candidate only: flag `@flywheel_gp_fused` in AGENT_CHAT; adoption belongs to that owner with DOPE checks. H186 performs no adoption.
