# Mission: Optional and Progressive Landmarking for Dope–JopeDime Fusion

**Status**: Proposed · 2026-10-27
**Owners**: JopeDime / Dope fusion track
**Related**: Dope MLC-v3 MultiLevelLandmarkModel, JopeDime encoder → policy pipeline, H27 landmark-routing study

## Executive Goal

Make landmark conditioning **optional, progressive, and extremely fast** inside the pre-trained Dope encoder that JopeDime consumes.

Today the Dope MLC-v3 encoder is conditioned on a fixed panel of 23 cross-validated landmark gains (plus data descriptors and inter-landmark statistics). Computing that panel is the dominant cold-start cost. We want:

1. **Fast / cold-start path** — zero landmark computation. Encoding proceeds immediately with a well-defined “no-landmarks” conditioning vector (all masks false, values zeroed). Target: encoding latency dominated by the base table transform and forward pass only.

2. **Progressive warm-up** — landmarks arrive asynchronously (none → partial → full). As each landmark (or group of landmarks) becomes available, the conditioning vector is updated and the encoding is refreshed. JopeDime’s fused representation and downstream policy can benefit without waiting for the full panel.

3. **Minimal viable landmark tests** — support stratified down-sampling (candidate sizes 100–500 rows, starting recommendation 300) so that even “full” landmark mode can be made much cheaper while preserving most of the conditioning signal.

The result is a single encoder surface that degrades gracefully:

```
no landmarks  →  partial landmarks  →  full landmarks
   (fast)            (warm-up)             (best)
```

All three modes must produce valid, deterministic encodings that the pre-trained MLC-v3 weights can consume without retraining the core model (or with only light adapter fine-tuning if measured retention requires it).

## User Scenarios

### Scenario A – Ultra-fast “no landmarks” mode
- User / appliance requests `--landmarks none` (or a new “fast” profile).
- Landmark computation is completely skipped.
- Dope encoder receives a zero-value / all-false-mask landmark vector plus the usual train-only table descriptors.
- Encoding returns in the time of the base forward pass.
- Downstream JopeDime policy / program search proceeds with this lower-fidelity but immediately available representation.
- Later, if the user wants higher quality, landmarks can be requested and the encoding updated.

### Scenario B – Progressive warm-up (cold-start + parallel AutoML)
- Encoding starts immediately with the zero-landmark path (Scenario A).
- In parallel, a landmark worker begins:
  - First: the 6–9 cheapest Macro landmarks (linear, poly2, ridge, etc.) on a stratified subsample.
  - Then: larger / interaction landmarks.
  - Finally: fine landmarks (knn, rff, mlp, catboost, …).
- Each time a new landmark (or batch) completes, the conditioning vector is patched, the Dope encoder is re-run (or incrementally updated if a stateful path is implemented), and the new encoding is published to JopeDime.
- The system can surface intermediate quality milestones (“macro landmarks ready”, “full panel ready”).

### Scenario C – Full landmarks with cheap tests
- User requests full conditioning but with `--landmark-subsample 300` (stratified).
- The 23-landmark panel is computed on the subsample only.
- Retention and routing quality are measured against the full-row baseline; the subsample size is tuned so that the drop is small (target ≤ 5 % relative on key metrics).

### Scenario D – Partial / flagged inputs from external systems
- An external AutoML or meta-feature service supplies only some of the 23 landmarks (or supplies NaNs for the rest).
- The encoder treats missing entries as masked (value = 0, mask = false) exactly as the existing `landmark_features.py` / `learned/landmark_features.rs` path already does.
- Partial vectors are first-class.

## Success Criteria

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Fast-mode latency | ≥ 5× faster than full-landmark encoding on typical tables (1 k–50 k rows) | Wall-clock encode time, cold start |
| Zero-landmark retention | Within 5–8 % relative of full-landmark retention on the existing MLC transfer suite | Downstream R / feature-importance metrics |
| Progressive monotonicity | Adding landmarks never degrades the encoding quality by more than noise; ideally improves it | Paired retention on the same tables |
| Minimal subsample | Stratified N = 300 retains ≥ 95 % of full-row landmark-conditioned retention | Ablation study (N = 100, 200, 300, 500, 1 000 vs full) |
| Determinism & compatibility | Bit-identical encodings for identical inputs; existing full-landmark paths unchanged | Unit + integration tests |
| Mask correctness | Missing landmarks are exactly zeroed + mask=false; never leak NaNs into the network | Existing + new mask tests |

## Non-Goals

- Retraining the core MLC-v3 weights from scratch (we prefer zero-shot or light adapter use of the existing pre-trained model).
- Changing the hierarchical level structure (Small / Medium / High / Ultra) beyond what is required to accept partial conditioning.
- Real-time incremental updates *during* prediction of individual rows (focus is the fit / encode phase).
- Replacing the existing Hyperion / foundation / target / chimera fusion path; Dope is an additional (or alternative) encoder source that can be fused into the same z-vector.

## Open Questions (to be resolved in the engineering spec)

- Exact mapping of a Dope encoding (or its conditioning latent) into JopeDime’s 1536-d fused z.
- Whether progressive updates re-run the full forward pass or use a lighter delta path.
- Whether the pre-trained model already generalizes well to zero-landmark inputs (the H27 study suggests the signal is useful but not yet proven for routing; we must measure retention).
- Minimal subsample size that keeps landmark value variance acceptable.

## Deliverable

A detailed engineering specification that an implementation agent can execute end-to-end, covering both repositories, exact files, data structures, APIs, test plans, and verification criteria.

---

## ADDENDUM — Jepson requirements (Oct 27 2026, binding)

1. **Landmarks are optional by training, not just by plumbing.** Train the Dope encoder with landmark dropout / random partial masks (none, partial, full) so it produces a strong encoding with zero landmarks and gets better when they are present. Report quality at each mask level.
2. **Pinned encoder contract.** JopeDime depends on a pinned Dope encoding layer. The same pinned weights must accept a bundle with or without landmarks; fixed output dimension either way.
3. **JopeDime choice (evaluate both, keep both optional):** (a) consume only the Dope encoding (landmark benefit flows through Dope), and/or (b) also feed the same encodings into JopeDime's own deep-learning model. Ablate and document.
4. **Flywheel must keep working.** The self-play/data flywheel stays live and green through every PR; add a flywheel regression test.
5. **Fast inference with no landmarks** is a first-class JopeDime mode, with a latency target and test.
6. **World-class binary, regression, and multi-class** support, with tests and benchmarks for each.
7. **Inference is 100% Rust, no Python.** Weights ship as safetensors and load in Rust. Python is allowed for training/research only.
