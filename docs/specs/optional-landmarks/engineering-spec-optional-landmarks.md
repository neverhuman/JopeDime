# Engineering Specification: Optional and Progressive Landmarking for Dope Encoder in JopeDime

**Status**: Ready for implementation · 2026-10-27
**Mission**: [mission-optional-landmarks.md](mission-optional-landmarks.md)
**Scope**: Both `neverhuman/dope` and `neverhuman/jopedime`
**Goal**: An implementing agent can execute this end-to-end with no further clarification.

## 0. Design Principles

1. **Leverage existing mask support**. Dope already represents landmarks as `(values: [f32; 23], mask: [bool; 23])` and zeros missing entries. We make this first-class and expose it at every API boundary.
2. **Zero-shot preference**. Prefer using the existing pre-trained MLC-v3 weights with zero/partial masks. Only introduce light adapters or fine-tuning if measured retention gaps are unacceptable.
3. **Progressive is additive**. Landmarks arrive in a fixed cheap-to-expensive order. Each update is idempotent and versioned by a conditioning hash.
4. **Determinism**. Identical inputs (including mask) produce bit-identical encodings. Full-landmark mode remains unchanged.
5. **Rust-first long-term**. Python research path may be used for rapid prototyping; the product surface must have a Rust API (consistent with JopeDime ADR-0006).

## 1. Landmark Representation (Shared Contract)

### 1.1 Canonical structure

```rust
// Both repos (or a shared schema)
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LandmarkBundle {
    /// Squashed gains (value / (1 + |value|)). Missing slots are exactly 0.0.
    pub values: [f32; 23],
    /// True if the corresponding landmark was computed and is valid.
    pub mask: [bool; 23],
    pub status: LandmarkStatus,
    /// Number of rows used for the CV (None = full training partition).
    pub subsample_n: Option<usize>,
    /// SHA-256 of the source rows + options used to produce this bundle.
    pub source_digest: String,
    /// Optional provenance (e.g. "native-v1", "subsample-300-stratified").
    pub provenance: Option<String>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum LandmarkStatus {
    None,          // all mask false
    Partial(u8),   // 1..=22 landmarks present
    Full,          // all 23 present
}
```

**NaN protocol**: Any incoming NaN or non-finite value in a landmark slot is treated as missing (value = 0.0, mask = false). This is already enforced in `landmark_features.py::validate_values`.

**Zero bundle**:
```rust
LandmarkBundle {
    values: [0.0; 23],
    mask: [false; 23],
    status: LandmarkStatus::None,
    subsample_n: None,
    source_digest: "zero".into(),
    provenance: Some("fast-path".into()),
}
```

### 1.2 Landmark order (cheap → expensive)

The 23 landmarks from `landmarks-native-v1` / fit-cv-v1 (order used for progressive arrival):

| Priority | IDs (approx.) | Family | Typical cost |
|----------|---------------|--------|--------------|
| 1 | linear, ridge, poly2 | Macro / linear | Very low |
| 2 | poly3, bspline, pchip, monotone, periodic, hinge | Macro | Low |
| 3 | pair, otree2, … (interaction) | Larger | Medium |
| 4 | knn, kreg, rff, mlp, catboost | Fine / kernel | Higher |

Exact IDs are taken from the existing panel definition in `engine/dope-core/src/landmark/`. Progressive mode may emit intermediate bundles after each priority group.

## 2. Dope Repository Changes (`neverhuman/dope`)

### 2.1 Research / Python path (primary for rapid implementation)

**Files to modify or create:**

| File | Change |
|------|--------|
| `research/levels/landmark_features.py` | Add `LandmarkBundle` dataclass (or plain dict). Methods: `zero()`, `from_optional_list(values: List[Optional[float]])`, `merge(other)`, `to_tensor(device)`, `status()`. Update `report_features` and `feature_tensor` to accept partial bundles. |
| `research/levels/multilevel_model.py` | Ensure `forward_conditioning`, `LandmarkHyperRouter`, and target net paths are robust to all-false masks. Add explicit `encode_table(..., landmarks: Optional[LandmarkBundle] = None)`. Zero-mask must not produce NaNs. Add smoke test path. |
| `research/levels/building_block_landmarks.py` | Add optional `subsample: Optional[int]` argument to computation entry points. Fast-path skip when landmarks status is None. |
| `research/levels/descriptors.py` (or equivalent) | Confirm base table descriptors remain available even when landmarks are zeroed. |
| **New** `research/levels/optional_landmarks.py` | Helpers: stratified subsample, progressive merger, conditioning hash. |
| **New** `research/levels/tests/test_optional_landmarks.py` | Unit tests (see §5). |
| `research/levels/s3_multilevel_trainer.py` | Optional ablation flag for zero/partial training (document only if not executed). |
| `docs/architecture.md` | Add subsection “Optional and progressive landmarks”. |
| `docs/multilevel-landmark-dl-study.md` | Note zero-mask and partial support. |
| `docs/cli.md` | Document new flags. |

**Key function signatures (Python):**

```python
def zero_bundle() -> LandmarkBundle: ...
def from_nans(values: Sequence[float | None]) -> LandmarkBundle: ...
def merge(base: LandmarkBundle, incoming: LandmarkBundle) -> LandmarkBundle: ...
def feature_tensor(bundle: LandmarkBundle, device) -> torch.Tensor:  # 46-d
def encode_with_landmarks(model, table, bundle: Optional[LandmarkBundle], level: str) -> Tensor: ...
```

### 2.2 Rust / engine path (product surface)

| File / Module | Change |
|---------------|--------|
| `engine/dope-core/src/learned/landmark_features.rs` | Already loads 46-d masked features. Add public `LandmarkBundle` type and `from_partial` / `zero` constructors. |
| `engine/dope-core/src/landmark/mod.rs` | Expose options for subsample size (already has `small_rows`, `ridge_rows`). Add progressive status. |
| `engine/dope-core/src/levels.rs` | Ensure level fitting accepts optional landmark signature. |
| CLI facade (`rust/main/facade/` or `dope` binary) | Add `--landmark-mode {none,fast,progressive,full}` and `--landmark-subsample N`. |
| `docs/architecture.md`, `docs/cli.md` | Document modes and NaN protocol. |

### 2.3 Minimal subsample support

- Stratified sampling: for regression, quantiles of the target; for classification, by class.
- Candidate sizes to evaluate: 100, 200, 300, 500, 1 000.
- Implementation lives in `research/levels/optional_landmarks.py` (or a small Rust helper) and is called from the landmark panel runner when `subsample_n` is set.
- The existing `LandmarkOptions` already has row caps (`ridge_rows`, `small_rows`); wire them.

## 3. JopeDime Repository Changes (`neverhuman/jopedime`)

### 3.1 Encoder integration

**New or extended files:**

| File | Change |
|------|--------|
| **New** `crates/jope-encoders/src/dope_encoder.rs` | `DopeEncoder` implementing the existing encoder trait (or a new one). Accepts `Option<LandmarkBundle>`. Produces a fixed-size embedding (recommended 512-d to slot into the Hyperion position) or a full conditioning vector that is projected. |
| `crates/jope-encoders/src/lib.rs` | Export the new module. |
| Fusion site (Hyperion / chimera / policy context) | Support Dope-conditioned path. Options (choose one in implementation): (a) replace Hyperion-512 slot, (b) concatenate and project back to 1536, (c) gated residual. Default: replace Hyperion slot when Dope is selected. |
| `crates/jope-pipeline/src/` (pipeline / encode path) | Add `EncodeMode { None, Progressive, Full }`. Progressive mode spawns background landmark jobs (rayon or async) and publishes updated encodings via a channel or callback. |
| `crates/jope-cli/` | Propagate `--landmark-mode` and `--landmark-subsample`. |
| `crates/jope-appliance/` (if terminal path is in scope) | Fast path for `jopedime fit` / `fit-predict`. |
| `schemas/` | Extend or add schema for encoder context / receipt to include `landmark_status`, `landmark_mask` (or digest). |
| `docs/architecture.md`, `docs/pipeline.md`, `README.md` | Update fusion diagram and document the new modes. |
| `crates/jope-encoders/tests/` | New tests (see §5). |

**Recommended embedding contract:**
- Dope encoder returns a 512-d vector that is robust to landmark quality (zero-mask is valid).
- Optional quality token: a small embedding of `LandmarkStatus` can be concatenated if later ranking experiments show value.

### 3.2 Progressive update flow (JopeDime)

1. Pipeline requests encoding with `mode = Progressive`.
2. Immediately encode with `LandmarkBundle::zero()`.
3. Emit initial context receipt / z-vector to the policy / search.
4. Spawn landmark worker (subsample + ordered panel).
5. On each completed group (or individual landmark):
   - Merge into current bundle.
   - Re-encode.
   - Publish updated z / receipt (with new conditioning hash).
6. Policy / search may optionally re-seed or continue with the improved context.

The update must be idempotent: applying the same bundle twice produces the identical encoding.

## 4. CLI and User Surface

```
# Fast / cold start
jopedime fit ... --landmark-mode none
# or
dope encode ... --landmark-mode none

# Progressive
jopedime fit ... --landmark-mode progressive --landmark-subsample 300

# Full (current behavior, optionally cheaper)
jopedime fit ... --landmark-mode full --landmark-subsample 300
```

Receipts must record `landmark_status`, `subsample_n`, and the conditioning digest.

## 5. Test Plan

### 5.1 Unit tests

| Test | Location | Assertion |
|------|----------|-----------|
| Zero bundle construction | Dope `test_optional_landmarks.py` + Rust | values all 0, mask all false, status None |
| NaN → mask | Same | Non-finite becomes 0 + mask false |
| Partial merge | Same | Mask union, values take non-missing |
| Mask propagation | multilevel_model + feature_tensor | 46-d tensor matches manual zeroing |
| Zero-mask forward pass | multilevel_model | No NaNs, finite output |
| Conditioning hash stability | optional_landmarks | Same bundle → same hash |
| Subsample determinism | optional_landmarks | Fixed seed → identical subsample |

### 5.2 Integration tests

| Test | Location | Assertion |
|------|----------|-----------|
| Fast mode end-to-end | JopeDime pipeline + Dope | Valid encoding / program, latency recorded |
| Progressive simulation | Both | Intermediate bundles produce valid, changing encodings; final matches full |
| Full mode bit-identical | Dope + JopeDime | Matches pre-change full-landmark path |
| Subsample vs full | Dope research script | Landmark vector cosine similarity + downstream retention delta |

### 5.3 Performance benchmarks

- Tables: `datasets/test/*` plus a few synthetic sizes (1 k, 10 k, 50 k rows).
- Metrics: wall-clock encode latency (p50 / p95), landmark compute time separately.
- Target: fast mode ≥ 5× faster than full-landmark mode.
- Record in a small JSON report under `benchmarks/` or Dope `research/levels/`.

### 5.4 Quality / retention suite

- Use existing MLC transfer metrics (retention R, feature-importance dynamics) and JopeDime same-cut benchmarks.
- Arms: zero-landmark, partial (macro only), full, subsample-300 full.
- Success: zero-landmark within ~5–8 % relative of full; subsample-300 ≥ 95 % of full-row landmark utility.
- Document any larger gaps; they become candidates for light fine-tuning.

### 5.5 Backward compatibility

- Existing full-landmark unit and integration tests must pass unchanged.
- Historical receipts and model bundles remain loadable.

## 6. Implementation Order

1. **Dope Python surface** — LandmarkBundle, zero/partial helpers, zero-mask smoke test, unit tests.
2. **Dope Rust surface** — Public LandmarkBundle + CLI flags (can be thin wrappers initially).
3. **JopeDime DopeEncoder skeleton** — Encode with zero bundle, produce fixed embedding, fuse into z.
4. **Progressive hooks** — Background worker + update channel in pipeline.
5. **Subsample study** — Script + ablation report; pick recommended N.
6. **CLI polish, receipts, docs**.
7. **Full verification suite** (unit + integration + perf + retention).

## 7. Verification Commands (example)

```bash
# Dope
cd dope
just fast
python -m pytest research/levels/tests/test_optional_landmarks.py -q
# (after implementation)
cargo test -p dope-core -- landmark

# JopeDime
cd jopedime
cargo test -p jope-encoders -- optional_dope
cargo test -p jope-pipeline -- progressive
# Appliance smoke
bash deploy/appliance/build.sh --profile fixture ...
./target/appliance/.../jopedime fit ... --landmark-mode none
```

## 8. Open Items / Decisions for Implementer

- Exact 512-d projection architecture from Dope conditioning (simple linear is acceptable for first cut).
- Whether progressive updates re-run the full forward pass (recommended for correctness) or attempt a cheaper delta (optional optimization).
- Final recommended subsample N after the ablation (start with 300).
- Whether a quality token is added to the policy input in the first PR (optional; can be a follow-up).

## 9. Done Criteria

- Fast mode works and meets latency target on the test suite.
- Progressive mode produces valid intermediate and final encodings.
- Subsample ablation report exists and a recommended N is documented.
- All new unit and integration tests pass.
- Existing full-landmark tests remain green.
- Docs and CLI help are updated.
- Receipts contain landmark status.

This specification is complete and self-contained for an implementation agent.
