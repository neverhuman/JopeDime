//! Frozen fused forward and parity/placement for RNK-12, spec §5.8/M2.
//!
//! The native DOPE/router, projections and adapter use the selected Candle device.
//! Scalar recipe preprocessing and admission remain CPU operations; learned dense
//! and attention tensors run on the selected backend. Real CUDA qualification is
//! an ignored, lock-gated test with source/bundle/fixture/device receipts.
//! The existing cpu_placement helper still describes CPU-only native evaluation.

mod cpu;
#[cfg(feature = "fused-cuda")]
mod deterministic;
mod forward;
pub mod release;
pub use forward::{FusedForward, StageOutput};
#[cfg(test)]
mod authenticated;
#[cfg(test)]
mod fixture;

use super::{
    admission,
    lm48::Lm48Projection,
    stage1b::{self, Stage1bParameters, TokenProjection, TokenProjector},
};
use anyhow::{ensure, Context, Result};
use jope_api::{error::ErrorCode, Placement};
use jope_runtime::backend::{self, DeviceRequest};
use rand_chacha::{rand_core::SeedableRng, ChaCha12Rng};
use std::{fs::OpenOptions, path::Path};

pub const ATOL: f64 = 1e-5;
pub const RTOL: f64 = 1e-4;

/// Entry point for a future synchronous GPU qualification operation. The caller
/// must supply the owner's existing `~/jope_plan/qualification.lock`, and the
/// complete GPU operation must finish before returning from the closure. Never
/// create a missing lock or wait ahead of the existing H58/H57 owner. CPU unit
/// controls use only private temporary files, never the live qualification lock.
pub fn with_gpu_qualification<T>(
    qualification_lock: &Path,
    run: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let lease = OpenOptions::new()
        .read(true)
        .write(true)
        .open(qualification_lock)
        .context("open existing GPU qualification lock")?;
    lease
        .try_lock()
        .map_err(|error| anyhow::anyhow!("GPU qualification lease unavailable: {error}"))?;
    let result = run();
    drop(lease);
    result
}

/// Current backend placement; explicit CUDA never silently runs on the CPU.
/// No device is probed or initialized. A future CUDA implementation must replace
/// this resolution with actual instantiated stage backends, not GPU presence.
pub fn cpu_placement(request: DeviceRequest, workers: u32) -> Result<Placement, ErrorCode> {
    if workers == 0 {
        return Err(ErrorCode::InvalidInput);
    }
    let choice = backend::select(request).map_err(|_| ErrorCode::BackendUnavailable)?;
    Ok(Placement {
        dope: choice.selected.into(),
        ranker: choice.selected.into(),
        evaluation: "cpu".into(),
        workers,
    })
}

/// Reference only for the currently implemented dense-lm48/Stage-1B primitives.
/// The caller supplies the DOPE output; no encoder or cost output is invented.
#[derive(Clone, Debug, PartialEq)]
pub struct CpuProjectionReference {
    pub lm48: Vec<f32>,
    pub stage1b: TokenProjection,
}

struct ProjectedLm48(Vec<f32>);

impl TokenProjector for ProjectedLm48 {
    fn project(&self, _: &[f32]) -> Vec<f32> {
        self.0.clone()
    }
}

pub fn cpu_projection_reference(
    projection: &Lm48Projection,
    input: &[f32],
    dope629: &[f32],
    parameters: &Stage1bParameters,
) -> Result<CpuProjectionReference> {
    ensure!(projection.in_dim > 0, "empty lm48 input width");
    ensure!(
        projection.in_dim == input.len(),
        "lm48 input width mismatch"
    );
    ensure!(projection.out_dim == 48, "lm48 output width must be 48");
    ensure!(
        projection.in_dim.checked_mul(48) == Some(projection.weight.len())
            && projection.bias.len() == 48,
        "lm48 parameter shape mismatch"
    );
    ensure!(
        projection
            .weight
            .iter()
            .chain(&projection.bias)
            .chain(input)
            .chain(dope629)
            .all(|v| v.is_finite()),
        "CPU reference inputs and parameters must be finite"
    );
    let lm48 = projection.project(input);
    ensure!(lm48.iter().all(|v| v.is_finite()), "nonfinite lm48 output");
    let stage1b = stage1b::project(&ProjectedLm48(lm48.clone()), input, dope629, parameters)?;
    ensure!(
        stage1b.tokens.iter().flatten().all(|v| v.is_finite()),
        "nonfinite Stage-1B output"
    );
    Ok(CpuProjectionReference { lm48, stage1b })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    V256,
    Col,
    Dope629,
    Lm48,
    Tokens,
    Mask,
    Costs,
}

/// Row-major values, including all batch/slate axes in `shape`.
#[derive(Clone, Copy, Debug)]
pub struct TensorView<'a> {
    pub shape: &'a [usize],
    pub values: &'a [f32],
}

impl TensorView<'_> {
    fn validate(self, stage: Stage) -> Result<()> {
        ensure!(
            !self.shape.is_empty() && self.shape.iter().all(|&dim| dim > 0),
            "{stage:?} has empty dimensions"
        );
        let len = self.shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d));
        ensure!(
            len == Some(self.values.len()),
            "{stage:?} shape/data mismatch"
        );
        let suffix: &[usize] = match stage {
            Stage::V256 => &[256],
            Stage::Col => &[64, 17],
            Stage::Dope629 => &[629],
            Stage::Lm48 => &[48],
            Stage::Tokens => &[10, 32],
            Stage::Mask => &[10],
            Stage::Costs => &[],
        };
        ensure!(
            self.shape.ends_with(suffix),
            "{stage:?} stage dimensions differ"
        );
        if stage == Stage::Costs {
            ensure!(
                matches!(self.shape.last().copied(), Some(12 | 20)),
                "costs require a serving-shaped 4+8 or 4+16 slate"
            );
        }
        ensure!(
            self.values.iter().all(|v| v.is_finite()),
            "{stage:?} contains NaN/Inf"
        );
        if stage == Stage::Mask {
            ensure!(
                self.values.iter().all(|&v| v == 0.0 || v == 1.0),
                "token mask must contain only zero or one"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StageComparison {
    pub stage: Stage,
    pub values: usize,
    pub max_absolute_error: f64,
}

/// Tolerance is asymmetric: the CPU/golden reference defines the relative term.
pub fn compare_stage(
    stage: Stage,
    reference: TensorView<'_>,
    actual: TensorView<'_>,
) -> Result<StageComparison> {
    reference.validate(stage)?;
    actual.validate(stage)?;
    ensure!(
        reference.shape == actual.shape,
        "{stage:?} reference/actual shapes differ"
    );
    let mut max_absolute_error = 0.0f64;
    for (index, (&expected, &observed)) in reference.values.iter().zip(actual.values).enumerate() {
        // Widen before subtraction so opposite finite F32 extrema cannot overflow.
        let error = (f64::from(observed) - f64::from(expected)).abs();
        let tolerance = ATOL + RTOL * f64::from(expected).abs();
        ensure!(
            error <= tolerance,
            "{stage:?}[{index}] error {error} exceeds {tolerance}"
        );
        max_absolute_error = max_absolute_error.max(error);
    }
    Ok(StageComparison {
        stage,
        values: reference.values.len(),
        max_absolute_error,
    })
}

/// Every M2 stage is mandatory; a costs-only trace cannot satisfy this checker.
/// A trace contains exactly one context, and its identity travels with every
/// supplied stage so reference/candidate rows cannot be paired across contexts.
#[derive(Clone, Copy, Debug)]
pub struct Trace<'a> {
    pub context_id: &'a str,
    pub v256: TensorView<'a>,
    pub col: TensorView<'a>,
    pub dope629: TensorView<'a>,
    pub lm48: TensorView<'a>,
    pub tokens: TensorView<'a>,
    pub mask: TensorView<'a>,
    pub costs: TensorView<'a>,
}

pub fn compare_trace(reference: Trace<'_>, actual: Trace<'_>) -> Result<Vec<StageComparison>> {
    ensure!(
        !reference.context_id.is_empty(),
        "trace context identity must not be empty"
    );
    ensure!(
        reference.context_id == actual.context_id,
        "reference and actual trace context identities differ"
    );
    let stages = [
        (Stage::V256, reference.v256, actual.v256),
        (Stage::Col, reference.col, actual.col),
        (Stage::Dope629, reference.dope629, actual.dope629),
        (Stage::Lm48, reference.lm48, actual.lm48),
        (Stage::Tokens, reference.tokens, actual.tokens),
        (Stage::Mask, reference.mask, actual.mask),
        (Stage::Costs, reference.costs, actual.costs),
    ];
    // Validate pairs first, including rank/suffix checks, before slicing shapes.
    let comparisons = stages
        .iter()
        .map(|&(stage, reference, actual)| compare_stage(stage, reference, actual))
        .collect::<Result<Vec<_>>>()?;
    let context_shape = &reference.v256.shape[..reference.v256.shape.len() - 1];
    ensure!(
        context_shape.is_empty() || context_shape == [1],
        "a trace identity must name exactly one context"
    );
    for (stage, reference, _) in stages {
        let trailing_axes = match stage {
            Stage::Col | Stage::Tokens => 2,
            _ => 1,
        };
        ensure!(
            &reference.shape[..reference.shape.len() - trailing_axes] == context_shape,
            "{stage:?} context/batch axes differ from V256"
        );
    }
    Ok(comparisons)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdmissionComparison {
    pub reference: [usize; 4],
    pub actual: [usize; 4],
    /// Count/report this decision; it is not proof of identical admitted sets.
    pub ambiguous_boundary: bool,
    pub boundary_gap: f64,
    pub tie_band: f64,
}

/// Check selection separately from numeric cost parity. Uses the existing
/// admission implementation and identical RNG state for the two exploring arms.
/// No caller RNG is consumed; the boundary is fourth/fifth normally, second/third
/// when two ranked proposals are combined with two random proposals.
pub fn compare_admission(
    reference: &[f32],
    actual: &[f32],
    explore: bool,
    seed: [u8; 32],
) -> Result<AdmissionComparison> {
    let reference: Vec<_> = reference.iter().map(|&v| f64::from(v)).collect();
    let actual: Vec<_> = actual.iter().map(|&v| f64::from(v)).collect();
    let selected_reference =
        admission::select_costs(&reference, explore, &mut ChaCha12Rng::from_seed(seed))?;
    let selected_actual =
        admission::select_costs(&actual, explore, &mut ChaCha12Rng::from_seed(seed))?;
    let costs = &reference[4..];
    let mut order: Vec<_> = (0..costs.len()).collect();
    order.sort_by(|&a, &b| {
        costs[a]
            .partial_cmp(&costs[b])
            .expect("validated finite costs")
            .then(a.cmp(&b))
    });
    let boundary = if explore { 2 } else { 4 };
    let last = costs[order[boundary - 1]];
    let gap = costs[order[boundary]] - last;
    let band = 2.0 * (ATOL + RTOL * last.abs());
    let ambiguous = gap <= band;
    let mut expected_set = selected_reference;
    let mut actual_set = selected_actual;
    expected_set.sort_unstable();
    actual_set.sort_unstable();
    ensure!(
        ambiguous || expected_set == actual_set,
        "admitted sets differ outside the tie band"
    );
    Ok(AdmissionComparison {
        reference: selected_reference,
        actual: selected_actual,
        ambiguous_boundary: ambiguous,
        boundary_gap: gap,
        tie_band: band,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use stage1b::{SvdSlot, DOPE_FEATURES, TOKEN_WIDTH};

    #[test]
    fn gpu_entry_point_refuses_busy_or_missing_lease_before_callback() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("qualification.lock");
        let invoked = std::cell::Cell::new(false);
        assert!(with_gpu_qualification(&path, || {
            invoked.set(true);
            Ok(())
        })
        .is_err());
        assert!(!invoked.get());
        assert!(!path.exists());
        let lease = std::fs::File::create(&path).unwrap();
        lease.lock().unwrap();
        assert!(with_gpu_qualification(&path, || {
            invoked.set(true);
            Ok(())
        })
        .is_err());
        assert!(!invoked.get());
    }

    #[test]
    fn gpu_lease_is_held_for_the_complete_callback_and_released_after_failure() {
        let lock = tempfile::NamedTempFile::new().unwrap();
        let result: Result<()> = with_gpu_qualification(lock.path(), || {
            let contender = OpenOptions::new()
                .read(true)
                .write(true)
                .open(lock.path())
                .unwrap();
            assert!(contender.try_lock().is_err());
            anyhow::bail!("GPU operation control failure")
        });
        assert!(result.unwrap_err().to_string().contains("control failure"));
        assert_eq!(with_gpu_qualification(lock.path(), || Ok(17)).unwrap(), 17);
    }

    fn slot(width: usize) -> SvdSlot {
        SvdSlot {
            mean: vec![0.0; width],
            std: vec![1.0; width],
            active: vec![1.0; width],
            projection: vec![0.0; width * TOKEN_WIDTH],
        }
    }

    // Arithmetic controls only: these are not model weights or M2 goldens.
    fn projection_fixture() -> (Lm48Projection, Stage1bParameters) {
        let mut projection = Lm48Projection {
            weight: vec![0.0; 96],
            bias: vec![0.0; 48],
            in_dim: 2,
            out_dim: 48,
        };
        projection.weight[0] = 2.0;
        projection.bias[0] = 1.0;
        projection.weight[3] = 1.0;
        let mut parameters = Stage1bParameters {
            lm80_mean: [0.0; 80],
            lm80_std: [1.0; 80],
            lm48_mean: [0.0; 48],
            lm48_std: [1.0; 48],
            slot0: slot(184),
            slot1: slot(48),
            slot2: slot(DOPE_FEATURES),
            slot4: slot(DOPE_FEATURES),
            slot5: slot(DOPE_FEATURES),
            slot7: slot(184),
            slot9: slot(64),
        };
        parameters.slot0.projection[104 * TOKEN_WIDTH] = 1.0;
        parameters.slot2.projection[0] = 1.0;
        (projection, parameters)
    }

    #[test]
    fn placement_serializes_actual_cpu_stages_and_worker_count() {
        for request in [DeviceRequest::Auto, DeviceRequest::Cpu] {
            let placement = cpu_placement(request, 7).unwrap();
            assert_eq!(
                serde_json::to_value(placement).unwrap(),
                serde_json::json!({
                    "dope":"cpu", "ranker":"cpu", "evaluation":"cpu", "workers":7,
                })
            );
        }
        assert_eq!(
            cpu_placement(DeviceRequest::Cpu, 0),
            Err(ErrorCode::InvalidInput)
        );
    }

    #[test]
    fn explicit_cuda_refuses_with_catalog_backend_error_without_probing_gpu() {
        for index in [0, 1, u32::MAX] {
            let error = cpu_placement(DeviceRequest::Cuda(index), 1).unwrap_err();
            assert_eq!(error, ErrorCode::BackendUnavailable);
            assert_eq!(error.as_str(), "BACKEND_UNAVAILABLE");
            assert_eq!(error.http_status(), Some(503));
            assert_eq!(error.exit_code(), Some(4));
        }
    }

    #[test]
    fn cpu_reference_executes_both_existing_projection_primitives() {
        let (projection, parameters) = projection_fixture();
        let mut dope = [0.0; 629];
        dope[0] = 1.5;
        let result =
            cpu_projection_reference(&projection, &[1.0, 2.0], &dope, &parameters).unwrap();
        assert_eq!(&result.lm48[..2], &[3.0, 2.0]);
        assert_eq!(result.stage1b.tokens[0][0], 3.0);
        assert_eq!(result.stage1b.tokens[2][0], 1.5);
        assert_eq!(
            result.stage1b.mask,
            [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        );
        let changed =
            cpu_projection_reference(&projection, &[2.0, 2.0], &dope, &parameters).unwrap();
        assert_eq!(changed.lm48[0], 5.0);
        assert_eq!(changed.stage1b.tokens[0][0], 5.0);
        dope[0] = -2.0;
        let changed =
            cpu_projection_reference(&projection, &[1.0, 2.0], &dope, &parameters).unwrap();
        assert_eq!(changed.stage1b.tokens[2][0], -2.0);
    }

    #[test]
    fn cpu_reference_rejects_invalid_shapes_and_nonfinite_arithmetic() {
        let (mut projection, mut parameters) = projection_fixture();
        assert!(cpu_projection_reference(&projection, &[1.0], &[0.0; 629], &parameters).is_err());
        assert!(
            cpu_projection_reference(&projection, &[1.0, 2.0], &[0.0; 628], &parameters).is_err()
        );
        projection.weight.pop();
        assert!(
            cpu_projection_reference(&projection, &[1.0, 2.0], &[0.0; 629], &parameters).is_err()
        );
        projection.weight.push(0.0);
        projection.weight[0] = f32::NAN;
        assert!(
            cpu_projection_reference(&projection, &[1.0, 2.0], &[0.0; 629], &parameters).is_err()
        );
        projection.weight[0] = f32::MAX;
        assert!(
            cpu_projection_reference(&projection, &[2.0, 2.0], &[0.0; 629], &parameters).is_err()
        );
        projection.weight[0] = 2.0;
        parameters.slot0.std[0] = 0.0;
        assert!(
            cpu_projection_reference(&projection, &[1.0, 2.0], &[0.0; 629], &parameters).is_err()
        );
    }

    fn view<'a>(shape: &'a [usize], values: &'a [f32]) -> TensorView<'a> {
        TensorView { shape, values }
    }

    #[test]
    fn compares_f32_absolute_and_reference_relative_tolerances() {
        let mut expected = [0.0; 48];
        expected[1] = -1000.0;
        let mut actual = expected;
        actual[0] = 1e-6;
        actual[1] += 0.05;
        let report =
            compare_stage(Stage::Lm48, view(&[48], &expected), view(&[48], &actual)).unwrap();
        assert_eq!(report.values, 48);
        assert!(report.max_absolute_error > 0.04);
        actual[0] = 0.001;
        assert!(compare_stage(Stage::Lm48, view(&[48], &expected), view(&[48], &actual)).is_err());
    }

    #[test]
    fn rejects_empty_overflowing_and_axis_mismatched_shapes() {
        for shape in [&[][..], &[0, 48][..], &[usize::MAX, 48][..], &[47][..]] {
            assert!(compare_stage(Stage::Lm48, view(shape, &[]), view(shape, &[])).is_err());
        }
        assert!(compare_stage(
            Stage::Lm48,
            view(&[1, 48], &[0.0; 48]),
            view(&[48], &[0.0; 48])
        )
        .is_err());
    }

    #[test]
    fn rejects_nonfinite_reference_or_actual_and_finite_extreme_drift() {
        let good = [0.0; 48];
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut bad = good;
            bad[0] = invalid;
            assert!(compare_stage(Stage::Lm48, view(&[48], &good), view(&[48], &bad)).is_err());
            assert!(compare_stage(Stage::Lm48, view(&[48], &bad), view(&[48], &good)).is_err());
        }
        assert!(compare_stage(
            Stage::Lm48,
            view(&[48], &[f32::MAX; 48]),
            view(&[48], &[-f32::MAX; 48])
        )
        .is_err());
    }

    #[test]
    fn binary_mask_and_serving_slate_dimensions_are_enforced() {
        assert!(compare_stage(
            Stage::Mask,
            view(&[10], &[0.0; 10]),
            view(&[10], &[1e-6; 10])
        )
        .is_err());
        for n in [12, 20] {
            let costs = vec![0.0; n];
            assert!(compare_stage(Stage::Costs, view(&[n], &costs), view(&[n], &costs)).is_ok());
        }
        assert!(compare_stage(Stage::Costs, view(&[8], &[0.0; 8]), view(&[8], &[0.0; 8])).is_err());
    }

    struct TraceFixture {
        context_id: &'static str,
        stages: Vec<Vec<f32>>,
    }

    impl TraceFixture {
        fn new() -> Self {
            Self {
                context_id: "context-a",
                stages: [256, 1088, 629, 48, 320, 10, 12]
                    .map(|n| vec![0.0; n])
                    .to_vec(),
            }
        }

        fn trace(&self) -> Trace<'_> {
            Trace {
                context_id: self.context_id,
                v256: view(&[256], &self.stages[0]),
                col: view(&[64, 17], &self.stages[1]),
                dope629: view(&[629], &self.stages[2]),
                lm48: view(&[48], &self.stages[3]),
                tokens: view(&[10, 32], &self.stages[4]),
                mask: view(&[10], &self.stages[5]),
                costs: view(&[12], &self.stages[6]),
            }
        }
    }

    #[test]
    fn rejects_reference_and_actual_traces_from_different_contexts() {
        let reference = TraceFixture::new();
        let mut actual = TraceFixture::new();
        actual.context_id = "context-b";
        let error = compare_trace(reference.trace(), actual.trace()).unwrap_err();
        assert!(error.to_string().contains("context identities differ"));

        actual.context_id = "";
        assert!(compare_trace(actual.trace(), actual.trace()).is_err());
    }

    #[test]
    fn every_stage_perturbation_is_detected_even_when_costs_are_unchanged() {
        let reference = TraceFixture::new();
        assert_eq!(
            compare_trace(reference.trace(), reference.trace())
                .unwrap()
                .len(),
            7
        );
        for stage in 0..7 {
            let mut actual = TraceFixture::new();
            actual.stages[stage][0] = 1.0;
            assert!(
                compare_trace(reference.trace(), actual.trace()).is_err(),
                "stage {stage}"
            );
        }
        let mut incomplete = TraceFixture::new();
        incomplete.stages[6].clear();
        assert!(compare_trace(reference.trace(), incomplete.trace()).is_err());
    }

    fn costs(proposals: usize) -> Vec<f32> {
        [-1000.0; 4]
            .into_iter()
            .chain((0..proposals).map(|n| n as f32))
            .collect()
    }

    #[test]
    fn compares_admitted_sets_outside_the_boundary_band() {
        let reference = costs(8);
        let same = compare_admission(&reference, &reference, false, [7; 32]).unwrap();
        assert!(!same.ambiguous_boundary);
        assert_eq!(same.reference, [0, 1, 2, 3]);
        assert_eq!(same.boundary_gap, 1.0);
        let mut disconnected = reference.clone();
        disconnected.swap(7, 8);
        assert!(compare_admission(&reference, &disconnected, false, [7; 32]).is_err());
    }

    #[test]
    fn counts_ambiguous_boundary_without_asserting_set_equality() {
        let mut reference = costs(8);
        reference[8] = reference[7];
        let mut actual = reference.clone();
        actual[7] += 1e-5;
        let report = compare_admission(&reference, &actual, false, [7; 32]).unwrap();
        assert!(report.ambiguous_boundary);
        assert_eq!(report.boundary_gap, 0.0);
        assert_ne!(report.reference, report.actual);
    }

    #[test]
    fn exploration_replays_rng_and_uses_second_third_boundary() {
        let reference = costs(16);
        let report = compare_admission(&reference, &reference, true, [19; 32]).unwrap();
        assert_eq!(report.reference, report.actual);
        assert_eq!(&report.reference[..2], &[0, 1]);
        assert!(!report.ambiguous_boundary);
        assert_eq!(report.tie_band, 2.0 * (ATOL + RTOL));
        let mut tied = reference;
        tied[6] = tied[5];
        assert!(
            compare_admission(&tied, &tied, true, [19; 32])
                .unwrap()
                .ambiguous_boundary
        );
    }

    #[test]
    fn signed_zero_ties_preserve_existing_admission_order() {
        let reference = [
            0.0, 0.0, 0.0, 0.0, -3.0, -2.0, -1.0, -0.0, 0.0, 1.0, 2.0, 3.0,
        ];
        let report = compare_admission(&reference, &reference, false, [0; 32]).unwrap();
        assert_eq!(report.reference, [0, 1, 2, 3]);
        assert!(report.ambiguous_boundary);
    }

    #[test]
    fn admission_rejects_bad_slate_and_nonfinite_costs_on_either_side() {
        let good = costs(8);
        assert!(compare_admission(&good, &good[..11], false, [0; 32]).is_err());
        assert!(compare_admission(&good, &good, true, [0; 32]).is_err());
        for invalid in [f32::NAN, f32::INFINITY] {
            let mut bad = good.clone();
            bad[4] = invalid;
            assert!(compare_admission(&good, &bad, false, [0; 32]).is_err());
            assert!(compare_admission(&bad, &good, false, [0; 32]).is_err());
        }
    }
}

#[cfg(test)]
mod context_axis_regression {
    use super::*;

    struct Fixture {
        shapes: [Vec<usize>; 7],
        values: [Vec<f32>; 7],
    }

    impl Fixture {
        fn new(prefix: &[usize], slate: usize) -> Self {
            let suffixes = [
                vec![256],
                vec![64, 17],
                vec![629],
                vec![48],
                vec![10, 32],
                vec![10],
                vec![slate],
            ];
            let shapes = suffixes.map(|suffix| {
                let mut shape = prefix.to_vec();
                shape.extend(suffix);
                shape
            });
            let values = shapes
                .each_ref()
                .map(|shape| vec![0.0; shape.iter().product()]);
            Self { shapes, values }
        }

        fn set_prefix(&mut self, stage: usize, prefix: &[usize]) {
            let trailing = if stage == 1 || stage == 4 { 2 } else { 1 };
            let mut shape = prefix.to_vec();
            shape.extend_from_slice(&self.shapes[stage][self.shapes[stage].len() - trailing..]);
            self.values[stage] = vec![0.0; shape.iter().product()];
            self.shapes[stage] = shape;
        }

        fn trace(&self) -> Trace<'_> {
            let v = |i: usize| TensorView {
                shape: self.shapes[i].as_slice(),
                values: self.values[i].as_slice(),
            };
            Trace {
                context_id: "synthetic-context",
                v256: v(0),
                col: v(1),
                dope629: v(2),
                lm48: v(3),
                tokens: v(4),
                mask: v(5),
                costs: v(6),
            }
        }
    }

    #[test]
    fn accepts_aligned_unbatched_or_single_context_on_both_slates() {
        for prefix in [&[][..], &[1][..]] {
            for slate in [12, 20] {
                let valid = Fixture::new(prefix, slate);
                assert_eq!(
                    compare_trace(valid.trace(), valid.trace()).unwrap().len(),
                    7
                );
            }
        }
    }

    #[test]
    fn rejects_forty_nine_contexts_in_any_one_stage_of_a_fifty_context_trace() {
        for stage in 0..7 {
            let mut wrong = Fixture::new(&[50], 12);
            wrong.set_prefix(stage, &[49]);
            assert!(
                compare_trace(wrong.trace(), wrong.trace()).is_err(),
                "stage {stage} accepted different context inventory"
            );
        }
    }

    #[test]
    fn rejects_different_context_axes_even_when_the_flattened_lengths_agree() {
        for stage in 0..7 {
            let mut wrong = Fixture::new(&[50], 20);
            let old_len = wrong.values[stage].len();
            wrong.set_prefix(stage, &[2, 25]);
            assert_eq!(wrong.values[stage].len(), old_len);
            assert!(
                compare_trace(wrong.trace(), wrong.trace()).is_err(),
                "stage {stage} accepted incompatible context axes"
            );
        }
    }

    #[test]
    fn rejects_implicit_single_context_broadcast_in_any_stage() {
        for stage in 0..7 {
            let mut wrong = Fixture::new(&[], 12);
            wrong.set_prefix(stage, &[1]);
            assert!(
                compare_trace(wrong.trace(), wrong.trace()).is_err(),
                "stage {stage} accepted an implicit context axis"
            );
        }
    }

    #[test]
    fn existing_bad_rank_nonfinite_mask_and_pair_shape_controls_remain_errors() {
        let valid = Fixture::new(&[1], 12);
        let mut bad = Fixture::new(&[1], 12);
        bad.shapes[0].clear();
        assert!(compare_trace(valid.trace(), bad.trace()).is_err());
        bad = Fixture::new(&[1], 12);
        bad.values[5][0] = 0.5;
        assert!(compare_trace(valid.trace(), bad.trace()).is_err());
        bad = Fixture::new(&[1], 12);
        bad.values[3][0] = f32::NAN;
        assert!(compare_trace(valid.trace(), bad.trace()).is_err());
        bad = Fixture::new(&[1], 12);
        bad.set_prefix(6, &[2]);
        assert!(compare_trace(valid.trace(), bad.trace()).is_err());
    }

    #[test]
    fn one_identity_cannot_certify_fifty_contexts_even_when_all_stage_axes_match() {
        for slate in [12, 20] {
            let wrong = Fixture::new(&[50], slate);
            assert!(compare_trace(wrong.trace(), wrong.trace()).is_err());
        }
    }

    #[test]
    fn existing_context_identity_checks_still_reject_mismatch_or_empty_id() {
        let valid = Fixture::new(&[1], 12);
        let reference = valid.trace();
        let mut actual = valid.trace();
        actual.context_id = "another-context";
        assert!(compare_trace(reference, actual).is_err());
        let mut reference = valid.trace();
        reference.context_id = "";
        assert!(compare_trace(reference, reference).is_err());
    }
}
