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
#[cfg(all(test, unix))]
pub(crate) use fixture::inherited_lease;

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

/// Tensor boundaries used by the CUDA projection comparison path.
///
/// This is a different domain from the research evidence lifecycle stages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorStage {
    V256,
    Col,
    Dope629,
    Lm48,
    Tokens,
    Mask,
    Costs,
}

/// Compatibility alias for callers of the original CUDA comparison API.
pub type Stage = TensorStage;

/// Row-major values, including all batch/slate axes in `shape`.
#[derive(Clone, Copy, Debug)]
pub struct TensorView<'a> {
    pub shape: &'a [usize],
    pub values: &'a [f32],
}

impl TensorView<'_> {
    fn validate(self, stage: TensorStage) -> Result<()> {
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
            TensorStage::V256 => &[256],
            TensorStage::Col => &[64, 17],
            TensorStage::Dope629 => &[629],
            TensorStage::Lm48 => &[48],
            TensorStage::Tokens => &[10, 32],
            TensorStage::Mask => &[10],
            TensorStage::Costs => &[],
        };
        ensure!(
            self.shape.ends_with(suffix),
            "{stage:?} stage dimensions differ"
        );
        if stage == TensorStage::Costs {
            ensure!(
                matches!(self.shape.last().copied(), Some(12 | 20)),
                "costs require a serving-shaped 4+8 or 4+16 slate"
            );
        }
        ensure!(
            self.values.iter().all(|v| v.is_finite()),
            "{stage:?} contains NaN/Inf"
        );
        if stage == TensorStage::Mask {
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
    pub stage: TensorStage,
    pub values: usize,
    pub max_absolute_error: f64,
}

/// Tolerance is asymmetric: the CPU/golden reference defines the relative term.
pub fn compare_stage(
    stage: TensorStage,
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
        (TensorStage::V256, reference.v256, actual.v256),
        (TensorStage::Col, reference.col, actual.col),
        (TensorStage::Dope629, reference.dope629, actual.dope629),
        (TensorStage::Lm48, reference.lm48, actual.lm48),
        (TensorStage::Tokens, reference.tokens, actual.tokens),
        (TensorStage::Mask, reference.mask, actual.mask),
        (TensorStage::Costs, reference.costs, actual.costs),
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
            TensorStage::Col | TensorStage::Tokens => 2,
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
#[path = "cuda/context_axis_regression.rs"]
mod context_axis_regression;
#[cfg(test)]
#[path = "cuda/comparison_tests.rs"]
mod tests;
