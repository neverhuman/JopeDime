//! Adaptive `SoloSingleLambda` vs `HybridFanOut` execution planner and Graviton2 memory-class
//! selector ([ADR-0007](../../../../docs/adr/ADR-0007-carbon-lambda-cells.md)).

use serde::{Deserialize, Serialize};

use crate::error::{CarbonError, CarbonResult};
use crate::role::{CarbonEffortTier, CarbonProductTier};

/// AWS Lambda `arm64` price per GB-second in USD (`$0.0000133334`).
pub const LAMBDA_ARM64_USD_PER_GB_SEC: f64 = 0.000_013_333_4;

/// AWS Lambda request invocation fee in USD (`$0.20` per `1,000,000` requests).
pub const LAMBDA_USD_PER_INVOCATION: f64 = 0.000_000_2;

/// Memory configured on `carbon-engine` (`3,008 MB` -> `~2` full vCPUs for neural Wave-0).
pub const ENGINE_MEMORY_MB: usize = 3_008;

/// `EvalMemoryClass` is an enum for the five Arm64 Graviton2 `carbon-eval` worker memory classes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvalMemoryClass {
    /// `512 MB` (`~0.29` vCPU) — Free / micro tables (`working_set <= 192 MB`).
    Xs512Mb,
    /// `1,024 MB` (`~0.58` vCPU) — Small tables (`working_set <= 512 MB`).
    S1024Mb,
    /// `1,769 MB` (`1.00` full vCPU sweet spot) — Standard tables (`working_set <= 1,024 MB`).
    M1769Mb,
    /// `3,538 MB` (`2.00` full vCPUs) — Large Pro tables (`working_set <= 2,304 MB`).
    L3538Mb,
    /// `7,076 MB` (`4.00` full vCPUs) — Extra-large tables (`working_set <= 5,120 MB`).
    Xl7076Mb,
}

impl EvalMemoryClass {
    /// Configured Lambda memory in MB.
    pub const fn memory_mb(self) -> usize {
        match self {
            Self::Xs512Mb => 512,
            Self::S1024Mb => 1_024,
            Self::M1769Mb => 1_769,
            Self::L3538Mb => 3_538,
            Self::Xl7076Mb => 7_076,
        }
    }

    /// Maximum safe working-set budget in MB (`~65–72%` of container RAM).
    pub const fn max_working_set_mb(self) -> usize {
        match self {
            Self::Xs512Mb => 192,
            Self::S1024Mb => 512,
            Self::M1769Mb => 1_024,
            Self::L3538Mb => 2_304,
            Self::Xl7076Mb => 5_120,
        }
    }

    /// Effective vCPU allocation proportional to `1,769 MB == 1.0 vCPU` on AWS Lambda.
    pub fn effective_vcpus(self) -> f64 {
        (self.memory_mb() as f64) / 1_769.0
    }

    /// Select the smallest `EvalMemoryClass` that safely holds `working_set_mb`.
    pub fn select_for_working_set_mb(working_set_mb: usize) -> CarbonResult<Self> {
        for class in [
            Self::Xs512Mb,
            Self::S1024Mb,
            Self::M1769Mb,
            Self::L3538Mb,
            Self::Xl7076Mb,
        ] {
            if working_set_mb <= class.max_working_set_mb() {
                return Ok(class);
            }
        }
        Err(CarbonError::WorkingSetExceedsLambdaClass {
            required_mb: working_set_mb,
            max_lambda_mb: Self::Xl7076Mb.memory_mb(),
        })
    }
}

/// `CarbonExecutionMode` is an enum for the adaptive planner's execution topology choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CarbonExecutionMode {
    /// Execute neural Wave-0 and all search waves inside `carbon-engine` (`zero` S3 fan-out overhead).
    SoloSingleLambda,
    /// Execute neural Wave-0 in `carbon-engine` and fan out candidate shards across `K` parallel `carbon-eval` workers per wave.
    HybridFanOut,
}

/// `CarbonPlanInput` is a struct for the dataset dimensions, tier, and per-candidate timing inputs.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CarbonPlanInput {
    pub product_tier: CarbonProductTier,
    pub effort: CarbonEffortTier,
    pub total_rows: usize,
    pub fit_rows: usize,
    pub features: usize,
    pub est_ms_per_candidate_1vcpu: f64,
}

/// `CarbonPlanDecision` is a struct for the deterministic execution topology, worker sizing, shard
/// count, and estimated cost/latency returned by `plan_carbon_execution`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CarbonPlanDecision {
    pub mode: CarbonExecutionMode,
    pub eval_class: EvalMemoryClass,
    pub estimated_working_set_mb: usize,
    pub shards_per_wave: u32,
    pub planned_waves: u32,
    pub solo_wall_ms: f64,
    pub hybrid_wall_ms: f64,
    pub solo_estimated_usd: f64,
    pub hybrid_estimated_usd: f64,
}

/// Estimate peak candidate-evaluation working-set footprint in MB from `(fit_rows, features)`.
pub fn estimate_working_set_mb(fit_rows: usize, features: usize) -> CarbonResult<usize> {
    if fit_rows == 0 || features == 0 {
        return Err(CarbonError::InvalidPlanInput {
            reason: "fit_rows and features must both be >= 1",
        });
    }
    let matrix_bytes = (fit_rows as u128) * (features as u128) * 8;
    let scratch_bytes = (fit_rows as u128) * (features.clamp(6, 32) as u128) * 8;
    let total_mb = 64u128 + (matrix_bytes + scratch_bytes).div_ceil(1_048_576);
    usize::try_from(total_mb).map_err(|_| CarbonError::InvalidPlanInput {
        reason: "working-set byte estimate overflowed usize",
    })
}

/// Deterministically choose `SoloSingleLambda` vs `HybridFanOut` and right-size `EvalMemoryClass`.
pub fn plan_carbon_execution(input: &CarbonPlanInput) -> CarbonResult<CarbonPlanDecision> {
    input
        .product_tier
        .validate_admission(input.total_rows, input.features, input.effort)?;
    if input.fit_rows == 0 || input.fit_rows >= input.total_rows {
        return Err(CarbonError::InvalidPlanInput {
            reason: "fit_rows must satisfy 1 <= fit_rows < total_rows",
        });
    }
    if !input.est_ms_per_candidate_1vcpu.is_finite() || input.est_ms_per_candidate_1vcpu <= 0.0 {
        return Err(CarbonError::InvalidPlanInput {
            reason: "est_ms_per_candidate_1vcpu must be finite and positive",
        });
    }

    let working_set_mb = estimate_working_set_mb(input.fit_rows, input.features)?;
    let base_class = EvalMemoryClass::select_for_working_set_mb(working_set_mb)?;
    let eval_class = if input.product_tier != CarbonProductTier::Free
        && input.effort >= CarbonEffortTier::High
        && base_class < EvalMemoryClass::M1769Mb
    {
        EvalMemoryClass::M1769Mb
    } else {
        base_class
    };

    let waves = input.effort.planned_waves();
    let cands_per_wave = input.effort.candidates_per_wave();
    let engine_vcpus = (ENGINE_MEMORY_MB as f64) / 1_769.0;
    let wave0_neural_ms = 180.0;

    let solo_wave_ms =
        (f64::from(cands_per_wave) * input.est_ms_per_candidate_1vcpu) / engine_vcpus;
    let solo_wall_ms = wave0_neural_ms + f64::from(waves) * solo_wave_ms;
    let engine_gb = (ENGINE_MEMORY_MB as f64) / 1_024.0;
    let solo_estimated_usd = (solo_wall_ms / 1_000.0) * engine_gb * LAMBDA_ARM64_USD_PER_GB_SEC
        + LAMBDA_USD_PER_INVOCATION;

    let target_cands_per_shard = 16u32;
    let shards_per_wave = cands_per_wave.div_ceil(target_cands_per_shard).clamp(2, 24);
    let cands_per_shard = cands_per_wave.div_ceil(shards_per_wave);
    let eval_vcpus = eval_class.effective_vcpus();
    let dispatch_overhead_ms = 115.0;
    let shard_compute_ms =
        (f64::from(cands_per_shard) * input.est_ms_per_candidate_1vcpu) / eval_vcpus;
    let hybrid_wave_ms = dispatch_overhead_ms + shard_compute_ms;
    let hybrid_wall_ms = wave0_neural_ms + f64::from(waves) * hybrid_wave_ms;

    let eval_gb = (eval_class.memory_mb() as f64) / 1_024.0;
    let total_eval_invocations = waves * shards_per_wave;
    let eval_compute_usd = f64::from(total_eval_invocations)
        * (shard_compute_ms / 1_000.0)
        * eval_gb
        * LAMBDA_ARM64_USD_PER_GB_SEC;
    let engine_coord_usd = (hybrid_wall_ms / 1_000.0) * engine_gb * LAMBDA_ARM64_USD_PER_GB_SEC;
    let invocation_fees_usd = f64::from(1 + total_eval_invocations) * LAMBDA_USD_PER_INVOCATION;
    let hybrid_estimated_usd = engine_coord_usd + eval_compute_usd + invocation_fees_usd;

    let lease_budget_ms = f64::from(input.effort.lease_ceiling_secs()) * 1_000.0;
    let mode = if input.effort == CarbonEffortTier::Fast || solo_wall_ms <= 8_000.0 {
        CarbonExecutionMode::SoloSingleLambda
    } else if solo_wall_ms > 0.45 * lease_budget_ms
        || (solo_wall_ms / hybrid_wall_ms >= 2.2
            && hybrid_estimated_usd <= solo_estimated_usd * 1.35)
    {
        CarbonExecutionMode::HybridFanOut
    } else {
        CarbonExecutionMode::SoloSingleLambda
    };

    Ok(CarbonPlanDecision {
        mode,
        eval_class,
        estimated_working_set_mb: working_set_mb,
        shards_per_wave: if mode == CarbonExecutionMode::HybridFanOut {
            shards_per_wave
        } else {
            0
        },
        planned_waves: waves,
        solo_wall_ms,
        hybrid_wall_ms,
        solo_estimated_usd,
        hybrid_estimated_usd,
    })
}
