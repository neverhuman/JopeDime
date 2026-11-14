//! Adaptive `SoloSingleLambda` vs `HybridFanOut` execution planner and Graviton2 memory-class
//! selector ([ADR-0007](../../../../docs/adr/ADR-0007-carbon-lambda-cells.md)).

use serde::{Deserialize, Serialize};

use crate::error::{CarbonError, CarbonResult};
use crate::role::{CarbonEffortTier, CarbonProductTier};

/// AWS Lambda `arm64` price per GB-second in USD (`$0.0000133334`).
pub const LAMBDA_ARM64_USD_PER_GB_SEC: f64 = 0.000_013_333_4;

/// AWS Lambda request invocation fee in USD (`$0.20` per `1,000,000` requests).
pub const LAMBDA_USD_PER_INVOCATION: f64 = 0.000_000_2;

/// AWS API Gateway HTTP API v2.0 request fee in USD (`$1.00` per `1,000,000` 512 KB units, `us-east-2`).
pub const APIGW_HTTP_API_USD_PER_UNIT: f64 = 0.000_001_0;

/// Amazon S3 Standard `PUT` / `POST` / `LIST` fee in USD (`$0.005` per `1,000` requests, `us-east-2`).
pub const S3_STANDARD_PUT_LIST_USD_PER_REQ: f64 = 0.000_005_0;

/// Amazon S3 Standard `GET` / `HEAD` fee in USD (`$0.0004` per `1,000` requests, `us-east-2`).
pub const S3_STANDARD_GET_USD_PER_REQ: f64 = 0.000_000_4;

/// Amazon S3 Standard storage fee in USD per decimal GB-month (`$0.023`, `us-east-2`).
pub const S3_STANDARD_USD_PER_GB_MONTH: f64 = 0.023;

/// AWS internet data transfer out fee in USD per decimal GB (`$0.09`).
pub const AWS_INTERNET_EGRESS_USD_PER_GB: f64 = 0.09;

/// Memory configured on `carbon-edge` (`256 MB` Graviton2).
pub const EDGE_MEMORY_MB: usize = 256;

/// Memory configured on `carbon-engine` (`3,008 MB` -> `~2` full vCPUs for neural Wave-0).
pub const ENGINE_MEMORY_MB: usize = 3_008;

/// Memory configured on `carbon-eval` in `deploy/carbon/cloudformation-cell.json` (`1,769 MB` = `1.0` full vCPU).
pub const DEPLOYED_EVAL_MEMORY_MB: u32 = 1_769;

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

/// `CarbonIngressCostMode` is an enum identifying how HTTP requests enter the Carbon cell for cost metering.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CarbonIngressCostMode {
    /// Public API Gateway HTTP API v2.0 (`*.execute-api.*.amazonaws.com`, `$1.00 / 1M` 512 KB units).
    ApiGatewayHttpV2,
    /// Direct IAM-authenticated Lambda Function URL (`*.lambda-url.*.on.aws`, `$0.00` gateway fee).
    LambdaFunctionUrl,
    /// Customer-managed VPC `carbon serve` (`$0.00` marginal AWS cloud fee).
    SelfHostedVpc,
}

/// `CarbonRunCostBreakdown` is a struct itemizing per-dataset AWS compute, API Gateway, S3 request,
/// storage, and egress costs in USD (`us-east-2` On-Demand rates, zero Free-Tier discount assumed).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CarbonRunCostBreakdown {
    pub ingress_mode: CarbonIngressCostMode,
    pub edge_invocations: u32,
    pub engine_invocations: u32,
    pub engine_billed_ms: u64,
    pub eval_invocations: u64,
    pub eval_memory_mb: u32,
    pub eval_billed_ms: u64,
    pub apigw_metered_units: u32,
    pub s3_put_list_requests: u32,
    pub s3_get_requests: u32,
    pub lambda_edge_usd: f64,
    pub lambda_engine_usd: f64,
    pub lambda_eval_usd: f64,
    pub apigw_http_api_usd: f64,
    pub s3_requests_usd: f64,
    pub s3_storage_and_transfer_usd: f64,
    pub total_estimated_usd: f64,
    pub runs_per_1_usd: Option<f64>,
}

/// `CarbonLiveCostInputs` is a struct supplying observed run telemetry to [`compute_live_run_cost`].
#[derive(Clone, Copy, Debug)]
pub struct CarbonLiveCostInputs {
    pub ingress_mode: CarbonIngressCostMode,
    pub minted_ephemeral_token: bool,
    pub engine_search_invocations: u32,
    pub poll_requests: u32,
    pub signaled_early_stop: bool,
    pub shredded_after_run: bool,
    pub waves_completed: u32,
    pub eval_shards: u32,
    pub eval_memory_mb: u32,
    pub eval_invocations: u64,
    pub eval_total_wall_ms: u64,
    pub search_wall_ms: f64,
    pub predict_wall_ms: f64,
    pub uploaded_cas_bytes: usize,
    pub http_request_bytes: usize,
    pub http_response_bytes: usize,
}

/// Compute an itemized per-dataset USD cost breakdown from actual observed Carbon run telemetry.
pub fn compute_live_run_cost(inputs: &CarbonLiveCostInputs) -> CarbonRunCostBreakdown {
    let auth_calls = u32::from(inputs.minted_ephemeral_token);
    let stop_calls = u32::from(inputs.signaled_early_stop);
    let shred_calls = u32::from(inputs.shredded_after_run);
    let search_legs = inputs.engine_search_invocations.max(1);

    // Edge HTTP calls: ping (1) + optional auth (0..1) + uploads/prepare (1) + start/resume legs + polls + optional stop + predict (1) + optional shred.
    let edge_invocations = 1u32
        .saturating_add(auth_calls)
        .saturating_add(1)
        .saturating_add(search_legs)
        .saturating_add(inputs.poll_requests)
        .saturating_add(stop_calls)
        .saturating_add(1)
        .saturating_add(shred_calls);
    let engine_invocations = search_legs.saturating_add(1); // +1 for synchronous `/predict`
    let engine_search_ms = (inputs.search_wall_ms.ceil().max(1.0)) as u64;
    let engine_predict_ms = (inputs.predict_wall_ms.ceil().max(1.0)) as u64;
    let engine_billed_ms = engine_search_ms.saturating_add(engine_predict_ms);

    let effective_eval_mb = if inputs.eval_invocations > 0 {
        inputs.eval_memory_mb.max(DEPLOYED_EVAL_MEMORY_MB)
    } else {
        inputs.eval_memory_mb
    };

    // API Gateway HTTP API v2.0 meters requests in 512 KiB increments.
    let extra_512k_units = (inputs.http_request_bytes / 524_288) as u32;
    let apigw_metered_units = if inputs.ingress_mode == CarbonIngressCostMode::ApiGatewayHttpV2 {
        edge_invocations.saturating_add(extra_512k_units)
    } else {
        0
    };

    // S3 PUT/LIST: 2 presigned CAS blobs + request.json + initial status.json + wave status writes
    // + (sealed_checkpoint.enc + sealed_champion.enc + final status.json) per search leg + optional stop.json + optional shred LIST.
    let s3_put_list_requests = 4u32
        .saturating_add(inputs.waves_completed)
        .saturating_add(3u32.saturating_mul(search_legs))
        .saturating_add(stop_calls)
        .saturating_add(shred_calls);

    // S3 GET: 2 CAS blob reads per search leg + 2 state reads per resume leg + status.json per poll
    // + 2 state reads on `/predict` + 1 cold `fit.jmat` read per unique `eval` shard worker (`LRU` cached across later waves).
    let resume_legs = search_legs.saturating_sub(1);
    let cold_eval_reads = if inputs.eval_invocations > 0 {
        inputs
            .eval_shards
            .max(1)
            .min(inputs.eval_invocations as u32)
    } else {
        0
    };
    let s3_get_requests = 2u32
        .saturating_mul(search_legs)
        .saturating_add(2u32.saturating_mul(resume_legs))
        .saturating_add(inputs.poll_requests)
        .saturating_add(2)
        .saturating_add(cold_eval_reads);

    if inputs.ingress_mode == CarbonIngressCostMode::SelfHostedVpc {
        return CarbonRunCostBreakdown {
            ingress_mode: inputs.ingress_mode,
            edge_invocations,
            engine_invocations,
            engine_billed_ms,
            eval_invocations: inputs.eval_invocations,
            eval_memory_mb: effective_eval_mb,
            eval_billed_ms: inputs.eval_total_wall_ms,
            apigw_metered_units: 0,
            s3_put_list_requests,
            s3_get_requests,
            lambda_edge_usd: 0.0,
            lambda_engine_usd: 0.0,
            lambda_eval_usd: 0.0,
            apigw_http_api_usd: 0.0,
            s3_requests_usd: 0.0,
            s3_storage_and_transfer_usd: 0.0,
            total_estimated_usd: 0.0,
            runs_per_1_usd: None,
        };
    }

    let edge_billed_ms = u64::from(edge_invocations) * 18;
    let edge_gb = (EDGE_MEMORY_MB as f64) / 1_024.0;
    let lambda_edge_usd = (edge_billed_ms as f64 / 1_000.0) * edge_gb * LAMBDA_ARM64_USD_PER_GB_SEC
        + f64::from(edge_invocations) * LAMBDA_USD_PER_INVOCATION;

    let engine_gb = (ENGINE_MEMORY_MB as f64) / 1_024.0;
    let lambda_engine_usd =
        (engine_billed_ms as f64 / 1_000.0) * engine_gb * LAMBDA_ARM64_USD_PER_GB_SEC
            + f64::from(engine_invocations) * LAMBDA_USD_PER_INVOCATION;

    let eval_gb = (effective_eval_mb as f64) / 1_024.0;
    let lambda_eval_usd = if inputs.eval_invocations == 0 {
        0.0
    } else {
        (inputs.eval_total_wall_ms as f64 / 1_000.0) * eval_gb * LAMBDA_ARM64_USD_PER_GB_SEC
            + (inputs.eval_invocations as f64) * LAMBDA_USD_PER_INVOCATION
    };

    let apigw_http_api_usd = f64::from(apigw_metered_units) * APIGW_HTTP_API_USD_PER_UNIT;
    let s3_requests_usd = f64::from(s3_put_list_requests) * S3_STANDARD_PUT_LIST_USD_PER_REQ
        + f64::from(s3_get_requests) * S3_STANDARD_GET_USD_PER_REQ;

    let cas_gb = (inputs.uploaded_cas_bytes as f64) / 1_000_000_000.0;
    let retention_months = if inputs.shredded_after_run {
        1.0 / 8_640.0 // < 5 minutes before cryptographic DELETE shred
    } else {
        1.0 / 30.0 // 1-day lifecycle rule on `blobs/`
    };
    let s3_storage_usd = cas_gb * S3_STANDARD_USD_PER_GB_MONTH * retention_months;
    let egress_gb = (inputs.http_response_bytes as f64) / 1_000_000_000.0;
    let s3_storage_and_transfer_usd = s3_storage_usd + egress_gb * AWS_INTERNET_EGRESS_USD_PER_GB;

    let total_estimated_usd = lambda_edge_usd
        + lambda_engine_usd
        + lambda_eval_usd
        + apigw_http_api_usd
        + s3_requests_usd
        + s3_storage_and_transfer_usd;
    let runs_per_1_usd = if total_estimated_usd > 0.0 {
        Some(1.0 / total_estimated_usd)
    } else {
        None
    };

    CarbonRunCostBreakdown {
        ingress_mode: inputs.ingress_mode,
        edge_invocations,
        engine_invocations,
        engine_billed_ms,
        eval_invocations: inputs.eval_invocations,
        eval_memory_mb: effective_eval_mb,
        eval_billed_ms: inputs.eval_total_wall_ms,
        apigw_metered_units,
        s3_put_list_requests,
        s3_get_requests,
        lambda_edge_usd,
        lambda_engine_usd,
        lambda_eval_usd,
        apigw_http_api_usd,
        s3_requests_usd,
        s3_storage_and_transfer_usd,
        total_estimated_usd,
        runs_per_1_usd,
    }
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
    pub full_dataset_cost_estimate: CarbonRunCostBreakdown,
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

/// Empirically calibrated Graviton2 (`1.0 vCPU`) per-candidate 5-fold inner-CV GP evaluation
/// latency in milliseconds as a function of `(fit_rows, features)`.
pub fn estimate_ms_per_candidate_1vcpu(fit_rows: usize, features: usize) -> f64 {
    let n = (fit_rows.max(1)) as f64;
    let p = (features.max(1)) as f64;
    let linear_term = 8.0 + 0.045 * n * (1.0 + 0.08 * p);
    let superlinear_term = if n > 350.0 {
        let excess = (n - 350.0) / 100.0;
        18.0 * excess.powf(2.15) * (1.0 + 0.05 * p)
    } else {
        0.0
    };
    (linear_term + superlinear_term).clamp(8.0, 25_000.0)
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

    let eval_class = if ((input.product_tier != CarbonProductTier::Free
        && input.effort >= CarbonEffortTier::High)
        || solo_wall_ms > 8_000.0)
        && base_class < EvalMemoryClass::M1769Mb
    {
        EvalMemoryClass::M1769Mb
    } else {
        base_class
    };

    let target_cands_per_shard = if input.est_ms_per_candidate_1vcpu >= 250.0 {
        4u32
    } else if input.est_ms_per_candidate_1vcpu >= 80.0 && cands_per_wave <= 64 {
        8u32
    } else {
        16u32
    };
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
    let mode = if solo_wall_ms <= 8_000.0 {
        CarbonExecutionMode::SoloSingleLambda
    } else if solo_wall_ms > 0.45 * lease_budget_ms
        || (solo_wall_ms / hybrid_wall_ms >= 2.0
            && hybrid_estimated_usd <= solo_estimated_usd * 1.45)
    {
        CarbonExecutionMode::HybridFanOut
    } else {
        CarbonExecutionMode::SoloSingleLambda
    };

    let selected_shards = if mode == CarbonExecutionMode::HybridFanOut {
        shards_per_wave
    } else {
        0
    };
    let selected_wall_ms = if mode == CarbonExecutionMode::HybridFanOut {
        hybrid_wall_ms
    } else {
        solo_wall_ms
    };
    let est_polls = if input.effort == CarbonEffortTier::Fast && selected_wall_ms < 1_000.0 {
        0
    } else if selected_wall_ms <= 5_000.0 {
        ((selected_wall_ms / 300.0).ceil() as u32).max(1)
    } else {
        12 + (((selected_wall_ms - 5_000.0) / 1_000.0).ceil() as u32)
    };
    let est_cas_bytes = input
        .total_rows
        .saturating_mul(input.features.saturating_add(1))
        .saturating_mul(8)
        .saturating_add(512);
    let est_eval_inv = if mode == CarbonExecutionMode::HybridFanOut {
        u64::from(total_eval_invocations)
    } else {
        0
    };
    let est_eval_wall_ms = if mode == CarbonExecutionMode::HybridFanOut {
        u64::from(total_eval_invocations) * (shard_compute_ms.ceil().max(1.0) as u64)
    } else {
        0
    };
    let ingress = if input.product_tier == CarbonProductTier::Enterprise {
        CarbonIngressCostMode::SelfHostedVpc
    } else {
        CarbonIngressCostMode::ApiGatewayHttpV2
    };
    let full_dataset_cost_estimate = compute_live_run_cost(&CarbonLiveCostInputs {
        ingress_mode: ingress,
        minted_ephemeral_token: false,
        engine_search_invocations: 1,
        poll_requests: est_polls,
        signaled_early_stop: false,
        shredded_after_run: true,
        waves_completed: waves,
        eval_shards: selected_shards,
        eval_memory_mb: eval_class.memory_mb() as u32,
        eval_invocations: est_eval_inv,
        eval_total_wall_ms: est_eval_wall_ms,
        search_wall_ms: selected_wall_ms,
        predict_wall_ms: 45.0,
        uploaded_cas_bytes: est_cas_bytes,
        http_request_bytes: 12_288,
        http_response_bytes: 8_192,
    });

    Ok(CarbonPlanDecision {
        mode,
        eval_class,
        estimated_working_set_mb: working_set_mb,
        shards_per_wave: selected_shards,
        planned_waves: waves,
        solo_wall_ms,
        hybrid_wall_ms,
        solo_estimated_usd,
        hybrid_estimated_usd,
        full_dataset_cost_estimate,
    })
}
