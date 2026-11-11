//! CTO 3-Gate (`S-1` Spend, `S-2` Load Shape / Duty Cycle, `S-3` Burstiness) evaluator deciding
//! when sustained regional load justifies warm-standby Lambda vs optional Enterprise EC2 Spot
//! overflow ([ADR-0007](../../../../docs/adr/ADR-0007-carbon-lambda-cells.md)).

use serde::{Deserialize, Serialize};

/// `MonthlyWindowMetrics` is a struct for a 30-day regional telemetry snapshot evaluated against
/// the three Spot-Hybrid break-even gates.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MonthlyWindowMetrics {
    pub region_code: [u8; 16],
    pub monthly_lambda_spend_usd: f64,
    pub p20_active_vcpus: f64,
    pub mean_active_vcpus: f64,
    pub p95_active_vcpus: f64,
    pub engine_cold_start_rate: f64,
}

impl MonthlyWindowMetrics {
    pub fn for_region(
        region: &str,
        monthly_lambda_spend_usd: f64,
        p20_active_vcpus: f64,
        mean_active_vcpus: f64,
        p95_active_vcpus: f64,
        engine_cold_start_rate: f64,
    ) -> Self {
        let mut region_code = [0u8; 16];
        let src = region.as_bytes();
        let copy_len = src.len().min(16);
        region_code[..copy_len].copy_from_slice(&src[..copy_len]);
        Self {
            region_code,
            monthly_lambda_spend_usd,
            p20_active_vcpus,
            mean_active_vcpus,
            p95_active_vcpus,
            engine_cold_start_rate,
        }
    }

    pub fn burstiness_ratio(&self) -> f64 {
        if self.mean_active_vcpus <= 1e-6 {
            f64::INFINITY
        } else {
            self.p95_active_vcpus / self.mean_active_vcpus
        }
    }
}

/// `SpotAdvisorThresholds` is a struct for the three conjunctive gates documented in `ADR-0007`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpotAdvisorThresholds {
    pub required_consecutive_months: usize,
    pub min_monthly_spend_usd: f64,
    pub min_p20_active_vcpus: f64,
    pub max_burstiness_ratio: f64,
    pub warm_standby_cold_start_threshold: f64,
}

impl Default for SpotAdvisorThresholds {
    fn default() -> Self {
        Self {
            required_consecutive_months: 2,
            min_monthly_spend_usd: 2_500.0,
            min_p20_active_vcpus: 16.0,
            max_burstiness_ratio: 3.5,
            warm_standby_cold_start_threshold: 0.05,
        }
    }
}

/// `SpotAdvisorRecommendation` is an enum for the architecture posture recommended by the 3-Gate advisor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpotAdvisorRecommendation {
    StayPureLambda,
    EnableWarmStandbyLambda,
    EligibleForEnterpriseSpotOverflow,
}

/// `SpotGateVerdict` is a struct recording per-gate pass counts and the final recommendation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpotGateVerdict {
    pub consecutive_qualifying_months: usize,
    pub latest_spend_gate_passed: bool,
    pub latest_duty_cycle_gate_passed: bool,
    pub latest_burstiness_gate_passed: bool,
    pub recommendation: SpotAdvisorRecommendation,
}

/// Evaluate chronological monthly regional windows against `thresholds`.
pub fn evaluate_spot_hybrid_gates(
    chronological_windows: &[MonthlyWindowMetrics],
    thresholds: &SpotAdvisorThresholds,
) -> SpotGateVerdict {
    let Some(latest) = chronological_windows.last() else {
        return SpotGateVerdict {
            consecutive_qualifying_months: 0,
            latest_spend_gate_passed: false,
            latest_duty_cycle_gate_passed: false,
            latest_burstiness_gate_passed: false,
            recommendation: SpotAdvisorRecommendation::StayPureLambda,
        };
    };

    let passes_window = |w: &MonthlyWindowMetrics| -> (bool, bool, bool) {
        let s1 = w.monthly_lambda_spend_usd >= thresholds.min_monthly_spend_usd;
        let s2 = w.p20_active_vcpus >= thresholds.min_p20_active_vcpus;
        let s3 = w.burstiness_ratio() <= thresholds.max_burstiness_ratio;
        (s1, s2, s3)
    };

    let (latest_s1, latest_s2, latest_s3) = passes_window(latest);
    let mut consecutive = 0usize;
    for w in chronological_windows.iter().rev() {
        let (s1, s2, s3) = passes_window(w);
        if s1 && s2 && s3 {
            consecutive += 1;
        } else {
            break;
        }
    }

    let recommendation = if consecutive >= thresholds.required_consecutive_months.max(1) {
        SpotAdvisorRecommendation::EligibleForEnterpriseSpotOverflow
    } else if latest.engine_cold_start_rate > thresholds.warm_standby_cold_start_threshold
        && latest.monthly_lambda_spend_usd >= 250.0
    {
        SpotAdvisorRecommendation::EnableWarmStandbyLambda
    } else {
        SpotAdvisorRecommendation::StayPureLambda
    };

    SpotGateVerdict {
        consecutive_qualifying_months: consecutive,
        latest_spend_gate_passed: latest_s1,
        latest_duty_cycle_gate_passed: latest_s2,
        latest_burstiness_gate_passed: latest_s3,
        recommendation,
    }
}
