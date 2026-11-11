//! CTO 3-Gate (`S-1` Spend, `S-2` Load Shape / Duty Cycle, `S-3` Burstiness) evaluator deciding
//! when sustained regional load justifies warm-standby Lambda vs optional Enterprise EC2 Spot
//! overflow ([ADR-0007](../../../docs/adr/ADR-0007-carbon-lambda-cells.md)).

use serde::{Deserialize, Serialize};

/// `MonthlyWindowMetrics` is a struct for a 30-day regional telemetry snapshot evaluated against
/// the three Spot-Hybrid break-even gates.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MonthlyWindowMetrics {
    /// AWS region identifier (for example `"us-east-2"`).
    pub region_code: [u8; 16],
    /// Monthly Lambda compute spend in USD for this region (`Gate S-1`).
    pub monthly_lambda_spend_usd: f64,
    /// Sustained 5-minute P20 active vCPU floor (`Gate S-2`: baseline trough load).
    pub p20_active_vcpus: f64,
    /// Monthly mean active vCPU load.
    pub mean_active_vcpus: f64,
    /// Monthly P95 active vCPU peak (`Gate S-3`: `burstiness = p95 / max(mean, eps)`).
    pub p95_active_vcpus: f64,
    /// Observed cold-start rate fraction (`0.0..=1.0`) on `carbon-engine`.
    pub engine_cold_start_rate: f64,
}

impl MonthlyWindowMetrics {
    /// Helper constructor encoding a region string into the fixed 16-byte key.
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

    /// Burstiness ratio `P95 / mean` (`f64::INFINITY` when `mean <= 1e-6`).
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
    /// Minimum consecutive qualifying 30-day windows required (`default: 2`).
    pub required_consecutive_months: usize,
    /// `Gate S-1`: Minimum monthly regional Lambda compute spend in USD (`default: $2,500.0`).
    pub min_monthly_spend_usd: f64,
    /// `Gate S-2`: Minimum sustained P20 baseline active vCPUs (`default: 16.0` vCPUs, ~38% duty cycle of a 2-AZ `c7g.4xlarge` pair + warm pool).
    pub min_p20_active_vcpus: f64,
    /// `Gate S-3`: Maximum `P95 / mean` burstiness ratio (`default: 3.5`; burstier traffic makes idle Spot capacity costlier than pure Lambda).
    pub max_burstiness_ratio: f64,
    /// Cold-start rate threshold above which Step 1.5 (`EnableWarmStandbyLambda`) triggers before Spot (`default: 0.05`).
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
    /// Keep 100% pure on-demand Lambda (`$0` idle cost; optimal for spiky or sub-threshold load).
    StayPureLambda,
    /// Step 1.5: enable `1-2` provisioned/warm-standby `carbon-engine` instances (`~$60/mo`) without Spot fleet ops.
    EnableWarmStandbyLambda,
    /// All 3 gates (`S-1`, `S-2`, `S-3`) passed for `>= required_consecutive_months`: eligible to route async `high`/`ultra` waves to `carbon serve` Spot pool with automatic Lambda fallback.
    EligibleForEnterpriseSpotOverflow,
}

/// `SpotGateVerdict` is a struct recording per-gate pass counts and the final recommendation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpotGateVerdict {
    /// Consecutive trailing windows passing all three gates (`S-1 && S-2 && S-3`).
    pub consecutive_qualifying_months: usize,
    /// Whether the most recent window passed `Gate S-1` (Spend).
    pub latest_spend_gate_passed: bool,
    /// Whether the most recent window passed `Gate S-2` (Duty-cycle floor).
    pub latest_duty_cycle_gate_passed: bool,
    /// Whether the most recent window passed `Gate S-3` (Bounded burstiness).
    pub latest_burstiness_gate_passed: bool,
    /// Final recommended execution posture.
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
