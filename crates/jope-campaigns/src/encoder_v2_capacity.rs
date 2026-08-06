//! Bounded, recovery-first capacity policy for the `jope-encoder-v2` lane.

use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

pub const ENCODER_PROJECT: &str = "jope-encoder-v2";
pub const CAPACITY_OBSERVATION_SCHEMA: &str = "JopeEncoderCapacityObservation.v1";
pub const CPUS_PER_WORKER: u32 = 4;
pub const MEMORY_BYTES_PER_WORKER: u64 = 16 * 1024 * 1024 * 1024;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EncoderCapacityObservation {
    pub schema: String,
    pub generated_at_unix_ms: u64,
    pub freshness_budget_seconds: u64,
    pub input_digests: Vec<String>,
    pub current_workers: u32,
    pub eligible_nodes: u32,
    pub successful_outputs_at_rung: u64,
    pub green_minutes_at_rung: u64,
    pub collision_count: u64,
    pub fallback_count: u64,
    pub oom_as_success_count: u64,
    pub unexplained_drift_count: u64,
    pub ram_headroom_ratio: f64,
    pub vram_headroom_ratio: f64,
    pub cpu_psi: f64,
    pub memory_psi: f64,
    pub io_psi: f64,
    pub free_memory_fraction: f64,
    pub supply_status_v2_fresh_green: bool,
    pub useful_work_at_target: bool,
    pub protected_complete_waves: u32,
    pub companion_cpu_credits: u32,
    pub trainer_active: bool,
    pub critical_runway: bool,
    pub blockers: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CapacityDecision {
    Hold { workers: u32 },
    Advance { workers: u32 },
    GracefulPreempt { reason: &'static str },
}

impl EncoderCapacityObservation {
    pub fn decide_at(&self, now_unix_ms: u64) -> Result<CapacityDecision> {
        ensure!(
            self.schema == CAPACITY_OBSERVATION_SCHEMA,
            "capacity observation schema drift"
        );
        ensure!(
            self.generated_at_unix_ms > 0 && self.generated_at_unix_ms <= now_unix_ms,
            "capacity observation has no valid timestamp"
        );
        ensure!(
            self.freshness_budget_seconds > 0
                && self.freshness_budget_seconds <= 300
                && now_unix_ms - self.generated_at_unix_ms
                    <= self.freshness_budget_seconds.saturating_mul(1_000),
            "capacity observation is stale or has an invalid freshness budget"
        );
        ensure!(
            !self.input_digests.is_empty()
                && self.input_digests.windows(2).all(|pair| pair[0] < pair[1])
                && self.input_digests.iter().all(|digest| {
                    digest.len() == 64
                        && digest
                            .bytes()
                            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
                }),
            "capacity input digests are absent, invalid, or noncanonical"
        );
        ensure!(
            self.current_workers <= self.eligible_nodes,
            "more Encoder V2 workers than eligible nodes"
        );
        for (value, field) in [
            (self.ram_headroom_ratio, "RAM headroom"),
            (self.vram_headroom_ratio, "VRAM headroom"),
            (self.cpu_psi, "CPU PSI"),
            (self.memory_psi, "memory PSI"),
            (self.io_psi, "I/O PSI"),
            (self.free_memory_fraction, "free-memory fraction"),
        ] {
            ensure!(value.is_finite() && value >= 0.0, "{field} is invalid");
        }
        if self.critical_runway
            || !self.useful_work_at_target
            || self.protected_complete_waves < 2
            || !self.supply_status_v2_fresh_green
        {
            return Ok(CapacityDecision::GracefulPreempt {
                reason: "recovery_runway",
            });
        }
        ensure!(
            self.blockers.is_empty(),
            "green capacity observation has blockers"
        );
        if self.cpu_psi > 0.25
            || self.memory_psi > 0.05
            || self.io_psi > 0.10
            || self.free_memory_fraction < 0.15
            || self.companion_cpu_credits < CPUS_PER_WORKER
            || self.trainer_active
        {
            return Ok(CapacityDecision::GracefulPreempt {
                reason: "resource_pressure",
            });
        }
        let clean = self.collision_count == 0
            && self.fallback_count == 0
            && self.oom_as_success_count == 0
            && self.unexplained_drift_count == 0;
        let proven = self.successful_outputs_at_rung >= 128
            && self.green_minutes_at_rung >= 30
            && self.ram_headroom_ratio >= 1.5
            && self.vram_headroom_ratio >= 1.5;
        let maximum = self.eligible_nodes;
        if clean && self.current_workers == 0 && maximum > 0 {
            return Ok(CapacityDecision::Advance { workers: 1 });
        }
        if clean && proven && self.current_workers < maximum {
            let next = match self.current_workers {
                0 => 1,
                1 => 2.min(maximum),
                _ => maximum,
            };
            return Ok(CapacityDecision::Advance { workers: next });
        }
        Ok(CapacityDecision::Hold {
            workers: self.current_workers,
        })
    }
}
