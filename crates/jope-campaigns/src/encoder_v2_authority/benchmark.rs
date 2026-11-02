use super::*;

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkDataset {
    pub task: String,
    pub source_family: String,
    pub row_log2_bin: u8,
    pub column_log2_bin: u8,
    pub byte_size_log2_bin: u8,
    pub input_identity_sha256: String,
    pub train_sha256: String,
}

impl BenchmarkDataset {
    fn validate(&self) -> Result<()> {
        ensure!(valid_task(&self.task), "benchmark dataset task is invalid");
        ensure!(
            !self.source_family.is_empty(),
            "benchmark source family is absent"
        );
        require_sha256(&self.input_identity_sha256, "benchmark input identity")?;
        require_sha256(&self.train_sha256, "benchmark train SHA-256")?;
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EncoderBenchmarkPlanV2 {
    pub schema: String,
    pub generated_at_unix_ms: u64,
    pub freshness_budget_seconds: u64,
    pub input_digests: Vec<String>,
    pub cohort_seed: u64,
    pub release_cohort: Vec<BenchmarkDataset>,
    pub tuning_cohort: Vec<BenchmarkDataset>,
    pub baseline_capsule_sha256: String,
    pub candidate_encoder_identity_sha256: String,
    pub hardware_capability_receipt_sha256: String,
    pub cache_reset_protocol_sha256: String,
    pub proof_chunk_size: u32,
    pub proof_datasets_per_task: u32,
    pub replay_chunk_size: u32,
    pub replay_datasets_per_task: u32,
    pub backlog_chunk_max: u32,
    pub untimed_warmups: u32,
    pub timed_repetitions: u32,
    pub cold_and_warm_lanes: bool,
    pub blockers: Vec<String>,
}

impl EncoderBenchmarkPlanV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == BENCHMARK_PLAN_SCHEMA,
            "invalid benchmark-plan schema"
        );
        validate_status(
            self.generated_at_unix_ms,
            self.freshness_budget_seconds,
            &self.input_digests,
        )?;
        ensure!(
            self.cohort_seed == COHORT_SEED,
            "benchmark cohort seed drift"
        );
        for (value, field) in [
            (&self.baseline_capsule_sha256, "baseline capsule"),
            (
                &self.candidate_encoder_identity_sha256,
                "candidate encoder identity",
            ),
            (
                &self.hardware_capability_receipt_sha256,
                "hardware capability receipt",
            ),
            (&self.cache_reset_protocol_sha256, "cache reset protocol"),
        ] {
            require_sha256(value, field)?;
        }
        ensure!(
            self.release_cohort.len() == 128,
            "release cohort is not 128 inputs"
        );
        ensure!(
            self.tuning_cohort.len() == 58,
            "tuning cohort is not 58 inputs"
        );
        let mut identities = BTreeSet::new();
        for dataset in self.release_cohort.iter().chain(self.tuning_cohort.iter()) {
            dataset.validate()?;
            ensure!(
                identities.insert(dataset.input_identity_sha256.as_str()),
                "benchmark tuning/release cohorts overlap"
            );
        }
        for task in ["regression", "binary", "multiclass", "timeseries"] {
            ensure!(
                self.release_cohort
                    .iter()
                    .filter(|row| row.task == task)
                    .count()
                    == 32,
                "release cohort does not contain 32 {task} inputs"
            );
        }
        for (task, count) in [
            ("regression", 16usize),
            ("binary", 16),
            ("multiclass", 16),
            ("timeseries", 10),
        ] {
            ensure!(
                self.tuning_cohort
                    .iter()
                    .filter(|row| row.task == task)
                    .count()
                    == count,
                "tuning cohort has the wrong {task} count"
            );
        }
        ensure!(
            self.release_cohort
                .windows(2)
                .all(|pair| benchmark_dataset_precedes(&pair[0], &pair[1]))
                && self
                    .tuning_cohort
                    .windows(2)
                    .all(|pair| benchmark_dataset_precedes(&pair[0], &pair[1])),
            "benchmark cohorts are not in frozen stratification order"
        );
        ensure!(
            self.proof_chunk_size == 32
                && self.proof_datasets_per_task == 8
                && self.replay_chunk_size == 128
                && self.replay_datasets_per_task == 32
                && self.backlog_chunk_max == 256,
            "benchmark chunk ladder drift"
        );
        ensure!(
            self.untimed_warmups == 1 && self.timed_repetitions >= 3,
            "benchmark repetitions are insufficient"
        );
        ensure!(
            self.cold_and_warm_lanes,
            "benchmark omits cold or warm lane"
        );
        ensure!(self.blockers.is_empty(), "benchmark plan has blockers");
        Ok(())
    }

    pub fn digest(&self) -> Result<String> {
        self.validate()?;
        canonical_json_sha256(&serde_json::to_value(self)?)
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkLaneMetrics {
    pub samples: u64,
    pub p50_latency_ms: f64,
    pub p90_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub throughput_per_gpu_slot: f64,
    pub cpu_hours: f64,
    pub load_time_ms: f64,
    pub preprocess_time_ms: f64,
    pub kernel_time_ms: f64,
    pub verification_publication_ms: f64,
    pub cache_lookup_ms: f64,
    pub peak_ram_bytes: u64,
    pub peak_vram_bytes: u64,
    pub failures: u64,
}

impl BenchmarkLaneMetrics {
    fn validate(&self) -> Result<()> {
        ensure!(self.samples > 0, "benchmark lane has no samples");
        for value in [
            self.p50_latency_ms,
            self.p90_latency_ms,
            self.p95_latency_ms,
            self.p99_latency_ms,
            self.throughput_per_gpu_slot,
            self.cpu_hours,
            self.load_time_ms,
            self.preprocess_time_ms,
            self.kernel_time_ms,
            self.verification_publication_ms,
            self.cache_lookup_ms,
        ] {
            ensure!(
                value.is_finite() && value >= 0.0,
                "benchmark metric is invalid"
            );
        }
        ensure!(
            self.p50_latency_ms <= self.p90_latency_ms
                && self.p90_latency_ms <= self.p95_latency_ms
                && self.p95_latency_ms <= self.p99_latency_ms,
            "benchmark latency percentiles are not monotonic"
        );
        ensure!(
            self.throughput_per_gpu_slot > 0.0,
            "benchmark throughput is zero"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EncoderBenchmarkReportV2 {
    pub schema: String,
    pub generated_at_unix_ms: u64,
    pub freshness_budget_seconds: u64,
    pub input_digests: Vec<String>,
    pub benchmark_plan_sha256: String,
    pub baseline: BenchmarkLaneMetrics,
    pub candidate: BenchmarkLaneMetrics,
    pub cache_hit: BenchmarkLaneMetrics,
    pub paired_bootstrap_throughput_ratio_lower_95: f64,
    pub control_plane_overhead_fraction: f64,
    pub output_drift_count: u64,
    pub fallback_count: u64,
    pub backend_mismatch_count: u64,
    pub collision_count: u64,
    pub oom_as_success_count: u64,
    pub useful_work_green: bool,
    pub protected_reserves_green: bool,
    pub companion_cpu_credits_green: bool,
    pub release_speed_gate_passed: bool,
    pub blockers: Vec<String>,
}

impl EncoderBenchmarkReportV2 {
    pub fn computed_release_gate(&self) -> Result<bool> {
        self.baseline.validate()?;
        self.candidate.validate()?;
        self.cache_hit.validate()?;
        ensure!(
            self.paired_bootstrap_throughput_ratio_lower_95.is_finite(),
            "benchmark bootstrap bound is invalid"
        );
        ensure!(
            self.control_plane_overhead_fraction.is_finite()
                && self.control_plane_overhead_fraction >= 0.0,
            "control-plane overhead is invalid"
        );
        let throughput_ratio =
            self.candidate.throughput_per_gpu_slot / self.baseline.throughput_per_gpu_slot;
        let cache_budget = 250.0_f64.min(self.candidate.p95_latency_ms / 20.0);
        Ok(throughput_ratio >= 3.0
            && self.paired_bootstrap_throughput_ratio_lower_95 >= 2.5
            && self.candidate.p95_latency_ms <= self.baseline.p95_latency_ms * 0.5
            && self.cache_hit.p95_latency_ms <= cache_budget
            && self.control_plane_overhead_fraction < 0.05
            && self.output_drift_count == 0
            && self.fallback_count == 0
            && self.backend_mismatch_count == 0
            && self.collision_count == 0
            && self.oom_as_success_count == 0
            && self.useful_work_green
            && self.protected_reserves_green
            && self.companion_cpu_credits_green)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == BENCHMARK_REPORT_SCHEMA,
            "invalid benchmark-report schema"
        );
        validate_status(
            self.generated_at_unix_ms,
            self.freshness_budget_seconds,
            &self.input_digests,
        )?;
        require_sha256(&self.benchmark_plan_sha256, "benchmark plan SHA-256")?;
        let computed = self.computed_release_gate()?;
        ensure!(
            self.release_speed_gate_passed == computed,
            "benchmark release-gate claim does not match measurements"
        );
        if computed {
            ensure!(self.blockers.is_empty(), "passing benchmark has blockers");
        } else {
            ensure!(
                !self.blockers.is_empty(),
                "failed benchmark lacks explicit blockers"
            );
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String> {
        self.validate()?;
        canonical_json_sha256(&serde_json::to_value(self)?)
    }
}
