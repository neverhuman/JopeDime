use super::*;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ThroughputProgressV2 {
    pub queue_depth: usize,
    pub ready_reserve: usize,
    pub dynamic_target: usize,
    pub p95_context_to_ready_ms: u64,
    pub freeze_to_publish_ms: u64,
    pub corpus_lag_labels: u64,
    pub supply_target_fraction: u16,
    pub disk_used_percent: u8,
    pub expansion_paused: bool,
}

impl ThroughputProgressV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.dynamic_target == 4_096
                && self.supply_target_fraction <= 1_000
                && self.disk_used_percent <= 100,
            "flywheel throughput progress is outside its declared bounds"
        );
        ensure!(
            self.disk_used_percent < 85 || self.expansion_paused,
            "catalog expansion must stop at 85% disk usage"
        );
        Ok(())
    }

    pub fn next_expansion_paused(&self, disk_used_percent: u8) -> bool {
        if self.expansion_paused {
            disk_used_percent >= 80
        } else {
            disk_used_percent >= 85
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FlywheelProgressV2 {
    pub schema: String,
    pub generated_at_ms: u128,
    pub accepted_generations: Vec<AcceptedGenerationV2>,
    pub causal_evidence_ledger_sha256: Sha256Digest,
    pub external_evaluation_sha256: BTreeMap<Task, Vec<Sha256Digest>>,
    pub throughput: ThroughputProgressV2,
    pub catalog_coverage: BTreeMap<Task, f64>,
    pub real_data_share: BTreeMap<Task, f64>,
    pub plateau: BTreeMap<Task, PlateauDiagnosisV1>,
    pub self_improving: bool,
    pub recursive_improvement_proven: bool,
}

impl FlywheelProgressV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == FLYWHEEL_PROGRESS_V2_SCHEMA,
            "flywheel progress schema is not v2"
        );
        self.throughput.validate()?;
        for generation in &self.accepted_generations {
            generation.validate()?;
        }
        for diagnosis in self.plateau.values() {
            diagnosis.validate()?;
        }
        ensure!(
            self.catalog_coverage
                .values()
                .chain(self.real_data_share.values())
                .all(|value| value.is_finite() && (0.0..=1.0).contains(value)),
            "flywheel breadth fraction is outside [0,1]"
        );
        let recursive = self.accepted_generations.iter().any(|child| {
            self.accepted_generations.iter().any(|parent| {
                child.task == parent.task
                    && child.parent_checkpoint_sha256 == parent.child_checkpoint_sha256
            })
        });
        ensure!(
            self.self_improving == !self.accepted_generations.is_empty()
                && self.recursive_improvement_proven == recursive,
            "flywheel improvement claim does not match its accepted generation chain"
        );
        Ok(())
    }
}
