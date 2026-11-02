use super::*;

fn lift_panel_dataset_precedes(left: &LiftPanelDataset, right: &LiftPanelDataset) -> bool {
    (
        task_rank(&left.task),
        &left.source_family,
        &left.input_identity_sha256,
        &left.test_sha256,
        left.outer_fold,
    ) < (
        task_rank(&right.task),
        &right.source_family,
        &right.input_identity_sha256,
        &right.test_sha256,
        right.outer_fold,
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SamplingWeight {
    pub numerator: u8,
    pub denominator: u8,
}

pub fn eight_slot_sampling_weights(
    multiclass_ready: bool,
    timeseries_ready: bool,
) -> BTreeMap<&'static str, SamplingWeight> {
    let missing = u8::from(!multiclass_ready) + u8::from(!timeseries_ready);
    let denominator = 16;
    let core = 6 + missing;
    BTreeMap::from([
        (
            "regression",
            SamplingWeight {
                numerator: core,
                denominator,
            },
        ),
        (
            "binary",
            SamplingWeight {
                numerator: core,
                denominator,
            },
        ),
        (
            "multiclass",
            SamplingWeight {
                numerator: if multiclass_ready { 2 } else { 0 },
                denominator,
            },
        ),
        (
            "timeseries",
            SamplingWeight {
                numerator: if timeseries_ready { 2 } else { 0 },
                denominator,
            },
        ),
    ])
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LiftPanelDataset {
    pub task: String,
    pub source_family: String,
    pub input_identity_sha256: String,
    pub test_sha256: String,
    pub outer_fold: u8,
}

impl LiftPanelDataset {
    fn validate(&self) -> Result<()> {
        ensure!(valid_task(&self.task), "lift-panel task is invalid");
        ensure!(
            !self.source_family.is_empty(),
            "lift-panel family is absent"
        );
        require_sha256(&self.input_identity_sha256, "lift input identity")?;
        require_sha256(&self.test_sha256, "lift test SHA-256")?;
        ensure!(self.outer_fold < 5, "lift outer fold is invalid");
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EncoderLiftExperimentV2 {
    pub schema: String,
    pub generated_at_unix_ms: u64,
    pub freshness_budget_seconds: u64,
    pub input_digests: Vec<String>,
    pub v1_capsule_sha256: String,
    pub v2_encoder_identity_sha256: String,
    pub panel: Vec<LiftPanelDataset>,
    pub common_contexts_sha256: String,
    pub common_candidates_sha256: String,
    pub common_seeds_sha256: String,
    pub common_evaluators_sha256: String,
    pub failure_policy_sha256: String,
    pub standardization_fit_policy: String,
    pub eight_slot_compute: bool,
    pub family_cluster_bootstrap_replicates: u32,
    pub multiplicity_correction: String,
    pub bounded_metric_noninferiority_margin: String,
    pub native_error_noninferiority_ratio: String,
    pub blockers: Vec<String>,
}

impl EncoderLiftExperimentV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == LIFT_EXPERIMENT_SCHEMA,
            "invalid lift experiment schema"
        );
        validate_status(
            self.generated_at_unix_ms,
            self.freshness_budget_seconds,
            &self.input_digests,
        )?;
        for (value, field) in [
            (&self.v1_capsule_sha256, "V1 capsule"),
            (&self.v2_encoder_identity_sha256, "V2 encoder identity"),
            (&self.common_contexts_sha256, "common contexts"),
            (&self.common_candidates_sha256, "common candidates"),
            (&self.common_seeds_sha256, "common seeds"),
            (&self.common_evaluators_sha256, "common evaluators"),
            (&self.failure_policy_sha256, "failure policy"),
        ] {
            require_sha256(value, field)?;
        }
        ensure!(self.panel.len() == 426, "lift panel is not 426 datasets");
        let mut identities = BTreeSet::new();
        let mut family_folds = BTreeMap::new();
        for dataset in &self.panel {
            dataset.validate()?;
            ensure!(
                identities.insert(dataset.input_identity_sha256.as_str()),
                "lift panel repeats an input identity"
            );
            if let Some(existing) = family_folds.insert(&dataset.source_family, dataset.outer_fold)
            {
                ensure!(
                    existing == dataset.outer_fold,
                    "source family appears in more than one held-out fold"
                );
            }
        }
        for (task, count) in [
            ("regression", 128usize),
            ("binary", 128),
            ("multiclass", 128),
            ("timeseries", 42),
        ] {
            ensure!(
                self.panel.iter().filter(|row| row.task == task).count() == count,
                "lift panel has the wrong {task} count"
            );
        }
        let folds = self
            .panel
            .iter()
            .map(|row| row.outer_fold)
            .collect::<BTreeSet<_>>();
        ensure!(
            folds == BTreeSet::from([0, 1, 2, 3, 4]),
            "lift panel omits a fold"
        );
        ensure!(
            self.panel
                .windows(2)
                .all(|pair| lift_panel_dataset_precedes(&pair[0], &pair[1])),
            "lift panel is not in frozen order"
        );
        ensure!(
            self.standardization_fit_policy == "outer_fold_training_families_only",
            "lift standardization policy permits held-out leakage"
        );
        ensure!(
            self.eight_slot_compute,
            "lift compute is not matched at eight slots"
        );
        ensure!(
            self.family_cluster_bootstrap_replicates == 10_000,
            "lift bootstrap replicate count drift"
        );
        ensure!(
            self.multiplicity_correction == "holm",
            "lift correction is not Holm"
        );
        ensure!(
            self.bounded_metric_noninferiority_margin == "-0.005"
                && self.native_error_noninferiority_ratio == "1.01",
            "lift noninferiority authority drift"
        );
        ensure!(self.blockers.is_empty(), "lift experiment has blockers");
        Ok(())
    }

    pub fn digest(&self) -> Result<String> {
        self.validate()?;
        canonical_json_sha256(&serde_json::to_value(self)?)
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LiftTaskResult {
    pub task: String,
    pub corrected_primary_delta_lower_95: Option<f64>,
    pub native_error_ratio_upper_95: Option<f64>,
    pub strict_win_rate_lower_95: f64,
    pub p95_latency_ratio: f64,
    pub failure_rate_delta: f64,
    pub bounded_metric_worst_delta: Option<f64>,
    pub native_error_worst_ratio: Option<f64>,
    pub directional_agreement: bool,
    pub typed_absence: Option<String>,
}

impl LiftTaskResult {
    fn validate_numbers(&self) -> Result<()> {
        ensure!(valid_task(&self.task), "lift result task is invalid");
        for value in [
            Some(self.strict_win_rate_lower_95),
            Some(self.p95_latency_ratio),
            Some(self.failure_rate_delta),
            self.corrected_primary_delta_lower_95,
            self.native_error_ratio_upper_95,
            self.bounded_metric_worst_delta,
            self.native_error_worst_ratio,
        ]
        .into_iter()
        .flatten()
        {
            ensure!(value.is_finite(), "lift task metric is nonfinite");
        }
        Ok(())
    }

    fn noninferior(&self) -> Result<bool> {
        self.validate_numbers()?;
        match self.task.as_str() {
            "binary" | "multiclass" => Ok(self
                .corrected_primary_delta_lower_95
                .is_some_and(|lower| lower > -0.005)),
            "regression" | "timeseries" => Ok(self
                .native_error_ratio_upper_95
                .is_some_and(|upper| upper < 1.01)),
            _ => unreachable!(),
        }
    }

    fn catastrophic(&self) -> bool {
        self.bounded_metric_worst_delta
            .is_some_and(|delta| delta < -0.02)
            || self
                .native_error_worst_ratio
                .is_some_and(|ratio| ratio > 1.05)
    }

    fn positive_quality(&self) -> bool {
        self.corrected_primary_delta_lower_95
            .is_some_and(|lower| lower > 0.005)
            || self
                .native_error_ratio_upper_95
                .is_some_and(|ratio| ratio <= 0.99)
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EncoderLiftReportV2 {
    pub schema: String,
    pub generated_at_unix_ms: u64,
    pub freshness_budget_seconds: u64,
    pub input_digests: Vec<String>,
    pub experiment_sha256: String,
    pub benchmark_report_sha256: String,
    pub speed_gate_passed: bool,
    pub tasks: Vec<LiftTaskResult>,
    pub leakage_detected: bool,
    pub identity_collision_detected: bool,
    pub reproducible: bool,
    pub advancement_allowed: bool,
    pub blockers: Vec<String>,
}

impl EncoderLiftReportV2 {
    pub fn computed_advancement(&self) -> Result<bool> {
        ensure!(
            self.tasks.len() == 4
                && self.tasks[0].task == "regression"
                && self.tasks[1].task == "binary"
                && self.tasks[2].task == "multiclass"
                && self.tasks[3].task == "timeseries",
            "lift report does not contain four ordered task rows"
        );
        for task in &self.tasks {
            task.validate_numbers()?;
        }
        let all_noninferior = self
            .tasks
            .iter()
            .map(LiftTaskResult::noninferior)
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .all(|value| value);
        let positive_quality = self.tasks.iter().any(LiftTaskResult::positive_quality);
        let common = self.tasks.iter().all(|task| {
            task.strict_win_rate_lower_95 > 0.5
                && task.p95_latency_ratio <= 1.25
                && task.failure_rate_delta <= 0.0025
                && task.directional_agreement
                && task.typed_absence.is_none()
                && !task.catastrophic()
        });
        Ok(all_noninferior
            && (positive_quality || self.speed_gate_passed)
            && common
            && !self.leakage_detected
            && !self.identity_collision_detected
            && self.reproducible)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == LIFT_REPORT_SCHEMA,
            "invalid lift-report schema"
        );
        validate_status(
            self.generated_at_unix_ms,
            self.freshness_budget_seconds,
            &self.input_digests,
        )?;
        require_sha256(&self.experiment_sha256, "lift experiment SHA-256")?;
        require_sha256(&self.benchmark_report_sha256, "benchmark report SHA-256")?;
        let computed = self.computed_advancement()?;
        ensure!(
            self.advancement_allowed == computed,
            "lift advancement claim does not match preregistered gates"
        );
        if computed {
            ensure!(
                self.blockers.is_empty(),
                "advancing lift report has blockers"
            );
        } else {
            ensure!(
                !self.blockers.is_empty(),
                "blocked lift report lacks explicit blockers"
            );
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String> {
        self.validate()?;
        canonical_json_sha256(&serde_json::to_value(self)?)
    }
}
