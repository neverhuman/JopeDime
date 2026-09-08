use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use super::identity::{canonical_sha256, validate_schema, validate_sha256};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchTaskV1 {
    Regression,
    Binary,
    Multiclass,
    Timeseries,
}

impl SearchTaskV1 {
    pub const ALL: [Self; 4] = [
        Self::Regression,
        Self::Binary,
        Self::Multiclass,
        Self::Timeseries,
    ];

    pub fn production_qualifiable(self) -> bool {
        matches!(self, Self::Regression | Self::Binary)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateProducerV1 {
    Gp,
    Dl,
    Hybrid,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchCandidateIdentityV1 {
    pub slot: u32,
    pub semantic_sha256: String,
    pub ancestry_sha256: String,
    pub producer: CandidateProducerV1,
    pub seed: u64,
    pub complexity: u64,
    pub compute_units: u64,
}

impl SearchCandidateIdentityV1 {
    pub fn validate(&self) -> Result<(), String> {
        validate_sha256("candidate semantic_sha256", &self.semantic_sha256)?;
        validate_sha256("candidate ancestry_sha256", &self.ancestry_sha256)?;
        if self.compute_units == 0 {
            return Err("candidate compute_units must be positive".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchCandidatePlanV1 {
    pub schema: String,
    pub generation: u32,
    pub candidates: Vec<SearchCandidateIdentityV1>,
}

impl SearchCandidatePlanV1 {
    pub const SCHEMA: &'static str = "SearchCandidatePlan.v1";
    pub const CANDIDATE_COUNT: usize = 60;

    pub fn validate(&self) -> Result<(), String> {
        validate_schema(&self.schema, Self::SCHEMA)?;
        if self.candidates.len() != Self::CANDIDATE_COUNT {
            return Err(format!(
                "candidate plan must contain exactly {} candidates, found {}",
                Self::CANDIDATE_COUNT,
                self.candidates.len()
            ));
        }
        let mut semantics = BTreeSet::new();
        for (slot, candidate) in self.candidates.iter().enumerate() {
            candidate.validate()?;
            if candidate.slot != slot as u32 {
                return Err(format!(
                    "candidate at vector index {slot} declares slot {}",
                    candidate.slot
                ));
            }
            if !semantics.insert(candidate.semantic_sha256.as_str()) {
                return Err(format!(
                    "duplicate candidate semantic identity {}",
                    candidate.semantic_sha256
                ));
            }
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RegisteredFitnessV1 {
    RobustR2 {
        max_rmse: f64,
    },
    BinaryAuc {
        lift_depths: Vec<f64>,
        max_log_loss: f64,
    },
    MulticlassMacroF1 {
        classes: u32,
        min_class_support: u64,
        max_log_loss: f64,
    },
    NegativeMase {
        max_mase: f64,
    },
}

impl RegisteredFitnessV1 {
    pub fn validate_for_task(&self, task: SearchTaskV1) -> Result<(), String> {
        let finite_positive = |name: &str, value: f64| {
            if value.is_finite() && value > 0.0 {
                Ok(())
            } else {
                Err(format!("{name} must be finite and positive"))
            }
        };
        match (task, self) {
            (SearchTaskV1::Regression, Self::RobustR2 { max_rmse }) => {
                finite_positive("max_rmse", *max_rmse)
            }
            (
                SearchTaskV1::Binary,
                Self::BinaryAuc {
                    lift_depths,
                    max_log_loss,
                },
            ) => {
                finite_positive("max_log_loss", *max_log_loss)?;
                if lift_depths.is_empty()
                    || lift_depths
                        .iter()
                        .any(|depth| !depth.is_finite() || *depth <= 0.0 || *depth > 1.0)
                {
                    return Err("binary lift_depths must be non-empty fractions in (0, 1]".into());
                }
                if !lift_depths.windows(2).all(|window| window[0] < window[1]) {
                    return Err("binary lift_depths must be strictly increasing".into());
                }
                Ok(())
            }
            (
                SearchTaskV1::Multiclass,
                Self::MulticlassMacroF1 {
                    classes,
                    min_class_support,
                    max_log_loss,
                },
            ) => {
                if *classes < 2 || *min_class_support == 0 {
                    return Err(
                        "multiclass fitness requires at least two classes and positive support"
                            .into(),
                    );
                }
                finite_positive("max_log_loss", *max_log_loss)
            }
            (SearchTaskV1::Timeseries, Self::NegativeMase { max_mase }) => {
                finite_positive("max_mase", *max_mase)
            }
            _ => Err("registered fitness does not match the declared task".into()),
        }
    }

    pub fn identity(&self) -> Result<String, String> {
        canonical_sha256(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchBudgetV1 {
    pub generations: u32,
    pub candidates_per_generation: u32,
    pub declared_fold_count: u32,
    pub stage_two_candidates: u32,
    pub finalist_candidates: u32,
}

impl SearchBudgetV1 {
    pub fn validate(&self, task: SearchTaskV1) -> Result<(), String> {
        if self.generations == 0
            || self.candidates_per_generation != 60
            || self.stage_two_candidates != 30
            || self.finalist_candidates != 8
        {
            return Err(
                "search budget must declare generations>0 and the frozen 60/30/8 schedule".into(),
            );
        }
        match task {
            SearchTaskV1::Timeseries if self.declared_fold_count < 3 => {
                Err("timeseries requires at least three declared origins".into())
            }
            SearchTaskV1::Timeseries => Ok(()),
            _ if self.declared_fold_count == 5 => Ok(()),
            _ => Err("tabular Search V3 tasks require exactly five folds".into()),
        }
    }

    pub fn fold_evaluation_count(&self) -> u64 {
        u64::from(self.candidates_per_generation)
            + u64::from(self.stage_two_candidates)
            + u64::from(self.finalist_candidates)
                * u64::from(self.declared_fold_count.saturating_sub(2))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConditioningPolicyV1 {
    pub policy_sha256: String,
    pub validation_truth_access: bool,
    pub held_out_release_access: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FutureExportRequirementsV1 {
    pub source_required: bool,
    pub predict_protocol_sha256: String,
    pub benchmark_protocol_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchRunContractV3 {
    pub schema: String,
    pub task: SearchTaskV1,
    pub task_semantics_sha256: String,
    pub training_view_sha256: String,
    pub fold_plan_sha256: String,
    pub candidate_plan_root_sha256: String,
    pub replay_protocol_sha256: String,
    pub evaluator_sha256: String,
    pub dsl_sha256: String,
    pub dependency_lockfile_sha256: String,
    pub math_policy_sha256: String,
    pub backend_policy_sha256: String,
    pub fitness: RegisteredFitnessV1,
    pub conditioning: ConditioningPolicyV1,
    pub budget: SearchBudgetV1,
    pub future_export: FutureExportRequirementsV1,
}

impl SearchRunContractV3 {
    pub const SCHEMA: &'static str = "SearchRunContract.v3";

    pub fn validate(&self) -> Result<(), String> {
        validate_schema(&self.schema, Self::SCHEMA)?;
        for (name, digest) in [
            ("task_semantics_sha256", self.task_semantics_sha256.as_str()),
            ("training_view_sha256", self.training_view_sha256.as_str()),
            ("fold_plan_sha256", self.fold_plan_sha256.as_str()),
            (
                "candidate_plan_root_sha256",
                self.candidate_plan_root_sha256.as_str(),
            ),
            (
                "replay_protocol_sha256",
                self.replay_protocol_sha256.as_str(),
            ),
            ("evaluator_sha256", self.evaluator_sha256.as_str()),
            ("dsl_sha256", self.dsl_sha256.as_str()),
            (
                "dependency_lockfile_sha256",
                self.dependency_lockfile_sha256.as_str(),
            ),
            ("math_policy_sha256", self.math_policy_sha256.as_str()),
            ("backend_policy_sha256", self.backend_policy_sha256.as_str()),
            (
                "conditioning.policy_sha256",
                self.conditioning.policy_sha256.as_str(),
            ),
            (
                "future_export.predict_protocol_sha256",
                self.future_export.predict_protocol_sha256.as_str(),
            ),
            (
                "future_export.benchmark_protocol_sha256",
                self.future_export.benchmark_protocol_sha256.as_str(),
            ),
        ] {
            validate_sha256(name, digest)?;
        }
        if self.conditioning.validation_truth_access || self.conditioning.held_out_release_access {
            return Err(
                "Search V3 conditioning cannot access validation truth or held-out release labels"
                    .into(),
            );
        }
        if !self.future_export.source_required {
            return Err("future export requirements must bind source emission".into());
        }
        self.fitness.validate_for_task(self.task)?;
        self.budget.validate(self.task)
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}
