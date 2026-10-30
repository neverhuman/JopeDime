//! Effort defaults and admission-time evaluation budget accounting.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Effort {
    Fast,
    Medium,
    High,
    Ultra,
}

impl Effort {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "fast" => Some(Self::Fast),
            "medium" => Some(Self::Medium),
            "high" => Some(Self::High),
            "ultra" => Some(Self::Ultra),
            _ => None,
        }
    }

    pub const fn limits(self) -> EffortLimits {
        match self {
            Self::Fast => EffortLimits {
                evaluations: 24,
                wall_seconds: 10,
                rung1_folds: 3,
                promotion_repeats: 1,
                patience: 2,
                stall_rounds: 1,
            },
            Self::Medium => EffortLimits {
                evaluations: 128,
                wall_seconds: 60,
                rung1_folds: 5,
                promotion_repeats: 1,
                patience: 8,
                stall_rounds: 4,
            },
            Self::High => EffortLimits {
                evaluations: 512,
                wall_seconds: 300,
                rung1_folds: 5,
                promotion_repeats: 1,
                patience: 24,
                stall_rounds: 12,
            },
            Self::Ultra => EffortLimits {
                evaluations: 4_096,
                wall_seconds: 1_800,
                rung1_folds: 5,
                promotion_repeats: 2,
                patience: 64,
                stall_rounds: 32,
            },
        }
    }

    pub fn limits_with_budget(self, budget: BudgetOverrides) -> EffortLimits {
        let defaults = self.limits();
        let patience = match budget.patience {
            Some(value) => value,
            None => defaults.patience,
        };
        EffortLimits {
            evaluations: budget.evaluations.unwrap_or(defaults.evaluations),
            wall_seconds: budget.wall_seconds.unwrap_or(defaults.wall_seconds),
            rung1_folds: budget.rung1_folds.unwrap_or(defaults.rung1_folds),
            promotion_repeats: budget
                .promotion_repeats
                .unwrap_or(defaults.promotion_repeats),
            patience,
            stall_rounds: patience / 2 + patience % 2,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BudgetOverrides {
    pub evaluations: Option<u64>,
    pub wall_seconds: Option<u64>,
    pub rung1_folds: Option<u8>,
    pub promotion_repeats: Option<u8>,
    pub patience: Option<u32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EffortLimits {
    pub evaluations: u64,
    pub wall_seconds: u64,
    pub rung1_folds: u8,
    pub promotion_repeats: u8,
    pub patience: u32,
    pub stall_rounds: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EvaluationBudget {
    limit: u64,
    admitted: u64,
}

impl EvaluationBudget {
    pub const fn new(limit: u64) -> Self {
        Self { limit, admitted: 0 }
    }

    /// Reserves one evaluation before dispatch. A failed evaluation remains admitted.
    pub fn admit(&mut self) -> bool {
        if self.admitted >= self.limit {
            return false;
        }
        self.admitted += 1;
        true
    }

    pub const fn admitted(&self) -> u64 {
        self.admitted
    }

    pub const fn remaining(&self) -> u64 {
        self.limit - self.admitted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effort_table_matches_spec_defaults() {
        let cases = [
            (Effort::Fast, 24, 10, 3, 1, 2, 1),
            (Effort::Medium, 128, 60, 5, 1, 8, 4),
            (Effort::High, 512, 300, 5, 1, 24, 12),
            (Effort::Ultra, 4_096, 1_800, 5, 2, 64, 32),
        ];

        for (effort, evaluations, wall_seconds, folds, repeats, patience, stall_rounds) in cases {
            assert_eq!(
                effort.limits(),
                EffortLimits {
                    evaluations,
                    wall_seconds,
                    rung1_folds: folds,
                    promotion_repeats: repeats,
                    patience,
                    stall_rounds,
                },
                "{effort:?}"
            );
        }
    }

    #[test]
    fn explicit_budget_fields_override_effort_defaults() {
        let limits = Effort::Medium.limits_with_budget(BudgetOverrides {
            evaluations: Some(32),
            wall_seconds: Some(12),
            rung1_folds: Some(3),
            promotion_repeats: Some(2),
            patience: Some(5),
        });

        assert_eq!(limits.evaluations, 32);
        assert_eq!(limits.wall_seconds, 12);
        assert_eq!(limits.rung1_folds, 3);
        assert_eq!(limits.promotion_repeats, 2);
        assert_eq!(limits.patience, 5);
        assert_eq!(limits.stall_rounds, 3);
    }

    #[test]
    fn failed_evaluations_remain_counted_from_admission() {
        let mut budget = EvaluationBudget::new(2);

        assert!(budget.admit());
        let failed_evaluation: Result<(), &str> = Err("evaluation failed");
        assert!(failed_evaluation.is_err());
        assert_eq!(budget.admitted(), 1);
        assert_eq!(budget.remaining(), 1);

        assert!(budget.admit());
        assert!(!budget.admit());
        assert_eq!(budget.admitted(), 2);
        assert_eq!(budget.remaining(), 0);
    }

    #[test]
    fn effort_names_are_exact() {
        for (name, expected) in [
            ("fast", Effort::Fast),
            ("medium", Effort::Medium),
            ("high", Effort::High),
            ("ultra", Effort::Ultra),
        ] {
            assert_eq!(Effort::from_name(name), Some(expected));
        }
        assert_eq!(Effort::from_name("FAST"), None);
        assert_eq!(Effort::from_name("unknown"), None);
    }
}
