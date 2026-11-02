use super::*;

pub(super) mod support {
    use super::*;
    use std::cell::RefCell;

    pub(super) fn parents() -> Vec<Program> {
        vec![
            Program::from_dsl("(pipe X (standardize) (cheb degree=3) (ridge lambda_frac=0.1))"),
            Program::from_dsl(
                "(pipe X (robust_scale) (winsor_clip q=0.05) (histgbm depth=3 lr=0.08 rounds=160))",
            ),
        ]
    }

    pub(super) fn request<'a>(
        parents: &'a [Program],
        lm: &'a [f64; 48],
        exploring: bool,
    ) -> Round<'a> {
        Round {
            seed: 42,
            round: 7,
            exploring,
            target_family: "target",
            raw_lm48: lm,
            parents,
        }
    }

    pub(super) struct Bank {
        pub(super) role: BankRole,
        pub(super) records: Vec<Retrieved>,
        pub(super) called: RefCell<Vec<([f64; 12], String, usize)>>,
    }

    impl Bank {
        pub(super) fn new() -> Self {
            Self {
                role: BankRole::Proposer,
                records: vec![
                    Retrieved {
                        program: Program::from_dsl("(pipe X (quantile_bins bins=16) (ridge))"),
                        family_id: "other-a".into(),
                        dl_train: true,
                        distance_squared: 1.0,
                    },
                    Retrieved {
                        program: Program::from_dsl("(pipe X (winsor_clip q=0.1) (ridge))"),
                        family_id: "other-b".into(),
                        dl_train: true,
                        distance_squared: 2.0,
                    },
                ],
                called: RefCell::new(vec![]),
            }
        }
    }

    impl ChampionBank for Bank {
        fn contract(&self) -> Option<BankContract> {
            Some(BankContract {
                role: self.role,
                coordinate_space: RAW_COORDINATES.into(),
                snapshot_sha256: "1".repeat(64),
            })
        }
        fn nearest(&self, raw: &[f64; 12], family: &str, limit: usize) -> Result<Vec<Retrieved>> {
            self.called.borrow_mut().push((*raw, family.into(), limit));
            Ok(self.records.iter().take(limit).cloned().collect())
        }
    }
}

mod frequency;
mod generation;
mod head_edits;
mod operators;
mod replay;
mod rng;
