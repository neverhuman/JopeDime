//! Typed opt-in scalar births reuse the unchanged base expression generator.
use super::*;
use crate::kernel::numerical_policy::NumericalPolicy;
use crate::kernel::scalar_program::{ScalarExpressionProgram, ScalarFunction, ScalarNode};
use crate::math_expr::{MathExpr, Var};
use crate::rng::Pcg32;

#[derive(Clone, Debug)]
pub struct ScalarCandidate {
    program: ScalarExpressionProgram,
    profile: ScalarProducerProfile,
    seed: u64,
    opt_in_drawn: bool,
    opt_in_admitted: bool,
    proposal_rejected: bool,
}

impl ScalarProducerProfile {
    /// Imported trees enter the same strict admission as grown/edited trees.
    /// This path draws no randomness and enables no unavailable token.
    pub fn admit_candidate(
        &self,
        program: ScalarExpressionProgram,
        seed: u64,
    ) -> Result<ScalarCandidate, String> {
        self.admit(&program)?;
        Ok(ScalarCandidate {
            program,
            profile: self.clone(),
            seed,
            opt_in_drawn: false,
            opt_in_admitted: false,
            proposal_rejected: false,
        })
    }

    fn function_enabled(&self, id: u16) -> bool {
        self.declaration.enable_functions
            && match id {
                13 => self.declaration.enable_numerical_policy,
                19 | 20 => self.declaration.enable_inverse_links,
                21 | 22 => self.declaration.enable_stable_exponentials,
                _ => false,
            }
    }

    pub fn grow(&self, seed: u64) -> Result<ScalarCandidate, String> {
        // The old body is drawn at identical scope/seed in both arms. New
        // admission never retries or changes the existing generator's stream.
        let body = MathExpr::grow_v2(&self.base, self.scope(), seed).map_err(|e| e.to_string())?;
        let parent = ScalarExpressionProgram::from_v1(&body).map_err(|e| e.to_string())?;
        let mut rng = Pcg32::new(seed, BIRTH_STREAM);
        let drawn = rng.next_u32().is_multiple_of(4);
        let mut program = parent.clone();
        let mut admitted = false;
        let mut rejected = false;
        if drawn {
            // Disabled choices retain their slots and reject without redraw.
            // Earlier families therefore keep their exact draw probabilities
            // in component-isolated ablations.
            let mut choices: Vec<_> = self
                .declaration
                .function_catalog
                .iter()
                .copied()
                .map(Some)
                .collect();
            if !self.declaration.policy_birth_choices.is_empty() {
                choices.push(None);
            }
            if !choices.is_empty() {
                let choice = choices[rng.next_u32() as usize % choices.len()];
                let proposal = if let Some(id) = choice {
                    ScalarExpressionProgram::new(ScalarNode::Call(
                        ScalarFunction::from_versioned_id(2, id).map_err(|e| e.to_string())?,
                        vec![parent.ast().clone()],
                    ))
                    .map_err(|e| e.to_string())?
                    .with_numerical_policy(if id == 13 {
                        NumericalPolicy::StableV2
                    } else {
                        NumericalPolicy::LegacyV1
                    })
                } else {
                    let choice = rng.next_u32() as usize % 5;
                    if choice == 0 {
                        parent
                            .clone()
                            .with_numerical_policy(NumericalPolicy::StableV2)
                    } else {
                        let var = [Var::RowSum, Var::RowMean, Var::RowL1, Var::RowL2][choice - 1];
                        ScalarExpressionProgram::new(ScalarNode::Var(var))
                            .map_err(|e| e.to_string())?
                            .with_numerical_policy(NumericalPolicy::StableV2)
                    }
                };
                let enabled = choice.map_or(self.declaration.enable_numerical_policy, |id| {
                    self.function_enabled(id)
                });
                if enabled {
                    if self.admit(&proposal).is_ok() {
                        program = proposal;
                        admitted = true;
                    } else {
                        rejected = true;
                    }
                }
            }
        }
        self.admit(&program)?;
        Ok(ScalarCandidate {
            program,
            profile: self.clone(),
            seed,
            opt_in_drawn: drawn,
            opt_in_admitted: admitted,
            proposal_rejected: rejected,
        })
    }
}

impl ScalarCandidate {
    pub fn program(&self) -> &ScalarExpressionProgram {
        &self.program
    }
    pub fn profile(&self) -> &ScalarProducerProfile {
        &self.profile
    }
    pub fn seed(&self) -> u64 {
        self.seed
    }
    pub fn birth_draw(&self) -> (bool, bool) {
        (self.opt_in_drawn, self.opt_in_admitted)
    }
    pub fn proposal_rejected(&self) -> bool {
        self.proposal_rejected
    }
}

#[path = "generation_eec.rs"]
mod eec;
pub use eec::ScalarEdit;
