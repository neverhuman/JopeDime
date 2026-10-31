//! Pre-stage input contract for the h40b `adapter_10tok.predict_costs` ranker.
//!
//! RES-09 has supplied shared SafeTensors/Candle loader support, but the RES-07
//! real-tensor fixtures needed for parity are still pending. This pre-stage
//! validates only F12 flattened input dimensions; it does not load adapter
//! weights, produce costs, or establish M2 parity.

use std::fmt;

pub const PROG_FEATURES: usize = 412;
pub const PROJ_FEATURES: usize = 16;
pub const TOKENS: usize = 10;
pub const TOKEN_WIDTH: usize = 32;
pub const CONTEXT_FEATURES: usize = 584;
pub const REVEALED: usize = 4;
pub const SOS_FEATURES: usize = 2;
pub const SCALE: f32 = 0.4208;

/// Flattened F12 inputs for one batched ranker call.
///
/// Candidate arrays are row-major `[batch, candidates, feature]`; token arrays
/// are `[batch, candidates, 10, 32]` and masks are `[batch, candidates, 10]`.
/// Context and prior inputs are the zero-valued constants specified by F12.
#[derive(Clone, Copy, Debug)]
pub struct PredictCostsInputs<'a> {
    pub batch: usize,
    pub candidates: usize,
    pub prog_412: &'a [f32],
    pub proj_16: &'a [f32],
    pub tokens: &'a [f32],
    pub token_mask: &'a [f32],
    pub revealed_indices: &'a [i64],
    pub revealed_fitness: &'a [f32],
    pub revealed_gates: &'a [f32],
    pub ctx_feat: &'a [f32],
    pub base_prior_cost: &'a [f32],
    pub sos_prior: &'a [f32],
    pub scale: f32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InputShapeError {
    pub field: &'static str,
    pub expected: usize,
    pub actual: usize,
}

impl fmt::Display for InputShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "adapter input {} has {} elements; expected {}",
            self.field, self.actual, self.expected
        )
    }
}

impl std::error::Error for InputShapeError {}

impl PredictCostsInputs<'_> {
    /// Check the F12 flattened dimensions before converting them to backend tensors.
    pub fn validate_shapes(&self) -> Result<(), InputShapeError> {
        let bn = self.batch.saturating_mul(self.candidates);
        let expected = [
            (
                "prog_412",
                bn.saturating_mul(PROG_FEATURES),
                self.prog_412.len(),
            ),
            (
                "proj_16",
                bn.saturating_mul(PROJ_FEATURES),
                self.proj_16.len(),
            ),
            (
                "tokens",
                bn.saturating_mul(TOKENS * TOKEN_WIDTH),
                self.tokens.len(),
            ),
            (
                "token_mask",
                bn.saturating_mul(TOKENS),
                self.token_mask.len(),
            ),
            (
                "revealed_indices",
                self.batch.saturating_mul(REVEALED),
                self.revealed_indices.len(),
            ),
            (
                "revealed_fitness",
                self.batch.saturating_mul(REVEALED),
                self.revealed_fitness.len(),
            ),
            (
                "revealed_gates",
                self.batch.saturating_mul(REVEALED),
                self.revealed_gates.len(),
            ),
            (
                "ctx_feat",
                self.batch.saturating_mul(CONTEXT_FEATURES),
                self.ctx_feat.len(),
            ),
            ("base_prior_cost", bn, self.base_prior_cost.len()),
            (
                "sos_prior",
                bn.saturating_mul(SOS_FEATURES),
                self.sos_prior.len(),
            ),
        ];

        for (field, expected, actual) in expected {
            if expected != actual {
                return Err(InputShapeError {
                    field,
                    expected,
                    actual,
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixture {
        prog_412: Vec<f32>,
        proj_16: Vec<f32>,
        tokens: Vec<f32>,
        token_mask: Vec<f32>,
        revealed_indices: Vec<i64>,
        revealed_fitness: Vec<f32>,
        revealed_gates: Vec<f32>,
        ctx_feat: Vec<f32>,
        base_prior_cost: Vec<f32>,
        sos_prior: Vec<f32>,
    }

    impl Fixture {
        fn serving(batch: usize, proposals: usize) -> Self {
            let candidates = REVEALED + proposals;
            let bn = batch * candidates;
            Self {
                prog_412: vec![0.0; bn * PROG_FEATURES],
                proj_16: vec![0.0; bn * PROJ_FEATURES],
                tokens: vec![0.0; bn * TOKENS * TOKEN_WIDTH],
                token_mask: vec![0.0; bn * TOKENS],
                revealed_indices: (0..batch).flat_map(|_| 0..REVEALED as i64).collect(),
                revealed_fitness: vec![0.0; batch * REVEALED],
                revealed_gates: vec![0.0; batch * REVEALED],
                ctx_feat: vec![0.0; batch * CONTEXT_FEATURES],
                base_prior_cost: vec![0.0; bn],
                sos_prior: vec![0.0; bn * SOS_FEATURES],
            }
        }

        fn inputs(&self, batch: usize, proposals: usize) -> PredictCostsInputs<'_> {
            PredictCostsInputs {
                batch,
                candidates: REVEALED + proposals,
                prog_412: &self.prog_412,
                proj_16: &self.proj_16,
                tokens: &self.tokens,
                token_mask: &self.token_mask,
                revealed_indices: &self.revealed_indices,
                revealed_fitness: &self.revealed_fitness,
                revealed_gates: &self.revealed_gates,
                ctx_feat: &self.ctx_feat,
                base_prior_cost: &self.base_prior_cost,
                sos_prior: &self.sos_prior,
                scale: SCALE,
            }
        }
    }

    #[test]
    fn accepts_both_serving_slate_shapes() {
        for proposals in [8, 16] {
            let fixture = Fixture::serving(2, proposals);
            assert_eq!(fixture.inputs(2, proposals).validate_shapes(), Ok(()));
        }
    }

    #[test]
    fn reports_a_mismatch_for_every_flattened_f12_input_shape() {
        for proposals in [8, 16] {
            let fixture = Fixture::serving(2, proposals);
            let valid = fixture.inputs(2, proposals);

            macro_rules! assert_one_short {
                ($field:ident, $name:literal) => {{
                    let actual = fixture.$field.len() - 1;
                    let malformed = PredictCostsInputs {
                        $field: &fixture.$field[..actual],
                        ..valid
                    };
                    let error = malformed.validate_shapes().unwrap_err();
                    assert_eq!(error.field, $name);
                    assert_eq!(error.expected, fixture.$field.len());
                    assert_eq!(error.actual, actual);
                }};
            }

            assert_one_short!(prog_412, "prog_412");
            assert_one_short!(proj_16, "proj_16");
            assert_one_short!(tokens, "tokens");
            assert_one_short!(token_mask, "token_mask");
            assert_one_short!(revealed_indices, "revealed_indices");
            assert_one_short!(revealed_fitness, "revealed_fitness");
            assert_one_short!(revealed_gates, "revealed_gates");
            assert_one_short!(ctx_feat, "ctx_feat");
            assert_one_short!(base_prior_cost, "base_prior_cost");
            assert_one_short!(sos_prior, "sos_prior");
        }
    }
}
