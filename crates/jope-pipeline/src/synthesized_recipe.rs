//! Lossless proposal custody and explicit raw-input adaptation for generated DSL.
use anyhow::{ensure, Result};
use feat_math::GenomeIr;
use serde::{Deserialize, Serialize};

pub const RAW_INPUT_POLICY: &str = "asm.generated-raw-input/median-onehot256-indicator/v1";
pub const INPUT_STAGE: &str =
    "(input numeric=median categorical=onehot missing=indicator min_count=1 max_onehot=256)";
/// EvaluationRequest.seed distinguishes scheduling identities. fit_table retains
/// explicit DSL seeds and engine defaults; it has no runtime seed parameter.
pub const FIT_RANDOMNESS_POLICY: &str =
    "dsl-seeds-and-engine-fixed-defaults/request-seed-scheduling-only/v1";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SynthesizedRecipe {
    pub original_proposal: String,
    pub original_bytes_sha256: String,
    pub proposal_canonical: String,
    pub proposal_canonical_sha256: String,
    pub executable_canonical: String,
    pub executable_sha256: String,
    pub input_policy: String,
    pub inserted_input_stage: bool,
    pub randomness_policy: String,
}
impl SynthesizedRecipe {
    pub fn new(original_proposal: String) -> Result<Self> {
        let genome = GenomeIr::parse(&original_proposal)?;
        let proposal_canonical = genome.canonical();
        let GenomeIr::Pipeline(mut pipeline) = genome else {
            anyhow::bail!("generated raw-table worker requires an executable typed pipeline; symbolic/portfolio forms are unsupported")
        };
        let inserted_input_stage = !pipeline.stages.iter().any(|s| s.op == "input");
        if inserted_input_stage {
            let GenomeIr::Pipeline(input) = GenomeIr::parse(&format!("(pipe X {INPUT_STAGE})"))?
            else {
                unreachable!()
            };
            let position = usize::from(
                pipeline
                    .stages
                    .first()
                    .is_some_and(|stage| feat_math::target_transform::is_target_op(&stage.op)),
            );
            pipeline.stages.insert(position, input.stages[0].clone());
        }
        let executable_canonical = GenomeIr::Pipeline(pipeline).canonical();
        Ok(Self {
            original_bytes_sha256: crate::sha256_hex(original_proposal.as_bytes()),
            original_proposal,
            proposal_canonical_sha256: crate::sha256_hex(proposal_canonical.as_bytes()),
            proposal_canonical,
            executable_sha256: crate::sha256_hex(executable_canonical.as_bytes()),
            executable_canonical,
            input_policy: RAW_INPUT_POLICY.into(),
            inserted_input_stage,
            randomness_policy: FIT_RANDOMNESS_POLICY.into(),
        })
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            *self == Self::new(self.original_proposal.clone())?,
            "generated recipe or explicit preprocessing derivation changed"
        );
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn lossless_original_and_explicit_input_derivation() -> Result<()> {
        let source = " (pipe X (ridge lambda=2)) ".to_owned();
        let recipe = SynthesizedRecipe::new(source.clone())?;
        assert_eq!(recipe.original_proposal, source);
        assert!(recipe.inserted_input_stage);
        assert_ne!(recipe.proposal_canonical_sha256, recipe.executable_sha256);
        let explicit = SynthesizedRecipe::new(recipe.executable_canonical.clone())?;
        assert!(!explicit.inserted_input_stage);
        assert_eq!(explicit.executable_sha256, recipe.executable_sha256);
        let mut changed = recipe;
        changed.executable_canonical.push(' ');
        assert!(changed.validate().is_err());
        Ok(())
    }
}
