//! Bounded paired search on actual frozen training rows. This controller never
//! receives sealed data and never passes development scoring labels to workers.
use super::{
    campaign::{CampaignPlan, Runtime},
    data::DevelopmentData,
    optimize_fit::{self, artifact, digest, immutable, immutable_bytes, FitAdapter},
};
use anyhow::{ensure, Context, Result};
use jope_datasets::catalog::Task;
use jope_pipeline::{
    context_qualification::{CompletedQualification, EncoderAssets},
    cpu_context_bridge::{CpuContextFiles, ExecutionProof},
    execution::{Artifact, WorkerCommand, WorkerJob, WorkerOutcome},
    experiment::{ColumnKind, EvaluationRequest, EvaluationStage, EvaluationStatus, Experiment},
    neural_search::{
        self, Backend, Comparison, ContextCost, EncodingPlan, InstallationPins, PairInputPlan,
        PolicyWorker, PoolAccess, PoolBackend, Variation,
    },
    search_context::{ContextFiles, TrainingIdentity},
    synthesized_recipe::{FIT_RANDOMNESS_POLICY, RAW_INPUT_POLICY},
};
use jope_policy::{
    context_producer::{self, TrainingInput, TrainingRow},
    producer::{QualifiedContextBundle, RequestPlan},
};
use serde::{Deserialize, Serialize};
use std::{fs, path::Path, sync::atomic::AtomicBool};

mod configuration;
pub use configuration::{
    classification_generic, generic_policy, variation_policy, Configuration, Incumbents,
    NativeChallenger, CLASSIFICATION_GENERIC_POLICY, CLASSIFICATION_VARIATION, CONFIG_RUNTIME,
    GENERIC_METHOD, GENERIC_POLICY, MAX_SEARCH_ROUND, NEURAL_METHOD, SCHEMA, VARIATION,
};
pub(crate) use configuration::{TaskVariation, TEACHER_HANDOFF};

mod execution;
mod pair;
mod prepare;

pub use execution::search_dataset;
pub(crate) use execution::{diagnostic, pending, proof};
pub use pair::{build_pair_input, DatasetSearch};
pub(crate) use pair::{
    challenger_checkpoint, incumbent_seed, pair_seed, pinned_checkpoint, proposal_files,
};
pub(crate) use prepare::prepare_regression_context;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn distinct_classification_sampler_retains_typed_heads_and_eighty_unique_slots() -> Result<()> {
        use feat_math::advanced_v3::{validate_v3_genome, ColumnSpec, DataSchema, RowSemantics};
        let schema = DataSchema {
            columns: vec![ColumnSpec::numeric("x")],
            row_semantics: RowSemantics::Iid,
        };
        for task in [Task::Binary, Task::Multiclass] {
            let mut seen = std::collections::BTreeSet::new();
            for seed in 0..80 {
                let source = classification_generic(task, seed)?;
                let recipe = jope_pipeline::synthesized_recipe::SynthesizedRecipe::new(source)?;
                validate_v3_genome(
                    &feat_math::GenomeIr::parse(&recipe.executable_canonical)?,
                    &super::super::methods::task_spec(task)?,
                    &schema,
                )?;
                seen.insert(recipe.executable_sha256);
            }
            assert_eq!(seen.len(), 80);
            assert_ne!(generic_policy(task), GENERIC_POLICY);
        }
        Ok(())
    }
}
