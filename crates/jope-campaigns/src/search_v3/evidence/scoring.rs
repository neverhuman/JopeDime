use super::*;

mod models;
mod prediction;
mod records;

pub use models::*;
pub use prediction::*;
pub use records::*;

fn invalid_fitness(
    task: SearchTaskV1,
    metric: &str,
    candidate_semantic_sha256: &str,
    complexity: u64,
    reason: String,
) -> FitnessVectorV1 {
    FitnessVectorV1 {
        schema: FitnessVectorV1::SCHEMA.into(),
        evidence_valid: false,
        invalid_reason: Some(reason),
        constraints_satisfied: false,
        primary: TypedMetricValueV1 {
            task,
            metric: metric.into(),
            value: 0.0,
        },
        stability_guards: Vec::new(),
        complexity,
        candidate_semantic_sha256: candidate_semantic_sha256.into(),
    }
}
