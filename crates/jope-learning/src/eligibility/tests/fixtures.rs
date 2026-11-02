use super::*;
use crate::{
    encoder_receipt::{Encoder, EncoderCapability, Inputs, Member, Product, Scope},
    hwm::{self, Candidate},
    record::{ComparisonContext, Cost, Evaluator, Genome, Provenance, Score, HWM_SCHEMA},
};
use jope_pipeline::evaluation_result::{Consumption, RESULT_SCHEMA};
use jope_pipeline::execution::{Artifact, ObservedConsumption, WorkerCommand, EXECUTION_SCHEMA};
use jope_pipeline::experiment::REQUEST_SCHEMA;
use serde_json::json;
use std::path::PathBuf;

pub(super) struct Fixture {
    pub(super) _directory: tempfile::TempDir,
    pub(super) state: State,
    pub(super) plan: FreezePlan,
    pub(super) record: HwmRecord,
    pub(super) evidence: OofEvidence,
    pub(super) rows: Vec<TargetRow>,
    pub(super) task: Task,
    pub(super) classes: Vec<String>,
    pub(super) experiment: Experiment,
    pub(super) window: ExecutionWindow,
    pub(super) created_at: u64,
}

pub(super) type Predict<'a> = &'a dyn Fn(u8, &[&TargetRow], Predictions) -> Predictions;

impl Fixture {
    /// Another candidate for the same frozen comparison, context and execution window.
    pub(super) fn candidate(&self, tag: &str, predict: Predict) -> (HwmRecord, OofEvidence) {
        let mut base = self.evidence.clone();
        base.fold_predictions_sha256.clear();
        candidate_in(
            &self.state,
            &self.plan.comparison,
            &self.rows,
            self.task,
            &self.classes,
            &self.experiment,
            &self.window,
            self.created_at,
            base,
            tag,
            predict,
        )
    }
}

/// Predictions exact on the first `improved` folds, the train-only baseline elsewhere.
fn exact_on(improved: usize) -> impl Fn(u8, &[&TargetRow], Predictions) -> Predictions {
    move |fold: u8, hold: &[&TargetRow], baseline: Predictions| {
        if (fold as usize) >= improved {
            return baseline;
        }
        match baseline {
            Predictions::Values(_) => {
                Predictions::Values(hold.iter().map(|row| row.target.parse().unwrap()).collect())
            }
            Predictions::Probabilities { classes, .. } => Predictions::Probabilities {
                values: hold
                    .iter()
                    .map(|row| {
                        classes
                            .iter()
                            .map(|class| f64::from(class == &row.target))
                            .collect()
                    })
                    .collect(),
                classes,
            },
        }
    }
}

fn context_plan() -> ContextPlan {
    ContextPlan {
        source_family: "synthetic-fixture-family".into(),
        encoder: Encoder {
            artifact_sha256: sha256_hex(b"fixture context encoder"),
            configuration_sha256: sha256_hex(b"fixture context configuration"),
            seed: 7,
            product: Product::TeachingContext2048,
        },
        producer: Some(producer_plan()),
    }
}

/// The synthetic W2 producer executable's bytes: never executed, only pinned.
const FIXTURE_PRODUCER: &[u8] = b"synthetic W2 context producer executable; never executed";

/// The W2 producer the fixture controller preregisters with every freeze.
pub(super) fn producer_plan() -> ProducerPlan {
    let digest = |name: &str| sha256_hex(format!("fixture {name}").as_bytes());
    ProducerPlan {
        release: HandoffRelease {
            encoder: HandoffEncoder {
                hyperion_weights: digest("hyperion_weights"),
                hyperion_manifest: digest("hyperion_manifest"),
                component_bundle: digest("component_bundle"),
                foundation_weights: digest("foundation_weights"),
                target_weights: digest("target_weights"),
                frozen_transforms: digest("frozen_transforms"),
                assembly: digest("assembly"),
                recipe: digest("recipe"),
            },
            source_sha256: "b".repeat(40),
            producer_sha256: sha256_hex(FIXTURE_PRODUCER),
            verification_policy_sha256: digest("verification policy"),
            reference_manifest_sha256: digest("reference manifest"),
        },
        transformations: [
            "fixture.row-embeddings/v1",
            "fixture.signature/v1",
            "fixture.foundation-raw/v1",
            "fixture.target/v1",
            "fixture.foundation-projected/v1",
            "fixture.final-context/v1",
        ]
        .map(String::from)
        .to_vec(),
        qualification_sha256: digest("W2 producer qualification"),
        qualification_execution_sha256: Some(fixture_case_execution_pins()),
    }
}

/// The controller's execution attestation of each W2 qualification case, as the fixture
/// controller pins them (`AdmissionPins::qualification_execution_sha256`), in manifest order.
pub(super) fn fixture_case_execution_pins() -> Vec<String> {
    (0..8)
        .map(|case| sha256_hex(format!("fixture case execution {case}").as_bytes()))
        .collect()
}

/// One W2 context handoff to teacher admission and its execution receipts, before they are
/// retained, so a test can tamper with any part.
pub(super) struct HandoffParts {
    pub(super) handoff: serde_json::Value,
    pub(super) job: WorkerJob,
    pub(super) outcome: WorkerOutcome,
    /// The bytes of the context artifact the job wrote.
    pub(super) artifact: Vec<u8>,
    /// The W2 qualification's case-execution pins the controller issued this admission under.
    pub(super) case_execution_pins: Vec<String>,
    pub(super) experiment: Experiment,
}

/// `jope_pipeline::cpu_context_bridge::execution_attestation`'s record, field for field in
/// its serialization order, written out here independently of the receiver.
#[derive(Serialize)]
struct FixtureAttestation {
    schema: &'static str,
    supervisor_receipt_sha256: String,
    program_sha256: String,
    input_sha256: String,
    output_sha256: String,
    completed: bool,
    cpu_limit: u32,
    memory_max: u64,
    swap_max: u64,
    threads: u32,
    elapsed_seconds: f64,
    deadline_seconds: f64,
}

/// `jope_policy::context_producer::TrainingIdentity`, in its serialization order.
#[derive(Serialize, Deserialize)]
struct FixtureTraining {
    dataset_sha256: String,
    prepared_sha256: String,
    training_sha256: String,
    fitting_rows_sha256: String,
    cut: u8,
}

/// `jope_policy::context_producer::AdmissionPins`, in its serialization order.
#[derive(Serialize)]
struct FixturePins {
    source_sha256: String,
    producer_sha256: String,
    artifact_sha256: String,
    qualification_sha256: String,
    reference_manifest_sha256: String,
    context_execution_sha256: String,
    qualification_execution_sha256: Vec<String>,
    verification_policy_sha256: String,
    training: FixtureTraining,
    input_sha256: String,
    preprocessing_sha256: String,
}
