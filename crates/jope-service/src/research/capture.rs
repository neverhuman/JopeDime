//! Private durable-capture inputs. No qualified production provider is registered.

use super::*;
use crate::{
    discover::r#loop::Plan,
    ranker::admission::{select_revealed, MeasuredAlgorithm, RevealedRule},
    store::{BlobRef, Store},
    worker::{Request, Response},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::Cursor,
    sync::Arc,
};

pub(crate) const CHECKPOINT_VERSION: u32 = 3;
const ENVELOPE_VERSION: u32 = 1;
const MAX_CAPTURE_BYTES: u64 = 256 * 1024 * 1024;

// Own-child process controls only; no production environment switch exists.
#[cfg(test)]
pub(crate) fn boundary(name: &str) {
    if std::env::var("QLT03_CAPTURE_STOP").as_deref() == Ok(name) {
        println!("QLT03_CAPTURE_BOUNDARY:{name}");
        std::io::stdout().flush().unwrap();
        loop {
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    }
}

/// Explicit storage/attempt ceilings, not a campaign-compute authorization.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureConfig {
    pub max_attempts: u64,
    pub max_rounds: usize,
    pub max_bytes: u64,
    pub evaluate_all_proposals: bool,
    pub seed_protocol_sha256: String,
}

impl CaptureConfig {
    fn validate(&self, budget: u64) -> Result<()> {
        ensure!(
            self.max_attempts > 0 && self.max_attempts <= budget,
            "invalid capture attempt bound"
        );
        ensure!(
            self.max_rounds > 0
                && self.max_rounds <= 4096
                && self.max_rounds as u64 <= self.max_attempts,
            "invalid capture round bound"
        );
        ensure!(
            self.max_bytes > 0 && self.max_bytes <= MAX_CAPTURE_BYTES,
            "invalid capture byte bound"
        );
        ensure!(
            hex(&self.seed_protocol_sha256, 64),
            "evaluation seed protocol is not pinned"
        );
        Ok(())
    }
}

/// Actual aligned program inputs and indices into the measured admitted pool.
pub struct SlateObservation {
    pub programs: Vec<RecordedProgram>,
    pub revealed_indices: Vec<usize>,
}

/// A same-invocation admission input; values are not manufactured by the loop.
pub struct ObservedSlate {
    pub programs: Vec<RecordedProgram>,
    pub revealed: Vec<RecordedReveal>,
}

pub(crate) trait CaptureProvider: Send + Sync {
    /// Engine observations require the authenticated native input adapter.
    /// Synthetic controls keep their existing explicitly tagged input path.
    fn native_inputs(&self) -> Option<&NativeCaptureInputs> {
        None
    }

    /// Refuse unavailable corpus, conditioning, evaluator, gate or seed inputs
    /// before any reservation or execution. This is not an identity-only check.
    fn validate_binding(&self, binding: &Binding, plan: &Plan) -> Result<()>;
    /// Prepare the actual request before it is journaled or dispatched.
    fn prepare_request(&self, _request: &mut Request) -> Result<()> {
        Ok(())
    }
    fn observe(
        &self,
        round: u64,
        dsls: &[String],
        measured: &[RecordedReveal],
    ) -> Result<SlateObservation>;
    fn measurement(&self, request: &Request, response: &Response) -> Result<Rung0Measurement>;
    fn label_seed(&self, round: u64, proposal_index: usize) -> Result<u64>;
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Binding {
    pub config: CaptureConfig,
    pub identity: RecordIdentity,
    pub seed: u64,
    pub tokens: Vec<Vec<f32>>,
    pub mask: Vec<bool>,
    pub matrix_sha256: String,
}

/// Opaque capability for a reviewed input/evaluator closure.
///
/// There is intentionally no public constructor or JSON importer. Production
/// resolution awaits the separately reviewed corpus/R0/gate/seed contract.
/// Capture execution controls use SyntheticControl bindings. Isolated input
/// boundary tests never export an Engine record.
pub struct CaptureContext {
    pub(crate) binding: Binding,
    pub(crate) provider: Arc<dyn CaptureProvider>,
}

impl CaptureContext {
    /// Create an Engine capability only from the trusted service provider.
    /// There is intentionally no serialized or caller-selectable constructor.
    #[allow(dead_code)] // Used by the qualified provider and trusted job loader.
    pub(crate) fn engine(
        binding: Binding,
        provider: Arc<dyn CaptureProvider>,
        plan: &Plan,
    ) -> Result<Self> {
        ensure!(
            binding.identity.evidence == RecordEvidence::Engine,
            "Engine context requires Engine evidence"
        );
        let context = Self { binding, provider };
        context.validate(plan)?;
        Ok(context)
    }

    #[cfg(test)]
    pub(crate) fn control(binding: Binding, provider: Arc<dyn CaptureProvider>) -> Result<Self> {
        ensure!(
            binding.identity.evidence == RecordEvidence::SyntheticControl,
            "control cannot claim Engine evidence"
        );
        Ok(Self { binding, provider })
    }

    pub(crate) fn validate(&self, plan: &Plan) -> Result<()> {
        self.binding.config.validate(plan.evaluations)?;
        let id = &self.binding.identity;
        ensure!(
            id.partition == "train"
                && hex(&id.source_commit, 40)
                && id.dope_weights_sha256 == DOPE_SHA256
                && [
                    &id.corpus_manifest_sha256,
                    &id.dataset_sha256,
                    &id.family_sha256,
                    &id.problem_sha256,
                    &id.bundle_sha256,
                    &id.conditioning_sha256,
                    &id.evaluator_sha256
                ]
                .iter()
                .all(|value| hex(value, 64)),
            "invalid capture input identities"
        );
        ensure!(
            self.binding.tokens.len() == 10
                && self.binding.mask.len() == 10
                && self
                    .binding
                    .tokens
                    .iter()
                    .all(|row| row.len() == 32 && row.iter().all(|v| v.is_finite())),
            "capture conditioning unavailable"
        );
        ensure!(
            self.binding.identity.run_id == plan.run_id.to_string()
                && self.binding.seed == plan.seed
                && self.binding.matrix_sha256 == plan.matrix_sha256
                && self.binding.identity.problem_sha256 == plan.problem_digest
                && self.binding.identity.bundle_sha256 == plan.bundle_digest,
            "capture/plan identity mismatch"
        );
        ensure!(
            self.binding.identity.task == RecordTask::Regression,
            "binary capture requires its qualified evaluator adapter"
        );
        if id.evidence == RecordEvidence::Engine {
            self.provider
                .native_inputs()
                .context("native research input closure unavailable")?
                .validate_binding(&self.binding, plan)?;
        }
        self.provider.validate_binding(&self.binding, plan)?;
        Ok(())
    }

    pub(crate) fn observe(
        &self,
        round: u64,
        dsls: &[String],
        measured: &[RecordedReveal],
    ) -> Result<ObservedSlate> {
        let native = if self.binding.identity.evidence == RecordEvidence::Engine {
            Some(
                self.provider
                    .native_inputs()
                    .context("native research input closure unavailable")?,
            )
        } else {
            None
        };
        if let Some(native) = native {
            native.validate_record_binding(&self.binding)?;
        }
        let observation = self.provider.observe(round, dsls, measured)?;
        if let Some(native) = native {
            native.validate_observation(&self.binding, dsls, &observation.programs)?;
        }
        ensure!(
            observation.programs.len() == dsls.len(),
            "program feature inventory mismatch"
        );
        for (program, dsl) in observation.programs.iter().zip(dsls) {
            validate_program(program)?;
            ensure!(
                program.program_sha256 == digest(dsl.as_bytes()),
                "program provenance mismatch"
            );
        }
        let revealed_indices = if self.binding.identity.evidence == RecordEvidence::Engine {
            validate_od13_provider_indices(round, measured, &observation.revealed_indices)?
        } else {
            ensure!(
                observation.revealed_indices.len() == if round == 0 { 0 } else { 4 },
                "missing actual reveal context"
            );
            let mut seen = BTreeSet::new();
            for index in &observation.revealed_indices {
                ensure!(seen.insert(*index), "duplicate revealed index");
                ensure!(measured.get(*index).is_some(), "unknown revealed index");
            }
            observation.revealed_indices
        };
        let revealed = revealed_indices
            .into_iter()
            .map(|index| measured[index].clone())
            .collect();
        Ok(ObservedSlate {
            programs: observation.programs,
            revealed,
        })
    }
}

/// Apply the owner-accepted OD13 rule to the actual measured admitted pool.
/// The ranker selector keeps input order on equal F values.
pub(crate) fn select_od13_indices(round: u64, measured: &[RecordedReveal]) -> Result<Vec<usize>> {
    if round == 0 {
        return Ok(Vec::new());
    }
    let candidates: Vec<_> = measured
        .iter()
        .map(|entry| MeasuredAlgorithm {
            id: &entry.program.program_sha256,
            surrogate_fitness: entry.surrogate,
            gate: entry.gate,
        })
        .collect();
    let selected = select_revealed(&candidates, RevealedRule::SurrogateFitness)
        .context("OD13 could not select four measured reveals")?;
    Ok(selected.into_iter().collect())
}

fn validate_od13_provider_indices(
    round: u64,
    measured: &[RecordedReveal],
    provider_indices: &[usize],
) -> Result<Vec<usize>> {
    let expected = select_od13_indices(round, measured)?;
    ensure!(
        provider_indices == expected,
        "provider reveal selection differs from accepted OD13 rule"
    );
    Ok(expected)
}

pub(crate) fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod od13_tests {
    use super::*;

    fn reveal(index: usize, surrogate: f64) -> RecordedReveal {
        RecordedReveal {
            program: RecordedProgram {
                program_sha256: format!("{index:064x}"),
                operator_family: "synthetic_control".into(),
                feat412: vec![0.0; 412],
                proj16: vec![0.0; 16],
            },
            surrogate,
            gate: index & 1 == 0,
        }
    }

    #[test]
    fn od13_selects_four_lowest_f_with_stable_source_order_ties() {
        let measured = [
            reveal(0, 2.0),
            reveal(1, -0.0),
            reveal(2, 1.0),
            reveal(3, 0.0),
            reveal(4, -1.0),
            reveal(5, 3.0),
        ];

        let selected = select_od13_indices(1, &measured).unwrap();

        assert_eq!(selected, [4, 1, 3, 2]);
        assert_eq!(
            validate_od13_provider_indices(1, &measured, &[4, 1, 3, 2]).unwrap(),
            selected
        );
        assert!(validate_od13_provider_indices(1, &measured, &[0, 1, 2, 3]).is_err());
        assert_eq!(measured[selected[0]].gate, measured[4].gate);
        assert_eq!(measured[selected[1]].gate, measured[1].gate);
    }

    #[test]
    fn od13_refuses_short_or_nonfinite_measured_pools_and_round_zero_reveals() {
        let short = (0..3).map(|index| reveal(index, 1.0)).collect::<Vec<_>>();
        assert!(select_od13_indices(1, &short).is_err());
        let mut nonfinite = vec![reveal(0, 1.0); 4];
        nonfinite[2].surrogate = f64::NAN;
        assert!(select_od13_indices(1, &nonfinite).is_err());
        assert!(select_od13_indices(0, &[reveal(0, 1.0)])
            .unwrap()
            .is_empty());
        assert!(validate_od13_provider_indices(0, &[reveal(0, 1.0)], &[0]).is_err());
        assert!(select_od13_indices(0, &[]).unwrap().is_empty());
    }
}

mod state;
pub(crate) use state::{
    AuxiliaryAttempt, AuxiliaryResult, CaptureState, CapturedAttempt, CapturedRound, Measured,
    Slate,
};
