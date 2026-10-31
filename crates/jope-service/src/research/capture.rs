//! Private durable-capture inputs. No qualified production provider is registered.

use super::*;
use crate::{
    discover::r#loop::Plan,
    store::{BlobRef, Store},
    worker::{Request, Response},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, io::Cursor, sync::Arc};

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
        ensure!(
            observation.revealed_indices.len() == if round == 0 { 0 } else { 4 },
            "missing actual reveal context"
        );
        let mut seen = BTreeSet::new();
        let mut revealed = vec![];
        for index in observation.revealed_indices {
            ensure!(seen.insert(index), "duplicate revealed index");
            revealed.push(
                measured
                    .get(index)
                    .context("unknown revealed index")?
                    .clone(),
            );
        }
        Ok(ObservedSlate {
            programs: observation.programs,
            revealed,
        })
    }
}

pub(crate) fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Slate {
    pub record: ResearchRound,
    pub dsls: Vec<String>,
    pub dispatch: Vec<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CapturedAttempt {
    pub proposal_index: usize,
    pub admitted_slot: Option<usize>,
    pub request: BlobRef,
    pub result: Option<BlobRef>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CapturedRound {
    pub slate: BlobRef,
    pub attempts: Vec<CapturedAttempt>,
    pub committed: Option<BlobRef>,
    pub aborted: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Measured {
    pub response: Response,
    pub outcome: RecordedOutcome,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct AuxiliaryAttempt {
    pub request: BlobRef,
    pub result: Option<BlobRef>,
    pub model: Option<BlobRef>,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum AuxiliaryResult {
    Evaluation { response: Response },
    Fit,
    Failure { worker: Option<Response> },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CaptureState {
    pub version: u32,
    pub binding: BlobRef,
    pub config: CaptureConfig,
    pub charged: u64,
    pub bytes: u64,
    pub label_counter: u32,
    pub tainted: bool,
    pub rounds: BTreeMap<u64, CapturedRound>,
    pub auxiliary: Vec<AuxiliaryAttempt>,
}

impl CaptureState {
    pub fn new(store: &Store, context: &CaptureContext) -> Result<Self> {
        let bytes = serde_json::to_vec(&context.binding)?;
        ensure!(
            bytes.len() as u64 <= context.binding.config.max_bytes,
            "capture binding exceeds byte budget"
        );
        let binding = store.publish_blob(Cursor::new(&bytes))?;
        Ok(Self {
            version: ENVELOPE_VERSION,
            binding,
            config: context.binding.config.clone(),
            charged: 0,
            bytes: bytes.len() as u64,
            label_counter: 0,
            tainted: false,
            rounds: BTreeMap::new(),
            auxiliary: vec![],
        })
    }

    pub fn check_binding(&self, store: &Store, context: &CaptureContext) -> Result<()> {
        ensure!(
            self.version == ENVELOPE_VERSION && self.config == context.binding.config,
            "capture version/config mismatch"
        );
        ensure!(
            store.read_blob(&self.binding)? == serde_json::to_vec(&context.binding)?,
            "capture input binding changed"
        );
        self.config.validate(context.binding.config.max_attempts)?;
        ensure!(
            self.charged <= self.config.max_attempts
                && self.bytes <= self.config.max_bytes
                && self.rounds.len() <= self.config.max_rounds,
            "capture accounting exceeds its bounds"
        );
        self.validate_storage(store)?;
        Ok(())
    }

    pub fn validate_storage(&self, store: &Store) -> Result<()> {
        ensure!(
            self.version == ENVELOPE_VERSION,
            "capture envelope version mismatch"
        );
        self.config.validate(self.config.max_attempts)?;
        let binding: Binding = Self::read(store, &self.binding)?;
        ensure!(self.config == binding.config, "capture config drift");
        let mut charged = 0_u64;
        for (index, (number, round)) in self.rounds.iter().enumerate() {
            ensure!(*number == index as u64, "capture round gap");
            let slate: Slate = Self::read(store, &round.slate)?;
            slate.record.validate()?;
            ensure!(
                slate.record.round == *number
                    && slate.record.identity == binding.identity
                    && slate.record.seed == binding.seed
                    && slate.record.tokens == binding.tokens
                    && slate.record.mask == binding.mask
                    && slate.record.evaluate_all_proposals == self.config.evaluate_all_proposals
                    && slate.dsls.len() == slate.record.proposals.len()
                    && slate
                        .dsls
                        .iter()
                        .zip(&slate.record.proposals)
                        .all(|(dsl, p)| digest(dsl.as_bytes()) == p.program_sha256),
                "capture slate binding mismatch"
            );
            let mut expected = slate.record.policy.admitted.clone();
            if slate.record.evaluate_all_proposals {
                expected.extend(
                    (0..slate.dsls.len()).filter(|i| !slate.record.policy.admitted.contains(i)),
                );
            }
            ensure!(
                expected.starts_with(&slate.dispatch),
                "capture dispatch mapping drift"
            );
            charged = charged
                .checked_add(slate.dispatch.len() as u64)
                .context("capture charge overflow")?;
            ensure!(
                round.attempts.len() == slate.dispatch.len(),
                "capture reservation inventory mismatch"
            );
            for (position, attempt) in round.attempts.iter().enumerate() {
                let request: Request = Self::read(store, &attempt.request)?;
                ensure!(
                    attempt.proposal_index == slate.dispatch[position]
                        && request.attempt.run.to_string() == binding.identity.run_id
                        && request.attempt.round == *number
                        && request.dsl == slate.dsls[attempt.proposal_index],
                    "capture request binding mismatch"
                );
                let admitted = slate
                    .record
                    .policy
                    .admitted
                    .iter()
                    .position(|i| *i == attempt.proposal_index);
                ensure!(
                    attempt.admitted_slot == admitted
                        && request.attempt.slot
                            == admitted.map_or(32 + attempt.proposal_index as u32, |s| s as u32),
                    "capture request slot mismatch"
                );
            }
            if let Some(reference) = &round.committed {
                let actual = self.completed_record(store, *number)?;
                ensure!(
                    store.read_blob(reference)? == serde_json::to_vec(&actual)?,
                    "committed capture differs from attempt journal"
                );
            }
        }
        for attempt in &self.auxiliary {
            let request: Request = Self::read(store, &attempt.request)?;
            ensure!(
                request.attempt.run.to_string() == binding.identity.run_id,
                "auxiliary request run mismatch"
            );
            if let Some(reference) = &attempt.result {
                let response = match Self::read(store, reference)? {
                    AuxiliaryResult::Evaluation { response } => Some(response),
                    AuxiliaryResult::Fit => {
                        ensure!(
                            attempt.model.as_ref().is_some_and(|model| model.bytes > 0),
                            "captured model root missing or empty"
                        );
                        None
                    }
                    AuxiliaryResult::Failure { worker } => {
                        ensure!(
                            worker
                                .as_ref()
                                .is_none_or(|response| response.failure.is_some()),
                            "auxiliary failure contains a successful response"
                        );
                        worker
                    }
                };
                if let Some(response) = response {
                    ensure!(
                        response.attempt == request.attempt
                            && response.valid()
                            && response.failure != Some(crate::worker::WorkerFailure::StaleAttempt),
                        "auxiliary response identity or outcome mismatch"
                    );
                }
            }
        }
        ensure!(
            charged == self.charged
                && charged <= self.config.max_attempts
                && self.bytes <= self.config.max_bytes
                && self.rounds.len() <= self.config.max_rounds
                && self.auxiliary.len() <= self.config.max_rounds.saturating_mul(2),
            "capture accounting mismatch"
        );
        let mut roots = BTreeMap::new();
        for root in self.roots() {
            store.verify_blob(&root)?;
            roots.insert(root.sha256, root.bytes);
        }
        ensure!(
            roots
                .values()
                .try_fold(0_u64, |sum, bytes| sum.checked_add(*bytes))
                .is_some_and(|total| total <= self.bytes),
            "capture byte accounting mismatch"
        );
        Ok(())
    }

    pub fn publish<T: Serialize>(&mut self, store: &Store, value: &T) -> Result<BlobRef> {
        let bytes = serde_json::to_vec(value)?;
        let total = self
            .bytes
            .checked_add(bytes.len() as u64)
            .context("capture byte counter exhausted")?;
        ensure!(
            total <= self.config.max_bytes,
            "capture byte budget exhausted"
        );
        let reference = store.publish_blob(Cursor::new(bytes))?;
        self.bytes = total;
        Ok(reference)
    }

    pub fn read<T: serde::de::DeserializeOwned>(store: &Store, reference: &BlobRef) -> Result<T> {
        ensure!(
            reference.bytes <= MAX_CAPTURE_BYTES,
            "capture blob exceeds ceiling"
        );
        Ok(serde_json::from_slice(&store.read_blob(reference)?)?)
    }

    pub fn roots(&self) -> Vec<BlobRef> {
        let mut roots = vec![self.binding.clone()];
        for round in self.rounds.values() {
            roots.push(round.slate.clone());
            roots.extend(round.committed.iter().cloned());
            for attempt in &round.attempts {
                roots.push(attempt.request.clone());
                roots.extend(attempt.result.iter().cloned());
            }
        }
        for attempt in &self.auxiliary {
            roots.push(attempt.request.clone());
            roots.extend(attempt.result.iter().cloned());
            roots.extend(attempt.model.iter().cloned());
        }
        roots
    }

    pub fn unknown(&self) -> bool {
        self.tainted
            || self.auxiliary.iter().any(|a| a.result.is_none())
            || self
                .rounds
                .values()
                .any(|r| r.aborted || r.attempts.iter().any(|a| a.result.is_none()))
    }

    pub fn measured_pool(&self, store: &Store, keys: &[String]) -> Result<Vec<RecordedReveal>> {
        let mut observed = BTreeMap::new();
        for round in self.rounds.values().filter(|r| r.committed.is_some()) {
            let record: ResearchRound = Self::read(store, round.committed.as_ref().unwrap())?;
            for outcome in record.outcomes {
                if !record.policy.admitted.contains(&outcome.proposal_index) {
                    continue;
                }
                if let Rung0Measurement::Success {
                    surrogate, gate, ..
                } = outcome.measurement
                {
                    let program = record.proposals[outcome.proposal_index].clone();
                    observed.insert(
                        program.program_sha256.clone(),
                        RecordedReveal {
                            program,
                            surrogate,
                            gate,
                        },
                    );
                }
            }
        }
        keys.iter()
            .map(|key| {
                observed
                    .get(key)
                    .cloned()
                    .context("elite lacks a committed measured gate")
            })
            .collect()
    }

    fn completed_record(&self, store: &Store, number: u64) -> Result<ResearchRound> {
        let round = self
            .rounds
            .get(&number)
            .context("capture round missing")?
            .clone();
        ensure!(!round.aborted, "capture round aborted");
        let mut slate: Slate = Self::read(store, &round.slate)?;
        ensure!(
            round.attempts.len() == slate.dispatch.len(),
            "incomplete capture reservations"
        );
        for attempt in &round.attempts {
            let measured: Measured = Self::read(
                store,
                attempt.result.as_ref().context("unknown capture outcome")?,
            )?;
            let request: Request = Self::read(store, &attempt.request)?;
            ensure!(
                measured.response.attempt == request.attempt
                    && measured.response.valid()
                    && measured.outcome.proposal_index == attempt.proposal_index,
                "captured attempt mismatch"
            );
            ensure!(
                match (&measured.response.failure, &measured.outcome.measurement) {
                    (Some(code), Rung0Measurement::Failed { code: recorded }) =>
                        code == recorded && *code != crate::worker::WorkerFailure::StaleAttempt,
                    (
                        None,
                        Rung0Measurement::Success {
                            surrogate,
                            objective,
                            ..
                        },
                    ) =>
                        measured.response.scores.len() == 3
                            && measured.response.scores[0] == *surrogate
                            && measured.response.scores[1] == *objective,
                    _ => false,
                },
                "capture measurement differs from response"
            );
            slate.record.dispatched.push(attempt.proposal_index);
            slate.record.outcomes.push(measured.outcome);
        }
        let expected = if slate.record.evaluate_all_proposals {
            slate.record.proposals.len()
        } else {
            slate.record.policy.admitted.len()
        };
        slate.record.capture_complete = slate.record.dispatched.len() == expected;
        slate.record.validate()?;
        ensure!(
            serde_json::to_vec(&slate.record)?.len() < MAX_LINE_BYTES,
            "capture record exceeds line bound"
        );
        Ok(slate.record)
    }

    pub fn finish_round(&mut self, store: &Store, number: u64) -> Result<()> {
        ensure!(
            self.rounds
                .get(&number)
                .is_some_and(|r| r.committed.is_none()),
            "capture round already committed"
        );
        let record = self.completed_record(store, number)?;
        let reference = self.publish(store, &record)?;
        self.rounds.get_mut(&number).unwrap().committed = Some(reference);
        Ok(())
    }
}
