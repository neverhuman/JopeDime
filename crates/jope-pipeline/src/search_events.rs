//! Observational draw/request/physical-attempt joins. No replay or fitting authority.

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Mutex;

// Reuse I02's actual codec without changing the coordinator-owned Cargo graph. The integrator
// can replace this source-module import with a jope-records dependency; no codec is forked.
// Extract I02's jope_records-importing doctests only in their owning crate. Runtime
// codec checks remain compiled here; I05's nonzero integration fixtures exercise this adapter.
#[cfg(not(doctest))]
#[allow(dead_code)]
#[path = "../../jope-records/src/wire_contract.rs"]
mod wire;

pub const LOG_SCHEMA: &str = "jope.search-events/v1";
pub const DECISION_SCHEMA: &str = "jope.search-decision/v1";
pub const EDIT_SCHEMA: &str = "jope.edit-transition/v1";

/// I02 canonical bytes, including exact float tags and length-prefixed domain separation.
pub fn canonical_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    Ok(
        wire::CanonicalPayload::parse_and_canonicalize(&serde_json::to_vec(value)?)?
            .as_bytes()
            .to_vec(),
    )
}

pub fn identity<T: Serialize>(value: &T) -> Result<String> {
    Ok(
        wire::CanonicalPayload::parse_and_canonicalize(&serde_json::to_vec(value)?)?
            .domain_separated_id()?
            .sha256_hex()
            .to_owned(),
    )
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FloatBits {
    pub f64_bits: String,
}

impl FloatBits {
    pub fn new(value: f64) -> Result<Self> {
        ensure!(value.is_finite(), "nonfinite event measurement");
        Ok(Self {
            f64_bits: format!("{:016x}", value.to_bits()),
        })
    }

    pub fn value(&self) -> Result<f64> {
        let value = wire::ExactF64::from_hex("measurement", &self.f64_bits)?.to_f64();
        ensure!(value.is_finite(), "nonfinite event measurement");
        Ok(value)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum EvidenceRef {
    Known { sha256: String },
    Unknown { reason: String },
}

impl EvidenceRef {
    pub fn unknown(reason: &str) -> Self {
        Self::Unknown {
            reason: reason.into(),
        }
    }
    fn check(&self) -> Result<()> {
        match self {
            Self::Known { sha256 } => ensure!(
                crate::experiment::sha256(sha256),
                "invalid evidence reference"
            ),
            Self::Unknown { reason } => {
                ensure!(!reason.is_empty(), "unknown evidence needs a reason")
            }
        }
        Ok(())
    }
}

/// Authenticated role/exposure remains an I04 input. Its presence here never confers eligibility.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchBinding {
    pub run_id: String,
    pub task: String,
    pub experiment: EvidenceRef,
    pub role_exposure: EvidenceRef,
    pub context_sha256: String,
    pub source_sha256: String,
    pub evaluator: EvidenceRef,
    pub profile: EvidenceRef,
    pub legal_set: EvidenceRef,
    pub fidelity: String,
}

impl SearchBinding {
    fn check(&self) -> Result<()> {
        ensure!(
            !self.run_id.is_empty() && !self.task.is_empty() && !self.fidelity.is_empty(),
            "missing search binding"
        );
        ensure!(
            crate::experiment::sha256(&self.context_sha256)
                && crate::experiment::sha256(&self.source_sha256),
            "invalid context/source identity"
        );
        for r in [
            &self.experiment,
            &self.role_exposure,
            &self.evaluator,
            &self.profile,
            &self.legal_set,
        ] {
            r.check()?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum BehaviorLaw {
    Unknown {
        reason: String,
    },
    NotApplicable {
        reason: String,
    },
    DiscreteProbability {
        subject_sha256: String,
        probability: FloatBits,
    },
    ContinuousDensity {
        subject_sha256: String,
        density: FloatBits,
        measure: String,
    },
}

impl BehaviorLaw {
    fn check(&self, source: &str, program: &Option<String>) -> Result<()> {
        // An edit's unconditional child law is not established by parent selection or seed law.
        if matches!(source, "mutate" | "crossover" | "random_fallback") {
            ensure!(
                matches!(self, Self::Unknown { .. }),
                "GP child law must be unknown; ancestry is not density"
            );
        }
        match self {
            Self::Unknown { reason } | Self::NotApplicable { reason } => {
                ensure!(!reason.is_empty(), "missing behavior-law reason")
            }
            Self::DiscreteProbability {
                subject_sha256,
                probability,
            } => {
                let p = probability.value()?;
                ensure!(
                    program.as_ref() == Some(subject_sha256) && p > 0. && p <= 1.,
                    "probability has wrong subject or range"
                );
            }
            Self::ContinuousDensity {
                subject_sha256,
                density,
                measure,
            } => {
                ensure!(
                    program.as_ref() == Some(subject_sha256)
                        && density.value()? >= 0.
                        && !measure.is_empty(),
                    "density has wrong subject or measure"
                );
            }
        }
        Ok(())
    }
}

/// Effort is per stage, never inferred from an absent physical evaluation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Effort {
    pub wall_seconds: Option<FloatBits>,
    pub cpu_seconds: Option<FloatBits>,
    pub gpu_seconds: Option<FloatBits>,
    pub peak_rss_bytes: Option<u64>,
    pub reason: String,
}

impl Effort {
    pub fn unknown(reason: &str) -> Self {
        Self {
            wall_seconds: None,
            cpu_seconds: None,
            gpu_seconds: None,
            peak_rss_bytes: None,
            reason: reason.into(),
        }
    }
    pub fn wall(seconds: f64) -> Result<Self> {
        Ok(Self {
            wall_seconds: Some(FloatBits::new(seconds)?),
            ..Self::unknown("caller-measured wall span; CPU/GPU/memory not instrumented")
        })
    }
    fn check(&self) -> Result<()> {
        ensure!(!self.reason.is_empty(), "missing cost reason");
        for n in [&self.wall_seconds, &self.cpu_seconds, &self.gpu_seconds]
            .into_iter()
            .flatten()
        {
            ensure!(n.value()? >= 0., "negative effort");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchDecision {
    pub schema: String,
    pub coordinate: String,
    pub source: String,
    pub seed_hex: Option<String>,
    pub program_sha256: Option<String>,
    pub canonical_sha256: Option<String>,
    pub request_id: Option<String>,
    #[serde(rename = "draw_disposition")]
    pub outcome: String,
    pub detail: Option<String>,
    pub behavior_law: BehaviorLaw,
    pub proposal_cost: Effort,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EditTransition {
    pub schema: String,
    pub draw_id: String,
    /// Ordered, including both choices even when crossover selected the same parent twice.
    pub parents: Vec<String>,
    pub parent_canonical_sha256: Vec<Option<String>>,
    pub donor: Option<String>,
    pub sibling_group: String,
    pub operator: String,
    pub parameters: BTreeMap<String, String>,
    pub post_canonical_sha256: Option<String>,
    pub legal: bool,
    pub refusal: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SearchEvent {
    Draw {
        decision: SearchDecision,
    },
    Edit {
        transition: EditTransition,
    },
    Admission {
        draw_id: String,
        disposition: String,
    },
    Started {
        draw_id: String,
        request_id: String,
        attempt_id: String,
        ordinal: u32,
        retry_of: Option<String>,
    },
    Finished {
        attempt_id: String,
        status: String,
        reason: Option<String>,
        observation_sha256: Option<String>,
        cost: Effort,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventRecord {
    pub event_id: String,
    pub event: SearchEvent,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchEventLog {
    pub schema: String,
    pub binding: SearchBinding,
    pub events: Vec<EventRecord>,
    /// Sorted capture inventory: losing even an unattempted/shadow draw is detectable.
    pub captured_event_ids: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Reconciliation {
    pub draws: usize,
    pub duplicate_draws: usize,
    pub migration_decisions: usize,
    pub unique_requests: usize,
    pub evaluation_attempts: usize,
    pub failed_attempts: usize,
    pub logical_completed: usize,
    pub pending_attempts: usize,
    pub duplicate_deliveries: usize,
    pub proposal_costs: usize,
    pub terminal_costs: usize,
    pub evaluation_wall_seconds: f64,
    pub unknown_evaluation_wall_costs: usize,
}

impl SearchEventLog {
    pub fn new(binding: SearchBinding) -> Result<Self> {
        binding.check()?;
        Ok(Self {
            schema: LOG_SCHEMA.into(),
            binding,
            events: Vec::new(),
            captured_event_ids: Vec::new(),
        })
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        // I02 rejects duplicate keys and malformed exact numbers before serde dispatch.
        let payload = wire::CanonicalPayload::from_canonical_bytes(bytes)?;
        let log: Self = serde_json::from_slice(payload.as_bytes())?;
        log.reconcile()?;
        Ok(log)
    }

    pub fn append(&mut self, event: SearchEvent) -> Result<String> {
        let event_id = self.event_id(&event)?;
        if let Err(index) = self.captured_event_ids.binary_search(&event_id) {
            self.captured_event_ids.insert(index, event_id.clone());
        }
        self.events.push(EventRecord {
            event_id: event_id.clone(),
            event,
        });
        Ok(event_id)
    }

    pub fn event_id(&self, event: &SearchEvent) -> Result<String> {
        identity(
            &serde_json::json!({"schema": "jope.search-event/v1", "binding": self.binding, "event": event}),
        )
    }

    pub fn request_id(
        &self,
        program_sha256: &str,
        canonical_sha256: &Option<String>,
    ) -> Result<String> {
        identity(
            &serde_json::json!({"schema": "jope.selfplay.draw-request/v1", "binding": self.binding, "program_sha256": program_sha256, "canonical_sha256": canonical_sha256}),
        )
    }

    pub fn start(
        &mut self,
        draw_id: &str,
        ordinal: u32,
        retry_of: Option<String>,
    ) -> Result<String> {
        let decision = self
            .events
            .iter()
            .find_map(|r| match &r.event {
                SearchEvent::Draw { decision } if r.event_id == draw_id => Some(decision),
                _ => None,
            })
            .context("attempt without draw")?;
        let request_id = decision
            .request_id
            .clone()
            .context("attempt without request")?;
        let attempt_id = identity(
            &serde_json::json!({"schema": "jope.search-attempt/v1", "binding": self.binding, "request_id": request_id, "ordinal": ordinal}),
        )?;
        ensure!(!self.events.iter().any(|r| matches!(&r.event, SearchEvent::Started { attempt_id: previous, .. } if previous == &attempt_id)), "physical attempt already started; a new start needs a new ordinal");
        self.append(SearchEvent::Started {
            draw_id: draw_id.into(),
            request_id,
            attempt_id: attempt_id.clone(),
            ordinal,
            retry_of,
        })?;
        Ok(attempt_id)
    }

    /// Pure reconciliation, independent of delivery/completion order. Repeated identical records
    /// are delivery duplicates; changed records, conflicting terminals and missing joins refuse.
    pub fn reconcile(&self) -> Result<Reconciliation> {
        ensure!(
            self.schema == LOG_SCHEMA,
            "unsupported search-event version"
        );
        self.binding.check()?;
        let mut summary = Reconciliation::default();
        let mut unique = BTreeMap::new();
        for record in &self.events {
            ensure!(
                record.event_id == self.event_id(&record.event)?,
                "altered search event"
            );
            if let Some(old) = unique.insert(&record.event_id, &record.event) {
                ensure!(old == &record.event, "conflicting event delivery");
                summary.duplicate_deliveries += 1;
            }
        }
        ensure!(
            self.captured_event_ids.iter().collect::<Vec<_>>()
                == unique.keys().copied().collect::<Vec<_>>(),
            "missing or conflicting captured event inventory"
        );
        let mut draws = BTreeMap::new();
        let mut coordinates = BTreeSet::new();
        let mut requests = BTreeSet::new();
        let mut starts = BTreeMap::new();
        let mut finishes = BTreeMap::new();
        let mut admissions = BTreeMap::new();
        let mut edits = BTreeSet::new();
        for (id, event) in &unique {
            match event {
                SearchEvent::Draw { decision: d } => {
                    ensure!(
                        d.schema == DECISION_SCHEMA
                            && !d.source.is_empty()
                            && coordinates.insert(&d.coordinate),
                        "conflicting draw coordinate"
                    );
                    if let Some(seed) = &d.seed_hex {
                        wire::HexU64::from_hex("seed_hex", seed)?;
                    }
                    d.proposal_cost.check()?;
                    d.behavior_law.check(&d.source, &d.program_sha256)?;
                    if let Some(program) = &d.program_sha256 {
                        ensure!(
                            crate::experiment::sha256(program)
                                && d.canonical_sha256
                                    .as_ref()
                                    .is_none_or(|h| crate::experiment::sha256(h)),
                            "invalid recipe identity"
                        );
                        ensure!(
                            d.request_id.as_ref()
                                == Some(&self.request_id(program, &d.canonical_sha256)?),
                            "wrong request identity"
                        );
                        if !d.source.starts_with("migration")
                            && !requests.insert(d.request_id.as_ref().unwrap())
                        {
                            summary.duplicate_draws += 1;
                        }
                    } else {
                        ensure!(
                            d.request_id.is_none() && d.canonical_sha256.is_none(),
                            "request without program"
                        );
                    }
                    ensure!(
                        matches!(
                            d.outcome.as_str(),
                            "candidate"
                                | "duplicate"
                                | "operator_failed"
                                | "migration_admitted"
                                | "migration_rejected"
                        ),
                        "unknown draw outcome"
                    );
                    draws.insert(*id, d);
                    if d.source == "migration" {
                        summary.migration_decisions += 1;
                    } else {
                        summary.draws += 1;
                        summary.proposal_costs += 1;
                    }
                }
                SearchEvent::Started { attempt_id, .. } => {
                    ensure!(
                        starts.insert(attempt_id, event).is_none(),
                        "conflicting physical start"
                    );
                }
                SearchEvent::Finished { attempt_id, .. } => {
                    ensure!(
                        finishes.insert(attempt_id, event).is_none(),
                        "conflicting attempt terminal"
                    );
                }
                SearchEvent::Admission {
                    draw_id,
                    disposition,
                } => {
                    ensure!(
                        matches!(
                            disposition.as_str(),
                            "admitted" | "duplicate" | "evaluations" | "seconds" | "cancelled"
                        ),
                        "unknown admission disposition"
                    );
                    ensure!(
                        admissions.insert(draw_id, disposition).is_none(),
                        "conflicting draw admission"
                    );
                }
                SearchEvent::Edit { .. } => {}
            }
        }
        summary.unique_requests = requests.len();
        let mut successful_requests = BTreeSet::new();
        let mut ordinals = BTreeSet::new();
        for event in starts.values() {
            let SearchEvent::Started {
                draw_id,
                request_id,
                attempt_id,
                ordinal,
                retry_of,
            } = event
            else {
                unreachable!()
            };
            let draw = draws
                .get(draw_id)
                .context("missing draw event for attempt")?;
            ensure!(
                draw.request_id.as_ref() == Some(request_id)
                    && !draw.source.starts_with("migration"),
                "attempt joined to wrong request or migration"
            );
            ensure!(
                *ordinal > 0 && ordinals.insert((request_id, ordinal)),
                "duplicate attempt ordinal"
            );
            ensure!(
                *attempt_id
                    == identity(
                        &serde_json::json!({"schema": "jope.search-attempt/v1", "binding": self.binding, "request_id": request_id, "ordinal": ordinal})
                    )?,
                "altered attempt ID"
            );
            match retry_of {
                None => ensure!(*ordinal == 1, "retry missing predecessor"),
                Some(prior) => {
                    let SearchEvent::Started {
                        request_id: previous_request,
                        ordinal: previous_ordinal,
                        ..
                    } = starts.get(prior).context("orphan retry start")?
                    else {
                        unreachable!()
                    };
                    ensure!(
                        request_id == previous_request
                            && previous_ordinal.checked_add(1) == Some(*ordinal),
                        "retry changed request/ordinal"
                    );
                    let SearchEvent::Finished { status, .. } = finishes
                        .get(prior)
                        .context("retry without retained failure")?
                    else {
                        unreachable!()
                    };
                    ensure!(status != "measured", "retry after successful completion");
                }
            }
            summary.evaluation_attempts += 1;
            if !finishes.contains_key(attempt_id) {
                summary.pending_attempts += 1;
            }
        }
        for event in finishes.values() {
            let SearchEvent::Finished {
                attempt_id,
                status,
                reason,
                observation_sha256,
                cost,
            } = event
            else {
                unreachable!()
            };
            let SearchEvent::Started { request_id, .. } =
                starts.get(attempt_id).context("orphan terminal cost")?
            else {
                unreachable!()
            };
            ensure!(
                matches!(
                    status.as_str(),
                    "measured"
                        | "parse_failed"
                        | "fit_failed"
                        | "score_failed"
                        | "panicked"
                        | "timed_out"
                        | "interrupted"
                ),
                "unknown attempt status"
            );
            ensure!(
                (status == "measured") == reason.is_none(),
                "terminal reason/status mismatch"
            );
            ensure!(
                reason.as_ref().is_none_or(|r| !r.is_empty()),
                "empty failure reason"
            );
            ensure!(
                observation_sha256
                    .as_ref()
                    .is_none_or(|h| crate::experiment::sha256(h)),
                "invalid observation reference"
            );
            cost.check()?;
            if status == "measured" {
                ensure!(
                    observation_sha256.is_some() && successful_requests.insert(request_id),
                    "missing result or multiple logical completions"
                );
                summary.logical_completed += 1;
            } else {
                summary.failed_attempts += 1;
            }
            summary.terminal_costs += 1;
            match &cost.wall_seconds {
                Some(seconds) => summary.evaluation_wall_seconds += seconds.value()?,
                None => summary.unknown_evaluation_wall_costs += 1,
            }
        }
        for event in unique.values() {
            if let SearchEvent::Edit { transition: e } = event {
                let draw = draws.get(&e.draw_id).context("edit missing draw")?;
                ensure!(
                    e.schema == EDIT_SCHEMA && edits.insert(&e.draw_id),
                    "conflicting edit"
                );
                ensure!(
                    e.operator == draw.source
                        && e.post_canonical_sha256 == draw.canonical_sha256
                        && e.parents.len() == e.parent_canonical_sha256.len(),
                    "edit subject mismatch"
                );
                let expected = match e.operator.as_str() {
                    "crossover" => 2,
                    "mutate" => 1,
                    _ => 0,
                };
                ensure!(
                    e.parents.len() == expected
                        && e.parents.iter().all(|h| crate::experiment::sha256(h)),
                    "missing edit parents"
                );
                ensure!(
                    e.donor.as_ref()
                        == if expected == 2 {
                            e.parents.get(1)
                        } else {
                            None
                        },
                    "wrong edit donor"
                );
                ensure!(
                    e.legal == e.refusal.is_none() && e.legal == draw.canonical_sha256.is_some(),
                    "edit legality mismatch"
                );
            }
        }
        for (id, draw) in draws {
            if matches!(
                draw.source.as_str(),
                "mutate" | "crossover" | "random_fallback"
            ) {
                ensure!(edits.contains(id), "missing edit transition");
            }
        }
        for id in admissions.keys() {
            ensure!(
                unique
                    .get(id)
                    .is_some_and(|e| matches!(e, SearchEvent::Draw { .. })),
                "admission missing draw"
            );
        }
        Ok(summary)
    }

    pub fn require_complete(&self) -> Result<Reconciliation> {
        let result = self.reconcile()?;
        ensure!(
            result.draws > 0 && result.evaluation_attempts > 0,
            "empty search evidence cannot qualify measurement closure"
        );
        ensure!(
            result.pending_attempts == 0,
            "incomplete search evidence: missing terminal cost"
        );
        Ok(result)
    }
}

/// A shared recorder permits starts/finishes on the actual evaluation worker. Failures are
/// retained and propagated at publication; a poisoned/failed recorder never publishes success.
#[derive(Debug)]
pub(crate) struct Recorder(Mutex<(SearchEventLog, Option<String>)>);

impl Recorder {
    pub(crate) fn new(log: SearchEventLog) -> Self {
        Self(Mutex::new((log, None)))
    }
    pub(crate) fn update<T>(&self, f: impl FnOnce(&mut SearchEventLog) -> Result<T>) -> Option<T> {
        let mut state = self.0.lock().unwrap_or_else(|p| p.into_inner());
        match f(&mut state.0) {
            Ok(value) => Some(value),
            Err(error) => {
                state.1 = Some(format!("{error:#}"));
                None
            }
        }
    }
    pub(crate) fn publish(&self, output: &Path) -> Result<()> {
        let state = self
            .0
            .lock()
            .map_err(|_| anyhow::anyhow!("search event recorder poisoned"))?;
        ensure!(
            state.1.is_none(),
            "search evidence incomplete: {:?}",
            state.1
        );
        state.0.reconcile()?;
        std::fs::create_dir_all(output)?;
        let bytes = canonical_bytes(&state.0)?;
        // Atomic sidecar replacement across exploration/comparison; old observation bytes stay
        // in their existing files. An I/O failure fails the caller before a success publication.
        let staged = output.join(".search-events.staging");
        let mut file = std::fs::File::create(&staged)?;
        use std::io::Write;
        file.write_all(&bytes)?;
        file.sync_all()?;
        std::fs::rename(staged, output.join("search-events.json"))?;
        std::fs::File::open(output)?.sync_all()?;
        Ok(())
    }
}

pub fn canonical_recipe(genome: &str) -> Option<String> {
    feat_math::genome::GenomeIr::parse(genome)
        .ok()
        .map(|ir| crate::sha256_hex(ir.canonical().as_bytes()))
}

/// Exact ordered input identity without exporting labels or floating-point JSON numbers.
pub fn cut_context_sha256(cut: &crate::prepare::Cut) -> Result<String> {
    let bits = |rows: &[Vec<f64>]| {
        rows.iter()
            .map(|r| {
                r.iter()
                    .map(|v| format!("{:016x}", v.to_bits()))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };
    Ok(crate::sha256_hex(&serde_json::to_vec(&(
        cut.index,
        bits(&cut.train_x),
        bits(&cut.hold_x),
        bits(&[cut.train_y.clone(), cut.hold_y.clone()]),
        &cut.hold_indices,
    ))?))
}

/// Logger/search/operator source identity. Evaluator/build qualification remains separate.
pub fn source_sha256() -> String {
    crate::sha256_hex(
        &[
            include_bytes!("search.rs").as_slice(),
            include_bytes!("search_events.rs").as_slice(),
            include_bytes!("../../feat-math/src/evolve.rs").as_slice(),
            include_bytes!("../../jope-records/src/wire_contract.rs").as_slice(),
        ]
        .concat(),
    )
}
