use super::*;
use anyhow::Context;
use std::collections::{BTreeMap, BTreeSet};

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
