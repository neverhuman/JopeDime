use super::*;

/// Explicit join to the already planned W8 unit; never reconstruct it from a winner or change
/// the original run/attempt identity. This is observation, not I04 replay eligibility.
#[derive(Clone, Debug)]
pub struct SearchWorkBinding {
    /// Identifier for the associated request.
    pub request_id: String,
    /// Identifier for the associated work.
    pub work_id: String,
}

#[derive(Clone, Debug)]
/// `SearchMapped` is a record for the search mapped contract in this module: W8 product integration: the engine's durable events as W8 progress objects.
pub struct SearchMapped {
    /// Mapped recorded for this search mapped.
    pub mapped: Mapped,
    /// Collection of events values carried by this search mapped.
    pub events: Vec<Decoded<Event>>,
    /// Reconciliation recorded for this search mapped.
    pub reconciliation: jope_pipeline::search_events::Reconciliation,
}

/// Map an authenticated event sidecar and original result blobs through the actual W8 event
/// types. The caller retains `mapped.blobs` (including the full sidecar/costs) before its objects.
/// Missing/changed evidence refuses the entire mapping. Duplicate deliveries keep their IDs.
pub fn search_events(
    manifest: &Decoded<RunManifest>,
    bytes: &[u8],
    expected_sha256: &str,
    bindings: &[SearchWorkBinding],
    observations: &[Vec<u8>],
) -> Result<SearchMapped> {
    use jope_pipeline::search_events::{SearchEvent, SearchEventLog};
    use std::collections::{BTreeMap, BTreeSet};
    ensure!(
        sha256_hex(bytes) == expected_sha256,
        "altered search-event artifact"
    );
    let log = SearchEventLog::from_bytes(bytes)?;
    ensure!(
        log.binding.run_id == manifest.value().run_id,
        "search event belongs to another original run"
    );
    let reconciliation = log.require_complete()?;
    let mut units = BTreeMap::new();
    let mut used_units = BTreeSet::new();
    for binding in bindings {
        planned(manifest, &binding.work_id)?;
        ensure!(
            manifest
                .value()
                .units
                .iter()
                .any(|unit| unit.work_id == binding.work_id
                    && unit.spec.inputs.contains(&binding.request_id)),
            "request absent from original planned work inputs"
        );
        ensure!(
            units
                .insert(&binding.request_id, &binding.work_id)
                .is_none()
                && used_units.insert(&binding.work_id),
            "conflicting request/work join"
        );
    }
    let mut results = BTreeMap::new();
    for observation in observations {
        results.insert(sha256_hex(observation), observation);
    }
    let mut starts = BTreeMap::new();
    let mut terminals = BTreeMap::new();
    for record in &log.events {
        match &record.event {
            SearchEvent::Started {
                request_id,
                ordinal,
                ..
            } => {
                starts.insert((request_id, ordinal), &record.event);
            }
            SearchEvent::Finished { attempt_id, .. } => {
                terminals.insert(attempt_id, (&record.event_id, &record.event));
            }
            _ => {}
        }
    }
    ensure!(
        units
            .keys()
            .all(|id| starts.keys().any(|(request, _)| request == id)),
        "work join without physical request"
    );
    let mut mapped = Mapped {
        blobs: vec![bytes.to_vec()],
        objects: Vec::new(),
    };
    let mut events = Vec::new();
    let mut failed: BTreeMap<&String, String> = BTreeMap::new();
    for ((request, ordinal), start) in starts {
        let SearchEvent::Started {
            attempt_id,
            retry_of,
            ..
        } = start
        else {
            unreachable!()
        };
        let work = units
            .get(request)
            .context("physical attempt missing original work binding")?;
        let opening = if *ordinal == 1 {
            EventBody::Scheduled
        } else {
            EventBody::Retry {
                failed_event: failed
                    .get(retry_of.as_ref().context("missing retry predecessor")?)
                    .context("retry missing original failure")?
                    .clone(),
            }
        };
        events.push(Decoded::new(Event::new(manifest, work, *ordinal, opening))?);
        events.push(Decoded::new(Event::new(
            manifest,
            work,
            *ordinal,
            EventBody::Started {
                worker: "pipeline-in-process".into(),
            },
        ))?);
        let (terminal_id, terminal_event) = terminals
            .get(attempt_id)
            .context("missing terminal/cost event")?;
        let SearchEvent::Finished {
            status,
            reason,
            observation_sha256,
            cost,
            ..
        } = terminal_event
        else {
            unreachable!()
        };
        ensure!(
            *ordinal <= manifest.value().max_attempts,
            "attempt exceeds original run budget"
        );
        let original = observation_sha256
            .as_ref()
            .map(|digest| {
                results
                    .get(digest)
                    .context("missing original observation bytes")
            })
            .transpose()?;
        if let Some(original) = original {
            let observation: jope_pipeline::search::ObservationRecord =
                serde_json::from_slice(original)?;
            let observation = observation.into_observation()?;
            ensure!(
                observation.seconds.is_finite() && observation.seconds >= 0.,
                "invalid original observation clock"
            );
            ensure!(
                observation.status == *status && observation.failure == *reason,
                "terminal status differs from original observation"
            );
            if let Some(wall) = &cost.wall_seconds {
                ensure!(
                    wall.value()? >= observation.seconds,
                    "attempt wall cost is smaller than its original observation span"
                );
            }
            let SearchEvent::Started { draw_id, .. } = start else {
                unreachable!()
            };
            let program = log
                .events
                .iter()
                .find_map(|r| match &r.event {
                    SearchEvent::Draw { decision } if &r.event_id == draw_id => {
                        decision.program_sha256.as_ref()
                    }
                    _ => None,
                })
                .context("missing original draw subject")?;
            ensure!(
                *program == observation.program_sha256,
                "result belongs to another drawn program"
            );
        }
        let body = if status == "measured" {
            let digest = observation_sha256
                .as_ref()
                .context("missing result digest")?;
            let original = results
                .get(digest)
                .context("missing original observation bytes")?;
            mapped.blobs.push((*original).clone());
            EventBody::Completed {
                result: BlobRef::of(original),
            }
        } else {
            // Retain original failure bytes too whenever the producer names them.
            if let Some(digest) = observation_sha256 {
                let original = results
                    .get(digest)
                    .context("missing original failed observation bytes")?;
                mapped.blobs.push((*original).clone());
            }
            let class = match status.as_str() {
                "parse_failed" => FailureClass::ParseRejection,
                "timed_out" => FailureClass::Timeout,
                "interrupted" => FailureClass::Cancelled,
                _ => FailureClass::WorkerError,
            };
            EventBody::Failed {
                class,
                retryable: *ordinal < manifest.value().max_attempts,
                detail: format!(
                    "{}; attempt {}; cost basis {}; terminal event {}",
                    reason.as_deref().context("failure reason missing")?,
                    attempt_id,
                    cost.reason,
                    terminal_id
                ),
            }
        };
        let terminal = Decoded::new(Event::new(manifest, work, *ordinal, body))?;
        if status != "measured" {
            failed.insert(attempt_id, terminal.digest().to_owned());
        }
        events.push(terminal);
    }
    for event in &events {
        mapped.objects.push(event.encoded()?);
    }
    Ok(SearchMapped {
        mapped,
        events,
        reconciliation,
    })
}
