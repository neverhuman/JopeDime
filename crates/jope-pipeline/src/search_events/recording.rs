use super::*;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Mutex;

/// A shared recorder permits starts/finishes on the actual evaluation worker. Failures are
/// retained and propagated at publication; a poisoned/failed recorder never publishes success.
#[derive(Debug)]
pub(crate) struct Recorder(Mutex<(SearchEventLog, Option<String>, Option<NarrationIndex>)>);

#[derive(Debug, Default)]
struct DrawLinks {
    request: Option<String>,
    admission: Option<String>,
    finished: Option<(String, Option<String>)>,
}
#[derive(Debug, Default)]
struct NarrationIndex {
    draws: BTreeMap<String, DrawLinks>,
    attempts: BTreeMap<String, String>,
}
impl NarrationIndex {
    fn observe(&mut self, records: &[EventRecord]) {
        for record in records {
            match &record.event {
                SearchEvent::Draw { decision } => {
                    self.draws
                        .entry(record.event_id.clone())
                        .or_default()
                        .request = decision.request_id.clone();
                }
                SearchEvent::Admission { draw_id, .. } => {
                    self.draws.entry(draw_id.clone()).or_default().admission =
                        Some(record.event_id.clone());
                }
                SearchEvent::Started {
                    draw_id,
                    attempt_id,
                    ..
                } => {
                    self.attempts.insert(attempt_id.clone(), draw_id.clone());
                }
                SearchEvent::Finished {
                    attempt_id,
                    observation_sha256,
                    ..
                } => {
                    if let Some(draw) = self.attempts.get(attempt_id) {
                        self.draws.entry(draw.clone()).or_default().finished =
                            Some((record.event_id.clone(), observation_sha256.clone()));
                    }
                }
                _ => {}
            }
        }
    }
}

impl Recorder {
    pub(crate) fn new(log: SearchEventLog) -> Self {
        Self(Mutex::new((log, None, None)))
    }
    pub(crate) fn update<T>(&self, f: impl FnOnce(&mut SearchEventLog) -> Result<T>) -> Option<T> {
        let mut state = self.0.lock().unwrap_or_else(|p| p.into_inner());
        let start = state.0.events.len();
        let result = f(&mut state.0);
        let (log, _, index) = &mut *state;
        if let Some(index) = index {
            index.observe(&log.events[start..]);
        }
        match result {
            Ok(value) => Some(value),
            Err(error) => {
                state.1 = Some(format!("{error:#}"));
                None
            }
        }
    }
    /// References to the existing evidence contract. This observer never appends events.
    /// A start can be described immediately before fitting: its deterministic identity is the
    /// same identity the ordered evidence fold subsequently appends.
    pub(crate) fn narration_links(
        &self,
        draw_id: &Option<String>,
        stage: &str,
    ) -> jope_narrator::Links {
        let Some(draw_id) = draw_id else {
            return jope_narrator::Links::default();
        };
        let mut state = self.0.lock().unwrap_or_else(|p| p.into_inner());
        if state.2.is_none() {
            let mut index = NarrationIndex::default();
            index.observe(&state.0.events);
            state.2 = Some(index);
        }
        let log = &state.0;
        let found = state.2.as_ref().and_then(|i| i.draws.get(draw_id));
        let Some(request_id) = found.and_then(|d| d.request.clone()) else {
            return jope_narrator::Links {
                draw_id: Some(draw_id.clone()),
                search_event_id: (stage == "draw").then(|| draw_id.clone()),
                ..Default::default()
            };
        };
        let mut links = jope_narrator::Links {
            draw_id: Some(draw_id.clone()),
            request_id: Some(request_id.clone()),
            ..Default::default()
        };
        if stage == "draw" {
            links.search_event_id = Some(draw_id.clone());
            return links;
        }
        if stage == "admission" {
            links.search_event_id = found.and_then(|d| d.admission.clone());
            return links;
        }
        let attempt_id = identity(&serde_json::json!({
            "schema": "jope.search-attempt/v1", "binding": log.binding,
            "request_id": request_id, "ordinal": 1u32,
        }))
        .ok();
        links.attempt_id = attempt_id.clone();
        if stage == "started" {
            if let Some(attempt_id) = attempt_id {
                links.search_event_id = log
                    .event_id(&SearchEvent::Started {
                        draw_id: draw_id.clone(),
                        request_id,
                        attempt_id,
                        ordinal: 1,
                        retry_of: None,
                    })
                    .ok();
            }
        } else if stage == "finished" {
            if let Some((event, observation)) = found.and_then(|d| d.finished.as_ref()) {
                links.search_event_id = Some(event.clone());
                links.observation_sha256 = observation.clone();
            }
        }
        links
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
