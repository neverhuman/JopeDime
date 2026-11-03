use super::recorded::qualification_rows;
use super::*;

impl Reader {
    pub fn start(config: &Config) -> Result<Self> {
        let latest = Arc::new(Mutex::new(Observed::default()));
        let shared = latest.clone();
        let config = config.clone();
        thread::Builder::new()
            .name("jope-science-reader".into())
            .spawn(move || {
                let rt = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(rt) => rt,
                    Err(_) => return,
                };
                loop {
                    let mut rows = Rows::default();
                    match catalog() {
                        Ok(c) => {
                            for source in c.entries {
                                let result = match &source {
                                    Source::RecordedQualification {
                                        id,
                                        artifact,
                                        original_at_unix_ms,
                                    } => checked(None, artifact, 16 << 20).and_then(|v| {
                                        qualification_rows(
                                            id,
                                            artifact,
                                            &v,
                                            *original_at_unix_ms,
                                            now()?,
                                        )
                                    }),
                                    Source::GraphRun {
                                        id,
                                        root,
                                        graph_state,
                                        worker_state,
                                        loop_state,
                                        config: a,
                                    } => rt.block_on(graph_run(
                                        id,
                                        crate::native_observer::science::GraphLocation {
                                            root,
                                            state: graph_state,
                                            worker: worker_state,
                                        },
                                        loop_state,
                                        a,
                                        &Client::new(&config.socket),
                                        config.tenant.clone(),
                                        config.project.clone(),
                                    )),
                                    Source::OrderedWinners {
                                        id,
                                        root,
                                        outbox,
                                        service_receipts,
                                    } => winners(id, root, outbox, service_receipts),
                                };
                                match result {
                                    Ok(r) => merge(&mut rows, r),
                                    Err(e) => rows.errors.push(text(format!(
                                        "Science {} unavailable: {e:#}",
                                        source.id()
                                    ))),
                                }
                                // Publish partial progress so one slow original closure cannot hide prior results.
                                if let Ok(mut value) = shared.lock() {
                                    *value = Observed {
                                        at: now().ok(),
                                        rows: rows.clone(),
                                    };
                                }
                            }
                        }
                        Err(e) => rows.errors.push(text(format!(
                            "Protected scientific catalog unavailable: {e:#}"
                        ))),
                    }
                    if let Ok(mut value) = shared.lock() {
                        *value = Observed {
                            at: now().ok(),
                            rows,
                        };
                    }
                    thread::sleep(Duration::from_secs(10));
                }
            })?;
        Ok(Self { latest })
    }
    pub fn enrich(&self, snapshot: &mut Snapshot) {
        // Operational Loop health is a bounded metadata read on the sampling
        // thread, independent of potentially long scientific closure verification.
        match catalog() {
            Ok(c) => {
                for source in c.entries {
                    if let Source::GraphRun {
                        id,
                        root,
                        loop_state,
                        ..
                    } = source
                    {
                        let result = (|| -> Result<String> {
                            ensure!(
                                loop_state.starts_with(&root),
                                "loop health outside pinned root"
                            );
                            let health: Value = serde_json::from_slice(&regular(
                                &loop_state.join("health.json"),
                                1 << 20,
                            )?)?;
                            ensure!(
                                health["schema"] == "jope.native_loop_health.v1"
                                    && health["model_improvement_claimed"] == false,
                                "loop health schema differs"
                            );
                            let stamp = health["timestamp_unix_ms"]
                                .as_u64()
                                .context("loop original health timestamp absent")?;
                            original_time(Some(stamp), snapshot.observed_at_unix_ms)?;
                            Ok(text(format!("Loop {id}: original health={stamp}, stale={}, declared progress={}",snapshot.observed_at_unix_ms.saturating_sub(stamp)>=STALE_MS,health["progress"])))
                        })();
                        match result {
                            Ok(note) => snapshot.notes.push(note),
                            Err(e) => snapshot
                                .source_errors
                                .push(text(format!("Loop {id} health unavailable: {e:#}"))),
                        }
                    }
                }
            }
            Err(e) => snapshot.source_errors.push(text(format!(
                "Science catalog currently unavailable: {e:#}"
            ))),
        }
        self.enrich_science(snapshot);
    }
    pub(super) fn enrich_science(&self, snapshot: &mut Snapshot) {
        let Ok(observed) = self.latest.lock().map(|v| v.clone()) else {
            return;
        };
        let stale = observed
            .at
            .is_none_or(|t| snapshot.observed_at_unix_ms.saturating_sub(t) >= STALE_MS);
        let mut rows = observed.rows;
        rows.notes.retain(|s| !s.starts_with("Loop "));
        for winner in &mut rows.winners {
            if let Some(status) = winner["identity"]
                .as_str()
                .and_then(|id| rows.publications.get(id))
            {
                winner["local_commit"] = status["local_commit"].clone();
                winner["publication_status"] = status["status"].clone();
                winner["remote_verified_recorded"] = status["remote_verified_recorded"].clone();
                winner["publication_source"] = status["source"].clone();
                winner["capture_sequence"] = status["capture_sequence"].clone();
                winner["scientific_verified_at_unix_ms"] =
                    status["scientific_verified_at_unix_ms"].clone();
            }
        }
        snapshot.notes.retain(|s| {
            !s.starts_with("No independently verified scientific receipt feed configured")
        });
        snapshot.notes.push(text(format!("Scientific verification last={:?}, stale={stale}; historical evidence is not live work or model improvement",observed.at)));
        let expired = |row: &Value| {
            stale
                || row["scientific_verified_at_unix_ms"]
                    .as_u64()
                    .is_none_or(|t| snapshot.observed_at_unix_ms.saturating_sub(t) >= STALE_MS)
        };
        for c in &mut rows.checkpoints {
            if expired(c) {
                c["status"] = json!(text(format!(
                    "STALE science: {}",
                    c["status"].as_str().unwrap_or("unknown")
                )));
            }
        }
        for c in &mut rows.continuations {
            if expired(c) {
                c["generation"] = json!(text(format!(
                    "STALE science: {}",
                    c["generation"].as_str().unwrap_or("unknown")
                )));
            }
        }
        for c in &mut rows.comparisons {
            if expired(c) {
                c["name"] = json!(text(format!(
                    "STALE science: {}",
                    c["name"].as_str().unwrap_or("unknown")
                )));
            }
        }
        for w in &mut rows.winners {
            if expired(w) {
                w["publication_status"] =
                    json!("STALE science; previous original publication evidence retained");
                w["remote_verified"] = Value::Null;
            }
        }
        if rows.corpus.as_ref().is_some_and(expired) {
            rows.notes
                .push("Current corpus withheld because its science sample is stale".into());
            rows.corpus = None;
        }
        for e in &mut rows.evidence {
            e["stale"] = json!(e["verified_at_unix_ms"]
                .as_u64()
                .is_none_or(|t| snapshot.observed_at_unix_ms.saturating_sub(t) >= STALE_MS));
        }
        snapshot.checkpoints = rows.checkpoints.into_iter().take(64).collect();
        snapshot.continuations = rows.continuations.into_iter().take(64).collect();
        snapshot.comparisons = rows.comparisons.into_iter().take(32).collect();
        snapshot.winners = rows.winners.into_iter().take(64).collect();
        snapshot.corpus = rows.corpus;
        snapshot.notes.extend(rows.notes);
        snapshot.notes.truncate(64);
        snapshot.source_errors.extend(rows.errors);
        snapshot.source_errors.truncate(32);
        snapshot.scientific_sources = rows.evidence.into_iter().take(64).collect();
        snapshot.scientific_verified_at_unix_ms = observed.at;
    }
}
