use super::*;

impl super::Session {
    pub(super) fn attach(&mut self, run: RunId, after: Option<DecimalU64>) -> anyhow::Result<()> {
        self.snapshot = Some(self.client.run_status(&run)?);
        let client = self.client.clone();
        let sender = self.event_sender.clone();
        thread::Builder::new()
            .name("jope-terminal-events".into())
            .spawn(move || {
                let mut cursor = after;
                loop {
                    if let Ok(mut stream) = client.follow_events(run.clone(), cursor) {
                        loop {
                            match stream.next() {
                                Some(Ok(StreamEvent::Durable(event))) => {
                                    cursor = Some(event.sequence);
                                    if sender.send(*event).is_err() {
                                        return;
                                    }
                                }
                                Some(Ok(StreamEvent::Telemetry(_))) => {}
                                Some(Err(_)) | None => {
                                    cursor = stream.last_event_id().or(cursor);
                                    break;
                                }
                            }
                        }
                    }
                    thread::sleep(Duration::from_millis(100));
                }
            })?;
        Ok(())
    }

    pub(super) fn latest_active_run(&self) -> anyhow::Result<RunId> {
        let value = serde_json::to_value(self.client.runs()?)?;
        let runs = value
            .get("runs")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| anyhow::anyhow!("engine did not return a run list"))?;
        for run in runs.iter().rev() {
            let Some(id) = run
                .get("run_id")
                .or_else(|| run.get("id"))
                .and_then(serde_json::Value::as_str)
            else {
                continue;
            };
            let Ok(id) = RunId::from_str(id) else {
                continue;
            };
            if let Ok(snapshot) = self.client.run_status(&id) {
                if matches!(
                    snapshot.state,
                    RunState::Running | RunState::Pausing | RunState::Paused
                ) {
                    return Ok(id);
                }
            }
        }
        anyhow::bail!("there is no active run to attach")
    }

    pub(super) fn control(&mut self, action: &str) -> anyhow::Result<()> {
        let current = self
            .snapshot
            .as_ref()
            .map(|snapshot| snapshot.run_id.clone())
            .ok_or_else(|| anyhow::anyhow!("there is no active run"))?;
        let snapshot = self.client.run_status(&current)?;
        let key = format!("tui_{}", ulid::Ulid::new());
        let request = Request::new(8, 0, &[("id", &current.to_string())])?
            .key(Some(&key))?
            .if_match(snapshot.revision)
            .json(&serde_json::json!({"action": action}))?;
        self.snapshot = Some(self.client.execute(request)?.json()?);
        Ok(())
    }

    pub(super) fn handle_event(&mut self, event: PublicEvent, width: usize) -> Vec<String> {
        update_snapshot_from_event(&mut self.snapshot, &event);
        crate::terminal::cards::render_transcript(
            &[event],
            width,
            crate::terminal::Theme::from_env(crate::terminal::ThemeMode::Auto),
            crate::terminal::symbols_from_env(),
        )
    }

    pub(super) fn leave(&self) {
        if let Some(snapshot) = &self.snapshot {
            eprintln!(
                "run continues in the background; reconnect with `jopedime attach {}`",
                snapshot.run_id
            );
        }
    }
}
