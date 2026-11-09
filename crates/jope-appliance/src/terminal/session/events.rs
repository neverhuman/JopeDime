use super::*;

impl super::Session {
    pub(super) fn attach(&mut self, run: RunId, after: Option<DecimalU64>) -> anyhow::Result<()> {
        self.snapshot = Some(self.client.run_status(&run)?);
        self.last_refresh = Instant::now();
        // The status bar's clock counts from here while the engine reports no elapsed time yet.
        if self.run_clock.as_ref().map(|(id, _)| id) != Some(&run) {
            self.run_clock = Some((run.clone(), Instant::now()));
        }
        // Lines already shown for this run are not replayed (#2143): resume after them.
        let after = match (after, self.shown.get(&run.to_string())) {
            (Some(after), Some(shown)) => Some(DecimalU64(after.0.max(shown.0))),
            (None, Some(shown)) => Some(*shown),
            (after, None) => after,
        };
        if let Some(previous) = self.follow_cancel.take() {
            previous.store(true, Ordering::SeqCst);
        }
        let cancel = Arc::new(AtomicBool::new(false));
        self.follow_cancel = Some(Arc::clone(&cancel));
        let client = self.client.clone();
        let sender = self.event_sender.clone();
        thread::Builder::new()
            .name("jope-terminal-events".into())
            .spawn(move || {
                let mut cursor = after;
                loop {
                    if cancel.load(Ordering::SeqCst) {
                        return;
                    }
                    if let Ok(mut stream) = client.follow_events(run.clone(), cursor) {
                        loop {
                            match stream.next() {
                                Some(Ok(StreamEvent::Durable(event))) => {
                                    cursor = Some(event.sequence);
                                    if cancel.load(Ordering::SeqCst) || sender.send(*event).is_err()
                                    {
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
        let run = event.run_id.to_string();
        if self
            .shown
            .get(&run)
            .is_some_and(|shown| event.sequence.0 <= shown.0)
        {
            return Vec::new();
        }
        self.shown.insert(run.clone(), event.sequence);
        update_snapshot_from_event(&mut self.snapshot, &event);
        let finished = matches!(event.event, EventBody::RunFinished(_));
        let mut lines = crate::terminal::cards::render_transcript(
            &[event],
            width,
            crate::terminal::Theme::from_env(crate::terminal::ThemeMode::Auto),
            crate::terminal::symbols_from_env(),
        );
        if finished {
            if let Some(snapshot) = self
                .snapshot
                .as_ref()
                .filter(|snapshot| snapshot.run_id.to_string() == run)
            {
                lines.extend(crate::terminal::status::budget_stop_line(snapshot));
                lines.push(crate::terminal::status::next_steps(
                    &run,
                    crate::cloud::connection().is_ok(),
                    true,
                ));
            }
        }
        lines
    }

    /// Re-read the attached run about once a second until it ends, so the status bar follows
    /// it even between events and shows the finished state (#2143). Returns whether it changed.
    pub(super) fn refresh_snapshot(&mut self) -> bool {
        let Some(current) = self.snapshot.as_ref() else {
            return false;
        };
        if crate::terminal::status::is_terminal(current.state)
            || self.last_refresh.elapsed() < Duration::from_secs(1)
        {
            return false;
        }
        self.last_refresh = Instant::now();
        match self.client.run_status(&current.run_id) {
            Ok(fresh) if Some(&fresh) != self.snapshot.as_ref() => {
                self.snapshot = Some(fresh);
                true
            }
            _ => false,
        }
    }

    pub(super) fn leave(&self) {
        if let Some(snapshot) = &self.snapshot {
            if !crate::terminal::status::is_terminal(snapshot.state) {
                eprintln!(
                    "run continues in the background; reconnect with `jain attach {}`",
                    snapshot.run_id
                );
            }
        }
        eprintln!("{ENGINE_STAYS}");
    }
}

/// Printed on quit: the engine the TUI started outlives it (#2148).
pub(super) const ENGINE_STAYS: &str = "The jAIn background engine stays up for about 30 minutes after its last job, then exits; `jain serve --stop` stops it now.";

#[cfg(test)]
mod tests {
    #[test]
    fn the_quit_line_names_the_idle_time_and_the_stop_command() {
        assert!(super::ENGINE_STAYS.contains("30 minutes"));
        assert!(super::ENGINE_STAYS.contains("`jain serve --stop`"));
    }
}
