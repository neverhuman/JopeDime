use super::*;

impl super::Session {
    pub(super) fn draw(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    ) -> io::Result<()> {
        let width = usize::from(terminal.size()?.width);
        let (status, activity) = self.status_lines(width);
        let completed = terminal.draw(|frame| {
            self.screen
                .draw(frame, &self.composer, &self.version, &status, &activity);
        })?;
        if self.links {
            crate::terminal::links::overlay(
                &mut io::stdout(),
                &self.screen.links,
                completed.buffer,
            )?;
        }
        Ok(())
    }

    pub(super) fn refresh_runs(&mut self) {
        self.screen
            .browser
            .get_or_insert_with(Default::default)
            .notice = "Loading runs...".into();
        let client = self.client.clone();
        let sender = self.run_list_sender.clone();
        thread::spawn(move || {
            let _ = sender.send(crate::cloud::run_entries(&client));
        });
    }

    pub(super) fn background_sync(&mut self, runs: Vec<RunId>, owner: Option<(String, String)>) {
        if self.cloud_busy.swap(true, Ordering::SeqCst) {
            return;
        }
        let busy = Arc::clone(&self.cloud_busy);
        let done = self.synced_terminal_sender.clone();
        let client = self.client.clone();
        let sender = self.background_sender.clone();
        thread::spawn(move || {
            let result = (|| -> anyhow::Result<Vec<String>> {
                let version = client.system_info()?.version;
                let mut lines = Vec::new();
                for run in runs {
                    let snapshot = client.run_status(&run)?;
                    lines.push(crate::cloud::sync_snapshot_for(
                        &snapshot,
                        &version,
                        owner.as_ref(),
                    )?);
                    if matches!(
                        snapshot.state,
                        RunState::Completed | RunState::Cancelled | RunState::Failed
                    ) {
                        // A finished run's leaderboard and profile, once; a failure is reported
                        // and retried by the next sync rather than stopping this one.
                        lines.push(
                            crate::cloud::sync_events(&client, &snapshot)
                                .unwrap_or_else(|error| format!("Run events: {error:#}")),
                        );
                        let _ = done.send(run);
                    }
                }
                Ok(lines)
            })();
            busy.store(false, Ordering::SeqCst);
            let _ = sender.send(BackgroundResult::Sync(result));
        });
    }

    pub(super) fn tick_sync(&mut self) {
        while let Ok(run) = self.synced_terminal.try_recv() {
            if self
                .auto_sync
                .as_ref()
                .is_some_and(|(current, _)| current == &run)
            {
                self.auto_sync = None;
            }
        }
        if self.last_sync.elapsed() >= Duration::from_secs(5) {
            if let Some((run, owner)) = self.auto_sync.clone() {
                let same =
                    crate::cloud::connection().is_ok_and(|c| (c.origin, c.account_id) == owner);
                if !same {
                    self.auto_sync = None;
                } else if !self.cloud_busy.load(Ordering::SeqCst) {
                    self.background_sync(vec![run], Some(owner));
                }
            }
            self.last_sync = Instant::now();
        }
    }

    pub(super) fn screen_lines(&self) -> Vec<String> {
        self.screen.transcript()
    }

    pub(super) fn status_lines(&self, width: usize) -> (String, String) {
        if let Some(snapshot) = &self.snapshot {
            crate::terminal::status::render(snapshot, &[], width)
        } else {
            (
                self.message
                    .clone()
                    .unwrap_or_else(|| crate::terminal::screen::IDLE_STATUS.into()),
                self.activity.clone(),
            )
        }
    }
}
