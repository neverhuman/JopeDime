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

    /// `/usage`: account, plan, current status and this machine's run statistics. The plan is
    /// read from the workspace in the background and arrives as one more line.
    pub(super) fn usage(&mut self) -> Vec<String> {
        use crate::terminal::{onboarding, usage};
        let runs = crate::terminal::runs::local_runs(&self.client).ok();
        let connection = crate::cloud::connection().ok();
        let state = onboarding::load(&self.home);
        let telemetry = if onboarding::telemetry_forced_off_from_env() {
            "Usage statistics off (DO_NOT_TRACK or JAIN_TELEMETRY=0)"
        } else if state.telemetry == Some(true) {
            "Usage statistics on"
        } else {
            "Usage statistics off"
        };
        let terms = match &state.terms_version {
            Some(version) if state.terms_current() => format!("{version} accepted"),
            Some(version) => format!("{version} accepted; newer terms pending"),
            None => "not accepted".into(),
        };
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|elapsed| elapsed.as_millis() as u64)
            .unwrap_or(0);
        let lines = usage::local_lines(
            runs.as_deref(),
            now_ms,
            connection
                .as_ref()
                .map(|c| (c.email.as_str(), c.origin.as_str())),
            self.snapshot.as_ref(),
            telemetry,
            &terms,
        );
        if let Some(connection) = connection {
            let sender = self.cloud_sender.clone();
            thread::spawn(move || {
                let _ = sender.send(vec![usage::fetch_plan_line(&connection)]);
            });
        }
        lines
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
