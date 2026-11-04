//! First-run dialog completion and the `/demo` command.
use super::*;

impl super::Session {
    /// Apply a finished first-run dialog: persist it, then log in or point at the demos.
    pub(super) fn finish_first_run(
        &mut self,
        outcome: crate::terminal::onboarding::Outcome,
        source: &str,
    ) -> Vec<String> {
        use crate::terminal::onboarding::{self, Choice, Outcome};
        let version = self
            .first_run
            .take()
            .map(|dialog| dialog.terms_version)
            .unwrap_or_else(|| onboarding::TERMS_VERSION.to_owned());
        self.screen.panel = None;
        let mut lines = Vec::new();
        if let Err(error) = onboarding::record(&self.home, outcome, &version, source) {
            lines.push(format!(
                "Could not save your choices ({error:#}); you may be asked again."
            ));
        }
        if let Outcome::Finished {
            terms_accepted,
            telemetry,
            choice,
        } = outcome
        {
            if terms_accepted {
                lines.push(format!(
                    "Terms of Service {version} accepted · {}",
                    onboarding::TERMS_URL
                ));
                // A signed-in account records the acceptance too (a new login sends it with
                // the device request instead).
                if crate::cloud::connection().is_ok() {
                    let sender = self.cloud_sender.clone();
                    let version = version.clone();
                    thread::spawn(move || {
                        let _ = sender.send(vec![crate::cloud::record_terms(&version)
                            .unwrap_or_else(|error| format!("Terms: {error:#}"))]);
                    });
                }
            }
            match telemetry {
                Some(true) => lines
                    .push("Usage statistics on · turn off with `jain cloud telemetry off`".into()),
                Some(false) => lines
                    .push("Usage statistics off · turn on with `jain cloud telemetry on`".into()),
                None => {}
            }
            match choice {
                Some(Choice::Login) => lines.extend(self.execute("/login")),
                Some(Choice::Demos) => lines.extend(crate::terminal::demos::listing()),
                None => {}
            }
        }
        lines
    }

    /// `/demo` lists the bundled demos; `/demo NAME` opens one with its target, task and
    /// objective preset and starts a fast run.
    pub(super) fn demo(&mut self, args: &[String]) -> anyhow::Result<Vec<String>> {
        use crate::terminal::demos;
        let Some(name) = args.first() else {
            return Ok(demos::listing());
        };
        anyhow::ensure!(args.len() == 1, "usage: /demo [NAME]");
        let Some(demo) = demos::find(name) else {
            let mut lines = vec![format!("There is no bundled demo named {name}.")];
            lines.extend(demos::listing());
            return Ok(lines);
        };
        let previous = std::mem::take(&mut self.draft);
        let presets = std::iter::once(format!("\\target {} --task {}", demo.target, demo.task))
            .chain(
                demo.objective
                    .map(|objective| format!("\\objective {objective}")),
            );
        for preset in presets {
            if let Err(error) = self.draft.apply(&preset) {
                self.draft = previous;
                anyhow::bail!(error);
            }
        }
        let mut lines = vec![format!(
            "Demo {} · {} · {} rows · target {} ({})",
            demo.name,
            demo.title,
            demo.rows(),
            demo.target,
            demo.task
        )];
        lines.extend(self.register(
            &format!("{}.csv", demo.name),
            Box::new(io::Cursor::new(demo.bytes)),
        )?);
        match self.start_run(&[]) {
            Ok(started) => lines.extend(started),
            // e.g. a build whose local engine runs regression only: keep the opened draft.
            Err(error) => lines.push(format!(
                "The dataset is open, but this engine could not start the {} demo: {error:#}. Try /demo regression.",
                demo.task
            )),
        }
        Ok(lines)
    }
}
