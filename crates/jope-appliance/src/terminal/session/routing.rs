//! `/run` routing (owner rule 2026-11-06): a bundled demo runs on this machine; the user's own
//! table trains in the veox Cloud, after the login and admission gate and the per-table upload
//! consent. Cloud progress in the transcript, limits in plain words, and the results sync of a
//! finished local demo run. The decision itself is `crate::cloud::route`; this file is the
//! dialogue.
use super::*;
use crate::cloud::{
    cloud_run::{CloudRefusal, Update},
    flow::{self, Prepared},
    route::{self, AccountTier, Force},
};

/// The table the draft was opened from.
#[derive(Clone, Debug, Default)]
pub(crate) struct Source {
    /// The file on disk, when there is one (not for bundled demos).
    pub path: Option<PathBuf>,
    pub demo: bool,
    pub rows: u64,
    pub columns: u64,
    /// The name it was registered under and its profile, remembered with a started run.
    pub name: String,
    pub profile: Option<jope_api::DatasetProfile>,
}

/// A cloud run waiting on an answer.
#[derive(Clone, Debug)]
pub(crate) struct CloudRequest {
    pub profile: &'static str,
    pub fallback: &'static str,
    pub target: Option<String>,
}

/// A yes/no question the next line answers.
pub(crate) enum Pending {
    Consent(CloudRequest, Prepared, String),
    SyncResults(RunId, (String, String)),
}

/// What the cloud-run thread reports.
pub(crate) enum CloudMsg {
    Update(Update),
    Refused(CloudRefusal),
    Error(String),
}

/// Session-side state for routing, kept together so `Session` gains one field.
#[derive(Default)]
pub(crate) struct Routing {
    pub source: Option<Source>,
    pub pending: Option<Pending>,
    pub cloud_cancel: Option<Arc<AtomicBool>>,
    /// The workspace link of the cloud run being followed.
    pub cloud_url: Option<String>,
    pub results_waiting: Option<(RunId, (String, String))>,
    pub results_handled: std::collections::BTreeSet<String>,
}

/// Split `--local` / `--cloud` off the `/run` arguments.
pub(crate) fn force_from(args: &[String]) -> anyhow::Result<(Option<Force>, Vec<String>)> {
    let mut force = None;
    let mut rest = Vec::new();
    for arg in args {
        let next = match arg.as_str() {
            "--local" => Some(Force::Local),
            "--cloud" => Some(Force::Cloud),
            _ => {
                rest.push(arg.clone());
                continue;
            }
        };
        anyhow::ensure!(
            force.is_none() || force == next,
            "use either --local or --cloud, not both"
        );
        force = next;
    }
    Ok((force, rest))
}

/// After `/open` of the user's own table, where it will train.
pub(crate) const OPENED_OWN: &str = "Type /run to train it in the veox Cloud (jAIn asks before uploading), or /auto for the defaults; /target COLUMN predicts another column.";

impl super::Session {
    /// `/run [--local|--cloud] [--seed S] [--idempotency-key KEY]`.
    pub(super) fn route_run(&mut self, args: &[String]) -> anyhow::Result<Vec<String>> {
        let (force, rest) = force_from(args)?;
        let Some(spec) = self.draft.run_spec(true) else {
            return self.start_run(&rest);
        };
        let source = self.routing.source.clone().unwrap_or_default();
        let decision = route::decide(route::Input {
            effort: spec.effort,
            rows: self.routing.source.as_ref().map(|s| s.rows),
            demo: source.demo,
            force,
            cloud_first: route::cloud_first(),
        });
        let request = CloudRequest {
            profile: route::profile(spec.effort),
            fallback: decision.cloud_fallback(),
            target: Some(spec.target.clone()),
        };
        match decision.target {
            route::Target::Refused => Ok(vec![decision.line]),
            route::Target::Local => {
                let mut lines = vec![decision.line];
                lines.extend(self.start_run(&rest)?);
                Ok(lines)
            }
            route::Target::Cloud => self.begin_cloud(request, decision.line),
        }
    }

    /// The own-data gate, then `line` (where it trains) and the upload question (once per
    /// table) or the run itself.
    pub(super) fn begin_cloud(
        &mut self,
        request: CloudRequest,
        line: String,
    ) -> anyhow::Result<Vec<String>> {
        let source = self.routing.source.clone().unwrap_or_default();
        let Some(path) = source.path.clone() else {
            return Ok(vec![
                "The bundled demos run on this machine: type /run without --cloud. /run --cloud uploads only your own files (/open FILE.csv)."
                    .into(),
            ]);
        };
        let (c, tier) = match flow::admitted() {
            Ok(admitted) => admitted,
            Err(gate) => return Ok(vec![gate.message(true)]),
        };
        let mut lines = vec![line];
        // Free's row ceiling is known: say so before uploading anything.
        if source.rows > route::FREE_MAX_DATA_ROWS && tier == Some(AccountTier::Free) {
            let limit = route::Limit {
                status: 400,
                limit: "size".into(),
                used: Some(source.rows),
                allowed: Some(route::FREE_MAX_DATA_ROWS),
                resets_at: None,
                upgrade_url: route::UPGRADE_URL.into(),
                message: String::new(),
            };
            lines.push(limit.render());
            return Ok(lines);
        }
        let mut request = request;
        if tier == Some(AccountTier::Free) && request.profile != "fast" {
            request.profile = "fast";
            lines.push(route::FREE_RUNS_FAST.into());
        }
        let prepared = Prepared::read(&path, source.rows, source.columns)?;
        let key = flow::consent_key(&c.origin, &c.account_id, &prepared.sha256);
        if flow::consented(
            &crate::terminal::onboarding::load(&self.home),
            &key,
            &prepared,
        ) {
            lines.extend(self.spawn_cloud(request, prepared));
            return Ok(lines);
        }
        lines.push(prepared.question());
        self.routing.pending = Some(Pending::Consent(request, prepared, key));
        Ok(lines)
    }

    fn spawn_cloud(&mut self, request: CloudRequest, prepared: Prepared) -> Vec<String> {
        let Ok(c) = crate::cloud::connection() else {
            return vec!["Not signed in; nothing was uploaded. Type /login first.".into()];
        };
        if let Some(previous) = self.routing.cloud_cancel.take() {
            previous.store(true, Ordering::SeqCst);
        }
        let cancel = Arc::new(AtomicBool::new(false));
        self.routing.cloud_cancel = Some(Arc::clone(&cancel));
        self.routing.cloud_url = None;
        // The status bar follows the cloud run, not the last local one.
        self.snapshot = None;
        self.message = Some(format!("veox Cloud · uploading {}", prepared.file_name));
        let sender = self.cloud_run_sender.clone();
        let name = prepared.file_name.clone();
        let size = route::size(prepared.bytes.len() as u64);
        thread::spawn(move || {
            let result = flow::submit_and_follow(
                &c,
                &prepared,
                request.profile,
                request.fallback,
                request.target.as_deref(),
                Duration::from_secs(2),
                &cancel,
                &mut |update| {
                    let _ = sender.send(CloudMsg::Update(update));
                },
            );
            match result {
                Ok(Ok(())) => {}
                Ok(Err(refusal)) => {
                    let _ = sender.send(CloudMsg::Refused(refusal));
                }
                Err(error) => {
                    let _ = sender.send(CloudMsg::Error(format!(
                        "The upload did not reach your workspace ({error:#}); no run started."
                    )));
                }
            }
        });
        vec![format!(
            "Uploading {name} ({size}) to your veox Cloud Workspace…"
        )]
    }

    /// Answer a pending question. `None`: the line was not an answer and runs as a command.
    pub(super) fn answer(&mut self, input: &str) -> Option<anyhow::Result<Vec<String>>> {
        let pending = self.routing.pending.take()?;
        let answer = match &pending {
            Pending::Consent(..) => route::consent_answer(input),
            Pending::SyncResults(..) => route::answer(input),
        };
        let Some(yes) = answer else {
            self.routing.pending = None;
            return None;
        };
        Some(match pending {
            Pending::Consent(request, prepared, key) if yes => {
                if let Err(error) = flow::remember_consent(&self.home, &key, &prepared) {
                    return Some(Err(error));
                }
                Ok(self.spawn_cloud(request, prepared))
            }
            Pending::Consent(..) => Ok(vec![
                "Nothing was uploaded, and no run started. Your own data trains only in the veox Cloud; the bundled demos run here: /demo".into(),
            ]),
            Pending::SyncResults(run, owner) => {
                let home = self.home.clone();
                if let Err(error) =
                    crate::terminal::onboarding::update(&home, |s| s.results_sync = Some(yes))
                {
                    return Some(Err(error));
                }
                if yes {
                    self.spawn_results(run, owner);
                    Ok(vec!["Results sync on for this and later local runs · turn off with /sync off (or `jain cloud results off`)".into()])
                } else {
                    Ok(vec![
                        "Results stay on this machine for this and later local runs · turn on with /sync on (or `jain cloud results on`)".into()
                    ])
                }
            }
        })
    }

    pub(super) fn spawn_results(&mut self, run: RunId, owner: (String, String)) {
        self.cloud_busy.store(true, Ordering::SeqCst);
        let busy = Arc::clone(&self.cloud_busy);
        let client = self.client.clone();
        let sender = self.background_sender.clone();
        thread::spawn(move || {
            let result = crate::cloud::sync_results(&client, &run, Some(&owner));
            busy.store(false, Ordering::SeqCst);
            let _ = sender.send(BackgroundResult::Sync(result));
        });
    }

    /// A finished local run, while logged in: sync its results, or ask the first time.
    pub(super) fn tick_results(&mut self) -> Vec<String> {
        if self.routing.pending.is_some() || self.cloud_busy.load(Ordering::SeqCst) {
            return Vec::new();
        }
        let Some((run, owner)) = self.routing.results_waiting.take() else {
            return Vec::new();
        };
        if !crate::cloud::connection()
            .is_ok_and(|c| (c.origin.clone(), c.account_id.clone()) == owner && c.may_publish)
        {
            return Vec::new();
        }
        if !self.routing.results_handled.insert(run.to_string()) {
            return Vec::new();
        }
        match crate::terminal::onboarding::load(&self.home).results_sync {
            Some(true) => {
                self.spawn_results(run, owner);
                vec!["Results sync is on: sending this run's model and code to your workspace… (turn off with /sync off)".into()]
            }
            Some(false) => Vec::new(),
            None => {
                self.routing.pending = Some(Pending::SyncResults(run, owner));
                vec![crate::cloud::RESULTS_QUESTION.into()]
            }
        }
    }

    /// Lines for one message from the cloud-run thread.
    pub(super) fn cloud_lines(&mut self, message: CloudMsg, width: usize) -> Vec<String> {
        let theme = crate::terminal::Theme::from_env(crate::terminal::ThemeMode::Auto);
        match message {
            CloudMsg::Update(Update::Started { url, .. }) => {
                self.message = Some(format!("veox Cloud · running · {url}"));
                self.routing.cloud_url = Some(url.clone());
                vec![format!("Cloud run started · {url}")]
            }
            CloudMsg::Update(Update::Thinking(text)) => {
                crate::terminal::cards::thinking_card(&text, width, theme)
            }
            CloudMsg::Update(Update::Finished(mut lines)) => {
                self.routing.cloud_cancel = None;
                let url = self.routing.cloud_url.take().unwrap_or_default();
                let done = lines.first().is_some_and(|l| l.starts_with('✓'));
                self.message = Some(if done {
                    "veox Cloud · run finished".to_owned()
                } else {
                    "veox Cloud · run ended".to_owned()
                });
                if done && !url.is_empty() {
                    lines.push(crate::cloud::cloud_run::next_line(&url, true));
                }
                lines
            }
            CloudMsg::Update(Update::Error(line)) | CloudMsg::Error(line) => {
                self.routing.cloud_cancel = None;
                self.message = Some("veox Cloud · not following a run".into());
                vec![line]
            }
            CloudMsg::Refused(CloudRefusal::Limit(limit)) => {
                self.routing.cloud_cancel = None;
                self.message = None;
                vec![limit.render()]
            }
            CloudMsg::Refused(CloudRefusal::Other(message)) => {
                self.routing.cloud_cancel = None;
                self.message = None;
                vec![format!("{message} No run started.")]
            }
        }
    }

    /// A finished `/login`: say what now opens up.
    pub(super) fn login_finished(&mut self, ok: bool) -> Vec<String> {
        if !ok {
            return Vec::new();
        }
        match &self.routing.source {
            Some(source) if !source.demo && source.path.is_some() => vec![
                "Signed in: type /run to train the opened table in the veox Cloud (jAIn asks before uploading).".into(),
            ],
            _ => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_overrides_are_split_from_the_other_arguments() {
        let args: Vec<String> = ["--cloud", "--seed", "3"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(
            force_from(&args).unwrap(),
            (
                Some(Force::Cloud),
                vec!["--seed".to_owned(), "3".to_owned()]
            )
        );
        assert_eq!(
            force_from(&["--local".into()]).unwrap().0,
            Some(Force::Local)
        );
        assert_eq!(force_from(&[]).unwrap().0, None);
        assert!(force_from(&["--local".into(), "--cloud".into()]).is_err());
    }
}
