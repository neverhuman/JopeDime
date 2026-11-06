//! `/run` routing: local or the veox Cloud Workspace, just-in-time login, the per-dataset upload
//! consent, cloud progress in the transcript, limits in plain words, and the results sync of a
//! finished local run. The decision itself is `crate::cloud::route`; this file is the dialogue.
use super::*;
use crate::cloud::{
    cloud_run::{Refusal, Update},
    flow::{self, Prepared},
    route::{self, Force, Tier},
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

/// A cloud run waiting on a login or an answer.
#[derive(Clone, Debug)]
pub(crate) struct CloudRequest {
    /// The `/run` arguments without `--cloud`/`--local`, for running locally instead.
    pub args: Vec<String>,
    pub profile: &'static str,
    pub fallback: &'static str,
    pub target: Option<String>,
}

/// A yes/no question the next line answers.
pub(crate) enum Pending {
    OfferCloud(CloudRequest),
    Consent(CloudRequest, Prepared, String),
    RunLocal(Vec<String>),
    SyncResults(RunId, (String, String)),
}

/// What the cloud-run thread reports.
pub(crate) enum CloudMsg {
    Update(Update),
    Refused(Refusal),
    Error(String),
}

/// Session-side state for routing, kept together so `Session` gains one field.
#[derive(Default)]
pub(crate) struct Routing {
    pub source: Option<Source>,
    pub pending: Option<Pending>,
    pub after_login: Option<CloudRequest>,
    pub cloud_cancel: Option<Arc<AtomicBool>>,
    pub results_waiting: Option<(RunId, (String, String))>,
    pub results_handled: std::collections::BTreeSet<String>,
    /// The `/run` arguments of the cloud run in flight, to run it locally instead.
    pub cloud_args: Vec<String>,
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

impl super::Session {
    /// `/run [--local|--cloud] [--seed S] [--idempotency-key KEY]`.
    pub(super) fn route_run(&mut self, args: &[String]) -> anyhow::Result<Vec<String>> {
        let (force, rest) = force_from(args)?;
        let Some(spec) = self.draft.run_spec(true) else {
            return self.start_run(&rest);
        };
        let source = self.routing.source.clone().unwrap_or_default();
        let connection = crate::cloud::connection().ok();
        let needs_tier = force != Some(Force::Local)
            && !source.demo
            && (spec.effort != jope_api::Effort::Fast || source.rows > route::LOCAL_ROW_THRESHOLD);
        let tier = match (&connection, needs_tier) {
            (Some(c), true) => flow::tier(c).ok().flatten(),
            _ => None,
        };
        let decision = route::decide(route::Input {
            effort: spec.effort,
            rows: self.routing.source.as_ref().map(|s| s.rows),
            demo: source.demo,
            force,
            tier,
        });
        let request = CloudRequest {
            args: rest.clone(),
            profile: route::profile(spec.effort),
            fallback: decision.cloud_fallback(),
            target: Some(spec.target.clone()),
        };
        let mut lines = vec![decision.line.clone()];
        match decision.target {
            route::Target::Local if decision.offer_cloud => {
                lines.push(route::OFFER_CLOUD.into());
                self.routing.pending = Some(Pending::OfferCloud(request));
            }
            route::Target::Local => lines.extend(self.start_run(&rest)?),
            route::Target::Cloud => lines.extend(self.begin_cloud(request)?),
        }
        Ok(lines)
    }

    /// Log in if needed, then continue the same run.
    pub(super) fn begin_cloud(&mut self, request: CloudRequest) -> anyhow::Result<Vec<String>> {
        let source = self.routing.source.clone().unwrap_or_default();
        if source.path.is_none() {
            let mut lines = vec![
                "This table was not opened from a file on this machine, so it runs locally.".into(),
            ];
            lines.extend(self.start_run(&request.args)?);
            return Ok(lines);
        }
        if crate::cloud::connection().is_ok() {
            return self.continue_cloud(request);
        }
        if let Some(cancel) = self.login_cancel.take() {
            cancel.store(true, Ordering::SeqCst);
        }
        let cancel = Arc::new(AtomicBool::new(false));
        crate::cloud::spawn_login(
            crate::cloud::default_workspace(),
            self.cloud_sender.clone(),
            Arc::clone(&cancel),
            Some(self.login_done_sender.clone()),
        );
        self.login_cancel = Some(cancel);
        self.routing.after_login = Some(request);
        Ok(vec![flow::NEEDS_LOGIN.into()])
    }

    /// Signed in: check what the plan allows locally, then ask (once per dataset) and upload.
    pub(super) fn continue_cloud(&mut self, request: CloudRequest) -> anyhow::Result<Vec<String>> {
        let c = crate::cloud::connection()?;
        let source = self.routing.source.clone().unwrap_or_default();
        let path = source
            .path
            .clone()
            .ok_or_else(|| anyhow::anyhow!("no file to upload"))?;
        // Free's row ceiling is known: say so before uploading anything.
        if source.rows > route::FREE_MAX_DATA_ROWS
            && flow::tier(&c).ok().flatten() == Some(Tier::Free)
        {
            let limit = route::Limit {
                status: 400,
                limit: "size".into(),
                used: Some(source.rows),
                allowed: Some(route::FREE_MAX_DATA_ROWS),
                resets_at: None,
                upgrade_url: route::UPGRADE_URL.into(),
                message: String::new(),
            };
            self.routing.pending = Some(Pending::RunLocal(request.args));
            return Ok(vec![limit.render()]);
        }
        let prepared = Prepared::read(&path, source.rows, source.columns)?;
        let key = flow::consent_key(&c.origin, &c.account_id, &prepared.sha256);
        if flow::consented(
            &crate::terminal::onboarding::load(&self.home),
            &key,
            &prepared,
        ) {
            return Ok(self.spawn_cloud(request, prepared));
        }
        let question = prepared.question();
        self.routing.pending = Some(Pending::Consent(request, prepared, key));
        Ok(vec![question])
    }

    fn spawn_cloud(&mut self, request: CloudRequest, prepared: Prepared) -> Vec<String> {
        let Ok(c) = crate::cloud::connection() else {
            return vec!["Not signed in; nothing was uploaded.".into()];
        };
        let cancel = Arc::new(AtomicBool::new(false));
        self.routing.cloud_cancel = Some(Arc::clone(&cancel));
        self.routing.cloud_args = request.args.clone();
        let sender = self.cloud_run_sender.clone();
        let name = prepared.file_name.clone();
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
                        "The upload did not reach your workspace ({error:#})."
                    )));
                }
            }
        });
        vec![format!("Uploading {name} to your veox Cloud Workspace…")]
    }

    /// Answer a pending question. `None`: the line was not an answer and runs as a command.
    pub(super) fn answer(&mut self, input: &str) -> Option<anyhow::Result<Vec<String>>> {
        let pending = self.routing.pending.take()?;
        let Some(yes) = route::answer(input) else {
            self.routing.pending = None;
            return None;
        };
        Some(match pending {
            Pending::OfferCloud(request) if yes => self.begin_cloud(request),
            Pending::OfferCloud(request) => self.start_run(&request.args),
            Pending::Consent(request, prepared, key) if yes => {
                if let Err(error) = flow::remember_consent(&self.home, &key, &prepared) {
                    return Some(Err(error));
                }
                Ok(self.spawn_cloud(request, prepared))
            }
            Pending::Consent(request, _, _) => {
                let mut lines = vec!["Nothing was uploaded.".into()];
                lines.push(route::Limit::local_question().into());
                self.routing.pending = Some(Pending::RunLocal(request.args));
                Ok(lines)
            }
            Pending::RunLocal(args) if yes => {
                let mut lines = vec!["Running locally on this machine.".to_owned()];
                match self.start_run(&args) {
                    Ok(more) => lines.extend(more),
                    Err(error) => return Some(Err(error)),
                }
                Ok(lines)
            }
            Pending::RunLocal(_) => Ok(vec!["Not started.".into()]),
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
                vec![format!("Cloud run started · {url}")]
            }
            CloudMsg::Update(Update::Thinking(text)) => {
                crate::terminal::cards::thinking_card(&text, width, theme)
            }
            CloudMsg::Update(Update::Finished(lines)) => {
                self.routing.cloud_cancel = None;
                lines
            }
            CloudMsg::Update(Update::Error(line)) | CloudMsg::Error(line) => {
                self.routing.cloud_cancel = None;
                vec![line]
            }
            CloudMsg::Refused(Refusal::Limit(limit)) => {
                self.routing.cloud_cancel = None;
                self.routing.pending = Some(Pending::RunLocal(self.last_run_args()));
                vec![limit.render()]
            }
            CloudMsg::Refused(Refusal::Other(message)) => {
                self.routing.cloud_cancel = None;
                self.routing.pending = Some(Pending::RunLocal(self.last_run_args()));
                vec![message, route::Limit::local_question().into()]
            }
        }
    }

    fn last_run_args(&self) -> Vec<String> {
        self.routing.cloud_args.clone()
    }

    /// A finished login: continue the run that needed it.
    pub(super) fn login_finished(&mut self, ok: bool) -> Vec<String> {
        let Some(request) = self.routing.after_login.take() else {
            return Vec::new();
        };
        if ok {
            match self.continue_cloud(request) {
                Ok(lines) => lines,
                Err(error) => vec![format!("error: {error:#}")],
            }
        } else {
            self.routing.pending = Some(Pending::RunLocal(request.args));
            vec![
                "The login did not finish, so nothing was uploaded.".into(),
                route::Limit::local_question().into(),
            ]
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
