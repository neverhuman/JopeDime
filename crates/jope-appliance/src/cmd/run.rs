use clap::Args;
use jope_api::{
    decimal::DecimalU64, ids::DatasetId, ExportKind, ExportRequest, Language, Refit, RunSnapshot,
    RunSpec, RunState,
};
use jope_client::{transport::Request, RunResult, Transport};
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::{io::Write, path::PathBuf, str::FromStr};

#[cfg(not(windows))]
type RunInterrupts = super::inproc::Interrupts;

// Keep the existing Windows request-boundary behavior, with a scoped registration.
// Two-signal pause/cancel qualification remains with the platform implementation.
#[cfg(windows)]
struct RunInterrupts {
    flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
    registration: signal_hook::SigId,
}

#[cfg(windows)]
impl RunInterrupts {
    fn install() -> std::io::Result<Self> {
        let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let registration =
            signal_hook::flag::register(signal_hook::consts::SIGINT, std::sync::Arc::clone(&flag))?;
        Ok(Self { flag, registration })
    }

    fn requested(&self) -> bool {
        self.flag.load(std::sync::atomic::Ordering::SeqCst)
    }
}

#[cfg(windows)]
impl Drop for RunInterrupts {
    fn drop(&mut self) {
        signal_hook::low_level::unregister(self.registration);
    }
}

#[derive(Args, Debug)]
pub struct RunArgs {
    /// Reuse the original key when retrying the same run request.
    #[arg(long)]
    pub idempotency_key: Option<String>,
    #[arg(long)]
    pub dataset: String,
    #[arg(long)]
    pub target: String,
    #[arg(long)]
    pub task: String,
    #[arg(long = "ignore-column")]
    pub ignore_columns: Vec<String>,
    #[arg(long)]
    pub test_dataset: Option<String>,
    #[arg(long)]
    pub split: String,
    #[arg(long)]
    pub objective: String,
    #[arg(long)]
    pub effort: String,
    #[arg(long)]
    pub budget: String,
    #[arg(long)]
    pub landmarks: String,
    #[arg(long)]
    pub guidance: String,
    #[arg(long = "language")]
    pub languages: Vec<String>,
    #[arg(long)]
    pub require_all_languages: bool,
    #[arg(long)]
    pub device: String,
    #[arg(long)]
    pub seed: String,
    #[arg(long, conflicts_with_all = ["detach", "out", "events"])]
    pub dry_run: bool,
    #[arg(long)]
    pub plan_digest: Option<String>,
    #[arg(long, conflicts_with_all = ["dry_run", "events", "out"])]
    pub detach: bool,
    #[arg(long)]
    pub out: Option<PathBuf>,
    #[arg(long)]
    pub json: bool,
    #[arg(long, value_parser = ["ndjson"])]
    pub events: Option<String>,
    #[arg(long, default_value_t = 30)]
    pub wait: u64,
    #[arg(long)]
    pub after_event: Option<String>,
    /// Internal M1 probe using a prepared FIT-only service plan.
    #[arg(
        long,
        hide = true,
        requires = "json",
        conflicts_with_all = ["dry_run", "detach", "out", "events", "plan_digest", "after_event", "wait"]
    )]
    pub m1_plan: Option<PathBuf>,
}

impl RunArgs {
    pub fn run_spec(&self) -> anyhow::Result<RunSpec> {
        let dataset = DatasetId::from_str(&self.dataset)
            .map_err(|_| anyhow::anyhow!("invalid dataset identifier"))?;
        let test_dataset = self
            .test_dataset
            .as_deref()
            .map(DatasetId::from_str)
            .transpose()
            .map_err(|_| anyhow::anyhow!("invalid test dataset identifier"))?;
        Ok(RunSpec {
            dataset,
            target: self.target.clone(),
            task: parse_value(&self.task)?,
            ignore_columns: self.ignore_columns.clone(),
            test_dataset,
            split: parse_json(&self.split)?,
            objective: parse_value(&self.objective)?,
            effort: parse_value(&self.effort)?,
            budget: parse_json(&self.budget)?,
            landmarks: parse_value(&self.landmarks)?,
            guidance: parse_value(&self.guidance)?,
            languages: self
                .languages
                .iter()
                .map(|value| parse_value(value))
                .collect::<anyhow::Result<Vec<Language>>>()?,
            require_all_languages: self.require_all_languages,
            device: self.device.clone(),
            seed: serde_json::from_value(Value::String(self.seed.clone()))?,
            dry_run: self.dry_run,
            plan_digest: self.plan_digest.clone(),
        })
    }
}

fn parse_json<T: DeserializeOwned>(raw: &str) -> anyhow::Result<T> {
    serde_json::from_str(raw).map_err(Into::into)
}

fn parse_value<T: DeserializeOwned>(raw: &str) -> anyhow::Result<T> {
    let value = serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_owned()));
    serde_json::from_value(value).map_err(Into::into)
}

/// Keep the hidden local probe ahead of the ordinary engine-client preflight.
pub fn dispatch(args: RunArgs) -> anyhow::Result<()> {
    if args.m1_plan.is_some() {
        return run_m1(&args);
    }
    use jope_client::discover::{self, Engine, Mode, Options};
    let selection = discover::discover(&Options {
        home: jope_paths::jopedime_home(),
        executable: std::env::current_exe()?,
        endpoint: None,
        mode: Mode::Batch,
        engine_available: cfg!(feature = "engine"),
    })?;
    match selection {
        Engine::Background(background) => run(args, &background.client),
        // An explicit endpoint is authoritative. Never substitute a daemon or
        // legacy server when the selected endpoint cannot be authenticated.
        Engine::Endpoint(_) => Err(DispatchUnavailable::EndpointCredentials.into()),
        Engine::InProcess => {
            #[cfg(feature = "engine")]
            {
                let token = format!("{}{}", ulid::Ulid::new(), ulid::Ulid::new());
                // The local authority never binds a socket. Derive a valid bearer
                // from fresh entropy without exposing it on the command line.
                let token = jope_runtime::sha256_hex(token.as_bytes());
                let transport = jope_client::InProcess::local(31234, &token)?;
                run(args, &transport)
            }
            #[cfg(not(feature = "engine"))]
            {
                Err(DispatchUnavailable::LocalHandlers.into())
            }
        }
    }
}

#[derive(Debug)]
enum DispatchUnavailable {
    EndpointCredentials,
    #[cfg(not(feature = "engine"))]
    LocalHandlers,
}

impl std::fmt::Display for DispatchUnavailable {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::EndpointCredentials => {
                "explicit run endpoints require supported bearer credentials"
            }
            #[cfg(not(feature = "engine"))]
            Self::LocalHandlers => "in-process batch runs are unavailable in this build",
        })
    }
}

impl std::error::Error for DispatchUnavailable {}

pub fn dispatch_exit_code(error: &anyhow::Error) -> Option<u8> {
    if let Some(code) = interrupted_exit_code(error) {
        return Some(code);
    }
    let top: &(dyn std::error::Error + 'static) = error.as_ref();
    if let Some(error) = top.downcast_ref::<jope_client::discover::Error>() {
        return Some(error.exit_code());
    }
    top.downcast_ref::<DispatchUnavailable>()
        .map(|error| match error {
            DispatchUnavailable::EndpointCredentials => 9,
            #[cfg(not(feature = "engine"))]
            DispatchUnavailable::LocalHandlers => 2,
        })
}

/// Only a direct, acknowledged interruption may select exit130. A surrounding
/// cleanup/IO failure remains an error even if its cause was an interruption.
pub fn interrupted_exit_code(error: &anyhow::Error) -> Option<u8> {
    #[cfg(not(windows))]
    {
        let top: &(dyn std::error::Error + 'static) = error.as_ref();
        top.downcast_ref::<super::inproc::Interrupted>()
            .map(super::inproc::Interrupted::exit_code)
    }
    #[cfg(windows)]
    {
        let _ = error;
        None
    }
}

fn finish_interrupt<T: Transport>(
    transport: &T,
    current: &RunSnapshot,
    interrupts: &RunInterrupts,
    stderr: &mut impl Write,
) -> anyhow::Result<()> {
    if !interrupts.requested() {
        return Ok(());
    }
    write_run_id(stderr, &current.run_id)?;
    #[cfg(not(windows))]
    if let Some(acknowledged) = interrupts.finish(transport, current.clone())? {
        return Err(acknowledged.into());
    }
    #[cfg(windows)]
    {
        let _ = transport;
        anyhow::bail!("interrupted");
    }
    #[cfg(not(windows))]
    Ok(())
}

#[cfg(feature = "engine")]
fn run_m1(args: &RunArgs) -> anyhow::Result<()> {
    use jope_runtime::Failure;
    // Authenticate before opening the input or creating a store/worker.
    let _package = crate::bootstrap::open()?;
    let path = args
        .m1_plan
        .as_deref()
        .ok_or_else(|| Failure::input("M1 requires a prepared plan"))?;
    let bytes =
        std::fs::read(path).map_err(|error| Failure::input(format!("read M1 plan: {error}")))?;
    let input: Value = serde_json::from_slice(&bytes)
        .map_err(|error| Failure::input(format!("M1 plan JSON: {error}")))?;
    let plan: jope_service::discover::r#loop::Plan = serde_json::from_value(input["plan"].clone())
        .map_err(|error| Failure::input(format!("M1 prepared plan: {error}")))?;
    let spec = args
        .run_spec()
        .map_err(|error| Failure::input(format!("M1 run options: {error}")))?;
    validate_m1_spec(&spec, &plan)?;
    let executable = std::env::current_exe()
        .map_err(|error| Failure::backend(format!("M1 worker executable: {error}")))?;
    let result = jope_service::discover::r#loop::run_m1(path, &executable)?;
    write_json(&result).map_err(|error| Failure::output(format!("M1 JSON output: {error}")))
}

#[cfg(not(feature = "engine"))]
fn run_m1(_args: &RunArgs) -> anyhow::Result<()> {
    let _package = crate::bootstrap::open()?;
    Err(jope_runtime::Failure::backend(
        "M1 discovery requires a build with the engine feature",
    ))
}

#[cfg(feature = "engine")]
fn validate_m1_spec(
    spec: &RunSpec,
    plan: &jope_service::discover::r#loop::Plan,
) -> anyhow::Result<()> {
    use jope_api::{Guidance, LandmarkMode, ObjectiveSpec, SplitKind, Task};
    use jope_runtime::Failure;
    if spec.dataset != plan.dataset_id
        || spec.target != plan.target
        || spec.objective != ObjectiveSpec::Name(plan.objective.clone())
        || spec.effort != plan.effort
        || spec.seed.0 != plan.seed
        || spec.budget.evaluations.map(|n| n.0) != Some(plan.evaluations)
        || spec.budget.wall_seconds.map(|n| n.0) != Some(plan.wall_seconds)
    {
        return Err(Failure::input(
            "M1 dataset, target, objective, effort, seed and budgets must match the prepared plan",
        ));
    }
    // This probe consumes an already prepared FIT matrix; it does not perform
    // public dataset registration, learned conditioning, finalization or export.
    if spec.task != Task::Regression
        || !spec.ignore_columns.is_empty()
        || spec.test_dataset.is_some()
        || spec.split.kind != SplitKind::Random
        || spec.split.holdout_fraction != 0.2
        || spec.split.group_column.is_some()
        || spec.split.time_column.is_some()
        || plan.groups.is_some()
        || plan.times.is_some()
        || plan.gap_rows != 0
        || spec.landmarks != LandmarkMode::None
        || !matches!(spec.guidance, Guidance::Auto | Guidance::Heuristic)
        || !spec.languages.is_empty()
        || spec.require_all_languages
        || !matches!(spec.device.as_str(), "cpu" | "auto")
    {
        return Err(Failure::input(
            "M1 supports prepared random 80/20 FIT regression, no landmarks, heuristic CPU selection and no test dataset or export languages",
        ));
    }
    Ok(())
}

pub fn run<T: Transport>(args: RunArgs, transport: &T) -> anyhow::Result<()> {
    let spec = args.run_spec()?;
    let key = crate::common::resolve_idempotency_key(args.idempotency_key.as_deref());
    let response = transport.run_start(&spec, Some(&key))?;
    match response {
        RunResult::Plan(plan) => write_json(&*plan),
        RunResult::Started(snapshot) if args.detach => {
            write_run_id(&mut std::io::stderr().lock(), &snapshot.run_id)?;
            write_json(&*snapshot)
        }
        RunResult::Started(snapshot) => {
            let mut current = *snapshot;
            let interrupts = RunInterrupts::install()?;
            if args.events.as_deref() == Some("ndjson") {
                let mut events = transport.follow_events(
                    current.run_id.clone(),
                    parse_after(args.after_event.as_deref())?,
                )?;
                loop {
                    if interrupts.requested() {
                        // Drop the response reader before requesting committed control.
                        drop(events);
                        finish_interrupt(
                            transport,
                            &current,
                            &interrupts,
                            &mut std::io::stderr().lock(),
                        )?;
                        break;
                    }
                    let Some(event) = events.next() else { break };
                    let event = match event {
                        Ok(event) => event,
                        Err(error) => {
                            if interrupts.requested() {
                                let _ =
                                    write_run_id(&mut std::io::stderr().lock(), &current.run_id);
                            }
                            return Err(error.into());
                        }
                    };
                    println!("{}", serde_json::to_string(&event_json(&event)?)?);
                }
            }
            while !matches!(
                current.state,
                RunState::Completed | RunState::Cancelled | RunState::Failed
            ) {
                finish_interrupt(
                    transport,
                    &current,
                    &interrupts,
                    &mut std::io::stderr().lock(),
                )?;
                let status: anyhow::Result<RunSnapshot> = if args.wait == 0 {
                    transport.run_status(&current.run_id).map_err(Into::into)
                } else {
                    wait_status(transport, &current, args.wait)
                };
                current = match status {
                    Ok(snapshot) => snapshot,
                    Err(error) => {
                        if interrupts.requested() {
                            let _ = write_run_id(&mut std::io::stderr().lock(), &current.run_id);
                        }
                        return Err(error);
                    }
                };
            }
            finish_interrupt(
                transport,
                &current,
                &interrupts,
                &mut std::io::stderr().lock(),
            )?;
            let manifest = args
                .out
                .as_deref()
                .map(|directory| export_best(transport, &current, &spec, directory))
                .transpose()?;
            finish_interrupt(
                transport,
                &current,
                &interrupts,
                &mut std::io::stderr().lock(),
            )?;
            if args.events.as_deref() == Some("ndjson") || manifest.is_some() {
                write_json(&receipt_value(&current, manifest))
            } else {
                write_json(&current)
            }
        }
    }
}

fn export_best<T: Transport>(
    transport: &T,
    snapshot: &RunSnapshot,
    spec: &RunSpec,
    directory: &std::path::Path,
) -> anyhow::Result<Value> {
    use std::fs;
    let best = snapshot
        .best
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("completed run has no best algorithm to export"))?;
    let request = ExportRequest {
        algorithm_id: best.algorithm_id.clone(),
        kind: ExportKind::Bundle,
        languages: spec.languages.clone(),
        require_all_languages: spec.require_all_languages,
        refit: Refit::Fit,
    };
    let key = crate::common::resolve_idempotency_key(None);
    let manifest = transport.model_export(&snapshot.run_id, &request, Some(&key))?;
    fs::create_dir_all(directory)?;
    let path = directory.join(manifest.artifact_id.to_string());
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    let mut response = transport.artifact_download(&manifest.artifact_id)?;
    std::io::copy(&mut response.body, &mut file)?;
    file.sync_all()?;
    Ok(serde_json::to_value(manifest)?)
}

fn wait_status<T: Transport>(
    transport: &T,
    snapshot: &RunSnapshot,
    seconds: u64,
) -> anyhow::Result<RunSnapshot> {
    let request = Request::new(7, 1, &[("id", &snapshot.run_id.to_string())])?.query(&[
        ("wait_seconds", &seconds.to_string()),
        ("after_event", &snapshot.event_watermark.0.to_string()),
    ]);
    Ok(transport.execute(request)?.json()?)
}

fn parse_after(value: Option<&str>) -> anyhow::Result<Option<DecimalU64>> {
    value
        .map(|value| serde_json::from_value(Value::String(value.to_owned())).map_err(Into::into))
        .transpose()
}

fn event_json(event: &jope_client::StreamEvent) -> anyhow::Result<Value> {
    match event {
        jope_client::StreamEvent::Durable(event) => Ok(serde_json::to_value(event)?),
        jope_client::StreamEvent::Telemetry(event) => Ok(serde_json::to_value(event)?),
    }
}

fn write_json(value: &impl serde::Serialize) -> anyhow::Result<()> {
    let stdout = std::io::stdout();
    let mut stdout = stdout.lock();
    stdout.write_all(encode_json_line(value)?.as_bytes())?;
    Ok(())
}

fn encode_json_line(value: &impl serde::Serialize) -> anyhow::Result<String> {
    Ok(format!("{}\n", serde_json::to_string(value)?))
}

fn write_run_id(stderr: &mut impl Write, run_id: &impl std::fmt::Display) -> std::io::Result<()> {
    writeln!(stderr, "{run_id}")
}

fn receipt_value(run: &impl serde::Serialize, export: Option<Value>) -> Value {
    serde_json::json!({"kind":"receipt","run":run,"export":export})
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use serde_json::json;

    #[derive(Parser)]
    struct Args {
        #[command(subcommand)]
        command: crate::cli::Command,
    }

    #[cfg(not(windows))]
    #[test]
    fn request_boundary_requires_acknowledged_pause_or_cancel_and_preserves_control_errors() {
        use jope_client::{
            transport::{header, HeaderMap},
            Request, Response,
        };
        use std::{io::Cursor, sync::Mutex};
        struct Control {
            current: Mutex<RunSnapshot>,
            actions: Mutex<Vec<String>>,
            refuse: bool,
        }
        impl Transport for Control {
            fn execute(&self, request: Request) -> jope_client::Result<Response> {
                let mut current = self.current.lock().unwrap();
                if request.method().as_str() == "POST" {
                    if self.refuse {
                        return Err(jope_client::Error::Protocol(
                            "unknown control commit outcome",
                        ));
                    }
                    let value: Value = serde_json::from_slice(request.body()).unwrap();
                    let action = value["action"].as_str().unwrap();
                    self.actions.lock().unwrap().push(action.into());
                    current.state = match action {
                        "pause" => RunState::Paused,
                        "cancel" => RunState::Cancelled,
                        _ => panic!("unexpected action"),
                    };
                    current.revision.0 += 1;
                }
                let mut headers = HeaderMap::new();
                headers.insert(header::CONTENT_TYPE, "application/json".parse().unwrap());
                headers.insert("jope-api-version", "1.0".parse().unwrap());
                Ok(Response::new(
                    200,
                    headers,
                    Cursor::new(serde_json::to_vec(&*current).unwrap()),
                ))
            }
        }
        let snapshot: RunSnapshot = serde_json::from_value(json!({
            "run_id":"run_01JBD8Q6N4V2T5Y8K3M7P9R1S0", "revision":"1",
            "state":"running", "phase":"searching", "pending":null, "effort":"fast",
            "budget_used":{"evaluations":"1","wall_seconds":"1"},
            "budget_total":{"evaluations":"128","wall_seconds":"60"}, "best":null,
            "event_watermark":"0", "warnings":[], "completion_reason":null
        }))
        .unwrap();
        for (count, refused, action) in [(1, false, "pause"), (2, false, "cancel"), (1, true, "")] {
            let control = Control {
                current: Mutex::new(snapshot.clone()),
                actions: Mutex::new(vec![]),
                refuse: refused,
            };
            let interrupts = RunInterrupts::delivered_for_test(count);
            let mut stderr = vec![];
            let error =
                finish_interrupt(&control, &snapshot, &interrupts, &mut stderr).unwrap_err();
            assert_eq!(stderr, format!("{}\n", snapshot.run_id).as_bytes());
            if refused {
                assert!(error.to_string().contains("unknown control commit outcome"));
                assert_eq!(interrupted_exit_code(&error), None);
                assert!(control.actions.lock().unwrap().is_empty());
            } else {
                assert_eq!(interrupted_exit_code(&error), Some(130));
                assert_eq!(*control.actions.lock().unwrap(), [action]);
            }
        }
    }

    #[cfg(not(windows))]
    #[test]
    fn cleanup_failure_around_an_acknowledgment_does_not_become_exit130() {
        let acknowledged: anyhow::Error = super::super::inproc::Interrupted {
            run_id: "run_01JBD8Q6N4V2T5Y8K3M7P9R1S0".parse().unwrap(),
            state: RunState::Paused,
        }
        .into();
        assert_eq!(interrupted_exit_code(&acknowledged), Some(130));
        let error = acknowledged.context("response reader shutdown failed");
        assert_eq!(interrupted_exit_code(&error), None);
        assert_eq!(dispatch_exit_code(&error), None);
    }

    #[test]
    fn batch_discovery_failures_keep_catalog_exit_codes_and_do_not_mask_cleanup_errors() {
        for (error, expected) in [
            (jope_client::discover::Error::NoEngine, 9),
            (jope_client::discover::Error::InvalidEndpoint, 2),
            (
                jope_client::discover::Error::Busy {
                    pid: Some(123),
                    command: "owned-fixture".into(),
                },
                9,
            ),
        ] {
            let error: anyhow::Error = error.into();
            assert_eq!(dispatch_exit_code(&error), Some(expected));
            assert_eq!(
                dispatch_exit_code(&error.context("response shutdown failed")),
                None
            );
        }
        assert_eq!(
            dispatch_exit_code(&DispatchUnavailable::EndpointCredentials.into()),
            Some(9)
        );
        #[cfg(not(feature = "engine"))]
        assert_eq!(
            dispatch_exit_code(&DispatchUnavailable::LocalHandlers.into()),
            Some(2)
        );
    }

    fn m1_argv(path: &std::path::Path) -> Vec<String> {
        [
            "jopedime",
            "run",
            "--dataset",
            "ds_21fbfe8b53dbcd605d79486b",
            "--target",
            "target",
            "--task",
            "regression",
            "--split",
            r#"{"kind":"random","holdout_fraction":0.2,"group_column":null,"time_column":null}"#,
            "--objective",
            "rmse",
            "--effort",
            "fast",
            "--budget",
            r#"{"evaluations":"1","wall_seconds":"60"}"#,
            "--landmarks",
            "none",
            "--guidance",
            "heuristic",
            "--device",
            "cpu",
            "--seed",
            "73",
            "--json",
            "--m1-plan",
        ]
        .into_iter()
        .map(str::to_owned)
        .chain(std::iter::once(path.to_string_lossy().into_owned()))
        .collect()
    }

    #[test]
    fn hidden_m1_requires_json_preserves_public_requirements_and_conflicts_with_transport_modes() {
        use clap::CommandFactory;
        let argv = m1_argv(std::path::Path::new("prepared.json"));
        let parsed = Args::try_parse_from(&argv).unwrap();
        let crate::cli::Command::Run(args) = parsed.command else {
            panic!("run command")
        };
        assert_eq!(args.m1_plan, Some("prepared.json".into()));
        assert!(!serde_json::to_string(&args.run_spec().unwrap())
            .unwrap()
            .contains("m1"));
        let mut cli = crate::cli::Cli::command();
        let help = cli.find_subcommand_mut("run").unwrap().render_long_help();
        assert!(!help.to_string().contains("m1"));
        let without_json: Vec<_> = argv.iter().filter(|a| a.as_str() != "--json").collect();
        assert!(Args::try_parse_from(without_json).is_err());
        assert!(Args::try_parse_from(["jopedime", "run", "--json"]).is_err());
        for extra in [
            vec!["--detach"],
            vec!["--dry-run"],
            vec!["--events", "ndjson"],
            vec!["--wait", "0"],
        ] {
            let mut conflict = argv.clone();
            conflict.extend(extra.into_iter().map(str::to_owned));
            assert!(Args::try_parse_from(conflict).is_err());
        }
    }

    #[cfg(feature = "engine")]
    fn m1_fixture(directory: &std::path::Path) -> std::path::PathBuf {
        use jope_service::discover::r#loop::Plan;
        let raw = include_bytes!("../../../../test-support/datasets/v1/d2-regression/train.csv");
        assert_eq!(
            jope_runtime::sha256_hex(raw),
            "d9a0b97e91ff9d04bc87fefa4671a49c40cee21dfac05be58a7fc7ce08045931"
        );
        let mut names: Vec<_> = (0..21).map(|j| format!("x{j}")).collect();
        names.push("target".into());
        let mut csv = names.join(",").into_bytes();
        csv.push(b'\n');
        csv.extend(raw);
        let mut table =
            jope_runtime::table::parse_csv(&csv, &jope_runtime::limits::Limits::default()).unwrap();
        let (fit, _) = jope_service::discover::splits::random_split(
            table.rows.len(),
            table.rows.len() * 4 / 5,
            73,
            "dsc08/fit",
        )
        .unwrap();
        let rows: Vec<_> = fit.into_iter().take(30).collect();
        table.rows = rows.iter().map(|&r| table.rows[r].clone()).collect();
        let bytes = jope_service::dataset::jmat::encode(&table).unwrap();
        let matrix = directory.join("fit.jmat");
        std::fs::write(&matrix, &bytes).unwrap();
        std::fs::write(
            directory.join("source-rows.json"),
            serde_json::to_vec(&rows).unwrap(),
        )
        .unwrap();
        let plan = Plan {
            run_id: "run_01JBD8Q6N4V2T5Y8K3M7P9R1S0".parse().unwrap(),
            dataset_id: "ds_21fbfe8b53dbcd605d79486b".parse().unwrap(),
            problem_digest: jope_runtime::sha256_hex(&bytes),
            bundle_digest: jope_runtime::sha256_hex(b"DSC08 heuristic M1, no learned bundle"),
            binary_version: "dsc08f01-cli-probe-v1".into(),
            matrix_path: matrix,
            matrix_sha256: jope_runtime::sha256_hex(&bytes),
            target: "target".into(),
            features: names[..21].to_vec(),
            objective: "rmse".into(),
            seed: 73,
            planned_workers: 8,
            effort: jope_api::Effort::Fast,
            evaluations: 1,
            wall_seconds: 60,
            evaluation_wall_ms: 2000,
            evaluation_rss_bytes: 128 * 1024 * 1024,
            groups: None,
            times: None,
            gap_rows: 0,
        };
        let path = directory.join("m1.json");
        std::fs::write(
            &path,
            serde_json::to_vec_pretty(
                &json!({"home":directory.join("store"),"plan":plan,"live_workers":2}),
            )
            .unwrap(),
        )
        .unwrap();
        path
    }

    #[cfg(feature = "engine")]
    #[test]
    fn hidden_m1_refuses_malformed_or_contradictory_input_before_creating_a_store() {
        let directory = tempfile::tempdir().unwrap();
        let path = m1_fixture(directory.path());
        let mut argv = m1_argv(&path);
        let objective = argv.iter().position(|a| a == "--objective").unwrap() + 1;
        argv[objective] = "mse".into();
        let crate::cli::Command::Run(args) = Args::try_parse_from(argv).unwrap().command else {
            panic!("run")
        };
        let error = dispatch(args).unwrap_err();
        assert_eq!(
            jope_runtime::error::kind_of(&error),
            jope_runtime::FailureKind::Input
        );
        assert!(!directory.path().join("store").exists());
        std::fs::write(&path, b"not json").unwrap();
        let crate::cli::Command::Run(args) = Args::try_parse_from(m1_argv(&path)).unwrap().command
        else {
            panic!("run")
        };
        assert_eq!(
            jope_runtime::error::kind_of(&dispatch(args).unwrap_err()),
            jope_runtime::FailureKind::Input
        );
        assert!(!directory.path().join("store").exists());
    }

    #[cfg(not(feature = "engine"))]
    #[test]
    fn hidden_m1_without_engine_refuses_as_backend_before_reading_a_plan() {
        let crate::cli::Command::Run(args) =
            Args::try_parse_from(m1_argv(std::path::Path::new("absent.json")))
                .unwrap()
                .command
        else {
            panic!("run")
        };
        assert_eq!(
            jope_runtime::error::kind_of(&dispatch(args).unwrap_err()),
            jope_runtime::FailureKind::Backend
        );
    }

    #[cfg(feature = "engine")]
    #[test]
    #[ignore = "actual CLI gate; build the current jopedime with dev-keys, set JOPE_M1_CLI, and explicitly run this test"]
    fn hidden_m1_actual_cli_runs_real_fit_without_a_daemon_and_refuses_damaged_package() {
        use std::process::Command;
        let executable = PathBuf::from(
            std::env::var_os("JOPE_M1_CLI").expect("build/select the current CLI first"),
        );
        let directory = tempfile::tempdir().unwrap();
        let input = m1_fixture(directory.path());
        let argv = m1_argv(&input);
        let output = Command::new(&executable)
            .args(&argv[1..])
            .env_clear()
            .env("JOPEDIME_HOME", directory.path().join("absent-daemon"))
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let value: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(output.stdout.iter().filter(|&&b| b == b'\n').count(), 1);
        assert_eq!(value["stage"], "search_finished");
        assert!(value["champion"]["cv_value"].as_f64().unwrap().is_finite());
        assert_eq!(value["nondeterminism"]["count"], "0");
        let text = String::from_utf8(output.stdout.clone()).unwrap();
        for forbidden in [
            "(pipe", "\"dsl\"", "admitted", "stalled", "round-", "slot-", "rung-",
        ] {
            assert!(!text.contains(forbidden), "public leak {forbidden}");
        }
        assert!(!directory.path().join("absent-daemon").exists());

        use jope_assets::writer::{assemble_development, stage, write_new, StageOptions};
        let code = std::fs::read(&executable).unwrap();
        let staged = stage(
            &code,
            vec![],
            &StageOptions {
                product: "jopedime".into(),
                product_version: env!("CARGO_PKG_VERSION").into(),
                target: "x86_64-unknown-linux-gnu".into(),
                profile: "fixture".into(),
                scope: jope_assets::Scope::Development,
                source: Default::default(),
                methods: vec!["rust.ridge".into()],
                key_id: jope_assets::verify::development_key().id,
                frame_bytes: 1 << 20,
                level: 3,
            },
        )
        .unwrap();
        let mut package = assemble_development(&code, &staged).unwrap();
        // Change an executable-region byte while preserving the loader header.
        package[code.len() - 1] ^= 1;
        let damaged = directory.path().join("damaged-jopedime");
        write_new(&package, &damaged).unwrap();
        let refused = Command::new(&damaged)
            .args(&argv[1..])
            .env_clear()
            .output()
            .unwrap();
        assert_eq!(
            refused.status.code(),
            Some(3),
            "{}",
            String::from_utf8_lossy(&refused.stderr)
        );
        assert!(refused.stdout.is_empty());
        if let Some(root) = std::env::var_os("JOPE_M1_RECEIPTS") {
            let root = PathBuf::from(root);
            std::fs::create_dir_all(&root).unwrap();
            std::fs::write(root.join("stdout.json"), &output.stdout).unwrap();
            std::fs::write(root.join("integrity.stderr"), &refused.stderr).unwrap();
            for name in ["fit.jmat", "source-rows.json", "m1.json"] {
                std::fs::copy(directory.path().join(name), root.join(name)).unwrap();
            }
            std::fs::write(root.join("cli.json"), serde_json::to_vec_pretty(&json!({"executable":executable,"sha256":jope_runtime::sha256_hex(&code),"argv":argv,"success_exit":output.status.code(),"integrity_exit":refused.status.code()})).unwrap()).unwrap();
        }
    }

    #[test]
    fn run_flags_round_trip_to_authoritative_runspec_json() {
        let parsed = Args::try_parse_from([
            "jopedime",
            "run",
            "--dataset",
            "ds_0123456789abcdef01234567",
            "--target",
            "y",
            "--task",
            "regression",
            "--ignore-column",
            "x1",
            "--ignore-column",
            "x2",
            "--test-dataset",
            "ds_abcdef0123456789abcdef01",
            "--split",
            r#"{"kind":"random","holdout_fraction":0.2,"group_column":null,"time_column":null}"#,
            "--objective",
            "rmse",
            "--effort",
            "medium",
            "--budget",
            r#"{"evaluations":"100","wall_seconds":"60"}"#,
            "--landmarks",
            "full",
            "--guidance",
            "auto",
            "--language",
            "python",
            "--language",
            "rust",
            "--require-all-languages",
            "--device",
            "cpu",
            "--seed",
            "7",
            "--dry-run",
            "--plan-digest",
            "abc",
        ])
        .unwrap();
        let crate::cli::Command::Run(args) = parsed.command else {
            panic!("run command")
        };
        let spec = args.run_spec().unwrap();
        let json = serde_json::to_value(spec).unwrap();
        assert_eq!(json["dataset"], "ds_0123456789abcdef01234567");
        assert_eq!(json["test_dataset"], "ds_abcdef0123456789abcdef01");
        assert_eq!(json["ignore_columns"], json!(["x1", "x2"]));
        assert_eq!(json["languages"], json!(["python", "rust"]));
        assert_eq!(json["dry_run"], true);
        assert_eq!(json["plan_digest"], "abc");
        let _: RunSpec = serde_json::from_value(json).unwrap();
    }

    #[test]
    fn api_json_output_is_one_golden_document_line() {
        let line = encode_json_line(
            &json!({"run_id":"run_01J00000000000000000000000","state":"completed"}),
        )
        .unwrap();
        assert_eq!(
            line,
            "{\"run_id\":\"run_01J00000000000000000000000\",\"state\":\"completed\"}\n"
        );
    }

    #[test]
    fn events_and_export_receipt_is_one_final_golden_json_line() {
        let receipt = receipt_value(
            &json!({"run_id":"run_01J00000000000000000000000","state":"completed"}),
            Some(json!({"artifact_id":"art_0123456789abcdef01234567"})),
        );
        assert_eq!(
            encode_json_line(&receipt).unwrap(),
            "{\"export\":{\"artifact_id\":\"art_0123456789abcdef01234567\"},\"kind\":\"receipt\",\"run\":{\"run_id\":\"run_01J00000000000000000000000\",\"state\":\"completed\"}}\n"
        );
    }

    #[test]
    fn detach_and_interrupt_report_run_id_on_stderr() {
        let mut stderr = Vec::new();
        write_run_id(&mut stderr, &"run_01J00000000000000000000000").unwrap();
        assert_eq!(stderr, b"run_01J00000000000000000000000\n");
    }

    #[test]
    fn noninteractive_modes_parse_without_prompting() {
        let required = [
            "--dataset",
            "ds_0123456789abcdef01234567",
            "--target",
            "y",
            "--task",
            "regression",
            "--split",
            r#"{"kind":"random","holdout_fraction":0.2,"group_column":null,"time_column":null}"#,
            "--objective",
            "rmse",
            "--effort",
            "medium",
            "--budget",
            r#"{"evaluations":"100","wall_seconds":"60"}"#,
            "--landmarks",
            "full",
            "--guidance",
            "auto",
            "--device",
            "cpu",
            "--seed",
            "7",
        ];
        for mode in [
            vec!["--detach"],
            vec!["--events", "ndjson", "--after-event", "7"],
            vec!["--out", "model-bundle"],
        ] {
            let mut argv = vec!["jopedime", "run"];
            argv.extend(required);
            argv.extend(mode);
            Args::try_parse_from(argv).unwrap();
        }
    }

    #[cfg(feature = "engine")]
    #[test]
    fn run_uses_the_typed_in_process_api_contract() {
        use jope_client::{InProcess, Transport};
        let client = InProcess::local(43123, &"ab".repeat(32)).unwrap();
        let parsed = Args::try_parse_from([
            "jopedime",
            "run",
            "--dataset",
            "ds_0123456789abcdef01234567",
            "--target",
            "y",
            "--task",
            "regression",
            "--split",
            r#"{"kind":"random","holdout_fraction":0.2,"group_column":null,"time_column":null}"#,
            "--objective",
            "rmse",
            "--effort",
            "medium",
            "--budget",
            r#"{"evaluations":"100","wall_seconds":"60"}"#,
            "--landmarks",
            "full",
            "--guidance",
            "auto",
            "--device",
            "cpu",
            "--seed",
            "7",
            "--dry-run",
        ])
        .unwrap();
        let crate::cli::Command::Run(args) = parsed.command else {
            panic!("run command")
        };
        let spec = args.run_spec().unwrap();
        assert!(matches!(
            client.run_start(&spec, Some("bat02_test")),
            Err(jope_client::Error::Problem(problem))
                if problem.code == jope_api::error::ErrorCode::CapabilityUnavailable
        ));
    }
}
