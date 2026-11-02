use jope_api::{
    decimal::DecimalU64, ExportKind, ExportRequest, Refit, RunSnapshot, RunSpec, RunState,
};
use jope_client::{transport::Request, RunResult, Transport};
use serde_json::Value;
use std::io::Write;

mod args;
pub use args::RunArgs;

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

pub fn run<T: Transport>(mut args: RunArgs, transport: &T) -> anyhow::Result<()> {
    let export_refit: Refit = parse_value(&args.refit)?;
    let key = crate::common::resolve_idempotency_key(args.idempotency_key.as_deref());
    if let Some(path) = args.input.take() {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| anyhow::anyhow!("CSV path must have a UTF-8 file name"))?;
        // Both local HTTP and socket-free transports support JSON path registration.
        let absolute = std::fs::canonicalize(&path).map_err(|error| {
            jope_runtime::Failure::input(format!("open CSV {}: {error}", path.display()))
        })?;
        let registration_key = format!("csv_{}", jope_runtime::sha256_hex(key.as_bytes()));
        let request = jope_api::registry::DatasetRegistration::Path {
            path: absolute
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("CSV path must be UTF-8"))?
                .into(),
            name: Some(name.into()),
        };
        let profile = transport.dataset_register(&request, Some(&registration_key))?;
        if args.task == "auto" {
            if let Some(task) = crate::draft::inferred_tasks(&profile).get(&args.target) {
                args.task = serde_json::to_value(task)?
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("task did not encode as a name"))?
                    .to_owned();
            }
        }
        if args.languages.is_empty() {
            args.languages = vec!["rust".into(), "python".into()];
        }
        args.dataset = Some(profile.dataset_id.to_string());
    }
    let spec = args.run_spec()?;
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
                let after = parse_after(args.after_event.as_deref())?;
                // The original idempotent response may predate completion. SSE
                // remains live, so use the committed watermark to avoid waiting
                // forever when the caller has already consumed the final event.
                current = transport.run_status(&current.run_id)?;
                let drained = matches!(
                    current.state,
                    RunState::Completed | RunState::Cancelled | RunState::Failed
                ) && after
                    .is_some_and(|cursor| cursor.0 >= current.event_watermark.0);
                if !drained {
                    let mut events = transport.follow_events(current.run_id.clone(), after)?;
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
                                    let _ = write_run_id(
                                        &mut std::io::stderr().lock(),
                                        &current.run_id,
                                    );
                                }
                                return Err(error.into());
                            }
                        };
                        let finished = match &event {
                            jope_client::StreamEvent::Durable(event) => match &event.event {
                                jope_api::EventBody::RunFinished(_) => true,
                                jope_api::EventBody::RunState(body) => {
                                    matches!(body.state, RunState::Failed | RunState::Cancelled)
                                }
                                _ => false,
                            },
                            _ => false,
                        };
                        println!("{}", serde_json::to_string(&event_json(&event)?)?);
                        if finished {
                            break;
                        }
                    }
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
            if current.state == RunState::Failed {
                write_run_id(&mut std::io::stderr().lock(), &current.run_id)?;
                return Err(jope_runtime::Failure::backend(
                    "The run failed; inspect its retained status and events.",
                ));
            }
            let manifest = args
                .out
                .as_deref()
                .map(|directory| export_best(transport, &current, &spec, directory, export_refit))
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
    refit: Refit,
) -> anyhow::Result<Value> {
    let best = snapshot
        .best
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("completed run has no best algorithm to export"))?;
    let request = ExportRequest {
        algorithm_id: best.algorithm_id.clone(),
        kind: ExportKind::Bundle,
        languages: spec.languages.clone(),
        require_all_languages: spec.require_all_languages,
        refit,
    };
    let key = crate::common::resolve_idempotency_key(None);
    let manifest = transport.model_export(&snapshot.run_id, &request, Some(&key))?;
    jope_client::files::download_export_artifact(
        transport,
        &manifest.artifact_id,
        directory,
        false,
    )?;
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
mod tests;
