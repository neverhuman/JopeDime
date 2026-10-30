use clap::Args;
use jope_api::{
    decimal::DecimalU64, ids::DatasetId, ExportKind, ExportRequest, Language, Refit, RunSnapshot,
    RunSpec, RunState,
};
use jope_client::{transport::Request, RunResult, Transport};
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::{
    io::Write,
    path::PathBuf,
    str::FromStr,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

#[derive(Args, Debug)]
pub struct RunArgs {
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

pub fn run<T: Transport>(args: RunArgs, transport: &T) -> anyhow::Result<()> {
    let spec = args.run_spec()?;
    let key = crate::common::resolve_idempotency_key(None);
    let response = transport.run_start(&spec, Some(&key))?;
    match response {
        RunResult::Plan(plan) => write_json(&*plan),
        RunResult::Started(snapshot) if args.detach => {
            eprintln!("{}", snapshot.run_id);
            write_json(&*snapshot)
        }
        RunResult::Started(snapshot) => {
            let mut current = *snapshot;
            let interrupted = Arc::new(AtomicBool::new(false));
            signal_hook::flag::register(signal_hook::consts::SIGINT, Arc::clone(&interrupted))?;
            if args.events.as_deref() == Some("ndjson") {
                let events = transport.follow_events(
                    current.run_id.clone(),
                    parse_after(args.after_event.as_deref())?,
                )?;
                for event in events {
                    if interrupted.load(Ordering::Relaxed) {
                        eprintln!("{}", current.run_id);
                        anyhow::bail!("interrupted");
                    }
                    let event = event?;
                    println!("{}", serde_json::to_string(&event_json(&event)?)?);
                }
            }
            while !matches!(
                current.state,
                RunState::Completed | RunState::Cancelled | RunState::Failed
            ) {
                if interrupted.load(Ordering::Relaxed) {
                    eprintln!("{}", current.run_id);
                    anyhow::bail!("interrupted");
                }
                let status: anyhow::Result<RunSnapshot> = if args.wait == 0 {
                    transport.run_status(&current.run_id).map_err(Into::into)
                } else {
                    wait_status(transport, &current, args.wait)
                };
                current = match status {
                    Ok(snapshot) => snapshot,
                    Err(error) => {
                        if interrupted.load(Ordering::Relaxed) {
                            eprintln!("{}", current.run_id);
                        }
                        return Err(error);
                    }
                };
            }
            if interrupted.load(Ordering::Relaxed) {
                eprintln!("{}", current.run_id);
                anyhow::bail!("interrupted");
            }
            let manifest = args
                .out
                .as_deref()
                .map(|directory| export_best(transport, &current, &spec, directory))
                .transpose()?;
            if args.events.as_deref() == Some("ndjson") || manifest.is_some() {
                write_json(&serde_json::json!({"kind":"receipt","run":current,"export":manifest}))
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
