use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use jope_telemetry::import;
use serde_json::Value;
use std::collections::VecDeque;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "jtel", about = "Inspect and recover JopeDime telemetry runs")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Tail {
        run_dir: PathBuf,
        #[arg(long, default_value_t = 50)]
        lines: usize,
    },
    Import {
        input: PathBuf,
        #[arg(long, value_enum, default_value_t = InputFormat::Auto)]
        format: InputFormat,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    Parquet {
        run_dir: PathBuf,
    },
    Recover {
        run_dir: PathBuf,
    },
    Summary {
        run_dir: PathBuf,
    },
    Diff {
        left: PathBuf,
        right: PathBuf,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum InputFormat {
    Auto,
    Engine,
    Epoch,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Tail { run_dir, lines } => tail(&run_dir, lines),
        Command::Import {
            input,
            format,
            output,
        } => import_file(&input, format, output.as_deref()),
        Command::Parquet { run_dir } => {
            let rows = jope_telemetry::sink::parquet::regenerate(
                &run_dir.join("events.jsonl"),
                &run_dir.join("metrics.parquet"),
            )?;
            println!("regenerated metrics.parquet from {rows} event rows");
            Ok(())
        }
        Command::Recover { run_dir } => recover(&run_dir),
        Command::Summary { run_dir } => summary(&run_dir),
        Command::Diff { left, right } => diff(&left, &right),
    }
}

fn recover(run_dir: &Path) -> Result<()> {
    let event_path = run_dir.join("events.jsonl");
    let truncated = jope_telemetry::sink::jsonl::recover_final_line(&event_path)?;
    if truncated > 0 {
        let manifest_path = run_dir.join("manifest.json");
        let mut manifest: Value = serde_json::from_slice(&fs::read(&manifest_path)?)?;
        let old = manifest
            .get("recovery_truncated_bytes")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        manifest["recovery_truncated_bytes"] = Value::from(old.saturating_add(truncated));
        let temp = run_dir.join(format!(".manifest.json.{}.tmp", std::process::id()));
        let mut file = File::create(&temp)?;
        serde_json::to_writer_pretty(&mut file, &manifest)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        fs::rename(temp, manifest_path)?;
    }
    let rows =
        jope_telemetry::sink::parquet::regenerate(&event_path, &run_dir.join("metrics.parquet"))?;
    println!("recovered event tail: truncated {truncated} bytes; Parquet has {rows} events");
    Ok(())
}

fn tail(run_dir: &Path, count: usize) -> Result<()> {
    if count > 10_000 {
        bail!("--lines is capped at 10,000");
    }
    let file = File::open(run_dir.join("events.jsonl"))?;
    let mut tail = VecDeque::with_capacity(count);
    for line in BufReader::new(file).lines() {
        let line = line?;
        if tail.len() == count {
            tail.pop_front();
        }
        if count > 0 {
            tail.push_back(line);
        }
    }
    for line in tail {
        println!("{line}");
    }
    Ok(())
}

fn import_file(input: &Path, format: InputFormat, output: Option<&Path>) -> Result<()> {
    let use_engine = match format {
        InputFormat::Engine => true,
        InputFormat::Epoch => false,
        InputFormat::Auto => input
            .file_name()
            .and_then(|s| s.to_str())
            .is_some_and(|s| s.contains("state")),
    };
    let scalars = if use_engine {
        import::engine_state_jsonl(input)?
    } else {
        import::generic_epoch_jsonl(input)?
    };
    let stdout = std::io::stdout();
    let mut writer: Box<dyn Write> = match output {
        Some(path) => {
            Box::new(File::create(path).with_context(|| format!("create {}", path.display()))?)
        }
        None => Box::new(stdout.lock()),
    };
    for scalar in scalars {
        serde_json::to_writer(&mut writer, &scalar)?;
        writer.write_all(b"\n")?;
    }
    Ok(())
}

fn summary(run_dir: &Path) -> Result<()> {
    let manifest: Value = serde_json::from_slice(&fs::read(run_dir.join("manifest.json"))?)?;
    let mut counts = std::collections::BTreeMap::<u64, u64>::new();
    let mut reason_counts = std::collections::BTreeMap::<u64, u64>::new();
    let file = File::open(run_dir.join("events.jsonl"))?;
    let mut events = 0_u64;
    let mut records = 0_u64;
    let mut missing = std::collections::BTreeSet::new();
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line = line?;
        let value: Value =
            serde_json::from_str(&line).with_context(|| format!("events.jsonl:{}", index + 1))?;
        if value.get("kind").and_then(Value::as_str) == Some("event") {
            events += 1;
            if let Some(id) = value.pointer("/event/metric_id").and_then(Value::as_u64) {
                *counts.entry(id).or_default() += 1;
                missing.insert(id);
            }
            if let Some(id) = value
                .pointer("/event/availability_reason_id")
                .and_then(Value::as_u64)
            {
                *reason_counts.entry(id).or_default() += 1;
            }
        } else {
            records += 1;
        }
    }
    let registered: std::collections::BTreeSet<u64> = manifest
        .get("metrics")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|m| m.get("id").and_then(Value::as_u64))
        .collect();
    let absent: Vec<u64> = registered.difference(&missing).copied().collect();
    let report = serde_json::json!({"manifest":manifest,"events_read":events,"records_read":records,"events_by_metric_id":counts,"availability_reason_counts":reason_counts,"availability_reasons":manifest.get("availability_reasons"),"metrics_not_observed":absent,"note":"unobserved metrics are not zero"});
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn diff(left: &Path, right: &Path) -> Result<()> {
    let a: Value = serde_json::from_slice(&fs::read(left.join("manifest.json"))?)?;
    let b: Value = serde_json::from_slice(&fs::read(right.join("manifest.json"))?)?;
    let identity = ["metadata", "metrics", "event_schema_version"];
    let incompatible: Vec<&str> = identity
        .into_iter()
        .filter(|key| a.get(*key) != b.get(*key))
        .collect();
    let report = serde_json::json!({"compatible":incompatible.is_empty(),"incompatible_fields":incompatible,"left_run_id":a.get("run_id"),"right_run_id":b.get("run_id"),"left_status":a.get("status"),"right_status":b.get("status"),"left_events":a.get("events_written"),"right_events":b.get("events_written"),"warning":"event-count differences are not model-quality comparisons"});
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
