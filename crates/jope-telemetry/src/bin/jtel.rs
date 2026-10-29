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
        /// Print the TRAIN+GP bottleneck report as text instead of the JSON report.
        #[arg(long)]
        text: bool,
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
    /// A flywheel output tree (`results/` and `s3_mirror/round-*/`); `--output` is a new run dir.
    Flywheel,
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
        Command::Summary { run_dir, text } => summary(&run_dir, text),
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
    if matches!(format, InputFormat::Flywheel) {
        let Some(out) = output else {
            bail!("--format flywheel needs --output <new run dir>");
        };
        return import_flywheel(input, out);
    }
    let use_engine = match format {
        InputFormat::Engine => true,
        InputFormat::Epoch => false,
        InputFormat::Flywheel => unreachable!(),
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

/// Backfill a run dir from flywheel outputs: every import adapter's records plus the per-round
/// HWM consistency check, as record lines. Imported records carry `source = "import"`.
fn import_flywheel(root: &Path, out: &Path) -> Result<()> {
    use jope_telemetry::{flywheel, gp};
    if out.join("events.jsonl").exists() {
        bail!(
            "{} already holds a run; import writes a new one",
            out.display()
        );
    }
    fs::create_dir_all(out)?;
    let mut recs: Vec<(String, Value)> = Vec::new();
    let report = root.join("results/flywheel_selfplay_report.json");
    if report.exists() {
        let rep = import::selfplay_report(&report)?;
        for r in &rep.rounds {
            recs.push((flywheel::ROUND_SCHEMA.into(), serde_json::to_value(r)?));
        }
        for q in &rep.queue {
            recs.push((import::QUEUE_SCHEMA.into(), serde_json::to_value(q)?));
        }
    }
    let mut rounds: Vec<PathBuf> = fs::read_dir(root.join("s3_mirror"))
        .map(|rd| rd.flatten().map(|e| e.path()).collect())
        .unwrap_or_default();
    rounds.retain(|p| {
        p.file_name()
            .is_some_and(|n| n.to_string_lossy().starts_with("round-"))
    });
    rounds.sort();
    for dir in rounds {
        let name = dir
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        let round: u32 = name["round-".len()..]
            .parse()
            .with_context(|| format!("round number in {name}"))?;
        let mut hwm = Vec::new();
        for e in fs::read_dir(dir.join("hwm"))
            .into_iter()
            .flatten()
            .flatten()
        {
            let h = import::hwm_file(&e.path())?;
            recs.push((import::HWM_SCHEMA.into(), serde_json::to_value(&h)?));
            hwm.push(h);
        }
        for e in fs::read_dir(dir.join("gp_distilled"))
            .into_iter()
            .flatten()
            .flatten()
        {
            let (_, ret) = import::distilled_round(&e.path())?;
            let c = import::hwm_consistency(round, &hwm, &ret, 60);
            recs.push((import::RETENTION_SCHEMA.into(), serde_json::to_value(&ret)?));
            recs.push((
                import::HWM_CONSISTENCY_SCHEMA.into(),
                serde_json::to_value(&c)?,
            ));
        }
        for e in fs::read_dir(dir.join("scores"))
            .into_iter()
            .flatten()
            .flatten()
        {
            let g = import::retained_scores(&e.path(), round, None)?;
            recs.push((gp::GEN_SCHEMA.into(), serde_json::to_value(&g)?));
        }
    }
    let mut f = std::io::BufWriter::new(File::create(out.join("events.jsonl"))?);
    for (kind, body) in &recs {
        serde_json::to_writer(&mut f, &serde_json::json!({"kind": kind, "body": body}))?;
        f.write_all(b"\n")?;
    }
    f.flush()?;
    let manifest = serde_json::json!({
        "source": "import",
        "status": "complete",
        "run_id": format!("import:{}", root.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()),
        "records_written": recs.len(),
    });
    fs::write(
        out.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    eprintln!("imported {} records into {}", recs.len(), out.display());
    Ok(())
}

/// The TRAIN+GP section: event scalars by registered metric name, plus every record line.
fn train_gp_summary(run_dir: &Path) -> Result<jope_telemetry::summary::Summary> {
    use jope_telemetry::summary::{summarize, ScalarIn};
    let file = File::open(run_dir.join("events.jsonl"))?;
    let mut scalars: Vec<(String, Option<u64>, f64)> = Vec::new();
    let mut records: Vec<(String, Value)> = Vec::new();
    for line in BufReader::new(file).lines() {
        let Ok(value) = serde_json::from_str::<Value>(&line?) else {
            continue;
        };
        match value.get("kind").and_then(Value::as_str) {
            Some("event") => {
                let ev = value.get("event").cloned().unwrap_or(Value::Null);
                let id = ev.get("metric_id").and_then(Value::as_u64).unwrap_or(0) as u16;
                let name = jope_telemetry::train::metric(id)
                    .map_or_else(|| format!("metric.{id}"), |m| m.name.to_owned());
                let v = ev.get("value").and_then(Value::as_f64).unwrap_or(f64::NAN);
                scalars.push((name, ev.get("step").and_then(Value::as_u64), v));
            }
            Some(kind) => records.push((
                kind.to_owned(),
                value.get("body").cloned().unwrap_or(Value::Null),
            )),
            None => {}
        }
    }
    Ok(summarize(
        scalars.iter().map(|(m, step, v)| ScalarIn {
            metric: m,
            group: None,
            step: *step,
            value: *v,
        }),
        records.iter().map(|(k, b)| (k.as_str(), b)),
    ))
}

fn summary(run_dir: &Path, text: bool) -> Result<()> {
    let train_gp = train_gp_summary(run_dir)?;
    if text {
        print!("{}", train_gp.render_text());
        return Ok(());
    }
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
    let report = serde_json::json!({"manifest":manifest,"events_read":events,"records_read":records,"events_by_metric_id":counts,"availability_reason_counts":reason_counts,"availability_reasons":manifest.get("availability_reasons"),"metrics_not_observed":absent,"note":"unobserved metrics are not zero","train_gp":train_gp});
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
