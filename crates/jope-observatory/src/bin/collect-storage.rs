//! Explicit metadata refresh, never triggered by a browser request. No bucket payload reads.
use anyhow::{bail, Context, Result};
use chrono::{Duration, Utc};
use clap::Parser;
use serde_json::{json, Value};
use std::{
    fs,
    io::Read,
    path::PathBuf,
    process::{Command, Stdio},
    thread,
    time::{Duration as StdDuration, Instant},
};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    catalog: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    profile: String,
    #[arg(long, default_value = "714044927654")]
    expected_account: String,
    #[arg(long, default_value = "aws")]
    aws: String,
}
fn aws(args: &Args, region: &str, command: &[&str]) -> Result<Value> {
    // Named profile is delegated to AWS CLI. Do not inspect credentials or return stderr secrets.
    let mut child = Command::new(&args.aws)
        .args([
            "--profile",
            &args.profile,
            "--region",
            region,
            "--output",
            "json",
            "--no-cli-pager",
        ])
        .args(command)
        .env("AWS_MAX_ATTEMPTS", "1")
        .env("AWS_PAGER", "")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()?;
    let stdout = child.stdout.take().context("AWS output pipe")?;
    let reader = thread::spawn(move || {
        let mut bytes = vec![];
        stdout
            .take(4 * 1024 * 1024 + 1)
            .read_to_end(&mut bytes)
            .map(|_| bytes)
    });
    let start = Instant::now();
    loop {
        if child.try_wait()?.is_some() {
            break;
        }
        if start.elapsed() > StdDuration::from_secs(25) {
            child.kill()?;
            let _ = child.wait();
            bail!("AWS metadata timeout");
        }
        thread::sleep(StdDuration::from_millis(100));
    }
    let status = child.wait()?;
    let bytes = reader
        .join()
        .map_err(|_| anyhow::anyhow!("AWS reader failed"))??;
    if !status.success() {
        bail!("AWS metadata unavailable or denied; no credential fallback");
    }
    if bytes.len() > 4 * 1024 * 1024 {
        bail!("metadata response too large");
    }
    serde_json::from_slice(&bytes).context("invalid AWS JSON")
}
fn main() -> Result<()> {
    let args = Args::parse();
    let identity = aws(&args, "us-east-2", &["sts", "get-caller-identity"])?;
    if identity["Account"].as_str() != Some(&args.expected_account) {
        bail!("AWS account mismatch");
    }
    let catalog: Value = serde_json::from_slice(&jope_observatory::read_bounded(
        &args.catalog,
        2 * 1024 * 1024,
    )?)?;
    let mut buckets = vec![];
    let mut errors = vec![];
    let start = (Utc::now() - Duration::days(4)).to_rfc3339();
    let end = Utc::now().to_rfc3339();
    for b in catalog["buckets"]
        .as_array()
        .context("catalog buckets required")?
        .iter()
        .take(16)
    {
        let bucket = b["bucket"].as_str().context("bucket name")?;
        let region = b["region"].as_str().context("bucket region")?;
        let dimensions = format!("Name=BucketName,Value={bucket}");
        let result = (|| -> Result<Value> {
            let list = aws(
                &args,
                region,
                &[
                    "cloudwatch",
                    "list-metrics",
                    "--namespace",
                    "AWS/S3",
                    "--dimensions",
                    &dimensions,
                    "--max-items",
                    "500",
                ],
            )?;
            if list["NextToken"].is_string() {
                bail!("metric inventory truncated");
            }
            let mut queries = vec![];
            let mut kinds = vec![];
            for (i, m) in list["Metrics"]
                .as_array()
                .context("metric list")?
                .iter()
                .filter(|m| {
                    m["MetricName"] == "BucketSizeBytes" || m["MetricName"] == "NumberOfObjects"
                })
                .take(100)
                .enumerate()
            {
                // All returned dimensions must match the requested bucket; no cross-bucket totals.
                let ds = m["Dimensions"].as_array().context("dimensions")?;
                if !ds
                    .iter()
                    .any(|d| d["Name"] == "BucketName" && d["Value"] == bucket)
                {
                    bail!("metric bucket mismatch");
                }
                if m["MetricName"] == "NumberOfObjects"
                    && !ds
                        .iter()
                        .any(|d| d["Name"] == "StorageType" && d["Value"] == "AllStorageTypes")
                {
                    continue;
                }
                queries.push(json!({"Id":format!("m{i}"),"MetricStat":{"Metric":m,"Period":86400,"Stat":"Average"},"ReturnData":true}));
                kinds.push((
                    format!("m{i}"),
                    m["MetricName"].as_str().unwrap_or_default().to_owned(),
                ));
            }
            if queries.is_empty() {
                bail!("no daily S3 metrics published");
            }
            let qs = serde_json::to_string(&queries)?;
            let result = aws(
                &args,
                region,
                &[
                    "cloudwatch",
                    "get-metric-data",
                    "--metric-data-queries",
                    &qs,
                    "--start-time",
                    &start,
                    "--end-time",
                    &end,
                    "--scan-by",
                    "TimestampDescending",
                    "--max-datapoints",
                    "1000",
                ],
            )?;
            if result["NextToken"].is_string() {
                bail!("daily metric response truncated");
            }
            // Sum storage types only at a COMMON published daily timestamp. No mixing days.
            let rows = result["MetricDataResults"]
                .as_array()
                .context("metric results")?;
            let mut common: Option<std::collections::BTreeSet<String>> = None;
            for row in rows {
                if row["StatusCode"] != "Complete" {
                    bail!("incomplete CloudWatch metric");
                }
                let times = row["Timestamps"]
                    .as_array()
                    .context("timestamps")?
                    .iter()
                    .filter_map(|t| t.as_str().map(str::to_owned))
                    .collect::<std::collections::BTreeSet<_>>();
                common = Some(match common {
                    None => times,
                    Some(v) => v.intersection(&times).cloned().collect(),
                });
            }
            if rows.len() != queries.len() {
                bail!("missing metric response");
            }
            let stamp = common
                .and_then(|s| s.last().cloned())
                .context("no common daily metric timestamp")?;
            let mut bytes = 0f64;
            let mut objects = None;
            for row in rows {
                let pos = row["Timestamps"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .position(|v| v == &stamp)
                    .context("time alignment")?;
                let value = row["Values"][pos]
                    .as_f64()
                    .filter(|n| n.is_finite() && *n >= 0.)
                    .context("metric value")?;
                if kinds
                    .iter()
                    .any(|(id, k)| row["Id"] == *id && k == "BucketSizeBytes")
                {
                    bytes += value;
                } else {
                    objects = Some(value as u64);
                }
            }
            if objects.is_none() {
                bail!("object count metric missing");
            }
            Ok(
                json!({"bucket":bucket,"region":region,"role":b["role"],"replica_of":b["replica_of"],"bytes":bytes as u64,"objects":objects,"observed_at":stamp,"status":"daily","metric_series":queries.len(),"scope":"CloudWatch daily physical storage, including versions/metadata; not current-key payload bytes"}),
            )
        })();
        match result {
            Ok(v) => buckets.push(v),
            Err(_) => {
                errors.push(format!("{bucket}: daily metadata unavailable; inspect scoped AWS access or publication delay"));
                buckets.push(json!({"bucket":bucket,"region":region,"role":b["role"],"bytes":null,"objects":null,"observed_at":null,"status":"unavailable"}));
            }
        }
    }
    let complete = !buckets.is_empty() && errors.is_empty();
    let bytes: u64 = buckets.iter().filter_map(|v| v["bytes"].as_u64()).sum();
    let objects: u64 = buckets.iter().filter_map(|v| v["objects"].as_u64()).sum();
    let value = json!({"schema":"jope.storage-observation/v1","captured_at":Utc::now().to_rfc3339(),"account":args.expected_account,"source":"AWS/S3 CloudWatch daily metrics, bounded metadata only","scope":"configured JopeDime buckets only, physical copies included; metric timestamp and collector refresh are distinct","complete":complete,"observed_bytes":if buckets.iter().any(|v|v["bytes"].is_number()){json!(bytes)}else{Value::Null},"observed_objects":if buckets.iter().any(|v|v["objects"].is_number()){json!(objects)}else{Value::Null},"buckets":buckets,"errors":errors});
    let temp = args
        .output
        .with_extension(format!("{}.new", std::process::id()));
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)?;
    fs::write(&temp, serde_json::to_vec_pretty(&value)?)?;
    fs::rename(&temp, &args.output)?;
    println!("wrote {} (complete={complete})", args.output.display());
    Ok(())
}
