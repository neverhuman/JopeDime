//! Thin, read-only view over producer receipts. Dispatch remains in the umbrella CLI.

use clap::{Parser, Subcommand};
use serde::Serialize;
use serde_json::Value;
use std::{fs, path::PathBuf, process::ExitCode};

#[derive(Debug, Parser)]
pub struct ResearchCli {
    #[command(subcommand)]
    pub command: ResearchCommand,
}

#[derive(Debug, Subcommand)]
pub enum ResearchCommand {
    /// Report dated evidence without changing producer receipts.
    Doctor(ReceiptArgs),
    /// Show separate evidence records and their freshness labels.
    Status(ReceiptArgs),
    /// Check that receipt files contain valid JSON objects.
    Verify(ReceiptArgs),
    /// Render a compact receipt-backed summary.
    Brief(ReceiptArgs),
}

#[derive(Debug, clap::Args)]
pub struct ReceiptArgs {
    /// Receipt JSON files supplied by their owning producers.
    #[arg(long = "receipt", required = true)]
    pub receipts: Vec<PathBuf>,
    /// UTC cutoff selected from the relevant evidence policy, e.g. 2026-10-17T00:00:00Z.
    #[arg(long)]
    pub stale_before_utc: Option<String>,
    /// Emit machine-readable JSON.
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Freshness {
    Dated,
    NotMarkedStale,
    Unknown,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionEvidence {
    NotRecorded,
    ConfiguredOnly,
    Executed,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ReceiptStatus {
    pub task: Option<String>,
    pub stage: Option<String>,
    pub producer_status: Option<String>,
    pub source_sha: Option<String>,
    pub recorded_at_utc: Option<String>,
    pub freshness: Freshness,
    pub execution_evidence: ExecutionEvidence,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct StatusReport {
    pub as_of_utc: String,
    pub receipts: Vec<ReceiptStatus>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct MachineFailure {
    pub ok: bool,
    pub exit_code: u8,
    pub reason: &'static str,
    pub capability: String,
}

impl MachineFailure {
    pub fn unavailable(capability: impl Into<String>) -> Self {
        Self {
            ok: false,
            exit_code: 2,
            reason: "capability_unavailable",
            capability: capability.into(),
        }
    }

    pub fn exit_status(&self) -> ExitCode {
        ExitCode::from(self.exit_code)
    }

    fn invalid_receipt() -> Self {
        Self {
            ok: false,
            exit_code: 2,
            reason: "invalid_receipt",
            capability: "receipt_json".to_owned(),
        }
    }
}

/// Build a status view from already-read producer receipts. Raw producer values are preserved.
pub fn evaluate_receipts(
    receipts: &[Value],
    as_of_utc: &str,
    stale_before_utc: Option<&str>,
) -> Result<StatusReport, MachineFailure> {
    if receipts.is_empty() {
        return Err(MachineFailure::unavailable("receipt_backed_status"));
    }
    let mut output = Vec::with_capacity(receipts.len());
    for receipt in receipts {
        if !receipt.is_object() {
            return Err(MachineFailure::invalid_receipt());
        }
        let recorded_at = first_string(
            receipt,
            &[
                &["verification", "recorded_at_utc"],
                &["verification", "recorded_at"],
                &["updated_at"],
                &["updated_utc"],
                &["created_utc"],
            ],
        );
        let freshness = match (recorded_at.as_deref(), stale_before_utc) {
            (Some(recorded), Some(cutoff)) => {
                match (utc_second_key(recorded), utc_second_key(cutoff)) {
                    (Some(recorded), Some(cutoff)) if recorded < cutoff => Freshness::Dated,
                    (Some(_), Some(_)) => Freshness::NotMarkedStale,
                    _ => Freshness::Unknown,
                }
            }
            _ => Freshness::Unknown,
        };
        output.push(ReceiptStatus {
            task: first_string(receipt, &[&["task_id"], &["task"]]),
            stage: first_string(receipt, &[&["stage"], &["phase"]]),
            producer_status: first_string(receipt, &[&["status"], &["state"]]),
            source_sha: first_string(
                receipt,
                &[
                    &["source_sha256"],
                    &["source", "implementation_head"],
                    &["head_commit"],
                    &["head"],
                ],
            ),
            recorded_at_utc: recorded_at,
            freshness,
            execution_evidence: execution_evidence(receipt),
        });
    }
    Ok(StatusReport {
        as_of_utc: as_of_utc.to_owned(),
        receipts: output,
    })
}

fn first_string(value: &Value, paths: &[&[&str]]) -> Option<String> {
    paths.iter().find_map(|path| {
        path.iter()
            .try_fold(value, |current, key| current.get(*key))
            .and_then(Value::as_str)
            .map(str::to_owned)
    })
}

fn execution_evidence(receipt: &Value) -> ExecutionEvidence {
    let configured = receipt.get("command").is_some()
        || receipt
            .get("verification")
            .and_then(|value| value.get("command"))
            .is_some();
    let executed = receipt
        .get("execution")
        .and_then(Value::as_object)
        .is_some_and(|execution| {
            execution.get("exit_code").and_then(Value::as_i64).is_some()
                && execution
                    .get("started_at_utc")
                    .and_then(Value::as_str)
                    .is_some()
                && execution
                    .get("ended_at_utc")
                    .and_then(Value::as_str)
                    .is_some()
                && execution.get("argv").and_then(Value::as_array).is_some()
                && execution
                    .get("source_sha256")
                    .and_then(Value::as_str)
                    .is_some()
                && execution
                    .get("input_sha256")
                    .and_then(Value::as_str)
                    .is_some()
        });
    if executed {
        ExecutionEvidence::Executed
    } else if configured {
        ExecutionEvidence::ConfiguredOnly
    } else {
        ExecutionEvidence::NotRecorded
    }
}

fn utc_second_key(value: &str) -> Option<String> {
    let utc = value.strip_suffix("+00:00").unwrap_or(value);
    let utc = utc.strip_suffix('Z')?;
    let (whole_seconds, fraction) = utc.split_once('.').unwrap_or((utc, ""));
    let bytes = whole_seconds.as_bytes();
    if bytes.len() != 19
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || bytes[10] != b'T'
        || bytes[13] != b':'
        || bytes[16] != b':'
        || bytes
            .iter()
            .enumerate()
            .any(|(index, byte)| ![4, 7, 10, 13, 16].contains(&index) && !byte.is_ascii_digit())
    {
        return None;
    }
    if !fraction.bytes().all(|byte| byte.is_ascii_digit()) || fraction.len() > 9 {
        return None;
    }
    Some(format!("{whole_seconds}.{fraction:0<9}"))
}

pub fn execute(cli: ResearchCli) -> ExitCode {
    let args = match cli.command {
        ResearchCommand::Doctor(args)
        | ResearchCommand::Status(args)
        | ResearchCommand::Verify(args)
        | ResearchCommand::Brief(args) => args,
    };
    let result = read_receipts(&args.receipts).and_then(|receipts| {
        evaluate_receipts(&receipts, &now_utc(), args.stale_before_utc.as_deref())
    });
    match result {
        Ok(report) => {
            if args.json {
                println!(
                    "{}",
                    serde_json::to_string(&report).expect("status serializes")
                );
            } else {
                for item in report.receipts {
                    println!(
                        "{} {} freshness={:?} execution={:?}",
                        item.task.as_deref().unwrap_or("unknown-task"),
                        item.producer_status.as_deref().unwrap_or("unknown-status"),
                        item.freshness,
                        item.execution_evidence
                    );
                }
            }
            ExitCode::SUCCESS
        }
        Err(failure) => {
            if args.json {
                println!(
                    "{}",
                    serde_json::to_string(&failure).expect("failure serializes")
                );
            } else {
                eprintln!("research: {} ({})", failure.capability, failure.reason);
            }
            failure.exit_status()
        }
    }
}

fn read_receipts(paths: &[PathBuf]) -> Result<Vec<Value>, MachineFailure> {
    if paths.is_empty() {
        return Err(MachineFailure::unavailable("receipt_backed_status"));
    }
    paths
        .iter()
        .map(|path| {
            fs::read(path)
                .ok()
                .and_then(|bytes| serde_json::from_slice(&bytes).ok())
                .ok_or_else(MachineFailure::invalid_receipt)
        })
        .collect()
}

fn now_utc() -> String {
    // The caller supplies the cutoff policy; this timestamp only records when the view ran.
    use std::time::{SystemTime, UNIX_EPOCH};
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs());
    let days = (seconds / 86_400) as i64;
    let day_seconds = seconds % 86_400;
    let shifted = days + 719_468;
    let era = shifted / 146_097;
    let day_of_era = shifted - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_part = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_part + 2) / 5 + 1;
    let month = month_part + if month_part < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        day_seconds / 3_600,
        day_seconds % 3_600 / 60,
        day_seconds % 60
    )
}
