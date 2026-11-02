//! Rebuild receipt-accurate, task-scoped causal evidence and fail promotion closed.

use anyhow::{ensure, Context, Result};
use clap::Parser;
use jope_dime_regressor::selfplay_ingest::{ingest_receipt_with_manifest_connection, open_corpus};
use jope_dime_regressor::selfplay_runtime::{
    atomic_json, default_root_str, now_ms, read_json, sha256_bytes, sha256_file,
};
use jope_dime_regressor::selfplay_shadow::{
    build_causal_evidence_ledger_v2, build_causal_evidence_ledger_v3, CausalEvaluationReceiptV2,
    CausalEvidenceEpochV1, CausalEvidenceLedgerV2, CausalEvidenceLedgerV3, CausalLane,
    DlAncestryKind, Sha256Digest, Task, CAUSAL_EVALUATION_RECEIPT_SCHEMA,
};
use rusqlite::{Connection, OpenFlags};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

const HISTORICAL_BINARY_LOWER_BOUND: usize = 286;
const HISTORICAL_REGRESSION_LOWER_BOUND: usize = 776;

#[derive(Parser, Debug)]
#[command(
    name = "jope-causal-evidence",
    about = "Build CausalEvidenceLedger.v2 from append-only raw receipts"
)]
struct Args {
    #[arg(long, default_value = default_root_str())]
    root: PathBuf,
    #[arg(long)]
    once: bool,
    #[arg(long)]
    backfill_raw_receipts: bool,
    #[arg(long, default_value_t = 600, value_parser = clap::value_parser!(u64).range(1..))]
    interval_seconds: u64,
}

#[derive(Clone, Debug)]
struct WaveState {
    checkpoint_sha256: String,
    completed: bool,
    zero_backfill: bool,
    requested_slots: BTreeMap<Task, usize>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.backfill_raw_receipts {
        let report = backfill_raw_receipts(&args.root)?;
        atomic_json(
            &args.root.join("ops/causal-receipt-backfill.v2.json"),
            &report,
        )?;
        println!("{}", serde_json::to_string(&report)?);
    }
    loop {
        match rebuild(&args.root) {
            Ok((ledger_v2, ledger_v3, audit)) => {
                atomic_json(
                    &args.root.join("training/causal-evidence-ledger.v2.json"),
                    &serde_json::to_value(&ledger_v2)?,
                )?;
                atomic_json(
                    &args.root.join("training/causal-evidence-ledger.v3.json"),
                    &serde_json::to_value(&ledger_v3)?,
                )?;
                atomic_json(&args.root.join("ops/causal-evidence-audit.v2.json"), &audit)?;
                atomic_json(
                    &args.root.join("ops/causal-pair-repair-queue.v1.json"),
                    &json!({
                        "schema": "CausalPairRepairQueue.v1",
                        "generated_at_ms": now_ms(),
                        "repairs": ledger_v2.pair_repairs,
                    }),
                )?;
                println!("{}", serde_json::to_string(&audit)?);
            }
            Err(error) => {
                let failure = json!({
                    "schema": "CausalEvidenceAuditError.v2",
                    "generated_at_ms": now_ms(),
                    "authoritative": false,
                    "production_gate_open": {"binary": false, "regression": false},
                    "error": format!("{error:#}"),
                });
                atomic_json(
                    &args.root.join("ops/causal-evidence-audit.v2.json"),
                    &failure,
                )?;
                eprintln!("{}", serde_json::to_string(&failure)?);
                if args.once {
                    return Err(error);
                }
            }
        }
        if args.once {
            return Ok(());
        }
        thread::sleep(Duration::from_secs(args.interval_seconds));
    }
}

mod epochs;
mod rebuild;
mod receipts;
pub(super) use epochs::*;
pub(super) use rebuild::*;
pub(super) use receipts::*;

#[cfg(test)]
#[path = "jope_causal_evidence/tests.rs"]
mod tests;
