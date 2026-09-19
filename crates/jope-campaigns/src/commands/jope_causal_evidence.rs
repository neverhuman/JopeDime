//! Rebuild receipt-accurate, task-scoped causal evidence and fail promotion closed.

use anyhow::{ensure, Context, Result};
use clap::Parser;
use crate::selfplay_ingest::{ingest_receipt_with_manifest_connection, open_corpus};
use crate::selfplay_runtime::{
    atomic_json, default_root_str, now_ms, read_json, sha256_bytes, sha256_file,
};
use crate::selfplay_shadow::{
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

pub fn main(args: Vec<String>) -> Result<()> {
    let args = Args::parse_from(&args);
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

fn rebuild(root: &Path) -> Result<(CausalEvidenceLedgerV2, CausalEvidenceLedgerV3, Value)> {
    let (accepted, accepted_lineage_valid) = accepted_parents(root)?;
    let waves = wave_states(root)?;
    let mut requested_slots = Task::ALL
        .into_iter()
        .map(|task| (task, 0))
        .collect::<BTreeMap<_, _>>();
    for wave in waves.values() {
        if !wave.completed {
            continue;
        }
        for task in Task::ALL {
            if accepted.get(&task).map(Sha256Digest::as_str)
                == Some(wave.checkpoint_sha256.as_str())
            {
                *requested_slots.entry(task).or_default() +=
                    wave.requested_slots.get(&task).copied().unwrap_or_default();
            }
        }
    }
    let corpus = root.join("corpus/strict-test-exact.sqlite");
    let connection = Connection::open_with_flags(
        &corpus,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
    )?;
    connection.busy_timeout(Duration::from_secs(60))?;
    let receipts = receipt_rows(&connection, &accepted, accepted_lineage_valid, &waves)?;
    let generated_at_ms = now_ms();
    let mut ledger_v2 = build_causal_evidence_ledger_v2(
        &accepted,
        &requested_slots,
        receipts.clone(),
        generated_at_ms,
    )?;
    apply_permanent_repair_failures(root, &mut ledger_v2)?;
    let (epochs, policy_pins_match, logged_slots, permanent_missing_slots) =
        load_promotion_epochs(root, &accepted)?;
    let ledger_v3 = build_causal_evidence_ledger_v3(
        ledger_v2.clone(),
        &epochs,
        &receipts,
        &policy_pins_match,
        &logged_slots,
        &permanent_missing_slots,
        generated_at_ms,
    )?;
    let binary = &ledger_v2.tasks[&Task::Binary];
    let regression = &ledger_v2.tasks[&Task::Regression];
    let unreplayable = unreplayable_pair_disposition(root, generated_at_ms)?;
    let audit = json!({
        "schema": "CausalEvidenceAudit.v3",
        "generated_at_ms": generated_at_ms,
        "authoritative": true,
        "source": "causal_evaluation_receipts_v2 joined to immutable wave manifests",
        "all_history_ledger": root.join("training/causal-evidence-ledger.v2.json"),
        "promotion_ledger": root.join("training/causal-evidence-ledger.v3.json"),
        "non_authoritative_pre_v2_ledger": root.join("training/causal-evidence-ledger.v1.json"),
        "promotion_authority": "only CausalEvidenceLedger.v3 promotion_epoch can open production",
        "historical_direct_outcome_lower_bound_non_authoritative": {
            "binary": HISTORICAL_BINARY_LOWER_BOUND,
            "regression": HISTORICAL_REGRESSION_LOWER_BOUND,
            "note": "286/776 is preserved only as a historical direct-outcome lower bound and cannot open a gate.",
        },
        "receipt_level_complete_pairs": {
            "binary": binary.complete_pairs,
            "regression": regression.complete_pairs,
        },
        "requested_slot_validity": {
            "binary": binary.proposal_validity,
            "regression": regression.proposal_validity,
        },
        "pair_repair_queue_depth": ledger_v2.pair_repairs.len(),
        "unreplayable_pair_disposition": unreplayable,
        "production_gate_open": {
            "binary": ledger_v3.promotion_epoch.get(&Task::Binary).is_some_and(|row| row.production_gate_open),
            "regression": ledger_v3.promotion_epoch.get(&Task::Regression).is_some_and(|row| row.production_gate_open),
        },
        "gate_reasons": {
            "binary": ledger_v3.promotion_epoch.get(&Task::Binary).map(|row| &row.gate_reasons),
            "regression": ledger_v3.promotion_epoch.get(&Task::Regression).map(|row| &row.gate_reasons),
        },
        "legacy_cumulative_gate_non_authoritative": {
            "binary": binary.production_gate_open,
            "regression": regression.production_gate_open,
        }
    });
    Ok((ledger_v2, ledger_v3, audit))
}

fn write_immutable(path: &Path, bytes: &[u8]) -> Result<()> {
    if path.is_file() {
        ensure!(
            fs::read(path)? == bytes,
            "immutable unreplayable-pair object collision at {}",
            path.display()
        );
        return Ok(());
    }
    fs::create_dir_all(path.parent().context("immutable object has no parent")?)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o444)
        .open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    let mut permissions = file.metadata()?.permissions();
    permissions.set_mode(0o444);
    fs::set_permissions(path, permissions)?;
    Ok(())
}

fn retained_evaluator(root: &Path, digest: &str) -> Result<bool> {
    let path = root
        .join("training/evaluator-vault")
        .join(digest)
        .join("corpus_eval");
    Ok(path.is_file() && sha256_file(&path)? == digest)
}

fn failure_class(root: &Path, entry: &Value) -> Result<&'static str> {
    let Some(evaluator) = entry
        .get("original_evaluator_sha256")
        .and_then(Value::as_str)
    else {
        return Ok("expected_evaluator_unrecorded");
    };
    ensure!(
        evaluator.len() == 64
            && evaluator
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "repair state contains an invalid expected evaluator digest"
    );
    if !retained_evaluator(root, evaluator)? {
        return Ok("evaluator_binary_unavailable");
    }
    if entry
        .get("identical_failures")
        .and_then(Value::as_u64)
        .unwrap_or_default()
        >= 3
    {
        return Ok("repeated_identical_failure");
    }
    Ok("permanent_repair_failure")
}

/// Publish permanent causal failures without evaluator substitution or outcome
/// imputation. Pair identities live in a content-addressed JSONL object so the
/// summary stays small and can be grouped without exposing a mutable ID list.
fn unreplayable_pair_disposition(root: &Path, generated_at_ms: u128) -> Result<Value> {
    let path = root.join("ops/causal-pair-repair-state.v1.json");
    let mut identities = BTreeMap::<(String, String), Value>::new();
    let mut groups = BTreeMap::<(String, String), usize>::new();
    if path.is_file() {
        let state = read_json(&path)?;
        ensure!(
            state.get("schema").and_then(Value::as_str) == Some("CausalPairRepairExecutorState.v1"),
            "causal pair repair state schema drift"
        );
        if let Some(entries) = state.get("entries").and_then(Value::as_object) {
            for entry in entries.values() {
                if entry.get("status").and_then(Value::as_str) != Some("permanent_failure") {
                    continue;
                }
                let pair_id = entry
                    .get("pair_id")
                    .and_then(Value::as_str)
                    .context("permanent failure lacks pair identity")?;
                let lane = entry
                    .get("missing_lane")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown");
                let expected = entry
                    .get("original_evaluator_sha256")
                    .and_then(Value::as_str);
                let class = failure_class(root, entry)?;
                *groups
                    .entry((
                        expected.unwrap_or("unrecorded").to_owned(),
                        class.to_owned(),
                    ))
                    .or_default() += 1;
                identities.insert(
                    (pair_id.to_owned(), lane.to_owned()),
                    json!({
                        "pair_id": pair_id,
                        "task": entry.get("task").and_then(Value::as_str),
                        "missing_lane": lane,
                        "expected_evaluator_sha256": expected,
                        "failure_class": class,
                        "promotion_eligible": false,
                    }),
                );
            }
        }
    }
    let mut bytes = Vec::new();
    for identity in identities.values() {
        bytes.extend(serde_json::to_vec(identity)?);
        bytes.push(b'\n');
    }
    let identities_sha256 = sha256_bytes(&bytes);
    let identities_path = root
        .join("training/unreplayable-pair-identities")
        .join(format!("{identities_sha256}.v2.jsonl"));
    write_immutable(&identities_path, &bytes)?;
    let grouped_counts = groups
        .into_iter()
        .map(|((evaluator, class), count)| {
            json!({
                "expected_evaluator_sha256": (evaluator != "unrecorded").then_some(evaluator),
                "failure_class": class,
                "count": count,
            })
        })
        .collect::<Vec<_>>();
    let disposition = json!({
        "schema": "JopeUnreplayablePairDisposition.v2",
        "generated_at_ms": generated_at_ms,
        "policy": "explicit-unreplayable-no-imputation",
        "source_repair_state": path,
        "source_repair_state_sha256": path.is_file().then(|| sha256_file(&path)).transpose()?,
        "pair_identities_path": identities_path,
        "pair_identities_sha256": identities_sha256,
        "count": identities.len(),
        "unique_pair_count": identities.keys().map(|(pair, _)| pair).collect::<BTreeSet<_>>().len(),
        "grouped_counts": grouped_counts,
        "promotion_eligible": false,
        "evaluator_substitution_allowed": false,
        "outcome_imputation_allowed": false,
    });
    atomic_json(
        &root.join("training/unreplayable-pair-disposition.v2.json"),
        &disposition,
    )?;
    Ok(disposition)
}

type EpochAccounting = (
    BTreeMap<Task, CausalEvidenceEpochV1>,
    BTreeMap<Task, bool>,
    BTreeMap<Task, usize>,
    BTreeMap<Task, usize>,
);

fn load_promotion_epochs(
    root: &Path,
    accepted: &BTreeMap<Task, Sha256Digest>,
) -> Result<EpochAccounting> {
    let mut epochs = BTreeMap::new();
    let mut policy_pins_match = BTreeMap::new();
    let mut logged_slots = BTreeMap::new();
    let mut permanent_missing_slots = BTreeMap::new();
    for task in Task::ALL {
        let path = root
            .join("training/causal-evidence-epochs")
            .join(format!("{}.v1.json", task.as_str()));
        if !path.is_file() {
            continue;
        }
        let epoch: CausalEvidenceEpochV1 = serde_json::from_value(read_json(&path)?)?;
        epoch.validate()?;
        ensure!(
            epoch.task == task
                && accepted.get(&task) == Some(&epoch.accepted_parent_checkpoint_sha256),
            "{task} promotion epoch is not pinned to the current accepted parent"
        );
        let accounting_path = root
            .join("training/causal-evidence-epoch-accounting")
            .join(format!("{}.v1.json", task.as_str()));
        let mut pins = false;
        let mut slots = 0;
        let mut missing = 0;
        if accounting_path.is_file() {
            let accounting = read_json(&accounting_path)?;
            ensure!(
                accounting.get("schema").and_then(Value::as_str)
                    == Some("CausalEvidenceEpochAccounting.v1")
                    && accounting.get("epoch_sha256").and_then(Value::as_str)
                        == Some(epoch.identity()?.as_str()),
                "{task} epoch accounting identity drift"
            );
            pins = accounting
                .get("policy_pins_match")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            slots = accounting
                .get("logged_slots")
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize;
            missing = accounting
                .get("permanent_missing_slots")
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize;
        }
        policy_pins_match.insert(task, pins);
        logged_slots.insert(task, slots);
        permanent_missing_slots.insert(task, missing);
        epochs.insert(task, epoch);
    }
    Ok((
        epochs,
        policy_pins_match,
        logged_slots,
        permanent_missing_slots,
    ))
}

fn apply_permanent_repair_failures(root: &Path, ledger: &mut CausalEvidenceLedgerV2) -> Result<()> {
    let path = root.join("ops/causal-pair-repair-state.v1.json");
    if !path.is_file() {
        return Ok(());
    }
    let state = read_json(&path)?;
    ensure!(
        state.get("schema").and_then(Value::as_str) == Some("CausalPairRepairExecutorState.v1"),
        "causal pair repair state schema drift"
    );
    let permanent = state
        .get("entries")
        .and_then(Value::as_object)
        .into_iter()
        .flatten()
        .filter_map(|(_, entry)| {
            if entry.get("status").and_then(Value::as_str) != Some("permanent_failure") {
                return None;
            }
            Some((
                entry.get("pair_id")?.as_str()?.to_owned(),
                entry.get("missing_lane")?.as_str()?.to_owned(),
            ))
        })
        .collect::<std::collections::BTreeSet<_>>();
    for repair in &mut ledger.pair_repairs {
        if permanent.contains(&(
            repair.pair_id.as_str().to_owned(),
            match repair.missing_lane {
                CausalLane::NativeDlDirect => "native_dl_direct",
                CausalLane::MatchedGpComparator => "matched_gp_comparator",
            }
            .to_owned(),
        )) {
            repair.permanent_failure = true;
        }
    }
    Ok(())
}

fn accepted_parents(root: &Path) -> Result<(BTreeMap<Task, Sha256Digest>, bool)> {
    let pointer = read_json(&root.join("training/accepted-by-task.json"))?;
    ensure!(
        pointer.get("schema").and_then(Value::as_str) == Some("JopeSelfplayAcceptedByTask.v1"),
        "accepted-by-task pointer schema is not live v1"
    );
    let mut output = BTreeMap::new();
    let mut valid = true;
    for task in Task::ALL {
        let accepted_row = pointer
            .get("tasks")
            .and_then(|value| value.get(task.as_str()));
        let anchor;
        let row = if let Some(row) = accepted_row {
            row
        } else if matches!(task, Task::Multiclass | Task::Timeseries) {
            let path = root
                .join("training/bootstrap-evaluation-anchors")
                .join(format!("{}.v1.json", task.as_str()));
            if !path.is_file() {
                continue;
            }
            anchor = read_json(&path)?;
            ensure!(
                anchor.get("schema").and_then(Value::as_str)
                    == Some("TaskBootstrapEvaluationAnchor.v1")
                    && anchor.get("task").and_then(Value::as_str) == Some(task.as_str())
                    && anchor.get("accepted").and_then(Value::as_bool) == Some(false),
                "{task} bootstrap evaluation anchor is not an unaccepted v1 anchor"
            );
            &anchor
        } else {
            anyhow::bail!("accepted pointer lacks {task}");
        };
        let digest = row
            .get("checkpoint_sha256")
            .and_then(Value::as_str)
            .with_context(|| format!("{task} accepted pointer lacks checkpoint digest"))?;
        let path = Path::new(
            row.get("checkpoint")
                .and_then(Value::as_str)
                .with_context(|| format!("{task} accepted pointer lacks checkpoint path"))?,
        );
        if !path.is_file() || sha256_file(path)? != digest {
            valid = false;
        }
        output.insert(task, Sha256Digest::parse(digest.to_owned())?);
    }
    Ok((output, valid))
}

fn wave_states(root: &Path) -> Result<BTreeMap<u64, WaveState>> {
    let mut output = BTreeMap::new();
    let waves = root.join("waves");
    if !waves.is_dir() {
        return Ok(output);
    }
    for entry in std::fs::read_dir(waves)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(wave) = parse_wave(&name) else {
            continue;
        };
        if wave < 191 {
            continue;
        }
        let summary_path = entry.path().join("summary.json");
        let cohort_path = entry.path().join("cohort.json");
        if !summary_path.is_file() || !cohort_path.is_file() {
            continue;
        }
        let summary = read_json(&summary_path)?;
        let eligibility = summary.get("wave_summary").and_then(|value| {
            value
                .get("evidence_eligibility")
                .or_else(|| summary.get("evidence_eligibility"))
        });
        let completed = summary.get("immutable").and_then(Value::as_bool) == Some(true)
            && eligibility
                .and_then(|value| value.get("promotion"))
                .and_then(Value::as_bool)
                == Some(true)
            && eligibility
                .and_then(|value| value.get("recovery_accounting_only"))
                .and_then(Value::as_bool)
                != Some(true);
        let zero_backfill = summary
            .get("cross_context_backfill_rows")
            .and_then(Value::as_u64)
            == Some(0);
        let checkpoint_sha256 = summary
            .get("checkpoint_sha256")
            .and_then(Value::as_str)
            .context("wave summary lacks checkpoint identity")?
            .to_owned();
        let cohort = read_json(&cohort_path)?;
        let mut requested_slots = Task::ALL
            .into_iter()
            .map(|task| (task, 0))
            .collect::<BTreeMap<_, _>>();
        for row in cohort
            .get("entries")
            .and_then(Value::as_array)
            .context("wave cohort lacks entries")?
        {
            let task: Task = row
                .get("task")
                .and_then(Value::as_str)
                .context("cohort entry lacks task")?
                .parse()?;
            *requested_slots.entry(task).or_default() += 8;
        }
        output.insert(
            wave,
            WaveState {
                checkpoint_sha256,
                completed,
                zero_backfill,
                requested_slots,
            },
        );
    }
    Ok(output)
}

fn receipt_rows(
    connection: &Connection,
    accepted: &BTreeMap<Task, Sha256Digest>,
    accepted_lineage_valid: bool,
    waves: &BTreeMap<u64, WaveState>,
) -> Result<Vec<CausalEvaluationReceiptV2>> {
    let exists: bool = connection.query_row(
        "SELECT EXISTS(
           SELECT 1 FROM sqlite_master
           WHERE type='table' AND name='causal_evaluation_receipts_v2'
         )",
        [],
        |row| row.get(0),
    )?;
    if !exists {
        return Ok(Vec::new());
    }
    let mut statement = connection.prepare(
        "SELECT receipt_sha256,manifest_sha256,candidate_sha256,task,
                context_identity,accepted_parent_checkpoint_sha256,wave,lane,
                pair_index,pair_id,score,success,proposal_valid,runtime_seconds,
                ancestry,cross_context_backfill,valid_lineage,canary
           FROM causal_evaluation_receipts_v2
          WHERE wave>=191
          ORDER BY wave,task,context_identity,pair_id,lane,receipt_sha256",
    )?;
    let raw = statement
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, u64>(6)?,
                row.get::<_, String>(7)?,
                row.get::<_, usize>(8)?,
                row.get::<_, String>(9)?,
                row.get::<_, Option<f64>>(10)?,
                row.get::<_, bool>(11)?,
                row.get::<_, bool>(12)?,
                row.get::<_, f64>(13)?,
                row.get::<_, String>(14)?,
                row.get::<_, bool>(15)?,
                row.get::<_, bool>(16)?,
                row.get::<_, bool>(17)?,
            ))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    let mut output = Vec::with_capacity(raw.len());
    for (
        receipt,
        manifest,
        candidate,
        task,
        context,
        parent,
        wave_number,
        lane,
        pair_index,
        pair_id,
        score,
        success,
        proposal_valid,
        runtime_seconds,
        ancestry,
        cross_context_backfill,
        valid_lineage,
        canary,
    ) in raw
    {
        let task: Task = task.parse()?;
        let Some(wave) = waves.get(&wave_number) else {
            continue;
        };
        let lane = match lane.as_str() {
            "native_dl_direct" => CausalLane::NativeDlDirect,
            "matched_gp_comparator" => CausalLane::MatchedGpComparator,
            _ => continue,
        };
        let ancestry = match ancestry.as_str() {
            "none" => DlAncestryKind::None,
            "accepted_dl" => DlAncestryKind::AcceptedDl,
            "shadow_dl" => DlAncestryKind::ShadowDl,
            _ => DlAncestryKind::Mixed,
        };
        output.push(CausalEvaluationReceiptV2 {
            schema: CAUSAL_EVALUATION_RECEIPT_SCHEMA.to_owned(),
            receipt_sha256: Sha256Digest::parse(receipt)?,
            manifest_sha256: Sha256Digest::parse(manifest)?,
            candidate_sha256: Sha256Digest::parse(candidate)?,
            task,
            context_identity: context,
            accepted_parent_checkpoint_sha256: Sha256Digest::parse(parent.clone())?,
            wave: wave_number,
            lane,
            pair_index,
            pair_id: Sha256Digest::parse(pair_id)?,
            score,
            success,
            proposal_valid,
            runtime_ms: (runtime_seconds.max(0.0) * 1_000.0).round() as u64,
            ancestry,
            cross_context_backfill: cross_context_backfill || !wave.zero_backfill,
            completed_causal_wave: wave.completed,
            valid_lineage: valid_lineage
                && accepted_lineage_valid
                && accepted.get(&task).map(Sha256Digest::as_str) == Some(parent.as_str())
                && wave.checkpoint_sha256 == parent,
            canary,
        });
    }
    Ok(output)
}

fn backfill_raw_receipts(root: &Path) -> Result<Value> {
    let corpus = root.join("corpus/strict-test-exact.sqlite");
    let mut connection = open_corpus(&corpus)?;
    // Receipt replay is idempotent and each row is protected by its immutable
    // triple key. WAL + NORMAL remains crash-consistent while avoiding one
    // durable fsync for every historical job; a lost tail is replayed on the
    // next run.
    connection.pragma_update(None, "synchronous", "NORMAL")?;
    let backlog = Connection::open_with_flags(
        root.join("ops/backlog.sqlite"),
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
    )?;
    backlog.busy_timeout(Duration::from_secs(60))?;
    let mut statement = backlog
        .prepare("SELECT job_id,manifest FROM claims WHERE state='ingested' ORDER BY job_id")?;
    let claims = statement
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                PathBuf::from(row.get::<_, String>(1)?),
            ))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    drop(statement);
    drop(backlog);
    let already = connection.query_row(
        "SELECT COUNT(*) FROM causal_evaluation_receipts_v2",
        [],
        |row| row.get::<_, u64>(0),
    )?;
    let mut eligible_jobs = 0_u64;
    let mut backfilled_jobs = 0_u64;
    let mut rows = 0_u64;
    for (job_id, manifest) in claims {
        let Some(wave) = manifest
            .components()
            .filter_map(|part| part.as_os_str().to_str())
            .find_map(parse_wave)
        else {
            continue;
        };
        if wave < 191 {
            continue;
        }
        eligible_jobs += 1;
        let output = root.join("receipts").join(&job_id);
        if !output.join("SHARD_OK").is_file() {
            continue;
        }
        let summary =
            ingest_receipt_with_manifest_connection(&mut connection, &job_id, &output, &manifest)?;
        backfilled_jobs += 1;
        rows += summary.rows as u64;
    }
    let after = connection.query_row(
        "SELECT COUNT(*) FROM causal_evaluation_receipts_v2",
        [],
        |row| row.get::<_, u64>(0),
    )?;
    Ok(json!({
        "schema": "CausalReceiptBackfill.v2",
        "generated_at_ms": now_ms(),
        "eligible_jobs": eligible_jobs,
        "backfilled_jobs": backfilled_jobs,
        "raw_rows_replayed": rows,
        "receipt_rows_before": already,
        "receipt_rows_after": after,
        "append_only_rows_added": after.saturating_sub(already),
    }))
}

fn parse_wave(value: &str) -> Option<u64> {
    value.strip_prefix("strict-wave-")?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::selfplay_shadow::{
        build_causal_evidence_ledger_v2, causal_pair_id, DlAncestryKind, PairRepairRequestV1,
        Sha256Digest, Task,
    };
    use std::fs;

    #[test]
    fn wave_189_is_excluded_and_receipt_era_starts_at_191() {
        assert_eq!(parse_wave("strict-wave-000189"), Some(189));
        assert_eq!(parse_wave("strict-wave-000191"), Some(191));
        assert_eq!(parse_wave("shadow-wave-000191"), None);
    }

    #[test]
    fn executor_permanent_failure_is_reflected_in_the_authoritative_queue() {
        let temporary = tempfile::tempdir().unwrap();
        fs::create_dir_all(temporary.path().join("ops")).unwrap();
        let parent = Sha256Digest::parse("a".repeat(64)).unwrap();
        let pair_id = causal_pair_id(&parent, 191, Task::Binary, "context", 0);
        let direct = CausalEvaluationReceiptV2 {
            schema: CAUSAL_EVALUATION_RECEIPT_SCHEMA.to_owned(),
            receipt_sha256: Sha256Digest::of_bytes(b"receipt"),
            manifest_sha256: Sha256Digest::of_bytes(b"manifest"),
            candidate_sha256: Sha256Digest::of_bytes(b"candidate"),
            task: Task::Binary,
            context_identity: "context".to_owned(),
            accepted_parent_checkpoint_sha256: parent.clone(),
            wave: 191,
            lane: CausalLane::NativeDlDirect,
            pair_index: 0,
            pair_id: pair_id.clone(),
            score: Some(1.0),
            success: true,
            proposal_valid: true,
            runtime_ms: 1,
            ancestry: DlAncestryKind::AcceptedDl,
            cross_context_backfill: false,
            completed_causal_wave: true,
            valid_lineage: true,
            canary: false,
        };
        let mut ledger = build_causal_evidence_ledger_v2(
            &BTreeMap::from([(Task::Binary, parent)]),
            &BTreeMap::from([(Task::Binary, 8)]),
            vec![direct],
            1,
        )
        .unwrap();
        assert_eq!(
            ledger.pair_repairs,
            vec![PairRepairRequestV1 {
                pair_id: pair_id.clone(),
                task: Task::Binary,
                missing_lane: CausalLane::MatchedGpComparator,
                original_manifest_sha256: Sha256Digest::of_bytes(b"manifest"),
                permanent_failure: false,
            }]
        );
        atomic_json(
            &temporary
                .path()
                .join("ops/causal-pair-repair-state.v1.json"),
            &json!({
                "schema": "CausalPairRepairExecutorState.v1",
                "updated_at_ms": 1,
                "entries": {
                    format!("{pair_id}:matched_gp_comparator"): {
                        "pair_id": pair_id,
                        "missing_lane": "matched_gp_comparator",
                        "status": "permanent_failure"
                    }
                }
            }),
        )
        .unwrap();
        apply_permanent_repair_failures(temporary.path(), &mut ledger).unwrap();
        assert!(ledger.pair_repairs[0].permanent_failure);
    }

    #[test]
    fn missing_evaluator_is_classified_unreplayable_without_imputation() {
        let temporary = tempfile::tempdir().unwrap();
        fs::create_dir_all(temporary.path().join("ops")).unwrap();
        atomic_json(
            &temporary
                .path()
                .join("ops/causal-pair-repair-state.v1.json"),
            &json!({
                "schema": "CausalPairRepairExecutorState.v1",
                "entries": {
                    "pair": {
                        "pair_id": "pair",
                        "status": "permanent_failure"
                    },
                    "replayable": {
                        "pair_id": "replayable",
                        "status": "permanent_failure",
                        "original_evaluator_sha256": "a".repeat(64)
                    }
                }
            }),
        )
        .unwrap();
        let disposition = unreplayable_pair_disposition(temporary.path(), 10).unwrap();
        assert_eq!(disposition["schema"], "JopeUnreplayablePairDisposition.v2");
        assert_eq!(disposition["count"], 2);
        assert_eq!(disposition["unique_pair_count"], 2);
        assert_eq!(disposition["promotion_eligible"], false);
        assert_eq!(disposition["evaluator_substitution_allowed"], false);
        assert_eq!(disposition["outcome_imputation_allowed"], false);
        let object = PathBuf::from(disposition["pair_identities_path"].as_str().unwrap());
        assert!(object.is_file());
        assert_eq!(
            sha256_file(&object).unwrap(),
            disposition["pair_identities_sha256"]
        );
        assert!(disposition.get("pair_ids").is_none());
    }
}
