//! Fail-closed SCQ queue audit/hold control and robust cross-dataset seed export.
//!
//! Queue mutations are deliberately limited to `hold`. Running/dispatching
//! work is never a candidate. Every candidate must join to a submitted backlog
//! claim whose immutable manifest hash and freshness scope validate.

use anyhow::{bail, Context, Result};
use clap::Parser;
use jope_dime_regressor::paths;
use jope_dime_regressor::selfplay_runtime::{
    canonical_json_bytes, default_root_str, default_scq_str, manifest_epoch,
    manifest_freshness_scoped, now_ms, scq_json, sha256_bytes, sha256_file,
};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use rusqlite::{params, Connection, OpenFlags, OptionalExtension, TransactionBehavior};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

mod blocker_plan;
mod inventory;
mod recovery;
mod retirement;
mod seed_export;
mod socket;
use blocker_plan::*;
use inventory::*;
use recovery::*;
use retirement::*;
use seed_export::*;
use socket::*;

const LEGACY_EPOCH: &str = "legacy-jope-testonly";
const LEGACY_PROJECT: &str = "jope-testonly";
const SELFPLAY_PROJECT: &str = "jope-selfplay-eval";
const MANIFEST_SCHEMA: &str = "JopeFlywheelCorpusEvalShard.v1";
const DESIRED_BLOCKER_QUERY: &str = "SELECT p.state,p.manifest_path,p.manifest_sha256,p.job_id
     FROM desired_pairs d
     CROSS JOIN pairs p INDEXED BY sqlite_autoindex_pairs_1
       ON p.task=d.task AND p.dataset_hash=d.dataset_hash
      AND p.semantic_sha256=d.semantic_sha256
      AND p.protocol_sha256=d.protocol_sha256
     WHERE p.state IN ('reserved','submitted','held')
       AND p.receipt_sha256 IS NULL";

#[derive(Parser, Debug)]
#[command(name = "jope-queue-control")]
struct Args {
    #[arg(long, default_value = default_root_str())]
    root: PathBuf,
    #[arg(long, default_value = default_scq_str())]
    scq: PathBuf,
    #[arg(long, default_value_os_t = paths::scq_socket())]
    socket: PathBuf,
    /// Hold only provenance-verified queued/starting legacy jope-testonly work.
    #[arg(long)]
    hold_legacy: bool,
    /// Hold only provenance-verified queued/starting older strict waves.
    #[arg(long)]
    hold_superseded_waves: bool,
    /// Cancel held, provenance-verified legacy work and supersede its claims/pairs.
    #[arg(long)]
    retire_held_legacy: bool,
    /// Cancel held, provenance-verified older strict waves and supersede claims/pairs.
    #[arg(long)]
    retire_held_superseded_waves: bool,
    /// Release a bounded bridge of held jobs bound entirely to one checkpoint.
    #[arg(long)]
    release_current_checkpoint_bridge: bool,
    /// Maximum bridge jobs to release (hard-capped at 512).
    #[arg(long, default_value_t = 512)]
    limit: usize,
    /// Required SHA-256 checkpoint identity for the temporary release bridge.
    #[arg(long)]
    active_checkpoint: Option<String>,
    /// Retire only proven-cancelled old manifests blocking a validated wave.
    #[arg(long)]
    retire_blockers_for_wave: Option<PathBuf>,
    /// Repair stale available bindings from an audited blocker retirement.
    #[arg(long)]
    repair_retired_bindings_for_wave: Option<PathBuf>,
    /// Recover completed/launched blockers with irretrievably missing receipts.
    #[arg(long)]
    recover_missing_receipt_blockers_for_wave: Option<PathBuf>,
    /// Validate and write the mutation preview without calling SCQ.
    #[arg(long)]
    preview_only: bool,
    /// Export repeated, robust, task-separated GP genomes from the corpus.
    #[arg(long)]
    export_seed_bank: Option<PathBuf>,
    #[arg(
        long,
        default_value_os_t = paths::corpus_db()
    )]
    corpus: PathBuf,
    #[arg(long, default_value_t = 512)]
    seeds_per_task: usize,
    #[arg(long, default_value_t = 2)]
    minimum_repeat_datasets: usize,
    /// Maximum bounded concurrency for manifest validation and SCQ hold calls.
    #[arg(long, default_value_t = 32)]
    max_hold_workers: usize,
    #[arg(long)]
    receipt_dir: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ValidationMode {
    Audit,
    HoldLegacy,
    HoldSupersededWaves,
    RetireHeldLegacy,
    RetireHeldSupersededWaves,
    ReleaseCurrentCheckpointBridge,
}

#[derive(Clone, Debug)]
struct Claim {
    manifest_sha256: String,
    manifest: PathBuf,
    state: String,
}

#[derive(Clone, Debug, Serialize)]
struct JobRecord {
    job_id: String,
    project: String,
    state: String,
    epoch: String,
    strict_wave: Option<String>,
    generation: Option<String>,
    checkpoint_sha256: Option<String>,
    claim_state: Option<String>,
    manifest: Option<String>,
    manifest_sha256: Option<String>,
    manifest_checked: bool,
    manifest_valid: bool,
    invalid_reason: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct HoldCandidate {
    job_id: String,
    project: String,
    observed_state: String,
    epoch: String,
    strict_wave: Option<String>,
    generation: Option<String>,
    manifest: String,
    manifest_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
struct HoldResult {
    job_id: String,
    state_before_hold: Option<String>,
    hold_requested: bool,
    command_error: Option<String>,
    state_after_hold: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
struct ReleaseResult {
    job_id: String,
    state_before_release: Option<String>,
    release_requested: bool,
    command_error: Option<String>,
    state_after_release: Option<String>,
}

#[derive(Debug, Serialize)]
struct WaveBlockerPlan {
    wave: String,
    wave_dir: String,
    summary_sha256: String,
    cohort_sha256: String,
    checkpoint_sha256: String,
    checkpoint_epoch: String,
    desired_pairs: usize,
    current_wave_active_pairs: usize,
    blocking_pairs: usize,
    blocking_claims: usize,
    exact_submitted_claims: usize,
    blocking_manifests: usize,
    proven_blocking_manifests: usize,
    unproven_blocking_manifests: usize,
    proof_errors: Vec<String>,
    candidates: Vec<HoldCandidate>,
}

#[derive(Debug, Serialize)]
struct RetiredBindingRepairPlan {
    wave: String,
    source_audit_path: String,
    source_audit_sha256: String,
    source_preview_sha256: String,
    manifest_count: usize,
    pair_count: usize,
    candidates: Vec<HoldCandidate>,
}

#[derive(Debug, Serialize)]
struct MissingReceiptRecoveryCandidate {
    job_id: String,
    manifest: String,
    manifest_sha256: String,
    claim_state: String,
    scq_evidence: String,
    desired_pair_count: usize,
    pair_count: usize,
    pairs: Vec<RegistryPairIdentity>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
struct RegistryPairIdentity {
    task: String,
    dataset_hash: String,
    semantic_sha256: String,
    protocol_sha256: String,
}

struct ManifestPairBinding {
    identity: RegistryPairIdentity,
    state: String,
    manifest_path: Option<String>,
    job_id: Option<String>,
    receipt_sha256: Option<String>,
    failure_class: Option<String>,
}

#[derive(Debug, Serialize)]
struct MissingReceiptRecoveryPlan {
    wave: String,
    wave_dir: String,
    desired_pairs: usize,
    blocker_pairs: usize,
    blocker_manifests: usize,
    eligible_desired_pairs: usize,
    eligible_total_pairs: usize,
    eligible_manifests: usize,
    reason_counts: BTreeMap<String, usize>,
    exclusions: Vec<String>,
    candidates: Vec<MissingReceiptRecoveryCandidate>,
}

#[derive(Clone, Debug, Serialize)]
struct RetirementResult {
    job_id: String,
    state_before_cancel: Option<String>,
    cancel_requested: bool,
    command_error: Option<String>,
    state_after_cancel: Option<String>,
    pair_updates: usize,
    claim_updated: bool,
    recovery_from_prior_preview: bool,
    registry_error: Option<String>,
}

#[derive(Debug, Serialize)]
struct QueueReport {
    schema: &'static str,
    generated_at_ms: u128,
    counts_by_project_state_epoch: BTreeMap<String, BTreeMap<String, BTreeMap<String, usize>>>,
    selfplay_counts_by_wave_state: BTreeMap<String, BTreeMap<String, usize>>,
    selfplay_counts_by_generation_state: BTreeMap<String, BTreeMap<String, usize>>,
    newest_strict_wave: Option<String>,
    scheduler_jobs: usize,
    joined_claims: usize,
    valid_manifests: usize,
    invalid_manifests: usize,
    unchecked_manifests: usize,
    unclaimed_jobs: usize,
}

#[derive(Debug, Serialize)]
struct SeedRow {
    task: String,
    semantic_sha256: String,
    canonical_genome: String,
    dataset_count: u64,
    mean_within_dataset_percentile: f64,
    worst_within_dataset_percentile: f64,
    strict_win_count: u64,
    /// True when the seed was selected from accepted or shadow native-DL
    /// evidence (combined DL-origin graduation, capped at floor(limit/3)).
    dl_origin: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.minimum_repeat_datasets < 2 {
        bail!("--minimum-repeat-datasets must be at least two");
    }
    if !(1..=64).contains(&args.max_hold_workers) {
        bail!("--max-hold-workers must be in 1..=64");
    }
    let mutation_count = [
        args.hold_legacy,
        args.hold_superseded_waves,
        args.retire_held_legacy,
        args.retire_held_superseded_waves,
        args.release_current_checkpoint_bridge,
        args.retire_blockers_for_wave.is_some(),
        args.repair_retired_bindings_for_wave.is_some(),
        args.recover_missing_receipt_blockers_for_wave.is_some(),
    ]
    .into_iter()
    .filter(|enabled| *enabled)
    .count();
    if mutation_count > 1 {
        bail!("select at most one queue mutation mode");
    }
    if args.preview_only && mutation_count != 1 {
        bail!("--preview-only requires exactly one queue mutation mode");
    }
    if args.limit == 0 || args.limit > 512 {
        bail!("--limit must be in 1..=512");
    }
    if args.release_current_checkpoint_bridge {
        let checkpoint = args
            .active_checkpoint
            .as_deref()
            .context("--release-current-checkpoint-bridge requires --active-checkpoint")?;
        if !is_sha256(checkpoint) {
            bail!("--active-checkpoint is not a SHA-256 value");
        }
    } else if args.active_checkpoint.is_some() {
        bail!("--active-checkpoint requires --release-current-checkpoint-bridge");
    }

    let backlog = args.root.join("ops/backlog.sqlite");
    let claims = load_claims(&backlog)?;
    let initial_jobs = scheduler_jobs(&args.scq, &args.socket)?;
    let mode = if args.hold_legacy {
        ValidationMode::HoldLegacy
    } else if args.hold_superseded_waves {
        ValidationMode::HoldSupersededWaves
    } else if args.retire_held_legacy {
        ValidationMode::RetireHeldLegacy
    } else if args.retire_held_superseded_waves {
        ValidationMode::RetireHeldSupersededWaves
    } else if args.release_current_checkpoint_bridge {
        ValidationMode::ReleaseCurrentCheckpointBridge
    } else {
        ValidationMode::Audit
    };
    let records = join_inventory(&initial_jobs, &claims, mode, args.max_hold_workers)?;
    let report = queue_report(&records);

    let receipt_dir = args
        .receipt_dir
        .unwrap_or_else(|| args.root.join("ops/queue-control-receipts"));
    let mut output = json!({"queue": report});

    if let Some(wave_dir) = args.recover_missing_receipt_blockers_for_wave.as_deref() {
        let scq_var = args
            .socket
            .parent()
            .context("SCQ socket has no parent directory")?;
        let plan = build_missing_receipt_recovery_plan(
            &args.root,
            wave_dir,
            &args.root.join("ops/pair-registry.sqlite"),
            &backlog,
            &states_by_id(&initial_jobs),
            &receipt_dir,
            &scq_var.join("events.jsonl"),
            &scq_var.join("runs"),
            &args.scq,
            &args.socket,
        )?;
        let preview = json!({
            "schema": "JopeMissingReceiptRecoveryPreview.v1",
            "action": "recover_missing_receipt_blockers_for_wave",
            "created_at_ms": now_ms(),
            "plan": &plan,
        });
        let (preview_path, preview_sha256) =
            write_content_addressed(&receipt_dir, "missing-receipt-preview", &preview)?;
        output["missing_receipt_recovery_preview"] = json!({
            "preview_path": preview_path,
            "preview_sha256": preview_sha256,
            "blocker_pairs": plan.blocker_pairs,
            "blocker_manifests": plan.blocker_manifests,
            "eligible_desired_pairs": plan.eligible_desired_pairs,
            "eligible_total_pairs": plan.eligible_total_pairs,
            "eligible_manifests": plan.eligible_manifests,
            "reason_counts": plan.reason_counts,
            "mutation_performed": false,
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
        if !args.preview_only {
            bail!(
                "missing-receipt recovery requires explicit apply authorization \
                 after reviewing preview {}",
                preview_path.display()
            );
        }
        return Ok(());
    }

    if let Some(wave_dir) = args.repair_retired_bindings_for_wave.as_deref() {
        let plan = build_retired_binding_repair_plan(
            wave_dir,
            &args.root.join("ops/pair-registry.sqlite"),
            &backlog,
            &receipt_dir,
        )?;
        let preview = json!({
            "schema": "JopeRetiredBindingRepairPreview.v1",
            "action": "repair_retired_bindings_for_wave",
            "created_at_ms": now_ms(),
            "plan": &plan,
        });
        let (preview_path, preview_sha256) =
            write_content_addressed(&receipt_dir, "binding-repair-preview", &preview)?;
        if args.preview_only {
            output["retired_binding_repair_preview"] = json!({
                "preview_path": preview_path,
                "preview_sha256": preview_sha256,
                "manifest_count": plan.manifest_count,
                "pair_count": plan.pair_count,
                "mutation_performed": false,
            });
            println!("{}", serde_json::to_string_pretty(&output)?);
            return Ok(());
        }
        let updated = apply_retired_binding_repair(
            &args.root.join("ops/pair-registry.sqlite"),
            &backlog,
            &plan,
            &preview_sha256,
        )?;
        let audit = json!({
            "schema": "JopeRetiredBindingRepairAudit.v1",
            "action": "repair_retired_bindings_for_wave",
            "created_at_ms": now_ms(),
            "preview_path": preview_path,
            "preview_sha256": preview_sha256,
            "source_audit_sha256": plan.source_audit_sha256,
            "manifest_count": plan.manifest_count,
            "pair_count": plan.pair_count,
            "updated_pairs": updated,
        });
        let (audit_path, audit_sha256) =
            write_content_addressed(&receipt_dir, "binding-repair-audit", &audit)?;
        output["retired_binding_repair"] = json!({
            "preview_path": preview_path,
            "preview_sha256": preview_sha256,
            "audit_path": audit_path,
            "audit_sha256": audit_sha256,
            "updated_pairs": updated,
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
        return Ok(());
    }

    if let Some(wave_dir) = args.retire_blockers_for_wave.as_deref() {
        let scq_events = args
            .socket
            .parent()
            .context("SCQ socket has no parent directory")?
            .join("events.jsonl");
        let plan = build_wave_blocker_plan(
            &args.root,
            wave_dir,
            &args.root.join("ops/pair-registry.sqlite"),
            &claims,
            &states_by_id(&initial_jobs),
            &receipt_dir,
            &scq_events,
        )?;
        let preview = json!({
            "schema": "JopeWaveBlockerRetirementPreview.v1",
            "action": "retire_blockers_for_wave",
            "created_at_ms": now_ms(),
            "plan": &plan,
        });
        let (preview_path, preview_sha256) =
            write_content_addressed(&receipt_dir, "blocker-preview", &preview)?;
        if args.preview_only {
            output["wave_blocker_preview"] = json!({
                "preview_path": preview_path,
                "preview_sha256": preview_sha256,
                "desired_pairs": plan.desired_pairs,
                "current_wave_active_pairs": plan.current_wave_active_pairs,
                "blocking_pairs": plan.blocking_pairs,
                "blocking_claims": plan.blocking_claims,
                "exact_submitted_claims": plan.exact_submitted_claims,
                "blocking_manifests": plan.blocking_manifests,
                "proven_blocking_manifests": plan.proven_blocking_manifests,
                "unproven_blocking_manifests": plan.unproven_blocking_manifests,
                "proof_errors": &plan.proof_errors,
                "mutation_performed": false,
            });
            println!("{}", serde_json::to_string_pretty(&output)?);
            return Ok(());
        }
        let fresh_jobs = scheduler_jobs(&args.scq, &args.socket)?;
        let fresh_states = states_by_id(&fresh_jobs);
        let fresh_claims = load_claims(&backlog)?;
        revalidate_wave_blocker_plan(
            &plan,
            &fresh_states,
            &fresh_claims,
            &receipt_dir,
            &scq_events,
        )?;
        let mut results = plan
            .candidates
            .iter()
            .map(|candidate| RetirementResult {
                job_id: candidate.job_id.clone(),
                state_before_cancel: (candidate.observed_state == "cancelled")
                    .then(|| "cancelled".to_owned()),
                cancel_requested: false,
                command_error: None,
                state_after_cancel: (candidate.observed_state == "cancelled")
                    .then(|| "cancelled".to_owned()),
                pair_updates: 0,
                claim_updated: false,
                recovery_from_prior_preview: true,
                registry_error: None,
            })
            .collect::<Vec<_>>();
        let pair_updates_total = apply_retirement_registry(
            &args.root.join("ops/pair-registry.sqlite"),
            &backlog,
            &plan.candidates,
            &mut results,
            &preview_sha256,
        )?;
        let audit = json!({
            "schema": "JopeWaveBlockerRetirementAudit.v1",
            "action": "retire_blockers_for_wave",
            "created_at_ms": now_ms(),
            "preview_sha256": preview_sha256,
            "preview_path": preview_path,
            "wave": plan.wave,
            "desired_pairs": plan.desired_pairs,
            "current_wave_active_pairs": plan.current_wave_active_pairs,
            "blocking_pairs": plan.blocking_pairs,
            "blocking_claims": plan.blocking_claims,
            "blocking_manifests": plan.blocking_manifests,
            "pair_updates_total": pair_updates_total,
            "results": results,
        });
        let (audit_path, audit_sha256) =
            write_content_addressed(&receipt_dir, "blocker-audit", &audit)?;
        output["wave_blocker_retirement"] = json!({
            "preview_path": preview_path,
            "preview_sha256": preview_sha256,
            "audit_path": audit_path,
            "audit_sha256": audit_sha256,
            "blocking_pairs": plan.blocking_pairs,
            "blocking_manifests": plan.blocking_manifests,
            "pair_updates_total": pair_updates_total,
        });
        println!("{}", serde_json::to_string_pretty(&output)?);
        return Ok(());
    }

    if args.hold_legacy || args.hold_superseded_waves {
        let newest_wave = newest_wave(&records);
        let action = if args.hold_legacy {
            "hold_legacy"
        } else {
            "hold_superseded_waves"
        };
        let candidates = hold_candidates(
            &records,
            args.hold_legacy,
            args.hold_superseded_waves,
            newest_wave.as_deref(),
        );
        let preview = json!({
            "schema": "JopeQueueHoldPreview.v1",
            "action": action,
            "created_at_ms": now_ms(),
            "newest_strict_wave": newest_wave,
            "candidate_count": candidates.len(),
            "candidates": candidates,
        });
        let (preview_path, preview_sha256) =
            write_content_addressed(&receipt_dir, "preview", &preview)?;
        if args.preview_only {
            output["hold_preview"] = json!({
                "preview_path": preview_path,
                "preview_sha256": preview_sha256,
                "candidate_count": candidates.len(),
                "mutation_performed": false,
            });
            println!("{}", serde_json::to_string_pretty(&output)?);
            return Ok(());
        }

        // Re-read immediately before mutation. Eligibility is intersected with
        // this fresh snapshot; jobs which advanced to running/dispatching are skipped.
        let before_jobs = scheduler_jobs(&args.scq, &args.socket)?;
        let before = states_by_id(&before_jobs);
        let before_claims = load_claims(&backlog)?;
        let mut results = hold_over_socket(
            &candidates,
            &before,
            &before_claims,
            &args.socket,
            args.max_hold_workers,
        )?;
        results.sort_by(|left, right| left.job_id.cmp(&right.job_id));

        let after_jobs = scheduler_jobs(&args.scq, &args.socket)?;
        let after = states_by_id(&after_jobs);
        for result in &mut results {
            result.state_after_hold = after.get(&result.job_id).map(|(_, state)| state.to_owned());
        }
        let audit = json!({
            "schema": "JopeQueueHoldAudit.v1",
            "action": action,
            "created_at_ms": now_ms(),
            "preview_sha256": preview_sha256,
            "preview_path": preview_path,
            "results": results,
            "post_action_counts": queue_report(&join_inventory(
                &after_jobs,
                &claims,
                ValidationMode::Audit,
                args.max_hold_workers,
            )?),
        });
        let (audit_path, audit_sha256) = write_content_addressed(&receipt_dir, "audit", &audit)?;
        output["hold"] = json!({
            "preview_path": preview_path,
            "preview_sha256": preview_sha256,
            "audit_path": audit_path,
            "audit_sha256": audit_sha256,
            "candidate_count": candidates.len(),
        });
    }

    if args.release_current_checkpoint_bridge {
        let active_checkpoint = args
            .active_checkpoint
            .as_deref()
            .context("active checkpoint disappeared")?;
        let candidates = checkpoint_bridge_candidates(&records, active_checkpoint, args.limit);
        let newest_selected_wave = candidates
            .first()
            .and_then(|candidate| candidate.strict_wave.clone());
        let preview = json!({
            "schema": "JopeQueueCheckpointBridgePreview.v1",
            "action": "release_current_checkpoint_bridge",
            "created_at_ms": now_ms(),
            "active_checkpoint_sha256": active_checkpoint,
            "limit": args.limit,
            "newest_selected_strict_wave": newest_selected_wave,
            "candidate_count": candidates.len(),
            "candidates": candidates,
        });
        let (preview_path, preview_sha256) =
            write_content_addressed(&receipt_dir, "bridge-preview", &preview)?;
        if args.preview_only {
            output["checkpoint_bridge_preview"] = json!({
                "preview_path": preview_path,
                "preview_sha256": preview_sha256,
                "candidate_count": candidates.len(),
                "mutation_performed": false,
            });
            println!("{}", serde_json::to_string_pretty(&output)?);
            return Ok(());
        }

        // Re-read both scheduler and claim state, then re-hash every manifest
        // immediately before the bounded release. A job which is no longer
        // held, submitted, self-play, or checkpoint-exact is skipped.
        let before_jobs = scheduler_jobs(&args.scq, &args.socket)?;
        let before = states_by_id(&before_jobs);
        let before_claims = load_claims(&backlog)?;
        let mut results = release_over_socket(
            &candidates,
            &before,
            &before_claims,
            &args.socket,
            args.max_hold_workers,
            active_checkpoint,
        )?;
        results.sort_by(|left, right| left.job_id.cmp(&right.job_id));

        let after_jobs = scheduler_jobs(&args.scq, &args.socket)?;
        let after = states_by_id(&after_jobs);
        for result in &mut results {
            result.state_after_release =
                after.get(&result.job_id).map(|(_, state)| state.to_owned());
        }
        let audit = json!({
            "schema": "JopeQueueCheckpointBridgeAudit.v1",
            "action": "release_current_checkpoint_bridge",
            "created_at_ms": now_ms(),
            "active_checkpoint_sha256": active_checkpoint,
            "limit": args.limit,
            "preview_sha256": preview_sha256,
            "preview_path": preview_path,
            "results": results,
            "post_action_counts": queue_report(&join_inventory(
                &after_jobs,
                &load_claims(&backlog)?,
                ValidationMode::Audit,
                args.max_hold_workers,
            )?),
        });
        let (audit_path, audit_sha256) =
            write_content_addressed(&receipt_dir, "bridge-audit", &audit)?;
        output["checkpoint_bridge"] = json!({
            "preview_path": preview_path,
            "preview_sha256": preview_sha256,
            "audit_path": audit_path,
            "audit_sha256": audit_sha256,
            "candidate_count": candidates.len(),
        });
    }

    if args.retire_held_legacy || args.retire_held_superseded_waves {
        let newest_wave = newest_wave(&records);
        let action = if args.retire_held_legacy {
            "retire_held_legacy"
        } else {
            "retire_held_superseded_waves"
        };
        let mut candidates = retirement_candidates(
            &records,
            args.retire_held_legacy,
            args.retire_held_superseded_waves,
            newest_wave.as_deref(),
        );
        let recovered = recover_retirement_candidates(
            &receipt_dir,
            action,
            &claims,
            &states_by_id(&initial_jobs),
        )?;
        let mut recovery_ids = candidates
            .iter()
            .filter(|candidate| candidate.observed_state == "cancelled")
            .map(|candidate| candidate.job_id.clone())
            .collect::<BTreeSet<_>>();
        recovery_ids.extend(recovered.iter().map(|candidate| candidate.job_id.clone()));
        candidates.extend(recovered);
        candidates.sort_by(|left, right| left.job_id.cmp(&right.job_id));
        candidates.dedup_by(|left, right| left.job_id == right.job_id);
        let preview = json!({
            "schema": "JopeQueueRetirementPreview.v1",
            "action": action,
            "created_at_ms": now_ms(),
            "newest_strict_wave": newest_wave,
            "candidate_count": candidates.len(),
            "recovery_candidate_count": recovery_ids.len(),
            "candidates": candidates,
        });
        let (preview_path, preview_sha256) =
            write_content_addressed(&receipt_dir, "preview", &preview)?;
        if args.preview_only {
            output["retirement_preview"] = json!({
                "preview_path": preview_path,
                "preview_sha256": preview_sha256,
                "candidate_count": candidates.len(),
                "recovery_candidate_count": recovery_ids.len(),
                "mutation_performed": false,
            });
            println!("{}", serde_json::to_string_pretty(&output)?);
            return Ok(());
        }

        let before_jobs = scheduler_jobs(&args.scq, &args.socket)?;
        let before = states_by_id(&before_jobs);
        let before_claims = load_claims(&backlog)?;
        let mut results = cancel_over_socket(
            &candidates,
            &before,
            &before_claims,
            &args.socket,
            args.max_hold_workers,
            &recovery_ids,
        )?;
        let after_jobs = scheduler_jobs(&args.scq, &args.socket)?;
        let after = states_by_id(&after_jobs);
        for result in &mut results {
            result.state_after_cancel =
                after.get(&result.job_id).map(|(_, state)| state.to_owned());
            if result.cancel_requested
                && result
                    .state_after_cancel
                    .as_deref()
                    .is_none_or(|state| state == "cancelled")
            {
                // Scheduler state is the authority if a response was lost and
                // the reconnect retry reported "already cancelled".
                result.command_error = None;
            }
        }
        let pair_registry = args.root.join("ops/pair-registry.sqlite");
        let pair_updates_total = apply_retirement_registry(
            &pair_registry,
            &backlog,
            &candidates,
            &mut results,
            &preview_sha256,
        )?;
        let audit = json!({
            "schema": "JopeQueueRetirementAudit.v1",
            "action": action,
            "created_at_ms": now_ms(),
            "preview_sha256": preview_sha256,
            "preview_path": preview_path,
            "pair_updates_total": pair_updates_total,
            "results": results,
            "post_action_counts": queue_report(&join_inventory(
                &after_jobs,
                &load_claims(&backlog)?,
                ValidationMode::Audit,
                args.max_hold_workers,
            )?),
        });
        let (audit_path, audit_sha256) =
            write_content_addressed(&receipt_dir, "retirement-audit", &audit)?;
        output["retirement"] = json!({
            "preview_path": preview_path,
            "preview_sha256": preview_sha256,
            "audit_path": audit_path,
            "audit_sha256": audit_sha256,
            "candidate_count": candidates.len(),
        });
    }

    if let Some(path) = args.export_seed_bank {
        let seed_bank = export_seed_bank(
            &args.corpus,
            args.seeds_per_task,
            args.minimum_repeat_datasets,
        )?;
        write_json_atomic(&path, &seed_bank)?;
        let bank_sha256 = sha256_file(&path)?;
        let receipt = json!({
            "schema": "JopeSeedBankExportReceipt.v1",
            "created_at_ms": now_ms(),
            "corpus": args.corpus,
            "bank_path": path,
            "bank_sha256": bank_sha256,
            "seeds_per_task": args.seeds_per_task,
            "minimum_repeat_datasets": args.minimum_repeat_datasets,
            "dl_origin_cap_per_task": args.seeds_per_task / 3,
            "dl_origin_counts": seed_bank["dl_origin_counts"].clone(),
        });
        let (receipt_path, receipt_sha256) =
            write_content_addressed(&receipt_dir, "seed-bank-export", &receipt)?;
        output["seed_bank"] = json!({
            "path": path,
            "sha256": bank_sha256,
            "seeds_per_task": args.seeds_per_task,
            "minimum_repeat_datasets": args.minimum_repeat_datasets,
            "dl_origin_cap_per_task": args.seeds_per_task / 3,
            "dl_origin_counts": seed_bank["dl_origin_counts"].clone(),
            "receipt_path": receipt_path,
            "receipt_sha256": receipt_sha256,
        });
    }

    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

#[cfg(test)]
mod tests;
