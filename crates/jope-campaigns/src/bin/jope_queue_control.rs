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

fn load_claims(path: &Path) -> Result<HashMap<String, Claim>> {
    let connection = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .with_context(|| format!("open backlog {}", path.display()))?;
    connection.busy_timeout(std::time::Duration::from_secs(60))?;
    let mut statement = connection.prepare(
        "SELECT job_id,manifest_sha256,manifest,state FROM claims WHERE job_id IS NOT NULL",
    )?;
    let rows = statement.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            Claim {
                manifest_sha256: row.get(1)?,
                manifest: PathBuf::from(row.get::<_, String>(2)?),
                state: row.get(3)?,
            },
        ))
    })?;
    let mut claims = HashMap::new();
    for row in rows {
        let (job_id, claim) = row?;
        if claims.insert(job_id.clone(), claim).is_some() {
            bail!("duplicate backlog job_id {job_id}");
        }
    }
    Ok(claims)
}

fn scheduler_jobs(scq: &Path, socket: &Path) -> Result<Vec<Value>> {
    let socket = socket
        .to_str()
        .context("SCQ socket path is not valid UTF-8")?;
    let output = scq_json(scq, &["ls", "--socket", socket])?;
    Ok(output
        .get("Jobs")
        .and_then(|value| value.get("jobs"))
        .or_else(|| output.get("jobs"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default())
}

fn join_inventory(
    jobs: &[Value],
    claims: &HashMap<String, Claim>,
    mode: ValidationMode,
    workers: usize,
) -> Result<Vec<JobRecord>> {
    // Manifest validation is CPU-heavy (content and semantic SHA-256 checks).
    // Bound concurrency so it is fast without flooding the shared filesystem.
    ThreadPoolBuilder::new()
        .num_threads(workers)
        .thread_name(|index| format!("queue-audit-{index}"))
        .build()?
        .install(|| {
            jobs.par_iter()
                .map(|job| join_job(job, claims, mode))
                .collect()
        })
}

fn join_job(
    job: &Value,
    claims: &HashMap<String, Claim>,
    mode: ValidationMode,
) -> Result<JobRecord> {
    let job_id = text_field(job, "id")
        .or_else(|| text_field(job, "job_id"))
        .unwrap_or_default();
    let project = text_field(job, "project").unwrap_or_else(|| "unknown".to_owned());
    let state = text_field(job, "state").unwrap_or_else(|| "unknown".to_owned());
    let Some(claim) = claims.get(&job_id) else {
        return Ok(JobRecord {
            job_id,
            project,
            state,
            epoch: "unclaimed".to_owned(),
            strict_wave: None,
            generation: None,
            checkpoint_sha256: None,
            claim_state: None,
            manifest: None,
            manifest_sha256: None,
            manifest_checked: false,
            manifest_valid: false,
            invalid_reason: Some("no_matching_backlog_claim".to_owned()),
        });
    };

    let should_validate = claim.state == "submitted"
        && match mode {
            ValidationMode::Audit => false,
            ValidationMode::HoldLegacy => project == LEGACY_PROJECT && is_holdable_state(&state),
            ValidationMode::HoldSupersededWaves => {
                project == SELFPLAY_PROJECT && is_holdable_state(&state)
            }
            ValidationMode::RetireHeldLegacy => {
                project == LEGACY_PROJECT && matches!(state.as_str(), "held" | "cancelled")
            }
            ValidationMode::RetireHeldSupersededWaves => {
                project == SELFPLAY_PROJECT && matches!(state.as_str(), "held" | "cancelled")
            }
            ValidationMode::ReleaseCurrentCheckpointBridge => {
                project == SELFPLAY_PROJECT && state == "held"
            }
        };
    let validation = should_validate.then(|| validate_manifest(claim));
    let (manifest, invalid_reason) = match validation {
        Some(Ok(value)) => (Some(value), None),
        Some(Err(error)) => (None, Some(format!("{error:#}"))),
        None => (None, None),
    };
    let wave_hint = strict_wave_from_path(&claim.manifest);
    let generation_hint = wave_hint.clone();
    Ok(JobRecord {
        job_id,
        project,
        state,
        epoch: manifest.as_ref().map(manifest_epoch).unwrap_or_else(|| {
            if should_validate {
                "invalid-manifest".to_owned()
            } else if wave_hint.is_some() {
                "manifest-wave-unchecked".to_owned()
            } else {
                LEGACY_EPOCH.to_owned()
            }
        }),
        strict_wave: manifest.as_ref().and_then(strict_wave).or(wave_hint),
        generation: manifest.as_ref().and_then(generation).or(generation_hint),
        checkpoint_sha256: manifest.as_ref().and_then(manifest_checkpoint_binding),
        claim_state: Some(claim.state.clone()),
        manifest: Some(claim.manifest.to_string_lossy().into_owned()),
        manifest_sha256: Some(claim.manifest_sha256.clone()),
        manifest_checked: should_validate,
        manifest_valid: manifest.is_some(),
        invalid_reason,
    })
}

fn validate_manifest(claim: &Claim) -> Result<Value> {
    if claim.manifest_sha256.len() != 64
        || !claim
            .manifest_sha256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        bail!("claim manifest hash is not SHA-256");
    }
    if !claim.manifest.is_file() {
        bail!("claimed manifest does not exist");
    }
    let actual = sha256_file(&claim.manifest)?;
    if actual != claim.manifest_sha256 {
        bail!("claimed manifest hash drift");
    }
    let manifest: Value = serde_json::from_slice(&fs::read(&claim.manifest)?)
        .with_context(|| format!("parse manifest {}", claim.manifest.display()))?;
    if manifest.get("schema").and_then(Value::as_str) != Some(MANIFEST_SCHEMA) {
        bail!("wrong manifest schema");
    }
    if !manifest_freshness_scoped(&manifest) || protected_lane(&manifest) {
        bail!("manifest is outside freshness scope");
    }
    let task = manifest
        .get("task")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if !matches!(task, "regression" | "binary" | "timeseries" | "multiclass") {
        bail!("unsupported task");
    }
    let candidates = manifest
        .get("candidates")
        .and_then(Value::as_array)
        .context("manifest candidates are absent")?;
    if candidates.is_empty() || candidates.len() > 100 {
        bail!("candidate count is outside 1..=100");
    }
    let mut semantics = BTreeSet::new();
    for candidate in candidates {
        let genome = candidate
            .get("canonical_genome")
            .and_then(Value::as_str)
            .context("candidate canonical genome is absent")?;
        let semantic = candidate
            .get("semantic_sha256")
            .and_then(Value::as_str)
            .context("candidate semantic hash is absent")?;
        if sha256_bytes(genome.as_bytes()) != semantic {
            bail!("candidate semantic hash drift");
        }
        if !semantics.insert(semantic) {
            bail!("duplicate candidate semantic");
        }
    }
    Ok(manifest)
}

fn protected_lane(manifest: &Value) -> bool {
    let lane = manifest
        .get("evaluation_lane")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_ascii_lowercase();
    lane.contains("gate") || lane.contains("sealed") || lane.contains("canary")
}

fn strict_wave(manifest: &Value) -> Option<String> {
    manifest
        .get("strict_wave")
        .and_then(Value::as_str)
        .or_else(|| {
            manifest
                .get("label_authority")
                .and_then(|value| value.get("strict_wave"))
                .and_then(Value::as_str)
        })
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

fn generation(manifest: &Value) -> Option<String> {
    manifest
        .get("generation")
        .and_then(Value::as_str)
        .or_else(|| {
            manifest
                .get("label_authority")
                .and_then(|value| value.get("generation"))
                .and_then(Value::as_str)
        })
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

fn manifest_checkpoint_binding(manifest: &Value) -> Option<String> {
    let candidates = manifest.get("candidates")?.as_array()?;
    let checkpoint = candidates
        .first()?
        .get("model_checkpoint_sha256")?
        .as_str()?;
    if !is_sha256(checkpoint)
        || !candidates.iter().all(|candidate| {
            candidate
                .get("model_checkpoint_sha256")
                .and_then(Value::as_str)
                == Some(checkpoint)
        })
        || manifest
            .get("label_authority")
            .and_then(|value| value.get("model_checkpoint_sha256"))
            .and_then(Value::as_str)
            != Some(checkpoint)
        || manifest
            .get("checkpoint_sha256")
            .and_then(Value::as_str)
            .is_some_and(|value| value != checkpoint)
    {
        return None;
    }
    Some(checkpoint.to_owned())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn strict_wave_from_path(path: &Path) -> Option<String> {
    path.ancestors()
        .filter_map(Path::file_name)
        .filter_map(|name| name.to_str())
        .find(|name| {
            name.strip_prefix("strict-wave-").is_some_and(|suffix| {
                !suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit())
            })
        })
        .map(str::to_owned)
}

fn queue_report(records: &[JobRecord]) -> QueueReport {
    let mut counts = BTreeMap::new();
    let mut waves = BTreeMap::new();
    let mut generations = BTreeMap::new();
    for record in records {
        *counts
            .entry(record.project.clone())
            .or_insert_with(BTreeMap::new)
            .entry(record.state.clone())
            .or_insert_with(BTreeMap::new)
            .entry(record.epoch.clone())
            .or_insert(0) += 1;
        if record.project == SELFPLAY_PROJECT {
            *waves
                .entry(
                    record
                        .strict_wave
                        .clone()
                        .unwrap_or_else(|| "none".to_owned()),
                )
                .or_insert_with(BTreeMap::new)
                .entry(record.state.clone())
                .or_insert(0) += 1;
            *generations
                .entry(
                    record
                        .generation
                        .clone()
                        .unwrap_or_else(|| "none".to_owned()),
                )
                .or_insert_with(BTreeMap::new)
                .entry(record.state.clone())
                .or_insert(0) += 1;
        }
    }
    QueueReport {
        schema: "JopeQueueControlStatus.v1",
        generated_at_ms: now_ms(),
        counts_by_project_state_epoch: counts,
        selfplay_counts_by_wave_state: waves,
        selfplay_counts_by_generation_state: generations,
        newest_strict_wave: newest_wave(records),
        scheduler_jobs: records.len(),
        joined_claims: records
            .iter()
            .filter(|row| row.claim_state.is_some())
            .count(),
        valid_manifests: records
            .iter()
            .filter(|row| row.manifest_checked && row.manifest_valid)
            .count(),
        invalid_manifests: records
            .iter()
            .filter(|row| row.manifest_checked && !row.manifest_valid)
            .count(),
        unchecked_manifests: records
            .iter()
            .filter(|row| row.claim_state.is_some() && !row.manifest_checked)
            .count(),
        unclaimed_jobs: records
            .iter()
            .filter(|row| row.claim_state.is_none())
            .count(),
    }
}

fn newest_wave(records: &[JobRecord]) -> Option<String> {
    records
        .iter()
        .filter(|row| row.project == SELFPLAY_PROJECT && row.claim_state.is_some())
        .filter_map(|row| row.strict_wave.as_ref())
        .max_by_key(|wave| wave_number(wave))
        .cloned()
}

fn wave_number(wave: &str) -> (u64, &str) {
    (
        wave.rsplit('-')
            .next()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0),
        wave,
    )
}

fn hold_candidates(
    records: &[JobRecord],
    legacy: bool,
    superseded_waves: bool,
    newest_wave: Option<&str>,
) -> Vec<HoldCandidate> {
    let mut candidates = records
        .iter()
        .filter(|row| {
            row.manifest_valid
                && row.claim_state.as_deref() == Some("submitted")
                && is_holdable_state(&row.state)
                && if legacy {
                    row.project == LEGACY_PROJECT
                } else if superseded_waves {
                    row.project == SELFPLAY_PROJECT
                        && row.strict_wave.is_some()
                        && row.strict_wave.as_deref() != newest_wave
                } else {
                    false
                }
        })
        .filter_map(|row| {
            Some(HoldCandidate {
                job_id: row.job_id.clone(),
                project: row.project.clone(),
                observed_state: row.state.clone(),
                epoch: row.epoch.clone(),
                strict_wave: row.strict_wave.clone(),
                generation: row.generation.clone(),
                manifest: row.manifest.clone()?,
                manifest_sha256: row.manifest_sha256.clone()?,
            })
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| left.job_id.cmp(&right.job_id));
    candidates
}

fn retirement_candidates(
    records: &[JobRecord],
    legacy: bool,
    superseded_waves: bool,
    newest_wave: Option<&str>,
) -> Vec<HoldCandidate> {
    let mut candidates = records
        .iter()
        .filter(|row| {
            row.manifest_valid
                && row.claim_state.as_deref() == Some("submitted")
                && matches!(row.state.as_str(), "held" | "cancelled")
                && if legacy {
                    row.project == LEGACY_PROJECT
                } else if superseded_waves {
                    row.project == SELFPLAY_PROJECT
                        && row.strict_wave.is_some()
                        && row.strict_wave.as_deref() != newest_wave
                } else {
                    false
                }
        })
        .filter_map(|row| {
            Some(HoldCandidate {
                job_id: row.job_id.clone(),
                project: row.project.clone(),
                observed_state: row.state.clone(),
                epoch: row.epoch.clone(),
                strict_wave: row.strict_wave.clone(),
                generation: row.generation.clone(),
                manifest: row.manifest.clone()?,
                manifest_sha256: row.manifest_sha256.clone()?,
            })
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| left.job_id.cmp(&right.job_id));
    candidates
}

fn checkpoint_bridge_candidates(
    records: &[JobRecord],
    active_checkpoint: &str,
    limit: usize,
) -> Vec<HoldCandidate> {
    let mut candidates = records
        .iter()
        .filter(|row| {
            row.project == SELFPLAY_PROJECT
                && row.state == "held"
                && row.claim_state.as_deref() == Some("submitted")
                && row.manifest_valid
                && row.strict_wave.is_some()
                && row.checkpoint_sha256.as_deref() == Some(active_checkpoint)
        })
        .filter_map(|row| {
            Some(HoldCandidate {
                job_id: row.job_id.clone(),
                project: row.project.clone(),
                observed_state: row.state.clone(),
                epoch: row.epoch.clone(),
                strict_wave: row.strict_wave.clone(),
                generation: row.generation.clone(),
                manifest: row.manifest.clone()?,
                manifest_sha256: row.manifest_sha256.clone()?,
            })
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| {
        wave_number(right.strict_wave.as_deref().unwrap_or_default())
            .cmp(&wave_number(
                left.strict_wave.as_deref().unwrap_or_default(),
            ))
            .then_with(|| left.job_id.cmp(&right.job_id))
    });
    candidates.truncate(limit);
    candidates
}

fn is_holdable_state(state: &str) -> bool {
    matches!(state, "queued" | "starting")
}

fn states_by_id(jobs: &[Value]) -> HashMap<String, (String, String)> {
    jobs.iter()
        .filter_map(|job| {
            let id = text_field(job, "id").or_else(|| text_field(job, "job_id"))?;
            Some((
                id,
                (
                    text_field(job, "project").unwrap_or_default(),
                    text_field(job, "state").unwrap_or_default(),
                ),
            ))
        })
        .collect()
}

fn text_field(value: &Value, field: &str) -> Option<String> {
    value.get(field).and_then(Value::as_str).map(str::to_owned)
}

struct ScqHoldClient {
    writer: UnixStream,
    reader: BufReader<UnixStream>,
}

impl ScqHoldClient {
    fn connect(socket: &Path) -> Result<Self> {
        let writer = UnixStream::connect(socket)
            .with_context(|| format!("connect SCQ socket {}", socket.display()))?;
        writer.set_read_timeout(Some(Duration::from_secs(30)))?;
        writer.set_write_timeout(Some(Duration::from_secs(30)))?;
        let reader = BufReader::new(writer.try_clone()?);
        Ok(Self { writer, reader })
    }

    fn hold(&mut self, job_id: &str) -> Result<()> {
        self.request("Hold", job_id)
    }

    fn cancel(&mut self, job_id: &str) -> Result<()> {
        self.request("Cancel", job_id)
    }

    fn release(&mut self, job_id: &str) -> Result<()> {
        self.request("Release", job_id)
    }

    fn request(&mut self, operation: &str, job_id: &str) -> Result<()> {
        let request = match operation {
            "Hold" => json!({"Hold": {"id": job_id}}),
            "Cancel" => json!({"Cancel": {"id": job_id}}),
            "Release" => json!({"Release": {"id": job_id}}),
            _ => bail!("unsupported SCQ mutation"),
        };
        serde_json::to_writer(&mut self.writer, &request)?;
        self.writer.write_all(b"\n")?;
        self.writer.flush()?;
        let mut response = String::new();
        if self.reader.read_line(&mut response)? == 0 {
            bail!("SCQ socket closed before {operation} response");
        }
        let response: Value =
            serde_json::from_str(response.trim()).context("parse SCQ mutation response")?;
        if let Some(error) = response
            .get("Error")
            .or_else(|| response.get("Err"))
            .or_else(|| response.get("error"))
            .filter(|value| !value.is_null())
        {
            bail!("SCQ {operation} rejected {job_id}: {error}");
        }
        if response.get("ok").and_then(Value::as_bool) == Some(false) {
            bail!("SCQ {operation} rejected {job_id}: {response}");
        }
        Ok(())
    }
}

fn claim_matches_candidate(claim: &Claim, candidate: &HoldCandidate) -> bool {
    claim.state == "submitted"
        && claim.manifest_sha256 == candidate.manifest_sha256
        && claim.manifest.to_string_lossy() == candidate.manifest
}

fn hold_over_socket(
    candidates: &[HoldCandidate],
    before: &HashMap<String, (String, String)>,
    before_claims: &HashMap<String, Claim>,
    socket: &Path,
    workers: usize,
) -> Result<Vec<HoldResult>> {
    if candidates.is_empty() {
        return Ok(Vec::new());
    }
    let chunk_size = candidates.len().div_ceil(workers);
    let pool = ThreadPoolBuilder::new()
        .num_threads(workers.min(candidates.len()))
        .thread_name(|index| format!("queue-hold-{index}"))
        .build()?;
    Ok(pool.install(|| {
        candidates
            .par_chunks(chunk_size)
            .flat_map_iter(|chunk| {
                let mut client = ScqHoldClient::connect(socket);
                chunk.iter().map(move |candidate| {
                    let live = before.get(&candidate.job_id);
                    let still_eligible = live.is_some_and(|(project, state)| {
                        project == &candidate.project && is_holdable_state(state)
                    }) && before_claims
                        .get(&candidate.job_id)
                        .is_some_and(|claim| claim_matches_candidate(claim, candidate));
                    let command_error = if still_eligible {
                        socket_mutation_with_reconnect(
                            &mut client,
                            socket,
                            "Hold",
                            &candidate.job_id,
                        )
                    } else {
                        None
                    };
                    HoldResult {
                        job_id: candidate.job_id.clone(),
                        state_before_hold: live.map(|(_, state)| state.clone()),
                        hold_requested: still_eligible,
                        command_error,
                        state_after_hold: None,
                    }
                })
            })
            .collect()
    }))
}

fn release_over_socket(
    candidates: &[HoldCandidate],
    before: &HashMap<String, (String, String)>,
    before_claims: &HashMap<String, Claim>,
    socket: &Path,
    workers: usize,
    active_checkpoint: &str,
) -> Result<Vec<ReleaseResult>> {
    if candidates.is_empty() {
        return Ok(Vec::new());
    }
    let chunk_size = candidates.len().div_ceil(workers);
    let pool = ThreadPoolBuilder::new()
        .num_threads(workers.min(candidates.len()))
        .thread_name(|index| format!("queue-release-{index}"))
        .build()?;
    Ok(pool.install(|| {
        candidates
            .par_chunks(chunk_size)
            .flat_map_iter(|chunk| {
                let mut client = ScqHoldClient::connect(socket);
                chunk.iter().map(move |candidate| {
                    let live = before.get(&candidate.job_id);
                    let claim = before_claims.get(&candidate.job_id);
                    let fresh_manifest_valid = claim
                        .filter(|claim| claim_matches_candidate(claim, candidate))
                        .and_then(|claim| validate_manifest(claim).ok())
                        .and_then(|manifest| manifest_checkpoint_binding(&manifest))
                        .as_deref()
                        == Some(active_checkpoint);
                    let still_eligible = live.is_some_and(|(project, state)| {
                        project == SELFPLAY_PROJECT && state == "held"
                    }) && fresh_manifest_valid;
                    let command_error = if still_eligible {
                        socket_mutation_with_reconnect(
                            &mut client,
                            socket,
                            "Release",
                            &candidate.job_id,
                        )
                    } else {
                        Some("state_claim_manifest_or_checkpoint_changed_before_release".to_owned())
                    };
                    ReleaseResult {
                        job_id: candidate.job_id.clone(),
                        state_before_release: live.map(|(_, state)| state.clone()),
                        release_requested: still_eligible,
                        command_error,
                        state_after_release: None,
                    }
                })
            })
            .collect()
    }))
}

fn cancel_over_socket(
    candidates: &[HoldCandidate],
    before: &HashMap<String, (String, String)>,
    before_claims: &HashMap<String, Claim>,
    socket: &Path,
    workers: usize,
    recovery_ids: &BTreeSet<String>,
) -> Result<Vec<RetirementResult>> {
    if candidates.is_empty() {
        return Ok(Vec::new());
    }
    let chunk_size = candidates.len().div_ceil(workers);
    let pool = ThreadPoolBuilder::new()
        .num_threads(workers.min(candidates.len()))
        .thread_name(|index| format!("queue-cancel-{index}"))
        .build()?;
    Ok(pool.install(|| {
        candidates
            .par_chunks(chunk_size)
            .flat_map_iter(|chunk| {
                let mut client = ScqHoldClient::connect(socket);
                chunk.iter().map(move |candidate| {
                    let live = before.get(&candidate.job_id);
                    let claim_matches = before_claims
                        .get(&candidate.job_id)
                        .is_some_and(|claim| claim_matches_candidate(claim, candidate));
                    let still_held = live.is_some_and(|(project, state)| {
                        project == &candidate.project && state == "held"
                    });
                    let recoverable_absence = recovery_ids.contains(&candidate.job_id)
                        && live.is_none_or(|(_, state)| state == "cancelled");
                    let cancel_requested = claim_matches && still_held;
                    let command_error = if cancel_requested {
                        socket_mutation_with_reconnect(
                            &mut client,
                            socket,
                            "Cancel",
                            &candidate.job_id,
                        )
                    } else if claim_matches && recoverable_absence {
                        None
                    } else {
                        Some("state_or_claim_changed_before_cancel".to_owned())
                    };
                    RetirementResult {
                        job_id: candidate.job_id.clone(),
                        state_before_cancel: live.map(|(_, state)| state.clone()),
                        cancel_requested,
                        command_error,
                        state_after_cancel: None,
                        pair_updates: 0,
                        claim_updated: false,
                        recovery_from_prior_preview: recoverable_absence,
                        registry_error: None,
                    }
                })
            })
            .collect()
    }))
}

fn socket_mutation_with_reconnect(
    client: &mut Result<ScqHoldClient>,
    socket: &Path,
    operation: &str,
    job_id: &str,
) -> Option<String> {
    let mut last_error = None;
    for _ in 0..2 {
        if client.is_err() {
            *client = ScqHoldClient::connect(socket);
        }
        let result = match client {
            Ok(client) if operation == "Hold" => client.hold(job_id),
            Ok(client) if operation == "Cancel" => client.cancel(job_id),
            Ok(client) if operation == "Release" => client.release(job_id),
            Ok(_) => return Some("unsupported SCQ mutation".to_owned()),
            Err(error) => Err(anyhow::anyhow!("{error:#}")),
        };
        match result {
            Ok(()) => return None,
            Err(error) => {
                last_error = Some(error.to_string());
                *client = ScqHoldClient::connect(socket);
            }
        }
    }
    last_error.or_else(|| Some("SCQ mutation failed without an error".to_owned()))
}

fn recover_retirement_candidates(
    receipt_dir: &Path,
    action: &str,
    claims: &HashMap<String, Claim>,
    scheduler: &HashMap<String, (String, String)>,
) -> Result<Vec<HoldCandidate>> {
    if !receipt_dir.is_dir() {
        return Ok(Vec::new());
    }
    let expected_project = if action == "retire_held_legacy" {
        LEGACY_PROJECT
    } else {
        SELFPLAY_PROJECT
    };
    let mut recovered = BTreeMap::<String, HoldCandidate>::new();
    for entry in fs::read_dir(receipt_dir)? {
        let path = entry?.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !name.starts_with("preview-") || !name.ends_with(".json") {
            continue;
        }
        let expected_hash = name
            .strip_prefix("preview-")
            .and_then(|value| value.strip_suffix(".json"))
            .unwrap_or_default();
        if expected_hash.len() != 64 || sha256_file(&path)? != expected_hash {
            continue;
        }
        let value: Value = serde_json::from_slice(&fs::read(&path)?)
            .with_context(|| format!("parse preview {}", path.display()))?;
        if value.get("schema").and_then(Value::as_str) != Some("JopeQueueRetirementPreview.v1")
            || value.get("action").and_then(Value::as_str) != Some(action)
        {
            continue;
        }
        let Some(rows) = value.get("candidates").and_then(Value::as_array) else {
            continue;
        };
        for row in rows {
            let candidate: HoldCandidate = serde_json::from_value(row.clone())?;
            // Present cancelled rows are already revalidated in the current
            // inventory. Prior previews are only needed after SCQ prunes a row.
            if candidate.project != expected_project || scheduler.contains_key(&candidate.job_id) {
                continue;
            }
            let Some(claim) = claims.get(&candidate.job_id) else {
                continue;
            };
            if !claim_matches_candidate(claim, &candidate) {
                continue;
            }
            recovered.insert(candidate.job_id.clone(), candidate);
        }
    }
    // Multiple durable previews intentionally repeat the same candidate. Hash
    // each immutable manifest once after deduplication, while still performing
    // a fresh validation immediately before any recovered mutation.
    for candidate in recovered.values() {
        let claim = claims
            .get(&candidate.job_id)
            .context("recovered retirement claim disappeared")?;
        validate_manifest(claim).with_context(|| format!("recover {}", candidate.job_id))?;
    }
    Ok(recovered.into_values().collect())
}

fn build_retired_binding_repair_plan(
    wave_dir: &Path,
    pair_registry: &Path,
    backlog: &Path,
    receipt_dir: &Path,
) -> Result<RetiredBindingRepairPlan> {
    let wave_dir = fs::canonicalize(wave_dir)?;
    let wave = wave_dir
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| name.starts_with("strict-wave-"))
        .context("repair wave directory name is invalid")?
        .to_owned();
    let mut audits = Vec::<(u64, PathBuf, String, Value)>::new();
    for entry in fs::read_dir(receipt_dir)? {
        let path = entry?.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let Some(expected_hash) = name
            .strip_prefix("blocker-audit-")
            .and_then(|value| value.strip_suffix(".json"))
        else {
            continue;
        };
        if !is_sha256(expected_hash) || sha256_file(&path)? != expected_hash {
            continue;
        }
        let value: Value = serde_json::from_slice(&fs::read(&path)?)?;
        if value.get("schema").and_then(Value::as_str) != Some("JopeWaveBlockerRetirementAudit.v1")
            || value.get("wave").and_then(Value::as_str) != Some(wave.as_str())
        {
            continue;
        }
        let expected_hash = expected_hash.to_owned();
        audits.push((
            value
                .get("created_at_ms")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            path,
            expected_hash,
            value,
        ));
    }
    audits.sort_by_key(|entry| entry.0);
    let (_, audit_path, audit_sha256, audit) = audits
        .pop()
        .context("no blocker retirement audit for wave")?;
    let source_preview_sha256 = required_json_str(&audit, "preview_sha256")?.to_owned();
    let source_preview_path = PathBuf::from(required_json_str(&audit, "preview_path")?);
    if !is_sha256(&source_preview_sha256)
        || sha256_file(&source_preview_path)? != source_preview_sha256
    {
        bail!("blocker retirement source preview hash drift");
    }
    let preview: Value = serde_json::from_slice(&fs::read(&source_preview_path)?)?;
    if preview.get("schema").and_then(Value::as_str) != Some("JopeWaveBlockerRetirementPreview.v1")
        || preview
            .get("plan")
            .and_then(|value| value.get("wave"))
            .and_then(Value::as_str)
            != Some(wave.as_str())
    {
        bail!("blocker retirement source preview schema or wave drift");
    }
    let results = audit
        .get("results")
        .and_then(Value::as_array)
        .context("blocker audit results are absent")?;
    let retired_jobs = results
        .iter()
        .map(|result| {
            if result.get("claim_updated").and_then(Value::as_bool) != Some(true)
                || result
                    .get("command_error")
                    .is_some_and(|value| !value.is_null())
                || result
                    .get("registry_error")
                    .is_some_and(|value| !value.is_null())
            {
                bail!("blocker audit contains an unsuccessful retirement");
            }
            Ok(required_json_str(result, "job_id")?.to_owned())
        })
        .collect::<Result<BTreeSet<_>>>()?;
    let all_candidates = preview
        .get("plan")
        .and_then(|value| value.get("candidates"))
        .and_then(Value::as_array)
        .context("blocker source candidates are absent")?;
    let mut candidates = all_candidates
        .iter()
        .map(|candidate| serde_json::from_value::<HoldCandidate>(candidate.clone()))
        .collect::<serde_json::Result<Vec<_>>>()?;
    candidates.retain(|candidate| retired_jobs.contains(&candidate.job_id));
    candidates.sort_by(|left, right| left.job_id.cmp(&right.job_id));
    if candidates.len() != retired_jobs.len() || candidates.is_empty() {
        bail!("blocker source audit/candidate identity mismatch");
    }

    let pair = Connection::open_with_flags(
        pair_registry,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    let owner_column = if sqlite_column_exists(&pair, "pairs", "reservation_owner")? {
        "reservation_owner"
    } else {
        "owner"
    };
    let backlog = Connection::open_with_flags(
        backlog,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    let retirement_events = {
        let mut statement = pair.prepare(
            "SELECT detail_json FROM events
             WHERE id > (SELECT COALESCE(MAX(id),0)-100000 FROM events)
               AND kind='superseded_unstarted'",
        )?;
        let details = statement
            .query_map([], |row| row.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        details
            .into_iter()
            .filter_map(|detail| serde_json::from_str::<Value>(&detail).ok())
            .filter_map(|detail| {
                Some((
                    detail.get("job_id")?.as_str()?.to_owned(),
                    detail.get("manifest_sha256")?.as_str()?.to_owned(),
                    detail.get("preview_sha256")?.as_str()?.to_owned(),
                ))
            })
            .collect::<BTreeSet<_>>()
    };
    let mut pair_count = 0usize;
    for candidate in &candidates {
        let claim: Option<(String, String, String)> = backlog
            .query_row(
                "SELECT state,manifest_sha256,manifest FROM claims WHERE job_id=?1",
                [&candidate.job_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()?;
        if claim.as_ref()
            != Some(&(
                "superseded_unstarted".to_owned(),
                candidate.manifest_sha256.clone(),
                candidate.manifest.clone(),
            ))
        {
            bail!("retired binding repair claim provenance mismatch");
        }
        let query = format!(
            "SELECT COUNT(*),
                    SUM(CASE WHEN state='available' AND {owner_column}='available'
                              AND receipt_sha256 IS NULL AND failure_class IS NULL
                              AND job_id=?2 AND manifest_path=?3
                             THEN 1 ELSE 0 END)
             FROM pairs INDEXED BY pair_manifest WHERE manifest_sha256=?1"
        );
        let (total, valid): (usize, Option<usize>) = pair.query_row(
            &query,
            params![
                candidate.manifest_sha256,
                candidate.job_id,
                candidate.manifest
            ],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        if total == 0 || valid.unwrap_or(0) != total {
            bail!("retired binding repair pair provenance mismatch");
        }
        if !retirement_events.contains(&(
            candidate.job_id.clone(),
            candidate.manifest_sha256.clone(),
            source_preview_sha256.clone(),
        )) {
            bail!("retired binding repair has no durable retirement event");
        }
        pair_count += total;
    }
    if audit.get("pair_updates_total").and_then(Value::as_u64) != Some(pair_count as u64) {
        bail!("retired binding repair pair count disagrees with source audit");
    }
    Ok(RetiredBindingRepairPlan {
        wave,
        source_audit_path: audit_path.to_string_lossy().into_owned(),
        source_audit_sha256: audit_sha256,
        source_preview_sha256,
        manifest_count: candidates.len(),
        pair_count,
        candidates,
    })
}

fn apply_retired_binding_repair(
    pair_registry: &Path,
    backlog: &Path,
    plan: &RetiredBindingRepairPlan,
    preview_sha256: &str,
) -> Result<usize> {
    let mut connection = Connection::open(pair_registry)?;
    connection.busy_timeout(Duration::from_secs(60))?;
    connection.execute(
        "ATTACH DATABASE ?1 AS backlog_db",
        [backlog.to_string_lossy().as_ref()],
    )?;
    let owner_column = if sqlite_column_exists(&connection, "pairs", "reservation_owner")? {
        "reservation_owner"
    } else {
        "owner"
    };
    let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    transaction.execute_batch(
        "CREATE TEMP TABLE repair_manifests(
           manifest_sha256 TEXT PRIMARY KEY,job_id TEXT NOT NULL,
           manifest_path TEXT NOT NULL,detail_json TEXT NOT NULL
         ) WITHOUT ROWID;",
    )?;
    for candidate in &plan.candidates {
        let detail = serde_json::to_string(&json!({
            "job_id": candidate.job_id,
            "manifest_sha256": candidate.manifest_sha256,
            "source_audit_sha256": plan.source_audit_sha256,
            "repair_preview_sha256": preview_sha256,
        }))?;
        transaction.execute(
            "INSERT INTO repair_manifests VALUES(?1,?2,?3,?4)",
            params![
                candidate.manifest_sha256,
                candidate.job_id,
                candidate.manifest,
                detail
            ],
        )?;
    }
    let invalid_claims: usize = transaction.query_row(
        "SELECT COUNT(*) FROM repair_manifests r
         LEFT JOIN backlog_db.claims c
           ON c.job_id=r.job_id AND c.manifest_sha256=r.manifest_sha256
          AND c.manifest=r.manifest_path
         WHERE c.state IS NULL OR c.state!='superseded_unstarted'",
        [],
        |row| row.get(0),
    )?;
    if invalid_claims != 0 {
        bail!("retired binding repair claim state changed");
    }
    let update = format!(
        "UPDATE pairs INDEXED BY pair_manifest
         SET manifest_path=NULL,manifest_sha256=NULL,job_id=NULL,updated_at_ms=?1
         WHERE state='available' AND {owner_column}='available'
           AND receipt_sha256 IS NULL AND failure_class IS NULL
           AND manifest_sha256 IN(SELECT manifest_sha256 FROM repair_manifests)
           AND job_id=(SELECT r.job_id FROM repair_manifests r
                       WHERE r.manifest_sha256=pairs.manifest_sha256)
           AND manifest_path=(SELECT r.manifest_path FROM repair_manifests r
                              WHERE r.manifest_sha256=pairs.manifest_sha256)"
    );
    let updated = transaction.execute(&update, [now_ms() as i64])?;
    if updated != plan.pair_count {
        bail!("retired binding repair pair count changed before commit");
    }
    transaction.execute(
        "INSERT INTO events(kind,detail_json,created_at_ms)
         SELECT 'retired_binding_repair',detail_json,?1 FROM repair_manifests",
        [now_ms() as i64],
    )?;
    transaction.commit()?;
    Ok(updated)
}

#[allow(clippy::too_many_arguments)]
fn build_missing_receipt_recovery_plan(
    root: &Path,
    wave_dir: &Path,
    pair_registry: &Path,
    backlog: &Path,
    scheduler: &HashMap<String, (String, String)>,
    receipt_dir: &Path,
    scq_events: &Path,
    runs_root: &Path,
    scq: &Path,
    socket: &Path,
) -> Result<MissingReceiptRecoveryPlan> {
    let claims = load_claims(backlog)?;
    // Reuse the strict blocker audit to validate the complete wave, current
    // checkpoint, immutable manifests, and exact pair/claim bindings.
    let _validated = build_wave_blocker_plan(
        root,
        wave_dir,
        pair_registry,
        &claims,
        scheduler,
        receipt_dir,
        scq_events,
    )?;
    let (wave, desired, current_manifest_hashes) = load_wave_desired_pairs(wave_dir)?;
    let pair = Connection::open_with_flags(
        pair_registry,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    pair.busy_timeout(Duration::from_secs(60))?;
    let mut lookup = pair.prepare(
        "SELECT state,manifest_path,manifest_sha256,job_id,receipt_sha256,failure_class
         FROM pairs
         WHERE task=?1 AND dataset_hash=?2 AND semantic_sha256=?3 AND protocol_sha256=?4",
    )?;
    let mut blockers = BTreeMap::<String, (String, String, Vec<RegistryPairIdentity>, bool)>::new();
    for identity in &desired {
        type PairRecoveryRow = (
            String,
            Option<String>,
            Option<String>,
            Option<String>,
            Option<String>,
            Option<String>,
        );
        let row: Option<PairRecoveryRow> = lookup
            .query_row(
                params![
                    identity.task,
                    identity.dataset_hash,
                    identity.semantic_sha256,
                    identity.protocol_sha256
                ],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                    ))
                },
            )
            .optional()?;
        let Some((state, path, manifest, job, receipt, failure)) = row else {
            continue;
        };
        if !matches!(state.as_str(), "reserved" | "submitted" | "held") || receipt.is_some() {
            continue;
        }
        let (Some(path), Some(manifest), Some(job)) = (path, manifest, job) else {
            continue;
        };
        if current_manifest_hashes.contains(&manifest) {
            continue;
        }
        let entry = blockers
            .entry(manifest.clone())
            .or_insert_with(|| (job.clone(), path.clone(), Vec::new(), false));
        if entry.0 != job || entry.1 != path {
            bail!("missing-receipt blocker manifest binding is inconsistent");
        }
        entry.2.push(identity.clone());
        entry.3 |= failure.is_some();
    }
    drop(lookup);
    let mut manifest_pairs_lookup = pair.prepare(
        "SELECT task,dataset_hash,semantic_sha256,protocol_sha256,
                state,manifest_path,job_id,receipt_sha256,failure_class
         FROM pairs INDEXED BY pair_manifest
         WHERE manifest_sha256=?1
         ORDER BY task,dataset_hash,semantic_sha256,protocol_sha256",
    )?;

    let jobs = blockers
        .values()
        .map(|(job, _, _, _)| job.clone())
        .collect::<BTreeSet<_>>();
    let start_evidence = load_scq_start_evidence(scq_events, &jobs)?;
    let corpus = Connection::open_with_flags(
        root.join("corpus/strict-test-exact.sqlite"),
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    let mut reason_counts = BTreeMap::new();
    let mut exclusions = Vec::new();
    let mut candidates = Vec::new();
    for (manifest_sha256, (job_id, manifest_path, pairs, has_failure)) in blockers.iter() {
        let exclude = |reason: String,
                       reason_counts: &mut BTreeMap<String, usize>,
                       exclusions: &mut Vec<String>| {
            let class = reason.split(':').next().unwrap_or("unknown").to_owned();
            *reason_counts.entry(class).or_insert(0) += 1;
            exclusions.push(format!("{job_id}:{reason}"));
        };
        let Some(claim) = claims.get(job_id) else {
            bail!("missing-receipt blocker claim disappeared");
        };
        if claim.manifest_sha256 != *manifest_sha256
            || claim.manifest.to_string_lossy() != *manifest_path
        {
            bail!("missing-receipt blocker claim hash/path mismatch");
        }
        if !(claim.state == "submitted" || claim.state.starts_with("ingest_failed:")) {
            exclude(
                format!("claim_state_not_retryable:{}", claim.state),
                &mut reason_counts,
                &mut exclusions,
            );
            continue;
        }
        let old_manifest = match validate_manifest(claim) {
            Ok(manifest) => manifest,
            Err(error) => {
                exclude(
                    format!("manifest_provenance_invalid:{error:#}"),
                    &mut reason_counts,
                    &mut exclusions,
                );
                continue;
            }
        };
        if strict_wave(&old_manifest)
            .is_some_and(|old_wave| wave_number(&old_wave) >= wave_number(&wave))
        {
            exclude(
                "current_or_future_wave_binding".to_owned(),
                &mut reason_counts,
                &mut exclusions,
            );
            continue;
        }
        if *has_failure {
            exclude(
                "semantic_or_terminal_pair_failure".to_owned(),
                &mut reason_counts,
                &mut exclusions,
            );
            continue;
        }
        let Some(evidence) = start_evidence.get(job_id) else {
            exclude(
                "no_durable_launch_or_completion_evidence".to_owned(),
                &mut reason_counts,
                &mut exclusions,
            );
            continue;
        };
        if scheduler.get(job_id).is_some_and(|(_, state)| {
            matches!(
                state.as_str(),
                "queued" | "starting" | "held" | "dispatching" | "running"
            )
        }) {
            exclude(
                "current_active_scheduler_state".to_owned(),
                &mut reason_counts,
                &mut exclusions,
            );
            continue;
        }
        let ingested: bool = corpus.query_row(
            "SELECT EXISTS(SELECT 1 FROM ingested_jobs WHERE job_id=?1)",
            [job_id],
            |row| row.get(0),
        )?;
        if ingested {
            exclude(
                "authoritative_ingested_receipt".to_owned(),
                &mut reason_counts,
                &mut exclusions,
            );
            continue;
        }
        let receipt_root = root.join("receipts").join(job_id);
        if tree_contains_named_file(&receipt_root, "SHARD_OK")? {
            exclude(
                "authoritative_local_receipt".to_owned(),
                &mut reason_counts,
                &mut exclusions,
            );
            continue;
        }
        if tree_contains_named_file(&receipt_root, "SHARD_FAIL.json")?
            || tree_contains_named_file(&runs_root.join(job_id), "SHARD_FAIL.json")?
        {
            exclude(
                "semantic_terminal_failure_output".to_owned(),
                &mut reason_counts,
                &mut exclusions,
            );
            continue;
        }
        if tree_contains_named_file(&runs_root.join(job_id), "SHARD_OK")? {
            exclude(
                "recoverable_run_receipt_exists".to_owned(),
                &mut reason_counts,
                &mut exclusions,
            );
            continue;
        }
        match probe_scq_output(scq, socket, job_id)? {
            OutputProbe::AuthoritativeReceipt => {
                exclude(
                    "recoverable_scq_receipt_exists".to_owned(),
                    &mut reason_counts,
                    &mut exclusions,
                );
                continue;
            }
            OutputProbe::SemanticFailure => {
                exclude(
                    "semantic_terminal_failure_output".to_owned(),
                    &mut reason_counts,
                    &mut exclusions,
                );
                continue;
            }
            OutputProbe::Unverified(error) => {
                exclude(
                    format!("output_unavailability_unverified:{error}"),
                    &mut reason_counts,
                    &mut exclusions,
                );
                continue;
            }
            OutputProbe::Unavailable(reason) => {
                let rows = manifest_pairs_lookup
                    .query_map([manifest_sha256], |row| {
                        Ok(ManifestPairBinding {
                            identity: RegistryPairIdentity {
                                task: row.get(0)?,
                                dataset_hash: row.get(1)?,
                                semantic_sha256: row.get(2)?,
                                protocol_sha256: row.get(3)?,
                            },
                            state: row.get(4)?,
                            manifest_path: row.get(5)?,
                            job_id: row.get(6)?,
                            receipt_sha256: row.get(7)?,
                            failure_class: row.get(8)?,
                        })
                    })?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                let Some(all_pairs) =
                    validate_missing_receipt_manifest_bindings(rows, manifest_path, job_id)
                else {
                    exclude(
                        "manifest_contains_authoritative_or_nonactive_pair".to_owned(),
                        &mut reason_counts,
                        &mut exclusions,
                    );
                    continue;
                };
                *reason_counts
                    .entry("eligible_infrastructure_retry_missing_receipt".to_owned())
                    .or_insert(0) += 1;
                candidates.push(MissingReceiptRecoveryCandidate {
                    job_id: job_id.clone(),
                    manifest: manifest_path.clone(),
                    manifest_sha256: manifest_sha256.clone(),
                    claim_state: claim.state.clone(),
                    scq_evidence: format!("{evidence}; {reason}"),
                    desired_pair_count: pairs.len(),
                    pair_count: all_pairs.len(),
                    pairs: all_pairs,
                });
            }
        }
    }
    drop(manifest_pairs_lookup);
    drop(pair);
    candidates.sort_by(|left, right| left.job_id.cmp(&right.job_id));
    exclusions.sort();
    let blocker_pairs = blockers.values().map(|entry| entry.2.len()).sum();
    let eligible_desired_pairs = candidates
        .iter()
        .map(|candidate| candidate.desired_pair_count)
        .sum();
    let eligible_total_pairs = candidates
        .iter()
        .map(|candidate| candidate.pair_count)
        .sum();
    Ok(MissingReceiptRecoveryPlan {
        wave,
        wave_dir: fs::canonicalize(wave_dir)?.to_string_lossy().into_owned(),
        desired_pairs: desired.len(),
        blocker_pairs,
        blocker_manifests: blockers.len(),
        eligible_desired_pairs,
        eligible_total_pairs,
        eligible_manifests: candidates.len(),
        reason_counts,
        exclusions,
        candidates,
    })
}

fn validate_missing_receipt_manifest_bindings(
    rows: Vec<ManifestPairBinding>,
    manifest_path: &str,
    job_id: &str,
) -> Option<Vec<RegistryPairIdentity>> {
    if rows.is_empty()
        || rows.iter().any(|row| {
            !matches!(row.state.as_str(), "reserved" | "submitted" | "held")
                || row.manifest_path.as_deref() != Some(manifest_path)
                || row.job_id.as_deref() != Some(job_id)
                || row.receipt_sha256.is_some()
                || row.failure_class.is_some()
        })
    {
        return None;
    }
    Some(rows.into_iter().map(|row| row.identity).collect())
}

fn load_wave_desired_pairs(
    wave_dir: &Path,
) -> Result<(String, BTreeSet<RegistryPairIdentity>, BTreeSet<String>)> {
    let wave_dir = fs::canonicalize(wave_dir)?;
    let wave = wave_dir
        .file_name()
        .and_then(|name| name.to_str())
        .context("wave directory name is invalid")?
        .to_owned();
    let summary: Value = serde_json::from_slice(&fs::read(wave_dir.join("summary.json"))?)?;
    let entries = summary
        .get("manifests")
        .and_then(Value::as_array)
        .context("wave summary manifests are absent")?;
    let mut desired = BTreeSet::new();
    let mut hashes = BTreeSet::new();
    for entry in entries {
        let path = wave_dir.join(required_json_str(entry, "relative_path")?);
        let expected_hash = required_json_str(entry, "sha256")?;
        if sha256_file(&path)? != expected_hash {
            bail!("wave desired manifest hash drift");
        }
        let manifest: Value = serde_json::from_slice(&fs::read(path)?)?;
        let task = required_json_str(&manifest, "task")?;
        let dataset = required_json_str(&manifest, "dataset_hash")?;
        let protocol = required_json_str(&manifest, "protocol_sha256")?;
        for candidate in manifest
            .get("candidates")
            .and_then(Value::as_array)
            .context("wave desired candidates are absent")?
        {
            desired.insert(RegistryPairIdentity {
                task: task.to_owned(),
                dataset_hash: dataset.to_owned(),
                semantic_sha256: required_json_str(candidate, "semantic_sha256")?.to_owned(),
                protocol_sha256: protocol.to_owned(),
            });
        }
        hashes.insert(expected_hash.to_owned());
    }
    if desired.len() % 256 != 0 || !matches!(desired.len() / 256, 128 | 192 | 256) {
        bail!(
            "wave desired identity count {} is not a legal micro-wave total",
            desired.len()
        );
    }
    Ok((wave, desired, hashes))
}

enum OutputProbe {
    AuthoritativeReceipt,
    SemanticFailure,
    Unavailable(String),
    Unverified(String),
}

fn probe_scq_output(scq: &Path, socket: &Path, job_id: &str) -> Result<OutputProbe> {
    let temporary = ProbeDirectory::create()?;
    let output = Command::new(scq)
        .arg("fetch")
        .arg(job_id)
        .arg("--output")
        .arg(&temporary.path)
        .arg("--socket")
        .arg(socket)
        .output()
        .with_context(|| format!("probe SCQ output for {job_id}"))?;
    if output.status.success() {
        if tree_contains_named_file(&temporary.path, "SHARD_OK")? {
            return Ok(OutputProbe::AuthoritativeReceipt);
        }
        if tree_contains_named_file(&temporary.path, "SHARD_FAIL.json")? {
            return Ok(OutputProbe::SemanticFailure);
        }
        return Ok(OutputProbe::Unavailable(
            "SCQ fetch returned no authoritative marker".to_owned(),
        ));
    }
    let error = String::from_utf8_lossy(if output.stderr.is_empty() {
        &output.stdout
    } else {
        &output.stderr
    })
    .trim()
    .to_owned();
    if error.contains("no such job")
        || error.contains("output is unavailable")
        || error.contains("no outputs")
    {
        Ok(OutputProbe::Unavailable(error))
    } else {
        Ok(OutputProbe::Unverified(error))
    }
}

struct ProbeDirectory {
    path: PathBuf,
}

impl ProbeDirectory {
    fn create() -> Result<Self> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "jope-scq-output-probe-{}-{}-{}",
            std::process::id(),
            now_ms(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path)?;
        Ok(Self { path })
    }
}

impl Drop for ProbeDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn tree_contains_named_file(root: &Path, name: &str) -> Result<bool> {
    if !root.exists() {
        return Ok(false);
    }
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        if kind.is_symlink() {
            bail!("output proof tree contains a symlink");
        }
        if kind.is_dir() {
            if tree_contains_named_file(&entry.path(), name)? {
                return Ok(true);
            }
        } else if entry.file_name() == name {
            return Ok(true);
        }
    }
    Ok(false)
}

fn build_wave_blocker_plan(
    root: &Path,
    requested_wave_dir: &Path,
    pair_registry: &Path,
    claims: &HashMap<String, Claim>,
    scheduler: &HashMap<String, (String, String)>,
    receipt_dir: &Path,
    scq_events: &Path,
) -> Result<WaveBlockerPlan> {
    let wave_dir = fs::canonicalize(requested_wave_dir)
        .with_context(|| format!("canonicalize wave {}", requested_wave_dir.display()))?;
    let waves_root = fs::canonicalize(root.join("waves"))?;
    if wave_dir.parent() != Some(waves_root.as_path()) {
        bail!("blocker retirement wave must be a direct child of the configured waves root");
    }
    let wave = wave_dir
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| name.starts_with("strict-wave-"))
        .context("wave directory name is not strict-wave-*")?
        .to_owned();
    let summary_path = wave_dir.join("summary.json");
    let cohort_path = wave_dir.join("cohort.json");
    let summary_bytes = fs::read(&summary_path)?;
    let cohort_bytes = fs::read(&cohort_path)?;
    let summary: Value = serde_json::from_slice(&summary_bytes)?;
    let cohort: Value = serde_json::from_slice(&cohort_bytes)?;
    let summary_sha256 = sha256_bytes(&summary_bytes);
    let cohort_sha256 = sha256_bytes(&cohort_bytes);
    let checkpoint_sha256 = required_json_str(&summary, "checkpoint_sha256")?.to_owned();
    let checkpoint_epoch = required_json_str(&summary, "checkpoint_epoch")?.to_owned();
    let catalog_sha256 = required_json_str(&summary, "catalog_sha256")?;
    if !is_sha256(&checkpoint_sha256)
        || checkpoint_epoch != format!("checkpoint-{checkpoint_sha256}")
        || !is_sha256(catalog_sha256)
        || summary.get("immutable").and_then(Value::as_bool) != Some(true)
        // Any legal micro-wave shape: tasks.len() * 64. The exact count is
        // cross-checked against the frozen cohort's entry list below, so this
        // is a range gate, not the authority.
        || !matches!(
            summary.get("datasets").and_then(Value::as_u64),
            Some(128 | 192 | 256)
        )
        || summary
            .get("candidates_per_dataset")
            .and_then(Value::as_u64)
            != Some(256)
        || summary.get("cohort_sha256").and_then(Value::as_str) != Some(cohort_sha256.as_str())
    {
        bail!("wave summary invariants failed");
    }
    let wave_datasets = summary
        .get("datasets")
        .and_then(Value::as_u64)
        .expect("validated above") as usize;
    if cohort.get("schema").and_then(Value::as_str) != Some("JopeMicroCohort.v2")
        || cohort.get("wave").and_then(Value::as_str) != Some(wave.as_str())
        || cohort.get("checkpoint_sha256").and_then(Value::as_str)
            != Some(checkpoint_sha256.as_str())
        || cohort.get("checkpoint_epoch").and_then(Value::as_str) != Some(checkpoint_epoch.as_str())
        || cohort.get("catalog_sha256").and_then(Value::as_str) != Some(catalog_sha256)
        || cohort.get("immutable").and_then(Value::as_bool) != Some(true)
        || cohort
            .get("entries")
            .and_then(Value::as_array)
            .map(Vec::len)
            != Some(wave_datasets)
    {
        bail!("wave cohort invariants failed");
    }

    let entries = summary
        .get("manifests")
        .and_then(Value::as_array)
        .context("wave summary manifests are absent")?;
    // manifests = datasets * ceil(256 / max_candidates_per_manifest), derived
    // from the summary's own declared shard cap (the historical literal 384 was
    // 128 datasets at the never-used cap of 100 -- 128 * 3 shards).
    let max_shard = summary
        .get("max_candidates_per_manifest")
        .and_then(Value::as_u64)
        .unwrap_or(100)
        .max(1) as usize;
    let expected_manifests = wave_datasets * 256usize.div_ceil(max_shard);
    if summary.get("manifest_count").and_then(Value::as_u64) != Some(entries.len() as u64)
        || entries.len() != expected_manifests
    {
        bail!(
            "wave summary manifest count {} is not the expected {expected_manifests}",
            entries.len()
        );
    }
    let mut desired = BTreeSet::<(String, String, String, String)>::new();
    let mut current_manifest_hashes = BTreeSet::new();
    let mut seen_paths = BTreeSet::new();
    for entry in entries {
        let relative = required_json_str(entry, "relative_path")?;
        let expected_hash = required_json_str(entry, "sha256")?;
        let expected_task = required_json_str(entry, "task")?;
        let expected_dataset = required_json_str(entry, "dataset_hash")?;
        let expected_candidates = entry
            .get("candidate_count")
            .and_then(Value::as_u64)
            .context("summary candidate_count is absent")?
            as usize;
        if !is_sha256(expected_hash)
            || !seen_paths.insert(relative.to_owned())
            || !matches!(
                expected_task,
                "regression" | "binary" | "timeseries" | "multiclass"
            )
        {
            bail!("wave summary manifest entry is invalid or duplicated");
        }
        let manifest_path = wave_dir.join(relative);
        let canonical_manifest = fs::canonicalize(&manifest_path)?;
        if canonical_manifest.parent() != Some(wave_dir.join("manifests").as_path()) {
            bail!("wave manifest escapes the immutable manifests directory");
        }
        if sha256_file(&canonical_manifest)? != expected_hash {
            bail!("current wave manifest hash drift");
        }
        let claim = Claim {
            manifest_sha256: expected_hash.to_owned(),
            manifest: canonical_manifest,
            state: "submitted".to_owned(),
        };
        let manifest = validate_manifest(&claim)?;
        if strict_wave(&manifest).as_deref() != Some(wave.as_str())
            || manifest.get("checkpoint_sha256").and_then(Value::as_str)
                != Some(checkpoint_sha256.as_str())
            || manifest.get("checkpoint_epoch").and_then(Value::as_str)
                != Some(checkpoint_epoch.as_str())
            || manifest.get("catalog_sha256").and_then(Value::as_str) != Some(catalog_sha256)
            || manifest.get("task").and_then(Value::as_str) != Some(expected_task)
            || manifest.get("dataset_hash").and_then(Value::as_str) != Some(expected_dataset)
        {
            bail!("current wave manifest disagrees with its summary");
        }
        let protocol = required_json_str(&manifest, "protocol_sha256")?;
        if !is_sha256(protocol) {
            bail!("current wave protocol is not SHA-256");
        }
        let candidates = manifest
            .get("candidates")
            .and_then(Value::as_array)
            .context("current wave candidates are absent")?;
        if candidates.len() != expected_candidates {
            bail!("current wave candidate count disagrees with summary");
        }
        for candidate in candidates {
            let semantic = required_json_str(candidate, "semantic_sha256")?;
            if !desired.insert((
                expected_task.to_owned(),
                expected_dataset.to_owned(),
                semantic.to_owned(),
                protocol.to_owned(),
            )) {
                bail!("current wave desired pair identity is duplicated");
            }
        }
        current_manifest_hashes.insert(expected_hash.to_owned());
    }
    if !desired.len().is_multiple_of(256) || !matches!(desired.len() / 256, 128 | 192 | 256) {
        bail!(
            "current wave desired pair count {} is not a legal micro-wave total",
            desired.len()
        );
    }

    let connection = Connection::open_with_flags(
        pair_registry,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    connection.busy_timeout(Duration::from_secs(60))?;
    connection.execute_batch(
        "PRAGMA temp_store=MEMORY;
         CREATE TEMP TABLE desired_pairs(
           task TEXT NOT NULL,dataset_hash TEXT NOT NULL,
           semantic_sha256 TEXT NOT NULL,protocol_sha256 TEXT NOT NULL,
           PRIMARY KEY(task,dataset_hash,semantic_sha256,protocol_sha256)
         ) WITHOUT ROWID;",
    )?;
    {
        let mut insert = connection.prepare("INSERT INTO desired_pairs VALUES(?1,?2,?3,?4)")?;
        for (task, dataset, semantic, protocol) in &desired {
            insert.execute(params![task, dataset, semantic, protocol])?;
        }
    }
    let mut statement = connection.prepare(DESIRED_BLOCKER_QUERY)?;
    let rows = statement
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, Option<String>>(1)?,
                row.get::<_, Option<String>>(2)?,
                row.get::<_, Option<String>>(3)?,
            ))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    drop(statement);
    drop(connection);

    let held_proofs = load_held_retirement_proofs(receipt_dir)?;
    let blocker_job_ids = rows
        .iter()
        .filter_map(|(_state, _path, manifest, job)| {
            let manifest = manifest.as_ref()?;
            (!current_manifest_hashes.contains(manifest))
                .then(|| job.clone())
                .flatten()
        })
        .collect::<BTreeSet<_>>();
    let started_jobs = load_scq_start_evidence(scq_events, &blocker_job_ids)?;
    let mut current_wave_active_pairs = 0usize;
    let mut blocking_pairs_by_manifest = BTreeMap::<String, usize>::new();
    let mut candidates_by_manifest = BTreeMap::<String, HoldCandidate>::new();
    let mut bindings_by_manifest = BTreeMap::<String, (String, String)>::new();
    let mut seen_jobs = BTreeMap::<String, String>::new();
    let mut proof_errors = Vec::new();
    let mut exact_submitted_claims = 0usize;
    for (_state, manifest_path, manifest_sha256, job_id) in rows {
        let (Some(manifest_path), Some(manifest_sha256), Some(job_id)) =
            (manifest_path, manifest_sha256, job_id)
        else {
            continue;
        };
        if current_manifest_hashes.contains(&manifest_sha256) {
            current_wave_active_pairs += 1;
            continue;
        }
        *blocking_pairs_by_manifest
            .entry(manifest_sha256.clone())
            .or_insert(0) += 1;
        if let Some(existing_job) = seen_jobs.insert(job_id.clone(), manifest_sha256.clone()) {
            if existing_job != manifest_sha256 {
                bail!("one stale job is bound to multiple manifest hashes");
            }
        }
        if let Some((existing_job, existing_path)) = bindings_by_manifest.get(&manifest_sha256) {
            if existing_job != &job_id || existing_path != &manifest_path {
                bail!("stale manifest pair bindings disagree on job or path");
            }
            continue;
        }
        bindings_by_manifest.insert(
            manifest_sha256.clone(),
            (job_id.clone(), manifest_path.clone()),
        );
        let claim = claims
            .get(&job_id)
            .with_context(|| format!("blocking job {job_id} has no backlog claim"))?;
        if claim.manifest_sha256 != manifest_sha256
            || claim.manifest.to_string_lossy() != manifest_path
        {
            bail!(
                "blocking job {job_id} manifest binding does not exactly match \
                 its claim (hash_match={}, path_match={})",
                claim.manifest_sha256 == manifest_sha256,
                claim.manifest.to_string_lossy() == manifest_path
            );
        }
        if claim.state != "submitted" {
            proof_errors.push(format!("{}: claim_not_submitted:{}", job_id, claim.state));
            continue;
        }
        exact_submitted_claims += 1;
        let old_manifest = match validate_manifest(claim) {
            Ok(manifest) => manifest,
            Err(error) => {
                proof_errors.push(format!(
                    "{job_id}: blocking_manifest_provenance_invalid:{error:#}"
                ));
                continue;
            }
        };
        let Some(old_wave) = strict_wave(&old_manifest) else {
            proof_errors.push(format!("{job_id}: blocking_manifest_has_no_strict_wave"));
            continue;
        };
        if wave_number(&old_wave) >= wave_number(&wave) {
            proof_errors.push(format!(
                "{job_id}: blocking_manifest_is_current_or_future_wave:{old_wave}"
            ));
            continue;
        }
        if let Some(reason) = started_jobs.get(&job_id) {
            proof_errors.push(format!("{job_id}: {reason}; excluded from recycling"));
            continue;
        }
        let observed_state = match scheduler.get(&job_id) {
            Some((project, state)) if project == SELFPLAY_PROJECT && state == "cancelled" => {
                Some("cancelled")
            }
            Some((_project, state)) => {
                proof_errors.push(format!("{job_id}: excluded live SCQ state {state}"));
                None
            }
            None if held_proofs.contains(&(
                job_id.clone(),
                manifest_sha256.clone(),
                manifest_path.clone(),
            )) =>
            {
                Some("held")
            }
            None => {
                proof_errors.push(format!(
                    "{job_id}: pruned from SCQ without held/unstarted retirement proof"
                ));
                None
            }
        };
        let Some(observed_state) = observed_state else {
            continue;
        };
        candidates_by_manifest.insert(
            manifest_sha256.clone(),
            HoldCandidate {
                job_id,
                project: SELFPLAY_PROJECT.to_owned(),
                observed_state: observed_state.to_owned(),
                epoch: manifest_epoch(&old_manifest),
                strict_wave: Some(old_wave),
                generation: generation(&old_manifest),
                manifest: manifest_path,
                manifest_sha256,
            },
        );
    }
    let blocking_pairs = blocking_pairs_by_manifest.values().sum();
    let candidates = candidates_by_manifest.into_values().collect::<Vec<_>>();
    let blocking_manifests = bindings_by_manifest.len();
    Ok(WaveBlockerPlan {
        wave,
        wave_dir: wave_dir.to_string_lossy().into_owned(),
        summary_sha256,
        cohort_sha256,
        checkpoint_sha256,
        checkpoint_epoch,
        desired_pairs: desired.len(),
        current_wave_active_pairs,
        blocking_pairs,
        blocking_claims: blocking_manifests,
        exact_submitted_claims,
        blocking_manifests,
        proven_blocking_manifests: candidates.len(),
        unproven_blocking_manifests: blocking_manifests - candidates.len(),
        proof_errors,
        candidates,
    })
}

fn load_held_retirement_proofs(receipt_dir: &Path) -> Result<BTreeSet<(String, String, String)>> {
    let mut proofs = BTreeSet::new();
    if !receipt_dir.is_dir() {
        return Ok(proofs);
    }
    for entry in fs::read_dir(receipt_dir)? {
        let path = entry?.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let Some(expected_hash) = name
            .strip_prefix("preview-")
            .and_then(|value| value.strip_suffix(".json"))
        else {
            continue;
        };
        if !is_sha256(expected_hash) || sha256_file(&path)? != expected_hash {
            continue;
        }
        let value: Value = serde_json::from_slice(&fs::read(&path)?)?;
        if value.get("schema").and_then(Value::as_str) != Some("JopeQueueRetirementPreview.v1") {
            continue;
        }
        let Some(candidates) = value.get("candidates").and_then(Value::as_array) else {
            continue;
        };
        for candidate in candidates {
            let candidate: HoldCandidate = serde_json::from_value(candidate.clone())?;
            if candidate.project == SELFPLAY_PROJECT && candidate.observed_state == "held" {
                proofs.insert((
                    candidate.job_id,
                    candidate.manifest_sha256,
                    candidate.manifest,
                ));
            }
        }
    }
    Ok(proofs)
}

fn load_scq_start_evidence(
    event_log: &Path,
    blocker_jobs: &BTreeSet<String>,
) -> Result<BTreeMap<String, String>> {
    if blocker_jobs.is_empty() {
        return Ok(BTreeMap::new());
    }
    let file = File::open(event_log)
        .with_context(|| format!("open SCQ event history {}", event_log.display()))?;
    let mut evidence = BTreeMap::new();
    for (line_number, line) in BufReader::new(file).lines().enumerate() {
        let line = line?;
        if !line.contains("\"job\"") {
            continue;
        }
        let event: Value = serde_json::from_str(&line).with_context(|| {
            format!(
                "parse SCQ event history {}:{}",
                event_log.display(),
                line_number + 1
            )
        })?;
        let Some(job_id) = event.get("job").and_then(Value::as_str) else {
            continue;
        };
        if !blocker_jobs.contains(job_id) {
            continue;
        }
        let kind = event
            .get("kind")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let class = event
            .get("class")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if kind == "exit" && class == "completed" {
            evidence.insert(
                job_id.to_owned(),
                "durable SCQ history records completion".to_owned(),
            );
        } else if matches!(kind, "launch" | "running") {
            evidence
                .entry(job_id.to_owned())
                .or_insert_with(|| "durable SCQ history records launch".to_owned());
        }
    }
    Ok(evidence)
}

fn revalidate_wave_blocker_plan(
    plan: &WaveBlockerPlan,
    scheduler: &HashMap<String, (String, String)>,
    claims: &HashMap<String, Claim>,
    receipt_dir: &Path,
    scq_events: &Path,
) -> Result<()> {
    let held_proofs = load_held_retirement_proofs(receipt_dir)?;
    let job_ids = plan
        .candidates
        .iter()
        .map(|candidate| candidate.job_id.clone())
        .collect::<BTreeSet<_>>();
    let started_jobs = load_scq_start_evidence(scq_events, &job_ids)?;
    for candidate in &plan.candidates {
        if let Some(reason) = started_jobs.get(&candidate.job_id) {
            bail!("blocker gained start evidence before transaction: {reason}");
        }
        let claim = claims
            .get(&candidate.job_id)
            .context("blocker claim disappeared before transaction")?;
        if !claim_matches_candidate(claim, candidate) {
            bail!("blocker claim changed before transaction");
        }
        let manifest = validate_manifest(claim)?;
        let old_wave = strict_wave(&manifest).context("blocker strict wave disappeared")?;
        if wave_number(&old_wave) >= wave_number(&plan.wave) {
            bail!("blocker provenance advanced before transaction");
        }
        match candidate.observed_state.as_str() {
            "cancelled"
                if scheduler
                    .get(&candidate.job_id)
                    .is_some_and(|(project, state)| {
                        project == SELFPLAY_PROJECT && state == "cancelled"
                    }) => {}
            "held"
                if !scheduler.contains_key(&candidate.job_id)
                    && held_proofs.contains(&(
                        candidate.job_id.clone(),
                        candidate.manifest_sha256.clone(),
                        candidate.manifest.clone(),
                    )) => {}
            _ => bail!("blocker SCQ state or held proof changed before transaction"),
        }
    }
    Ok(())
}

fn required_json_str<'a>(value: &'a Value, field: &str) -> Result<&'a str> {
    value
        .get(field)
        .and_then(Value::as_str)
        .with_context(|| format!("{field} is absent"))
}

fn apply_retirement_registry(
    pair_registry: &Path,
    backlog: &Path,
    candidates: &[HoldCandidate],
    results: &mut [RetirementResult],
    preview_sha256: &str,
) -> Result<usize> {
    if !pair_registry.is_file() || !backlog.is_file() {
        bail!("retirement requires pair registry and backlog databases");
    }
    let by_job = candidates
        .iter()
        .map(|candidate| (candidate.job_id.as_str(), candidate))
        .collect::<HashMap<_, _>>();
    let mut connection = Connection::open(pair_registry)
        .with_context(|| format!("open pair registry {}", pair_registry.display()))?;
    connection.busy_timeout(Duration::from_secs(60))?;
    connection.execute_batch("PRAGMA cache_size=-262144; PRAGMA temp_store=MEMORY;")?;
    connection.execute(
        "ATTACH DATABASE ?1 AS backlog_db",
        [backlog.to_string_lossy().as_ref()],
    )?;
    let owner_column = if sqlite_column_exists(&connection, "pairs", "reservation_owner")? {
        "reservation_owner"
    } else if sqlite_column_exists(&connection, "pairs", "owner")? {
        "owner"
    } else {
        bail!("pair registry has no ownership column");
    };
    let has_attempts = sqlite_table_exists(&connection, "attempts")?;
    let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    transaction.execute_batch(
        "CREATE TEMP TABLE retiring_jobs(
           manifest_sha256 TEXT PRIMARY KEY,
           job_id TEXT NOT NULL UNIQUE,
           detail_json TEXT NOT NULL
         ) WITHOUT ROWID;",
    )?;
    let mut accepted = BTreeSet::new();
    for result in results.iter_mut() {
        if result.command_error.is_some() {
            continue;
        }
        let safely_unstarted = result
            .state_after_cancel
            .as_deref()
            .is_none_or(|state| state == "cancelled");
        if !safely_unstarted || (!result.cancel_requested && !result.recovery_from_prior_preview) {
            result.registry_error = Some("cancel_not_confirmed_unstarted".to_owned());
            continue;
        }
        let candidate = by_job
            .get(result.job_id.as_str())
            .context("retirement candidate disappeared")?;
        let claim_state: Option<String> = transaction
            .query_row(
                "SELECT state FROM backlog_db.claims
                 WHERE job_id=?1 AND manifest_sha256=?2 AND manifest=?3",
                params![
                    candidate.job_id,
                    candidate.manifest_sha256,
                    candidate.manifest
                ],
                |row| row.get(0),
            )
            .optional()?;
        if claim_state.as_deref() == Some("superseded_unstarted") {
            continue;
        }
        if claim_state.as_deref() != Some("submitted") {
            result.registry_error = Some("backlog_claim_no_longer_submitted".to_owned());
            continue;
        }
        let detail = serde_json::to_string(&json!({
            "job_id": candidate.job_id,
            "manifest_sha256": candidate.manifest_sha256,
            "preview_sha256": preview_sha256,
            "reason": "superseded_unstarted",
        }))?;
        transaction.execute(
            "INSERT INTO retiring_jobs VALUES(?1,?2,?3)",
            params![candidate.manifest_sha256, candidate.job_id, detail],
        )?;
        accepted.insert(candidate.job_id.clone());
    }

    let timestamp = now_ms() as i64;
    transaction.execute(
        "INSERT INTO events(kind,detail_json,created_at_ms)
         SELECT 'superseded_unstarted',r.detail_json,?1 FROM retiring_jobs r",
        [timestamp],
    )?;
    if has_attempts {
        transaction.execute(
            "UPDATE attempts SET state='superseded_unstarted',
               receipt_sha256=COALESCE(receipt_sha256,?1),
               job_id=COALESCE(
                 job_id,
                 (SELECT r.job_id FROM retiring_jobs r
                  WHERE r.manifest_sha256=attempts.manifest_sha256)
               ),finalized_at_ms=?2
             WHERE manifest_sha256 IN (SELECT manifest_sha256 FROM retiring_jobs)
               AND state IN ('reserved','submitted','held')
               AND finalized_at_ms IS NULL",
            params![preview_sha256, timestamp],
        )?;
    }
    let bound_pairs: usize = transaction.query_row(
        "SELECT COUNT(*) FROM pairs INDEXED BY pair_manifest
         WHERE manifest_sha256 IN (SELECT manifest_sha256 FROM retiring_jobs)",
        [],
        |row| row.get(0),
    )?;
    let missing_manifest: Option<String> = transaction
        .query_row(
            "SELECT r.job_id FROM retiring_jobs r
             WHERE NOT EXISTS(
               SELECT 1 FROM pairs p INDEXED BY pair_manifest
               WHERE p.manifest_sha256=r.manifest_sha256
             ) LIMIT 1",
            [],
            |row| row.get(0),
        )
        .optional()?;
    if let Some(job_id) = missing_manifest {
        bail!("{job_id}: pair retirement has no pair-registry provenance");
    }
    // Sparse blocker repair is fastest through pair_manifest; mass retirement
    // must scan the WITHOUT ROWID primary table to avoid hundreds of thousands
    // of random primary-row fetches. Both paths retain immutable old bindings
    // until the next reservation, avoiding secondary-index churn.
    let update = if candidates.len() <= 128 {
        format!(
            "UPDATE pairs INDEXED BY pair_manifest
             SET state='available',{owner_column}='available',
               manifest_path=NULL,manifest_sha256=NULL,job_id=NULL,
               failure_class=NULL,updated_at_ms=?1
             WHERE manifest_sha256 IN(
               SELECT manifest_sha256 FROM retiring_jobs
             )
               AND state IN ('reserved','submitted','held')
               AND receipt_sha256 IS NULL
               AND (
                 job_id IS NULL OR job_id=(
                   SELECT r.job_id FROM retiring_jobs r
                   WHERE r.manifest_sha256=pairs.manifest_sha256
                 )
               )"
        )
    } else {
        format!(
            "UPDATE pairs NOT INDEXED
             SET state='available',{owner_column}='available',
               manifest_path=NULL,manifest_sha256=NULL,job_id=NULL,
               failure_class=NULL,updated_at_ms=?1
             WHERE state IN ('reserved','submitted','held')
               AND receipt_sha256 IS NULL
               AND EXISTS(
                 SELECT 1 FROM retiring_jobs r
                 WHERE r.manifest_sha256=pairs.manifest_sha256
                   AND (pairs.job_id IS NULL OR pairs.job_id=r.job_id)
               )"
        )
    };
    let updated_pairs = transaction.execute(&update, [timestamp])?;
    // The pre-update count is index-only. Equality proves every pair bound to a retiring
    // manifest passed the active-state, empty-receipt, and job provenance
    // predicates above. Any authoritative or mismatched row rolls back all
    // pair, attempt, event, and claim changes.
    if bound_pairs != updated_pairs {
        bail!(
            "pair retirement provenance mismatch: {updated_pairs} active rows \
             but {bound_pairs} total manifest-bound rows"
        );
    }
    let remaining_bound: usize = transaction.query_row(
        "SELECT COUNT(*) FROM pairs INDEXED BY pair_manifest
         WHERE manifest_sha256 IN (SELECT manifest_sha256 FROM retiring_jobs)",
        [],
        |row| row.get(0),
    )?;
    if remaining_bound != 0 {
        bail!("pair retirement did not clear active manifest bindings");
    }
    let updated_claims = transaction.execute(
        "UPDATE backlog_db.claims
         SET state='superseded_unstarted',updated_at_ms=?1
         WHERE state='submitted' AND EXISTS(
           SELECT 1 FROM retiring_jobs r
           WHERE r.job_id=backlog_db.claims.job_id
             AND r.manifest_sha256=backlog_db.claims.manifest_sha256
         )",
        [timestamp],
    )?;
    if updated_claims != accepted.len() {
        bail!("bulk backlog retirement count changed");
    }
    transaction.commit()?;
    for result in results.iter_mut() {
        if accepted.contains(&result.job_id) {
            result.claim_updated = true;
        }
    }
    Ok(updated_pairs)
}

fn sqlite_table_exists(connection: &Connection, table: &str) -> Result<bool> {
    Ok(connection.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name=?1)",
        [table],
        |row| row.get(0),
    )?)
}

fn sqlite_column_exists(connection: &Connection, table: &str, column: &str) -> Result<bool> {
    let query = format!("SELECT COUNT(*) FROM pragma_table_info('{table}') WHERE name=?1");
    Ok(connection.query_row(&query, [column], |row| row.get::<_, i64>(0))? != 0)
}

fn write_content_addressed(
    directory: &Path,
    prefix: &str,
    value: &Value,
) -> Result<(PathBuf, String)> {
    fs::create_dir_all(directory)?;
    let mut bytes = canonical_json_bytes(value)?;
    bytes.push(b'\n');
    let digest = sha256_bytes(&bytes);
    let path = directory.join(format!("{prefix}-{digest}.json"));
    match OpenOptions::new().write(true).create_new(true).open(&path) {
        Ok(mut file) => {
            file.write_all(&bytes)?;
            file.sync_all()?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            if sha256_file(&path)? != digest {
                bail!("content-addressed receipt collision at {}", path.display());
            }
        }
        Err(error) => return Err(error.into()),
    }
    Ok((path, digest))
}

fn write_json_atomic(path: &Path, value: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension(format!("tmp.{}", std::process::id()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    file.write_all(&canonical_json_bytes(value)?)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    fs::rename(&temporary, path)?;
    Ok(())
}

/// The exact ranking key the seed-bank SQL uses (`ORDER BY mean_percentile
/// DESC, worst_percentile DESC, strict_wins DESC, dataset_count DESC,
/// semantic_sha256`). Both source pools are sorted by this key, so merging
/// preserves the global merit order.
fn seed_rank_order(left: &SeedRow, right: &SeedRow) -> std::cmp::Ordering {
    right
        .mean_within_dataset_percentile
        .total_cmp(&left.mean_within_dataset_percentile)
        .then_with(|| {
            right
                .worst_within_dataset_percentile
                .total_cmp(&left.worst_within_dataset_percentile)
        })
        .then_with(|| right.strict_win_count.cmp(&left.strict_win_count))
        .then_with(|| right.dataset_count.cmp(&left.dataset_count))
        .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256))
}

/// Merge one task's non-DL and DL seed pools (each already in rank order)
/// keeping global merit order, with DL-origin entries capped at `dl_cap`.
/// A semantic present in both pools earned its slot without DL evidence and
/// is kept as the non-DL entry only, so it never consumes DL cap.
fn merge_seed_pools(
    non_dl: Vec<SeedRow>,
    dl: Vec<SeedRow>,
    limit: usize,
    dl_cap: usize,
) -> Vec<SeedRow> {
    let non_dl_semantics: BTreeSet<String> = non_dl
        .iter()
        .map(|row| row.semantic_sha256.clone())
        .collect();
    let mut dl: Vec<SeedRow> = dl
        .into_iter()
        .filter(|row| !non_dl_semantics.contains(&row.semantic_sha256))
        .collect();
    dl.truncate(dl_cap);
    let mut merged = Vec::new();
    let mut non_dl = non_dl.into_iter().peekable();
    let mut dl = dl.into_iter().peekable();
    while merged.len() < limit {
        let take_dl = match (non_dl.peek(), dl.peek()) {
            (Some(left), Some(right)) => seed_rank_order(right, left) == std::cmp::Ordering::Less,
            (None, Some(_)) => true,
            (Some(_), None) => false,
            (None, None) => break,
        };
        let row = if take_dl { dl.next() } else { non_dl.next() };
        merged.push(row.expect("peeked branch is non-empty"));
    }
    merged
}

fn query_seed_rows(
    connection: &Connection,
    minimum_datasets: usize,
    limit: usize,
    dl_origin: bool,
) -> Result<Vec<SeedRow>> {
    let source_predicate = if dl_origin {
        "candidate_source IN ('accepted_dl_checkpoint','shadow_dl_direct')"
    } else {
        "candidate_source NOT IN ('accepted_dl_checkpoint','shadow_dl_direct')"
    };
    let sql = format!(
        r#"
WITH ranked AS (
  SELECT task,dataset_hash,semantic_sha256,canonical_genome,score,
         ROW_NUMBER() OVER (
           PARTITION BY task,dataset_hash ORDER BY score,semantic_sha256
         ) AS ascending_rank,
         COUNT(*) OVER (PARTITION BY task,dataset_hash) AS dataset_size,
         RANK() OVER (
           PARTITION BY task,dataset_hash ORDER BY score DESC
         ) AS descending_rank
  FROM observations
  WHERE task IN ('regression','binary')
    AND {source_predicate}
),
aggregated AS (
  SELECT task,semantic_sha256,MAX(canonical_genome) AS canonical_genome,
         COUNT(*) AS dataset_count,
         AVG(CASE WHEN dataset_size=1 THEN 1.0
                  ELSE 1.0*(ascending_rank-1)/(dataset_size-1) END) AS mean_percentile,
         MIN(CASE WHEN dataset_size=1 THEN 1.0
                  ELSE 1.0*(ascending_rank-1)/(dataset_size-1) END) AS worst_percentile,
         SUM(CASE WHEN descending_rank=1 THEN 1 ELSE 0 END) AS strict_wins
  FROM ranked
  GROUP BY task,semantic_sha256
  HAVING COUNT(*) >= ?1
),
numbered AS (
  SELECT *,
         ROW_NUMBER() OVER (
           PARTITION BY task
           ORDER BY mean_percentile DESC,worst_percentile DESC,
                    strict_wins DESC,dataset_count DESC,semantic_sha256
         ) AS robust_rank
  FROM aggregated
)
SELECT task,semantic_sha256,canonical_genome,dataset_count,
       mean_percentile,worst_percentile,strict_wins
FROM numbered
WHERE robust_rank <= ?2
ORDER BY task,robust_rank
"#
    );
    let mut statement = connection.prepare(&sql)?;
    let rows = statement.query_map(
        rusqlite::params![minimum_datasets as i64, limit as i64],
        |row| {
            Ok(SeedRow {
                task: row.get(0)?,
                semantic_sha256: row.get(1)?,
                canonical_genome: row.get(2)?,
                dataset_count: row.get(3)?,
                mean_within_dataset_percentile: row.get(4)?,
                worst_within_dataset_percentile: row.get(5)?,
                strict_win_count: row.get(6)?,
                dl_origin,
            })
        },
    )?;
    let mut seeds = Vec::new();
    for row in rows {
        let row = row?;
        if sha256_bytes(row.canonical_genome.as_bytes()) != row.semantic_sha256 {
            bail!("corpus semantic hash drift for {}", row.semantic_sha256);
        }
        seeds.push(row);
    }
    Ok(seeds)
}

fn export_seed_bank(path: &Path, limit: usize, minimum_datasets: usize) -> Result<Value> {
    let connection = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .with_context(|| format!("open corpus {}", path.display()))?;
    connection.busy_timeout(std::time::Duration::from_secs(60))?;
    connection.execute_batch("PRAGMA temp_store=FILE; PRAGMA cache_size=-262144;")?;
    let dl_cap = limit / 3;
    let non_dl_rows = query_seed_rows(&connection, minimum_datasets, limit, false)?;
    // Fetch the DL pool at full limit so semantics deduped into the non-DL
    // pool do not underfill the cap; merge_seed_pools truncates to dl_cap
    // after dedupe.
    let dl_rows = query_seed_rows(&connection, minimum_datasets, limit, true)?;
    let mut non_dl_by_task: BTreeMap<String, Vec<SeedRow>> = BTreeMap::new();
    for row in non_dl_rows {
        non_dl_by_task
            .entry(row.task.clone())
            .or_default()
            .push(row);
    }
    let mut dl_by_task: BTreeMap<String, Vec<SeedRow>> = BTreeMap::new();
    for row in dl_rows {
        dl_by_task.entry(row.task.clone()).or_default().push(row);
    }
    let mut by_task: BTreeMap<String, Vec<SeedRow>> = BTreeMap::new();
    let tasks: BTreeSet<String> = non_dl_by_task
        .keys()
        .chain(dl_by_task.keys())
        .cloned()
        .collect();
    let mut dl_origin_counts = serde_json::Map::new();
    for task in tasks {
        let merged = merge_seed_pools(
            non_dl_by_task.remove(&task).unwrap_or_default(),
            dl_by_task.remove(&task).unwrap_or_default(),
            limit,
            dl_cap,
        );
        let dl_count = merged.iter().filter(|row| row.dl_origin).count();
        dl_origin_counts.insert(
            task.clone(),
            json!({
                "total": merged.len(),
                "dl_origin": dl_count,
                "non_dl": merged.len() - dl_count,
            }),
        );
        by_task.insert(task, merged);
    }
    Ok(json!({
        "schema": "JopeRobustGpSeedBank.v1",
        "created_at_ms": now_ms(),
        "source": path,
        "selection": {
            "task_separated": true,
            "minimum_repeat_datasets": minimum_datasets,
            "limit_per_task": limit,
            "within_dataset_rank": "percentile; higher is better",
            "order": [
                "mean_within_dataset_percentile_desc",
                "worst_within_dataset_percentile_desc",
                "strict_win_count_desc",
                "dataset_count_desc",
                "semantic_sha256_asc"
            ],
            "dl_origin_source": ["accepted_dl_checkpoint", "shadow_dl_direct"],
            "dl_origin_cap_per_task": dl_cap,
            "dl_origin_policy": "non-DL and accepted/shadow DL pools ranked separately by the same key, then merged in rank order; their combined DL-origin entries are capped at floor(limit/3); a semantic present in both pools counts as non-DL"
        },
        "dl_origin_counts": dl_origin_counts,
        "tasks": by_task,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::net::UnixListener;

    fn record(project: &str, state: &str, epoch: &str, wave: Option<&str>) -> JobRecord {
        JobRecord {
            job_id: format!("{project}-{state}-{epoch}-{}", wave.unwrap_or("none")),
            project: project.to_owned(),
            state: state.to_owned(),
            epoch: epoch.to_owned(),
            strict_wave: wave.map(str::to_owned),
            generation: wave.map(str::to_owned),
            checkpoint_sha256: None,
            claim_state: Some("submitted".to_owned()),
            manifest: Some("/manifest".to_owned()),
            manifest_sha256: Some("a".repeat(64)),
            manifest_checked: true,
            manifest_valid: true,
            invalid_reason: None,
        }
    }

    #[test]
    fn legacy_hold_excludes_running_and_unclaimed() {
        let mut rows = vec![
            record(LEGACY_PROJECT, "queued", LEGACY_EPOCH, None),
            record(LEGACY_PROJECT, "running", LEGACY_EPOCH, None),
        ];
        let mut unclaimed = record(LEGACY_PROJECT, "queued", LEGACY_EPOCH, None);
        unclaimed.claim_state = None;
        rows.push(unclaimed);
        assert_eq!(hold_candidates(&rows, true, false, None).len(), 1);
    }

    #[test]
    fn superseded_wave_hold_keeps_newest_and_all_running() {
        let rows = vec![
            record(
                SELFPLAY_PROJECT,
                "queued",
                LEGACY_EPOCH,
                Some("strict-wave-000009"),
            ),
            record(
                SELFPLAY_PROJECT,
                "running",
                LEGACY_EPOCH,
                Some("strict-wave-000009"),
            ),
            record(
                SELFPLAY_PROJECT,
                "queued",
                LEGACY_EPOCH,
                Some("strict-wave-000010"),
            ),
        ];
        assert_eq!(newest_wave(&rows).as_deref(), Some("strict-wave-000010"));
        let held = hold_candidates(&rows, false, true, Some("strict-wave-000010"));
        assert_eq!(held.len(), 1);
        assert_eq!(held[0].strict_wave.as_deref(), Some("strict-wave-000009"));
    }

    #[test]
    fn protected_lanes_fail_scope() {
        for lane in ["checkpoint_gate", "sealed_audit", "evaluator_canary"] {
            assert!(protected_lane(&json!({"evaluation_lane": lane})));
        }
    }

    #[test]
    fn checkpoint_bridge_is_exact_bounded_and_newest_wave_first() {
        let checkpoint = "f".repeat(64);
        let mut older = record(
            SELFPLAY_PROJECT,
            "held",
            "checkpoint-1",
            Some("strict-wave-000009"),
        );
        older.job_id = "job-a".to_owned();
        older.checkpoint_sha256 = Some(checkpoint.clone());
        let mut newer_b = record(
            SELFPLAY_PROJECT,
            "held",
            "checkpoint-1",
            Some("strict-wave-000010"),
        );
        newer_b.job_id = "job-b".to_owned();
        newer_b.checkpoint_sha256 = Some(checkpoint.clone());
        let mut newer_a = newer_b.clone();
        newer_a.job_id = "job-a-new".to_owned();
        let mut wrong_checkpoint = older.clone();
        wrong_checkpoint.job_id = "job-wrong".to_owned();
        wrong_checkpoint.checkpoint_sha256 = Some("e".repeat(64));
        let mut running = newer_b.clone();
        running.job_id = "job-running".to_owned();
        running.state = "running".to_owned();

        let selected = checkpoint_bridge_candidates(
            &[older, newer_b, newer_a, wrong_checkpoint, running],
            &checkpoint,
            2,
        );
        assert_eq!(
            selected
                .iter()
                .map(|candidate| candidate.job_id.as_str())
                .collect::<Vec<_>>(),
            ["job-a-new", "job-b"]
        );

        let manifest = json!({
            "checkpoint_sha256": checkpoint.clone(),
            "label_authority": {"model_checkpoint_sha256": checkpoint.clone()},
            "candidates": [
                {"model_checkpoint_sha256": checkpoint.clone()},
                {"model_checkpoint_sha256": checkpoint.clone()}
            ]
        });
        assert_eq!(
            manifest_checkpoint_binding(&manifest).as_deref(),
            Some(checkpoint.as_str())
        );
        let mut drifted = manifest;
        drifted["candidates"][1]["model_checkpoint_sha256"] = json!("e".repeat(64));
        assert!(manifest_checkpoint_binding(&drifted).is_none());
    }

    #[test]
    fn blocker_lookup_scans_desired_and_probes_pair_primary_key() {
        let connection = Connection::open_in_memory().unwrap();
        connection
            .execute_batch(
                "CREATE TABLE pairs(
                   task TEXT NOT NULL,dataset_hash TEXT NOT NULL,
                   semantic_sha256 TEXT NOT NULL,protocol_sha256 TEXT NOT NULL,
                   state TEXT NOT NULL,manifest_path TEXT,manifest_sha256 TEXT,
                   job_id TEXT,receipt_sha256 TEXT,
                   PRIMARY KEY(task,dataset_hash,semantic_sha256,protocol_sha256)
                 ) WITHOUT ROWID;
                 CREATE TEMP TABLE desired_pairs(
                   task TEXT NOT NULL,dataset_hash TEXT NOT NULL,
                   semantic_sha256 TEXT NOT NULL,protocol_sha256 TEXT NOT NULL,
                   PRIMARY KEY(task,dataset_hash,semantic_sha256,protocol_sha256)
                 ) WITHOUT ROWID;",
            )
            .unwrap();
        let mut statement = connection
            .prepare(&format!("EXPLAIN QUERY PLAN {DESIRED_BLOCKER_QUERY}"))
            .unwrap();
        let plan = statement
            .query_map([], |row| row.get::<_, String>(3))
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap()
            .join("\n");
        assert!(plan.contains("SCAN d"));
        assert!(plan.contains("SEARCH p USING PRIMARY KEY"));
    }

    #[test]
    fn missing_receipt_recovery_requires_every_manifest_pair_to_be_active_and_unscored() {
        let binding = |suffix: &str,
                       state: &str,
                       path: &str,
                       job: &str,
                       receipt: Option<&str>,
                       failure: Option<&str>| ManifestPairBinding {
            identity: RegistryPairIdentity {
                task: "regression".to_owned(),
                dataset_hash: format!("dataset-{suffix}"),
                semantic_sha256: format!("semantic-{suffix}"),
                protocol_sha256: "protocol".to_owned(),
            },
            state: state.to_owned(),
            manifest_path: Some(path.to_owned()),
            job_id: Some(job.to_owned()),
            receipt_sha256: receipt.map(str::to_owned),
            failure_class: failure.map(str::to_owned),
        };
        let pairs = validate_missing_receipt_manifest_bindings(
            vec![
                binding("a", "reserved", "/manifest", "job", None, None),
                binding("b", "submitted", "/manifest", "job", None, None),
                binding("c", "held", "/manifest", "job", None, None),
            ],
            "/manifest",
            "job",
        )
        .unwrap();
        assert_eq!(pairs.len(), 3);

        for invalid in [
            binding("a", "available", "/manifest", "job", None, None),
            binding("a", "submitted", "/manifest", "job", Some("receipt"), None),
            binding(
                "a",
                "submitted",
                "/manifest",
                "job",
                None,
                Some("semantic_failure"),
            ),
            binding("a", "submitted", "/other", "job", None, None),
            binding("a", "submitted", "/manifest", "other", None, None),
        ] {
            assert!(
                validate_missing_receipt_manifest_bindings(vec![invalid], "/manifest", "job")
                    .is_none()
            );
        }
        assert!(
            validate_missing_receipt_manifest_bindings(Vec::new(), "/manifest", "job").is_none()
        );
    }

    #[test]
    fn seed_bank_requires_repeats_and_ranks_within_task_dataset() {
        let directory = tempfile::tempdir().unwrap();
        let database = directory.path().join("corpus.sqlite");
        let connection = Connection::open(&database).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE observations(
                   task TEXT NOT NULL,dataset_hash TEXT NOT NULL,
                   semantic_sha256 TEXT NOT NULL,canonical_genome TEXT NOT NULL,
                   candidate_source TEXT NOT NULL,score REAL NOT NULL
                 );",
            )
            .unwrap();
        let strong = "(pipe X (ridge))";
        let weak = "(pipe X (standardize) (ridge))";
        let single = "(pipe X (rank) (ridge))";
        for (dataset, strong_score, weak_score) in
            [("dataset-a", 0.9, 0.2), ("dataset-b", 0.8, 0.1)]
        {
            for (genome, score) in [(strong, strong_score), (weak, weak_score)] {
                connection
                    .execute(
                        "INSERT INTO observations VALUES(?1,?2,?3,?4,'gp',?5)",
                        rusqlite::params![
                            "regression",
                            dataset,
                            sha256_bytes(genome.as_bytes()),
                            genome,
                            score
                        ],
                    )
                    .unwrap();
            }
        }
        connection
            .execute(
                "INSERT INTO observations VALUES('regression','dataset-a',?1,?2,'gp',1.0)",
                rusqlite::params![sha256_bytes(single.as_bytes()), single],
            )
            .unwrap();
        drop(connection);

        let bank = export_seed_bank(&database, 10, 2).unwrap();
        let rows = bank["tasks"]["regression"].as_array().unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["canonical_genome"], strong);
        assert_eq!(rows[0]["dataset_count"], 2);
        assert!(rows
            .iter()
            .all(|row| row["canonical_genome"].as_str() != Some(single)));
    }

    fn seed(semantic: &str, mean: f64, dl_origin: bool) -> SeedRow {
        SeedRow {
            task: "regression".to_owned(),
            semantic_sha256: semantic.to_owned(),
            canonical_genome: semantic.to_owned(),
            dataset_count: 2,
            mean_within_dataset_percentile: mean,
            worst_within_dataset_percentile: mean,
            strict_win_count: 0,
            dl_origin,
        }
    }

    #[test]
    fn merge_seed_pools_caps_dl_and_keeps_rank_order() {
        let non_dl = vec![
            seed("a", 0.9, false),
            seed("b", 0.5, false),
            seed("c", 0.3, false),
        ];
        // "b" also appears in the DL pool: it must count as non-DL and never
        // consume DL cap. Cap floor(4/3)=1 admits only the best novel DL seed.
        let dl = vec![
            seed("b", 0.95, true),
            seed("d", 0.8, true),
            seed("e", 0.7, true),
        ];
        let merged = merge_seed_pools(non_dl, dl, 4, 4 / 3);
        let order: Vec<(&str, bool)> = merged
            .iter()
            .map(|row| (row.semantic_sha256.as_str(), row.dl_origin))
            .collect();
        assert_eq!(
            order,
            vec![("a", false), ("d", true), ("b", false), ("c", false)]
        );
        // Overall limit still binds after the cap.
        let merged = merge_seed_pools(
            vec![seed("a", 0.9, false), seed("b", 0.5, false)],
            vec![seed("d", 0.8, true)],
            2,
            1,
        );
        let order: Vec<&str> = merged
            .iter()
            .map(|row| row.semantic_sha256.as_str())
            .collect();
        assert_eq!(order, vec!["a", "d"]);
        // Zero cap excludes DL entirely.
        assert!(merge_seed_pools(Vec::new(), vec![seed("d", 0.8, true)], 4, 0).is_empty());
    }

    #[test]
    fn seed_bank_admits_dl_on_merit_capped_per_task() {
        let directory = tempfile::tempdir().unwrap();
        let database = directory.path().join("corpus.sqlite");
        let connection = Connection::open(&database).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE observations(
                   task TEXT NOT NULL,dataset_hash TEXT NOT NULL,
                   semantic_sha256 TEXT NOT NULL,canonical_genome TEXT NOT NULL,
                   candidate_source TEXT NOT NULL,score REAL NOT NULL
                 );",
            )
            .unwrap();
        let strong = "(pipe X (ridge))";
        let weak = "(pipe X (standardize) (ridge))";
        let dl_top = "(pipe X (rank) (ridge))";
        let dl_second = "(pipe X (winsor) (ridge))";
        for dataset in ["dataset-a", "dataset-b"] {
            for (genome, source, score) in [
                (strong, "gp", 0.9),
                (weak, "gp", 0.2),
                (dl_top, "accepted_dl_checkpoint", 0.95),
                (dl_second, "accepted_dl_checkpoint", 0.5),
            ] {
                connection
                    .execute(
                        "INSERT INTO observations VALUES(?1,?2,?3,?4,?5,?6)",
                        rusqlite::params![
                            "regression",
                            dataset,
                            sha256_bytes(genome.as_bytes()),
                            genome,
                            source,
                            score
                        ],
                    )
                    .unwrap();
            }
        }
        drop(connection);

        // limit=3 -> dl_cap=1: dl_top graduates on merit, dl_second is capped
        // out, and non-DL fills the remaining slots.
        let bank = export_seed_bank(&database, 3, 2).unwrap();
        let rows = bank["tasks"]["regression"].as_array().unwrap();
        assert_eq!(rows.len(), 3);
        let dl_rows: Vec<&Value> = rows.iter().filter(|row| row["dl_origin"] == true).collect();
        assert_eq!(dl_rows.len(), 1);
        assert_eq!(dl_rows[0]["canonical_genome"], dl_top);
        assert!(rows
            .iter()
            .all(|row| row["canonical_genome"].as_str() != Some(dl_second)));
        assert!(rows.iter().any(
            |row| row["canonical_genome"].as_str() == Some(strong) && row["dl_origin"] == false
        ));
        assert_eq!(bank["selection"]["dl_origin_cap_per_task"], 1);
        assert_eq!(
            bank["dl_origin_counts"]["regression"],
            json!({"total": 3, "dl_origin": 1, "non_dl": 2})
        );
    }

    #[test]
    fn mutation_client_reuses_one_newline_json_socket() {
        let directory = tempfile::tempdir().unwrap();
        let socket = directory.path().join("scqd.sock");
        let listener = UnixListener::bind(&socket).unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            for (operation, expected) in
                [("Hold", "job-a"), ("Cancel", "job-b"), ("Release", "job-c")]
            {
                let mut request = String::new();
                reader.read_line(&mut request).unwrap();
                let request: Value = serde_json::from_str(request.trim()).unwrap();
                assert_eq!(request[operation]["id"], expected);
                stream.write_all(b"{\"Ok\":null}\n").unwrap();
                stream.flush().unwrap();
            }
        });
        let mut client = ScqHoldClient::connect(&socket).unwrap();
        client.hold("job-a").unwrap();
        client.cancel("job-b").unwrap();
        client.release("job-c").unwrap();
        server.join().unwrap();
    }

    #[test]
    fn mutation_reconnects_when_server_closes_between_requests() {
        let directory = tempfile::tempdir().unwrap();
        let socket = directory.path().join("scqd.sock");
        let listener = UnixListener::bind(&socket).unwrap();
        let server = std::thread::spawn(move || {
            for expected in ["job-a", "job-b"] {
                let (mut stream, _) = listener.accept().unwrap();
                let mut request = String::new();
                BufReader::new(stream.try_clone().unwrap())
                    .read_line(&mut request)
                    .unwrap();
                let request: Value = serde_json::from_str(request.trim()).unwrap();
                assert_eq!(request["Cancel"]["id"], expected);
                stream.write_all(b"{\"Ok\":null}\n").unwrap();
            }
        });
        let mut client = ScqHoldClient::connect(&socket);
        assert!(socket_mutation_with_reconnect(&mut client, &socket, "Cancel", "job-a").is_none());
        assert!(socket_mutation_with_reconnect(&mut client, &socket, "Cancel", "job-b").is_none());
        server.join().unwrap();
    }

    #[test]
    fn retirement_transaction_preserves_history_and_is_idempotent() {
        let directory = tempfile::tempdir().unwrap();
        let pair_path = directory.path().join("pairs.sqlite");
        let backlog_path = directory.path().join("backlog.sqlite");
        let manifest_hash = "a".repeat(64);
        let preview_hash = "b".repeat(64);
        let job_id = "job-held";
        let pair = Connection::open(&pair_path).unwrap();
        pair.execute_batch(
            "CREATE TABLE pairs(
               task TEXT,dataset_hash TEXT,semantic_sha256 TEXT,protocol_sha256 TEXT,
               state TEXT,reservation_owner TEXT,manifest_path TEXT,
               manifest_sha256 TEXT,job_id TEXT,receipt_sha256 TEXT,
               failure_class TEXT,updated_at_ms INTEGER
             );
             CREATE TABLE events(
               id INTEGER PRIMARY KEY,kind TEXT,task TEXT,dataset_hash TEXT,
               semantic_sha256 TEXT,protocol_sha256 TEXT,detail_json TEXT,
               created_at_ms INTEGER
             );
             CREATE INDEX pair_manifest ON pairs(manifest_sha256);",
        )
        .unwrap();
        pair.execute(
            "INSERT INTO pairs VALUES(
               'regression','dataset',?1,?2,'reserved','owner','/manifest',?3,
               NULL,NULL,NULL,1
             )",
            params!["c".repeat(64), "d".repeat(64), manifest_hash],
        )
        .unwrap();
        drop(pair);
        let backlog = Connection::open(&backlog_path).unwrap();
        backlog
            .execute_batch(
                "CREATE TABLE claims(
                   manifest_sha256 TEXT,manifest TEXT,job_id TEXT,state TEXT,
                   updated_at_ms INTEGER
                 );",
            )
            .unwrap();
        backlog
            .execute(
                "INSERT INTO claims VALUES(?1,'/manifest',?2,'submitted',1)",
                params![manifest_hash, job_id],
            )
            .unwrap();
        drop(backlog);
        let candidate = HoldCandidate {
            job_id: job_id.to_owned(),
            project: LEGACY_PROJECT.to_owned(),
            observed_state: "cancelled".to_owned(),
            epoch: LEGACY_EPOCH.to_owned(),
            strict_wave: None,
            generation: None,
            manifest: "/manifest".to_owned(),
            manifest_sha256: manifest_hash.clone(),
        };
        let mut results = vec![RetirementResult {
            job_id: job_id.to_owned(),
            state_before_cancel: Some("cancelled".to_owned()),
            cancel_requested: false,
            command_error: None,
            state_after_cancel: Some("cancelled".to_owned()),
            pair_updates: 0,
            claim_updated: false,
            recovery_from_prior_preview: true,
            registry_error: None,
        }];
        apply_retirement_registry(
            &pair_path,
            &backlog_path,
            std::slice::from_ref(&candidate),
            &mut results,
            &preview_hash,
        )
        .unwrap();
        assert!(results[0].claim_updated);
        let pair = Connection::open(&pair_path).unwrap();
        let (state, retained_manifest): (String, Option<String>) = pair
            .query_row("SELECT state,manifest_sha256 FROM pairs", [], |row| {
                Ok((row.get(0)?, row.get(1)?))
            })
            .unwrap();
        assert_eq!(state, "available");
        assert!(retained_manifest.is_none());
        let events: i64 = pair
            .query_row("SELECT COUNT(*) FROM events", [], |row| row.get(0))
            .unwrap();
        assert_eq!(events, 1);
        let backlog = Connection::open(&backlog_path).unwrap();
        let state: String = backlog
            .query_row("SELECT state FROM claims", [], |row| row.get(0))
            .unwrap();
        assert_eq!(state, "superseded_unstarted");

        let mut rerun = vec![RetirementResult {
            recovery_from_prior_preview: true,
            state_after_cancel: Some("cancelled".to_owned()),
            command_error: None,
            cancel_requested: false,
            job_id: job_id.to_owned(),
            state_before_cancel: Some("cancelled".to_owned()),
            pair_updates: 0,
            claim_updated: false,
            registry_error: None,
        }];
        apply_retirement_registry(
            &pair_path,
            &backlog_path,
            &[candidate],
            &mut rerun,
            &preview_hash,
        )
        .unwrap();
        assert!(!rerun[0].claim_updated);

        let pair = Connection::open(&pair_path).unwrap();
        pair.execute(
            "UPDATE pairs SET state='submitted',reservation_owner='owner',
               manifest_sha256=?1,receipt_sha256=?2",
            params![manifest_hash, "e".repeat(64)],
        )
        .unwrap();
        let backlog = Connection::open(&backlog_path).unwrap();
        backlog
            .execute(
                "UPDATE claims SET state='submitted' WHERE job_id=?1",
                [job_id],
            )
            .unwrap();
        drop(pair);
        drop(backlog);
        let mut rejected = vec![RetirementResult {
            recovery_from_prior_preview: true,
            state_after_cancel: Some("cancelled".to_owned()),
            command_error: None,
            cancel_requested: false,
            job_id: job_id.to_owned(),
            state_before_cancel: Some("cancelled".to_owned()),
            pair_updates: 0,
            claim_updated: false,
            registry_error: None,
        }];
        let receipt_candidate = HoldCandidate {
            job_id: job_id.to_owned(),
            project: LEGACY_PROJECT.to_owned(),
            observed_state: "cancelled".to_owned(),
            epoch: LEGACY_EPOCH.to_owned(),
            strict_wave: None,
            generation: None,
            manifest: "/manifest".to_owned(),
            manifest_sha256: manifest_hash,
        };
        assert!(apply_retirement_registry(
            &pair_path,
            &backlog_path,
            &[receipt_candidate],
            &mut rejected,
            &preview_hash,
        )
        .is_err());
        let backlog = Connection::open(&backlog_path).unwrap();
        let state: String = backlog
            .query_row("SELECT state FROM claims", [], |row| row.get(0))
            .unwrap();
        assert_eq!(state, "submitted");
    }
}
