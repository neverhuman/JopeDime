//! Proactive Rust-only supply supervisor for immutable self-play micro-waves.
//!
//! This process never submits, holds, releases, or cancels SCQ jobs. It keeps a
//! bounded reserve of current-checkpoint manifests ahead of the queue keeper by
//! invoking only the Rust `jope_selfplay_wave` planner/sanitizer/builder.

use anyhow::{bail, ensure, Context, Result};
use clap::Parser;
use jope_dime_regressor::dime_runtime_contracts::{
    append_infrastructure_attempt, digest_label, load_or_rebuild_recovery_projection,
    matching_drain_intent, save_recovery_projection, terminal_classification,
    DimeInfrastructureAttemptV1, DimeNamedDigestV1, DimeRecoveryProjectionV1, DimeRoleV1,
    DimeScientificPayloadV1, DimeTerminalClassV1, DimeTerminalEvidenceV1,
};
use jope_dime_regressor::paths;
use jope_dime_regressor::selfplay_runtime::{
    atomic_json, checkpoint_epoch, default_root_str, default_scq_str, now_ms, sha256_bytes,
    sha256_file,
};
use rusqlite::{Connection, OpenFlags};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::Duration;

#[path = "common/flock.rs"]
mod flock;

fn reconcile_proposer_projection(
    root: &Path,
    state: &mut PersistedSupplyState,
    projection: &DimeRecoveryProjectionV1,
) -> Result<()> {
    let desired_attempt = u32::try_from(projection.current_attempt_number.saturating_sub(1))
        .context("proposer recovery attempt overflows primary schema")?;
    if state.proposal_attempt == desired_attempt
        && state.proposal_job_id.as_deref() == Some(projection.current_job_identity.as_str())
    {
        return Ok(());
    }
    ensure!(
        desired_attempt == state.proposal_attempt.saturating_add(1),
        "proposer recovery projection is not the next rollback-compatible attempt"
    );
    let prior = state
        .runtime_pinned_attempts
        .last()
        .cloned()
        .context("proposer recovery lacks a pinned prior intent")?;
    let job_identity = projection.current_job_identity.clone();
    let prior_body = fs::read_to_string(&prior.immutable_spec_path)?;
    let replacement = format!("id = {}", toml_quote(&job_identity)?);
    let mut replaced = false;
    let body = prior_body
        .lines()
        .map(|line| {
            if !replaced && line.starts_with("id = ") {
                replaced = true;
                replacement.as_str()
            } else {
                line
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    if !replaced {
        bail!("pinned cohort proposer spec lacks a job identity");
    }
    let directory = root.join("specs/cohort-propose");
    fs::create_dir_all(&directory)?;
    let spec = directory.join(format!("{job_identity}.toml"));
    if spec.is_file() {
        if fs::read_to_string(&spec)? != body {
            bail!("immutable cohort proposer retry spec drift");
        }
    } else {
        fs::write(&spec, &body)?;
    }
    let pinned = runtime_pinned_intent_from_spec(&spec, desired_attempt)?;
    if pinned.runtime_bundle_sha256 != prior.runtime_bundle_sha256
        || pinned.catalog_snapshot_sha256 != prior.catalog_snapshot_sha256
        || pinned.proposer_inputs != prior.proposer_inputs
    {
        bail!("cohort proposer retry changed its pinned bundle or inputs");
    }
    state.runtime_pinned_attempts.push(pinned);
    state.proposal_attempt = desired_attempt;
    state.proposal_job_id = Some(job_identity);
    Ok(())
}

fn proposer_recovery_projection(
    root: &Path,
    state: &PersistedSupplyState,
    payload: &DimeScientificPayloadV1,
) -> Result<DimeRecoveryProjectionV1> {
    let initial_job = state
        .runtime_pinned_attempts
        .first()
        .map(|intent| intent.job_identity.as_str())
        .context("proposer recovery has no initial pinned identity")?;
    load_or_rebuild_recovery_projection(root, DimeRoleV1::CohortProposer, payload, initial_job)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProposerRetryDisposition {
    Retried,
    Backoff,
    Drained,
    ScientificCircuitOpen,
}

fn proposer_scientific_payload(
    state: &PersistedSupplyState,
    pinned: &RuntimePinnedIntentV1,
) -> Result<DimeScientificPayloadV1> {
    pinned.validate()?;
    let parent_intent_sha256 = sha256_bytes(&serde_json::to_vec(&json!({
        "domain": "dime-cohort-proposer-parent-intent-v1",
        "wave": state.wave,
        "checkpoint_sha256": state.checkpoint_sha256,
        "cohort_sha256": state.cohort_sha256,
        "catalog_sha256": state.catalog_sha256,
        "decoder_configuration_sha256": state.decoder_configuration_sha256,
        "seed": state.seed,
        "proposal_reservoir_per_context": state.proposal_reservoir_per_context,
    }))?);
    let binary_sha256 = pinned
        .proposer_inputs
        .iter()
        .find(|input| input.name == "generator")
        .map(|input| input.sha256.clone())
        .context("pinned proposer has no generator digest")?;
    let mut inputs = pinned
        .proposer_inputs
        .iter()
        .map(|input| DimeNamedDigestV1 {
            name: input.name.clone(),
            sha256: input.sha256.clone(),
        })
        .collect::<Vec<_>>();
    inputs.sort_by(|left, right| left.name.cmp(&right.name));
    let payload = DimeScientificPayloadV1 {
        parent_intent_sha256,
        runtime_bundle_sha256: pinned.runtime_bundle_sha256.clone(),
        binary_sha256,
        configuration_sha256: state.decoder_configuration_sha256.clone(),
        train_data_sha256: state.cohort_sha256.clone(),
        protocol_sha256: digest_label("jope-cohort-proposer-protocol-v1"),
        metric_sha256: digest_label("jope-proposal-validity-metric-v1"),
        seed: state.seed,
        inputs,
    };
    payload.validate()?;
    Ok(payload)
}

fn scientific_failure_receipt(
    root: &Path,
    job_identity: &str,
    payload: &DimeScientificPayloadV1,
) -> Result<Option<String>> {
    let path = root
        .join("ops/dime-scientific-failures")
        .join(format!("{job_identity}.json"));
    if !path.is_file() {
        return Ok(None);
    }
    let value: Value = serde_json::from_slice(&fs::read(&path)?)?;
    let evidence_sha256 = value
        .get("evidence_sha256")
        .and_then(Value::as_str)
        .context("scientific failure receipt lacks evidence")?;
    validate_sha256(evidence_sha256)?;
    ensure!(
        value.get("schema").and_then(Value::as_str) == Some("DimeScientificFailure.v1")
            && value.get("job_identity").and_then(Value::as_str) == Some(job_identity)
            && value
                .get("scientific_payload_sha256")
                .and_then(Value::as_str)
                == Some(payload.identity_sha256()?.as_str())
            && value.get("deterministic").and_then(Value::as_bool) == Some(true),
        "scientific failure receipt is not hash-bound to this proposer intent"
    );
    Ok(Some(evidence_sha256.to_owned()))
}

fn append_supply_event(root: &Path, kind: &str, detail: &Value) -> Result<()> {
    append_event(
        &root.join("ops/micro-wave-supply-events.v1.jsonl"),
        &json!({
            "schema": "JopeMicroWaveSupplyEvent.v1",
            "generated_at_ms": now_ms(),
            "kind": kind,
            "detail": detail,
        }),
    )
}

mod options;
use options::*;
mod persisted;
use persisted::*;
mod constants;
use constants::*;
mod lock;
use lock::*;
mod proposer_context;
mod proposer_input;
mod runtime;
mod supply;
mod wave_builder;
use proposer_context::*;
use proposer_input::*;
use runtime::*;
use supply::*;
use wave_builder::*;

#[cfg(test)]
mod tests;

fn main() -> Result<()> {
    let args = Args::parse();
    validate_args(&args)?;
    let _lock = SingleInstanceLock::acquire(&args.root.join("ops/micro-wave-supply-builder.lock"))?;
    let circuit_path = args.root.join("ops/cohort-supply-circuit-breaker.v1.json");
    let (mut previous_error, mut identical_failures) =
        load_supply_failure_streak(&circuit_path).unwrap_or_default();
    loop {
        let mut retry_after_seconds = args.interval_seconds;
        match cycle(&args) {
            Ok(report) => {
                let value = serde_json::to_value(&report)?;
                atomic_json(&args.root.join("ops/micro-wave-supply-v2.json"), &value)?;
                previous_error = None;
                identical_failures = 0;
                atomic_json(
                    &circuit_path,
                    &json!({
                        "schema": "SupplyCircuitBreaker.v1",
                        "generated_at_ms": now_ms(),
                        "open": false,
                        "identical_failures": 0,
                        "critical": false,
                        "retry_after_seconds": args.interval_seconds,
                        "recoverable_state": args.root.join("ops/cohort-supply-state-v2"),
                    }),
                )?;
                println!("{}", serde_json::to_string(&report)?);
            }
            Err(error) => {
                let error_text = format!("{error:#}");
                if previous_error.as_deref() == Some(error_text.as_str()) {
                    identical_failures = identical_failures.saturating_add(1);
                } else {
                    previous_error = Some(error_text.clone());
                    identical_failures = 1;
                }
                let exponent = identical_failures.saturating_sub(1).min(31);
                retry_after_seconds = args
                    .interval_seconds
                    .saturating_mul(1_u64 << exponent)
                    .min(ERROR_BACKOFF_MAX_SECONDS);
                let failure = json!({
                    "schema": "JopeMicroWaveSupplyError.v1",
                    "generated_at_ms": now_ms(),
                    "mode": if args.build { "build" } else { "audit" },
                    "error": error_text,
                    "builds_completed": 0,
                    "identical_failures": identical_failures,
                    "critical": identical_failures >= 3,
                    "retry_after_seconds": retry_after_seconds,
                });
                atomic_json(&args.root.join("ops/micro-wave-supply-v2.json"), &failure)?;
                atomic_json(
                    &circuit_path,
                    &json!({
                        "schema": "SupplyCircuitBreaker.v1",
                        "generated_at_ms": now_ms(),
                        "open": true,
                        "error": previous_error,
                        "identical_failures": identical_failures,
                        "critical": identical_failures >= 3,
                        "retry_after_seconds": retry_after_seconds,
                        "recoverable_state": args.root.join("ops/cohort-supply-state-v2"),
                    }),
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
        thread::sleep(Duration::from_secs(retry_after_seconds));
    }
}
