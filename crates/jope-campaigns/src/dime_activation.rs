//! DIME-only runtime activation and rollback transaction.
//!
//! The transaction snapshots and mutates only the runtime pointer, active
//! document, DIME user units, their drop-ins, and their service state. Science
//! pointers are observed before every step and are never restoration inputs.

use crate::recovery_contracts::{JsonEvidenceRefV2, RawEvidenceRefV2, REQUIRED_NODES};
use crate::selfplay_runtime::sha256_bytes;
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::os::unix::fs::{symlink, OpenOptionsExt, PermissionsExt};
use std::path::{Component, Path, PathBuf};

mod host;
pub use host::{ActivationLock, SystemdUserHost};
mod hash;
use hash::sha256_file;

mod evidence;
use evidence::*;
mod types;
pub use types::*;
mod unit_ops;
use unit_ops::*;

pub fn execute_activation_transaction(
    request: &ActivationRequest<'_>,
    host: &mut dyn ActivationHost,
    stop_after: Option<ActivationStageV1>,
) -> Result<DimeActivationTransactionV1> {
    require_sha256(request.candidate_bundle_sha256, "activation candidate")?;
    require_sha256(
        request.rollback_capsule_sha256,
        "activation rollback capsule",
    )?;
    if let Some(digest) = request.external_gate_sha256 {
        require_sha256(digest, "activation external gate")?;
    }
    if let Some(digest) = request.candidate_review_gate_sha256 {
        require_sha256(digest, "activation candidate review gate")?;
    }
    let candidate_root = request.output_root.join(request.candidate_bundle_sha256);
    ensure!(
        candidate_root.join("bundle.json").is_file(),
        "activation candidate is absent"
    );
    fs::create_dir_all(request.transaction_root)?;
    let transaction_path = request.transaction_root.join("transaction.v1.json");
    let mut transaction = if transaction_path.is_file() {
        let value: DimeActivationTransactionV1 =
            serde_json::from_slice(&fs::read(&transaction_path)?)?;
        ensure!(
            value.stage != ActivationStageV1::Aborted
                && value.candidate_bundle_sha256 == request.candidate_bundle_sha256
                && value.rollback_capsule_sha256 == request.rollback_capsule_sha256
                && value.candidate_review_gate_sha256
                    == request.candidate_review_gate_sha256.map(str::to_owned)
                && value.external_gate_sha256 == request.external_gate_sha256.map(str::to_owned)
                && value.active_document == request.active_document
                && value.operation == request.operation,
            "an incompatible or aborted activation transaction already exists"
        );
        if value.stage == ActivationStageV1::Complete {
            return Ok(value);
        }
        value
    } else {
        prepare_transaction(request, host, &candidate_root)?
    };
    write_transaction(&transaction_path, &transaction)?;
    if stop_after == Some(transaction.stage) {
        return Ok(transaction);
    }

    loop {
        match transaction.stage {
            ActivationStageV1::Prepared => {
                stage_units(
                    request.transaction_root,
                    &candidate_root,
                    &transaction.candidate_units,
                )?;
                transaction.stage = ActivationStageV1::UnitsStaged;
            }
            ActivationStageV1::UnitsStaged => {
                validate_candidate(&candidate_root, &transaction.candidate_units)?;
                let names = transaction
                    .candidate_units
                    .iter()
                    .map(|row| row.unit.clone())
                    .collect::<Vec<_>>();
                host.validate_candidate(&request.transaction_root.join("staged"), &names)?;
                transaction.stage = ActivationStageV1::CandidateValidated;
            }
            ActivationStageV1::CandidateValidated => {
                install_units(
                    request.user_unit_root,
                    request.transaction_root,
                    &transaction.preimage.units,
                    &transaction.candidate_units,
                )?;
                transaction.stage = ActivationStageV1::UnitsInstalled;
            }
            ActivationStageV1::UnitsInstalled => {
                switch_current(request.output_root, request.candidate_bundle_sha256)?;
                transaction.stage = ActivationStageV1::RuntimeSwitched;
            }
            ActivationStageV1::RuntimeSwitched => {
                atomic_json(
                    &request.output_root.join("active.json"),
                    &transaction.active_document,
                )?;
                transaction.stage = ActivationStageV1::ActivePublished;
            }
            ActivationStageV1::ActivePublished => {
                host.daemon_reload()?;
                transaction.stage = ActivationStageV1::DaemonReloaded;
            }
            ActivationStageV1::DaemonReloaded => {
                host.enable_timers(&transaction.candidate_timers)?;
                transaction.stage = ActivationStageV1::TimersEnabled;
            }
            ActivationStageV1::TimersEnabled => {
                host.try_restart_services(&transaction.affected_services)?;
                transaction.stage = ActivationStageV1::ServicesRestarted;
            }
            ActivationStageV1::ServicesRestarted => {
                transaction.stage = ActivationStageV1::Complete;
            }
            ActivationStageV1::Complete | ActivationStageV1::Aborted => return Ok(transaction),
        }
        write_transaction(&transaction_path, &transaction)?;
        if stop_after == Some(transaction.stage) {
            return Ok(transaction);
        }
    }
}

pub fn abort_activation_transaction(
    output_root: &Path,
    user_unit_root: &Path,
    transaction_root: &Path,
    host: &mut dyn ActivationHost,
) -> Result<DimeActivationTransactionV1> {
    let path = transaction_root.join("transaction.v1.json");
    let mut transaction: DimeActivationTransactionV1 = serde_json::from_slice(&fs::read(&path)?)?;
    ensure!(
        transaction.stage != ActivationStageV1::Complete
            && transaction.stage != ActivationStageV1::Aborted,
        "activation transaction is already terminal"
    );
    restore_units(
        user_unit_root,
        transaction_root,
        &transaction.preimage.units,
    )?;
    restore_raw_optional(
        &output_root.join("active.json"),
        transaction_root,
        transaction.preimage.active_json.as_ref(),
    )?;
    restore_current(
        output_root,
        transaction.preimage.current_symlink_target.as_deref(),
    )?;
    host.daemon_reload()?;
    host.restore_states(&transaction.preimage.units)?;
    transaction.stage = ActivationStageV1::Aborted;
    write_transaction(&path, &transaction)?;
    Ok(transaction)
}

fn prepare_transaction(
    request: &ActivationRequest<'_>,
    host: &mut dyn ActivationHost,
    candidate_root: &Path,
) -> Result<DimeActivationTransactionV1> {
    let candidate_units = candidate_units(candidate_root)?;
    let current_units = current_unit_names(request.output_root)?;
    let all_units = candidate_units
        .iter()
        .map(|row| row.unit.clone())
        .chain(current_units)
        .collect::<BTreeSet<_>>();
    let mut preimages = Vec::new();
    for unit in all_units {
        validate_unit_name(&unit)?;
        let base_path = request.user_unit_root.join(&unit);
        let base_file = if base_path.is_file() {
            Some(store_raw(
                request.transaction_root,
                &format!("preimages/units/{unit}"),
                &fs::read(&base_path)?,
            )?)
        } else {
            None
        };
        let drop_in_root = request.user_unit_root.join(format!("{unit}.d"));
        let mut drop_ins = Vec::new();
        if drop_in_root.is_dir() {
            let mut paths = collect_files(&drop_in_root)?;
            paths.sort();
            for path in paths {
                let relative = path
                    .strip_prefix(&drop_in_root)?
                    .to_string_lossy()
                    .into_owned();
                validate_relative(&relative, "drop-in preimage")?;
                drop_ins.push(ActivationDropInPreimageV1 {
                    relative_path: relative.clone(),
                    file: store_raw(
                        request.transaction_root,
                        &format!("preimages/drop-ins/{unit}/{relative}"),
                        &fs::read(path)?,
                    )?,
                });
            }
        }
        preimages.push(ActivationUnitPreimageV1 {
            unit: unit.clone(),
            base_file,
            drop_ins,
            enabled: host.is_enabled(&unit)?,
            active: host.is_active(&unit)?,
        });
    }
    let active_json = if request.output_root.join("active.json").is_file() {
        Some(store_raw(
            request.transaction_root,
            "preimages/active.json",
            &fs::read(request.output_root.join("active.json"))?,
        )?)
    } else {
        None
    };
    let current = request.output_root.join("current");
    let current_symlink_target = if current.is_symlink() {
        Some(fs::read_link(&current)?.to_string_lossy().into_owned())
    } else {
        ensure!(
            !current.exists(),
            "runtime current is not an atomic symlink"
        );
        None
    };
    let recovery_tool_path = current.join("runtime/bin/jope_selfplay_bundle");
    ensure!(
        recovery_tool_path.is_file(),
        "current runtime lacks its recovery tool"
    );
    let recovery_tool = store_raw(
        request.transaction_root,
        "recovery-tool/jope_selfplay_bundle",
        &fs::read(recovery_tool_path)?,
    )?;
    fs::set_permissions(
        request.transaction_root.join(&recovery_tool.path),
        fs::Permissions::from_mode(0o700),
    )?;
    let affected_services = preimages
        .iter()
        .filter(|preimage| preimage.unit.ends_with(".service"))
        .filter(|preimage| {
            let candidate = candidate_units.iter().find(|row| row.unit == preimage.unit);
            candidate.map(|row| row.sha256.as_str())
                != preimage
                    .base_file
                    .as_ref()
                    .map(|file| file.raw_file_sha256.as_str())
                || !preimage.drop_ins.is_empty()
        })
        .map(|preimage| preimage.unit.clone())
        .collect::<Vec<_>>();
    let candidate_timers = active_timer_units(candidate_root, &candidate_units)?;
    let transaction_id = sha256_bytes(
        format!(
            "{}\0{}\0{:?}",
            request.candidate_bundle_sha256, request.rollback_capsule_sha256, request.operation
        )
        .as_bytes(),
    );
    Ok(DimeActivationTransactionV1 {
        schema: ACTIVATION_TRANSACTION_SCHEMA.to_owned(),
        transaction_id,
        operation: request.operation,
        stage: ActivationStageV1::Prepared,
        candidate_bundle_sha256: request.candidate_bundle_sha256.to_owned(),
        rollback_capsule_sha256: request.rollback_capsule_sha256.to_owned(),
        candidate_review_gate_sha256: request.candidate_review_gate_sha256.map(str::to_owned),
        external_gate_sha256: request.external_gate_sha256.map(str::to_owned),
        active_document: request.active_document.clone(),
        preimage: ActivationPreimageV1 {
            active_json,
            current_symlink_target,
            units: preimages,
            recovery_tool,
        },
        candidate_units,
        candidate_timers,
        affected_services,
        protected_science_sha256: snapshot_protected_science(
            request
                .output_root
                .parent()
                .context("runtime root has no state parent")?,
        )?,
    })
}

#[cfg(test)]
mod tests;
