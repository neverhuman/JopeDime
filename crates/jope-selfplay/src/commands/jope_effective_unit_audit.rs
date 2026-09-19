//! Audit effective user-systemd units against the active runtime bundle.

use anyhow::{ensure, Context, Result};
use clap::Parser;
use jope_dime_regressor::paths;
use jope_dime_regressor::selfplay_runtime::{
    atomic_json, default_root_str, now_ms, read_json, sha256_bytes, sha256_file,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const SCHEMA: &str = "JopeEffectiveUnitAudit.v1";
const SERVICE_MANIFEST_SCHEMA: &str = "JopeSelfplayServiceManifest.v3";
const POINTER_SCHEMA: &str = "JopeSelfplayRuntimeBundlePointer.v2";
// Checkout names are intentionally absent: the project boundary forbids
// embedding sibling-project roots, and the exact expected bundle path below
// already rejects every executable outside the active immutable bundle.
const FORBIDDEN_EXECSTART: &[&str] = &["/target/", "/build/"];

#[derive(Parser, Debug)]
#[command(name = "jope-effective-unit-audit")]
struct Args {
    #[arg(long, default_value = default_root_str())]
    root: PathBuf,
    #[arg(long, default_value = "/usr/bin/systemctl")]
    systemctl: PathBuf,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long, default_value_os_t = paths::scq_projects())]
    scq_projects_active: PathBuf,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ServiceManifest {
    schema: String,
    binaries: Vec<String>,
    services: Vec<ServiceRow>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ServiceRow {
    unit: String,
    binary: String,
    owner: String,
    active: bool,
}

#[derive(Clone, Debug, Serialize)]
struct UnitAudit {
    unit: String,
    binary: String,
    owner: String,
    fragment_path: Option<String>,
    drop_in_paths: Vec<String>,
    active_state: Option<String>,
    sub_state: Option<String>,
    effective_exec_start: Option<String>,
    main_pid: Option<u32>,
    process_exe: Option<String>,
    process_sha256: Option<String>,
    expected_binary_sha256: Option<String>,
    systemctl_cat_sha256: String,
    healthy: bool,
    violations: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
struct EffectiveUnitAudit {
    schema: &'static str,
    generated_at_ms: u128,
    root: PathBuf,
    bundle_sha256: String,
    bundle_manifest_path: PathBuf,
    bundle_manifest_sha256: String,
    current_bundle_path: PathBuf,
    synchronization_receipt_path: PathBuf,
    synchronization_receipt_sha256: String,
    rollback_bundle_sha256: String,
    service_manifest_sha256: String,
    scq_projects_active_path: PathBuf,
    scq_projects_active_sha256: Option<String>,
    bundled_scq_projects_sha256: String,
    units: Vec<UnitAudit>,
    violations: Vec<String>,
    healthy: bool,
}

fn property<'a>(document: &'a str, key: &str) -> Option<&'a str> {
    document
        .lines()
        .find_map(|line| line.strip_prefix(&format!("{key}=")))
        .filter(|value| !value.trim().is_empty())
}

fn split_paths(value: Option<&str>) -> Vec<String> {
    value
        .into_iter()
        .flat_map(str::split_whitespace)
        .filter(|path| !path.is_empty())
        .map(str::to_owned)
        .collect()
}

fn drop_in_overrides_execstart(document: &str) -> bool {
    let mut in_drop_in = false;
    for line in document.lines() {
        if let Some(path) = line.strip_prefix("# ") {
            in_drop_in = path.trim().ends_with(".conf");
            continue;
        }
        let line = line.trim_start();
        if in_drop_in
            && (line.starts_with("ExecStart=")
                || line.starts_with("ExecStartPre=")
                || line.starts_with("ExecStartPost="))
        {
            return true;
        }
    }
    false
}

fn inspect_unit(row: &ServiceRow, root: &Path, show: &str, cat: &str) -> UnitAudit {
    let exec = property(show, "ExecStart").map(str::to_owned);
    let drop_in_paths = split_paths(property(show, "DropInPaths"));
    let expected = format!(
        "{}/runtime-bundles/current/runtime/bin/{}",
        root.display(),
        row.binary
    );
    let mut violations = Vec::new();
    match exec.as_deref() {
        Some(value) if value.contains(&expected) => {}
        Some(_) => {
            violations.push("effective ExecStart does not reference the bundled binary".to_owned())
        }
        None => violations.push("systemctl show returned no effective ExecStart".to_owned()),
    }
    if let Some(value) = exec.as_deref() {
        for forbidden in FORBIDDEN_EXECSTART {
            if value.contains(forbidden) {
                violations.push(format!(
                    "effective ExecStart contains forbidden mutable path {forbidden}"
                ));
            }
        }
        if value.contains(".py")
            && !value.contains(&format!(
                "{}/runtime-bundles/current/pytorch/",
                root.display()
            ))
        {
            violations.push("effective ExecStart references unbundled Python".to_owned());
        }
    }
    if !drop_in_paths.is_empty() && drop_in_overrides_execstart(cat) {
        violations.push("unknown ExecStart drop-in override is installed".to_owned());
    }
    if !cat.contains(&expected) {
        violations.push("systemctl cat inventory does not contain the bundled binary".to_owned());
    }
    UnitAudit {
        unit: row.unit.clone(),
        binary: row.binary.clone(),
        owner: row.owner.clone(),
        fragment_path: property(show, "FragmentPath").map(str::to_owned),
        drop_in_paths,
        active_state: property(show, "ActiveState").map(str::to_owned),
        sub_state: property(show, "SubState").map(str::to_owned),
        effective_exec_start: exec,
        main_pid: property(show, "MainPID").and_then(|value| value.parse().ok()),
        process_exe: None,
        process_sha256: None,
        expected_binary_sha256: None,
        systemctl_cat_sha256: sha256_bytes(cat.as_bytes()),
        healthy: violations.is_empty(),
        violations,
    }
}

fn systemctl(systemctl: &Path, action: &str, unit: &str) -> Result<String> {
    let mut command = Command::new(systemctl);
    command
        .arg("--user")
        .arg(action)
        .arg(unit)
        .arg("--no-pager");
    if action == "show" {
        command.arg("--property=FragmentPath,DropInPaths,ExecStart,ActiveState,SubState,MainPID");
    }
    let output = command
        .output()
        .with_context(|| format!("run systemctl {action} {unit}"))?;
    ensure!(
        output.status.success(),
        "systemctl {action} {unit} failed: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    String::from_utf8(output.stdout).context("systemctl output is not UTF-8")
}

fn build(args: &Args) -> Result<EffectiveUnitAudit> {
    let bundles = args.root.join("runtime-bundles");
    let pointer_path = bundles.join("active.json");
    let pointer = read_json(&pointer_path)?;
    ensure!(
        pointer.get("schema").and_then(Value::as_str) == Some(POINTER_SCHEMA),
        "runtime pointer schema drift"
    );
    let bundle_sha256 = pointer
        .get("bundle_sha256")
        .and_then(Value::as_str)
        .context("runtime pointer lacks bundle digest")?
        .to_owned();
    let bundle_path = bundles.join(&bundle_sha256);
    let bundle_manifest_path = bundle_path.join("bundle.json");
    let bundle_manifest_sha256 = sha256_file(&bundle_manifest_path)?;
    ensure!(
        pointer
            .get("bundle_manifest_sha256")
            .and_then(Value::as_str)
            == Some(bundle_manifest_sha256.as_str()),
        "runtime pointer bundle manifest hash drift"
    );
    let current = bundles.join("current");
    let current_bundle_path =
        fs::canonicalize(&current).context("runtime current symlink is broken")?;
    ensure!(
        current_bundle_path == fs::canonicalize(&bundle_path)?,
        "runtime current symlink does not match active pointer"
    );
    let synchronization_receipt_path = pointer
        .get("synchronization_receipt")
        .and_then(Value::as_str)
        .map(PathBuf::from)
        .context("runtime pointer lacks synchronization receipt")?;
    let synchronization_receipt_sha256 = sha256_file(&synchronization_receipt_path)?;
    ensure!(
        pointer
            .get("synchronization_receipt_sha256")
            .and_then(Value::as_str)
            == Some(synchronization_receipt_sha256.as_str()),
        "runtime pointer synchronization receipt hash drift"
    );
    let synchronization = read_json(&synchronization_receipt_path)?;
    let synchronized_nodes = synchronization
        .get("receipts")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|row| row.get("verified").and_then(Value::as_bool) == Some(true))
        .filter_map(|row| row.get("node").and_then(Value::as_str))
        .collect::<BTreeSet<_>>();
    ensure!(
        synchronization.get("schema").and_then(Value::as_str)
            == Some("JopeBundleSynchronizationReceipt.v1")
            && synchronization.get("complete").and_then(Value::as_bool) == Some(true)
            && synchronized_nodes
                == BTreeSet::from(["xbabe0", "xbabe1", "xbabe2", "xbabe3", "xbabe4"]),
        "active bundle does not have five-node synchronization evidence"
    );
    let rollback_bundle_sha256 = pointer
        .get("rollback_bundle_sha256")
        .and_then(Value::as_str)
        .context("runtime pointer lacks rollback bundle identity")?
        .to_owned();
    ensure!(
        rollback_bundle_sha256
            == "9521ffcef11159339f314cb26430739d5472a54e1b521a868bd9ccb08acf6bd4"
            && bundles
                .join(&rollback_bundle_sha256)
                .join("bundle.json")
                .is_file(),
        "verified rollback bundle is absent"
    );
    let manifest = read_json(&bundle_manifest_path)?;
    ensure!(
        manifest.get("schema").and_then(Value::as_str) == Some("RuntimeBundle.v3")
            && manifest.get("bundle_sha256").and_then(Value::as_str)
                == Some(bundle_sha256.as_str()),
        "active bundle manifest identity drift"
    );
    let service_value = manifest
        .get("service_manifest")
        .cloned()
        .context("bundle lacks embedded service manifest")?;
    let service_manifest_sha256 = sha256_bytes(&serde_json::to_vec(&service_value)?);
    let service: ServiceManifest = serde_json::from_value(service_value)?;
    ensure!(
        service.schema == SERVICE_MANIFEST_SCHEMA,
        "service manifest schema drift"
    );
    let binary_inventory = service.binaries.iter().collect::<BTreeSet<_>>();
    ensure!(
        binary_inventory.len() == service.binaries.len(),
        "service manifest repeats a binary"
    );
    let active_unit_count = service.services.iter().filter(|row| row.active).count();
    let files = manifest
        .get("files")
        .and_then(Value::as_array)
        .context("bundle manifest lacks file inventory")?;
    let bundled_digest = |target: &str| {
        files.iter().find_map(|row| {
            (row.get("target").and_then(Value::as_str) == Some(target))
                .then(|| row.get("sha256").and_then(Value::as_str))
                .flatten()
        })
    };
    let bundled_scq_projects_sha256 = bundled_digest("external/scq-projects.active.toml")
        .context("bundle lacks the active SCQ project policy")?
        .to_owned();
    ensure!(
        sha256_file(&bundle_path.join("external/scq-projects.active.toml"))?
            == bundled_scq_projects_sha256,
        "bundled SCQ project policy hash drift"
    );
    let scq_projects_active_sha256 = if args.scq_projects_active.is_file() {
        Some(sha256_file(&args.scq_projects_active)?)
    } else {
        None
    };
    let mut units = Vec::new();
    let mut violations = Vec::new();
    match scq_projects_active_sha256.as_deref() {
        Some(digest) if digest == bundled_scq_projects_sha256 => {}
        Some(_) => violations.push("active SCQ project policy differs from the bundle".to_owned()),
        None => violations.push("active SCQ project policy is absent".to_owned()),
    }
    for row in service.services.iter().filter(|row| row.active) {
        if !binary_inventory.contains(&row.binary) {
            violations.push(format!(
                "{} references unbundled binary {}",
                row.unit, row.binary
            ));
            continue;
        }
        let show = systemctl(&args.systemctl, "show", &row.unit);
        let cat = systemctl(&args.systemctl, "cat", &row.unit);
        match (show, cat) {
            (Ok(show), Ok(cat)) => {
                let mut audit = inspect_unit(row, &args.root, &show, &cat);
                let target = format!("runtime/bin/{}", row.binary);
                match bundled_digest(&target) {
                    Some(expected) => {
                        audit.expected_binary_sha256 = Some(expected.to_owned());
                        match audit.main_pid {
                            Some(pid) if pid > 0 => {
                                let process_path = PathBuf::from(format!("/proc/{pid}/exe"));
                                match (fs::read_link(&process_path), sha256_file(&process_path)) {
                                    (Ok(exe), Ok(digest)) => {
                                        audit.process_exe =
                                            Some(exe.to_string_lossy().into_owned());
                                        audit.process_sha256 = Some(digest.clone());
                                        if digest != expected {
                                            audit.violations.push(
                                                "running process hash differs from the bundle"
                                                    .to_owned(),
                                            );
                                        }
                                    }
                                    (exe, digest) => audit.violations.push(format!(
                                        "cannot resolve running process: exe={} hash={}",
                                        exe.err()
                                            .map(|error| error.to_string())
                                            .unwrap_or_else(|| "ok".to_owned()),
                                        digest
                                            .err()
                                            .map(|error| format!("{error:#}"))
                                            .unwrap_or_else(|| "ok".to_owned()),
                                    )),
                                }
                            }
                            _ if audit.active_state.as_deref() == Some("active")
                                && audit.sub_state.as_deref() == Some("running") =>
                            {
                                audit
                                    .violations
                                    .push("running unit has no resolvable MainPID".to_owned());
                            }
                            _ => {}
                        }
                    }
                    None => audit
                        .violations
                        .push("service binary is absent from bundle inventory".to_owned()),
                }
                audit.healthy = audit.violations.is_empty();
                violations.extend(
                    audit
                        .violations
                        .iter()
                        .map(|violation| format!("{}: {violation}", row.unit)),
                );
                units.push(audit);
            }
            (show, cat) => violations.push(format!(
                "{}: {}{}",
                row.unit,
                show.err()
                    .map(|error| format!("show failed: {error:#}"))
                    .unwrap_or_default(),
                cat.err()
                    .map(|error| format!(" cat failed: {error:#}"))
                    .unwrap_or_default(),
            )),
        }
    }
    Ok(EffectiveUnitAudit {
        schema: SCHEMA,
        generated_at_ms: now_ms(),
        root: args.root.clone(),
        bundle_sha256,
        bundle_manifest_path,
        bundle_manifest_sha256,
        current_bundle_path,
        synchronization_receipt_path,
        synchronization_receipt_sha256,
        rollback_bundle_sha256,
        service_manifest_sha256,
        scq_projects_active_path: args.scq_projects_active.clone(),
        scq_projects_active_sha256,
        bundled_scq_projects_sha256,
        healthy: violations.is_empty() && units.len() == active_unit_count,
        units,
        violations,
    })
}

fn main() -> Result<()> {
    let args = Args::parse();
    let audit = build(&args)?;
    let output = args
        .output
        .clone()
        .unwrap_or_else(|| args.root.join("ops/effective-unit-audit.v1.json"));
    atomic_json(&output, &serde_json::to_value(&audit)?)?;
    println!("{}", serde_json::to_string(&audit)?);
    ensure!(audit.healthy, "effective unit audit found drift");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row() -> ServiceRow {
        ServiceRow {
            unit: "queue.service".to_owned(),
            binary: "jope_selfplay_queue".to_owned(),
            owner: "queue".to_owned(),
            active: true,
        }
    }

    #[test]
    fn exact_bundle_execstart_is_healthy() {
        let root = Path::new("/srv/flywheel");
        let exec = "/srv/flywheel/runtime-bundles/current/runtime/bin/jope_selfplay_queue --submit";
        let show = format!(
            "FragmentPath=/units/queue.service\nDropInPaths=\nExecStart={exec}\nActiveState=active\nSubState=running\n"
        );
        let cat = format!("[Service]\nExecStart={exec}\n");
        assert!(inspect_unit(&row(), root, &show, &cat).healthy);
    }

    #[test]
    fn mutable_or_overridden_execstart_fails_closed() {
        let root = Path::new("/srv/flywheel");
        let show = format!(
            "FragmentPath=/units/queue.service\nDropInPaths=/override.conf\nExecStart={}/target/release/jope_selfplay_queue\nActiveState=active\nSubState=running\n",
            paths::jope_home().display()
        );
        let show = show.as_str();
        let cat = format!("{show}# /override.conf\n[Service]\nExecStart=/tmp/override\n");
        let audit = inspect_unit(&row(), root, show, &cat);
        assert!(!audit.healthy);
        assert!(audit
            .violations
            .iter()
            .any(|row| row.contains("mutable path")));
        assert!(audit.violations.iter().any(|row| row.contains("drop-in")));
    }
}
