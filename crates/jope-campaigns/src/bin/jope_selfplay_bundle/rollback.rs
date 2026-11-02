use super::*;

pub(super) fn rollback_derivation_path(output_root: &Path, bundle_sha256: &str) -> PathBuf {
    output_root
        .join("rollback-derivations")
        .join(format!("{bundle_sha256}.v1.json"))
}

pub(super) fn verify_intrinsic_existing(output_root: &Path, bundle_sha256: &str) -> Result<Value> {
    require_bundle_sha256(bundle_sha256)?;
    let root = output_root.join(bundle_sha256);
    let path = root.join("bundle.json");
    let manifest: Value = serde_json::from_reader(std::io::BufReader::new(
        File::open(&path).with_context(|| format!("open staged bundle {}", path.display()))?,
    ))?;
    if manifest
        .get("clean_detached_source")
        .and_then(Value::as_bool)
        == Some(true)
    {
        verify_existing(&root, &manifest)?;
    } else {
        ensure!(
            bundle_sha256 == HISTORICAL_ROLLBACK_PARENT_SHA256
                || rollback_derivation_path(output_root, bundle_sha256).is_file(),
            "legacy-provenance bundle is neither the historical rollback parent nor a strict derivation"
        );
        verify_rollback_candidate(&root, &manifest, bundle_sha256)?;
        let rows = manifest
            .get("files")
            .and_then(Value::as_array)
            .context("legacy bundle lacks file inventory")?;
        ensure!(
            manifest_inventory_identity(rows)? == bundle_sha256,
            "legacy bundle inventory identity drift"
        );
        if bundle_sha256 != HISTORICAL_ROLLBACK_PARENT_SHA256 {
            verify_rollback_derivation(output_root, bundle_sha256)?;
        }
    }
    Ok(manifest)
}

pub(super) fn rollback_bundle_files(manifest: &Value) -> Result<Vec<RollbackBundleFileV2>> {
    let mut files = manifest
        .get("files")
        .and_then(Value::as_array)
        .context("rollback bundle lacks file inventory")?
        .iter()
        .map(|row| {
            Ok(RollbackBundleFileV2 {
                target: row
                    .get("target")
                    .and_then(Value::as_str)
                    .context("rollback file target is absent")?
                    .to_owned(),
                bytes: row
                    .get("bytes")
                    .and_then(Value::as_u64)
                    .context("rollback file size is absent")?,
                sha256: row
                    .get("sha256")
                    .and_then(Value::as_str)
                    .context("rollback file digest is absent")?
                    .to_owned(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    files.sort_by(|left, right| left.target.cmp(&right.target));
    ensure!(
        files.windows(2).all(|pair| pair[0].target < pair[1].target),
        "rollback file inventory is duplicated"
    );
    Ok(files)
}

pub(super) fn parse_systemd_artifact(
    root: &Path,
    target: &str,
    expected_sha256: &str,
) -> Result<RollbackSystemdArtifactV2> {
    let unit = Path::new(target)
        .file_name()
        .and_then(|name| name.to_str())
        .context("systemd target has no unit name")?
        .to_owned();
    let kind = if unit.ends_with(".service") {
        "service"
    } else if unit.ends_with(".timer") {
        "timer"
    } else {
        bail!("unsupported systemd artifact: {unit}")
    };
    let contents = fs::read_to_string(root.join(target))?;
    ensure!(
        sha256_bytes(contents.as_bytes()) == expected_sha256,
        "systemd artifact digest differs from bundle inventory: {unit}"
    );
    let logical = systemd_logical_lines(&contents)?;
    let exec_start_argv = if kind == "service" {
        let line = logical
            .iter()
            .find_map(|line| line.trim().strip_prefix("ExecStart="))
            .with_context(|| format!("rollback service has no ExecStart: {unit}"))?;
        line.split_ascii_whitespace()
            .map(|argument| argument.trim_matches(['\'', '"']).to_owned())
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let working_directory = logical
        .iter()
        .find_map(|line| line.trim().strip_prefix("WorkingDirectory="))
        .map(str::to_owned);
    let mut artifact = RollbackSystemdArtifactV2 {
        unit,
        kind: kind.to_owned(),
        bundled_target: target.to_owned(),
        unit_file_sha256: expected_sha256.to_owned(),
        resolved_effective_unit_sha256: String::new(),
        exec_start_argv,
        working_directory,
        drop_ins: Vec::<ArtifactIdentity>::new(),
    };
    artifact.resolved_effective_unit_sha256 = artifact.computed_effective_unit_sha256()?;
    Ok(artifact)
}

pub(super) fn systemd_closure(
    root: &Path,
    manifest: &Value,
) -> Result<Vec<RollbackSystemdArtifactV2>> {
    let mut artifacts = manifest
        .get("files")
        .and_then(Value::as_array)
        .context("rollback bundle lacks file inventory")?
        .iter()
        .filter_map(|row| {
            let target = row.get("target")?.as_str()?;
            (target.starts_with("runtime/systemd/")
                && (target.ends_with(".service") || target.ends_with(".timer")))
            .then_some((target, row.get("sha256")?.as_str()?))
        })
        .map(|(target, digest)| parse_systemd_artifact(root, target, digest))
        .collect::<Result<Vec<_>>>()?;
    artifacts.sort_by(|left, right| left.unit.cmp(&right.unit));
    ensure!(
        artifacts.windows(2).all(|pair| pair[0].unit < pair[1].unit),
        "rollback systemd inventory is duplicated"
    );
    Ok(artifacts)
}

pub(super) fn unchanged_inventory_root(
    rows: &[Value],
    changed: &BTreeSet<String>,
) -> Result<String> {
    canonical_json_sha256(&Value::Array(
        rows.iter()
            .filter(|row| {
                row.get("target")
                    .and_then(Value::as_str)
                    .is_some_and(|target| !changed.contains(target))
            })
            .cloned()
            .collect(),
    ))
}

pub(super) fn git_blob(source: &Path, commit: &str, path: &str) -> Result<Vec<u8>> {
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(source)
        .arg("show")
        .arg(format!("{commit}:{path}"))
        .output()
        .with_context(|| format!("read reviewed source {commit}:{path}"))?;
    ensure!(
        output.status.success(),
        "reviewed source is unreachable: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    Ok(output.stdout)
}

pub(super) fn transform_rollback_unit(
    unit: &str,
    before: &[u8],
    reviewed: &[u8],
) -> Result<Vec<u8>> {
    let before_text = String::from_utf8(before.to_vec())?;
    let from_line = format!(
        "WorkingDirectory={rollback_working_directory_from_str}",
        rollback_working_directory_from_str = rollback_working_directory_from_str()
    );
    let to_line = format!(
        "WorkingDirectory={immutable_flywheel_root_str}",
        immutable_flywheel_root_str = immutable_flywheel_root_str()
    );
    ensure!(
        before_text
            .lines()
            .filter(|line| *line == from_line)
            .count()
            == 1,
        "rollback parent unit does not contain exactly one permitted line: {unit}"
    );
    let reviewed_text = String::from_utf8(reviewed.to_vec())?;
    ensure!(
        reviewed_text.lines().any(|line| line == to_line),
        "reviewed source does not contain the canonical WorkingDirectory: {unit}"
    );
    Ok(before_text.replacen(&from_line, &to_line, 1).into_bytes())
}

pub(super) fn derive_rollback_baseline(
    source: &Path,
    output_root: &Path,
) -> Result<RollbackBaselineDerivationV1> {
    let parent_root = output_root.join(HISTORICAL_ROLLBACK_PARENT_SHA256);
    let mut manifest = verify_intrinsic_existing(output_root, HISTORICAL_ROLLBACK_PARENT_SHA256)?;
    let parent_manifest_path = parent_root.join("bundle.json");
    let parent_manifest_file_sha256 = sha256_file(&parent_manifest_path)?;
    let parent_manifest_canonical_json_sha256 = canonical_json_sha256(&manifest)?;
    let rows = manifest
        .get("files")
        .and_then(Value::as_array)
        .context("rollback parent has no file inventory")?
        .clone();
    let changed_targets = ROLLBACK_REBASE_UNITS
        .iter()
        .map(|unit| format!("runtime/systemd/{unit}"))
        .collect::<BTreeSet<_>>();
    ensure!(
        changed_targets.len() == 7,
        "rollback rebase unit allowlist is not exactly seven"
    );

    let staging = output_root.join(format!(".rollback-rebase.tmp-{}", std::process::id()));
    ensure!(
        !staging.exists(),
        "rollback rebase staging directory already exists"
    );
    fs::create_dir_all(&staging)?;
    let result = (|| -> Result<RollbackBaselineDerivationV1> {
        let mut transformations = Vec::new();
        for row in &rows {
            let target = row
                .get("target")
                .and_then(Value::as_str)
                .context("rollback parent row lacks target")?;
            let destination = staging.join(target);
            if let Some(parent) = destination.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(parent_root.join(target), &destination)?;
        }
        let manifest_rows = manifest
            .get_mut("files")
            .and_then(Value::as_array_mut)
            .context("rollback parent file inventory is not mutable")?;
        for unit in ROLLBACK_REBASE_UNITS {
            let target = format!("runtime/systemd/{unit}");
            let path = staging.join(&target);
            let before = fs::read(&path)?;
            let reviewed = git_blob(
                source,
                REVIEWED_ROLLBACK_UNIT_COMMIT,
                &format!("systemd/{unit}"),
            )?;
            let after = transform_rollback_unit(unit, &before, &reviewed)?;
            fs::write(&path, &after)?;
            let before_sha256 = sha256_bytes(&before);
            let after_sha256 = sha256_bytes(&after);
            let manifest_row = manifest_rows
                .iter_mut()
                .find(|row| row.get("target").and_then(Value::as_str) == Some(target.as_str()))
                .with_context(|| format!("rollback parent inventory omits {target}"))?;
            ensure!(
                manifest_row.get("sha256").and_then(Value::as_str) == Some(before_sha256.as_str()),
                "rollback parent unit inventory drift: {unit}"
            );
            manifest_row["sha256"] = Value::String(after_sha256.clone());
            manifest_row["bytes"] = Value::from(after.len() as u64);
            transformations.push(RollbackUnitTransformationV1 {
                target,
                before_sha256,
                after_sha256,
                reviewed_source_sha256: sha256_bytes(&reviewed),
                from: rollback_working_directory_from_str().to_owned(),
                to: immutable_flywheel_root_str().to_owned(),
            });
        }
        transformations.sort_by(|left, right| left.target.cmp(&right.target));
        let derived_bundle_sha256 = manifest_inventory_identity(manifest_rows)?;
        manifest["bundle_sha256"] = Value::String(derived_bundle_sha256.clone());
        atomic_json(&staging.join("bundle.json"), &manifest)?;
        let final_root = output_root.join(&derived_bundle_sha256);
        if final_root.exists() {
            let existing = verify_intrinsic_existing(output_root, &derived_bundle_sha256)?;
            ensure!(
                existing == manifest,
                "existing derived rollback bundle manifest drift"
            );
            fs::remove_dir_all(&staging)?;
        } else {
            fs::rename(&staging, &final_root)?;
        }
        let derivation = RollbackBaselineDerivationV1 {
            schema: ROLLBACK_BASELINE_DERIVATION_SCHEMA.to_owned(),
            parent_bundle_sha256: HISTORICAL_ROLLBACK_PARENT_SHA256.to_owned(),
            derived_bundle_sha256: derived_bundle_sha256.clone(),
            parent_manifest_file_sha256,
            parent_manifest_canonical_json_sha256,
            reviewed_source_commit: REVIEWED_ROLLBACK_UNIT_COMMIT.to_owned(),
            transformations,
            unchanged_inventory_root_sha256: unchanged_inventory_root(&rows, &changed_targets)?,
        };
        derivation.validate()?;
        atomic_json(
            &rollback_derivation_path(output_root, &derived_bundle_sha256),
            &serde_json::to_value(&derivation)?,
        )?;
        verify_rollback_derivation(output_root, &derived_bundle_sha256)?;
        Ok(derivation)
    })();
    if result.is_err() && staging.exists() {
        fs::remove_dir_all(&staging)?;
    }
    result
}

pub(super) fn verify_rollback_derivation(
    output_root: &Path,
    derived_bundle_sha256: &str,
) -> Result<()> {
    let derivation_path = rollback_derivation_path(output_root, derived_bundle_sha256);
    let derivation: RollbackBaselineDerivationV1 =
        serde_json::from_reader(std::io::BufReader::new(
            File::open(&derivation_path).with_context(|| {
                format!("open rollback derivation {}", derivation_path.display())
            })?,
        ))?;
    derivation.validate()?;
    ensure!(
        derivation.derived_bundle_sha256 == derived_bundle_sha256,
        "rollback derivation output identity mismatch"
    );
    let parent_root = output_root.join(&derivation.parent_bundle_sha256);
    let derived_root = output_root.join(derived_bundle_sha256);
    let parent_manifest_path = parent_root.join("bundle.json");
    let parent_manifest: Value =
        serde_json::from_reader(std::io::BufReader::new(File::open(&parent_manifest_path)?))?;
    let derived_manifest: Value = serde_json::from_reader(std::io::BufReader::new(File::open(
        derived_root.join("bundle.json"),
    )?))?;
    ensure!(
        sha256_file(&parent_manifest_path)? == derivation.parent_manifest_file_sha256
            && canonical_json_sha256(&parent_manifest)?
                == derivation.parent_manifest_canonical_json_sha256,
        "rollback derivation parent manifest drift"
    );
    let changed = derivation
        .transformations
        .iter()
        .map(|row| row.target.clone())
        .collect::<BTreeSet<_>>();
    let parent_rows = parent_manifest
        .get("files")
        .and_then(Value::as_array)
        .context("rollback parent lacks files")?;
    let derived_rows = derived_manifest
        .get("files")
        .and_then(Value::as_array)
        .context("derived rollback lacks files")?;
    ensure!(
        parent_rows.len() == derived_rows.len()
            && unchanged_inventory_root(parent_rows, &changed)?
                == derivation.unchanged_inventory_root_sha256,
        "rollback derivation inventory shape/root drift"
    );
    for (parent, derived) in parent_rows.iter().zip(derived_rows) {
        let target = parent
            .get("target")
            .and_then(Value::as_str)
            .context("rollback parent target is absent")?;
        ensure!(
            derived.get("target").and_then(Value::as_str) == Some(target),
            "rollback derivation reordered or replaced an inventory target"
        );
        if !changed.contains(target) {
            ensure!(
                parent == derived,
                "rollback derivation changed a non-unit manifest row: {target}"
            );
            ensure!(
                sha256_file(&parent_root.join(target))? == sha256_file(&derived_root.join(target))?,
                "rollback derivation changed non-unit bytes: {target}"
            );
            continue;
        }
        let transformation = derivation
            .transformations
            .iter()
            .find(|row| row.target == target)
            .context("rollback derivation changed an unrecorded target")?;
        let before = fs::read(parent_root.join(target))?;
        let after = fs::read(derived_root.join(target))?;
        ensure!(
            sha256_bytes(&before) == transformation.before_sha256
                && sha256_bytes(&after) == transformation.after_sha256,
            "rollback derivation unit digest drift: {target}"
        );
        let expected = String::from_utf8(before)?.replacen(
            &format!("WorkingDirectory={}", transformation.from),
            &format!("WorkingDirectory={}", transformation.to),
            1,
        );
        ensure!(
            expected.as_bytes() == after.as_slice(),
            "rollback derivation made a change other than the permitted WorkingDirectory replacement"
        );
    }
    ensure!(
        manifest_inventory_identity(derived_rows)? == derived_bundle_sha256,
        "derived rollback bundle identity drift"
    );
    Ok(())
}

pub(super) fn verify_rollback_candidate(
    root: &Path,
    manifest: &Value,
    expected_sha256: &str,
) -> Result<()> {
    ensure!(
        manifest.get("bundle_sha256").and_then(Value::as_str) == Some(expected_sha256)
            && root.file_name().and_then(|name| name.to_str()) == Some(expected_sha256),
        "rollback candidate identity drift"
    );
    let rows = manifest
        .get("files")
        .and_then(Value::as_array)
        .context("rollback bundle lacks file inventory")?;
    let targets = rows
        .iter()
        .map(|row| {
            let target = row
                .get("target")
                .and_then(Value::as_str)
                .context("rollback file target is absent")?;
            let expected = row
                .get("sha256")
                .and_then(Value::as_str)
                .context("rollback file digest is absent")?;
            ensure!(
                sha256_file(&root.join(target))? == expected,
                "rollback bundle content drift: {target}"
            );
            Ok(target)
        })
        .collect::<Result<BTreeSet<_>>>()?;
    let service: ServiceManifest = serde_json::from_value(
        manifest
            .get("service_manifest")
            .cloned()
            .context("rollback bundle lacks service manifest")?,
    )?;
    validate_service_manifest(&service)?;
    for row in &service.services {
        ensure!(
            targets.contains(format!("runtime/bin/{}", row.binary).as_str())
                && targets.contains(format!("runtime/systemd/{}", row.unit).as_str()),
            "rollback service/unit closure is incomplete for {}",
            row.unit
        );
    }
    ensure!(
        targets.contains("runtime/bin/corpus_eval"),
        "rollback bundle lacks evaluator"
    );
    Ok(())
}

pub(super) fn verify_rollback_capsule(
    output_root: &Path,
    evidence_root: &Path,
    capsule: &RollbackCapsuleV2,
    now_unix_ms: u64,
) -> Result<String> {
    capsule.validate_evidence_at(evidence_root, now_unix_ms)?;
    let runtime_root = output_root.join(&capsule.runtime_bundle_sha256);
    let runtime_manifest_path = runtime_root.join("bundle.json");
    ensure!(
        sha256_file(&runtime_manifest_path)? == capsule.runtime_bundle_manifest_file_sha256,
        "rollback runtime manifest digest mismatch"
    );
    let runtime_manifest: Value =
        serde_json::from_reader(std::io::BufReader::new(File::open(&runtime_manifest_path)?))?;
    ensure!(
        canonical_json_sha256(&runtime_manifest)?
            == capsule.runtime_bundle_manifest_canonical_json_sha256,
        "rollback runtime manifest canonical digest mismatch"
    );
    verify_rollback_candidate(
        &runtime_root,
        &runtime_manifest,
        &capsule.runtime_bundle_sha256,
    )?;

    let service_value = runtime_manifest
        .get("service_manifest")
        .cloned()
        .context("rollback runtime lacks service manifest")?;
    ensure!(
        canonical_json_sha256(&service_value)?
            == capsule.embedded_service_manifest_canonical_json_sha256,
        "rollback embedded service manifest canonical digest mismatch"
    );
    let service: ServiceManifest = serde_json::from_value(service_value)?;
    let service_target = "runtime/manifests/selfplay/service-manifest.v3.json";
    let service_path = runtime_root.join(service_target);
    ensure!(
        sha256_file(&service_path)? == capsule.embedded_service_manifest_file_sha256
            && runtime_manifest
                .get("service_manifest_sha256")
                .and_then(Value::as_str)
                == Some(capsule.embedded_service_manifest_file_sha256.as_str()),
        "rollback service-manifest raw-file digest mismatch"
    );
    let expected_units = service
        .services
        .iter()
        .map(|row| row.unit.as_str())
        .collect::<BTreeSet<_>>();
    let capsule_services = capsule
        .systemd_closure
        .iter()
        .filter(|row| row.kind == "service")
        .map(|row| row.unit.as_str())
        .collect::<BTreeSet<_>>();
    ensure!(
        capsule_services == expected_units,
        "rollback systemd closure does not contain the complete service inventory"
    );
    let actual_files = rollback_bundle_files(&runtime_manifest)?;
    ensure!(
        actual_files == capsule.files,
        "rollback capsule file inventory differs from the bundle manifest"
    );
    let actual_systemd = systemd_closure(&runtime_root, &runtime_manifest)?;
    ensure!(
        actual_systemd == capsule.systemd_closure,
        "rollback capsule systemd closure differs from bundled unit bytes"
    );
    capsule.digest_at(evidence_root, now_unix_ms)
}

pub(super) fn verify_existing(root: &Path, manifest: &Value) -> Result<()> {
    ensure!(manifest["runtime"] == "rust", "bundle runtime is not Rust");
    ensure!(
        manifest["operational_python_allowed"] == false,
        "bundle permits operational Python"
    );
    ensure!(
        manifest
            .get("clean_detached_source")
            .and_then(Value::as_bool)
            == Some(true)
            && manifest
                .get("integration_commit")
                .and_then(Value::as_str)
                .is_some_and(|commit| {
                    commit.len() == 40 && commit.bytes().all(|byte| byte.is_ascii_hexdigit())
                }),
        "bundle lacks clean detached integration provenance"
    );
    ensure!(
        manifest
            .pointer("/prime_support/source_bundle_sha256")
            .and_then(Value::as_str)
            == Some(PRIME_SOURCE_BUNDLE_SHA256),
        "bundle does not bind immutable Prime support"
    );
    let required_nodes = manifest
        .get("required_sync_nodes")
        .and_then(Value::as_array)
        .context("bundle lacks five-node synchronization contract")?;
    ensure!(
        required_nodes
            == &GPU_SYNC_NODES
                .iter()
                .map(|node| Value::String((*node).to_owned()))
                .collect::<Vec<_>>(),
        "bundle synchronization node set drift"
    );
    ensure!(
        manifest
            .pointer("/five_node_synchronization/schema")
            .and_then(Value::as_str)
            == Some("JopeBundleSynchronizationContract.v1")
            && manifest
                .pointer("/five_node_synchronization/receipt_schema")
                .and_then(Value::as_str)
                == Some("JopeBundleSynchronizationReceipt.v1")
            && manifest
                .pointer("/five_node_synchronization/activation_requires_complete")
                .and_then(Value::as_bool)
                == Some(true)
            && manifest
                .pointer("/five_node_synchronization/required_nodes")
                .and_then(Value::as_array)
                == Some(required_nodes)
            && manifest
                .pointer("/five_node_synchronization/receipt_relative_path")
                .and_then(Value::as_str)
                == manifest
                    .get("bundle_sha256")
                    .and_then(Value::as_str)
                    .map(|identity| format!("synchronization/{identity}.v1.json"))
                    .as_deref(),
        "bundle synchronization receipt contract drift"
    );
    let rows = manifest["files"].as_array().context("manifest files")?;
    let service_manifest: ServiceManifest = serde_json::from_value(
        manifest
            .get("service_manifest")
            .cloned()
            .context("bundle lacks embedded service manifest")?,
    )?;
    validate_recovery_service_manifest(&service_manifest)?;
    let identity_rows: Vec<_> = rows
        .iter()
        .map(|row| {
            json!({
                "target": row["target"],
                "sha256": row["sha256"],
                "bytes": row["bytes"],
                "python_role": row.get("python_role").cloned().unwrap_or(Value::Null),
            })
        })
        .collect();
    let computed_identity = sha256_bytes(&serde_json::to_vec(&identity_rows)?);
    ensure!(
        manifest["bundle_sha256"].as_str() == Some(computed_identity.as_str()),
        "bundle manifest identity drift"
    );
    ensure!(
        root.file_name().and_then(|name| name.to_str()) == Some(computed_identity.as_str()),
        "bundle directory does not match its identity"
    );
    for row in rows {
        let target = row["target"].as_str().context("file target")?;
        let expected = row["sha256"].as_str().context("file sha256")?;
        ensure!(
            sha256_file(&root.join(target))? == expected,
            "existing bundle content drift: {target}"
        );
        if target.ends_with(".service") {
            validate_systemd_unit(&fs::read_to_string(root.join(target))?)?;
        }
        if target.ends_with(".py") {
            let role = row
                .get("python_role")
                .and_then(Value::as_str)
                .context("unclassified Python in bundle")?;
            validate_manifest_python_target(target, role)?;
        }
        if target.ends_with(".sh") {
            let contents = fs::read_to_string(root.join(target))?;
            if target == "pytorch/encoding/runner.sh" {
                validate_encoding_runner(&contents)?;
            } else {
                validate_no_python_runtime(&contents)?;
            }
        }
    }
    for service in service_manifest.services.iter().filter(|row| row.active) {
        ensure!(
            rows.iter().any(|row| {
                row.get("target").and_then(Value::as_str)
                    == Some(format!("runtime/bin/{}", service.binary).as_str())
            }) && rows.iter().any(|row| {
                row.get("target").and_then(Value::as_str)
                    == Some(format!("runtime/systemd/{}", service.unit).as_str())
            }),
            "bundle service/unit closure is incomplete for {}",
            service.unit
        );
    }
    let prime_files = manifest
        .pointer("/prime_support/files")
        .and_then(Value::as_array)
        .context("bundle lacks Prime per-file provenance")?;
    let mut canonical_prime_files =
        serde_json::from_value::<Vec<PrimeSupportFile>>(Value::Array(prime_files.clone()))?;
    canonical_prime_files.sort_by(|left, right| {
        left.source_bundle_target
            .cmp(&right.source_bundle_target)
            .then_with(|| left.bundle_target.cmp(&right.bundle_target))
    });
    let computed_prime_tree_sha256 = prime_support_identity(&canonical_prime_files)?;
    ensure!(
        manifest
            .pointer("/prime_support/package_tree_sha256")
            .and_then(Value::as_str)
            == Some(computed_prime_tree_sha256.as_str()),
        "Prime package-tree digest drift"
    );
    for prime in prime_files {
        let target = prime
            .get("bundle_target")
            .and_then(Value::as_str)
            .context("Prime support row lacks bundle target")?;
        let digest = prime
            .get("sha256")
            .and_then(Value::as_str)
            .context("Prime support row lacks digest")?;
        ensure!(
            rows.iter().any(|row| {
                row.get("target").and_then(Value::as_str) == Some(target)
                    && row.get("sha256").and_then(Value::as_str) == Some(digest)
            }),
            "Prime support row is absent from bundle inventory: {target}"
        );
    }
    for required in [
        "pytorch/jope/__init__.py",
        "pytorch/jope/policy/model_v3.py",
        "pytorch/jope/policy/decode.py",
        "pytorch/proposals/test_exact_context_proposals.py",
    ] {
        ensure!(
            root.join(required).is_file(),
            "bundle lacks required JOPE proposal dependency: {required}"
        );
    }
    Ok(())
}

pub(super) fn validate_manifest_python_target(target: &str, role: &str) -> Result<()> {
    if role == "pytorch_dependency_package"
        && target.starts_with("pytorch/jope/")
        && target.ends_with(".py")
    {
        return Ok(());
    }
    ensure!(
        PYTORCH_ALLOWLIST
            .iter()
            .any(|(_, allowed_target, allowed_role)| {
                target == *allowed_target && role == *allowed_role
            }),
        "unallowlisted Python in bundle: {target}"
    );
    Ok(())
}

pub(super) fn validate_service_manifest(manifest: &ServiceManifest) -> Result<()> {
    ensure!(
        manifest.schema == "JopeSelfplayServiceManifest.v3",
        "unsupported service manifest schema"
    );
    let binaries = manifest.binaries.iter().collect::<BTreeSet<_>>();
    ensure!(
        binaries.len() == manifest.binaries.len() && !binaries.is_empty(),
        "service manifest binary inventory is empty or duplicated"
    );
    let mut units = BTreeSet::new();
    for row in &manifest.services {
        ensure!(
            row.unit.ends_with(".service")
                && !row.unit.contains('/')
                && !row.owner.trim().is_empty(),
            "service manifest contains an invalid unit or owner"
        );
        ensure!(
            units.insert(row.unit.as_str()),
            "service manifest repeats unit {}",
            row.unit
        );
        ensure!(
            binaries.contains(&row.binary),
            "service {} references unbundled binary {}",
            row.unit,
            row.binary
        );
    }
    Ok(())
}

pub(super) fn validate_recovery_service_manifest(manifest: &ServiceManifest) -> Result<()> {
    validate_service_manifest(manifest)?;
    ensure!(
        manifest.services.iter().any(|row| {
            row.unit == "jope-reporting.service" && row.binary == "jope_reporting" && row.active
        }),
        "recovery bundle must install and activate jope-reporting.service"
    );
    ensure!(
        manifest.services.iter().any(|row| {
            row.unit == "jope-selfplay-expansion.service"
                && row.binary == "jope_selfplay_expansion_driver"
                && !row.active
        }),
        "recovery bundle must carry expansion as an inactive Encoder V2-gated unit"
    );
    Ok(())
}

pub(super) fn ensure_bundle_target(root: &Path, target: &Path) -> Result<()> {
    ensure!(
        target.starts_with(root),
        "bundle target escaped temporary directory"
    );
    Ok(())
}
