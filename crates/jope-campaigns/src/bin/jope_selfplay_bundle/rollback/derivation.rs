use super::*;

pub(crate) fn derive_rollback_baseline(
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

pub(crate) fn verify_rollback_derivation(
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

pub(crate) fn verify_rollback_candidate(
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

pub(crate) fn verify_rollback_capsule(
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
