use super::*;

pub(super) fn candidate_units(candidate_root: &Path) -> Result<Vec<ActivationUnitCandidateV1>> {
    let root = candidate_root.join("runtime/systemd");
    ensure!(root.is_dir(), "candidate has no systemd directory");
    let mut units = fs::read_dir(root)?
        .collect::<std::io::Result<Vec<_>>>()?
        .into_iter()
        .filter(|entry| {
            entry
                .path()
                .extension()
                .and_then(|value| value.to_str())
                .is_some_and(|extension| matches!(extension, "service" | "timer"))
        })
        .map(|entry| {
            let unit = entry.file_name().to_string_lossy().into_owned();
            validate_unit_name(&unit)?;
            Ok(ActivationUnitCandidateV1 {
                unit,
                sha256: sha256_file(&entry.path())?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    units.sort_by(|left, right| left.unit.cmp(&right.unit));
    ensure!(
        !units.is_empty() && units.windows(2).all(|pair| pair[0].unit < pair[1].unit),
        "candidate systemd inventory is empty or duplicated"
    );
    Ok(units)
}

pub(super) fn active_timer_units(
    candidate_root: &Path,
    candidates: &[ActivationUnitCandidateV1],
) -> Result<Vec<String>> {
    let manifest: Value = serde_json::from_slice(&fs::read(candidate_root.join("bundle.json"))?)?;
    let active_services = manifest
        .pointer("/service_manifest/services")
        .and_then(Value::as_array)
        .context("candidate bundle lacks its embedded service policy")?
        .iter()
        .filter(|row| row.get("active").and_then(Value::as_bool) == Some(true))
        .filter_map(|row| row.get("unit").and_then(Value::as_str))
        .collect::<BTreeSet<_>>();
    let mut timers = Vec::new();
    for candidate in candidates.iter().filter(|row| row.unit.ends_with(".timer")) {
        let contents =
            fs::read_to_string(candidate_root.join("runtime/systemd").join(&candidate.unit))?;
        let target = contents
            .lines()
            .map(str::trim)
            .find_map(|line| line.strip_prefix("Unit="))
            .map(str::to_owned)
            .unwrap_or_else(|| format!("{}.service", candidate.unit.trim_end_matches(".timer")));
        if active_services.contains(target.as_str()) {
            timers.push(candidate.unit.clone());
        }
    }
    Ok(timers)
}

pub(super) fn current_unit_names(output_root: &Path) -> Result<Vec<String>> {
    let root = output_root.join("current/runtime/systemd");
    if !root.is_dir() {
        return Ok(Vec::new());
    }
    Ok(fs::read_dir(root)?
        .collect::<std::io::Result<Vec<_>>>()?
        .into_iter()
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            (name.ends_with(".service") || name.ends_with(".timer")).then_some(name)
        })
        .collect())
}

pub(super) fn validate_candidate(
    candidate_root: &Path,
    units: &[ActivationUnitCandidateV1],
) -> Result<()> {
    for unit in units {
        ensure!(
            sha256_file(&candidate_root.join("runtime/systemd").join(&unit.unit))? == unit.sha256,
            "candidate unit bytes drifted: {}",
            unit.unit
        );
    }
    Ok(())
}

pub(super) fn stage_units(
    transaction_root: &Path,
    candidate_root: &Path,
    units: &[ActivationUnitCandidateV1],
) -> Result<()> {
    let staged = transaction_root.join("staged/etc/systemd/user");
    fs::create_dir_all(&staged)?;
    for unit in units {
        let destination = staged.join(&unit.unit);
        fs::copy(
            candidate_root.join("runtime/systemd").join(&unit.unit),
            &destination,
        )?;
        ensure!(
            sha256_file(&destination)? == unit.sha256,
            "staged unit copy drift"
        );
    }
    Ok(())
}

pub(super) fn install_units(
    user_unit_root: &Path,
    transaction_root: &Path,
    preimages: &[ActivationUnitPreimageV1],
    candidates: &[ActivationUnitCandidateV1],
) -> Result<()> {
    fs::create_dir_all(user_unit_root)?;
    for preimage in preimages {
        validate_unit_name(&preimage.unit)?;
        let base = user_unit_root.join(&preimage.unit);
        if base.exists() {
            fs::remove_file(&base)?;
        }
        let drop_ins = user_unit_root.join(format!("{}.d", preimage.unit));
        if drop_ins.exists() {
            ensure!(
                drop_ins.starts_with(user_unit_root),
                "drop-in removal escaped user-unit root"
            );
            fs::remove_dir_all(&drop_ins)?;
        }
    }
    for candidate in candidates {
        let source = transaction_root
            .join("staged/etc/systemd/user")
            .join(&candidate.unit);
        let destination = user_unit_root.join(&candidate.unit);
        fs::copy(&source, &destination)?;
        ensure!(
            sha256_file(&destination)? == candidate.sha256,
            "installed unit copy drift"
        );
    }
    Ok(())
}

pub(super) fn restore_units(
    user_unit_root: &Path,
    transaction_root: &Path,
    preimages: &[ActivationUnitPreimageV1],
) -> Result<()> {
    for preimage in preimages {
        let base = user_unit_root.join(&preimage.unit);
        restore_raw_optional(&base, transaction_root, preimage.base_file.as_ref())?;
        let drop_in_root = user_unit_root.join(format!("{}.d", preimage.unit));
        if drop_in_root.exists() {
            fs::remove_dir_all(&drop_in_root)?;
        }
        for drop_in in &preimage.drop_ins {
            let destination = drop_in_root.join(&drop_in.relative_path);
            restore_raw_optional(&destination, transaction_root, Some(&drop_in.file))?;
        }
    }
    Ok(())
}

pub(super) fn switch_current(output_root: &Path, bundle_sha256: &str) -> Result<()> {
    ensure!(
        output_root
            .join(bundle_sha256)
            .join("bundle.json")
            .is_file(),
        "runtime target is absent"
    );
    let current = output_root.join("current");
    ensure!(
        !current.exists() || current.is_symlink(),
        "runtime current is not a symlink"
    );
    let temporary = output_root.join(format!(".current.activation-{}", std::process::id()));
    if temporary.is_symlink() {
        fs::remove_file(&temporary)?;
    }
    symlink(bundle_sha256, &temporary)?;
    fs::rename(temporary, current)?;
    Ok(())
}

pub(super) fn restore_current(output_root: &Path, target: Option<&str>) -> Result<()> {
    let current = output_root.join("current");
    if current.is_symlink() {
        fs::remove_file(&current)?;
    }
    if let Some(target) = target {
        let temporary = output_root.join(format!(".current.restore-{}", std::process::id()));
        symlink(target, &temporary)?;
        fs::rename(temporary, current)?;
    }
    Ok(())
}
