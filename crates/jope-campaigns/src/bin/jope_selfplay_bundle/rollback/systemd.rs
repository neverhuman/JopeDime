use super::*;

pub(crate) fn rollback_derivation_path(output_root: &Path, bundle_sha256: &str) -> PathBuf {
    output_root
        .join("rollback-derivations")
        .join(format!("{bundle_sha256}.v1.json"))
}

pub(crate) fn verify_intrinsic_existing(output_root: &Path, bundle_sha256: &str) -> Result<Value> {
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

pub(crate) fn rollback_bundle_files(manifest: &Value) -> Result<Vec<RollbackBundleFileV2>> {
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

pub(crate) fn parse_systemd_artifact(
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

pub(crate) fn systemd_closure(
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

pub(crate) fn unchanged_inventory_root(
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

pub(crate) fn git_blob(source: &Path, commit: &str, path: &str) -> Result<Vec<u8>> {
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

pub(crate) fn transform_rollback_unit(
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
