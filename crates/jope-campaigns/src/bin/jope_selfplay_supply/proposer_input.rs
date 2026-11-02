use super::*;

pub(super) fn proposer_inputs_from_spec(body: &str) -> Result<Vec<PinnedProposerInput>> {
    let mut inputs = Vec::new();
    let mut current: Option<PinnedProposerInput> = None;
    let flush = |current: &mut Option<PinnedProposerInput>,
                 inputs: &mut Vec<PinnedProposerInput>|
     -> Result<()> {
        if let Some(input) = current.take() {
            if input.name.is_empty() || input.path.as_os_str().is_empty() {
                bail!("cohort proposer spec contains an incomplete input");
            }
            validate_sha256(&input.sha256)?;
            inputs.push(input);
        }
        Ok(())
    };
    for line in body.lines() {
        if line.trim() == "[[input]]" {
            flush(&mut current, &mut inputs)?;
            current = Some(PinnedProposerInput {
                name: String::new(),
                path: PathBuf::new(),
                sha256: String::new(),
            });
            continue;
        }
        let Some(input) = current.as_mut() else {
            continue;
        };
        if let Some(value) = toml_string_assignment(line, "name") {
            input.name = value;
        } else if let Some(value) = toml_string_assignment(line, "source") {
            input.path = PathBuf::from(value);
        } else if let Some(value) = toml_string_assignment(line, "sha256") {
            input.sha256 = value;
        } else if line.trim().starts_with("[[") || line.trim().starts_with('[') {
            flush(&mut current, &mut inputs)?;
        }
    }
    flush(&mut current, &mut inputs)?;
    inputs.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(inputs)
}

pub(super) fn runtime_pinned_intent_from_spec(
    spec: &Path,
    attempt: u32,
) -> Result<RuntimePinnedIntentV1> {
    let body = fs::read_to_string(spec)
        .with_context(|| format!("read immutable proposer spec {}", spec.display()))?;
    let job_identity = body
        .lines()
        .find_map(|line| toml_string_assignment(line, "id"))
        .context("cohort proposer spec lacks job identity")?;
    let pythonpath = fs::canonicalize(proposer_env_path(&body, "PYTHONPATH")?)
        .context("canonicalize pinned proposer PYTHONPATH")?;
    if pythonpath.file_name().and_then(|value| value.to_str()) != Some("pytorch") {
        bail!("pinned proposer PYTHONPATH is not inside a runtime bundle");
    }
    let runtime_bundle = pythonpath
        .parent()
        .context("pinned proposer PYTHONPATH has no runtime bundle parent")?
        .to_path_buf();
    let runtime_bundle_sha256 = runtime_bundle
        .file_name()
        .and_then(|value| value.to_str())
        .context("pinned proposer runtime bundle has no content identity")?
        .to_owned();
    validate_sha256(&runtime_bundle_sha256)?;
    let proposer_inputs = proposer_inputs_from_spec(&body)?;
    let pointer = proposer_inputs
        .iter()
        .find(|input| input.name == "pointer")
        .context("pinned proposer spec has no catalog pointer input")?;
    let value = RuntimePinnedIntentV1 {
        schema: RUNTIME_PINNED_INTENT_SCHEMA.to_owned(),
        runtime_bundle,
        runtime_bundle_sha256,
        catalog_snapshot: pointer.path.clone(),
        catalog_snapshot_sha256: pointer.sha256.clone(),
        proposer_inputs,
        immutable_spec_path: fs::canonicalize(spec).unwrap_or_else(|_| spec.to_path_buf()),
        immutable_spec_sha256: sha256_file(spec)?,
        job_identity,
        attempt,
    };
    value.validate()?;
    Ok(value)
}

pub(super) fn adaptive_proposal_reservoir(root: &Path) -> usize {
    let waves = root.join("waves");
    let Ok(entries) = fs::read_dir(waves) else {
        return 64;
    };
    let mut summaries = entries
        .flatten()
        .filter_map(|entry| {
            let wave = entry.file_name().to_string_lossy().into_owned();
            let number = parse_wave_id(&wave)?;
            let summary = entry.path().join("summary.json");
            summary.is_file().then_some((number, summary))
        })
        .collect::<Vec<_>>();
    summaries.sort_by_key(|(wave, _)| std::cmp::Reverse(*wave));
    let deficient = summaries
        .into_iter()
        .take(2)
        .map(|(_, path)| {
            fs::read(path)
                .ok()
                .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
                .and_then(|value| {
                    value
                        .pointer("/wave_summary/actual_d_by_task/binary")
                        .and_then(Value::as_array)
                        .map(|values| {
                            values
                                .iter()
                                .filter_map(Value::as_u64)
                                .any(|actual| actual < COHORT_NATIVE_D as u64)
                        })
                })
                .unwrap_or(true)
        })
        .collect::<Vec<_>>();
    match deficient.as_slice() {
        [true, true] => 256,
        [true, ..] => 128,
        _ => 64,
    }
}

pub(super) fn proposal_decoder_configuration_sha256(
    args: &Args,
    reservoir: usize,
) -> Result<String> {
    Ok(sha256_bytes(&serde_json::to_vec(&json!({
        "generator_sha256": sha256_file(&args.proposal_generator)?,
        "legacy_generator_sha256": sha256_file(&args.proposal_legacy_generator)?,
        "common_sha256": sha256_file(&args.proposal_common)?,
        "seed": COHORT_PROPOSAL_SEED,
        "proposal_reservoir_start": 64,
        "proposal_reservoir_maximum": 256,
        "legacy_adaptive_reservoir": reservoir,
        "selected_native_d": COHORT_NATIVE_D,
        "context_addressed": true,
        "selection_policy": "calibrated-or-empirical-dispersion-head-diversity-v3",
    }))?))
}

pub(super) fn append_spec_input(
    lines: &mut Vec<String>,
    name: &str,
    source: &Path,
    target: &str,
) -> Result<()> {
    lines.push("[[input]]".into());
    lines.push(format!("name = {}", toml_quote(name)?));
    lines.push(format!(
        "source = {}",
        toml_quote(&source.to_string_lossy())?
    ));
    lines.push(format!("target = {}", toml_quote(target)?));
    lines.push(format!("sha256 = {}", toml_quote(&sha256_file(source)?)?));
    lines.push(String::new());
    Ok(())
}

#[cfg(test)]
pub(super) fn normalized_proposer_spec(body: &str) -> String {
    body.lines()
        .map(|line| {
            if line.starts_with("PYTHONPATH = ") {
                "PYTHONPATH = \"<content-verified-runtime>\""
            } else if line.starts_with("JOPE_OPERATOR_REGISTRY = ") {
                "JOPE_OPERATOR_REGISTRY = \"<content-verified-runtime-registry>\""
            } else if line.starts_with("source = ")
                && (line.ends_with("/pytorch/proposals/test_exact_context_proposals.py\"")
                    || line.ends_with("/pytorch/proposals/test_exact_dl_generate.py\"")
                    || line.ends_with("/pytorch/shared/test_exact_common.py\"")
                    || line.ends_with("/pytorch/registry/operator_registry.json\""))
            {
                "source = \"<content-verified-runtime-input>\""
            } else {
                line
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub(super) fn proposer_env_path(body: &str, key: &str) -> Result<PathBuf> {
    let prefix = format!("{key} = \"");
    let line = body
        .lines()
        .find(|line| line.starts_with(&prefix))
        .with_context(|| format!("cohort proposer spec lacks {key}"))?;
    let value = line
        .strip_prefix(&prefix)
        .and_then(|value| value.strip_suffix('"'))
        .with_context(|| format!("cohort proposer {key} is malformed"))?;
    if value.is_empty() || value.contains(['\\', '\n', '\r', '"']) {
        bail!("cohort proposer {key} is unsafe");
    }
    Ok(PathBuf::from(value))
}

#[cfg(test)]
pub(super) fn validate_proposer_runtime(
    existing_body: &str,
    expected_body: &str,
    args: &Args,
) -> Result<()> {
    let prime_checkout = jope_paths::to_string(&paths::python_root());
    if existing_body.contains(&prime_checkout) || expected_body.contains(&prime_checkout) {
        bail!("cohort proposer spec references the mutable Prime checkout");
    }
    if normalized_proposer_spec(existing_body) != normalized_proposer_spec(expected_body) {
        bail!("immutable cohort proposer spec drift");
    }
    let pythonpath = proposer_env_path(existing_body, "PYTHONPATH")?;
    let operator_registry = proposer_env_path(existing_body, "JOPE_OPERATOR_REGISTRY")?;
    if !pythonpath.is_dir()
        || operator_registry != pythonpath.join("registry/operator_registry.json")
        || sha256_file(&operator_registry)?
            != sha256_file(
                &args
                    .proposal_pythonpath
                    .join("registry/operator_registry.json"),
            )?
        || sha256_file(&pythonpath.join("proposals/test_exact_context_proposals.py"))?
            != sha256_file(&args.proposal_generator)?
        || sha256_file(&pythonpath.join("proposals/test_exact_dl_generate.py"))?
            != sha256_file(&args.proposal_legacy_generator)?
        || sha256_file(&pythonpath.join("shared/test_exact_common.py"))?
            != sha256_file(&args.proposal_common)?
        || !pythonpath.join("jope/__init__.py").is_file()
        || !pythonpath.join("jope/policy/model_v3.py").is_file()
        || !pythonpath.join("jope/policy/decode.py").is_file()
    {
        bail!("immutable cohort proposer runtime dependency closure drift");
    }
    Ok(())
}

pub(super) fn immutable_json(path: &Path, value: &Value) -> Result<()> {
    if path.is_file() {
        let existing: Value = serde_json::from_slice(&fs::read(path)?)?;
        if existing != *value {
            bail!("immutable JSON artifact drift: {}", path.display());
        }
        return Ok(());
    }
    atomic_json(path, value)
}

pub(super) fn pointer_target(pointer_path: &Path, target: &str) -> PathBuf {
    let path = PathBuf::from(target);
    if path.is_absolute() {
        path
    } else {
        pointer_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(path)
    }
}
