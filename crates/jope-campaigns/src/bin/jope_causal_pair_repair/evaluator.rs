use super::*;

pub(super) fn parse_evaluator_input(spec: &Path) -> Result<(PathBuf, String)> {
    let text = fs::read_to_string(spec)
        .with_context(|| format!("read original evaluator spec {}", spec.display()))?;
    let mut in_input = false;
    let mut name = None;
    let mut source = None;
    let mut sha256 = None;
    for line in text.lines().map(str::trim) {
        if line == "[[input]]" {
            if name.as_deref() == Some("evaluator") {
                break;
            }
            in_input = true;
            name = None;
            source = None;
            sha256 = None;
            continue;
        }
        if line.starts_with('[') {
            if name.as_deref() == Some("evaluator") {
                break;
            }
            in_input = false;
            continue;
        }
        if !in_input {
            continue;
        }
        let Some((key, raw)) = line.split_once('=') else {
            continue;
        };
        let value = || -> Result<String> {
            serde_json::from_str(raw.trim()).context("generated TOML string is not JSON-quoted")
        };
        match key.trim() {
            "name" => name = Some(value()?),
            "source" => source = Some(PathBuf::from(value()?)),
            "sha256" => sha256 = Some(value()?),
            _ => {}
        }
    }
    ensure!(
        name.as_deref() == Some("evaluator"),
        "spec lacks evaluator input"
    );
    Ok((
        source.context("evaluator input lacks source")?,
        sha256.context("evaluator input lacks sha256")?,
    ))
}

pub(super) fn evaluator_from_spec(
    root: &Path,
    spec: &Path,
    bundle_evaluators: &BTreeMap<String, PathBuf>,
) -> Result<EvaluatorBinding> {
    let (source, expected) = parse_evaluator_input(spec)?;
    if source.is_file() && sha256_file(&source)? == expected {
        return Ok(EvaluatorBinding {
            path: source,
            sha256: expected,
        });
    }
    let path = bundle_evaluators.get(&expected).cloned().with_context(|| {
        format!(
            "original evaluator {expected} is not retained under {}",
            root.join("runtime-bundles").display()
        )
    })?;
    Ok(EvaluatorBinding {
        path,
        sha256: expected,
    })
}

pub(super) fn evaluator_binding(
    root: &Path,
    identity: &PairIdentity,
    target: &TargetCandidate,
    bundle_evaluators: &BTreeMap<String, PathBuf>,
) -> Result<EvaluatorBinding> {
    let backlog = Connection::open_with_flags(
        root.join("ops/backlog.sqlite"),
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
    )?;
    backlog.busy_timeout(Duration::from_secs(60))?;
    let target_spec = backlog
        .query_row(
            "SELECT spec FROM claims WHERE manifest_sha256=?1
             ORDER BY attempts DESC,updated_at_ms DESC LIMIT 1",
            [&target.manifest_sha256],
            |row| row.get::<_, String>(0),
        )
        .ok();
    if let Some(spec) = target_spec {
        return evaluator_from_spec(root, Path::new(&spec), bundle_evaluators);
    }

    let corpus = Connection::open_with_flags(
        root.join("corpus/strict-test-exact.sqlite"),
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
    )?;
    corpus.busy_timeout(Duration::from_secs(60))?;
    for receipt_sha256 in &identity.receipt_sha256s {
        let jobs = {
            let mut statement = corpus.prepare(
                "SELECT job_id FROM ingested_jobs WHERE receipt_sha256=?1 ORDER BY job_id",
            )?;
            let jobs = statement
                .query_map([receipt_sha256], |row| row.get::<_, String>(0))?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            jobs
        };
        for job_id in jobs {
            let spec = backlog
                .query_row(
                    "SELECT spec FROM claims WHERE job_id=?1 LIMIT 1",
                    [&job_id],
                    |row| row.get::<_, String>(0),
                )
                .ok();
            if let Some(spec) = spec {
                return evaluator_from_spec(root, Path::new(&spec), bundle_evaluators);
            }
        }
    }
    bail!("no immutable original evaluator binding exists for the repair pair")
}

/// Content-addressed evaluators available for replay, keyed by sha256.
///
/// Bundles are the primary source, but a bundle directory can be gutted by
/// cleanup while the evaluator it carried is still referenced by historical
/// repair state -- which is exactly the live failure: evaluator 6bf93239 is
/// referenced by 40 pairs, survives nowhere under `runtime-bundles`, and has
/// accumulated 120,349 identical retry failures because this lookup could not
/// see it. `retain_evaluators` already hard-links every bundle evaluator into
/// `training/evaluator-vault/<digest>/corpus_eval` for precisely this reason,
/// and `jope_causal_evidence::retained_evaluator` already resolves against the
/// vault. This function was the odd one out.
///
/// The vault is content-addressed, so admitting it invents no provenance: the
/// digest IS the identity, and it is re-verified here rather than trusted from
/// the path. A bundle entry always wins over a vault entry for the same digest.
pub(super) fn bundle_evaluators(root: &Path) -> Result<BTreeMap<String, PathBuf>> {
    let mut output = BTreeMap::new();
    let bundles = root.join("runtime-bundles");
    if bundles.is_dir() {
        for entry in fs::read_dir(bundles)? {
            let evaluator = entry?.path().join("runtime/bin/corpus_eval");
            if evaluator.is_file() {
                output.entry(sha256_file(&evaluator)?).or_insert(evaluator);
            }
        }
    }
    let vault = root.join("training/evaluator-vault");
    if vault.is_dir() {
        for entry in fs::read_dir(vault)? {
            let dir = entry?.path();
            let evaluator = dir.join("corpus_eval");
            if !evaluator.is_file() {
                continue;
            }
            let digest = sha256_file(&evaluator)?;
            // The directory name is a claim; the bytes are the proof. Skip any
            // entry whose contents do not hash to the name it is filed under.
            if dir.file_name().and_then(|n| n.to_str()) != Some(digest.as_str()) {
                continue;
            }
            output.entry(digest).or_insert(evaluator);
        }
    }
    Ok(output)
}

pub(super) fn runtime_runner(args: &Args) -> Result<PathBuf> {
    if let Some(runner) = &args.runner {
        ensure!(runner.is_file(), "configured repair runner is absent");
        return Ok(runner.clone());
    }
    let executable = fs::canonicalize(std::env::current_exe()?)?;
    let bundle = executable
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .context("repair executor is not running from a runtime bundle")?;
    let runner = bundle.join("runtime/scripts/fweval_runner.sh");
    ensure!(runner.is_file(), "runtime-bundle repair runner is absent");
    Ok(runner)
}
