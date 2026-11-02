use super::*;

pub(crate) fn checkpoint_inputs(args: &Args, status: &QueueStatus) -> Result<CheckpointInputs> {
    let active = status
        .active_checkpoint_sha256
        .as_deref()
        .context("queue keeper has no active checkpoint")?;
    validate_sha256(active)?;
    let checkpoint_pointer = if let Some(path) = &args.checkpoint_pointer {
        path.clone()
    } else {
        ["training/learning.json", "training/accepted.json"]
            .into_iter()
            .map(|relative| args.root.join(relative))
            .find(|path| path.is_file())
            .unwrap_or_else(|| args.proposals_pointer())
    };
    if pointer_checkpoint(&checkpoint_pointer)? != active {
        bail!("authoritative checkpoint pointer differs from queue keeper status");
    }
    let checkpoint_value: Value = serde_json::from_slice(&fs::read(&checkpoint_pointer)?)?;
    let declared_checkpoint = checkpoint_value
        .get("checkpoint")
        .or_else(|| checkpoint_value.pointer("/legacy_adopted/checkpoint"))
        .and_then(Value::as_str)
        .map(PathBuf::from);
    let checkpoint_artifact = [
        declared_checkpoint,
        Some(
            args.root
                .join("training/immutable/checkpoints")
                .join(format!("{active}.model.pt")),
        ),
        Some(paths::legacy_checkpoints_root().join(format!("{active}.model.pt"))),
    ]
    .into_iter()
    .flatten()
    .find(|path| path.is_file() && sha256_file(path).ok().as_deref() == Some(active))
    .context("active checkpoint artifact is absent or content-invalid")?;
    let proposals_pointer = args.proposals_pointer();
    if pointer_checkpoint(&proposals_pointer)? != active {
        bail!("proposal pointer is not from the active checkpoint");
    }
    Ok(CheckpointInputs {
        checkpoint_sha256: active.to_owned(),
        checkpoint_pointer,
        checkpoint_artifact,
        proposals_pointer,
    })
}

pub(crate) fn assert_checkpoint_unchanged(args: &Args, expected: &str) -> Result<()> {
    let status = load_queue_status(args)?;
    let inputs = checkpoint_inputs(args, &status)?;
    if inputs.checkpoint_sha256 != expected {
        bail!("checkpoint advanced while a micro-wave was being built");
    }
    Ok(())
}

pub(crate) fn pointer_checkpoint(path: &Path) -> Result<String> {
    let value: Value = serde_json::from_slice(&fs::read(path)?)
        .with_context(|| format!("parse checkpoint pointer {}", path.display()))?;
    let digest = value
        .get("checkpoint_sha256")
        .and_then(Value::as_str)
        .context("checkpoint pointer lacks checkpoint_sha256")?;
    validate_sha256(digest)?;
    Ok(digest.to_owned())
}

pub(crate) fn validate_sha256(value: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("invalid SHA-256 identity");
    }
    Ok(())
}

pub(crate) fn claimed_manifest_hashes(path: &Path) -> Result<BTreeSet<String>> {
    if !path.is_file() {
        return Ok(BTreeSet::new());
    }
    let connection = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    connection.busy_timeout(Duration::from_secs(60))?;
    let mut statement = connection
        .prepare("SELECT DISTINCT manifest_sha256 FROM claims WHERE manifest_sha256!=''")?;
    let mut hashes = statement
        .query_map([], |row| row.get(0))?
        .collect::<rusqlite::Result<BTreeSet<_>>>()?;
    drop(statement);
    let has_rejections: bool = connection.query_row(
        "SELECT EXISTS(
           SELECT 1 FROM sqlite_master
           WHERE type='table' AND name='rejected_manifests'
         )",
        [],
        |row| row.get(0),
    )?;
    if !has_rejections {
        bail!("rejection-aware keeper backlog lacks rejected_manifests");
    }
    let mut rejected =
        connection.prepare("SELECT manifest,manifest_sha256 FROM rejected_manifests")?;
    for row in rejected.query_map([], |row| {
        Ok((
            PathBuf::from(row.get::<_, String>(0)?),
            row.get::<_, Option<String>>(1)?,
        ))
    })? {
        let (manifest, recorded) = row?;
        let digest = match recorded.filter(|value| !value.is_empty()) {
            Some(digest) => digest,
            None if manifest.is_file() => sha256_file(&manifest)?,
            None => bail!(
                "rejected manifest lacks both a content hash and readable file: {}",
                manifest.display()
            ),
        };
        validate_sha256(&digest)?;
        hashes.insert(digest);
    }
    Ok(hashes)
}
