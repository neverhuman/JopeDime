use super::*;

#[derive(Debug)]
struct StillPublished {
    path: PathBuf,
    what: String,
    cause: String,
}

impl std::fmt::Display for StillPublished {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}; {} is still published (could not remove it: {})",
            self.what,
            self.path.display(),
            self.cause
        )
    }
}

/// Remove `published`, which this call's hard link created, after `err`. When it cannot be
/// removed the error carries [`StillPublished`], so callers know the file is theirs and remains.
fn unpublish(err: std::io::Error, what: String, published: &Path) -> anyhow::Error {
    match fs::remove_file(published) {
        Ok(()) => anyhow::Error::new(err).context(what),
        Err(e) => anyhow::Error::new(err).context(StillPublished {
            path: published.to_path_buf(),
            what,
            cause: e.to_string(),
        }),
    }
}

/// Move `partial` to `target` without ever replacing an existing file: hard-link (which fails
/// with `AlreadyExists` if `target` appeared meanwhile), then drop the temporary name. An error
/// means `target` was not published by this call, unless the error says it is still published.
fn publish_no_replace(partial: &Path, target: &Path) -> Result<()> {
    match fs::hard_link(partial, target) {
        Ok(()) => {
            if let Err(err) = fs::remove_file(partial) {
                return Err(unpublish(
                    err,
                    format!("remove {}; that partial file remains", partial.display()),
                    target,
                ));
            }
            Ok(())
        }
        Err(err) => {
            let _ = fs::remove_file(partial);
            if err.kind() == std::io::ErrorKind::AlreadyExists {
                bail!("refusing to overwrite {}", target.display());
            }
            Err(err).with_context(|| format!("publish {}", target.display()))
        }
    }
}

fn verify_round_trip(tensors: &[Named], path: &Path) -> Result<()> {
    let expected: BTreeMap<&str, &Named> = tensors.iter().map(|t| (t.name.as_str(), t)).collect();

    let reread: BTreeMap<String, Tensor> = Tensor::read_safetensors(path)?.into_iter().collect();
    ensure!(
        reread
            .keys()
            .map(String::as_str)
            .eq(expected.keys().copied()),
        "tch reload: tensor name set differs"
    );
    for (name, tensor) in &reread {
        let source = expected[name.as_str()];
        ensure!(
            tensor.kind() == source.tensor.kind() && tensor.size() == source.tensor.size(),
            "tch reload: {name} dtype or shape differs"
        );
        ensure!(
            raw_bytes(tensor)? == source.bytes,
            "tch reload: {name} bytes differ"
        );
    }

    let parsed = parse_safetensors(path)?;
    ensure!(
        parsed
            .keys()
            .map(String::as_str)
            .eq(expected.keys().copied()),
        "header parse: tensor name set differs"
    );
    for (name, (dtype, shape, bytes)) in &parsed {
        let source = expected[name.as_str()];
        ensure!(
            dtype == dtype_name(source.tensor.kind())?
                && *shape == source.tensor.size()
                && *bytes == source.bytes,
            "header parse: {name} differs"
        );
    }
    Ok(())
}

/// Write `tensors` to `output` (refusing to overwrite), re-read it through tch and through
/// `parse_safetensors`, requiring a bit-exact match of every tensor, then publish it with no
/// replacement (ot-export `lib.rs:255-294`). An error means `output` is not published, unless it
/// carries [`StillPublished`] naming it.
fn write_and_verify(tensors: &[Named], output: &Path) -> Result<Value> {
    ensure!(absent(output), "refusing to overwrite {}", output.display());
    let parent = output.parent().context("output has no parent")?;
    fs::create_dir_all(parent)?;
    // Opened before the partial is written, so its fsync cannot be refused after publication.
    let directory = fs::File::open(parent).with_context(|| format!("open {}", parent.display()))?;
    let partial = weights_partial(output)?;
    // Never remove or write through an entry this call did not create (also pre-checked by
    // `export_on_cpu` before any write).
    ensure!(
        absent(&partial),
        "refusing to replace an existing temporary {}",
        partial.display()
    );
    let named: Vec<(&str, &Tensor)> = tensors
        .iter()
        .map(|t| (t.name.as_str(), &t.tensor))
        .collect();
    if let Err(err) = Tensor::write_safetensors(&named, &partial) {
        let _ = fs::remove_file(&partial);
        return Err(err).with_context(|| format!("write {}", partial.display()));
    }
    if let Err(err) = fs::File::open(&partial).and_then(|f| f.sync_all()) {
        let _ = fs::remove_file(&partial);
        return Err(err).with_context(|| format!("sync {}", partial.display()));
    }
    if let Err(err) = verify_round_trip(tensors, &partial) {
        let _ = fs::remove_file(&partial);
        return Err(err);
    }
    // Describe the inode before it is published: nothing but the fsync may fail afterwards.
    let described =
        file_hash(&partial).and_then(|sha256| Ok((sha256, fs::metadata(&partial)?.len())));
    let (sha256, file_bytes) = match described {
        Ok(described) => described,
        Err(err) => {
            let _ = fs::remove_file(&partial);
            return Err(err);
        }
    };
    publish_no_replace(&partial, output)?;
    if let Err(err) = directory.sync_all() {
        // This call's hard link created `output`: never leave it published without a receipt.
        return Err(unpublish(err, format!("sync {}", parent.display()), output));
    }
    Ok(json!({
        "path": output,
        "sha256": sha256,
        "file_bytes": file_bytes,
        "round_trip": {
            "readers": ["tch::Tensor::read_safetensors", "independent header parse"],
            "compared": ["name set", "dtype", "shape", "raw bytes"],
            "bit_exact": true,
        },
    }))
}

/// Pretty JSON plus a newline, published with no replacement and its directory fsynced, as
/// `write_and_verify` does for the weights; returns its sha256 (ot-export `lib.rs:541-550`).
/// An error means no receipt exists at `path`, unless it carries [`StillPublished`] naming it.
fn write_receipt(path: &Path, receipt: &Value) -> Result<String> {
    ensure!(absent(path), "refusing to overwrite {}", path.display());
    let parent = path.parent().context("receipt has no parent")?;
    // Opened before any write, so its fsync cannot be refused after publication.
    let directory = fs::File::open(parent).with_context(|| format!("open {}", parent.display()))?;
    let mut bytes = serde_json::to_vec_pretty(receipt)?;
    bytes.push(b'\n');
    let partial = receipt_partial(path);
    // Created exclusively: an existing entry (a stale file, a symlink) is neither followed,
    // truncated nor removed; only what this call created is cleaned up. `export_on_cpu` also
    // refuses such an entry before any write.
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&partial)
        .with_context(|| format!("write {}", partial.display()))?;
    if let Err(err) = file.write_all(&bytes).and_then(|()| file.sync_all()) {
        let _ = fs::remove_file(&partial);
        return Err(err).with_context(|| format!("write {}", partial.display()));
    }
    drop(file);
    publish_no_replace(&partial, path)?;
    if let Err(err) = directory.sync_all() {
        // This call's hard link created `path`.
        return Err(unpublish(err, format!("sync {}", parent.display()), path));
    }
    Ok(hash(&bytes))
}

fn runtime_section() -> Value {
    let libtorch = std::env::var_os("LIBTORCH").map(PathBuf::from);
    let build_version = libtorch
        .as_ref()
        .and_then(|p| fs::read_to_string(p.join("build-version")).ok())
        .map(|s| s.trim().to_string());
    json!({
        "tool": "jope-parallel export-policy",
        "tool_version": env!("CARGO_PKG_VERSION"),
        "tch": "0.26.0",
        "libtorch": libtorch,
        "libtorch_build_version": build_version,
        "device": "cpu",
        "cuda_visible_devices": "",
    })
}

/// Export one checkpoint and write its receipt; returns a summary. CPU only.
pub fn export(request: &ExportRequest) -> Result<Value> {
    require_cpu_only(std::env::var("CUDA_VISIBLE_DEVICES").ok().as_deref())?;
    export_on_cpu(request)
}

/// The nearest ancestor of `path` (itself included) that exists, found without following links.
/// A dangling link above `path` is refused: whatever it points at could be created later, under
/// `/tmp` or anywhere else, and `create_dir_all` would then write through it. A dangling link at
/// `path` itself is skipped here; the overwrite checks refuse it as an existing entry.
fn nearest_existing(path: &Path) -> Result<&Path> {
    for ancestor in path.ancestors() {
        match fs::symlink_metadata(ancestor) {
            Ok(meta) if meta.file_type().is_symlink() && !ancestor.exists() => ensure!(
                ancestor == path,
                "export-policy writes only to durable absolute paths, never /tmp: {} passes \
                 through the dangling link {}",
                path.display(),
                ancestor.display()
            ),
            Ok(_) => return Ok(ancestor),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e).with_context(|| format!("inspect {}", ancestor.display())),
        }
    }
    bail!("{} has no existing ancestor", path.display())
}

/// The temporary name `write_and_verify` writes the weights to, beside the published name.
fn weights_partial(output: &Path) -> Result<PathBuf> {
    let parent = output.parent().context("output has no parent")?;
    let file_name = output
        .file_name()
        .context("output has no file name")?
        .to_string_lossy();
    // tch picks the writer by extension, so keep `.safetensors` on the temporary name.
    Ok(parent.join(format!(".partial-{}-{file_name}", std::process::id())))
}

/// The temporary name `write_receipt` writes the receipt to.
fn receipt_partial(path: &Path) -> PathBuf {
    path.with_extension(format!("partial-{}", std::process::id()))
}

/// After `write_receipt` failed for `receipt`: keep the published `weights` only when this
/// call's own receipt could not be undone ([`StillPublished`] naming `receipt`); otherwise remove
/// them, since weights without their binding receipt would refuse every retry.
fn after_receipt_failure(err: anyhow::Error, receipt: &Path, weights: &Path) -> anyhow::Error {
    if err
        .downcast_ref::<StillPublished>()
        .is_some_and(|s| s.path == receipt)
    {
        return err.context(format!(
            "receipt {} is still published after a failed write; kept {}",
            receipt.display(),
            weights.display()
        ));
    }
    let message = match fs::remove_file(weights) {
        Err(e) => format!(
            "receipt not written; could not remove the unreceipted {}: {e}",
            weights.display()
        ),
        Ok(()) => {
            let synced = weights
                .parent()
                .context("weights have no parent")
                .and_then(|d| Ok(fs::File::open(d)?.sync_all()?));
            match synced {
                Ok(()) => format!(
                    "receipt not written; removed the unreceipted {}",
                    weights.display()
                ),
                Err(e) => format!(
                    "receipt not written; removed the unreceipted {} but could not fsync its \
                     directory: {e:#}",
                    weights.display()
                ),
            }
        }
    };
    err.context(message)
}

/// Refuse a relative path, a `..` component, or a path under `/tmp`, lexically or once the
/// symlinks of its nearest existing ancestor are resolved.
fn require_durable_path(path: &Path) -> Result<()> {
    ensure!(
        path.is_absolute()
            && !path.components().any(|c| c == Component::ParentDir)
            && !path.starts_with("/tmp"),
        "export-policy writes only to durable absolute paths, never /tmp: {}",
        path.display()
    );
    let existing = nearest_existing(path)?;
    let resolved =
        fs::canonicalize(existing).with_context(|| format!("resolve {}", existing.display()))?;
    // `/tmp` itself may be a link on some hosts; compare with what it resolves to as well.
    let tmp = fs::canonicalize("/tmp").unwrap_or_else(|_| PathBuf::from("/tmp"));
    ensure!(
        !resolved.starts_with("/tmp") && !resolved.starts_with(&tmp),
        "export-policy writes only to durable absolute paths, never /tmp: {} resolves to {}",
        path.display(),
        resolved.display()
    );
    Ok(())
}

/// [`export`] after the CPU-only check. Every refusal precedes the first write, and a receipt
/// that still fails to be written removes the just-published policy file again, unless the
/// receipt itself remains published ([`StillPublished`]), in which case both stay.
fn export_on_cpu(request: &ExportRequest) -> Result<Value> {
    for path in [&request.output, &request.receipt] {
        require_durable_path(path)?;
    }
    // An output directory that is itself a dangling link would be written through by
    // `create_dir_all`; refuse it before any write.
    ensure!(
        request.output.exists() || absent(&request.output),
        "export-policy writes only to durable absolute paths, never /tmp: {} is a dangling link",
        request.output.display()
    );
    let output = request.output.join(POLICY_OUTPUT);
    // The receipt is written after the policy file, so every way it could fail to be written
    // that can be seen without writing is refused here, before the policy file exists.
    ensure!(
        request.receipt.parent().is_some_and(Path::is_dir),
        "receipt directory does not exist: {}",
        request.receipt.display()
    );
    ensure!(
        request.receipt != request.output && request.receipt != output,
        "receipt must not name the policy output: {}",
        request.receipt.display()
    );
    // The same check on resolved paths: through symlinks, and for an output directory not created
    // yet (resolved from its nearest existing ancestor; `..` was refused above). The receipt may be
    // neither the output directory, nor a directory that would be created above it, nor the
    // policy file. The receipt directory exists (checked above).
    let receipt_parent = request.receipt.parent().context("receipt has no parent")?;
    let resolved_receipt = fs::canonicalize(receipt_parent)
        .with_context(|| format!("resolve {}", receipt_parent.display()))?
        .join(
            request
                .receipt
                .file_name()
                .context("receipt has no file name")?,
        );
    let base = nearest_existing(&request.output)?;
    let resolved_output = fs::canonicalize(base)
        .with_context(|| format!("resolve {}", base.display()))?
        .join(request.output.strip_prefix(base)?);
    ensure!(
        !resolved_output.starts_with(&resolved_receipt)
            && resolved_receipt != resolved_output.join(POLICY_OUTPUT),
        "receipt must not name the policy output: {}",
        request.receipt.display()
    );
    for path in [&request.receipt, &output] {
        ensure!(absent(path), "refusing to overwrite {}", path.display());
    }
    // A stale temporary at either name would stop the export part way; refuse it now instead.
    for partial in [weights_partial(&output)?, receipt_partial(&request.receipt)] {
        ensure!(
            absent(&partial),
            "refusing to replace an existing temporary {}",
            partial.display()
        );
    }
    // Each directory must open for its fsync after publication; an existing one is checked here.
    for directory in [request.receipt.parent(), Some(request.output.as_path())]
        .into_iter()
        .flatten()
        .filter(|d| d.exists())
    {
        fs::File::open(directory).with_context(|| {
            format!(
                "directory cannot be opened for its fsync: {}",
                directory.display()
            )
        })?;
    }
    let child = load_checkpoint(&request.checkpoint, Some(&request.expect_state_sha256))?;
    let parent = request
        .parent
        .as_deref()
        .map(|p| load_checkpoint(p, None))
        .transpose()?;
    let relative = parent
        .as_ref()
        .map(|p| relative_change(p, &child))
        .transpose()?;
    let entries = tensor_entries(&child.tensors)?;
    let total: usize = child.tensors.iter().map(|t| t.bytes.len()).sum();
    let inventory_sha256 = hash(&serde_json::to_vec(&entries)?);
    let model_path = child.directory.join("model.ot");
    let model_bytes = fs::metadata(&model_path)?.len();
    let accepted_identity = child.state_sha256 == ACCEPTED_STATE_SHA256;
    // Everything above only reads. The policy file is written first, then the receipt.
    let written = write_and_verify(&child.tensors, &output)?;
    let receipt = json!({
        "schema": RECEIPT_SCHEMA,
        "runtime": runtime_section(),
        "checkpoint": {
            "directory": child.directory,
            "state_sha256": child.state_sha256,
            "state_bytes": child.state_bytes,
            "expected_state_sha256": request.expect_state_sha256,
            "accepted_identity": accepted_identity,
            "refused_identities": [REJECTED_STATE_SHA256],
            "refused_steps": [REJECTED_STEP],
            "step": child.step,
            "model_sha256": child.model_sha256,
            "config_sha256": child.config_sha256,
            "config": child.config_value,
            "verified_bindings": ["model_sha256", "config_sha256", "autocast_cache_lifetime", "parameter_sha256"],
            "carried_bindings": {
                "corpus_sha256": child.state["corpus_sha256"],
                "vocabulary_sha256": child.state["vocabulary_sha256"],
            },
            "parameter_sha256": child.state["parameter_sha256"],
            "parameter_sha256_recomputed": child.parameter_sha256,
        },
        "policy": {
            "source": {"path": model_path, "sha256": child.model_sha256, "file_bytes": model_bytes},
            "output": written,
            "tensor_count": child.tensors.len(),
            "total_tensor_bytes": total,
            "inventory_sha256": inventory_sha256,
            "tensors": entries,
            "architecture": "neural::Policy::new names and shapes; F32; finite",
        },
        "relative_change": relative,
    });
    let receipt_sha256 = write_receipt(&request.receipt, &receipt)
        .map_err(|err| after_receipt_failure(err, &request.receipt, &output))?;
    Ok(json!({
        "receipt": request.receipt,
        "receipt_sha256": receipt_sha256,
        "state_sha256": child.state_sha256,
        "accepted_identity": accepted_identity,
        "policy_sha256": written["sha256"],
        "policy_file_bytes": written["file_bytes"],
        "tensor_count": child.tensors.len(),
        "total_tensor_bytes": total,
        "parameter_sha256": child.parameter_sha256,
        "relative_change_total": relative.as_ref().map_or(Value::Null, |r| r["total"].clone()),
    }))
}
