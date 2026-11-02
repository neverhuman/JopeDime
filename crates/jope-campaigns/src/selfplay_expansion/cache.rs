use super::prelude::*;
use super::{
    inventory::{cache_path, compare_mirror},
    live::SIDECAR_NAMES,
    model::{
        load_v1_quarantine, CacheManifest, ExpansionPaths, FileLock, InventoryEntry,
        ENCODER_PROTOCOL, OPTIONAL_ENCODINGS, REQUIRED_ENCODINGS,
    },
    sidecars::validate_sidecar,
};

pub(super) fn copy_reflink_or_copy(source: &Path, destination: &Path) -> Result<()> {
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)?;
    }
    let input = File::open(source)?;
    let output = OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(input.metadata()?.permissions().mode())
        .open(destination)?;
    // Linux FICLONE; unsupported filesystems return an error and fall back to
    // buffered std::fs::copy.
    let cloned =
        // SAFETY: FICLONE (_IOW(0x94, 9, int)) takes the source fd by value, not a pointer;
        // `output` and `input` are open Files that outlive this call.
        unsafe { libc::ioctl(output.as_raw_fd(), 0x4004_9409_u64, input.as_raw_fd()) } == 0;
    drop(output);
    if !cloned {
        fs::copy(source, destination)?;
        fs::set_permissions(destination, input.metadata()?.permissions())?;
    }
    Ok(())
}

pub(super) fn encoder_protocol_sha256() -> Result<String> {
    canonical_json_sha256(&json!({"protocol": ENCODER_PROTOCOL, "train_only": true}))
}

/// The numerical backend this process will hand the encoder, recorded into
/// every new cache manifest as `encoder_backend`.
///
/// `CUDA_VISIBLE_DEVICES` set to an empty string is how every caller pins the
/// encoder to CPU, so an empty-or-unset value means the CUDA build will use a
/// GPU if one is free. This is an ATTESTATION of intent at manifest-writing
/// time, not proof of what the kernel did -- the encoder can still fall back
/// to CPU under GPU-memory pressure. Proving the executed backend requires the
/// encoder itself to report it; until then this narrows an unknown to a
/// declared expectation, and a mismatch between the two is exactly the audit
/// this field makes possible.
pub(super) fn observed_encoder_backend() -> Option<String> {
    match std::env::var("CUDA_VISIBLE_DEVICES") {
        Ok(value) if value.trim().is_empty() => Some("cpu".to_owned()),
        Ok(_) => Some("cuda-requested".to_owned()),
        Err(_) => Some("cuda-requested".to_owned()),
    }
}

pub fn validate_cache(path: &Path, expected: Option<&InventoryEntry>) -> Result<CacheManifest> {
    ensure!(
        !path
            .components()
            .any(|component| component.as_os_str() == "incident-quarantine"),
        "quarantined V1 caches cannot be validated for consumption"
    );
    let manifest_path = path.join("cache-manifest.v1.json");
    let manifest: CacheManifest = serde_json::from_value(read_json(&manifest_path)?)?;
    ensure!(
        manifest.schema == "JopeExpandedEncodingCache.v1"
            && manifest.immutable
            && manifest.encoder_protocol == ENCODER_PROTOCOL
            && manifest.train_only_encoding,
        "expanded cache envelope is invalid: {}",
        path.display()
    );
    ensure!(
        manifest.encoder_protocol_sha256 == encoder_protocol_sha256()?,
        "expanded cache encoder protocol drift"
    );
    if let Some(entry) = expected {
        ensure!(
            manifest.task == entry.task
                && manifest.dataset_id == entry.dataset_id
                && manifest.dataset_hash == entry.dataset_hash
                && manifest.train_sha256 == entry.train_sha256
                && manifest.test_sha256 == entry.test_sha256
                && manifest.source_family == entry.source_family,
            "expanded cache identity drift: {}",
            path.display()
        );
    }
    for required in [
        "train.csv",
        "test.csv",
        "meta.json",
        "signature_manifest.json",
        "hyperion3train128dim.vs",
        "train3.vs",
        "train_target.vs",
    ] {
        ensure!(
            manifest.files.contains_key(required),
            "expanded cache inventory is missing {required}"
        );
    }
    for (name, digest) in &manifest.files {
        let file = path.join(name);
        ensure!(
            file.is_file() && sha256_file(&file)? == *digest,
            "expanded cache content drift: {}",
            file.display()
        );
    }
    manifest.cache_identity().validate()?;
    Ok(manifest)
}

pub(super) fn is_legacy_cache(path: &Path) -> Result<bool> {
    let value = read_json(&path.join("cache-manifest.v1.json"))?;
    Ok(
        value["schema"].as_str() == Some("JopeExpandedEncodingCache.v1")
            && value["immutable"].as_bool() == Some(true)
            && value["train_only_encoding"].as_bool() == Some(true)
            && value["dataset_id"].is_null()
            && value["source_family"].is_null()
            && value["train_sha256"].is_null()
            && value["test_sha256"].is_null(),
    )
}

/// Upgrade the first-generation sidecar-only cache envelope in place.
///
/// The old encoder produced trustworthy sidecars, but its manifest omitted
/// dataset identity, source CSVs, and the signature manifest required by the
/// current activation contract. Every old manifest/file hash is checked before
/// any new file is written; the only added data files come from the audited
/// authoritative source.
pub(super) fn repair_legacy_cache(path: &Path, entry: &InventoryEntry) -> Result<CacheManifest> {
    ensure!(
        path.is_dir(),
        "legacy cache directory is absent: {}",
        path.display()
    );
    ensure!(
        !path.symlink_metadata()?.file_type().is_symlink(),
        "legacy cache directory must not be a symlink: {}",
        path.display()
    );
    let legacy = read_json(&path.join("cache-manifest.v1.json"))?;
    ensure!(
        legacy["schema"].as_str() == Some("JopeExpandedEncodingCache.v1")
            && legacy["immutable"].as_bool() == Some(true)
            && legacy["train_only_encoding"].as_bool() == Some(true),
        "existing cache is not a recognized legacy envelope: {}",
        path.display()
    );
    ensure!(
        legacy["task"].as_str() == Some(entry.task.as_str())
            && legacy["dataset_hash"].as_str() == Some(entry.dataset_hash.as_str())
            && legacy["source"].as_str() == entry.source.to_str(),
        "legacy cache identity drift: {}",
        path.display()
    );
    ensure!(
        legacy["encoder_protocol"].as_str() == Some(ENCODER_PROTOCOL),
        "legacy cache encoder protocol drift: {}",
        path.display()
    );
    let legacy_files = legacy["files"]
        .as_object()
        .context("legacy cache manifest lacks files")?;
    for (name, digest) in legacy_files {
        let expected = digest
            .as_str()
            .with_context(|| format!("legacy cache file hash is invalid: {name}"))?;
        let file = path.join(name);
        ensure!(
            file.is_file() && sha256_file(&file)? == expected,
            "legacy cache content drift: {}",
            file.display()
        );
    }
    for name in REQUIRED_ENCODINGS {
        validate_sidecar(&path.join(name), name)
            .with_context(|| format!("legacy cache sidecar validation failed: {name}"))?;
    }
    for name in SIDECAR_NAMES {
        let file = path.join(name);
        if file.is_file() {
            validate_sidecar(&file, name)
                .with_context(|| format!("legacy cache sidecar validation failed: {name}"))?;
        }
    }
    ensure!(
        sha256_file(&entry.source.join("train.csv"))? == entry.train_sha256
            && sha256_file(&entry.source.join("test.csv"))? == entry.test_sha256
            && sha256_file(&entry.source.join("meta.json"))? == entry.meta_sha256,
        "authoritative source changed during legacy cache repair"
    );

    for name in ["train.csv", "test.csv", "meta.json"] {
        let destination = path.join(name);
        let expected = match name {
            "train.csv" => &entry.train_sha256,
            "test.csv" => &entry.test_sha256,
            "meta.json" => &entry.meta_sha256,
            _ => unreachable!(),
        };
        if destination.is_file() {
            ensure!(
                sha256_file(&destination)? == *expected,
                "legacy cache data drift: {}",
                destination.display()
            );
            continue;
        }
        let temporary = path.join(format!(
            ".{}.repair-{}-{}",
            name,
            std::process::id(),
            now_ms()
        ));
        copy_reflink_or_copy(&entry.source.join(name), &temporary)?;
        let result = (|| -> Result<()> {
            ensure!(
                sha256_file(&temporary)? == *expected,
                "repaired cache data hash drift: {}",
                temporary.display()
            );
            fs::rename(&temporary, &destination)?;
            Ok(())
        })();
        if result.is_err() {
            cleanup_staging(&temporary);
        }
        result?;
    }

    let encoding_hashes = REQUIRED_ENCODINGS
        .into_iter()
        .chain(OPTIONAL_ENCODINGS)
        .filter(|name| path.join(name).is_file())
        .map(|name| Ok((name.to_owned(), sha256_file(&path.join(name))?)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    let signature = json!({
        "schema": "JopeExpandedSignatureManifest.v1",
        "task": entry.task,
        "dataset_id": entry.dataset_id,
        "train_only": true,
        "encoder_protocol": ENCODER_PROTOCOL,
        "encoder_protocol_sha256": encoder_protocol_sha256()?,
        "files": encoding_hashes,
    });
    let signature_path = path.join("signature_manifest.json");
    if signature_path.is_file() {
        ensure!(
            read_json(&signature_path)? == signature,
            "legacy cache signature manifest drift: {}",
            signature_path.display()
        );
    } else {
        atomic_json(&signature_path, &signature)?;
    }

    let files = fs::read_dir(path)?
        .filter_map(std::result::Result::ok)
        .filter(|item| {
            item.file_type().is_ok_and(|kind| kind.is_file())
                && item.file_name() != "cache-manifest.v1.json"
        })
        .map(|item| {
            let name = item.file_name().to_string_lossy().to_string();
            Ok((name, sha256_file(&item.path())?))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    ensure!(
        files["train.csv"] == entry.train_sha256
            && files["test.csv"] == entry.test_sha256
            && files["meta.json"] == entry.meta_sha256,
        "repaired cache data inventory drift"
    );
    let manifest = CacheManifest {
        schema: "JopeExpandedEncodingCache.v1".to_owned(),
        immutable: true,
        task: entry.task.clone(),
        dataset_id: entry.dataset_id.clone(),
        dataset_hash: entry.dataset_hash.clone(),
        source: entry.source.clone(),
        source_family: entry.source_family.clone(),
        train_sha256: entry.train_sha256.clone(),
        test_sha256: entry.test_sha256.clone(),
        encoder_protocol: ENCODER_PROTOCOL.to_owned(),
        encoder_protocol_sha256: encoder_protocol_sha256()?,
        train_only_encoding: true,
        encoder_backend: observed_encoder_backend(),
        files,
        created_at_ms: now_ms(),
    };
    manifest.cache_identity().validate()?;
    atomic_json(
        &path.join("cache-manifest.v1.json"),
        &serde_json::to_value(&manifest)?,
    )?;
    validate_cache(path, Some(entry))
}

pub(super) fn verify_native_encoder(path: &Path) -> Result<()> {
    ensure!(path.is_file(), "Rust encoder is absent: {}", path.display());
    let mut header = [0_u8; 4];
    File::open(path)?.read_exact(&mut header)?;
    ensure!(
        header == [0x7f, b'E', b'L', b'F'],
        "encoder must be a native Rust binary (ELF), never a Python/script entrypoint"
    );
    Ok(())
}

pub(super) fn run_native_encoder(
    encoder: &Path,
    source: &Path,
    destination: &Path,
    task: &str,
) -> Result<()> {
    verify_native_encoder(encoder)?;
    let output = Command::new(encoder)
        .args([
            "--train-csv",
            source.join("train.csv").to_string_lossy().as_ref(),
            "--output-dir",
            destination.to_string_lossy().as_ref(),
            "--task",
            task,
            "--train-only",
        ])
        .output()
        .with_context(|| format!("run native Hyperion encoder {}", encoder.display()))?;
    ensure!(
        output.status.success(),
        "native Hyperion encoder failed: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    Ok(())
}

pub(super) fn cleanup_staging(path: &Path) {
    if path.is_dir() {
        let _ = fs::remove_dir_all(path);
    } else if path.exists() {
        let _ = fs::remove_file(path);
    }
}

pub fn materialize_cache(
    paths: &ExpansionPaths,
    entry: &InventoryEntry,
    rust_encoder: Option<&Path>,
) -> Result<CacheManifest> {
    if let Some(quarantine) = load_v1_quarantine(paths)? {
        quarantine.require_operation_allowed(&entry.dataset_id, "V1 cache materialization")?;
    }
    entry.dataset_identity().validate()?;
    // Recheck immediately before materialization; an audit result is not a
    // permission to ignore later mirror drift.
    compare_mirror(paths, &entry.task, &entry.dataset_id, &entry.source)?;
    ensure!(
        sha256_file(&entry.source.join("train.csv"))? == entry.train_sha256
            && sha256_file(&entry.source.join("test.csv"))? == entry.test_sha256,
        "authoritative dataset changed after discovery"
    );
    let final_path = cache_path(paths, &entry.task, &entry.dataset_id);
    let lock_path = final_path
        .parent()
        .context("cache path has no parent")?
        .join(".locks")
        .join(format!("{}.lock", entry.dataset_id));
    let _lock = FileLock::exclusive(&lock_path)?;
    if final_path.is_dir() {
        match validate_cache(&final_path, Some(entry)) {
            Ok(manifest) => return Ok(manifest),
            Err(validation_error) => {
                if is_legacy_cache(&final_path)? {
                    return repair_legacy_cache(&final_path, entry).with_context(|| {
                        format!(
                            "upgrade legacy expanded cache {} after validation failure: {validation_error}",
                            final_path.display()
                        )
                    });
                }
                return Err(validation_error);
            }
        }
    }
    let staging = final_path.with_file_name(format!(
        ".{}.tmp-{}-{}",
        entry.dataset_id,
        std::process::id(),
        now_ms()
    ));
    fs::create_dir_all(&staging)?;
    let result = (|| -> Result<CacheManifest> {
        for name in ["train.csv", "test.csv", "meta.json"] {
            copy_reflink_or_copy(&entry.source.join(name), &staging.join(name))?;
        }
        for name in REQUIRED_ENCODINGS.into_iter().chain(OPTIONAL_ENCODINGS) {
            let source_file = entry.source.join(name);
            if source_file.is_file() {
                copy_reflink_or_copy(&source_file, &staging.join(name))?;
            }
        }
        let missing = REQUIRED_ENCODINGS
            .into_iter()
            .filter(|name| !staging.join(name).is_file())
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            let encoder = rust_encoder.with_context(|| {
                format!(
                    "{}:{} missing {}; no native Rust Hyperion encoder supplied",
                    entry.task,
                    entry.dataset_id,
                    missing.join(",")
                )
            })?;
            run_native_encoder(encoder, &entry.source, &staging, &entry.task)?;
        }
        for name in REQUIRED_ENCODINGS {
            ensure!(
                staging.join(name).is_file(),
                "native encoder did not produce {name}"
            );
        }
        let encoding_hashes = REQUIRED_ENCODINGS
            .into_iter()
            .chain(OPTIONAL_ENCODINGS)
            .filter(|name| staging.join(name).is_file())
            .map(|name| Ok((name.to_owned(), sha256_file(&staging.join(name))?)))
            .collect::<Result<BTreeMap<_, _>>>()?;
        atomic_json(
            &staging.join("signature_manifest.json"),
            &json!({
                "schema": "JopeExpandedSignatureManifest.v1",
                "task": entry.task,
                "dataset_id": entry.dataset_id,
                "train_only": true,
                "encoder_protocol": ENCODER_PROTOCOL,
                "encoder_protocol_sha256": encoder_protocol_sha256()?,
                "files": encoding_hashes,
            }),
        )?;
        let files = fs::read_dir(&staging)?
            .filter_map(std::result::Result::ok)
            .filter(|item| item.file_type().is_ok_and(|kind| kind.is_file()))
            .map(|item| {
                let name = item.file_name().to_string_lossy().to_string();
                Ok((name, sha256_file(&item.path())?))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        ensure!(
            files["train.csv"] == entry.train_sha256
                && files["test.csv"] == entry.test_sha256
                && files["meta.json"] == entry.meta_sha256,
            "staged data hash validation failed"
        );
        let manifest = CacheManifest {
            schema: "JopeExpandedEncodingCache.v1".to_owned(),
            immutable: true,
            task: entry.task.clone(),
            dataset_id: entry.dataset_id.clone(),
            dataset_hash: entry.dataset_hash.clone(),
            source: entry.source.clone(),
            source_family: entry.source_family.clone(),
            train_sha256: entry.train_sha256.clone(),
            test_sha256: entry.test_sha256.clone(),
            encoder_protocol: ENCODER_PROTOCOL.to_owned(),
            encoder_protocol_sha256: encoder_protocol_sha256()?,
            train_only_encoding: true,
            encoder_backend: observed_encoder_backend(),
            files,
            created_at_ms: now_ms(),
        };
        manifest.cache_identity().validate()?;
        atomic_json(
            &staging.join("cache-manifest.v1.json"),
            &serde_json::to_value(&manifest)?,
        )?;
        validate_cache(&staging, Some(entry))?;
        fs::create_dir_all(final_path.parent().context("cache path has no parent")?)?;
        fs::rename(&staging, &final_path)?;
        validate_cache(&final_path, Some(entry))
    })();
    if result.is_err() {
        cleanup_staging(&staging);
    }
    result
}
