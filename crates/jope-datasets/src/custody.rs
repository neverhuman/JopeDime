//! Preserve every real or unresolved file in a complete `remote_super` census. Dataset catalog
//! rows remain separate; this immutable path catalog also covers incomplete datasets and research
//! outputs that cannot be represented by a train/test row.

use crate::hash::digest_file;
use crate::store::Store;
use anyhow::{anyhow, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::os::unix::ffi::OsStringExt;
use std::path::{Component, Path, PathBuf};

const PREFIX: &str = "custody/quant";

#[derive(Debug, Deserialize)]
struct InventoryRow {
    source_root: Option<String>,
    path_hex: Option<String>,
    #[serde(rename = "type")]
    kind: Option<String>,
    classification: Option<String>,
    size: Option<u64>,
    sha256: Option<String>,
    error: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CatalogEntry {
    pub path_hex: String,
    pub sha256: String,
    pub bytes: u64,
    pub classification: String,
    pub s3_key: String,
    pub s3_version: Option<String>,
}

fn source_path(root: &Path, path_hex: &str) -> Result<PathBuf> {
    let bytes = hex::decode(path_hex).context("invalid inventory path_hex")?;
    let relative = PathBuf::from(OsString::from_vec(bytes));
    ensure!(
        relative
            .components()
            .all(|part| matches!(part, Component::Normal(_))),
        "inventory path is not relative"
    );
    Ok(root.join(relative))
}

fn quant_file(root: &Path, row: &InventoryRow) -> Result<Option<(PathBuf, CatalogEntry)>> {
    ensure!(
        row.error.is_none(),
        "inventory has unreadable or changing entries"
    );
    let Some(path_hex) = &row.path_hex else {
        return Ok(None);
    };
    let path = source_path(root, path_hex)?;
    let relative = path.strip_prefix(root)?;
    if relative
        .components()
        .next()
        .is_none_or(|part| part.as_os_str() != "quant")
        || row.kind.as_deref() != Some("file")
    {
        return Ok(None);
    }
    ensure!(
        row.source_root.as_deref() == Some(root.to_string_lossy().as_ref()),
        "inventory source root differs from requested root"
    );
    let class = row.classification.as_deref().unwrap_or("unresolved");
    if class == "confirmed_synthetic"
        && matches!(
            relative.file_name().and_then(|name| name.to_str()),
            Some("train.csv" | "test.csv" | "meta.json")
        )
    {
        return Ok(None);
    }
    let sha = row
        .sha256
        .as_deref()
        .ok_or_else(|| anyhow!("{}: no SHA-256", path.display()))?;
    ensure!(
        sha.len() == 64 && sha.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "{}: invalid SHA-256",
        path.display()
    );
    let bytes = row
        .size
        .ok_or_else(|| anyhow!("{}: no size", path.display()))?;
    Ok(Some((
        path,
        CatalogEntry {
            path_hex: path_hex.clone(),
            sha256: sha.to_owned(),
            bytes,
            classification: class.to_owned(),
            s3_key: format!("{PREFIX}/blobs/sha256/{sha}"),
            s3_version: None,
        },
    )))
}

fn read_catalog(bytes: &[u8]) -> Result<BTreeMap<(String, String), CatalogEntry>> {
    let mut entries = BTreeMap::new();
    for (index, line) in bytes
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .enumerate()
    {
        let entry: CatalogEntry = serde_json::from_slice(line)
            .with_context(|| format!("custody catalog line {}", index + 1))?;
        ensure!(
            entry.s3_key == format!("{PREFIX}/blobs/sha256/{}", entry.sha256),
            "custody catalog line {} has mismatched key",
            index + 1
        );
        ensure!(
            entries
                .insert((entry.path_hex.clone(), entry.sha256.clone()), entry)
                .is_none(),
            "custody catalog has duplicate path/hash"
        );
    }
    Ok(entries)
}

/// Upload or verify every selected source file and the full immutable inventory, then publish a
/// superset catalog. `out` is a new local catalog; `out.partial` is a resumable verified ledger.
pub async fn upload(inventory: &Path, root: &Path, out: &Path, store: &Store) -> Result<usize> {
    ensure!(
        inventory.extension().is_some_and(|ext| ext == "jsonl"),
        "custody upload requires a completed .jsonl inventory"
    );
    let root = root.canonicalize()?;
    ensure!(
        root.file_name().is_some_and(|name| name == "remote_super"),
        "custody upload requires the remote_super root"
    );
    ensure!(
        out.parent().is_some_and(Path::is_dir),
        "output parent does not exist"
    );
    let partial = out.with_extension("jsonl.partial");
    let mut resumed = BTreeMap::new();
    if partial.exists() {
        let raw = fs::read(&partial)?;
        let complete = raw
            .iter()
            .rposition(|byte| *byte == b'\n')
            .map_or(0, |index| index + 1);
        for (identity, entry) in read_catalog(&raw[..complete])? {
            resumed.insert(identity, entry);
        }
        if complete != raw.len() {
            File::options()
                .write(true)
                .open(&partial)?
                .set_len(complete as u64)?;
        }
    }
    let mut ledger = BufWriter::new(File::options().append(true).create(true).open(&partial)?);
    let store_is_s3 = store.url.starts_with("s3://");
    let mut entries = BTreeMap::new();
    let mut verified_sha = HashSet::new();
    let reader = BufReader::new(File::open(inventory)?);
    for (line_no, line) in reader.lines().enumerate() {
        let row: InventoryRow = serde_json::from_str(&line?)
            .with_context(|| format!("inventory line {}", line_no + 1))?;
        let Some((path, mut entry)) = quant_file(&root, &row)? else {
            continue;
        };
        let identity = (entry.path_hex.clone(), entry.sha256.clone());
        if let Some(previous) = resumed.remove(&identity) {
            ensure!(
                previous.bytes == entry.bytes
                    && previous.classification == entry.classification
                    && previous.s3_key == entry.s3_key,
                "{}: resumed custody row differs from inventory",
                path.display()
            );
            let local = digest_file(&path)?;
            ensure!(
                local.sha256 == entry.sha256 && local.bytes == entry.bytes,
                "{}: source changed since inventory",
                path.display()
            );
            if verified_sha.insert(entry.sha256.clone()) {
                ensure!(
                    store
                        .verify_sha256(&entry.s3_key, &entry.sha256, entry.bytes)
                        .await?,
                    "{}: resumed blob is missing",
                    entry.s3_key
                );
            }
            entry.s3_version = store.version(&entry.s3_key).await?;
            entries.insert(identity, entry);
            continue;
        }
        if verified_sha.insert(entry.sha256.clone()) {
            store
                .put_file_verified(&entry.s3_key, &path, &entry.sha256, entry.bytes)
                .await?;
        } else {
            let local = digest_file(&path)?;
            ensure!(
                local.sha256 == entry.sha256 && local.bytes == entry.bytes,
                "{}: source changed since inventory",
                path.display()
            );
        }
        entry.s3_version = store.version(&entry.s3_key).await?;
        ensure!(
            !store_is_s3 || entry.s3_version.is_some(),
            "{}: S3 version missing",
            entry.s3_key
        );
        writeln!(ledger, "{}", serde_json::to_string(&entry)?)?;
        ledger.flush()?;
        entries.insert(identity, entry);
    }
    ensure!(
        resumed.is_empty(),
        "resumed ledger contains rows absent from inventory"
    );
    ledger.flush()?;
    ledger.into_inner()?.sync_all()?;
    let observed = store
        .get_observed(&format!("{PREFIX}/catalog/LATEST"))
        .await?;
    if let Some(pointer) = &observed {
        let sha = std::str::from_utf8(&pointer.bytes)?.trim();
        ensure!(
            sha.len() == 64 && sha.bytes().all(|byte| byte.is_ascii_hexdigit()),
            "invalid custody catalog pointer"
        );
        let key = format!("{PREFIX}/catalog/sha256/{sha}.jsonl");
        let old = store
            .get_bytes(&key)
            .await?
            .ok_or_else(|| anyhow!("{key}: missing"))?;
        ensure!(
            crate::hash::sha256_hex(&old) == sha,
            "{key}: prior catalog body differs from LATEST"
        );
        for (identity, entry) in read_catalog(&old)? {
            if verified_sha.insert(entry.sha256.clone()) {
                ensure!(
                    store
                        .verify_sha256(&entry.s3_key, &entry.sha256, entry.bytes)
                        .await?,
                    "{}: missing prior custody blob",
                    entry.s3_key
                );
            }
            entries.entry(identity).or_insert(entry);
        }
    }
    let digest = digest_file(inventory)?;
    let inventory_key = format!("{PREFIX}/inventories/sha256/{}.jsonl", digest.sha256);
    store
        .put_file_verified(&inventory_key, inventory, &digest.sha256, digest.bytes)
        .await?;
    let mut catalog_bytes = Vec::new();
    for entry in entries.values() {
        writeln!(catalog_bytes, "{}", serde_json::to_string(entry)?)?;
    }
    if out.exists() {
        ensure!(
            fs::read(out)? == catalog_bytes,
            "existing local catalog differs"
        );
    } else {
        let mut catalog = File::options().write(true).create_new(true).open(out)?;
        catalog.write_all(&catalog_bytes)?;
        catalog.sync_all()?;
    }
    let catalog_digest = digest_file(out)?;
    let catalog_key = format!("{PREFIX}/catalog/sha256/{}.jsonl", catalog_digest.sha256);
    store
        .put_file_verified(
            &catalog_key,
            out,
            &catalog_digest.sha256,
            catalog_digest.bytes,
        )
        .await?;
    if observed.as_ref().map(|row| row.bytes.as_slice())
        != Some(format!("{}\n", catalog_digest.sha256).as_bytes())
    {
        store
            .put_conditional(
                &format!("{PREFIX}/catalog/LATEST"),
                format!("{}\n", catalog_digest.sha256).into_bytes(),
                observed.as_ref(),
            )
            .await?;
    }
    fs::rename(&partial, out.with_extension("verified-ledger.jsonl"))?;
    Ok(entries.len())
}
