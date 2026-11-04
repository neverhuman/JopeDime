#[derive(Clone)]
/// `CapturedMember` is a record for the captured member contract in this module: Local, nonqualifying custody-v2 observations. No S3 client or retention executor.
pub struct CapturedMember {
    /// Row recorded for this captured member.
    pub row: Value,
    /// Body recorded for this captured member.
    pub body: Arc<[u8]>,
}

/// Fault hooks run before the real operation; they cannot skip persistence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PersistStep {
    /// The `Write` case of `PersistStep`.
    Write,
    /// The `FileSync` case of `PersistStep`.
    FileSync,
    /// The `ParentSync` case of `PersistStep`.
    ParentSync,
}

/// Performs the durable file operation and returns an error if it cannot be completed.
pub fn durable_file(
    path: &Path,
    raw: &[u8],
    hook: &mut impl FnMut(&Path, PersistStep) -> Result<()>,
) -> Result<()> {
    hook(path, PersistStep::Write)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)?;
    file.write_all(raw)?;
    hook(path, PersistStep::FileSync)?;
    file.sync_all()?;
    hook(path, PersistStep::ParentSync)?;
    File::open(path.parent().context("parent directory")?)?.sync_all()?;
    Ok(())
}

/// Performs the pack local operation and returns an error if it cannot be completed.
pub fn pack_local(destination: &Path, members: &[CapturedMember]) -> Result<Value> {
    pack_with(destination, members, &mut |_, _| Ok(()))
}

fn durable_directory(
    path: &Path,
    hook: &mut impl FnMut(&Path, PersistStep) -> Result<()>,
) -> Result<()> {
    fs::DirBuilder::new().mode(0o700).create(path)?;
    let result = (|| {
        hook(path, PersistStep::ParentSync)?;
        File::open(path.parent().context("directory parent")?)?.sync_all()?;
        Ok(())
    })();
    if result.is_err() {
        mark_aborted(path);
    }
    result
}

/// Performs the pack with operation and returns an error if it cannot be completed.
pub fn pack_with(
    destination: &Path,
    members: &[CapturedMember],
    hook: &mut impl FnMut(&Path, PersistStep) -> Result<()>,
) -> Result<Value> {
    ensure!(
        !members.is_empty() && members.len() <= MAX_MEMBERS,
        "bounded physical reference roster"
    );
    let mut refs = Vec::new();
    let mut physical = BTreeSet::new();
    let mut blobs = BTreeMap::new();
    let mut unique_bytes = 0u64;
    for m in members {
        let id = identity(&m.row)?;
        ensure!(
            physical.insert(id.clone()),
            "duplicate physical object version"
        );
        ensure!(id[2] != "null", "immutable version required for capture");
        ensure!(
            unsigned(&m.row["bytes"], "member bytes")? == m.body.len() as u64,
            "captured size mismatch"
        );
        let sha = sha256_hex(&m.body);
        if let Some(old) = blobs.get(&sha) {
            ensure!(old == &m.body, "content digest collision");
        } else {
            unique_bytes = unique_bytes
                .checked_add(m.body.len() as u64)
                .context("payload overflow")?;
            ensure!(unique_bytes <= MAX_PAYLOAD, "local pack raw-byte cap");
            blobs.insert(sha.clone(), m.body.clone());
        }
        refs.push(json!({"bucket":id[0],"key":id[1],"version_id":id[2],"bytes":m.body.len(),"sha256":sha}));
    }
    refs.sort_by_key(|r| identity(r).unwrap_or_default());
    let raw = custody_archive::encode(&blobs)?;
    let pack_sha = sha256_hex(&raw);
    let index = json!({"schema":"jope.custody-local-pack/v2","refs":refs,
        "blobs":blobs.iter().map(|(h,b)|json!({"name":format!("members/{h}"),"sha256":h,"bytes":b.len()})).collect::<Vec<_>>(),
        "pack_sha256":pack_sha,"qualified":false,"remote_publication":false});
    durable_directory(destination, hook)?;
    // Retain incomplete bytes and an abort witness on every caught persistence/readback error.
    let result = (|| {
        durable_file(&destination.join(format!("{pack_sha}.tar.gz")), &raw, hook)?;
        let manifest_raw = canonical_json(&index)?;
        let verified = verify_manifest_bytes(&manifest_raw, destination)?;
        durable_file(&destination.join("manifest.json"), &manifest_raw, hook)?;
        Ok(
            json!({"manifest_sha256":verified["manifest_sha256"],"physical_refs":refs.len(),
            "unique_blobs":blobs.len(),"pack_bytes":raw.len(),"qualified":false,"remote_writes":0}),
        )
    })();
    if result.is_err() {
        mark_aborted(destination);
    }
    result
}

fn mark_aborted(directory: &Path) {
    // This witness is defensive. Failure to persist it never turns an error into success.
    let _ = durable_file(
        &directory.join("ABORTED"),
        b"incomplete custody publication\n",
        &mut |_, _| Ok(()),
    );
}

/// Verify completion with the receipt SHA returned by a successful publisher, not a directory guess.
pub fn verify_complete(directory: &Path, receipt_sha: &str) -> Result<Value> {
    ensure!(
        is_hex64(receipt_sha) && !directory.join("ABORTED").exists(),
        "completed publication identity required"
    );
    let raw = bounded_read(&directory.join("receipt.json"), MAX_MANIFEST)?;
    ensure!(
        sha256_hex(&raw) == receipt_sha,
        "terminal receipt SHA mismatch"
    );
    let receipt: Value = serde_json::from_slice(&raw)?;
    ensure!(
        receipt["schema"] == "jope.custody-v2-local-dry-run/v1"
            && receipt["complete"] == true
            && receipt["qualified"] == false
            && receipt["remote_calls"] == 0
            && receipt["remote_mutations"] == 0,
        "nonqualifying completed receipt required"
    );
    let ledger = bounded_read(&directory.join("object-ledger.json"), 32 * 1024 * 1024)?;
    ensure!(
        receipt["ledger_sha256"] == sha256_hex(&ledger),
        "ledger SHA mismatch"
    );
    let verified = verify_saved(&directory.join("pack/manifest.json"))?;
    ensure!(
        receipt["pack_sample"]["manifest_sha256"] == verified["manifest_sha256"],
        "manifest receipt join"
    );
    Ok(
        json!({"receipt_sha256":receipt_sha,"readback":verified,"complete":true,"qualified":false,"remote_calls":0}),
    )
}
