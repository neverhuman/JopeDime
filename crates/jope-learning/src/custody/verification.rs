fn manifest(raw: &[u8]) -> Result<(Value, Vec<Blob>)> {
    ensure!(raw.len() as u64 <= MAX_MANIFEST, "bounded manifest bytes");
    let index: Value = serde_json::from_slice(raw)?;
    ensure!(index.is_object(), "pack manifest object");
    ensure!(
        index["schema"] == "jope.custody-local-pack/v2"
            && index["qualified"] == false
            && index["remote_publication"] == false,
        "closed non-qualifying pack"
    );
    let refs = index["refs"]
        .as_array()
        .context("bounded physical reference roster")?;
    let blobs = index["blobs"].as_array().context("bounded blob roster")?;
    ensure!(
        !refs.is_empty() && refs.len() <= MAX_MEMBERS,
        "bounded physical reference roster"
    );
    ensure!(
        !blobs.is_empty() && blobs.len() <= MAX_MEMBERS,
        "bounded blob roster"
    );
    ensure!(
        index["pack_sha256"].as_str().is_some_and(is_hex64),
        "content-addressed pack SHA"
    );
    let mut roster = Vec::new();
    let mut expected = BTreeMap::new();
    for b in blobs {
        let sha = b["sha256"].as_str().context("safe content member")?;
        ensure!(
            is_hex64(sha) && b["name"] == format!("members/{sha}"),
            "safe content member"
        );
        let bytes = unsigned(&b["bytes"], "blob bytes")?;
        ensure!(
            expected.insert(sha.to_owned(), bytes).is_none(),
            "duplicate blob declaration"
        );
        roster.push(Blob {
            name: format!("members/{sha}"),
            sha256: sha.to_owned(),
            bytes,
        });
    }
    let mut physical = BTreeSet::new();
    let mut used = BTreeSet::new();
    for row in refs {
        let id = identity(row)?;
        ensure!(
            id[2] != "null" && physical.insert(id),
            "unique immutable captured reference"
        );
        let sha = row["sha256"]
            .as_str()
            .context("every reference joins a verified blob")?;
        let bytes = unsigned(&row["bytes"], "reference bytes")?;
        ensure!(
            expected.get(sha) == Some(&bytes),
            "every reference joins a verified blob"
        );
        used.insert(sha.to_owned());
    }
    ensure!(
        used.len() == expected.len(),
        "every blob has a source reference"
    );
    Ok((index, roster))
}

/// Report the hash of the exact manifest bytes verified, even if its pathname is replaced.
pub fn verify_manifest_bytes(raw: &[u8], directory: &Path) -> Result<Value> {
    let (index, roster) = manifest(raw)?;
    let sha = index["pack_sha256"].as_str().context("pack SHA")?;
    verify_parsed(
        File::open(directory.join(format!("{sha}.tar.gz")))?,
        raw,
        &index,
        &roster,
    )
}

/// Performs the verify pack operation and returns an error if it cannot be completed.
pub fn verify_pack(path: &Path, raw: &[u8]) -> Result<Value> {
    let (index, roster) = manifest(raw)?;
    verify_parsed(File::open(path)?, raw, &index, &roster)
}

fn verify_parsed(file: File, raw: &[u8], index: &Value, roster: &[Blob]) -> Result<Value> {
    custody_archive::verify(
        file,
        index["pack_sha256"].as_str().context("pack SHA")?,
        roster,
    )?;
    Ok(
        json!({"physical_refs":index["refs"].as_array().context("refs")?.len(),
        "verified_blobs":roster.len(),"all_member_shas_verified":true,
        "manifest_sha256":sha256_hex(raw),"qualified":false,"remote_calls":0}),
    )
}
/// Performs the verify saved operation and returns an error if it cannot be completed.
pub fn verify_saved(path: &Path) -> Result<Value> {
    let raw = bounded_read(path, MAX_MANIFEST).context("bounded manifest bytes")?;
    verify_manifest_bytes(&raw, path.parent().context("manifest directory")?)
}
