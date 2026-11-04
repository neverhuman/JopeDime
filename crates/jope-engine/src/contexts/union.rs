use super::*;
/// Merge indices by reference. No vectors, source rows, or model assets are copied.
pub fn union(root: &Path, indexes: &[PathBuf], output: &Path) -> Result<()> {
    ensure!(
        !indexes.is_empty(),
        "context union requires at least one index"
    );
    output
        .strip_prefix(root)
        .context("context union must be inside workspace")?;
    let mut rows = BTreeMap::<(String, u64), Value>::new();
    let mut assembly = None;
    let mut qualification = false;
    let mut sources = Vec::new();
    for path in indexes {
        let index: Value = read_json(path)?;
        ensure!(
            index["schema"] == "parallel.context_index.v1" && index["dimensions"] == 2048,
            "invalid source context index"
        );
        let sha = index["assembly_sha256"]
            .as_str()
            .context("assembly hash absent")?
            .to_string();
        if let Some(old) = &assembly {
            ensure!(*old == sha, "context union has different frozen assemblies");
        } else {
            assembly = Some(sha.clone());
        }
        qualification |= index["qualification"]
            .as_bool()
            .context("qualification flag absent")?;
        for row in index["encodings"]
            .as_array()
            .context("context rows absent")?
        {
            let key = (
                row["dataset"]
                    .as_str()
                    .context("dataset absent")?
                    .to_string(),
                row["cut"].as_u64().context("cut absent")?,
            );
            let payload = resolve(
                root,
                Path::new(row["path"].as_str().context("context path absent")?),
            )?;
            ensure!(
                file_hash(&payload)? == row["sha256"],
                "source context bytes changed"
            );
            let encoding: Encoding = read_json(&payload.with_extension("json"))?;
            if let Some(receipt_sha) = row["receipt_sha256"].as_str() {
                ensure!(
                    file_hash(&payload.with_extension("json"))? == receipt_sha,
                    "source context metadata changed"
                );
            }
            ensure!(
                encoding.encoder_spec_sha256 == sha
                    && encoding.output_sha256 == row["sha256"]
                    && encoding.prepared_sha256 == row["prepared_sha256"]
                    && encoding.dataset == key.0
                    && encoding.cut as u64 == key.1
                    && encoding.family == row["family"]
                    && encoding.partition == row["partition"],
                "source context receipt mismatch"
            );
            if let Some(old) = rows.get(&key) {
                ensure!(*old == *row, "conflicting duplicate context in union");
            } else {
                rows.insert(key, row.clone());
            }
        }
        sources.push(json!({"path":path.strip_prefix(root)?,"sha256":file_hash(path)?}));
    }
    immutable_json(
        output,
        &json!({"schema":"parallel.context_index.v1","assembly_sha256":assembly.context("assembly absent")?,"qualification":qualification,"dimensions":2048,"sources":sources,"copied_vector_bytes":0,"encodings":rows.into_values().collect::<Vec<_>>()}),
    )
}
