pub(crate) fn raw_feature_view(training: &str, preprocessing: &[u8]) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(
        &json!({"protocol":"raw_training_fold_v1",
        "training_sha256":training,"preprocessing_sha256":hash(preprocessing)}),
    )?)
}
fn relative_artifact(root: &Path, path: &Path) -> Result<serde_json::Value> {
    let absolute = path.canonicalize()?;
    let root = root.canonicalize()?;
    let relative = absolute
        .strip_prefix(&root)
        .context("original fit artifact outside workspace")?;
    Ok(json!({"path":relative,"sha256":file_hash(&absolute)?}))
}
fn relative_existing_reference(
    root: &Path,
    reference: &serde_json::Value,
) -> Result<serde_json::Value> {
    let path = Path::new(
        reference["path"]
            .as_str()
            .context("original fit reference missing")?,
    );
    let result = relative_artifact(
        root,
        &if path.is_absolute() {
            path.to_path_buf()
        } else {
            root.join(path)
        },
    )?;
    ensure!(
        result["sha256"] == reference["sha256"],
        "original fit reference changed"
    );
    Ok(result)
}

fn search_artifact_path(root: &Path, path: &Path) -> Result<std::path::PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let relative = absolute
        .strip_prefix(root)
        .context("search artifact outside declared root")?;
    resolve(root, relative)
}
