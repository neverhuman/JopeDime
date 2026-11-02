use super::*;

pub(crate) fn verify_declared_inputs(root: &Path, input: &FoundationDatasetInputV2) -> Result<()> {
    let actual = collect_files(root)?;
    let expected = input
        .files_read
        .iter()
        .map(|read| read.artifact.path.clone())
        .collect::<BTreeSet<_>>();
    ensure!(
        actual == expected,
        "job-private input tree differs from declared inputs"
    );
    for read in &input.files_read {
        verify_artifact(root, &read.artifact)?;
    }
    Ok(())
}

pub(crate) fn collect_files(root: &Path) -> Result<BTreeSet<String>> {
    fn visit(root: &Path, directory: &Path, output: &mut BTreeSet<String>) -> Result<()> {
        for entry in fs::read_dir(directory)? {
            let entry = entry?;
            let metadata = fs::symlink_metadata(entry.path())?;
            ensure!(
                !metadata.file_type().is_symlink(),
                "job tree contains a symlink"
            );
            if metadata.is_dir() {
                visit(root, &entry.path(), output)?;
            } else {
                ensure!(metadata.is_file(), "job tree contains a non-regular file");
                output.insert(
                    entry
                        .path()
                        .strip_prefix(root)?
                        .to_string_lossy()
                        .into_owned(),
                );
            }
        }
        Ok(())
    }
    let mut output = BTreeSet::new();
    visit(root, root, &mut output)?;
    Ok(output)
}

pub(crate) fn encoder_artifacts(encoder: &FoundationEncoderIdentityV2) -> Vec<ArtifactIdentity> {
    let mut artifacts = vec![
        encoder.protocol.clone(),
        encoder.python_encoder.clone(),
        encoder.wrapper.clone(),
        encoder.batch_driver.clone(),
        encoder.native_supervisor.clone(),
        encoder.model_config.clone(),
    ];
    artifacts.extend(encoder.python_support.clone());
    artifacts.extend(encoder.model_manifests.clone());
    artifacts.extend(encoder.model_weights.clone());
    artifacts.extend(encoder.prime_support.clone());
    artifacts.extend(encoder.build.lockfiles.clone());
    artifacts.sort_by(|left, right| left.path.cmp(&right.path));
    artifacts
}

pub(crate) fn verify_runtime_artifacts(root: &Path, artifacts: &[ArtifactIdentity]) -> Result<()> {
    let mut paths = BTreeSet::new();
    for artifact in artifacts {
        ensure!(
            paths.insert(artifact.path.as_str()),
            "runtime artifact path repeats"
        );
        verify_artifact(root, artifact)?;
    }
    Ok(())
}

pub(crate) fn verify_artifact(root: &Path, artifact: &ArtifactIdentity) -> Result<()> {
    artifact.validate()?;
    let path = root.join(&artifact.path);
    let metadata = fs::symlink_metadata(&path)?;
    ensure!(
        metadata.is_file() && !metadata.file_type().is_symlink(),
        "artifact is not regular"
    );
    ensure!(
        metadata.len() == artifact.size_bytes,
        "artifact size drift: {}",
        artifact.path
    );
    ensure!(
        sha256_file(&path)? == artifact.sha256,
        "artifact hash drift: {}",
        artifact.path
    );
    Ok(())
}

pub(crate) fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hash.update(&buffer[..read]);
    }
    Ok(hex::encode(hash.finalize()))
}

pub(crate) fn sha256_bytes(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}
