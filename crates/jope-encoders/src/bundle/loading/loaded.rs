use super::*;

impl LoadedModelBundle {
    pub fn load(root: impl AsRef<Path>) -> Result<Self> {
        let (root, root_dir) = open_bundle_root(root.as_ref())?;
        let manifest_bytes = read_manifest(&root_dir)?;
        let manifest_sha256 = sha256_bytes(&manifest_bytes);
        if manifest_sha256 != MODEL_BUNDLE_MANIFEST_SHA256 {
            bail!(
                "model bundle manifest identity changed: {manifest_sha256}, expected {MODEL_BUNDLE_MANIFEST_SHA256}"
            );
        }
        let manifest: ModelBundleV1 =
            serde_json::from_slice(&manifest_bytes).context("decode model bundle manifest")?;
        manifest.validate_metadata()?;
        let mut inventories = BTreeMap::<String, Vec<TensorSpec>>::new();
        let mut json_files = BTreeMap::<String, Vec<u8>>::new();
        for file in &manifest.files {
            let bytes = read_regular_bytes(&root_dir, &file.path, file.bytes)?;
            let inventory = verify_file_bytes(file, &bytes)?;
            if let Some(inventory) = inventory {
                inventories.insert(file.path.clone(), inventory);
            } else {
                json_files.insert(file.path.clone(), bytes);
            }
        }
        verify_notices(&root_dir, &manifest)?;
        verify_operator_registry(
            json_files
                .get(OPERATOR_REGISTRY)
                .context("operator registry is not listed in the model bundle")?,
        )?;
        verify_jope_metadata(
            json_files
                .get(JOPE_POLICY_METADATA)
                .context("JOPE policy metadata is not listed in the model bundle")?,
            &manifest,
            inventories
                .get(JOPE_POLICY_ARTIFACT)
                .context("JOPE policy tensor inventory is missing")?,
            inventories
                .get(JOPE_PREPROCESSING_ARTIFACT)
                .context("JOPE preprocessing tensor inventory is missing")?,
        )?;
        let target_metadata = verify_target_metadata(
            json_files
                .get(TARGET_METADATA)
                .context("target metadata is not listed in the model bundle")?,
            &manifest,
            inventories,
        )?;
        Ok(Self {
            root,
            root_dir,
            manifest,
            target_metadata,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn path(&self, relative: &str) -> Result<PathBuf> {
        validate_relative_path(relative)?;
        if !self.manifest.files.iter().any(|file| file.path == relative) {
            bail!("unlisted model bundle path {relative:?}");
        }
        Ok(self.root.join(relative))
    }

    /// Open descriptor-relative without following symlinks, then bound and
    /// re-verify the exact bytes returned to the caller.
    pub fn read_verified(&self, relative: &str) -> Result<Vec<u8>> {
        validate_relative_path(relative)?;
        let file = self
            .manifest
            .files
            .iter()
            .find(|file| file.path == relative)
            .with_context(|| format!("unlisted model bundle path {relative:?}"))?;
        let bytes = read_regular_bytes(&self.root_dir, relative, file.bytes)?;
        verify_file_bytes(file, &bytes)?;
        Ok(bytes)
    }

    pub fn target_seed(&self) -> u64 {
        self.target_metadata.seed
    }
}
