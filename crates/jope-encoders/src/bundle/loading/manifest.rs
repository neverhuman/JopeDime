use super::*;

impl ModelBundleV1 {
    pub fn validate_metadata(&self) -> Result<()> {
        if self.schema != MODEL_BUNDLE_V1 {
            bail!("unsupported model bundle schema {:?}", self.schema);
        }
        if self.bundle_id != JOPE_BUNDLE_ID
            || self.source_checkpoint != JOPE_SOURCE_CHECKPOINT
            || self.source_checkpoint_sha256 != JOPE_SOURCE_CHECKPOINT_SHA256
            || self.policy_version != JOPE_POLICY_VERSION
        {
            bail!("model bundle does not bind the reviewed JOPE gen31 checkpoint identity");
        }
        if self.vocabulary_sha256 != JOPE_VOCABULARY_SHA256
            || self.vocabulary_canonical_sha256 != JOPE_VOCABULARY_CANONICAL_SHA256
            || self.grammar_sha256 != JOPE_GRAMMAR_SHA256
            || self.registry_sha256 != OPERATOR_REGISTRY_SHA256
            || self.registry_semantic_sha256 != OPERATOR_REGISTRY_SEMANTIC_SHA256
        {
            bail!("model bundle vocabulary, grammar, and registry identities are not reviewed content identities");
        }
        if self.tensor_inventory_algorithm != TENSOR_INVENTORY_V1 {
            bail!("unsupported tensor inventory algorithm");
        }
        let required_encoders = ["foundation", "hyperion", "target"];
        if self.encoders.len() != required_encoders.len()
            || required_encoders
                .iter()
                .any(|name| !self.encoders.contains_key(*name))
        {
            bail!("model bundle must identify foundation, hyperion, and target encoders");
        }
        if self.files.is_empty() {
            bail!("model bundle contains no files");
        }
        if self.notices.is_empty() {
            bail!("model bundle contains no bound notices");
        }
        for (path, digest) in &self.notices {
            validate_relative_path(path)?;
            if !path.starts_with("docs/notices/") || !valid_sha256(digest) {
                bail!("model bundle notice {path} has an invalid path or digest");
            }
        }
        let mut paths = HashSet::with_capacity(self.files.len());
        let mut referenced_notices = HashSet::new();
        for file in &self.files {
            validate_relative_path(&file.path)?;
            if !paths.insert(file.path.as_str()) {
                bail!("duplicate model bundle path {}", file.path);
            }
            if file.bytes == 0 || file.bytes > MAX_MODEL_FILE_BYTES || !valid_sha256(&file.sha256) {
                bail!("invalid digest or size for bundle file {}", file.path);
            }
            let extension = Path::new(&file.path)
                .extension()
                .and_then(|v| v.to_str())
                .unwrap_or("");
            if matches!(extension, "pt" | "ckpt" | "npy" | "npz")
                || !matches!(extension, "safetensors" | "json")
            {
                bail!("unsupported deployed model file type: {}", file.path);
            }
            if file.kind.trim().is_empty() || (extension == "safetensors" && file.shape.is_empty())
            {
                bail!("bundle file {} is missing kind or tensor shape", file.path);
            }
            match extension {
                "safetensors" => {
                    if file.tensor_count == 0
                        || !file
                            .tensor_inventory_sha256
                            .as_deref()
                            .is_some_and(valid_sha256)
                    {
                        bail!("bundle file {} is missing its tensor inventory", file.path);
                    }
                }
                "json" => {
                    if file.tensor_count != 0 || file.tensor_inventory_sha256.is_some() {
                        bail!("JSON bundle file {} cannot declare tensors", file.path);
                    }
                }
                _ => unreachable!("extension checked above"),
            }
            validate_license(&file.license, file)?;
            for notice in &file.license.notice_paths {
                if !self.notices.contains_key(notice) {
                    bail!(
                        "bundle file {} references unbound notice {notice}",
                        file.path
                    );
                }
                referenced_notices.insert(notice.as_str());
            }
        }
        if referenced_notices.len() != self.notices.len()
            || self
                .notices
                .keys()
                .any(|notice| !referenced_notices.contains(notice.as_str()))
        {
            bail!("model bundle contains an unreferenced notice identity");
        }
        for (name, encoder) in &self.encoders {
            validate_relative_path(&encoder.artifact)?;
            if encoder.version.trim().is_empty()
                || encoder.recipe.trim().is_empty()
                || !valid_sha256(&encoder.sha256)
                || !valid_sha256(&encoder.recipe_sha256)
                || !valid_source_revision(&encoder.source_revision)
                || !valid_source_uri(&encoder.source_uri)
            {
                bail!("encoder {name} has incomplete identity metadata");
            }
            let file = self
                .files
                .iter()
                .find(|file| file.path == encoder.artifact)
                .with_context(|| format!("encoder {name} artifact is not listed"))?;
            if file.sha256 != encoder.sha256 {
                bail!("encoder {name} artifact identity does not match its bundle file");
            }
        }
        let foundation = &self.encoders["foundation"];
        if foundation.version != concat!("foundation_enc.v1.", "tab", "pfn")
            || foundation.artifact != FOUNDATION_ARTIFACT
            || foundation.sha256 != FOUNDATION_SHA256
            || foundation.recipe_sha256 != FOUNDATION_RECIPE_SHA256
            || foundation.source_revision != format!("sha256:{FOUNDATION_RECIPE_SHA256}")
            || foundation.source_uri != format!("urn:jain:model-source:sha256:{FOUNDATION_SHA256}")
            || !foundation.recipe.contains(concat!("Tab", "PFNv3Reg"))
            || !foundation.recipe.contains("never TabICL")
        {
            bail!("foundation encoder must bind the reviewed Starlight recipe");
        }
        let foundation_file = self
            .files
            .iter()
            .find(|file| file.path == FOUNDATION_ARTIFACT)
            .context("reviewed foundation artifact is not listed")?;
        if foundation_file.kind != "starlight_regressor"
            || foundation_file.bytes != FOUNDATION_BYTES
            || foundation_file.sha256 != FOUNDATION_SHA256
            || foundation_file.tensor_count != FOUNDATION_TENSORS
            || foundation_file.tensor_inventory_sha256.as_deref()
                != Some(FOUNDATION_TENSOR_INVENTORY_SHA256)
            || foundation_file.license.spdx_expression
                != "LicenseRef-Jain-ThirdParty-Provenance-Pending"
            || foundation_file.license.license_class != "restricted"
            || foundation_file.license.redistribution != "prohibited"
            || foundation_file.license.policy_eligibility.eligible
            || foundation_file.license.policy_eligibility.open_only
            || foundation_file.license.notice_paths.len() != 1
            || foundation_file.license.notice_paths[0]
                != "docs/notices/third-party-provenance-pending.txt"
        {
            bail!("foundation model, tensor, or fail-closed license identity changed");
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String> {
        self.validate_metadata()?;
        let bytes = serde_json::to_vec(self)?;
        Ok(sha256_bytes(&bytes))
    }
}
