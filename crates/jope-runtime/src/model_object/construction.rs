use super::*;

impl ModelObject {
    pub fn id(&self) -> &str {
        &self.id
    }
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }
    pub fn manifest_bytes(&self) -> &[u8] {
        &self.manifest_bytes
    }
    pub fn parts(&self) -> &BTreeMap<String, Vec<u8>> {
        &self.parts
    }

    pub fn new(
        manifest: Manifest,
        parts: BTreeMap<String, Vec<u8>>,
        max_bytes: u64,
    ) -> Result<Self> {
        let bytes = serde_json::to_vec_pretty(&manifest)?;
        Self::decode(bytes, parts, max_bytes)
    }

    /// Identity is SHA-256 of the *exact* manifest bytes. Readers never reserialize JSON to
    /// derive it, so Python/JS/C++/Rust do not need a shared float/string formatter.
    pub fn decode(
        manifest_bytes: Vec<u8>,
        parts: BTreeMap<String, Vec<u8>>,
        max_bytes: u64,
    ) -> Result<Self> {
        if manifest_bytes.len() as u64 > MAX_MANIFEST_BYTES {
            return Err(Failure::integrity("manifest exceeds its byte limit"));
        }
        let manifest: Manifest = crate::strict_json::decode(&manifest_bytes)
            .map_err(|e| Failure::integrity(format!("model manifest: {e}")))?;
        manifest.validate(max_bytes)?;
        if parts.len() != manifest.files.len() {
            return Err(Failure::integrity(
                "model object has missing or unlisted parts",
            ));
        }
        for part in &manifest.files {
            let bytes = parts
                .get(&part.path)
                .ok_or_else(|| Failure::integrity("missing part"))?;
            if bytes.len() as u64 != part.bytes || sha256_hex(bytes) != part.sha256 {
                return Err(Failure::integrity(format!(
                    "part {} checksum/length mismatch",
                    part.path
                )));
            }
        }
        Ok(Self {
            id: format!("sha256-{}", sha256_hex(&manifest_bytes)),
            manifest,
            manifest_bytes,
            parts,
        })
    }

    /// Wrap the entire native fitted file; no fit or parameter conversion occurs.
    pub fn from_fitted(model: &FittedModel) -> Result<Self> {
        Self::from_fitted_bytes(model.to_bytes()?)
    }

    /// Import an already serialized fitted file without normalizing its JSON state bytes.
    pub fn from_fitted_bytes(bytes: Vec<u8>) -> Result<Self> {
        let model = FittedModel::from_bytes(&bytes)?;
        let manifest = Manifest {
            schema: SCHEMA.into(),
            minimum_reader_version: 1,
            format: APPLIANCE.into(),
            producer: Producer::default(),
            primary: "model.jope".into(),
            required_capabilities: vec![],
            files: vec![Part::new(
                "model.jope",
                "application/vnd.jopedime.fitted",
                &bytes,
            )],
            metadata: json!({"task": model.header.task, "target": model.header.target,
                "method": model.header.method.id, "input_schema": model.header.input_schema,
                "retained_training_context": model.header.retained_training_context,
                "source_sha40": model.header.created_by.source_sha40}),
        };
        Self::new(
            manifest,
            BTreeMap::from([("model.jope".into(), bytes)]),
            crate::limits::Limits::default().max_model_bytes,
        )
    }

    /// All current bound GP forms use the existing validated state codec. Future forms use
    /// new format/capability identifiers instead of silently changing this decoder.
    pub fn from_bound(model: &feat_math::advanced_v3::BoundModel) -> Result<Self> {
        model.validate_binding()?;
        let bytes = serde_json::to_vec(model)?;
        Self::new(
            Manifest {
                schema: SCHEMA.into(),
                minimum_reader_version: 1,
                format: BOUND.into(),
                producer: Producer::default(),
                primary: "state.json".into(),
                required_capabilities: vec![],
                files: vec![Part::new("state.json", "application/json", &bytes)],
                metadata: json!({"input_schema": model.schema, "task": model.task}),
            },
            BTreeMap::from([("state.json".into(), bytes)]),
            crate::limits::Limits::default().max_model_bytes,
        )
    }
}
