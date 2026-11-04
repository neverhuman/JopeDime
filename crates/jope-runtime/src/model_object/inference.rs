use super::*;

impl ModelObject {
    /// The input contract of a Platt-calibrated object, or `None` when it has no calibration.
    /// It exposes no executable head, so prediction still goes through [`Self::predict`].
    pub fn calibrated_input(&self) -> Result<Option<CalibratedInput>> {
        let Some(calibration) = self.platt_calibration()? else {
            return Ok(None);
        };
        self.require_executable()?;
        let head = self.platt_head(&calibration)?;
        let header = match self.manifest.format.as_str() {
            APPLIANCE => Some(FittedModel::from_bytes(&self.parts[&self.manifest.primary])?.header),
            _ => None,
        };
        Ok(Some(CalibratedInput {
            schema: head.schema,
            genome_sha256: head.genome_sha256,
            header,
        }))
    }

    /// Predict through the complete object. Calibrated binary rows retain explicit class order.
    /// Math pipelines keep their separate numeric-input API, and a bare uncalibrated binary
    /// bound object lacks class names; wrap it in an appliance object or attach its calibration.
    pub fn predict(
        &self,
        table: &feat_math::advanced_v3::RawTable,
    ) -> Result<crate::methods::native::Predictions> {
        use crate::methods::native::{predict_bound, Predictions};
        self.require_executable()?;
        if let Some(calibration) = self.platt_calibration()? {
            let head = self.platt_head(&calibration)?;
            let Predictions::Probabilities { classes, values } =
                predict_bound(&head, table, &calibration.class_order)?
            else {
                return Err(Failure::integrity(
                    "Platt head did not return binary probabilities",
                ));
            };
            let values = values
                .into_iter()
                .map(|row| {
                    let p = calibration.sigmoid.apply(row[1])?;
                    Ok(vec![1.0 - p, p])
                })
                .collect::<Result<Vec<_>>>()?;
            return Ok(Predictions::Probabilities { classes, values });
        }
        match self.manifest.format.as_str() {
            APPLIANCE => self.fitted()?.predict(table),
            BOUND => predict_bound(&self.bound()?, table, &[]),
            _ => Err(Failure::backend(
                "use the registered decoder's prediction API for this object format",
            )),
        }
    }

    pub fn fitted(&self) -> Result<FittedModel> {
        self.require_executable()?;
        if self.platt_calibration()?.is_some() {
            return Err(Failure::backend("standalone fitted export would drop Platt calibration; use the complete object and ModelObject::predict"));
        }
        if self.manifest.format != APPLIANCE {
            return Err(Failure::backend(
                "this command needs an appliance fitted object",
            ));
        }
        FittedModel::from_bytes(&self.parts[&self.manifest.primary])
    }

    pub fn bound(&self) -> Result<feat_math::advanced_v3::BoundModel> {
        self.require_executable()?;
        if self.platt_calibration()?.is_some() {
            return Err(Failure::backend(
                "bound/source export would drop Platt calibration; export the complete object",
            ));
        }
        if self.manifest.format == MATH_PIPELINE {
            return Err(Failure::backend(
                "a math-pipeline object has no bound state; export object or javascript instead",
            ));
        }
        if self.manifest.format == APPLIANCE {
            let fitted = self.fitted()?;
            if fitted
                .header
                .output_transform
                .is_some_and(|map| !map.is_identity())
            {
                return Err(Failure::backend(
                    "bound/source export cannot preserve this model's output calibration; export object or fitted state instead",
                ));
            }
            return Ok(fitted.model);
        }
        let model: feat_math::advanced_v3::BoundModel =
            crate::strict_json::decode(&self.parts[&self.manifest.primary])
                .map_err(|e| Failure::integrity(format!("bound state: {e}")))?;
        model
            .validate_binding()
            .map_err(|e| Failure::integrity(format!("bound state: {e:#}")))?;
        Ok(model)
    }

    /// Decode a `math-pipeline/v1` object: a fitted feat-math pipeline behind a numeric input
    /// contract. See [`crate::math_pipeline`].
    pub fn math_pipeline(&self) -> Result<crate::math_pipeline::MathPipeline> {
        self.require_executable()?;
        if self.platt_calibration()?.is_some() {
            return Err(Failure::backend(
                "math-pipeline decode/source export cannot preserve Platt calibration; export the complete object",
            ));
        }
        if self.manifest.format != MATH_PIPELINE {
            return Err(Failure::backend(
                "this command needs a math-pipeline object",
            ));
        }
        crate::math_pipeline::MathPipeline::decode(&self.parts, &self.manifest.primary)
    }

    /// Lower a loaded model using the existing portable emitter's explicit coverage gate.
    pub fn export_source(&self, language: &str) -> Result<Vec<u8>> {
        if self.manifest.format == MATH_PIPELINE {
            let pipeline = self.math_pipeline()?;
            return match language {
                "javascript" => pipeline.export_javascript(),
                "rust" => crate::fitted_rs::emit_pipeline(&pipeline)
                    .map(String::into_bytes)
                    .map_err(|error| Failure::backend(error.to_string())),
                "python" => crate::fitted_py::emit_math_pipeline(&pipeline),
                _ => Err(Failure::input(
                    "source language must be python, rust or javascript",
                )),
            };
        }
        let model = self.bound()?;
        let reason = feat_math::advanced_v3::fitted_portable_unsupported(&model)
            .map_err(|e| Failure::integrity(e.to_string()))?;
        if let Some(reason) = reason {
            return Err(Failure::backend(format!(
                "portable source export unsupported: {reason:?}"
            )));
        }
        let sources = feat_math::advanced_v3::emit_fitted_portable(&model)?;
        let source = match language {
            "python" => sources.python,
            "rust" => sources.rust,
            "javascript" => sources
                .javascript
                .ok_or_else(|| Failure::backend("JavaScript export unsupported for this head"))?,
            _ => {
                return Err(Failure::input(
                    "source language must be python, rust or javascript",
                ))
            }
        };
        Ok(source.into_bytes())
    }

    pub(super) fn require_executable(&self) -> Result<()> {
        let unsupported = self.unsupported();
        if !unsupported.is_empty() {
            return Err(Failure::backend(format!(
                "unsupported model capabilities: {}",
                unsupported.join(", ")
            )));
        }
        Ok(())
    }

    /// Verify the complete executable state without reducing an appliance to its native model.
    pub fn verify_executable(&self) -> Result<()> {
        self.require_executable()?;
        if let Some(calibration) = self.platt_calibration()? {
            self.platt_head(&calibration)?;
        } else if self.manifest.format == APPLIANCE {
            self.fitted()?;
        } else if self.manifest.format == MATH_PIPELINE {
            self.math_pipeline()?;
        } else {
            self.bound()?;
        }
        Ok(())
    }

    pub fn inspect(&self) -> Result<Inspection> {
        let unsupported = self.unsupported();
        if unsupported.is_empty() {
            self.verify_executable()?;
        }
        Ok(Inspection {
            schema: INSPECTION_SCHEMA,
            id: self.id.clone(),
            integrity_verified: true,
            executable: unsupported.is_empty(),
            unsupported_capabilities: unsupported,
            format: self.manifest.format.clone(),
            primary: self.manifest.primary.clone(),
            files: self.manifest.files.clone(),
            metadata: self.manifest.metadata.clone(),
        })
    }

    /// `manifest.json` is written last using an atomic link. A failed or interrupted publisher
    /// leaves an uncommitted directory which readers refuse; it never damages another object.
    pub fn save(&self, root: &Path, max_bytes: u64) -> Result<PathBuf> {
        self.manifest.validate(max_bytes)?;
        // Decode known state before any filesystem mutation. Unsupported codecs remain opaque.
        self.inspect()?;
        if root.to_str().is_none() {
            return Err(Failure::input("model root must be UTF-8"));
        }
        fs::create_dir_all(root)
            .map_err(|e| Failure::output(format!("{}: {e}", root.display())))?;
        let destination = root.join(&self.id);
        let mut builder = fs::DirBuilder::new();
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        match builder.create(&destination) {
            Ok(()) => (),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                if !fs::symlink_metadata(&destination)?.file_type().is_dir() {
                    return Err(Failure::integrity(
                        "object destination is not a regular directory",
                    ));
                }
                if fs::symlink_metadata(destination.join(MANIFEST)).is_ok() {
                    let existing = Self::read(&destination, max_bytes)?;
                    if existing.manifest_bytes != self.manifest_bytes {
                        return Err(Failure::integrity("existing object differs"));
                    }
                    return Ok(destination);
                }
            }
            Err(e) => return Err(Failure::output(format!("{}: {e}", destination.display()))),
        }
        // Resume an interrupted publication only when every retained declared part agrees.
        // Never overwrite or reclaim state; a concurrent identical publisher is safe.
        for (name, data) in &self.parts {
            if let Ok(metadata) = fs::symlink_metadata(destination.join(name)) {
                if !metadata.file_type().is_file()
                    || read_bounded(&destination.join(name), data.len() as u64)? != *data
                {
                    return Err(Failure::integrity(
                        "incomplete object contains conflicting state",
                    ));
                }
            }
        }
        for (name, data) in &self.parts {
            publish_identical(&destination.join(name), data)?;
        }
        publish_identical(&destination.join(MANIFEST), &self.manifest_bytes)?;
        #[cfg(unix)]
        {
            File::open(&destination)
                .and_then(|f| f.sync_all())
                .map_err(|e| Failure::output(e.to_string()))?;
        }
        #[cfg(unix)]
        {
            File::open(root)
                .and_then(|f| f.sync_all())
                .map_err(|e| Failure::output(e.to_string()))?;
        }
        Ok(destination)
    }

    pub fn read(path: &Path, max_bytes: u64) -> Result<Self> {
        if !fs::symlink_metadata(path)
            .map_err(|e| Failure::input(e.to_string()))?
            .file_type()
            .is_dir()
        {
            return Err(Failure::integrity(
                "model object must be a directory (no symlinks)",
            ));
        }
        let manifest_bytes = read_bounded(&path.join(MANIFEST), MAX_MANIFEST_BYTES)?;
        let manifest: Manifest = crate::strict_json::decode(&manifest_bytes)
            .map_err(|e| Failure::integrity(format!("model manifest: {e}")))?;
        manifest.validate(max_bytes)?;
        let id = format!("sha256-{}", sha256_hex(&manifest_bytes));
        let resolved = fs::canonicalize(path).map_err(|e| Failure::input(e.to_string()))?;
        if resolved.file_name().and_then(|n| n.to_str()) != Some(&id) {
            return Err(Failure::integrity(
                "object directory does not match manifest identity",
            ));
        }
        let mut parts = BTreeMap::new();
        for part in &manifest.files {
            parts.insert(
                part.path.clone(),
                read_bounded(&path.join(&part.path), part.bytes)?,
            );
        }
        Self::decode(manifest_bytes, parts, max_bytes)
    }
}
