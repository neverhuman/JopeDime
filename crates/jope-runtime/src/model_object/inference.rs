use super::*;

impl ModelObject {
    /// The input contract of a Platt- or Temperature-calibrated object, or `None` when it has no
    /// calibration. It exposes no executable head, so prediction still goes through [`Self::predict`].
    pub fn calibrated_input(&self) -> Result<Option<CalibratedInput>> {
        let platt = self.platt_calibration()?;
        let temperature = self.temperature_calibration()?;
        let head = match (platt, temperature) {
            (None, None) => return Ok(None),
            (Some(_), Some(_)) => {
                return Err(Failure::integrity(
                    "model object cannot carry both Platt and Temperature calibration sections",
                ))
            }
            (Some(calibration), None) => {
                self.require_executable()?;
                self.platt_head(&calibration)?
            }
            (None, Some(calibration)) => {
                self.require_executable()?;
                self.temperature_head(&calibration)?
            }
        };
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

    /// Predict through the complete object. Calibrated binary and multiclass rows retain explicit
    /// class order. Math pipelines keep their separate numeric-input API, and a bare uncalibrated
    /// classification bound object lacks class names; wrap it in an appliance object or attach its
    /// calibration.
    pub fn predict(
        &self,
        table: &feat_math::advanced_v3::RawTable,
    ) -> Result<crate::methods::native::Predictions> {
        use crate::methods::native::{predict_bound, Predictions};
        self.require_executable()?;
        let platt = self.platt_calibration()?;
        let temperature = self.temperature_calibration()?;
        if platt.is_some() && temperature.is_some() {
            return Err(Failure::integrity(
                "model object cannot carry both Platt and Temperature calibration sections",
            ));
        }
        if let Some(calibration) = platt {
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
        if let Some(calibration) = temperature {
            let head = self.temperature_head(&calibration)?;
            let Predictions::Probabilities { classes, values } =
                predict_bound(&head, table, &calibration.class_order)?
            else {
                return Err(Failure::integrity(
                    "Temperature head did not return multiclass probabilities",
                ));
            };
            let values = values
                .into_iter()
                .map(|row| calibration.apply(&row))
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
        if self.temperature_calibration()?.is_some() {
            return Err(Failure::backend("standalone fitted export would drop Temperature calibration; use the complete object and ModelObject::predict"));
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
                "bound JSON export would drop Platt calibration; export the complete object or standalone source",
            ));
        }
        if self.temperature_calibration()?.is_some() {
            return Err(Failure::backend(
                "bound JSON export would drop Temperature calibration; export the complete object or standalone source",
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
        if self.temperature_calibration()?.is_some() {
            return Err(Failure::backend(
                "math-pipeline decode/source export cannot preserve Temperature calibration; export the complete object",
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
        let platt = self.platt_calibration()?;
        let temperature = self.temperature_calibration()?;
        if platt.is_some() && temperature.is_some() {
            return Err(Failure::integrity(
                "model object cannot carry both Platt and Temperature calibration sections",
            ));
        }
        let (model, platt, temperature) = match (platt, temperature) {
            (Some(calibration), None) => {
                self.require_executable()?;
                let head = self.platt_head(&calibration)?;
                (head, Some(calibration), None)
            }
            (None, Some(calibration)) => {
                self.require_executable()?;
                let head = self.temperature_head(&calibration)?;
                (head, None, Some(calibration))
            }
            (None, None) => (self.bound()?, None, None),
            (Some(_), Some(_)) => unreachable!(),
        };
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
        let source = if let Some(calibration) = &platt {
            wrap_portable_with_platt(language, source, calibration)?
        } else if let Some(calibration) = &temperature {
            wrap_portable_with_temperature(language, source, calibration)?
        } else {
            source
        };
        Ok(source.into_bytes())
    }

    /// Lower a numeric math pipeline with an explicit Rust/Python recipe source profile.
    /// Standard source export continues to use the default profile.
    pub fn export_math_pipeline_source_with_options(
        &self,
        language: &str,
        options: &feat_math::transpile::EmitOptions,
    ) -> Result<Vec<u8>> {
        let pipeline = self.math_pipeline()?;
        match language {
            "rust" => crate::fitted_rs::emit_pipeline_with_options(&pipeline, options)
                .map(String::into_bytes)
                .map_err(|error| Failure::backend(error.to_string())),
            "python" => crate::fitted_py::emit_math_pipeline_with_options(&pipeline, options),
            _ => Err(Failure::input(
                "recipe source options require Rust or Python",
            )),
        }
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
        let platt = self.platt_calibration()?;
        let temperature = self.temperature_calibration()?;
        if platt.is_some() && temperature.is_some() {
            return Err(Failure::integrity(
                "model object cannot carry both Platt and Temperature calibration sections",
            ));
        }
        if let Some(calibration) = platt {
            self.platt_head(&calibration)?;
        } else if let Some(calibration) = temperature {
            self.temperature_head(&calibration)?;
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

fn py_hex(value: f64) -> String {
    let bits = value.to_bits();
    let sign = if bits >> 63 == 0 { "" } else { "-" };
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1u64 << 52) - 1);
    if exponent == 0 {
        if fraction == 0 {
            format!("\"{sign}0x0.0p+0\"")
        } else {
            format!("\"{sign}0x0.{fraction:013x}p-1022\"")
        }
    } else {
        format!("\"{sign}0x1.{fraction:013x}p{:+}\"", exponent - 1023)
    }
}

fn js_float(value: f64) -> String {
    if value == 0.0 {
        if value.is_sign_negative() {
            "-0".into()
        } else {
            "0".into()
        }
    } else {
        format!("{value:?}")
    }
}

fn wrap_portable_with_platt(
    language: &str,
    source: String,
    calibration: &calibration_platt::Calibration,
) -> Result<String> {
    use feat_math::transpile::lint_shipped_text;

    calibration
        .validate()
        .map_err(|e| Failure::integrity(format!("invalid Platt calibration: {e:#}")))?;
    let s = &calibration.sigmoid;
    let class_order_json = serde_json::to_string(&calibration.class_order)
        .map_err(|e| Failure::integrity(e.to_string()))?;

    match language {
        "python" => {
            const PY_TAIL: &str =
                "def predict_encoded(rows):\n    return [_predict(_encode(row)) for row in rows]";
            let prefix = source
                .trim_end_matches('\n')
                .strip_suffix(PY_TAIL)
                .ok_or_else(|| {
                    Failure::integrity("unexpected portable Python entrypoint for Platt wrapping")
                })?;
            let render_tail = |classes: &str| {
                format!(
                    "PLATT_CLASS_ORDER = {classes}\n\n\n\
                     def fitted_platt(score):\n\
                     \x20   if not math.isfinite(score) or score < 0.0 or score > 1.0:\n\
                     \x20       raise ValueError(\"invalid Platt score\")\n\
                     \x20   slope = float.fromhex({})\n\
                     \x20   intercept = float.fromhex({})\n\
                     \x20   scale = float.fromhex({})\n\
                     \x20   center = float.fromhex({})\n\
                     \x20   z = intercept if slope == 0.0 else intercept + slope * (score / scale - center)\n\
                     \x20   if z >= 0.0:\n\
                     \x20       p = 1.0 / (1.0 + math.exp(-z))\n\
                     \x20   else:\n\
                     \x20       e = math.exp(z)\n\
                     \x20       p = e / (1.0 + e)\n\
                     \x20   return [1.0 - p, p]\n\n\n\
                     def predict_encoded(rows):\n\
                     \x20   return [fitted_platt(_predict(_encode(row))[1]) for row in rows]\n",
                    py_hex(s.slope),
                    py_hex(s.intercept),
                    py_hex(s.scale),
                    py_hex(s.center),
                )
            };
            let hits = lint_shipped_text(&render_tail("[\"\",\"\"]"));
            if !hits.is_empty() {
                return Err(Failure::integrity(format!(
                    "banned vocabulary in Python Platt artifact: {}",
                    hits.join(", ")
                )));
            }
            Ok(format!("{prefix}{}", render_tail(&class_order_json)))
        }
        "rust" => {
            const RS_TAIL: &str =
                "pub fn predict_encoded(rows: &[Vec<String>]) -> Result<Vec<Vec<f64>>, String> {\n    rows.iter().map(|row| encode(row).map(|x| predict(&x))).collect()\n}";
            let prefix = source
                .trim_end_matches('\n')
                .strip_suffix(RS_TAIL)
                .ok_or_else(|| {
                    Failure::integrity("unexpected portable Rust entrypoint for Platt wrapping")
                })?;
            let render_tail = |c0: &str, c1: &str| {
                format!(
                    "pub const PLATT_CLASS_ORDER: [&str; 2] = [{c0:?}, {c1:?}];\n\
                     pub fn fitted_platt(score: f64) -> Result<[f64; 2], String> {{\n\
                     \x20   if !score.is_finite() || !(0.0..=1.0).contains(&score) {{ return Err(\"invalid Platt score\".into()); }}\n\
                     \x20   let slope = f64::from_bits(0x{:016x});\n\
                     \x20   let intercept = f64::from_bits(0x{:016x});\n\
                     \x20   let scale = f64::from_bits(0x{:016x});\n\
                     \x20   let center = f64::from_bits(0x{:016x});\n\
                     \x20   let z = if slope == 0.0 {{ intercept }} else {{ intercept + slope * (score / scale - center) }};\n\
                     \x20   let p = if z >= 0.0 {{ 1.0 / (1.0 + (-z).exp()) }} else {{ let e = z.exp(); e / (1.0 + e) }};\n\
                     \x20   Ok([1.0 - p, p])\n\
                     }}\n\
                     pub fn predict_encoded(rows: &[Vec<String>]) -> Result<Vec<Vec<f64>>, String> {{\n\
                     \x20   rows.iter()\n\
                     \x20       .map(|row| {{\n\
                     \x20           let x = encode(row)?;\n\
                     \x20           let raw = predict(&x);\n\
                     \x20           let pair = fitted_platt(raw[1])?;\n\
                     \x20           Ok(vec![pair[0], pair[1]])\n\
                     \x20       }})\n\
                     \x20       .collect()\n\
                     }}\n",
                    s.slope.to_bits(),
                    s.intercept.to_bits(),
                    s.scale.to_bits(),
                    s.center.to_bits(),
                )
            };
            let hits = lint_shipped_text(&render_tail("", ""));
            if !hits.is_empty() {
                return Err(Failure::integrity(format!(
                    "banned vocabulary in Rust Platt artifact: {}",
                    hits.join(", ")
                )));
            }
            Ok(format!(
                "{prefix}{}",
                render_tail(&calibration.class_order[0], &calibration.class_order[1])
            ))
        }
        "javascript" => {
            const JS_TAIL: &str =
                "function predictEncoded(rows) { return rows.map(row => predict(encode(row))); }\nmodule.exports = { predictEncoded };";
            let prefix = source
                .trim_end_matches('\n')
                .strip_suffix(JS_TAIL)
                .ok_or_else(|| {
                    Failure::integrity(
                        "unexpected portable JavaScript entrypoint for Platt wrapping",
                    )
                })?;
            let render_tail = |classes: &str| {
                format!(
                    "const PLATT_CLASS_ORDER = Object.freeze({classes});\n\
                     function fittedPlatt(score) {{\n\
                     \x20 if (!Number.isFinite(score) || score < 0 || score > 1) throw new Error(\"invalid Platt score\");\n\
                     \x20 const slope = {}, intercept = {}, scale = {}, center = {};\n\
                     \x20 const z = slope === 0 ? intercept : intercept + slope * (score / scale - center);\n\
                     \x20 let p;\n\
                     \x20 if (z >= 0) p = 1 / (1 + Math.exp(-z)); else {{ const e = Math.exp(z); p = e / (1 + e); }}\n\
                     \x20 return [1 - p, p];\n\
                     }}\n\
                     function predictEncoded(rows) {{ return rows.map(row => fittedPlatt(predict(encode(row))[1])); }}\n\
                     module.exports = {{ predictEncoded, fittedPlatt, PLATT_CLASS_ORDER }};\n",
                    js_float(s.slope),
                    js_float(s.intercept),
                    js_float(s.scale),
                    js_float(s.center),
                )
            };
            let hits = lint_shipped_text(&render_tail("[\"\",\"\"]"));
            if !hits.is_empty() {
                return Err(Failure::integrity(format!(
                    "banned vocabulary in JavaScript Platt artifact: {}",
                    hits.join(", ")
                )));
            }
            Ok(format!("{prefix}{}", render_tail(&class_order_json)))
        }
        _ => Err(Failure::input(
            "source language must be python, rust or javascript",
        )),
    }
}

fn wrap_portable_with_temperature(
    language: &str,
    source: String,
    calibration: &calibration_temperature::Calibration,
) -> Result<String> {
    use feat_math::transpile::lint_shipped_text;

    calibration
        .validate()
        .map_err(|e| Failure::integrity(format!("invalid Temperature calibration: {e:#}")))?;
    let k = calibration.class_order.len();
    let class_order_json = serde_json::to_string(&calibration.class_order)
        .map_err(|e| Failure::integrity(e.to_string()))?;
    let dummy_classes = vec![""; k];
    let dummy_classes_json =
        serde_json::to_string(&dummy_classes).map_err(|e| Failure::integrity(e.to_string()))?;
    let floor = calibration_temperature::PROBABILITY_FLOOR;
    let sum_tol = crate::fitness_multiclass::PROBABILITY_SUM_TOLERANCE;

    match language {
        "python" => {
            const PY_TAIL: &str =
                "def predict_encoded(rows):\n    return [_predict(_encode(row)) for row in rows]";
            let prefix = source
                .trim_end_matches('\n')
                .strip_suffix(PY_TAIL)
                .ok_or_else(|| {
                    Failure::integrity(
                        "unexpected portable Python entrypoint for Temperature wrapping",
                    )
                })?;
            let render_tail = |classes: &str| {
                format!(
                    "TEMPERATURE_CLASS_ORDER = {classes}\n\n\n\
                     def fitted_temperature(probs):\n\
                     \x20   if len(probs) != {k}:\n\
                     \x20       raise ValueError(\"invalid Temperature probability length\")\n\
                     \x20   total = 0.0\n\
                     \x20   for p in probs:\n\
                     \x20       if not math.isfinite(p) or p < 0.0 or p > 1.0:\n\
                     \x20           raise ValueError(\"invalid Temperature probability\")\n\
                     \x20       total += p\n\
                     \x20   if abs(total - 1.0) > float.fromhex({}):\n\
                     \x20       raise ValueError(\"invalid Temperature probability sum\")\n\
                     \x20   inv_t = 1.0 / float.fromhex({})\n\
                     \x20   floor = float.fromhex({})\n\
                     \x20   logits = [math.log(p if p >= floor else floor) * inv_t for p in probs]\n\
                     \x20   max_logit = max(logits)\n\
                     \x20   exps = [math.exp(z - max_logit) for z in logits]\n\
                     \x20   denom = 0.0\n\
                     \x20   for e in exps:\n\
                     \x20       denom += e\n\
                     \x20   if not math.isfinite(denom) or denom <= 0.0:\n\
                     \x20       raise ValueError(\"invalid Temperature normalizer\")\n\
                     \x20   return [e / denom for e in exps]\n\n\n\
                     def predict_encoded(rows):\n\
                     \x20   return [fitted_temperature(_predict(_encode(row))) for row in rows]\n",
                    py_hex(sum_tol),
                    py_hex(calibration.temperature),
                    py_hex(floor),
                )
            };
            let hits = lint_shipped_text(&render_tail(&dummy_classes_json));
            if !hits.is_empty() {
                return Err(Failure::integrity(format!(
                    "banned vocabulary in Python Temperature artifact: {}",
                    hits.join(", ")
                )));
            }
            Ok(format!("{prefix}{}", render_tail(&class_order_json)))
        }
        "rust" => {
            const RS_TAIL: &str =
                "pub fn predict_encoded(rows: &[Vec<String>]) -> Result<Vec<Vec<f64>>, String> {\n    rows.iter().map(|row| encode(row).map(|x| predict(&x))).collect()\n}";
            let prefix = source
                .trim_end_matches('\n')
                .strip_suffix(RS_TAIL)
                .ok_or_else(|| {
                    Failure::integrity(
                        "unexpected portable Rust entrypoint for Temperature wrapping",
                    )
                })?;
            let format_rs_classes = |items: &[&str]| {
                items
                    .iter()
                    .map(|item| format!("{item:?}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            let render_tail = |classes_literal: &str| {
                format!(
                    "pub const TEMPERATURE_CLASS_ORDER: [&str; {k}] = [{classes_literal}];\n\
                     pub fn fitted_temperature(probs: &[f64]) -> Result<Vec<f64>, String> {{\n\
                     \x20   if probs.len() != {k} {{ return Err(\"invalid Temperature probability length\".into()); }}\n\
                     \x20   let mut total = 0.0;\n\
                     \x20   for &p in probs {{\n\
                     \x20       if !p.is_finite() || !(0.0..=1.0).contains(&p) {{ return Err(\"invalid Temperature probability\".into()); }}\n\
                     \x20       total += p;\n\
                     \x20   }}\n\
                     \x20   if (total - 1.0).abs() > f64::from_bits(0x{:016x}) {{ return Err(\"invalid Temperature probability sum\".into()); }}\n\
                     \x20   let inv_t = 1.0 / f64::from_bits(0x{:016x});\n\
                     \x20   let floor = f64::from_bits(0x{:016x});\n\
                     \x20   let mut logits = Vec::with_capacity(probs.len());\n\
                     \x20   let mut max_logit = f64::NEG_INFINITY;\n\
                     \x20   for &p in probs {{\n\
                     \x20       let z = p.max(floor).ln() * inv_t;\n\
                     \x20       if z > max_logit {{ max_logit = z; }}\n\
                     \x20       logits.push(z);\n\
                     \x20   }}\n\
                     \x20   let mut exps = Vec::with_capacity(probs.len());\n\
                     \x20   let mut denom = 0.0;\n\
                     \x20   for z in logits {{\n\
                     \x20       let e = (z - max_logit).exp();\n\
                     \x20       denom += e;\n\
                     \x20       exps.push(e);\n\
                     \x20   }}\n\
                     \x20   if !denom.is_finite() || denom <= 0.0 {{ return Err(\"invalid Temperature normalizer\".into()); }}\n\
                     \x20   Ok(exps.into_iter().map(|e| e / denom).collect())\n\
                     }}\n\
                     pub fn predict_encoded(rows: &[Vec<String>]) -> Result<Vec<Vec<f64>>, String> {{\n\
                     \x20   rows.iter()\n\
                     \x20       .map(|row| {{\n\
                     \x20           let x = encode(row)?;\n\
                     \x20           let raw = predict(&x);\n\
                     \x20           fitted_temperature(&raw)\n\
                     \x20       }})\n\
                     \x20       .collect()\n\
                     }}\n",
                    sum_tol.to_bits(),
                    calibration.temperature.to_bits(),
                    floor.to_bits(),
                )
            };
            let hits = lint_shipped_text(&render_tail(&format_rs_classes(&dummy_classes)));
            if !hits.is_empty() {
                return Err(Failure::integrity(format!(
                    "banned vocabulary in Rust Temperature artifact: {}",
                    hits.join(", ")
                )));
            }
            let real_refs: Vec<&str> = calibration.class_order.iter().map(String::as_str).collect();
            Ok(format!(
                "{prefix}{}",
                render_tail(&format_rs_classes(&real_refs))
            ))
        }
        "javascript" => {
            const JS_TAIL: &str =
                "function predictEncoded(rows) { return rows.map(row => predict(encode(row))); }\nmodule.exports = { predictEncoded };";
            let prefix = source
                .trim_end_matches('\n')
                .strip_suffix(JS_TAIL)
                .ok_or_else(|| {
                    Failure::integrity(
                        "unexpected portable JavaScript entrypoint for Temperature wrapping",
                    )
                })?;
            let render_tail = |classes: &str| {
                format!(
                    "const TEMPERATURE_CLASS_ORDER = Object.freeze({classes});\n\
                     function fittedTemperature(probs) {{\n\
                     \x20 if (!Array.isArray(probs) || probs.length !== {k}) throw new Error(\"invalid Temperature probability length\");\n\
                     \x20 let total = 0.0;\n\
                     \x20 for (const p of probs) {{\n\
                     \x20   if (!Number.isFinite(p) || p < 0 || p > 1) throw new Error(\"invalid Temperature probability\");\n\
                     \x20   total += p;\n\
                     \x20 }}\n\
                     \x20 if (Math.abs(total - 1) > {}) throw new Error(\"invalid Temperature probability sum\");\n\
                     \x20 const invT = 1 / {}, floor = {};\n\
                     \x20 const logits = [];\n\
                     \x20 let maxLogit = -Infinity;\n\
                     \x20 for (const p of probs) {{\n\
                     \x20   const z = Math.log(p >= floor ? p : floor) * invT;\n\
                     \x20   if (z > maxLogit) maxLogit = z;\n\
                     \x20   logits.push(z);\n\
                     \x20 }}\n\
                     \x20 const exps = [];\n\
                     \x20 let denom = 0.0;\n\
                     \x20 for (const z of logits) {{\n\
                     \x20   const e = Math.exp(z - maxLogit);\n\
                     \x20   denom += e;\n\
                     \x20   exps.push(e);\n\
                     \x20 }}\n\
                     \x20 if (!Number.isFinite(denom) || denom <= 0) throw new Error(\"invalid Temperature normalizer\");\n\
                     \x20 return exps.map(e => e / denom);\n\
                     }}\n\
                     function predictEncoded(rows) {{ return rows.map(row => fittedTemperature(predict(encode(row)))); }}\n\
                     module.exports = {{ predictEncoded, fittedTemperature, TEMPERATURE_CLASS_ORDER }};\n",
                    js_float(sum_tol),
                    js_float(calibration.temperature),
                    js_float(floor),
                )
            };
            let hits = lint_shipped_text(&render_tail(&dummy_classes_json));
            if !hits.is_empty() {
                return Err(Failure::integrity(format!(
                    "banned vocabulary in JavaScript Temperature artifact: {}",
                    hits.join(", ")
                )));
            }
            Ok(format!("{prefix}{}", render_tail(&class_order_json)))
        }
        _ => Err(Failure::input(
            "source language must be python, rust or javascript",
        )),
    }
}
