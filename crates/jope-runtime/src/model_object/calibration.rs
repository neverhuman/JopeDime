use super::*;

impl ModelObject {
    pub(super) fn unsupported(&self) -> Vec<String> {
        let mut out: Vec<_> = self
            .manifest
            .required_capabilities
            .iter()
            .filter(|c| {
                !matches!(
                    c.as_str(),
                    calibration_platt::CAPABILITY | calibration_temperature::CAPABILITY
                )
            })
            .cloned()
            .collect();
        if !matches!(
            self.manifest.format.as_str(),
            APPLIANCE | BOUND | MATH_PIPELINE
        ) {
            out.push(format!("decoder:{}", self.manifest.format));
        }
        out
    }

    /// A versioned required section: neither its bytes nor its capability may be omitted.
    pub fn platt_calibration(&self) -> Result<Option<calibration_platt::Calibration>> {
        let required = self
            .manifest
            .required_capabilities
            .iter()
            .any(|c| c == calibration_platt::CAPABILITY);
        match (required, self.parts.get(calibration_platt::PART)) {
            (false, None) => Ok(None),
            (true, Some(bytes)) => {
                if bytes.len() > 64 * 1024 {
                    return Err(Failure::integrity("Platt section exceeds 64 KiB"));
                }
                let calibration: calibration_platt::Calibration = crate::strict_json::decode(bytes)
                    .map_err(|e| Failure::integrity(format!("Platt section: {e}")))?;
                calibration
                    .validate()
                    .map_err(|e| Failure::integrity(format!("Platt section: {e:#}")))?;
                Ok(Some(calibration))
            }
            _ => Err(Failure::integrity(
                "Platt section and required capability must occur together",
            )),
        }
    }

    /// Versioned Temperature Scaling section (`org.jopedime/temperature-calibration/v1` + `temperature.json`).
    pub fn temperature_calibration(&self) -> Result<Option<calibration_temperature::Calibration>> {
        let required = self
            .manifest
            .required_capabilities
            .iter()
            .any(|c| c == calibration_temperature::CAPABILITY);
        match (required, self.parts.get(calibration_temperature::PART)) {
            (false, None) => Ok(None),
            (true, Some(bytes)) => {
                if bytes.len() > 64 * 1024 {
                    return Err(Failure::integrity("Temperature section exceeds 64 KiB"));
                }
                let calibration: calibration_temperature::Calibration =
                    crate::strict_json::decode(bytes)
                        .map_err(|e| Failure::integrity(format!("Temperature section: {e}")))?;
                calibration
                    .validate()
                    .map_err(|e| Failure::integrity(format!("Temperature section: {e:#}")))?;
                Ok(Some(calibration))
            }
            _ => Err(Failure::integrity(
                "Temperature section and required capability must occur together",
            )),
        }
    }

    pub(super) fn platt_head(
        &self,
        calibration: &calibration_platt::Calibration,
    ) -> Result<feat_math::advanced_v3::BoundModel> {
        if self.temperature_calibration()?.is_some() {
            return Err(Failure::integrity(
                "model object cannot carry both Platt and Temperature calibration sections",
            ));
        }
        let model = match self.manifest.format.as_str() {
            APPLIANCE => {
                let fitted = FittedModel::from_bytes(&self.parts[&self.manifest.primary])?;
                if fitted.header.class_order != calibration.class_order {
                    return Err(Failure::integrity(
                        "Platt class order differs from fitted head",
                    ));
                }
                fitted.model
            }
            BOUND => crate::strict_json::decode(&self.parts[&self.manifest.primary])
                .map_err(|e| Failure::integrity(format!("bound state: {e}")))?,
            _ => {
                return Err(Failure::integrity(
                    "Platt section requires an appliance or bound binary head",
                ))
            }
        };
        model
            .validate_binding()
            .map_err(|e| Failure::integrity(format!("Platt head: {e:#}")))?;
        if model.task.kind != feat_math::advanced_v3::TaskKind::Binary {
            return Err(Failure::integrity(
                "Platt section is only valid for binary heads",
            ));
        }
        Ok(model)
    }

    pub(super) fn temperature_head(
        &self,
        calibration: &calibration_temperature::Calibration,
    ) -> Result<feat_math::advanced_v3::BoundModel> {
        if self.platt_calibration()?.is_some() {
            return Err(Failure::integrity(
                "model object cannot carry both Platt and Temperature calibration sections",
            ));
        }
        let model = match self.manifest.format.as_str() {
            APPLIANCE => {
                let fitted = FittedModel::from_bytes(&self.parts[&self.manifest.primary])?;
                if fitted.header.class_order != calibration.class_order {
                    return Err(Failure::integrity(
                        "Temperature class order differs from fitted head",
                    ));
                }
                fitted.model
            }
            BOUND => crate::strict_json::decode(&self.parts[&self.manifest.primary])
                .map_err(|e| Failure::integrity(format!("bound state: {e}")))?,
            _ => {
                return Err(Failure::integrity(
                    "Temperature section requires an appliance or bound multiclass head",
                ))
            }
        };
        model
            .validate_binding()
            .map_err(|e| Failure::integrity(format!("Temperature head: {e:#}")))?;
        if model.task.kind != feat_math::advanced_v3::TaskKind::Multiclass {
            return Err(Failure::integrity(
                "Temperature section is only valid for multiclass heads",
            ));
        }
        let headers: Vec<String> = model
            .schema
            .columns
            .iter()
            .map(|c| c.name.clone())
            .collect();
        let empty = feat_math::advanced_v3::RawTable::new(headers, Vec::new())
            .map_err(|e| Failure::integrity(format!("Temperature head schema: {e:#}")))?;
        let probe = model
            .predict_multiclass_table(&empty)
            .map_err(|e| Failure::integrity(format!("Temperature head: {e:#}")))?;
        if probe.classes.len() != calibration.class_order.len()
            || !(0..calibration.class_order.len()).all(|idx| probe.classes.contains(&(idx as i64)))
        {
            return Err(Failure::integrity(
                "Temperature class count or label IDs differ from bound multiclass head",
            ));
        }
        Ok(model)
    }

    /// Attach a final map learned from out-of-fold scores, preserving every native state byte.
    /// For bare bound heads, `calibration.class_order` declares the native IDs' 0/1 label order.
    pub fn with_platt_calibration(
        &self,
        calibration: calibration_platt::Calibration,
    ) -> Result<Self> {
        self.require_executable()?;
        if self.platt_calibration()?.is_some() || self.temperature_calibration()?.is_some() {
            return Err(Failure::input("model object already has a calibration"));
        }
        calibration.validate()?;
        self.platt_head(&calibration)?;
        let bytes = serde_json::to_vec(&calibration)?;
        let mut manifest = self.manifest.clone();
        manifest
            .required_capabilities
            .push(calibration_platt::CAPABILITY.into());
        manifest.files.push(Part::new(
            calibration_platt::PART,
            "application/json",
            &bytes,
        ));
        let mut parts = self.parts.clone();
        parts.insert(calibration_platt::PART.into(), bytes);
        Self::new(
            manifest,
            parts,
            crate::limits::Limits::default().max_model_bytes,
        )
    }

    /// Attach a cross-fitted scalar Temperature Scaling map learned on out-of-fold K-class
    /// probability rows, preserving every native state byte.
    pub fn with_temperature_calibration(
        &self,
        calibration: calibration_temperature::Calibration,
    ) -> Result<Self> {
        self.require_executable()?;
        if self.platt_calibration()?.is_some() || self.temperature_calibration()?.is_some() {
            return Err(Failure::input("model object already has a calibration"));
        }
        calibration.validate()?;
        self.temperature_head(&calibration)?;
        let bytes = serde_json::to_vec(&calibration)?;
        let mut manifest = self.manifest.clone();
        manifest
            .required_capabilities
            .push(calibration_temperature::CAPABILITY.into());
        manifest.files.push(Part::new(
            calibration_temperature::PART,
            "application/json",
            &bytes,
        ));
        let mut parts = self.parts.clone();
        parts.insert(calibration_temperature::PART.into(), bytes);
        Self::new(
            manifest,
            parts,
            crate::limits::Limits::default().max_model_bytes,
        )
    }
}
