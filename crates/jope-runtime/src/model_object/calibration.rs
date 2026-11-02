use super::*;

impl ModelObject {
    fn unsupported(&self) -> Vec<String> {
        let mut out: Vec<_> = self
            .manifest
            .required_capabilities
            .iter()
            .filter(|c| c.as_str() != calibration_platt::CAPABILITY)
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

    pub(super) fn platt_head(
        &self,
        calibration: &calibration_platt::Calibration,
    ) -> Result<feat_math::advanced_v3::BoundModel> {
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

    /// Attach a final map learned from out-of-fold scores, preserving every native state byte.
    /// For bare bound heads, `calibration.class_order` declares the native IDs' 0/1 label order.
    pub fn with_platt_calibration(
        &self,
        calibration: calibration_platt::Calibration,
    ) -> Result<Self> {
        self.require_executable()?;
        if self.platt_calibration()?.is_some() {
            return Err(Failure::input(
                "model object already has a Platt calibration",
            ));
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
}
