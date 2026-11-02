use super::*;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CompileFeature {
    pub role: FeatureRole,
    pub origin: FeatureOrigin,
    pub missing_mask: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CompileInputLayout {
    pub(super) features: Vec<CompileFeature>,
}

impl CompileInputLayout {
    pub(crate) fn new(features: Vec<CompileFeature>) -> Result<Self> {
        let layout = Self { features };
        layout.validate()?;
        Ok(layout)
    }

    pub(crate) fn from_feature_block(block: &FeatureBlock) -> Result<Self> {
        block.validate()?;
        let features = block
            .roles
            .iter()
            .zip(&block.lineage)
            .enumerate()
            .map(|(column, (&role, lineage))| CompileFeature {
                role,
                origin: lineage.origin,
                missing_mask: block
                    .missing_masks
                    .iter()
                    .any(|row| row.get(column).copied().unwrap_or(false)),
            })
            .collect();
        Self::new(features)
    }

    pub(crate) fn features(&self) -> &[CompileFeature] {
        &self.features
    }

    pub(crate) fn widths(&self) -> FeatureWidths {
        widths_of(&self.features)
    }

    pub(super) fn validate(&self) -> Result<()> {
        let widths = self.widths();
        check_generated_dense(widths.generated_dense)?;
        for feature in &self.features {
            match feature.role {
                FeatureRole::Numeric
                | FeatureRole::CategoricalNative
                | FeatureRole::CategoricalFrequency => {
                    if feature.origin != FeatureOrigin::RawDerived {
                        bail!("input role {:?} must be raw-derived", feature.role);
                    }
                }
                FeatureRole::MissingIndicator | FeatureRole::CategoricalOneHot => {
                    if feature.origin != FeatureOrigin::GeneratedSparse || feature.missing_mask {
                        bail!(
                            "input role {:?} must be observed generated-sparse",
                            feature.role
                        );
                    }
                }
                FeatureRole::CategoricalOrdered => {
                    if !matches!(
                        feature.origin,
                        FeatureOrigin::RawDerived | FeatureOrigin::GeneratedDense
                    ) {
                        bail!("categorical ordered input must be raw-derived or generated-dense");
                    }
                }
                FeatureRole::Generated => bail!("input layout cannot contain generated role"),
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CompileV3Options<'a> {
    pub mode: CompileMode,
    pub requested_emitters: &'a [EmitterTarget],
    pub admitted_families: &'a [&'a str],
    pub input_layout: Option<&'a CompileInputLayout>,
}

impl<'a> CompileV3Options<'a> {
    pub(crate) const fn prototype(input_layout: Option<&'a CompileInputLayout>) -> Self {
        Self {
            mode: CompileMode::PrototypeCompatibility,
            requested_emitters: &[
                EmitterTarget::Runtime,
                EmitterTarget::Python,
                EmitterTarget::Rust,
            ],
            admitted_families: &[],
            input_layout,
        }
    }

    pub(crate) const fn production(
        requested_emitters: &'a [EmitterTarget],
        admitted_families: &'a [&'a str],
        input_layout: Option<&'a CompileInputLayout>,
    ) -> Self {
        Self {
            mode: CompileMode::Production,
            requested_emitters,
            admitted_families,
            input_layout,
        }
    }
}
