use super::*;
#[derive(Clone, Debug)]
pub(super) struct FeatureShape {
    pub(super) features: Vec<CompileFeature>,
}

impl FeatureShape {
    pub(super) fn new(layout: CompileInputLayout) -> Self {
        Self {
            features: layout.features,
        }
    }

    pub(super) fn width(&self) -> usize {
        self.features.len()
    }

    pub(super) fn widths(&self) -> FeatureWidths {
        widths_of(&self.features)
    }

    pub(super) fn generated(width: usize, sparse: bool) -> Result<Vec<CompileFeature>> {
        let origin = if sparse {
            FeatureOrigin::GeneratedSparse
        } else {
            FeatureOrigin::GeneratedDense
        };
        Ok((0..width)
            .map(|_| CompileFeature {
                role: if sparse {
                    FeatureRole::MissingIndicator
                } else {
                    FeatureRole::Generated
                },
                origin,
                missing_mask: false,
            })
            .collect())
    }

    pub(super) fn replace_or_append_generated(&mut self, width: usize, append: bool) -> Result<()> {
        let generated = Self::generated(width, false)?;
        if append {
            self.features
                .len()
                .checked_add(generated.len())
                .context("feature width overflow")?;
            self.features.extend(generated);
        } else {
            self.features = generated;
        }
        self.validate()
    }

    pub(super) fn append_one_to_one(&mut self) -> Result<()> {
        let generated = Self::generated(self.width(), false)?;
        self.features
            .len()
            .checked_add(generated.len())
            .context("feature width overflow")?;
        self.features.extend(generated);
        self.validate()
    }

    pub(super) fn validate(&self) -> Result<()> {
        check_generated_dense(self.widths().generated_dense)
    }
}
