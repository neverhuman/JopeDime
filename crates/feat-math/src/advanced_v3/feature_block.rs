use crate::Rows;
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const MAX_GENERATED_DENSE_WIDTH: usize = 512;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FeatureRole {
    Numeric,
    MissingIndicator,
    CategoricalNative,
    CategoricalFrequency,
    CategoricalOneHot,
    CategoricalOrdered,
    Generated,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FeatureOrigin {
    RawDerived,
    GeneratedDense,
    GeneratedSparse,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct FeatureLineage {
    pub source_columns: Vec<usize>,
    pub operations: Vec<String>,
    pub origin: FeatureOrigin,
}

impl FeatureLineage {
    pub fn raw(source_column: usize, operation: impl Into<String>) -> Self {
        Self {
            source_columns: vec![source_column],
            operations: vec![operation.into()],
            origin: FeatureOrigin::RawDerived,
        }
    }

    pub fn generated_dense(source_columns: Vec<usize>, operation: impl Into<String>) -> Self {
        Self {
            source_columns,
            operations: vec![operation.into()],
            origin: FeatureOrigin::GeneratedDense,
        }
    }

    pub fn generated_sparse(source_columns: Vec<usize>, operation: impl Into<String>) -> Self {
        Self {
            source_columns,
            operations: vec![operation.into()],
            origin: FeatureOrigin::GeneratedSparse,
        }
    }

    fn with_operation(&self, operation: &str) -> Self {
        let mut lineage = self.clone();
        lineage.operations.push(operation.to_string());
        lineage
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FeatureWidths {
    pub raw_derived: usize,
    pub generated_dense: usize,
    pub generated_sparse: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FeatureBlock {
    pub values: Rows,
    pub roles: Vec<FeatureRole>,
    /// One row-major mask per row. Generated features may be missing when a
    /// table-aware transform has insufficient or all-missing history.
    pub missing_masks: Vec<Vec<bool>>,
    pub sparse_columns: Vec<usize>,
    pub lineage: Vec<FeatureLineage>,
    pub raw_source_width: usize,
}

impl FeatureBlock {
    pub fn new(
        values: Rows,
        roles: Vec<FeatureRole>,
        missing_masks: Vec<Vec<bool>>,
        sparse_columns: Vec<usize>,
        lineage: Vec<FeatureLineage>,
        raw_source_width: usize,
    ) -> Result<Self> {
        let block = Self {
            values,
            roles,
            missing_masks,
            sparse_columns,
            lineage,
            raw_source_width,
        };
        block.validate()?;
        Ok(block)
    }

    pub fn check_generated_dense_width(width: usize) -> Result<()> {
        if width > MAX_GENERATED_DENSE_WIDTH {
            bail!("generated dense width {width} exceeds {MAX_GENERATED_DENSE_WIDTH}");
        }
        Ok(())
    }

    pub fn widths(&self) -> FeatureWidths {
        let mut widths = FeatureWidths::default();
        for lineage in &self.lineage {
            match lineage.origin {
                FeatureOrigin::RawDerived => widths.raw_derived += 1,
                FeatureOrigin::GeneratedDense => widths.generated_dense += 1,
                FeatureOrigin::GeneratedSparse => widths.generated_sparse += 1,
            }
        }
        widths
    }

    pub fn with_one_to_one_values(&self, values: Rows, operation: &str) -> Result<Self> {
        if operation.trim().is_empty() {
            bail!("feature-lineage operation must not be blank");
        }
        Self::new(
            values,
            self.roles.clone(),
            self.missing_masks.clone(),
            self.sparse_columns.clone(),
            self.lineage
                .iter()
                .map(|lineage| lineage.with_operation(operation))
                .collect(),
            self.raw_source_width,
        )
    }

    pub fn with_generated_columns(
        &self,
        generated: Rows,
        roles: Vec<FeatureRole>,
        lineage: Vec<FeatureLineage>,
        append: bool,
    ) -> Result<Self> {
        if roles.len() != lineage.len() {
            bail!("generated feature role/lineage width mismatch");
        }
        if lineage
            .iter()
            .any(|item| item.origin == FeatureOrigin::RawDerived)
        {
            bail!("generated columns cannot have raw-derived origin");
        }
        let generated_width = roles.len();
        for row in &generated {
            if row.len() != generated_width {
                bail!("generated feature rows are ragged");
            }
        }
        if generated.len() != self.values.len() {
            bail!("generated feature row count mismatch");
        }

        let existing_dense = if append {
            self.widths().generated_dense
        } else {
            0
        };
        let added_dense = lineage
            .iter()
            .filter(|item| item.origin == FeatureOrigin::GeneratedDense)
            .count();
        Self::check_generated_dense_width(existing_dense + added_dense)?;

        let base_width = if append { self.roles.len() } else { 0 };
        let mut values = if append {
            self.values.clone()
        } else {
            vec![Vec::new(); generated.len()]
        };
        for (row, added) in values.iter_mut().zip(generated) {
            row.extend(added);
        }

        let mut output_roles = if append {
            self.roles.clone()
        } else {
            Vec::new()
        };
        output_roles.extend(roles);
        let mut missing_masks = if append {
            self.missing_masks.clone()
        } else {
            vec![Vec::new(); values.len()]
        };
        for mask in &mut missing_masks {
            mask.resize(output_roles.len(), false);
        }
        let mut sparse_columns = if append {
            self.sparse_columns.clone()
        } else {
            Vec::new()
        };
        sparse_columns.extend(lineage.iter().enumerate().filter_map(|(index, item)| {
            (item.origin == FeatureOrigin::GeneratedSparse).then_some(base_width + index)
        }));
        let mut output_lineage = if append {
            self.lineage.clone()
        } else {
            Vec::new()
        };
        output_lineage.extend(lineage);

        Self::new(
            values,
            output_roles,
            missing_masks,
            sparse_columns,
            output_lineage,
            self.raw_source_width,
        )
    }

    pub fn validate(&self) -> Result<()> {
        let width = self.roles.len();
        if self.lineage.len() != width {
            bail!("feature block lineage width mismatch");
        }
        if self.missing_masks.len() != self.values.len() {
            bail!("feature block missing-mask row count mismatch");
        }
        for row in &self.values {
            if row.len() != width {
                bail!("feature block is ragged");
            }
        }
        for row in &self.missing_masks {
            if row.len() != width {
                bail!("feature block missing mask is ragged");
            }
        }

        if self
            .sparse_columns
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        {
            bail!("feature block sparse indexes must be sorted and unique");
        }
        if self.sparse_columns.iter().any(|index| *index >= width) {
            bail!("feature block sparse index is out of bounds");
        }

        let expected_sparse: Vec<usize> = self
            .lineage
            .iter()
            .enumerate()
            .filter_map(|(index, lineage)| {
                (lineage.origin == FeatureOrigin::GeneratedSparse).then_some(index)
            })
            .collect();
        if self.sparse_columns != expected_sparse {
            bail!("feature block sparse indexes do not match column origins");
        }

        let mut raw_sources = BTreeSet::new();
        for (index, (role, lineage)) in self.roles.iter().zip(&self.lineage).enumerate() {
            if lineage.source_columns.is_empty() {
                bail!("feature column {index} has no source lineage");
            }
            if lineage
                .source_columns
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            {
                bail!("feature column {index} source lineage must be sorted and unique");
            }
            if lineage
                .source_columns
                .iter()
                .any(|source| *source >= self.raw_source_width)
            {
                bail!("feature column {index} source lineage is out of bounds");
            }
            if lineage.operations.is_empty()
                || lineage
                    .operations
                    .iter()
                    .any(|operation| operation.trim().is_empty())
            {
                bail!("feature column {index} has blank operation lineage");
            }

            match (role, lineage.origin) {
                (
                    FeatureRole::MissingIndicator | FeatureRole::CategoricalOneHot,
                    FeatureOrigin::GeneratedSparse,
                ) => {}
                (FeatureRole::MissingIndicator | FeatureRole::CategoricalOneHot, _) => {
                    bail!("sparse indicator role at column {index} has a nonsparse origin")
                }
                (FeatureRole::Generated, FeatureOrigin::RawDerived) => {
                    bail!("generated role at column {index} has raw-derived origin")
                }
                (
                    FeatureRole::Numeric
                    | FeatureRole::CategoricalNative
                    | FeatureRole::CategoricalFrequency
                    | FeatureRole::CategoricalOrdered,
                    FeatureOrigin::GeneratedSparse,
                ) => bail!("dense role at column {index} has generated-sparse origin"),
                _ => {}
            }

            if lineage.origin == FeatureOrigin::RawDerived {
                if lineage.source_columns.len() != 1 {
                    bail!("raw-derived column {index} must have exactly one source");
                }
                if !raw_sources.insert(lineage.source_columns[0]) {
                    bail!(
                        "raw source {} has multiple one-to-one columns",
                        lineage.source_columns[0]
                    );
                }
            }
        }
        Self::check_generated_dense_width(self.widths().generated_dense)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wide_raw_block(width: usize) -> FeatureBlock {
        FeatureBlock::new(
            vec![vec![1.0; width], vec![2.0; width]],
            vec![FeatureRole::Numeric; width],
            vec![vec![false; width]; 2],
            Vec::new(),
            (0..width)
                .map(|source| FeatureLineage::raw(source, "input:numeric"))
                .collect(),
            width,
        )
        .unwrap()
    }

    #[test]
    fn raw_derived_width_is_not_capped_at_512() {
        let block = wide_raw_block(1_776);
        assert_eq!(block.values[0].len(), 1_776);
        assert_eq!(block.widths().raw_derived, 1_776);
        assert_eq!(block.widths().generated_dense, 0);
        assert_eq!(block.widths().generated_sparse, 0);
    }

    #[test]
    fn generated_dense_width_is_preflighted() {
        assert!(FeatureBlock::check_generated_dense_width(511).is_ok());
        assert!(FeatureBlock::check_generated_dense_width(512).is_ok());
        assert!(FeatureBlock::check_generated_dense_width(513).is_err());
    }

    #[test]
    fn one_to_one_transform_preserves_metadata_without_truncation() {
        let block = wide_raw_block(1_024);
        let transformed = block
            .with_one_to_one_values(vec![vec![3.0; 1_024], vec![4.0; 1_024]], "maxabs")
            .unwrap();
        assert_eq!(transformed.values[0].len(), 1_024);
        assert_eq!(transformed.roles, block.roles);
        assert_eq!(transformed.missing_masks, block.missing_masks);
        assert_eq!(
            transformed.lineage[0].operations,
            ["input:numeric", "maxabs"]
        );
    }

    #[test]
    fn generated_columns_are_accounted_without_truncation() {
        let block = wide_raw_block(1_024);
        let generated = block
            .with_generated_columns(
                vec![vec![5.0, 1.0], vec![6.0, 0.0]],
                vec![FeatureRole::Generated, FeatureRole::MissingIndicator],
                vec![
                    FeatureLineage::generated_dense(vec![0], "derived:dense"),
                    FeatureLineage::generated_sparse(vec![1], "derived:sparse"),
                ],
                true,
            )
            .unwrap();
        assert_eq!(generated.values[0].len(), 1_026);
        assert_eq!(generated.widths().raw_derived, 1_024);
        assert_eq!(generated.widths().generated_dense, 1);
        assert_eq!(generated.widths().generated_sparse, 1);
        assert_eq!(generated.sparse_columns, [1_025]);
    }

    #[test]
    fn invalid_sparse_indexes_and_origins_are_rejected() {
        let mut block = wide_raw_block(2);
        block.sparse_columns = vec![1, 1];
        assert!(block.validate().is_err());

        let block = FeatureBlock::new(
            vec![vec![1.0]],
            vec![FeatureRole::MissingIndicator],
            vec![vec![false]],
            Vec::new(),
            vec![FeatureLineage::generated_dense(vec![0], "bad")],
            1,
        );
        assert!(block.is_err());

        let block = FeatureBlock::new(
            vec![vec![1.0]],
            vec![FeatureRole::Generated],
            vec![vec![true]],
            Vec::new(),
            vec![FeatureLineage::generated_dense(vec![0], "row-op-missing")],
            1,
        );
        assert!(block.is_ok());
    }
}
