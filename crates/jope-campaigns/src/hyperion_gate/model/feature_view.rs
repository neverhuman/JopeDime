use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureViewKindV1 {
    Raw,
    HyperionCell,
    HyperionRow128,
    HyperionBoth,
}

impl FeatureViewKindV1 {
    pub const ALL: [Self; 4] = [
        Self::Raw,
        Self::HyperionCell,
        Self::HyperionRow128,
        Self::HyperionBoth,
    ];

    pub const EXPANDED: [Self; 3] = [Self::HyperionCell, Self::HyperionRow128, Self::HyperionBoth];
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JopeFeatureViewV1 {
    pub schema: String,
    pub view: FeatureViewKindV1,
    pub raw_features: u32,
    pub output_features: u32,
    pub cell_preprocessing: Option<String>,
    pub row_preprocessing: Option<String>,
}

impl JopeFeatureViewV1 {
    pub const SCHEMA: &'static str = "JopeFeatureView.v1";
    pub const CELL_PREPROCESSING: &'static str = "fold_train_empirical_percentile_per_column.v1";
    pub const ROW_PREPROCESSING: &'static str = "fold_train_zscore_per_dimension.v1";

    pub fn new(view: FeatureViewKindV1, raw_features: usize) -> Result<Self, String> {
        if raw_features == 0 || raw_features > u32::MAX as usize {
            return Err("feature view requires a positive representable raw width".into());
        }
        let (cell, row, output) = match view {
            FeatureViewKindV1::Raw => (None, None, raw_features),
            FeatureViewKindV1::HyperionCell => (
                Some(Self::CELL_PREPROCESSING.into()),
                None,
                raw_features * 2,
            ),
            FeatureViewKindV1::HyperionRow128 => (
                None,
                Some(Self::ROW_PREPROCESSING.into()),
                raw_features + ROW_EMBEDDING_WIDTH,
            ),
            FeatureViewKindV1::HyperionBoth => (
                Some(Self::CELL_PREPROCESSING.into()),
                Some(Self::ROW_PREPROCESSING.into()),
                raw_features * 2 + ROW_EMBEDDING_WIDTH,
            ),
        };
        let value = Self {
            schema: Self::SCHEMA.into(),
            view,
            raw_features: raw_features as u32,
            output_features: u32::try_from(output)
                .map_err(|_| "expanded feature width is not representable")?,
            cell_preprocessing: cell,
            row_preprocessing: row,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA || self.raw_features == 0 {
            return Err("feature view schema or raw width is invalid".into());
        }
        let expected = Self::new_unchecked(self.view, self.raw_features);
        if self.output_features != expected.output_features
            || self.cell_preprocessing != expected.cell_preprocessing
            || self.row_preprocessing != expected.row_preprocessing
        {
            return Err("feature view layout or preprocessing policy drifted".into());
        }
        Ok(())
    }

    fn new_unchecked(view: FeatureViewKindV1, raw: u32) -> Self {
        let raw_usize = raw as usize;
        let (cell, row, output) = match view {
            FeatureViewKindV1::Raw => (None, None, raw_usize),
            FeatureViewKindV1::HyperionCell => {
                (Some(Self::CELL_PREPROCESSING.into()), None, raw_usize * 2)
            }
            FeatureViewKindV1::HyperionRow128 => (
                None,
                Some(Self::ROW_PREPROCESSING.into()),
                raw_usize + ROW_EMBEDDING_WIDTH,
            ),
            FeatureViewKindV1::HyperionBoth => (
                Some(Self::CELL_PREPROCESSING.into()),
                Some(Self::ROW_PREPROCESSING.into()),
                raw_usize * 2 + ROW_EMBEDDING_WIDTH,
            ),
        };
        Self {
            schema: Self::SCHEMA.into(),
            view,
            raw_features: raw,
            output_features: output as u32,
            cell_preprocessing: cell,
            row_preprocessing: row,
        }
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

pub fn validate_feature_view_feasibility(
    view: FeatureViewKindV1,
    raw_features: usize,
    fit_rows: usize,
) -> Result<(), String> {
    if view == FeatureViewKindV1::Raw {
        return Ok(());
    }
    if fit_rows < MINIMUM_FOLD_FIT_ROWS {
        return Err(format!(
            "expanded view has {fit_rows} fit rows; minimum is {MINIMUM_FOLD_FIT_ROWS}"
        ));
    }
    JopeFeatureViewV1::new(view, raw_features)?;

    // The cell block is one signal per existing raw column.  A dataset that is
    // already wide (p > n) must not lose this candidate merely because adding
    // its aligned cell block keeps it wide.  The frozen AFPM heads already
    // support wide inputs and the paired CV gate, not an arbitrary p <= n
    // rule, decides whether the block is useful.  Row128 is different: it is a
    // dense, unaligned representation, so retain the conservative fit-row
    // guard for any view that includes it.
    if matches!(
        view,
        FeatureViewKindV1::HyperionRow128 | FeatureViewKindV1::HyperionBoth
    ) && ROW_EMBEDDING_WIDTH > fit_rows
    {
        return Err(format!(
            "row128 view adds {ROW_EMBEDDING_WIDTH} dense features but fold has only {fit_rows} fit rows"
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MatrixArtifactV1 {
    pub role: String,
    pub rows: u32,
    pub columns: u32,
    pub sha256: String,
}

impl MatrixArtifactV1 {
    pub fn validate(&self) -> Result<(), String> {
        if self.role.trim().is_empty() || self.rows == 0 || self.columns == 0 {
            return Err("sidecar matrix identity or shape is incomplete".into());
        }
        validate_sha256("sidecar matrix", &self.sha256)
    }
}
