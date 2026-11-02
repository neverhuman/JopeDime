use super::*;

/// A single-read sealed boundary. The private labels stay in the finalizer
/// object; the only method intended for the encoder accepts feature rows and
/// appends constant dummy targets.
#[derive(Debug, Clone, PartialEq)]
pub struct HyperionSealedFeatureStageV1 {
    pub schema: String,
    pub dataset_sha256: String,
    pub test_sha256: String,
    pub feature_count: usize,
    pub feature_rows: Vec<Vec<f64>>,
    labels: Vec<f64>,
    pub test_access_ordinal: u32,
}

impl HyperionSealedFeatureStageV1 {
    pub const SCHEMA: &'static str = "HyperionSealedFeatureStage.v1";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA
            || self.test_access_ordinal != 1
            || self.feature_count == 0
            || self.feature_rows.is_empty()
            || self.labels.len() != self.feature_rows.len()
            || self.feature_rows.iter().any(|row| {
                row.len() != self.feature_count || row.iter().any(|value| !value.is_finite())
            })
            || self.labels.iter().any(|value| !value.is_finite())
        {
            return Err("sealed feature stage has invalid shape or access boundary".into());
        }
        validate_sha256("sealed dataset", &self.dataset_sha256)?;
        validate_sha256("sealed test", &self.test_sha256)
    }

    pub fn hyperion_dummy_target_rows(&self) -> Result<Vec<Vec<f64>>, String> {
        self.validate()?;
        Ok(self
            .feature_rows
            .iter()
            .map(|row| {
                let mut output = row.clone();
                output.push(0.0);
                output
            })
            .collect())
    }

    pub fn finalizer_labels(&self) -> Result<&[f64], String> {
        self.validate()?;
        Ok(&self.labels)
    }
}

/// Parse the already-opened sealed bytes exactly once. Feature bytes are
/// materialized before labels are retained, and the resulting stage has one
/// immutable access ordinal for the finalizer receipt.
pub fn stage_hyperion_sealed_features(
    dataset_sha256: &str,
    test_sha256: &str,
    bytes: &[u8],
    feature_count: usize,
) -> Result<HyperionSealedFeatureStageV1, String> {
    validate_sha256("sealed dataset", dataset_sha256)?;
    validate_sha256("sealed test", test_sha256)?;
    if feature_count == 0 || bytes.is_empty() {
        return Err("sealed test bytes or feature count is empty".into());
    }
    let mut feature_rows = Vec::new();
    let mut labels = Vec::new();
    for (line_index, line) in std::str::from_utf8(bytes)
        .map_err(|error| format!("sealed test is not UTF-8: {error}"))?
        .lines()
        .enumerate()
    {
        if line.trim().is_empty() {
            continue;
        }
        let parsed = line
            .split(',')
            .map(|field| field.trim().parse::<f64>())
            .collect::<Result<Vec<_>, _>>();
        let mut values = match parsed {
            Ok(values) => values,
            Err(_) if line_index == 0 => continue,
            Err(error) => return Err(format!("sealed test row {line_index} is invalid: {error}")),
        };
        if values.len() != feature_count + 1 {
            return Err(format!(
                "sealed test row {line_index} has {} columns, expected {}",
                values.len(),
                feature_count + 1
            ));
        }
        let label = values.pop().expect("feature count checked");
        if !label.is_finite() || values.iter().any(|value| !value.is_finite()) {
            return Err(format!(
                "sealed test row {line_index} contains a nonfinite value"
            ));
        }
        feature_rows.push(values);
        labels.push(label);
    }
    let stage = HyperionSealedFeatureStageV1 {
        schema: HyperionSealedFeatureStageV1::SCHEMA.into(),
        dataset_sha256: dataset_sha256.into(),
        test_sha256: test_sha256.into(),
        feature_count,
        feature_rows,
        labels,
        test_access_ordinal: 1,
    };
    stage.validate()?;
    Ok(stage)
}

/// The fallback decision is made from encoder/shape proof only. It accepts no
/// labels and therefore cannot be retuned by sealed outcomes.
pub fn choose_predeclared_raw_fallback(encoding_verified: bool, shape_verified: bool) -> bool {
    !(encoding_verified && shape_verified)
}

/// The sealed finalizer may only accept a fresh full-train encoder result if
/// its bytes are identical to the registry-frozen full-train cache.  This
/// comparison is deliberately independent of test labels.
pub fn verify_full_train_replay(
    expected_cache_sha256: &str,
    fresh_train_bytes: &[u8],
) -> Result<String, String> {
    validate_sha256("full-train cache", expected_cache_sha256)?;
    if fresh_train_bytes.is_empty() {
        return Err("fresh full-train encoder output is empty".into());
    }
    let actual = raw_sha256(fresh_train_bytes);
    if actual != expected_cache_sha256 {
        return Err("fresh full-train encoder output differs from frozen cache".into());
    }
    Ok(actual)
}
