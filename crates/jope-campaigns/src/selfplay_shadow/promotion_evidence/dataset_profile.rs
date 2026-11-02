use super::*;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetOriginV2 {
    Real,
    Synthetic,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LicenseDecisionV2 {
    AutomaticAllow,
    ManualApproval,
    Quarantine,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SyntheticFamilyKeyV2 {
    pub generator_name: String,
    pub generator_version: String,
    pub scenario: String,
    pub target_family: String,
    pub feature_family_set: BTreeSet<String>,
    pub shift_pathology_class: String,
}

impl SyntheticFamilyKeyV2 {
    pub fn exact_family_sha256(&self) -> Result<Sha256Digest> {
        ensure!(
            [
                self.generator_name.as_str(),
                self.generator_version.as_str(),
                self.scenario.as_str(),
                self.target_family.as_str(),
                self.shift_pathology_class.as_str(),
            ]
            .into_iter()
            .all(|value| !value.trim().is_empty())
                && !self.feature_family_set.is_empty(),
            "synthetic family key has an empty stable component"
        );
        Ok(Sha256Digest::of_bytes(&serde_json::to_vec(self)?))
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetProvenanceV2 {
    pub schema: String,
    pub provider: String,
    pub provider_version: String,
    pub upstream_root: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license_spdx: Option<String>,
    #[serde(default)]
    pub manual_license_approved: bool,
    pub raw_sha256: BTreeMap<String, Sha256Digest>,
    pub origin: DatasetOriginV2,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generator_superfamily: Option<String>,
    pub exact_family_sha256: Sha256Digest,
    pub near_duplicate_cluster_sha256: Sha256Digest,
}

impl DatasetProvenanceV2 {
    pub fn license_decision(&self) -> LicenseDecisionV2 {
        match self
            .license_spdx
            .as_deref()
            .map(|value| value.to_ascii_uppercase())
        {
            Some(value) if matches!(value.as_str(), "CC0-1.0" | "CC-BY-4.0" | "CC-BY-4.0-ONLY") => {
                LicenseDecisionV2::AutomaticAllow
            }
            Some(_) if self.manual_license_approved => LicenseDecisionV2::ManualApproval,
            _ => LicenseDecisionV2::Quarantine,
        }
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == DATASET_PROVENANCE_V2_SCHEMA,
            "dataset provenance schema is not v2"
        );
        ensure!(
            [
                self.provider.as_str(),
                self.provider_version.as_str(),
                self.upstream_root.as_str(),
            ]
            .into_iter()
            .all(|value| !value.trim().is_empty())
                && !self.raw_sha256.is_empty(),
            "dataset provenance is incomplete"
        );
        ensure!(
            self.origin == DatasetOriginV2::Synthetic || self.generator_superfamily.is_none(),
            "real datasets cannot claim a generator superfamily"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetSizeSliceV1 {
    Small,
    Medium,
    Large,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MissingnessSliceV1 {
    None,
    Low,
    High,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TargetShapeSliceV1 {
    Balanced,
    Imbalanced,
    LightTailed,
    HeavyTailed,
    Multimodal,
    Unknown,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetProfileV1 {
    pub schema: String,
    pub rows: usize,
    pub encoded_features: usize,
    pub missing_fraction: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub class_count: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minority_majority_ratio: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub numeric_target_distinct_finite: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finite_target_fraction: Option<f64>,
    pub size_slice: DatasetSizeSliceV1,
    pub missingness_slice: MissingnessSliceV1,
    pub target_shape_slice: TargetShapeSliceV1,
    pub embedding_novelty: f64,
    #[serde(default)]
    pub sensitive_or_pii_like: bool,
    #[serde(default)]
    pub ambiguous_target: bool,
    #[serde(default)]
    pub sparse_text: bool,
    #[serde(default)]
    pub grouped_or_time_series: bool,
    #[serde(default)]
    pub duplicate_or_leakage_prone: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetEligibilityV1 {
    pub eligible: bool,
    #[serde(default)]
    pub quarantine_reasons: Vec<String>,
}

impl DatasetProfileV1 {
    pub fn eligibility(
        &self,
        task: Task,
        provenance: &DatasetProvenanceV2,
    ) -> Result<DatasetEligibilityV1> {
        provenance.validate()?;
        ensure!(
            self.schema == DATASET_PROFILE_V1_SCHEMA,
            "dataset profile schema is not v1"
        );
        ensure!(
            self.missing_fraction.is_finite()
                && (0.0..=1.0).contains(&self.missing_fraction)
                && self.embedding_novelty.is_finite()
                && self
                    .minority_majority_ratio
                    .is_none_or(|value| value.is_finite())
                && self
                    .finite_target_fraction
                    .is_none_or(|value| value.is_finite()),
            "dataset profile contains a non-finite value"
        );
        let mut reasons = Vec::new();
        if provenance.license_decision() == LicenseDecisionV2::Quarantine {
            reasons.push("license is absent or not approved".to_owned());
        }
        if !(500..=100_000).contains(&self.rows) {
            reasons.push("row count is outside 500..100000".to_owned());
        }
        if self.encoded_features > 5_000 {
            reasons.push("encoded feature count exceeds 5000".to_owned());
        }
        for (condition, reason) in [
            (self.sensitive_or_pii_like, "sensitive or PII-like"),
            (self.ambiguous_target, "ambiguous target"),
            (self.sparse_text, "sparse text"),
            (self.grouped_or_time_series, "grouped or time-series"),
            (
                self.duplicate_or_leakage_prone,
                "duplicate or leakage-prone",
            ),
        ] {
            if condition {
                reasons.push(reason.to_owned());
            }
        }
        match task {
            Task::Binary => {
                if self.class_count != Some(2) {
                    reasons.push("binary task does not have exactly two classes".to_owned());
                }
                if self
                    .minority_majority_ratio
                    .is_none_or(|ratio| ratio < 0.05)
                {
                    reasons.push("minority/majority ratio is below 0.05".to_owned());
                }
            }
            Task::Regression => {
                if self
                    .numeric_target_distinct_finite
                    .is_none_or(|count| count < 50)
                {
                    reasons.push("regression target has fewer than 50 finite values".to_owned());
                }
                if self
                    .finite_target_fraction
                    .is_none_or(|fraction| fraction < 0.99)
                {
                    reasons.push("finite regression targets are below 99%".to_owned());
                }
            }
            // The OpenML acquisition lane only sources the two legacy tasks;
            // quarantine anything else fail-closed until per-task eligibility
            // rules exist for the new lanes.
            Task::Timeseries | Task::Multiclass => {
                reasons.push(format!(
                    "{task} datasets are not eligible for the OpenML acquisition lane"
                ));
            }
        }
        Ok(DatasetEligibilityV1 {
            eligible: reasons.is_empty(),
            quarantine_reasons: reasons,
        })
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct OpenMlDatasetMetadataV1 {
    pub dataset_id: u64,
    pub version: u64,
    pub name: String,
    #[serde(alias = "licence")]
    pub license: Option<String>,
    pub default_target_attribute: Option<String>,
    pub original_data_url: Option<String>,
    pub download_url: String,
    pub rows: usize,
    pub features: usize,
    pub missing_values: usize,
}

impl OpenMlDatasetMetadataV1 {
    pub fn deterministic_key(&self) -> (u64, u64, &str) {
        (self.dataset_id, self.version, self.name.as_str())
    }

    pub fn discovery_quarantine_reasons(&self) -> Vec<String> {
        let mut reasons = Vec::new();
        if self.version == 0 {
            reasons.push("OpenML dataset version is not immutable".to_owned());
        }
        if self.license.as_deref().is_none_or(str::is_empty) {
            reasons.push("OpenML license metadata is absent".to_owned());
        }
        if self
            .default_target_attribute
            .as_deref()
            .is_none_or(str::is_empty)
        {
            reasons.push("OpenML default target is absent".to_owned());
        }
        if !(500..=100_000).contains(&self.rows) {
            reasons.push("OpenML row count is outside initial eligibility".to_owned());
        }
        reasons
    }
}

pub fn deterministic_openml_discovery(
    mut datasets: Vec<OpenMlDatasetMetadataV1>,
) -> Vec<OpenMlDatasetMetadataV1> {
    datasets.sort_by(|left, right| left.deterministic_key().cmp(&right.deterministic_key()));
    datasets.dedup_by(|left, right| {
        left.dataset_id == right.dataset_id && left.version == right.version
    });
    datasets
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct UciDatasetMetadataV1 {
    pub persistent_id: String,
    pub version: String,
    pub name: String,
    pub license: Option<String>,
    pub dataset_url: String,
    pub target_columns: Vec<String>,
    pub rows: usize,
    pub features: usize,
}

impl UciDatasetMetadataV1 {
    pub fn discovery_quarantine_reasons(&self) -> Vec<String> {
        let mut reasons = Vec::new();
        if self.persistent_id.trim().is_empty() || self.version.trim().is_empty() {
            reasons.push("UCI dataset lacks an immutable identity/version".to_owned());
        }
        if self.license.as_deref().is_none_or(str::is_empty) {
            reasons.push("UCI per-dataset license is absent".to_owned());
        }
        if self.target_columns.len() != 1 {
            reasons.push("UCI target is absent or ambiguous".to_owned());
        }
        if !(500..=100_000).contains(&self.rows) {
            reasons.push("UCI row count is outside initial eligibility".to_owned());
        }
        reasons
    }
}

pub fn deterministic_uci_discovery(
    mut datasets: Vec<UciDatasetMetadataV1>,
) -> Vec<UciDatasetMetadataV1> {
    datasets.sort_by(|left, right| {
        (&left.persistent_id, &left.version, &left.name).cmp(&(
            &right.persistent_id,
            &right.version,
            &right.name,
        ))
    });
    datasets.dedup_by(|left, right| {
        left.persistent_id == right.persistent_id && left.version == right.version
    });
    datasets
}

pub fn cross_provider_duplicate_cluster(
    raw_sha256: &BTreeMap<String, Sha256Digest>,
) -> Result<Sha256Digest> {
    ensure!(
        !raw_sha256.is_empty(),
        "duplicate clustering requires raw content hashes"
    );
    let mut digests = raw_sha256.values().collect::<Vec<_>>();
    digests.sort();
    digests.dedup();
    Ok(Sha256Digest::of_bytes(&serde_json::to_vec(&digests)?))
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct CurationCandidateV1 {
    pub dataset_sha256: Sha256Digest,
    pub dimensions: BTreeMap<String, String>,
    pub embedding_novelty: f64,
}

fn curation_distance(left: &CurationCandidateV1, right: &CurationCandidateV1) -> f64 {
    let dimensions = left
        .dimensions
        .keys()
        .chain(right.dimensions.keys())
        .collect::<BTreeSet<_>>();
    let categorical = dimensions
        .into_iter()
        .filter(|dimension| left.dimensions.get(*dimension) != right.dimensions.get(*dimension))
        .count() as f64;
    categorical + (left.embedding_novelty - right.embedding_novelty).abs()
}

pub fn deterministic_max_min_selection(
    candidates: &[CurationCandidateV1],
    count: usize,
) -> Result<Vec<Sha256Digest>> {
    ensure!(
        count <= candidates.len()
            && candidates
                .iter()
                .all(|candidate| candidate.embedding_novelty.is_finite()),
        "invalid deterministic max-min request"
    );
    if count == 0 {
        return Ok(Vec::new());
    }
    let mut remaining = candidates.iter().collect::<Vec<_>>();
    remaining.sort_by(|left, right| left.dataset_sha256.cmp(&right.dataset_sha256));
    let mut selected = vec![remaining.remove(0)];
    while selected.len() < count {
        let (index, _) = remaining
            .iter()
            .enumerate()
            .map(|(index, candidate)| {
                let distance = selected
                    .iter()
                    .map(|known| curation_distance(candidate, known))
                    .fold(f64::INFINITY, f64::min);
                (index, distance)
            })
            .max_by(
                |(left_index, left_distance), (right_index, right_distance)| {
                    left_distance.total_cmp(right_distance).then_with(|| {
                        remaining[*right_index]
                            .dataset_sha256
                            .cmp(&remaining[*left_index].dataset_sha256)
                    })
                },
            )
            .context("max-min selection exhausted candidates")?;
        selected.push(remaining.remove(index));
    }
    Ok(selected
        .into_iter()
        .map(|candidate| candidate.dataset_sha256.clone())
        .collect())
}
