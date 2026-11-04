use super::*;

mod outer_folds;
pub use outer_folds::{
    build_grouped_fold_bank_v2, build_grouped_outer_plan_v2, GroupedFoldBankV2, GroupedOuterFoldV2,
    GroupedOuterPlanV2,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureViewKindV2 {
    Raw,
    HyperionCellOnly,
    HyperionCellAppend,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CellEvidenceScopeV2 {
    HistoricalFullTrainDiagnostic,
    FoldLocalOuterFit,
}

/// V2 binds the real `train3.vs` bytes even for the matched raw arm. This is
/// what makes three separately executed views a single matched experiment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeatureViewContractV2 {
    pub schema: String,
    pub view: FeatureViewKindV2,
    pub train_sha256: String,
    pub real_cell_sha256: String,
    pub encoder_identity_sha256: String,
    pub rows: usize,
    pub raw_width: usize,
    pub cell_width: usize,
    pub output_width: usize,
    pub cell_scope: CellEvidenceScopeV2,
    pub test_material_present: bool,
}

impl FeatureViewContractV2 {
    pub const SCHEMA: &'static str = "JopeFeatureView.v2";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA || self.test_material_present {
            return Err("V2 feature view schema drifted or contains test material".into());
        }
        for (name, value) in [
            ("feature-view train", self.train_sha256.as_str()),
            ("feature-view real cell", self.real_cell_sha256.as_str()),
            (
                "feature-view encoder",
                self.encoder_identity_sha256.as_str(),
            ),
        ] {
            validate_sha256(name, value)?;
        }
        if self.rows == 0
            || self.raw_width == 0
            || self.cell_width != self.raw_width
            || self.output_width
                != match self.view {
                    FeatureViewKindV2::Raw | FeatureViewKindV2::HyperionCellOnly => self.raw_width,
                    FeatureViewKindV2::HyperionCellAppend => self.raw_width * 2,
                }
        {
            return Err("V2 feature-view dimensions are invalid".into());
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }

    pub fn can_support_activation(&self) -> bool {
        self.cell_scope == CellEvidenceScopeV2::FoldLocalOuterFit
    }
}

pub fn assemble_feature_view_v2(
    contract: &FeatureViewContractV2,
    raw: &[Vec<f64>],
    real_cell: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, String> {
    contract.validate()?;
    let raw_shape = matrix_shape("raw features", raw)?;
    let cell_shape = matrix_shape("real Hyperion cell features", real_cell)?;
    if raw_shape != (contract.rows, contract.raw_width)
        || cell_shape != (contract.rows, contract.cell_width)
    {
        return Err("feature-view matrix shape differs from its V2 contract".into());
    }
    Ok(match contract.view {
        FeatureViewKindV2::Raw => raw.to_vec(),
        FeatureViewKindV2::HyperionCellOnly => real_cell.to_vec(),
        FeatureViewKindV2::HyperionCellAppend => raw
            .iter()
            .zip(real_cell)
            .map(|(raw_row, cell_row)| {
                let mut row = Vec::with_capacity(contract.output_width);
                row.extend_from_slice(raw_row);
                row.extend_from_slice(cell_row);
                row
            })
            .collect(),
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DevelopmentGenomeObservationV1 {
    pub development_dataset_sha256: String,
    pub canonical_genome: String,
    pub semantic_sha256: String,
    pub leave_one_dataset_percentile_micros: u32,
    pub production_compatible: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FixedCandidateV1 {
    pub rank: u32,
    pub canonical_genome: String,
    pub semantic_sha256: String,
    pub median_leave_one_dataset_percentile_micros: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FixedCandidateBankV1 {
    pub schema: String,
    pub development_dataset_sha256s: Vec<String>,
    pub candidates: Vec<FixedCandidateV1>,
}

impl FixedCandidateBankV1 {
    pub const SCHEMA: &'static str = "HyperionFixedCandidateBank.v1";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA
            || self.development_dataset_sha256s.len() != DEVELOPMENT_DATASET_COUNT
            || self.candidates.len() != FIXED_BANK_SIZE
        {
            return Err("fixed candidate bank schema or cardinality drifted".into());
        }
        let development = self
            .development_dataset_sha256s
            .iter()
            .collect::<BTreeSet<_>>();
        if development.len() != DEVELOPMENT_DATASET_COUNT {
            return Err("fixed candidate bank development cohort is duplicated".into());
        }
        for digest in development {
            validate_sha256("development dataset", digest)?;
        }
        let mut semantics = BTreeSet::new();
        for (index, candidate) in self.candidates.iter().enumerate() {
            validate_sha256("candidate semantic", &candidate.semantic_sha256)?;
            if candidate.rank as usize != index + 1
                || candidate.canonical_genome.trim().is_empty()
                || candidate.median_leave_one_dataset_percentile_micros > 1_000_000
                || !semantics.insert(candidate.semantic_sha256.as_str())
            {
                return Err("fixed candidate bank entry is invalid or duplicated".into());
            }
            if index > 0 {
                let prior = &self.candidates[index - 1];
                if prior.median_leave_one_dataset_percentile_micros
                    < candidate.median_leave_one_dataset_percentile_micros
                    || (prior.median_leave_one_dataset_percentile_micros
                        == candidate.median_leave_one_dataset_percentile_micros
                        && prior.semantic_sha256 >= candidate.semantic_sha256)
                {
                    return Err(
                        "fixed candidate bank is not in its deterministic rank order".into(),
                    );
                }
            }
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

fn integer_median(values: &mut [u32]) -> u32 {
    values.sort_unstable();
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        ((values[middle - 1] as u64 + values[middle] as u64) / 2) as u32
    } else {
        values[middle]
    }
}

/// Build the fixed bank from complete leave-one-development-dataset evidence.
/// Any missing development score excludes that genome rather than imputing it.
pub fn build_fixed_candidate_bank(
    development_dataset_sha256s: &[String],
    observations: &[DevelopmentGenomeObservationV1],
) -> Result<FixedCandidateBankV1, String> {
    let development = development_dataset_sha256s
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    if development.len() != DEVELOPMENT_DATASET_COUNT {
        return Err("candidate bank requires exactly 22 unique development datasets".into());
    }
    for digest in &development {
        validate_sha256("development dataset", digest)?;
    }
    let mut grouped: BTreeMap<String, (String, BTreeMap<String, u32>)> = BTreeMap::new();
    for observation in observations {
        if !observation.production_compatible {
            continue;
        }
        validate_sha256("candidate semantic", &observation.semantic_sha256)?;
        validate_sha256(
            "candidate development dataset",
            &observation.development_dataset_sha256,
        )?;
        if !development.contains(&observation.development_dataset_sha256)
            || observation.leave_one_dataset_percentile_micros > 1_000_000
            || observation.canonical_genome.trim().is_empty()
        {
            return Err("candidate observation is foreign or invalid".into());
        }
        let (canonical, scores) = grouped
            .entry(observation.semantic_sha256.clone())
            .or_insert_with(|| (observation.canonical_genome.clone(), BTreeMap::new()));
        if canonical != &observation.canonical_genome
            || scores
                .insert(
                    observation.development_dataset_sha256.clone(),
                    observation.leave_one_dataset_percentile_micros,
                )
                .is_some()
        {
            return Err("candidate semantic has conflicting or duplicate observations".into());
        }
    }
    let mut eligible = grouped
        .into_iter()
        .filter_map(|(semantic, (canonical, scores))| {
            if scores.len() != DEVELOPMENT_DATASET_COUNT {
                return None;
            }
            let mut values = scores.into_values().collect::<Vec<_>>();
            Some((integer_median(&mut values), semantic, canonical))
        })
        .collect::<Vec<_>>();
    eligible.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    if eligible.len() < FIXED_BANK_SIZE {
        return Err(
            "fewer than twelve production-compatible genomes have complete evidence".into(),
        );
    }
    let candidates = eligible
        .into_iter()
        .take(FIXED_BANK_SIZE)
        .enumerate()
        .map(
            |(index, (median, semantic_sha256, canonical_genome))| FixedCandidateV1 {
                rank: (index + 1) as u32,
                canonical_genome,
                semantic_sha256,
                median_leave_one_dataset_percentile_micros: median,
            },
        )
        .collect();
    let bank = FixedCandidateBankV1 {
        schema: FixedCandidateBankV1::SCHEMA.into(),
        development_dataset_sha256s: development.into_iter().collect(),
        candidates,
    };
    bank.validate()?;
    Ok(bank)
}
