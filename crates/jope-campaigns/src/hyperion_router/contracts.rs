use super::*;

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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroupedOuterFoldV2 {
    pub fold_index: u32,
    pub fit_rows: Vec<usize>,
    pub hold_rows: Vec<usize>,
    pub fit_groups: Vec<String>,
    pub hold_groups: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroupedOuterPlanV2 {
    pub schema: String,
    pub seed: u64,
    pub row_count: usize,
    pub atomic_group_count: usize,
    pub row_atomic_groups: Vec<String>,
    pub folds: Vec<GroupedOuterFoldV2>,
}

impl GroupedOuterPlanV2 {
    pub const SCHEMA: &'static str = "HyperionGroupedOuterPlan.v2";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA
            || !ROUTER_SEEDS.contains(&self.seed)
            || self.row_count != self.row_atomic_groups.len()
            || self.row_count == 0
            || self.folds.len() != OUTER_FOLDS
            || self.row_atomic_groups.iter().any(|group| group.is_empty())
        {
            return Err("grouped outer plan schema, seed, or shape drifted".into());
        }
        let all_groups = self
            .row_atomic_groups
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        if all_groups.len() != self.atomic_group_count
            || self.atomic_group_count < MINIMUM_ATOMIC_GROUPS
        {
            return Err("grouped outer plan has fewer than fourteen atomic groups".into());
        }
        let mut coverage = vec![0u8; self.row_count];
        for (index, fold) in self.folds.iter().enumerate() {
            if fold.fold_index as usize != index
                || fold.fit_rows.len() < MINIMUM_FIT_ROWS
                || fold.hold_rows.len() < MINIMUM_HOLD_ROWS
                || fold.hold_groups.len() < MINIMUM_HOLD_GROUPS
            {
                return Err(format!(
                    "outer fold {index} violates the minimum structural guards"
                ));
            }
            let fit_rows = fold.fit_rows.iter().copied().collect::<BTreeSet<_>>();
            let hold_rows = fold.hold_rows.iter().copied().collect::<BTreeSet<_>>();
            if fit_rows.len() != fold.fit_rows.len()
                || hold_rows.len() != fold.hold_rows.len()
                || !fit_rows.is_disjoint(&hold_rows)
                || fit_rows.len() + hold_rows.len() != self.row_count
                || fit_rows.union(&hold_rows).copied().ne(0..self.row_count)
            {
                return Err(format!(
                    "outer fold {index} does not partition every row once"
                ));
            }
            let expected_hold_groups = hold_rows
                .iter()
                .map(|row| self.row_atomic_groups[*row].clone())
                .collect::<BTreeSet<_>>();
            let expected_fit_groups = fit_rows
                .iter()
                .map(|row| self.row_atomic_groups[*row].clone())
                .collect::<BTreeSet<_>>();
            let actual_hold_groups = fold.hold_groups.iter().cloned().collect::<BTreeSet<_>>();
            let actual_fit_groups = fold.fit_groups.iter().cloned().collect::<BTreeSet<_>>();
            if expected_hold_groups != actual_hold_groups
                || expected_fit_groups != actual_fit_groups
                || !actual_hold_groups.is_disjoint(&actual_fit_groups)
                || actual_hold_groups.len() != fold.hold_groups.len()
                || actual_fit_groups.len() != fold.fit_groups.len()
            {
                return Err(format!(
                    "outer fold {index} leaks or misstates atomic groups"
                ));
            }
            for row in hold_rows {
                coverage[row] = coverage[row].saturating_add(1);
            }
        }
        if coverage.iter().any(|count| *count != 1) {
            return Err(
                "outer fold bank does not provide exactly one hold prediction per row".into(),
            );
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

fn seeded_group_key(seed: u64, group: &str) -> String {
    raw_sha256(format!("hyperion-grouped-cv7-v2|{seed}|{group}").as_bytes())
}

pub fn build_grouped_outer_plan_v2(
    row_atomic_groups: &[String],
    seed: u64,
) -> Result<GroupedOuterPlanV2, String> {
    if !ROUTER_SEEDS.contains(&seed) {
        return Err("outer plan seed is outside the frozen 303/404 bank".into());
    }
    if row_atomic_groups.is_empty() || row_atomic_groups.iter().any(|group| group.is_empty()) {
        return Err("row atomic groups are empty".into());
    }
    let mut group_rows = BTreeMap::<String, Vec<usize>>::new();
    for (row, group) in row_atomic_groups.iter().enumerate() {
        group_rows.entry(group.clone()).or_default().push(row);
    }
    if group_rows.len() < MINIMUM_ATOMIC_GROUPS {
        return Err("Hyperion routing requires at least fourteen atomic groups".into());
    }
    let mut ordered = group_rows.keys().cloned().collect::<Vec<_>>();
    ordered.sort_by(|a, b| {
        seeded_group_key(seed, a)
            .cmp(&seeded_group_key(seed, b))
            .then(a.cmp(b))
    });
    let mut fold_groups = vec![Vec::<String>::new(); OUTER_FOLDS];
    let mut fold_rows = [0usize; OUTER_FOLDS];
    for (index, group) in ordered.into_iter().enumerate() {
        let fold = if index < OUTER_FOLDS * MINIMUM_HOLD_GROUPS {
            index % OUTER_FOLDS
        } else {
            (0..OUTER_FOLDS)
                .min_by_key(|fold| (fold_rows[*fold], fold_groups[*fold].len(), *fold))
                .expect("seven folds")
        };
        fold_rows[fold] += group_rows[&group].len();
        fold_groups[fold].push(group);
    }
    let all_rows = (0..row_atomic_groups.len()).collect::<BTreeSet<_>>();
    let all_groups = group_rows.keys().cloned().collect::<BTreeSet<_>>();
    let folds = fold_groups
        .into_iter()
        .enumerate()
        .map(|(fold_index, groups)| {
            let hold_groups = groups.into_iter().collect::<BTreeSet<_>>();
            let hold_rows = hold_groups
                .iter()
                .flat_map(|group| group_rows[group].iter().copied())
                .collect::<BTreeSet<_>>();
            let fit_rows = all_rows.difference(&hold_rows).copied().collect::<Vec<_>>();
            GroupedOuterFoldV2 {
                fold_index: fold_index as u32,
                fit_rows,
                hold_rows: hold_rows.into_iter().collect(),
                fit_groups: all_groups.difference(&hold_groups).cloned().collect(),
                hold_groups: hold_groups.into_iter().collect(),
            }
        })
        .collect();
    let plan = GroupedOuterPlanV2 {
        schema: GroupedOuterPlanV2::SCHEMA.into(),
        seed,
        row_count: row_atomic_groups.len(),
        atomic_group_count: group_rows.len(),
        row_atomic_groups: row_atomic_groups.to_vec(),
        folds,
    };
    plan.validate()?;
    Ok(plan)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroupedFoldBankV2 {
    pub schema: String,
    pub plans: Vec<GroupedOuterPlanV2>,
}

impl GroupedFoldBankV2 {
    pub const SCHEMA: &'static str = "HyperionGroupedFoldBank.v2";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA || self.plans.len() != ROUTER_SEEDS.len() {
            return Err("grouped fold bank schema or plan cardinality drifted".into());
        }
        for (index, plan) in self.plans.iter().enumerate() {
            plan.validate()?;
            if plan.seed != ROUTER_SEEDS[index] {
                return Err("grouped fold bank seed order drifted".into());
            }
            if index > 0
                && (plan.row_count != self.plans[0].row_count
                    || plan.row_atomic_groups != self.plans[0].row_atomic_groups)
            {
                return Err(
                    "grouped fold bank plans do not describe the same rows and groups".into(),
                );
            }
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

pub fn build_grouped_fold_bank_v2(
    row_atomic_groups: &[String],
) -> Result<GroupedFoldBankV2, String> {
    let bank = GroupedFoldBankV2 {
        schema: GroupedFoldBankV2::SCHEMA.into(),
        plans: ROUTER_SEEDS
            .iter()
            .map(|seed| build_grouped_outer_plan_v2(row_atomic_groups, *seed))
            .collect::<Result<Vec<_>, _>>()?,
    };
    bank.validate()?;
    Ok(bank)
}
