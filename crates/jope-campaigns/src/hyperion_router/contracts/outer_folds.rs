use super::*;

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
