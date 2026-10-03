//! Train-side selection closure. No target values or outer validation features
//! enter this plan. Exact ordered row identities come from the frozen panel.
use crate::{
    experiment::{EvaluationRequest, EvaluationStage},
    sha256_hex,
};
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const INNER_GROUP_POLICY: &str = "asm.inner-groups/largest-first-lexicographic-balance-five/v1";
/// Deterministic group-preserving assignment, without target values.
pub fn grouped_inner_folds(rows: &[(String, String)]) -> Result<Vec<Vec<String>>> {
    let mut groups = std::collections::BTreeMap::<&str, usize>::new();
    for (_, group) in rows {
        *groups.entry(group).or_default() += 1;
    }
    ensure!(
        groups.len() >= 5,
        "fewer than five training groups for OOF selection"
    );
    let mut groups = groups.into_iter().collect::<Vec<_>>();
    groups.sort_by(|(a, an), (b, bn)| bn.cmp(an).then(a.cmp(b)));
    let mut sizes = [0usize; 5];
    let mut assignment = std::collections::BTreeMap::new();
    for (group, n) in groups {
        let fold = (0..5).min_by_key(|&i| (sizes[i], i)).expect("five folds");
        sizes[fold] += n;
        assignment.insert(group, fold);
    }
    let mut folds = vec![vec![]; 5];
    for (id, group) in rows {
        folds[assignment[group.as_str()]].push(id.clone());
    }
    Ok(folds)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectionPlan {
    outer: EvaluationRequest,
    fitting_rows: Vec<String>,
    hold_rows: Vec<String>,
    inner_validation_rows: Vec<Vec<String>>,
    fitting_groups: Vec<String>,
    hold_groups: Vec<String>,
}
impl SelectionPlan {
    pub fn new(
        outer: EvaluationRequest,
        fitting_rows: Vec<String>,
        hold_rows: Vec<String>,
        inner_validation_rows: Vec<Vec<String>>,
        fitting_groups: Vec<String>,
        hold_groups: Vec<String>,
    ) -> Result<Self> {
        let plan = Self {
            outer,
            fitting_rows,
            hold_rows,
            inner_validation_rows,
            fitting_groups,
            hold_groups,
        };
        plan.validate()?;
        Ok(plan)
    }
    pub fn validate(&self) -> Result<()> {
        self.outer.validate()?;
        ensure!(
            matches!(
                self.outer.stage,
                EvaluationStage::Screen | EvaluationStage::Confirm
            ),
            "outer request must be a frozen development fold"
        );
        ensure!(
            self.fitting_rows.len() == self.outer.fitting_rows
                && self.hold_rows.len() == self.outer.scoring_rows
                && rows_sha(&self.fitting_rows)? == self.outer.fitting_rows_sha256
                && rows_sha(&self.hold_rows)? == self.outer.scoring_rows_sha256,
            "outer row assignments differ from frozen panel request"
        );
        ensure!(
            self.fitting_rows
                .iter()
                .chain(&self.hold_rows)
                .all(|r| crate::experiment::sha256(r)),
            "invalid original panel row ID"
        );
        let train: BTreeSet<_> = self.fitting_rows.iter().cloned().collect();
        let hold: BTreeSet<_> = self.hold_rows.iter().cloned().collect();
        ensure!(
            train.len() == self.fitting_rows.len()
                && hold.len() == self.hold_rows.len()
                && train.is_disjoint(&hold),
            "duplicate or overlapping outer rows"
        );
        ensure!(
            self.inner_validation_rows.len() == 5,
            "selection needs five training-side folds"
        );
        ensure!(
            self.fitting_groups.len() == self.fitting_rows.len()
                && self.hold_groups.len() == self.hold_rows.len()
                && self
                    .fitting_groups
                    .iter()
                    .chain(&self.hold_groups)
                    .all(|g| crate::experiment::sha256(g)),
            "missing frozen group identities"
        );
        let train_groups: BTreeSet<_> = self.fitting_groups.iter().collect();
        let hold_groups: BTreeSet<_> = self.hold_groups.iter().collect();
        ensure!(
            train_groups.is_disjoint(&hold_groups),
            "group crossed frozen outer fold"
        );
        let row_groups: std::collections::BTreeMap<_, _> =
            self.fitting_rows.iter().zip(&self.fitting_groups).collect();
        let mut group_folds = std::collections::BTreeMap::new();
        let mut covered = BTreeSet::new();
        for (fold_index, fold) in self.inner_validation_rows.iter().enumerate() {
            ensure!(
                !fold.is_empty() && fold.len() < train.len(),
                "empty inner fitting or validation partition"
            );
            for row in fold {
                ensure!(
                    train.contains(row) && covered.insert(row.clone()),
                    "inner fold repeats rows or includes outer heldout rows"
                );
                let group = row_groups
                    .get(row)
                    .ok_or_else(|| anyhow::anyhow!("inner row group absent"))?;
                if let Some(previous) = group_folds.insert(*group, fold_index) {
                    ensure!(
                        previous == fold_index,
                        "group crossed inner training-side folds"
                    );
                }
            }
        }
        ensure!(
            covered == train,
            "inner OOF folds do not cover exact outer training rows"
        );
        Ok(())
    }
    pub fn identity(&self) -> Result<String> {
        self.validate()?;
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }
    pub fn outer(&self) -> &EvaluationRequest {
        &self.outer
    }
    pub fn fitting_rows(&self) -> &[String] {
        &self.fitting_rows
    }
    pub fn inner_rows(&self, fold: usize) -> Result<(Vec<String>, &[String])> {
        self.validate()?;
        let validation = self
            .inner_validation_rows
            .get(fold)
            .ok_or_else(|| anyhow::anyhow!("invalid inner fold"))?;
        let selected: BTreeSet<_> = validation.iter().collect();
        Ok((
            self.fitting_rows
                .iter()
                .filter(|row| !selected.contains(row))
                .cloned()
                .collect(),
            validation,
        ))
    }
    pub fn selection_requests(
        &self,
        canonical_recipe_sha256: &str,
        candidate_sha256: &str,
    ) -> Result<Vec<EvaluationRequest>> {
        self.validate()?;
        (0..5)
            .map(|fold| {
                let (fit, validation) = self.inner_rows(fold)?;
                let mut request = self.outer.clone();
                request.recipe_sha256 = canonical_recipe_sha256.into();
                request.seed = candidate_seed(self.outer.seed, candidate_sha256)?;
                request.stage = EvaluationStage::TrainOof;
                request.fold = fold as u8;
                request.split_sha256 = self.identity()?;
                request.fitting_rows_sha256 = rows_sha(&fit)?;
                request.scoring_rows_sha256 = rows_sha(validation)?;
                request.fitting_rows = fit.len();
                request.scoring_rows = validation.len();
                request.validate()?;
                Ok(request)
            })
            .collect()
    }
    #[cfg(any(feature = "neural-search", test))]
    pub(crate) fn refit_request(
        &self,
        canonical_recipe_sha256: &str,
        candidate_sha256: &str,
    ) -> Result<EvaluationRequest> {
        self.validate()?;
        let mut request = self.outer.clone();
        request.recipe_sha256 = canonical_recipe_sha256.into();
        // Outer comparison uses the frozen experiment seed for every method.
        // Candidate-specific seeds isolate train-side requests only.
        ensure!(
            crate::experiment::sha256(candidate_sha256),
            "invalid candidate identity"
        );
        request.seed = self.outer.seed;
        request.stage = EvaluationStage::Confirm;
        request.validate()?;
        Ok(request)
    }
}
fn rows_sha(rows: &[String]) -> Result<String> {
    Ok(sha256_hex(&serde_json::to_vec(rows)?))
}
pub fn candidate_seed(root: u64, candidate_sha256: &str) -> Result<u64> {
    ensure!(
        crate::experiment::sha256(candidate_sha256),
        "invalid candidate seed binding"
    );
    let digest = sha256_hex(&serde_json::to_vec(&(
        "asm.candidate-request-seed/v1",
        root,
        candidate_sha256,
    ))?);
    Ok(u64::from_str_radix(&digest[..16], 16)?)
}

/// Controller-owned scoring proof, generated only after all five output bundles
/// verify. Actual starts and fitted identities remain separate measurement fields.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OofEvidence {
    pub selection_sha256: String,
    pub scored_rows_sha256: String,
    pub request_sha256: Vec<String>,
    pub physical_fit_starts: Vec<u64>,
    pub fitted_artifact_sha256: Vec<String>,
}
impl OofEvidence {
    pub fn for_candidate(
        plan: &SelectionPlan,
        recipe: &str,
        candidate: &str,
        physical_fit_starts: Vec<u64>,
        fitted_artifact_sha256: Vec<String>,
    ) -> Result<Self> {
        Ok(Self {
            physical_fit_starts,
            fitted_artifact_sha256,
            selection_sha256: plan.identity()?,
            scored_rows_sha256: plan.outer.fitting_rows_sha256.clone(),
            request_sha256: plan
                .selection_requests(recipe, candidate)?
                .iter()
                .map(EvaluationRequest::identity)
                .collect::<Result<_>>()?,
        })
    }
    pub fn validate(&self, plan: &SelectionPlan, recipe: &str, candidate: &str) -> Result<()> {
        ensure!(
            serde_json::to_value(self)?
                == serde_json::to_value(Self::for_candidate(
                    plan,
                    recipe,
                    candidate,
                    self.physical_fit_starts.clone(),
                    self.fitted_artifact_sha256.clone()
                )?)?,
            "selection score is not bound to complete training-side OOF requests"
        );
        ensure!(
            self.physical_fit_starts.len() == 5
                && self.physical_fit_starts.iter().all(|n| *n > 0)
                && self.fitted_artifact_sha256.len() == 5
                && self
                    .fitted_artifact_sha256
                    .iter()
                    .all(|s| crate::experiment::sha256(s)),
            "OOF selection requires five measured fits and fitted artifacts"
        );
        Ok(())
    }
    pub fn total_fit_starts(&self) -> Option<u64> {
        self.physical_fit_starts
            .iter()
            .try_fold(0u64, |sum, n| sum.checked_add(*n))
    }
    pub fn fitted_bundle_sha256(&self) -> Result<String> {
        Ok(sha256_hex(&serde_json::to_vec(
            &self.fitted_artifact_sha256,
        )?))
    }
}

#[cfg(test)]
fn row_id(i: usize) -> String {
    sha256_hex(format!("row{i}").as_bytes())
}
#[cfg(test)]
pub(crate) fn fixture() -> SelectionPlan {
    use crate::experiment::{Metric, REQUEST_SCHEMA};
    let train: Vec<_> = (0..10).map(row_id).collect();
    let hold = vec![row_id(20), row_id(21)];
    SelectionPlan::new(
        EvaluationRequest {
            schema: REQUEST_SCHEMA.into(),
            experiment_sha256: sha256_hex(b"synthetic experiment"),
            dataset_sha256: sha256_hex(b"synthetic dataset"),
            split_sha256: sha256_hex(b"synthetic outer split"),
            fitting_rows_sha256: rows_sha(&train).unwrap(),
            scoring_rows_sha256: rows_sha(&hold).unwrap(),
            fitting_rows: 10,
            scoring_rows: 2,
            recipe_sha256: sha256_hex(b"placeholder"),
            evaluator_sha256: sha256_hex(b"synthetic evaluator"),
            checkpoint_sha256: None,
            task: jope_datasets::catalog::Task::Regression,
            metric: Metric::Rmse,
            classes: vec![],
            fold: 0,
            stage: EvaluationStage::Screen,
            seed: 71,
            numerical_policy: "synthetic fixture".into(),
        },
        train,
        hold,
        (0..5)
            .map(|i| vec![row_id(2 * i), row_id(2 * i + 1)])
            .collect(),
        (0..10)
            .map(|i| sha256_hex(format!("group{i}").as_bytes()))
            .collect(),
        (20..22)
            .map(|i| sha256_hex(format!("group{i}").as_bytes()))
            .collect(),
    )
    .unwrap()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_inner_closure_and_arm_seeds_are_bound() -> Result<()> {
        let plan = fixture();
        let recipe = sha256_hex(b"canonical recipe");
        let a = plan.selection_requests(&recipe, &sha256_hex(b"arm a"))?;
        let b = plan.selection_requests(&recipe, &sha256_hex(b"arm b"))?;
        assert_ne!(a[0].identity()?, b[0].identity()?);
        assert_eq!(a[0].recipe_sha256, b[0].recipe_sha256);
        for (i, request) in a.iter().enumerate() {
            let (train, val) = plan.inner_rows(i)?;
            assert!(train
                .iter()
                .all(|r| !val.contains(r) && plan.fitting_rows.contains(r)));
            assert!(val.iter().all(|r| plan.fitting_rows.contains(r)));
            assert_eq!(request.stage, EvaluationStage::TrainOof);
        }
        Ok(())
    }
    #[test]
    fn heldout_or_repeated_or_missing_rows_are_rejected() {
        let mut plan = fixture();
        plan.inner_validation_rows[0][0] = row_id(20);
        assert!(plan.validate().is_err());
        let mut plan = fixture();
        plan.inner_validation_rows[0][0] = row_id(2);
        assert!(plan.validate().is_err());
        let mut plan = fixture();
        plan.inner_validation_rows[0].pop();
        assert!(plan.validate().is_err());
        let mut plan = fixture();
        plan.fitting_groups[0] = plan.fitting_groups[2].clone();
        assert!(plan.validate().is_err());
        let mut plan = fixture();
        plan.fitting_rows.swap(0, 1);
        assert!(plan.validate().is_err());
    }
    #[test]
    fn frozen_refit_uses_common_outer_seed_and_canonical_recipe() -> Result<()> {
        let plan = fixture();
        let recipe = sha256_hex(b"canonical DSL");
        let a = plan.refit_request(&recipe, &sha256_hex(b"candidate a"))?;
        let b = plan.refit_request(&recipe, &sha256_hex(b"candidate b"))?;
        assert_eq!(a.seed, plan.outer().seed);
        assert_eq!(a.identity()?, b.identity()?);
        assert_eq!(a.recipe_sha256, recipe);
        Ok(())
    }
    #[test]
    fn grouped_assignment_keeps_whole_groups_and_input_order() {
        let rows = [
            ("a1", "a"),
            ("b1", "b"),
            ("a2", "a"),
            ("c1", "c"),
            ("d1", "d"),
            ("e1", "e"),
            ("f1", "f"),
        ]
        .into_iter()
        .map(|(id, group)| (id.to_string(), group.to_string()))
        .collect::<Vec<_>>();
        let folds = grouped_inner_folds(&rows).unwrap();
        assert_eq!(folds[0], vec!["a1", "a2"]);
        assert_eq!(folds.iter().map(Vec::len).sum::<usize>(), rows.len());
        assert_eq!(folds, grouped_inner_folds(&rows).unwrap());
        assert!(grouped_inner_folds(&rows[..3]).is_err());
    }
}
