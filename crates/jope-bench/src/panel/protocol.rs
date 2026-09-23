//! Screen every declared method on fold zero, then freeze recipes before confirmation.
//! Scores must already have been computed by the controller from checked predictions.

use anyhow::{ensure, Context, Result};
use jope_pipeline::experiment::{EvaluationRequest, EvaluationStage, EvaluationStatus};
use jope_pipeline::experiment_v2::EvaluationResult;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Observation {
    pub method: String,
    pub request: EvaluationRequest,
    pub result: EvaluationResult,
}
impl Observation {
    pub fn validate(&self) -> Result<()> {
        ensure!(!self.method.is_empty(), "missing method identity");
        self.request.validate()?;
        self.result.validate(&self.request)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenRecipe {
    pub method: String,
    pub recipe_sha256: String,
    pub checkpoint_sha256: Option<String>,
    pub screen_request_sha256: String,
    /// An additional ASM recipe is confirmed even when it misses the top five.
    pub diagnostic: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenSelection {
    pub protocol: String,
    pub experiment_sha256: String,
    pub dataset_sha256: String,
    pub split_sha256: String,
    pub screen_evidence_sha256: String,
    pub recipes: Vec<FrozenRecipe>,
}

/// `expected` is the full applicable registry, including unavailable methods. A missing
/// or duplicate outcome is an error; it cannot quietly reduce the comparator set.
pub fn freeze_screen(
    expected: &[(String, EvaluationRequest)],
    observations: &[Observation],
    asm_method: Option<&str>,
) -> Result<FrozenSelection> {
    ensure!(!expected.is_empty(), "empty screen registry");
    let reference = &expected[0].1;
    reference.validate()?;
    ensure!(
        reference.stage == EvaluationStage::Screen && reference.fold == 0,
        "selection requires fold-zero screen requests"
    );
    let mut expected_by_method = BTreeMap::new();
    for (method, request) in expected {
        request.validate()?;
        ensure!(
            !method.is_empty() && expected_by_method.insert(method, request).is_none(),
            "empty or duplicate screen method"
        );
        ensure!(
            same_cut(reference, request),
            "screen methods use different cuts"
        );
    }
    ensure!(
        observations.len() == expected.len(),
        "incomplete screen inventory"
    );
    let mut by_method = BTreeMap::new();
    for observation in observations {
        observation.validate()?;
        let expected = expected_by_method
            .get(&observation.method)
            .context("unexpected screen method")?;
        ensure!(
            observation.request.identity()? == expected.identity()?,
            "stale screen request"
        );
        ensure!(
            by_method.insert(&observation.method, observation).is_none(),
            "duplicate screen outcome"
        );
    }
    let mut ranked = Vec::new();
    for observation in by_method.values() {
        if observation.result.status == EvaluationStatus::Completed {
            ranked.push((
                observation
                    .request
                    .metric
                    .loss(observation.result.primary.context("score")?)?,
                *observation,
            ));
        }
    }
    ranked.sort_by(|(a, oa), (b, ob)| a.total_cmp(b).then(oa.method.cmp(&ob.method)));
    let mut canonical_recipes = BTreeSet::new();
    let mut recipes: Vec<_> = ranked
        .iter()
        .filter(|(_, o)| canonical_recipes.insert(o.request.recipe_sha256.clone()))
        .take(5)
        .map(|(_, observation)| frozen_recipe(observation, false))
        .collect::<Result<_>>()?;
    if let Some(method) = asm_method {
        let observation = by_method
            .get(&method.to_owned())
            .context("ASM method is absent from the screen registry")?;
        if !recipes.iter().any(|recipe| recipe.method == method) {
            recipes.push(frozen_recipe(observation, true)?);
        }
    }
    // BTreeMap order makes this independent of completion order and fair scheduling.
    let ordered: Vec<_> = by_method.values().collect();
    Ok(FrozenSelection {
        protocol: super::PROTOCOL.into(),
        experiment_sha256: reference.experiment_sha256.clone(),
        dataset_sha256: reference.dataset_sha256.clone(),
        split_sha256: reference.split_sha256.clone(),
        screen_evidence_sha256: jope_pipeline::sha256_hex(&serde_json::to_vec(&ordered)?),
        recipes,
    })
}

fn frozen_recipe(observation: &Observation, diagnostic: bool) -> Result<FrozenRecipe> {
    Ok(FrozenRecipe {
        method: observation.method.clone(),
        recipe_sha256: observation.request.recipe_sha256.clone(),
        checkpoint_sha256: observation.request.checkpoint_sha256.clone(),
        screen_request_sha256: observation.request.identity()?,
        diagnostic,
    })
}

fn same_cut(a: &EvaluationRequest, b: &EvaluationRequest) -> bool {
    a.experiment_sha256 == b.experiment_sha256
        && a.dataset_sha256 == b.dataset_sha256
        && a.split_sha256 == b.split_sha256
        && a.fitting_rows_sha256 == b.fitting_rows_sha256
        && a.scoring_rows_sha256 == b.scoring_rows_sha256
        && a.fitting_rows == b.fitting_rows
        && a.scoring_rows == b.scoring_rows
        && a.evaluator_sha256 == b.evaluator_sha256
        && a.task == b.task
        && a.metric == b.metric
        && a.classes == b.classes
        && a.fold == b.fold
        && a.stage == b.stage
        && a.seed == b.seed
        && a.numerical_policy == b.numerical_policy
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConfirmedRecipe {
    pub method: String,
    pub diagnostic: bool,
    pub completed_folds: usize,
    /// Missing, failed and unavailable folds never become a score of zero.
    pub mean_loss: Option<f64>,
    pub fold_losses: [Option<f64>; 5],
}

/// Recompute selection from the full screen evidence before accepting a persisted freeze.
/// Confirmation requests are supplied by the controller from the actual five prepared cuts.
pub fn confirm(
    frozen: &FrozenSelection,
    expected_screen: &[(String, EvaluationRequest)],
    screen: &[Observation],
    asm_method: Option<&str>,
    expected_confirm: &[(String, EvaluationRequest)],
    observations: &[Observation],
) -> Result<Vec<ConfirmedRecipe>> {
    ensure!(
        *frozen == freeze_screen(expected_screen, screen, asm_method)?,
        "stale or altered frozen selection"
    );
    ensure!(
        expected_confirm.len() == frozen.recipes.len() * 5
            && observations.len() == expected_confirm.len(),
        "confirmation inventory must cover five folds of every frozen recipe"
    );
    let recipes: BTreeMap<_, _> = frozen.recipes.iter().map(|r| (&r.method, r)).collect();
    let mut expected_by_key = BTreeMap::new();
    let mut cuts = BTreeMap::new();
    for (method, request) in expected_confirm {
        request.validate()?;
        let recipe = recipes
            .get(method)
            .context("unselected confirmation recipe")?;
        ensure!(
            request.stage == EvaluationStage::Confirm
                && request.experiment_sha256 == frozen.experiment_sha256
                && request.dataset_sha256 == frozen.dataset_sha256
                && request.split_sha256 == frozen.split_sha256
                && request.recipe_sha256 == recipe.recipe_sha256
                && request.checkpoint_sha256 == recipe.checkpoint_sha256,
            "confirmation changes a frozen recipe or experiment"
        );
        let screen_request = &expected_screen[0].1;
        ensure!(
            request.task == screen_request.task
                && request.metric == screen_request.metric
                && request.classes == screen_request.classes
                && request.evaluator_sha256 == screen_request.evaluator_sha256
                && request.numerical_policy == screen_request.numerical_policy
                && request.seed == screen_request.seed,
            "confirmation changes the screen evaluator"
        );
        if request.fold == 0 {
            let mut screen_cut = request.clone();
            screen_cut.stage = EvaluationStage::Screen;
            ensure!(
                same_cut(screen_request, &screen_cut),
                "confirmation changes fold zero"
            );
        }
        ensure!(
            expected_by_key
                .insert((method, request.fold), request)
                .is_none(),
            "duplicate confirmation request"
        );
        if let Some(reference) = cuts.insert(request.fold, request) {
            ensure!(
                same_cut(reference, request),
                "confirmation methods use different cuts"
            );
        }
    }
    let mut seen = BTreeSet::new();
    let mut losses: BTreeMap<&String, [Option<f64>; 5]> = frozen
        .recipes
        .iter()
        .map(|r| (&r.method, [None; 5]))
        .collect();
    for observation in observations {
        observation.validate()?;
        let key = (&observation.method, observation.request.fold);
        let request = expected_by_key
            .get(&key)
            .context("unexpected confirmation outcome")?;
        ensure!(
            seen.insert(key) && request.identity()? == observation.request.identity()?,
            "duplicate or stale confirmation outcome"
        );
        if observation.result.status == EvaluationStatus::Completed {
            losses.get_mut(&observation.method).context("method")?
                [observation.request.fold as usize] = Some(
                request
                    .metric
                    .loss(observation.result.primary.context("score")?)?,
            );
        }
    }
    Ok(frozen
        .recipes
        .iter()
        .map(|recipe| {
            let fold_losses = losses[&recipe.method];
            let completed_folds = fold_losses.iter().flatten().count();
            ConfirmedRecipe {
                method: recipe.method.clone(),
                diagnostic: recipe.diagnostic,
                completed_folds,
                mean_loss: (completed_folds == 5)
                    .then(|| fold_losses.iter().flatten().map(|v| v / 5.).sum()),
                fold_losses,
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use jope_datasets::catalog::Task;
    use jope_pipeline::experiment::{Metric, REQUEST_SCHEMA};
    use jope_pipeline::experiment_v2::{Consumption, RESULT_SCHEMA};

    fn observation(method: usize, score: Option<f64>) -> Observation {
        let request = EvaluationRequest {
            schema: REQUEST_SCHEMA.into(),
            experiment_sha256: "a".repeat(64),
            dataset_sha256: "b".repeat(64),
            split_sha256: "c".repeat(64),
            fitting_rows_sha256: "d".repeat(64),
            scoring_rows_sha256: "e".repeat(64),
            fitting_rows: 80,
            scoring_rows: 20,
            recipe_sha256: jope_pipeline::sha256_hex(method.to_string().as_bytes()),
            evaluator_sha256: "f".repeat(64),
            checkpoint_sha256: None,
            task: Task::Regression,
            metric: Metric::Rmse,
            classes: vec![],
            fold: 0,
            stage: EvaluationStage::Screen,
            seed: 42,
            numerical_policy: "test-fixture".into(),
        };
        let result = EvaluationResult {
            schema: RESULT_SCHEMA.into(),
            request_sha256: request.identity().unwrap(),
            status: if score.is_some() {
                EvaluationStatus::Completed
            } else {
                EvaluationStatus::Unavailable
            },
            reason: score.is_none().then(|| "fixture missing dependency".into()),
            primary: score,
            secondary: BTreeMap::new(),
            scored_rows: if score.is_some() { 20 } else { 0 },
            predictions_sha256: score.map(|_| "0".repeat(64)),
            fitted_state_sha256: score.map(|_| "1".repeat(64)),
            worker_receipt_sha256: "2".repeat(64),
            reused_from_request_sha256: None,
            consumption: Consumption {
                wall_seconds: Some(0.),
                cpu_seconds: Some(0.),
                gpu_seconds: None,
                peak_rss_bytes: None,
                cgroup_peak_memory_bytes: None,
                sampled_peak_rss_bytes: None,
                accounting_reason: "synthetic protocol fixture".into(),
            },
        };
        Observation {
            method: format!("method-{method}"),
            request,
            result,
        }
    }
    fn expected(observations: &[Observation]) -> Vec<(String, EvaluationRequest)> {
        observations
            .iter()
            .map(|o| (o.method.clone(), o.request.clone()))
            .collect()
    }

    #[test]
    fn smaller_rmse_wins_ties_are_stable_and_asm_is_additional() {
        let mut rows: Vec<_> = (0..7).map(|i| observation(i, Some(i as f64))).collect();
        rows.push(observation(7, None));
        let declared = expected(&rows);
        let frozen = freeze_screen(&declared, &rows, Some("method-6")).unwrap();
        assert_eq!(
            frozen
                .recipes
                .iter()
                .map(|r| r.method.as_str())
                .collect::<Vec<_>>(),
            ["method-0", "method-1", "method-2", "method-3", "method-4", "method-6"]
        );
        assert!(frozen.recipes[5].diagnostic);
        rows.reverse();
        assert_eq!(
            frozen,
            freeze_screen(&declared, &rows, Some("method-6")).unwrap()
        );
        rows.pop();
        assert!(freeze_screen(&declared, &rows, None).is_err());
    }

    #[test]
    fn auroc_is_converted_to_loss_and_stale_results_are_rejected() {
        let mut rows: Vec<_> = (0..7)
            .map(|i| observation(i, Some(i as f64 / 10.)))
            .collect();
        for row in &mut rows {
            row.request.task = Task::Binary;
            row.request.metric = Metric::Auroc;
            row.request.classes = vec!["negative".into(), "positive".into()];
            row.result.request_sha256 = row.request.identity().unwrap();
        }
        let declared = expected(&rows);
        let frozen = freeze_screen(&declared, &rows, None).unwrap();
        assert_eq!(frozen.recipes[0].method, "method-6");
        rows[0].request.classes.reverse();
        assert!(freeze_screen(&declared, &rows, None).is_err());
        rows[0].result.request_sha256 = rows[0].request.identity().unwrap();
        assert!(freeze_screen(&declared, &rows, None).is_err());
    }

    #[test]
    fn confirmation_requires_all_folds_and_an_unchanged_freeze() {
        let screen = vec![observation(0, Some(3.))];
        let declared = expected(&screen);
        let frozen = freeze_screen(&declared, &screen, None).unwrap();
        let mut confirmation = Vec::new();
        for fold in 0..5 {
            let mut row = observation(0, Some(3.));
            row.request.stage = EvaluationStage::Confirm;
            row.request.fold = fold;
            if fold > 0 {
                row.request.fitting_rows_sha256 = jope_pipeline::sha256_hex(&[fold, 0]);
                row.request.scoring_rows_sha256 = jope_pipeline::sha256_hex(&[fold, 1]);
            }
            row.result.request_sha256 = row.request.identity().unwrap();
            confirmation.push(row);
        }
        let confirmations = expected(&confirmation);
        let report = confirm(
            &frozen,
            &declared,
            &screen,
            None,
            &confirmations,
            &confirmation,
        )
        .unwrap();
        assert_eq!(report[0].mean_loss, Some(3.));
        confirmation[3].result.status = EvaluationStatus::TimedOut;
        confirmation[3].result.reason = Some("fit deadline".into());
        confirmation[3].result.primary = None;
        confirmation[3].result.scored_rows = 0;
        let report = confirm(
            &frozen,
            &declared,
            &screen,
            None,
            &confirmations,
            &confirmation,
        )
        .unwrap();
        assert_eq!(report[0].completed_folds, 4);
        assert_eq!(report[0].mean_loss, None);
        let mut stale = frozen.clone();
        stale.recipes[0].recipe_sha256 = "9".repeat(64);
        assert!(confirm(
            &stale,
            &declared,
            &screen,
            None,
            &confirmations,
            &confirmation
        )
        .is_err());
        confirmation.pop();
        assert!(confirm(
            &frozen,
            &declared,
            &screen,
            None,
            &confirmations,
            &confirmation
        )
        .is_err());
    }
    #[test]
    fn top_five_deduplicates_recipes_and_retains_required_asm_alias() -> Result<()> {
        let mut data = (0..7)
            .map(|i| observation(i, Some(i as f64)))
            .collect::<Vec<_>>();
        data[1].request.recipe_sha256 = data[0].request.recipe_sha256.clone();
        data[1].result.request_sha256 = data[1].request.identity()?;
        let frozen = freeze_screen(&expected(&data), &data, Some("method-1"))?;
        assert_eq!(frozen.recipes.iter().filter(|r| !r.diagnostic).count(), 5);
        assert_eq!(
            frozen
                .recipes
                .iter()
                .filter(|r| !r.diagnostic)
                .map(|r| &r.recipe_sha256)
                .collect::<BTreeSet<_>>()
                .len(),
            5
        );
        assert!(frozen
            .recipes
            .iter()
            .any(|r| r.method == "method-1" && r.diagnostic));
        Ok(())
    }
}
