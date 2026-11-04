use super::{
    planning::{asset, build, digest, hash, hex, read_bytes, verify_asset, PROTOCOL},
    types::*,
};
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

pub(crate) fn rmse(actual: &[f64], predicted: &[f64]) -> Result<f64> {
    ensure!(
        !actual.is_empty() && actual.len() == predicted.len(),
        "prediction length mismatch"
    );
    let mse = actual
        .iter()
        .zip(predicted)
        .map(|(a, p)| (a - p).powi(2))
        .sum::<f64>()
        / actual.len() as f64;
    ensure!(mse.is_finite(), "nonfinite score");
    Ok(mse.sqrt())
}
#[derive(Clone, Copy, Debug, Serialize)]
pub(super) struct Gains {
    pub(super) audit: f64,
    outer: f64,
    pub(super) contribution: f64,
    pub(super) audit_skill: f64,
    outer_skill: f64,
    parent_audit_skill: f64,
    parent_outer_skill: f64,
}
pub(crate) fn gains(audit: &[(f64, f64, f64)], outer: (f64, f64, f64)) -> Result<Gains> {
    ensure!(audit.len() == 5, "five audit folds required");
    let gain = |(p, c, b): (f64, f64, f64)| -> Result<(f64, f64)> {
        ensure!(
            [p, c, b].iter().all(|v| v.is_finite() && *v >= 0.0) && b > 0.0,
            "invalid baseline/loss"
        );
        let result = ((p - c) / b, 1.0 - c / b);
        ensure!(
            result.0.is_finite() && result.1.is_finite(),
            "nonfinite normalized score"
        );
        Ok(result)
    };
    let folds = audit.iter().map(|x| gain(*x)).collect::<Result<Vec<_>>>()?;
    let a = folds.iter().map(|x| x.0 / 5.0).sum::<f64>();
    let skill = folds.iter().map(|x| x.1 / 5.0).sum::<f64>();
    let (o, outer_skill) = gain(outer)?;
    Ok(Gains {
        audit: a,
        outer: o,
        contribution: a.min(o),
        audit_skill: skill,
        outer_skill,
        parent_audit_skill: audit.iter().map(|(p, _, b)| (1.0 - p / b) / 5.0).sum(),
        parent_outer_skill: 1.0 - outer.0 / outer.2,
    })
}

pub fn report(
    plan_path: &Path,
    freeze_path: &Path,
    release_path: &Path,
    release_sha256: &str,
    output: &Path,
) -> Result<Value> {
    match report_validated(plan_path, freeze_path, release_path, release_sha256, output) {
        Ok(result) => Ok(result),
        Err(error) => {
            // Preserve coverage even when authentication or numerical validation fails.
            // Counts here are explicitly untrusted until the whole plan verifies.
            let claimed: Option<Plan> = read_bytes(plan_path)
                .ok()
                .and_then(|bytes| serde_json::from_slice(&bytes).ok());
            let failure = json!({"schema":"jope.transfer-report/v1","ok":false,
                "kpi_percentage_points":null,"status":"invalid evidence",
                "claimed_planned_units_unverified":claimed.as_ref().map(|p|p.planned_units),
                "failures":[format!("{error:#}")],"scores_released":false,"fits":0});
            if !output.exists() {
                super::super::worker::write_private(output, &failure)?;
            }
            Err(error)
        }
    }
}

fn report_validated(
    plan_path: &Path,
    freeze_path: &Path,
    release_path: &Path,
    release_sha256: &str,
    output: &Path,
) -> Result<Value> {
    let plan_bytes = read_bytes(plan_path)?;
    let plan: Plan = serde_json::from_slice(&plan_bytes)?;
    let plan_hash = hash(&plan_bytes);
    let rebuilt = build(plan.specification.clone())?;
    ensure!(
        digest(&plan)? == digest(&rebuilt)?,
        "altered plan or assets; regenerate a new round"
    );
    let release: Release = serde_json::from_slice(&asset(&Asset {
        path: release_path.into(),
        sha256: release_sha256.into(),
    })?)?;
    let freeze_bytes = read_bytes(freeze_path)?;
    let freeze: Freeze = serde_json::from_slice(&freeze_bytes)?;
    ensure!(
        release.schema == "jope.transfer-release/v1" && freeze.schema == "jope.transfer-freeze/v1",
        "invalid freeze/release schema"
    );
    ensure!(
        release.plan_sha256 == plan_hash
            && freeze.plan_sha256 == plan_hash
            && release.freeze_sha256 == hash(&freeze_bytes),
        "freeze/release identity mismatch"
    );
    ensure!(
        release.executor_source_sha == plan.specification.source_sha
            && release.independently_verified
            && release.all_predictions_committed_before_scoring
            && release.scoring_labels_excluded_from_selection_and_fitting
            && release.no_cross_fold_result_feedback,
        "separately admitted, independently verified executor evidence required"
    );
    verify_asset(&release.isolation_evidence)?;
    ensure!(
        freeze.frozen_sequence > 0
            && release.first_scored_result_sequence > freeze.frozen_sequence
            && release.released_sequence >= release.first_scored_result_sequence,
        "premature result release"
    );
    let mut failures = plan.failures.clone();
    let mut predictions = BTreeMap::new();
    let search_map: BTreeMap<_, _> = plan
        .searches
        .iter()
        .map(|s| Ok((digest(s)?, s)))
        .collect::<Result<_>>()?;
    for a in &freeze.predictions {
        let p: Prediction = serde_json::from_slice(&asset(a)?)?;
        ensure!(
            p.schema == PROTOCOL && p.plan_sha256 == plan_hash,
            "prediction identity mismatch"
        );
        let search = search_map
            .get(&p.search_sha256)
            .context("unplanned prediction search")?;
        let arm = plan
            .specification
            .arms
            .iter()
            .find(|a| a.id == p.arm)
            .context("unplanned arm")?;
        ensure!(
            p.checkpoint_sha256 == arm.checkpoint.sha256
                && p.procedure_sha256 == plan.specification.procedure.sha256
                && hex(&p.recipe_sha256, 64),
            "prediction recipe/checkpoint/procedure mismatch"
        );
        ensure!(
            p.completed_sequence > 0 && p.completed_sequence < freeze.frozen_sequence,
            "prediction completed after freeze"
        );
        ensure!(
            p.fitting_rows == search.train && p.selection_label_rows == search.train,
            "fitting/selection label closure differs from permitted rows"
        );
        ensure!(
            p.recipe.sha256 == p.recipe_sha256,
            "recipe asset identity mismatch"
        );
        verify_asset(&p.recipe)?;
        let selection: SelectionReceipt = serde_json::from_slice(&asset(&p.selection_receipt)?)?;
        ensure!(
            selection.schema == "jope.transfer-nested-selection/v1"
                && selection.complete_search_repeated
                && selection.search_sha256 == p.search_sha256
                && selection.procedure_sha256 == p.procedure_sha256
                && selection.recipe_sha256 == p.recipe_sha256
                && selection.inner_score == search.inner_score
                && selection.label_dependencies == search.train,
            "invalid independent full-search selection receipt"
        );
        ensure!(
            p.predictions.len() == search.score.len()
                && p.predictions.iter().all(|x| x.is_finite()),
            "invalid prediction values"
        );
        ensure!(
            predictions
                .insert((p.search_sha256.clone(), p.arm.clone()), p)
                .is_none(),
            "duplicate prediction"
        );
    }
    let mut terminal_slots: BTreeSet<_> = predictions.keys().cloned().collect();
    let mut unavailable = Vec::new();
    for failed in &freeze.failures {
        ensure!(
            search_map.contains_key(&failed.search_sha256)
                && plan.specification.arms.iter().any(|a| a.id == failed.arm),
            "unplanned failed slot"
        );
        ensure!(
            ["failed", "unavailable"].contains(&failed.status.as_str())
                && !failed.reason.trim().is_empty()
                && failed.completed_sequence > 0
                && failed.completed_sequence < freeze.frozen_sequence,
            "invalid terminal failure receipt"
        );
        ensure!(
            terminal_slots.insert((failed.search_sha256.clone(), failed.arm.clone())),
            "duplicate terminal slot"
        );
        unavailable.push(json!({"search_sha256":failed.search_sha256,"arm":failed.arm,"status":failed.status,"reason":failed.reason}));
    }
    for identity in search_map.keys() {
        for arm in &plan.specification.arms {
            if !terminal_slots.contains(&(identity.clone(), arm.id.clone())) {
                unavailable.push(json!({"search_sha256":identity,"arm":arm.id,"status":"missing","reason":"no committed terminal outcome"}));
            }
        }
    }
    // A partial freeze cannot release any scores, including successful subsets.
    let expected = plan.searches.len() * plan.specification.arms.len();
    if !unavailable.is_empty() {
        failures.push(format!(
            "missing comparisons: {} of {expected} planned predictions committed",
            predictions.len()
        ));
    }
    if !failures.is_empty() {
        let result = json!({"schema":"jope.transfer-report/v1","ok":false,"kpi_percentage_points":null,"status":"not yet measured","planned_units":plan.planned_units,"planned_predictions":expected,"committed_predictions":predictions.len(),"terminal_outcomes":terminal_slots.len(),"unavailable":unavailable,"failures":failures,"scores_released":false,"fits":0});
        super::super::worker::write_private(output, &result)?;
        return Ok(result);
    }
    let mut losses = BTreeMap::new();
    for d in &plan.specification.datasets {
        let labels: BTreeMap<String, f64> = serde_json::from_slice(&asset(&d.labels)?)?;
        ensure!(
            labels.len() == d.rows.len()
                && d.rows
                    .iter()
                    .all(|r| labels.get(&r.id).is_some_and(|v| v.is_finite())),
            "label row identity mismatch"
        );
        for s in plan.searches.iter().filter(|s| s.dataset == d.id) {
            let identity = digest(s)?;
            let mean = s
                .train
                .iter()
                .map(|r| labels[r] / s.train.len() as f64)
                .sum::<f64>();
            let actual: Vec<_> = s.score.iter().map(|r| labels[r]).collect();
            let baseline = rmse(&actual, &vec![mean; actual.len()])?;
            for arm in &plan.specification.arms {
                let p = &predictions[&(identity.clone(), arm.id.clone())];
                losses.insert(
                    (identity.clone(), arm.id.clone()),
                    (rmse(&actual, &p.predictions)?, baseline),
                );
            }
        }
    }
    let mut units = Vec::new();
    let mut families: BTreeMap<String, Vec<Gains>> = BTreeMap::new();
    for d in &plan.specification.datasets {
        let mut dataset_gains = Vec::new();
        for seed in &plan.specification.seeds {
            for outer in 0..5 {
                let mut audit = Vec::new();
                let mut outer_loss = None;
                for s in plan
                    .searches
                    .iter()
                    .filter(|s| s.dataset == d.id && s.seed == *seed && s.outer == outer)
                {
                    let id = digest(s)?;
                    let (p, b) = losses[&(id.clone(), "parent".into())];
                    let (c, _) = losses[&(id, "challenger".into())];
                    if s.audit.is_some() {
                        audit.push((p, c, b));
                    } else {
                        outer_loss = Some((p, c, b));
                    }
                }
                match gains(&audit, outer_loss.context("missing outer search")?) {
                    Ok(g) => {
                        dataset_gains.push(g);
                        units.push(json!({"dataset":d.id,"family":d.family,"seed":seed,"outer":outer,"gains":g}));
                    }
                    Err(e) => {
                        failures.push(format!("{} seed {seed} outer {outer}: {e:#}", d.id));
                        units.push(json!({"dataset":d.id,"family":d.family,"seed":seed,"outer":outer,"gains":null,"reason":e.to_string()}));
                    }
                }
            }
        }
        if dataset_gains.len() == plan.specification.seeds.len() * 5 {
            families
                .entry(d.family.clone())
                .or_default()
                .push(average(&dataset_gains));
        }
    }
    let family_means: BTreeMap<_, _> = families
        .iter()
        .map(|(f, ds)| (f.clone(), average(ds)))
        .collect();
    let complete = failures.is_empty() && !family_means.is_empty();
    let global = complete.then(|| average(&family_means.values().copied().collect::<Vec<_>>()));
    let scatter: Vec<_> = units.iter().filter_map(|u| u.get("gains").filter(|v| !v.is_null()).map(|g| json!({"dataset":u["dataset"],"family":u["family"],"seed":u["seed"],"outer":u["outer"],"x_audit":g["audit"],"y_outer":g["outer"]}))).collect();
    let result = json!({"schema":"jope.transfer-report/v1","ok":complete,"plan_sha256":plan_hash,"release_sha256":release_sha256,"status":if complete {"retained-evidence descriptive score"} else {"incomplete"},"kpi_percentage_points":global.map(|g|100.0*g.contribution),"global":global,"absolute_skill_statement":if global.is_some_and(|g|g.audit_skill>0.0 && g.outer_skill>0.0){"positive mean absolute skill on audit and outer"}else{"no absolute skill demonstrated"},"worst_family_outer_gain":family_means.values().map(|g|g.outer).reduce(f64::min),"worst_family_contribution":family_means.values().map(|g|g.contribution).reduce(f64::min),"sign_agreement":if scatter.is_empty(){None}else{Some(scatter.iter().filter(|g|sign(g["x_audit"].as_f64().unwrap())==sign(g["y_outer"].as_f64().unwrap())).count() as f64/scatter.len() as f64)},"planned_units":plan.planned_units,"valid_units":scatter.len(),"planned_families":plan.specification.datasets.iter().map(|d|&d.family).collect::<BTreeSet<_>>().len(),"families":family_means,"units":units,"paired_audit_outer_scatter":scatter,"scatter_svg":scatter_svg(&scatter),"sign_agreement_convention":"negative, zero, positive; zero agrees only with zero","losses":losses.iter().map(|((search,arm),(rmse,baseline))|json!({"search_sha256":search,"arm":arm,"rmse":rmse,"training_mean_rmse":baseline})).collect::<Vec<_>>(),"failures":failures,"fits":0,"controls":"reported separately as losses; never pooled with parent contrast","interpretation":"conservative descriptive score, not a confidence bound or causal proof; folds and seeds are not independent samples","subsequent_use":"released outcomes are exposed development evidence for subsequent rounds","provenance_limit":"hashes authenticate bytes; executor isolation requires independent admission and verification outside this reporter","teacher_admission":false});
    super::super::worker::write_private(output, &result)?;
    Ok(result)
}
pub(crate) fn average(gs: &[Gains]) -> Gains {
    let n = gs.len() as f64;
    Gains {
        audit: gs.iter().map(|g| g.audit / n).sum(),
        outer: gs.iter().map(|g| g.outer / n).sum(),
        contribution: gs.iter().map(|g| g.contribution / n).sum(),
        audit_skill: gs.iter().map(|g| g.audit_skill / n).sum(),
        outer_skill: gs.iter().map(|g| g.outer_skill / n).sum(),
        parent_audit_skill: gs.iter().map(|g| g.parent_audit_skill / n).sum(),
        parent_outer_skill: gs.iter().map(|g| g.parent_outer_skill / n).sum(),
    }
}
pub(crate) fn sign(x: f64) -> i8 {
    if x > 0.0 {
        1
    } else if x < 0.0 {
        -1
    } else {
        0
    }
}
fn scatter_svg(points: &[Value]) -> String {
    let extent = points
        .iter()
        .flat_map(|p| {
            [
                p["x_audit"].as_f64().unwrap().abs(),
                p["y_outer"].as_f64().unwrap().abs(),
            ]
        })
        .fold(0.01, f64::max)
        * 1.1;
    let mut svg = String::from(
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 560 540"><rect width="560" height="540" fill="white"/><g stroke="#999"><path d="M60 260H500M280 40V480"/><path d="M60 480L500 40" stroke-dasharray="5 5"/></g><g font-family="sans-serif" font-size="14" fill="#222"><text x="130" y="520">Audit gain (baseline RMSE units)</text><text x="20" y="25">Outer gain (baseline RMSE units)</text><text x="288" y="278">0</text></g>"##,
    );
    for p in points {
        let x = 280.0 + 220.0 * p["x_audit"].as_f64().unwrap() / extent;
        let y = 260.0 - 220.0 * p["y_outer"].as_f64().unwrap() / extent;
        svg.push_str(&format!(
            r##"<circle cx="{x:.3}" cy="{y:.3}" r="4" fill="#2463a0" fill-opacity="0.55"/>"##
        ));
    }
    svg.push_str(&format!(r#"<text x="360" y="25" font-family="sans-serif" font-size="12">Axis range ±{extent:.4}</text></svg>"#));
    svg
}
