//! Reporting-only nested procedure audit. No fitting or search entry point is linked here.
//! Hashes authenticate retained bytes, not the truth of an executor's provenance claim.
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

const PROTOCOL: &str = "jope.transfer-procedure-audit/v1";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Asset {
    pub path: PathBuf,
    pub sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Row {
    pub id: String,
    pub group: String,
    pub outer: u8,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Dataset {
    pub id: String,
    pub family: String,
    pub features: Asset,
    pub labels: Asset,
    pub rows: Vec<Row>,
    /// "untouched", "adaptive_development", or "unknown", supported by the exposure asset.
    pub evaluation_exposure: String,
    pub checkpoint_exposure: String,
    pub encoder_exposure: String,
    pub exposure_evidence: Asset,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Arm {
    pub id: String,
    pub checkpoint: Asset,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Limits {
    pub max_searches: u64,
    pub candidates_per_search: u64,
    pub candidate_fit_seconds: u64,
    pub max_fit_seconds: u64,
    pub cpus: u32,
    pub memory_bytes: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Specification {
    pub schema: String,
    pub round: String,
    pub source_sha: String,
    pub procedure: Asset,
    pub encoder: Asset,
    pub assets: Vec<Asset>,
    /// Parent and challenger are mandatory; controls are retained but never pooled into the KPI.
    pub arms: Vec<Arm>,
    pub seeds: Vec<u64>,
    pub datasets: Vec<Dataset>,
    pub limits: Limits,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Search {
    pub dataset: String,
    pub seed: u64,
    pub outer: u8,
    pub audit: Option<u8>,
    pub train: Vec<String>,
    pub score: Vec<String>,
    /// Five inner selection partitions wholly within train. Full search repeats for each Search.
    pub inner_score: Vec<Vec<String>>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub schema: String,
    pub specification: Specification,
    pub searches: Vec<Search>,
    pub failures: Vec<String>,
    pub planned_units: usize,
    pub planned_searches: u64,
    pub estimated_candidate_fits: u64,
    pub estimated_fit_seconds: u64,
}

fn hash(bytes: &[u8]) -> String {
    jope_pipeline::sha256_hex(bytes)
}
fn digest(value: &impl Serialize) -> Result<String> {
    Ok(hash(&serde_json::to_vec(value)?))
}
fn hex(s: &str, n: usize) -> bool {
    s.len() == n
        && s.bytes()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
}
fn read_bytes(path: &Path) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)?
        .take(128 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    ensure!(bytes.len() <= 128 * 1024 * 1024, "artifact exceeds 128 MiB");
    Ok(bytes)
}
fn read<T: for<'a> Deserialize<'a>>(path: &Path) -> Result<T> {
    Ok(serde_json::from_slice(&read_bytes(path)?)?)
}

fn verify_asset(a: &Asset) -> Result<()> {
    ensure!(hex(&a.sha256, 64), "invalid asset digest");
    let mut file =
        std::fs::File::open(&a.path).with_context(|| format!("read {}", a.path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0; 65536];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    ensure!(
        format!("{:x}", digest.finalize()) == a.sha256,
        "altered asset {}",
        a.path.display()
    );
    Ok(())
}
fn asset(a: &Asset) -> Result<Vec<u8>> {
    ensure!(
        std::fs::metadata(&a.path)?.len() <= 128 * 1024 * 1024,
        "JSON evidence exceeds 128 MiB"
    );
    let bytes = std::fs::read(&a.path)?;
    ensure!(
        hex(&a.sha256, 64) && hash(&bytes) == a.sha256,
        "altered asset {}",
        a.path.display()
    );
    Ok(bytes)
}

fn distinct<'a>(items: impl Iterator<Item = &'a str>) -> bool {
    let mut seen = BTreeSet::new();
    items.into_iter().all(|x| !x.is_empty() && seen.insert(x))
}
fn partition(rows: &[&Row]) -> Vec<Vec<String>> {
    let groups: BTreeSet<_> = rows.iter().map(|r| &r.group).collect();
    let folds: BTreeMap<_, _> = groups
        .into_iter()
        .enumerate()
        .map(|(i, g)| (g, i % 5))
        .collect();
    (0..5)
        .map(|fold| {
            rows.iter()
                .filter(|r| folds[&r.group] == fold)
                .map(|r| r.id.clone())
                .collect()
        })
        .collect()
}
fn build(specification: Specification) -> Result<Plan> {
    let s = &specification;
    ensure!(
        s.schema == "jope.transfer-spec/v1" && !s.round.is_empty() && hex(&s.source_sha, 40),
        "invalid specification identity"
    );
    ensure!(
        distinct(s.datasets.iter().map(|d| d.id.as_str())) && !s.datasets.is_empty(),
        "duplicate/empty datasets"
    );
    ensure!(
        distinct(s.arms.iter().map(|a| a.id.as_str()))
            && s.arms.iter().any(|a| a.id == "parent")
            && s.arms.iter().any(|a| a.id == "challenger"),
        "unique parent/challenger arms required"
    );
    ensure!(
        !s.seeds.is_empty() && s.seeds.iter().collect::<BTreeSet<_>>().len() == s.seeds.len(),
        "duplicate/empty seeds"
    );
    ensure!(
        s.limits.cpus > 0
            && s.limits.memory_bytes > 0
            && s.limits.candidates_per_search > 0
            && s.limits.candidate_fit_seconds > 0,
        "positive resource estimates required"
    );
    let mut failures = Vec::new();
    // Check all byte identities, including unavailable inputs, without reading label values.
    for a in [&s.procedure, &s.encoder]
        .into_iter()
        .chain(s.assets.iter())
        .chain(s.arms.iter().map(|a| &a.checkpoint))
    {
        if let Err(e) = verify_asset(a) {
            failures.push(format!("asset: {e:#}"));
        }
    }
    let mut searches = Vec::new();
    let mut dataset_hashes = BTreeSet::new();
    let adaptive: Value = serde_json::from_str(include_str!("../../../../benchmarks/panel.json"))?;
    for d in &s.datasets {
        if adaptive["datasets"]
            .as_array()
            .context("compiled panel inventory")?
            .iter()
            .any(|known| {
                known["id"] == d.id
                    || known["source"]["sha256"] == d.features.sha256
                    || known["split"]["development_sha256"] == d.features.sha256
            })
        {
            failures.push(format!(
                "{}: existing adaptive panel is not untouched evaluation",
                d.id
            ));
        }
        ensure!(
            !d.family.is_empty() && distinct(d.rows.iter().map(|r| r.id.as_str())),
            "invalid family/duplicate rows"
        );
        ensure!(
            dataset_hashes.insert(&d.features.sha256),
            "duplicate dataset content (including family aliases)"
        );
        for exposure in [
            &d.evaluation_exposure,
            &d.checkpoint_exposure,
            &d.encoder_exposure,
        ] {
            ensure!(
                ["untouched", "adaptive_development", "unknown"].contains(&exposure.as_str()),
                "invalid exposure category"
            );
        }
        if d.evaluation_exposure != "untouched"
            || d.checkpoint_exposure != "untouched"
            || d.encoder_exposure != "untouched"
        {
            failures.push(format!("{}: contaminated or unknown evaluation/checkpoint/encoder exposure; no unseen-family claim", d.id));
        }
        for a in [&d.features, &d.labels, &d.exposure_evidence] {
            if let Err(e) = verify_asset(a) {
                failures.push(format!("{}: {e:#}", d.id));
            }
        }
        let mut groups = BTreeMap::new();
        for r in &d.rows {
            ensure!(
                r.outer < 5 && !r.group.is_empty(),
                "invalid outer fold/group"
            );
            if let Some(previous) = groups.insert(&r.group, r.outer) {
                ensure!(previous == r.outer, "group crosses outer folds");
            }
        }
        for seed in &s.seeds {
            for outer in 0..5 {
                let train: Vec<_> = d.rows.iter().filter(|r| r.outer != outer).collect();
                let score: Vec<_> = d
                    .rows
                    .iter()
                    .filter(|r| r.outer == outer)
                    .map(|r| r.id.clone())
                    .collect();
                let audit_folds = partition(&train);
                for audit in std::iter::once(None).chain((0..5).map(Some)) {
                    let (fit, scoring) = match audit {
                        None => (train.clone(), score.clone()),
                        Some(k) => (
                            train
                                .iter()
                                .copied()
                                .filter(|r| !audit_folds[k as usize].contains(&r.id))
                                .collect(),
                            audit_folds[k as usize].clone(),
                        ),
                    };
                    let inner_score = partition(&fit);
                    if scoring.is_empty() || inner_score.iter().any(Vec::is_empty) {
                        failures.push(format!("{} seed {seed} outer {outer} audit {audit:?}: insufficient groups/rows for five inner folds", d.id));
                    }
                    searches.push(Search {
                        dataset: d.id.clone(),
                        seed: *seed,
                        outer,
                        audit,
                        train: fit.iter().map(|r| r.id.clone()).collect(),
                        score: scoring,
                        inner_score,
                    });
                }
            }
        }
    }
    let planned_units = s
        .datasets
        .len()
        .checked_mul(s.seeds.len())
        .and_then(|n| n.checked_mul(5))
        .context("unit count overflow")?;
    let planned_searches = (planned_units as u64)
        .checked_mul(6)
        .and_then(|n| n.checked_mul(s.arms.len() as u64))
        .context("search count overflow")?;
    // Five inner fits per candidate plus one final refit per complete search.
    let per_search = s
        .limits
        .candidates_per_search
        .checked_mul(5)
        .and_then(|n| n.checked_add(1))
        .context("fit count overflow")?;
    let estimated_candidate_fits = planned_searches
        .checked_mul(per_search)
        .context("fit count overflow")?;
    let estimated_fit_seconds = estimated_candidate_fits
        .checked_mul(s.limits.candidate_fit_seconds)
        .context("time overflow")?;
    if planned_searches > s.limits.max_searches || estimated_fit_seconds > s.limits.max_fit_seconds
    {
        failures.push(
            "insufficient declared search/fit resources; estimates exclude search and I/O overhead"
                .into(),
        );
    }
    Ok(Plan {
        schema: PROTOCOL.into(),
        specification,
        searches,
        failures,
        planned_units,
        planned_searches,
        estimated_candidate_fits,
        estimated_fit_seconds,
    })
}

pub fn plan(input: &Path, output: &Path) -> Result<Value> {
    let plan = build(read(input)?)?;
    super::worker::write_private(output, &plan)?;
    Ok(
        json!({"ok":plan.failures.is_empty(),"plan":output,"plan_sha256":hash(&std::fs::read(output)?),"planned_units":plan.planned_units,"planned_searches":plan.planned_searches,"failures":plan.failures,"fits":0,"execution_admitted":false,"kpi":"not yet measured"}),
    )
}

/// A receipt is distinct from fixed-program teacher admission. Its closure must be
/// attested by a separately admitted executor; this module cannot establish isolation.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Prediction {
    pub schema: String,
    pub plan_sha256: String,
    pub search_sha256: String,
    pub arm: String,
    pub checkpoint_sha256: String,
    pub procedure_sha256: String,
    pub recipe_sha256: String,
    pub recipe: Asset,
    pub selection_receipt: Asset,
    pub fitting_rows: Vec<String>,
    pub selection_label_rows: Vec<String>,
    pub predictions: Vec<f64>,
    pub completed_sequence: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectionReceipt {
    pub schema: String,
    pub search_sha256: String,
    pub procedure_sha256: String,
    pub recipe_sha256: String,
    pub inner_score: Vec<Vec<String>>,
    pub label_dependencies: Vec<String>,
    pub complete_search_repeated: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FailedPrediction {
    pub search_sha256: String,
    pub arm: String,
    pub status: String,
    pub reason: String,
    pub completed_sequence: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Freeze {
    pub schema: String,
    pub plan_sha256: String,
    pub predictions: Vec<Asset>,
    pub failures: Vec<FailedPrediction>,
    pub frozen_sequence: u64,
}
/// Supplied by separately admitted custody, with its digest pinned outside the evidence
/// bundle. A self-authored JSON file is not independent proof of chronological isolation.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Release {
    pub schema: String,
    pub plan_sha256: String,
    pub freeze_sha256: String,
    pub released_sequence: u64,
    pub first_scored_result_sequence: u64,
    pub executor_source_sha: String,
    pub isolation_evidence: Asset,
    pub all_predictions_committed_before_scoring: bool,
    pub scoring_labels_excluded_from_selection_and_fitting: bool,
    pub no_cross_fold_result_feedback: bool,
    pub independently_verified: bool,
}
fn rmse(actual: &[f64], predicted: &[f64]) -> Result<f64> {
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
struct Gains {
    audit: f64,
    outer: f64,
    contribution: f64,
    audit_skill: f64,
    outer_skill: f64,
    parent_audit_skill: f64,
    parent_outer_skill: f64,
}
fn gains(audit: &[(f64, f64, f64)], outer: (f64, f64, f64)) -> Result<Gains> {
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
            let claimed: Option<Plan> = read(plan_path).ok();
            let failure = json!({"schema":"jope.transfer-report/v1","ok":false,
                "kpi_percentage_points":null,"status":"invalid evidence",
                "claimed_planned_units_unverified":claimed.as_ref().map(|p|p.planned_units),
                "failures":[format!("{error:#}")],"scores_released":false,"fits":0});
            if !output.exists() {
                super::worker::write_private(output, &failure)?;
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
        super::worker::write_private(output, &result)?;
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
    super::worker::write_private(output, &result)?;
    Ok(result)
}
fn average(gs: &[Gains]) -> Gains {
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

#[cfg(test)]
#[path = "transfer_tests.rs"]
mod tests;

fn sign(x: f64) -> i8 {
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
