use super::types::*;
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::Read,
    path::{Path, PathBuf},
};

pub(crate) const PROTOCOL: &str = "jope.transfer-procedure-audit/v1";
pub(crate) fn hash(bytes: &[u8]) -> String {
    jope_pipeline::sha256_hex(bytes)
}
pub(crate) fn digest(value: &impl Serialize) -> Result<String> {
    Ok(hash(&serde_json::to_vec(value)?))
}
pub(crate) fn hex(s: &str, n: usize) -> bool {
    s.len() == n
        && s.bytes()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
}
pub(crate) fn read_bytes(path: &Path) -> Result<Vec<u8>> {
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

pub(crate) fn verify_asset(a: &Asset) -> Result<()> {
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
pub(crate) fn asset(a: &Asset) -> Result<Vec<u8>> {
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
pub(crate) fn build(specification: Specification) -> Result<Plan> {
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
    let adaptive: Value =
        serde_json::from_str(include_str!("../../../../../benchmarks/panel.json"))?;
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
    super::super::worker::write_private(output, &plan)?;
    Ok(
        json!({"ok":plan.failures.is_empty(),"plan":output,"plan_sha256":hash(&std::fs::read(output)?),"planned_units":plan.planned_units,"planned_searches":plan.planned_searches,"failures":plan.failures,"fits":0,"execution_admitted":false,"kpi":"not yet measured"}),
    )
}
