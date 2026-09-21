//! Family reporting: the reporter's adaptive / best-known tables.
//!
//! Methods are merged into families by the configured prefix patterns (best score per dataset
//! wins), a family is *global-eligible* when it has scores in every task, the analysis
//! blacklists must be exactly the datasets some global family lacks, and every task table rates
//! its participants with [`crate::elo::ratings`]. Ported from the SOTArena reporter's `src/main.rs`
//! (`canonical_participants`, `global_family_ids`, `validate_blacklist_complements`,
//! `apply_analysis_blacklists`, `build_task_table`, `build_global`, `build_valid_report`,
//! `build_report`).

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{bail, Result};
use serde::Serialize;

use crate::elo::{metric_better, ratings, TaskResults};
use crate::reporter::{
    pattern_matches, AnalysisBlacklists, Config, DatasetCatalog, LoadedResults, ResultCatalog,
    TaskConfig,
};

/// Family- or method-keyed scores for one task (`family:<id>` / `method:<id>`).
#[derive(Clone, Debug)]
pub struct CanonicalParticipants {
    /// Participant → dataset → best score.
    pub scores: TaskResults,
    /// Participant → display name.
    pub display_names: BTreeMap<String, String>,
}

/// Task → participants.
pub type ParticipantCatalog = BTreeMap<String, CanonicalParticipants>;

/// `sotarena.report.v2`.
#[derive(Debug, Serialize)]
pub struct Report {
    /// Schema id.
    pub schema: &'static str,
    /// Datasets over every task.
    pub datasets_total: usize,
    /// Result JSON files read.
    pub result_jsons_total: usize,
    /// Scored (participant, dataset) cells over every task.
    pub score_rows: usize,
    /// Weighted family ladder.
    pub global: Vec<GlobalRow>,
    /// Per-task tables.
    pub tasks: Vec<TaskTable>,
    /// Strict `benchmark_valid` tables.
    pub benchmark_valid: BenchmarkValidReport,
}

/// The strict section of a [`Report`].
#[derive(Debug, Serialize)]
pub struct BenchmarkValidReport {
    /// Always `None` (no cross-task valid evidence yet).
    pub global: Option<Vec<GlobalRow>>,
    /// Per-task availability.
    pub tasks: Vec<ValidTaskTable>,
}

/// One task of the strict section.
#[derive(Debug, Serialize)]
pub struct ValidTaskTable {
    /// Task id.
    pub task: String,
    /// `available` / `unavailable`.
    pub status: &'static str,
    /// Why unavailable.
    pub reason: Option<&'static str>,
    /// The table when available.
    pub table: Option<TaskTable>,
}

/// One task's ladder.
#[derive(Debug, Serialize)]
pub struct TaskTable {
    /// Task id.
    pub task: String,
    /// Metric id.
    pub metric: String,
    /// `maximize` / `minimize`.
    pub metric_direction: &'static str,
    /// Global weight.
    pub weight: f64,
    /// Datasets in the task.
    pub datasets: usize,
    /// Datasets after the analysis blacklist.
    pub analysis_datasets: usize,
    /// Blacklisted datasets.
    pub blacklisted_datasets: usize,
    /// Result JSON files for the task.
    pub result_jsons: usize,
    /// Scored cells.
    pub score_rows: usize,
    /// Participants.
    pub methods: usize,
    /// Top-10 rows plus every global family.
    pub rows: Vec<TaskRow>,
}

/// One ladder row.
#[derive(Debug, Serialize)]
pub struct TaskRow {
    /// 1-based rank among full-coverage participants.
    pub rank: usize,
    /// Display name.
    pub method_id: String,
    /// Rating.
    pub elo: f64,
    /// Datasets scored.
    pub datasets_scored: usize,
}

/// One row of the global family ladder.
#[derive(Debug, Serialize)]
pub struct GlobalRow {
    /// 1-based rank.
    pub rank: usize,
    /// Family id.
    pub family_id: String,
    /// Family display name.
    pub display_name: String,
    /// Weighted sum of per-task family ratings.
    pub elo: f64,
}

/// The family (by first matching pattern) a method id belongs to.
pub fn family_of<'a>(
    config: &'a Config,
    method: &str,
) -> Option<&'a crate::reporter::FamilyConfig> {
    config.families.iter().find(|family| {
        family
            .patterns
            .iter()
            .any(|pattern| pattern_matches(pattern, method))
    })
}

/// Merge methods into families (best score per dataset), other methods stand alone.
pub fn canonical_participants(
    config: &Config,
    task: &TaskConfig,
    results: &TaskResults,
) -> CanonicalParticipants {
    let mut scores: TaskResults = BTreeMap::new();
    let mut display_names = BTreeMap::new();
    for (method, method_scores) in results {
        let (key, display_name) = match family_of(config, method) {
            Some(family) => (format!("family:{}", family.id), family.display_name.clone()),
            None => (format!("method:{method}"), method.clone()),
        };
        display_names.insert(key.clone(), display_name);
        let participant_scores = scores.entry(key).or_default();
        for (dataset, value) in method_scores {
            match participant_scores.get_mut(dataset) {
                Some(current) if metric_better(*value, *current, task.direction) => {
                    *current = *value;
                }
                Some(_) => {}
                None => {
                    participant_scores.insert(dataset.clone(), *value);
                }
            }
        }
    }
    CanonicalParticipants {
        scores,
        display_names,
    }
}

/// [`canonical_participants`] for every task.
pub fn canonical_catalog(config: &Config, results: &ResultCatalog) -> ParticipantCatalog {
    config
        .tasks
        .iter()
        .map(|task| {
            (
                task.id.clone(),
                canonical_participants(config, task, &results[&task.id]),
            )
        })
        .collect()
}

/// Families with at least one score in every task.
pub fn global_family_ids(config: &Config, participants: &ParticipantCatalog) -> BTreeSet<String> {
    config
        .families
        .iter()
        .filter(|family| {
            let key = format!("family:{}", family.id);
            config.tasks.iter().all(|task| {
                participants[&task.id]
                    .scores
                    .get(&key)
                    .is_some_and(|scores| !scores.is_empty())
            })
        })
        .map(|family| family.id.clone())
        .collect()
}

/// Each task's blacklist must be exactly the datasets some global family has not scored.
pub fn validate_blacklist_complements(
    config: &Config,
    datasets: &DatasetCatalog,
    participants: &ParticipantCatalog,
    global_families: &BTreeSet<String>,
    blacklists: &AnalysisBlacklists,
) -> Result<()> {
    for task in &config.tasks {
        let task_participants = &participants[&task.id];
        let expected: BTreeSet<_> = datasets[&task.id]
            .keys()
            .filter(|dataset| {
                global_families.iter().any(|family| {
                    !task_participants
                        .scores
                        .get(&format!("family:{family}"))
                        .is_some_and(|scores| scores.contains_key(*dataset))
                })
            })
            .cloned()
            .collect();
        let actual = &blacklists[&task.id];
        if actual != &expected {
            let remove = actual
                .difference(&expected)
                .next()
                .map(|hash| format!("; remove {hash}"))
                .unwrap_or_default();
            let add = expected
                .difference(actual)
                .next()
                .map(|hash| format!("; add {hash}"))
                .unwrap_or_default();
            bail!(
                "stale analysis blacklist for {}: {} listed, {} required{remove}{add}",
                task.id,
                actual.len(),
                expected.len()
            );
        }
    }
    Ok(())
}

/// Drop blacklisted datasets (and participants left without scores).
pub fn apply_analysis_blacklists(
    participants: &mut ParticipantCatalog,
    blacklists: &AnalysisBlacklists,
) {
    for (task, task_participants) in participants {
        let blacklist = &blacklists[task];
        let display_names = &mut task_participants.display_names;
        task_participants.scores.retain(|method, scores| {
            scores.retain(|dataset, _| !blacklist.contains(dataset));
            if scores.is_empty() {
                display_names.remove(method);
                false
            } else {
                true
            }
        });
    }
}

/// Rate a task's participants; rank the full-coverage ones; keep the top 10 plus global families.
pub fn build_task_table(
    config: &Config,
    task: &TaskConfig,
    datasets: usize,
    blacklisted_datasets: usize,
    result_jsons: usize,
    participants: &CanonicalParticipants,
    global_families: &BTreeSet<String>,
) -> TaskTable {
    let analysis_datasets = datasets - blacklisted_datasets;
    let method_ratings = ratings(task.direction, &config.elo, &participants.scores);
    let mut order: Vec<_> = participants.scores.keys().collect();
    order.retain(|method| participants.scores[*method].len() == analysis_datasets);
    order.sort_by(|left, right| {
        method_ratings[*right]
            .total_cmp(&method_ratings[*left])
            .then_with(|| left.cmp(right))
    });
    let rows = order
        .into_iter()
        .enumerate()
        .filter_map(|(index, method)| {
            let global_family = method
                .strip_prefix("family:")
                .is_some_and(|family| global_families.contains(family));
            (index < 10 || global_family).then(|| TaskRow {
                rank: index + 1,
                method_id: participants.display_names[method].clone(),
                elo: method_ratings[method],
                datasets_scored: participants.scores[method].len(),
            })
        })
        .collect();
    TaskTable {
        task: task.id.clone(),
        metric: task.metric.clone(),
        metric_direction: task.direction.as_str(),
        weight: task.weight,
        datasets,
        analysis_datasets,
        blacklisted_datasets,
        result_jsons,
        score_rows: participants.scores.values().map(BTreeMap::len).sum(),
        methods: participants.scores.len(),
        rows,
    }
}

/// Weighted family ladder over the eligible families.
pub fn build_global(
    config: &Config,
    participants: &ParticipantCatalog,
    eligible: &BTreeSet<String>,
) -> Vec<GlobalRow> {
    let family_ratings: BTreeMap<_, _> = config
        .tasks
        .iter()
        .map(|task| {
            let scores: TaskResults = eligible
                .iter()
                .map(|family| {
                    (
                        family.clone(),
                        participants[&task.id].scores[&format!("family:{family}")].clone(),
                    )
                })
                .collect();
            (
                task.id.clone(),
                ratings(task.direction, &config.elo, &scores),
            )
        })
        .collect();

    let mut global = Vec::new();
    for family in &config.families {
        if eligible.contains(&family.id) {
            let elo = config
                .tasks
                .iter()
                .map(|task| task.weight * family_ratings[&task.id][&family.id])
                .sum();
            global.push(GlobalRow {
                rank: 0,
                family_id: family.id.clone(),
                display_name: family.display_name.clone(),
                elo,
            });
        }
    }
    global.sort_by(|a, b| {
        b.elo
            .total_cmp(&a.elo)
            .then_with(|| a.family_id.cmp(&b.family_id))
    });
    for (index, row) in global.iter_mut().enumerate() {
        row.rank = index + 1;
    }
    global
}

/// Strict tables from `benchmark_valid` rows only (binary and regression).
pub fn build_valid_report(config: &Config, loaded: &LoadedResults) -> BenchmarkValidReport {
    let tasks = config
        .tasks
        .iter()
        .map(|task| {
            if !matches!(task.id.as_str(), "binary" | "regression") {
                return ValidTaskTable {
                    task: task.id.clone(),
                    status: "unavailable",
                    reason: Some("no explicit benchmark-valid ASPFM evidence"),
                    table: None,
                };
            }
            let participants =
                canonical_participants(config, task, &loaded.valid_catalog[&task.id]);
            let datasets = participants
                .scores
                .values()
                .flat_map(BTreeMap::keys)
                .collect::<BTreeSet<_>>()
                .len();
            if datasets == 0 {
                return ValidTaskTable {
                    task: task.id.clone(),
                    status: "unavailable",
                    reason: Some("no explicit benchmark-valid rows"),
                    table: None,
                };
            }
            ValidTaskTable {
                task: task.id.clone(),
                status: "available",
                reason: None,
                table: Some(build_task_table(
                    config,
                    task,
                    datasets,
                    0,
                    loaded.result_jsons[&task.id],
                    &participants,
                    &BTreeSet::new(),
                )),
            }
        })
        .collect();
    BenchmarkValidReport {
        global: None,
        tasks,
    }
}

/// The full report.
pub fn build_report(
    config: &Config,
    datasets: &DatasetCatalog,
    loaded: &LoadedResults,
    blacklists: &AnalysisBlacklists,
) -> Result<Report> {
    let mut participants = canonical_catalog(config, &loaded.catalog);
    let global_families = global_family_ids(config, &participants);
    validate_blacklist_complements(
        config,
        datasets,
        &participants,
        &global_families,
        blacklists,
    )?;
    apply_analysis_blacklists(&mut participants, blacklists);
    let tasks = config
        .tasks
        .iter()
        .map(|task| {
            build_task_table(
                config,
                task,
                datasets[&task.id].len(),
                blacklists[&task.id].len(),
                loaded.result_jsons[&task.id],
                &participants[&task.id],
                &global_families,
            )
        })
        .collect::<Vec<_>>();
    let score_rows = tasks.iter().map(|task| task.score_rows).sum();
    let global = build_global(config, &participants, &global_families);
    Ok(Report {
        schema: "sotarena.report.v2",
        datasets_total: datasets.values().map(BTreeMap::len).sum(),
        result_jsons_total: loaded.result_jsons_total,
        score_rows,
        global,
        tasks,
        benchmark_valid: build_valid_report(config, loaded),
    })
}

/// Load a reporter root and build its report.
pub fn render_report(root: &std::path::Path) -> Result<Report> {
    let config = crate::reporter::load_config(root)?;
    let datasets = crate::reporter::load_datasets(root, &config)?;
    let loaded = crate::reporter::load_results(root, &config, &datasets)?;
    let blacklists = crate::reporter::load_analysis_blacklists(root, &config, &datasets)?;
    build_report(&config, &datasets, &loaded, &blacklists)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elo::Direction;
    use crate::reporter::{fixtures, FamilyConfig};

    /// The reporter's `configured_families_match_and_merge_historical_variants`, with the
    /// production family list.
    #[test]
    fn configured_families_match_and_merge_historical_variants() {
        let mut config = fixtures::test_config(Direction::Maximize);
        config.families = vec![
            ("aspfm", "ASPFM", vec!["jope-*", "aspfm*"]),
            ("tabicl", "TabICL", vec!["tabicl*", "mc20_tabicl*"]),
            (
                "tabfm",
                "TabFM",
                vec!["tabfm", "tabfm_classifier*", "tabfm_regressor*"],
            ),
            (
                "tabfm-ensemble",
                "TabFM Ensemble",
                vec!["tabfm_ensemble*", "mc20_tabfm_ensemble*"],
            ),
            (
                "tabpfn-v3",
                "TabPFN v3",
                vec!["tabpfn_v3*", "mc20_tabpfn_v3*", "tabpfn_ts_3"],
            ),
            ("fable-5", "Fable 5", vec!["fable5*"]),
            ("grok-4-5", "Grok 4.5", vec!["grok4p5*"]),
            ("openai-sol", "OpenAI *sol", vec!["gpt5p6sol_*"]),
            ("kimi-k3", "Kimi K3", vec!["kimik3*"]),
            ("catboost", "CatBoost", vec!["mc20_catboost*"]),
            ("timesfm", "TimesFM", vec!["timesfm*"]),
        ]
        .into_iter()
        .map(|(id, name, patterns)| FamilyConfig {
            id: id.to_owned(),
            display_name: name.to_owned(),
            patterns: patterns.into_iter().map(str::to_owned).collect(),
        })
        .collect();
        crate::reporter::validate_config(&config).unwrap();
        let cases = [
            ("jope-dime--attempt-a", "aspfm"),
            ("tabicl--attempt-a", "tabicl"),
            ("mc20_tabicl_v2", "tabicl"),
            ("tabfm_regressor--attempt-a", "tabfm"),
            ("tabfm_ensemble_regressor--attempt-a", "tabfm-ensemble"),
            ("mc20_tabfm_ensemble", "tabfm-ensemble"),
            ("tabpfn_v3_classifier--attempt-a", "tabpfn-v3"),
            ("mc20_tabpfn_v3", "tabpfn-v3"),
            ("tabpfn_ts_3", "tabpfn-v3"),
            ("fable5_regressor--attempt-a", "fable-5"),
            ("grok4p5_regressor--attempt-a", "grok-4-5"),
            ("gpt5p6sol_regressor--attempt-a", "openai-sol"),
            ("kimik3_regressor--attempt-a", "kimi-k3"),
        ];
        for (method, expected) in cases {
            assert_eq!(
                family_of(&config, method).map(|f| f.id.as_str()),
                Some(expected),
                "unexpected family for {method}"
            );
        }
        let variants = [
            ("jope-dime", "aspfm-binary", "aspfm"),
            ("tabicl--attempt-a", "mc20_tabicl_v2", "tabicl"),
            (
                "tabfm_ensemble_regressor--attempt-a",
                "mc20_tabfm_ensemble",
                "tabfm-ensemble",
            ),
            ("tabpfn_v3_classifier", "tabpfn_ts_3", "tabpfn-v3"),
            ("fable5_regressor", "fable5_regressor--attempt-a", "fable-5"),
            (
                "grok4p5_regressor",
                "grok4p5_regressor--attempt-a",
                "grok-4-5",
            ),
            (
                "gpt5p6sol_regressor",
                "gpt5p6sol_regressor--attempt-a",
                "openai-sol",
            ),
            ("kimik3_regressor", "kimik3_regressor--attempt-a", "kimi-k3"),
        ];
        let results: TaskResults = variants
            .iter()
            .flat_map(|(first, second, _)| {
                [
                    ((*first).to_owned(), BTreeMap::from([("a".to_owned(), 0.6)])),
                    (
                        (*second).to_owned(),
                        BTreeMap::from([("a".to_owned(), 0.9)]),
                    ),
                ]
            })
            .collect();
        let participants = canonical_participants(&config, &config.tasks[0], &results);
        for (_, _, family) in variants {
            assert_eq!(participants.scores[&format!("family:{family}")]["a"], 0.9);
        }
        assert!(!participants.scores.contains_key("family:tabfm"));
    }

    #[test]
    fn report_ranks_families_and_validates_blacklists() {
        let dir = tempfile::tempdir().unwrap();
        fixtures::write_reporter_root(
            dir.path(),
            &[
                ("family-a", "0123456789abcdef", 0.9, true),
                ("family-b", "0123456789abcdef", 0.7, false),
                ("solo", "0123456789abcdef", 0.8, false),
            ],
        );
        let report = render_report(dir.path()).unwrap();
        assert_eq!(report.schema, "sotarena.report.v2");
        assert_eq!(report.datasets_total, 1);
        assert_eq!(report.tasks.len(), 1);
        let table = &report.tasks[0];
        assert_eq!(
            table.methods, 2,
            "family-a and family-b merge into one participant"
        );
        assert_eq!(table.rows[0].method_id, "Family");
        assert_eq!(table.rows[1].method_id, "solo");
        assert!(table.rows[0].elo > table.rows[1].elo);
        assert_eq!(report.global.len(), 1);
        assert_eq!(report.global[0].family_id, "family");
        assert_eq!(report.global[0].rank, 1);
        assert_eq!(report.benchmark_valid.tasks[0].status, "available");
        assert_eq!(
            report.benchmark_valid.tasks[0]
                .table
                .as_ref()
                .unwrap()
                .methods,
            1
        );

        // A stale blacklist (listing the only dataset) is rejected with the fix spelled out.
        std::fs::write(
            dir.path().join("data/binary/analysis-blacklist.json"),
            r#"{"schema":"sotarena.analysis-blacklist","task":"binary","datasets":["0123456789abcdef"]}"#,
        )
        .unwrap();
        let error = render_report(dir.path()).unwrap_err().to_string();
        assert!(
            error.contains("stale analysis blacklist for binary"),
            "{error}"
        );
        assert!(error.contains("remove 0123456789abcdef"), "{error}");
    }
}
