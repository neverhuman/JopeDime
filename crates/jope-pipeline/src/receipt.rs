//! `jope.pipeline-receipt/v3`: what the run did, stage by stage.
//!
//! The committed JSON Schema under `schemas/` is the public contract; the Rust
//! type derives `JsonSchema` so `tests/schema_agreement.rs` can assert the two
//! agree on every object's required set and property list, exactly as
//! `jope-datasets` does for its catalog. Validation reuses
//! `jope_datasets::schema`, the workspace's JSON Schema 2020-12 subset validator.
//!
//! v3 adds `search.seeding` (the genomes generation 0 was seeded with, their digests and the
//! parent receipt they came from). v1 and v2 receipts keep validating against their own
//! committed schemas; nothing about them is reinterpreted.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub const RECEIPT_SCHEMA_ID: &str = "jope.pipeline-receipt/v3";

/// The committed receipt schema (JSON Schema 2020-12).
pub const RECEIPT_SCHEMA_JSON: &str =
    include_str!("../../../schemas/jope.pipeline-receipt.v3.schema.json");

/// The previous current schema: total-budget accounting and complete-coverage selection,
/// without seeding.
pub const V2_SCHEMA_ID: &str = "jope.pipeline-receipt/v2";
pub const V2_SCHEMA_JSON: &str =
    include_str!("../../../schemas/jope.pipeline-receipt.v2.schema.json");

pub const LEGACY_SCHEMA_ID: &str = "jope.pipeline-receipt/v1";
pub const LEGACY_SCHEMA_JSON: &str =
    include_str!("../../../schemas/jope.pipeline-receipt.v1.schema.json");

/// One genome generation 0 was seeded with.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Seed {
    /// Canonical pipeline DSL, exactly as it entered generation 0 with origin `incumbent`.
    pub genome: String,
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub program_sha256: String,
    /// `incumbent` (the parent receipt's selected genome) or `seeds_file`.
    pub source: String,
}

/// v3: how generation 0 was seeded. Present, with an empty `seeds`, on an unseeded run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Seeding {
    /// Genomes offered by `--incumbent` and `--seed-genomes` before canonicalisation,
    /// deduplication (including against the bootstrap) and the cap at the elite count.
    pub offered: u64,
    /// Admitted seeds, in the order they entered generation 0 after the bootstrap. Every one is
    /// re-measured on this run's cuts; no score is carried over from where it came from.
    pub seeds: Vec<Seed>,
    /// sha256 of the `--incumbent` receipt's bytes, which verified with its on-disk evidence.
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub parent_receipt_sha256: Option<String>,
    /// sha256 of the `--seed-genomes` file's bytes.
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub seeds_file_sha256: Option<String>,
}

/// Per-phase admission limits and actual work. One non-preemptible fit can
/// exceed seconds_limit; no later fit is admitted in that phase.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Phase {
    pub evaluation_limit: u64,
    pub seconds_limit: f64,
    pub completed_evaluations: u64,
    pub successful_measurements: u64,
    pub failed_measurements: u64,
    pub seconds: f64,
    pub stopped_by: String,
}

impl Phase {
    pub fn not_requested() -> Self {
        Self {
            evaluation_limit: 0,
            seconds_limit: 0.0,
            completed_evaluations: 0,
            successful_measurements: 0,
            failed_measurements: 0,
            seconds: 0.0,
            stopped_by: crate::search::NOT_REQUESTED.to_owned(),
        }
    }
}

/// Required for v2; absent for historical v1 receipts, which made no total-budget guarantee.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Accounting {
    pub exploration: Phase,
    pub comparison: Phase,
    pub active_seconds: f64,
    pub deadline_overrun: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Coverage {
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub program_sha256: String,
    pub measured_cuts: Vec<u64>,
    pub failed_cuts: Vec<u64>,
    pub missing_cuts: Vec<u64>,
}

/// A stage either ran, was deliberately skipped with a reason, or failed.
pub const STATUSES: [&str; 3] = ["passed", "skipped", "failed"];

/// One pipeline stage.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Stage {
    pub name: String,
    /// `passed`, `skipped` or `failed`.
    pub status: String,
    /// Why, for anything that is not a plain pass. A skip always carries one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    pub seconds: f64,
}

/// One file of the dataset, as hashed at resolution time.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct DatasetFile {
    pub name: String,
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub sha256: String,
    pub bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Dataset {
    pub id: String,
    #[schemars(regex(pattern = r"^[0-9a-f]{16}$"))]
    pub dataset_hash: String,
    /// Where the dataset was read from, as given on the command line.
    pub path: String,
    pub task: String,
    pub rows: u64,
    pub columns: u64,
    pub training_rows: u64,
    pub files: Vec<DatasetFile>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Prepared {
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub sha256: String,
    pub seed: u64,
    pub rows: u64,
    /// The partition each cut holds one fold out of.
    pub folds: u64,
    /// The cut indices that were built, in build order.
    pub cuts: Vec<u64>,
}

/// One cut's search, as its `search.json` published it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CutSearch {
    pub cut: u64,
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub search_sha256: String,
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub observations_sha256: String,
    /// Rows the fits on this cut saw, and rows they were scored on. Disjoint.
    pub train_rows: u64,
    pub hold_rows: u64,
    pub generations: u64,
    /// Fits attempted. One fit is one evaluation, whatever its outcome, so this
    /// is exactly `successful + failed`.
    pub completed_evaluations: u64,
    pub successful_measurements: u64,
    pub failed_or_unavailable_measurements: u64,
    /// Single cut: exploration's `evaluations`, `seconds`, or `search_space_exhausted`.
    /// Multiple cuts: comparison's outcome, with `seconds` before `evaluations`
    /// when both are reached, else `comparison_pool_exhausted`. Exploration's
    /// original reason remains in accounting.exploration.stopped_by.
    pub stopped_by: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(required)]
    pub accounting: Option<Accounting>,
    /// The best held-out RMSE measured on this cut, in the dataset's target units.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub best_validation_rmse: Option<f64>,
    /// That same candidate's held-out R^2.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub best_validation_r2: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub best_program: Option<String>,
}

/// The selected program and its measured evidence.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Best {
    pub genome: String,
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub program_sha256: String,
    /// Mean held-out RMSE over the cuts this program was measured on, in the
    /// dataset's own target units. There is no normalisation to undo: the search
    /// fits the committed rows as they are.
    pub validation_rmse: f64,
    /// Mean held-out R^2 over the same cuts — the fits never saw these rows.
    pub validation_r2: f64,
    /// Mean in-sample R^2, for the gap against `validation_r2`.
    pub train_r2: f64,
    /// Fit-and-score seconds this program cost across those cuts.
    pub seconds: f64,
    pub cuts_measured: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Search {
    /// The search implementation that produced these measurements.
    pub engine: String,
    pub rng_seed: u64,
    /// The `--budget` bounds, per cut.
    pub evaluations: u64,
    pub seconds: u64,
    /// Candidates alive per generation, and the transform depth they may reach.
    pub population: u64,
    pub max_transforms: u64,
    /// The genome generation 0 always contains, so a measured champion is never
    /// worse than it.
    pub bootstrap_genome: String,
    pub cuts: Vec<CutSearch>,
    /// v2: require every requested cut, then minimum mean RMSE, then canonical
    /// genome. v1 used coverage count before RMSE, including different partial sets.
    pub selection: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(required)]
    pub budget_policy: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(required)]
    pub compared_cuts: Option<Vec<u64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(required)]
    pub coverage: Option<Vec<Coverage>>,
    /// The genome handed to the emitters, always present.
    pub selected_genome: String,
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub selected_program_sha256: String,
    /// Absent when no cut produced a successful measurement.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub best: Option<Best>,
    /// v3: the seeds generation 0 started from and where they came from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(required)]
    pub seeding: Option<Seeding>,
}

/// One emitted source file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Artifact {
    pub language: String,
    /// Relative to the run's `emit/` directory. `cpp17` and `cpp20` share one
    /// path: there is a single C++ emitter and the standard is a compile flag.
    pub path: String,
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub sha256: String,
    pub bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub build: Option<String>,
}

/// A requested language the selected genome is not covered for.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Exclusion {
    pub language: String,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Receipt {
    pub schema: String,
    pub ok: bool,
    pub tool_version: String,
    /// The checkout this ran from, when it could be determined.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub git_sha: Option<String>,
    pub created_at: String,
    /// The `--policy` the run was given: `auto`, `skip` or `require`.
    pub policy: String,
    pub dataset: Dataset,
    pub prepared: Prepared,
    pub stages: Vec<Stage>,
    pub search: Search,
    pub artifacts: Vec<Artifact>,
    pub emit_exclusions: Vec<Exclusion>,
    /// sha256 over every cut's `observations.json` digest, concatenated as hex
    /// in cut order: one value binding the run to all of its evidence.
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub observations_sha256: String,
}

impl Receipt {
    /// Structural invariants the schema cannot express.
    pub fn invariant_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if ![RECEIPT_SCHEMA_ID, V2_SCHEMA_ID, LEGACY_SCHEMA_ID].contains(&self.schema.as_str()) {
            errors.push(format!(
                "schema is {:?}, expected {RECEIPT_SCHEMA_ID:?}",
                self.schema
            ));
        }
        for stage in &self.stages {
            if !STATUSES.contains(&stage.status.as_str()) {
                errors.push(format!(
                    "stage {}: status {:?} is not one of {STATUSES:?}",
                    stage.name, stage.status
                ));
            }
            if stage.status != "passed" && stage.reason.as_ref().is_none_or(String::is_empty) {
                errors.push(format!(
                    "stage {}: a {} stage must record a reason",
                    stage.name, stage.status
                ));
            }
        }
        for required in ["dataset", "prepare", "propose", "search", "select", "emit"] {
            if !self.stages.iter().any(|s| s.name == required) {
                errors.push(format!("stage {required} is missing"));
            }
        }
        if self.search.cuts.len() != self.prepared.cuts.len() {
            errors.push(format!(
                "{} cut searches for {} prepared cuts",
                self.search.cuts.len(),
                self.prepared.cuts.len()
            ));
        }
        for cut in &self.search.cuts {
            if cut
                .successful_measurements
                .checked_add(cut.failed_or_unavailable_measurements)
                != Some(cut.completed_evaluations)
            {
                errors.push(format!(
                    "cut {}: {} evaluations but {} + {} outcomes",
                    cut.cut,
                    cut.completed_evaluations,
                    cut.successful_measurements,
                    cut.failed_or_unavailable_measurements
                ));
            }
            if cut.successful_measurements > 0 && cut.best_program.is_none() {
                errors.push(format!(
                    "cut {}: {} measurement(s) but no best program",
                    cut.cut, cut.successful_measurements
                ));
            }
        }
        let measured: u64 = self
            .search
            .cuts
            .iter()
            .map(|c| c.successful_measurements)
            .fold(0u64, u64::saturating_add);
        if measured > 0 && self.search.selection == crate::select::UNMEASURED_FALLBACK {
            errors.push(format!(
                "{measured} successful measurement(s) but the selection is the unmeasured fallback"
            ));
        }
        // A passed emit stage with nothing emitted would be a receipt that says the run produced
        // code while listing none, which no reader could catch without re-deriving it.
        if let Some(emit) = self.stages.iter().find(|s| s.name == "emit") {
            if emit.status == "passed" && self.artifacts.is_empty() {
                errors.push("stage emit passed but the receipt names no artifact".to_owned());
            }
            if emit.status == "failed" && !self.artifacts.is_empty() {
                errors.push(format!(
                    "stage emit failed but the receipt names {} artifact(s)",
                    self.artifacts.len()
                ));
            }
        }
        if self.ok && self.stages.iter().any(|s| s.status == "failed") {
            errors.push("ok is true but a stage failed".to_owned());
        }
        if let Some(best) = &self.search.best {
            if best.genome != self.search.selected_genome {
                errors.push("the selected genome is not the best one".to_owned());
            }
            if best.cuts_measured == 0 {
                errors.push("a best program measured on no cut".to_owned());
            }
        } else if self.search.selection != crate::select::UNMEASURED_FALLBACK {
            errors.push(format!(
                "selection {:?} without a best program",
                self.search.selection
            ));
        }
        if self.schema == RECEIPT_SCHEMA_ID || self.schema == V2_SCHEMA_ID {
            self.v2_errors(&mut errors);
        }
        if self.schema == RECEIPT_SCHEMA_ID {
            self.v3_errors(&mut errors);
        } else if self.search.seeding.is_some() {
            errors.push(format!("{} receipts carry no seeding record", self.schema));
        }
        errors
    }

    /// v3: the seeding record is internally consistent and every seed was attempted on every
    /// requested cut, so its place in the selection rests on this run's measurements alone.
    fn v3_errors(&self, errors: &mut Vec<String>) {
        use std::collections::BTreeSet;
        let Some(seeding) = &self.search.seeding else {
            errors.push("v3 requires a seeding record".into());
            return;
        };
        if seeding.seeds.len() > crate::search::ELITES
            || seeding.seeds.len() as u64 > seeding.offered
        {
            errors.push(format!(
                "{} seed(s) from {} offered exceeds the cap of {}",
                seeding.seeds.len(),
                seeding.offered,
                crate::search::ELITES
            ));
        }
        let mut distinct = BTreeSet::new();
        for seed in &seeding.seeds {
            if crate::sha256_hex(seed.genome.as_bytes()) != seed.program_sha256
                || !distinct.insert(&seed.genome)
                || seed.genome == crate::search::BOOTSTRAP_GENOME
            {
                errors.push(format!(
                    "seed {}: digest, duplicate or bootstrap mismatch",
                    seed.program_sha256
                ));
            }
            match seed.source.as_str() {
                crate::seeds::FROM_INCUMBENT if seeding.parent_receipt_sha256.is_none() => {
                    errors.push("an incumbent seed without a parent receipt digest".into())
                }
                crate::seeds::FROM_SEEDS_FILE if seeding.seeds_file_sha256.is_none() => {
                    errors.push("a seeds-file seed without a seeds file digest".into())
                }
                crate::seeds::FROM_INCUMBENT | crate::seeds::FROM_SEEDS_FILE => {}
                other => errors.push(format!("seed source {other:?} is unknown")),
            }
            let covered = self.search.coverage.as_ref().is_some_and(|coverage| {
                coverage
                    .iter()
                    .any(|p| p.program_sha256 == seed.program_sha256 && p.missing_cuts.is_empty())
            });
            if !covered {
                errors.push(format!(
                    "seed {} was not attempted on every requested cut",
                    seed.program_sha256
                ));
            }
        }
    }

    fn v2_errors(&self, errors: &mut Vec<String>) {
        use std::collections::BTreeSet;
        let search = &self.search;
        let expected: BTreeSet<u64> = self.prepared.cuts.iter().copied().collect();
        let actual: BTreeSet<u64> = search.cuts.iter().map(|c| c.cut).collect();
        if expected.is_empty()
            || expected.len() != self.prepared.cuts.len()
            || actual != expected
            || actual.len() != search.cuts.len()
        {
            errors.push("cut identities must match the distinct prepared cuts".into());
        }
        if search.budget_policy.as_deref() != Some(crate::search::BUDGET_POLICY)
            || search.evaluations == 0
            || search.evaluations > crate::search::MAX_EVALUATIONS as u64
            || search.seconds == 0
            || search.seconds > crate::search::MAX_SECONDS
        {
            errors.push("invalid v2 total budget or missing budget policy".into());
        }
        if search.selection != crate::select::MIN_MEAN_VALIDATION_RMSE
            || search.compared_cuts.as_ref() != Some(&expected.iter().copied().collect::<Vec<_>>())
        {
            errors.push("v2 selection must compare all requested cuts".into());
        }
        let Some(coverage) = &search.coverage else {
            errors.push("v2 requires program coverage".into());
            return;
        };
        let mut programs = BTreeSet::new();
        for program in coverage {
            let all: Vec<u64> = program
                .measured_cuts
                .iter()
                .chain(&program.failed_cuts)
                .chain(&program.missing_cuts)
                .copied()
                .collect();
            if !programs.insert(&program.program_sha256)
                || all.len() != expected.len()
                || all.iter().copied().collect::<BTreeSet<_>>() != expected
            {
                errors.push(
                    "program coverage must partition the requested cuts without duplicates".into(),
                );
            }
        }
        for cut in &search.cuts {
            let Some(a) = &cut.accounting else {
                errors.push(format!("cut {}: v2 requires phase accounting", cut.cut));
                continue;
            };
            let (x, y) = (&a.exploration, &a.comparison);
            for phase in [x, y] {
                if phase
                    .successful_measurements
                    .checked_add(phase.failed_measurements)
                    != Some(phase.completed_evaluations)
                    || phase.completed_evaluations > phase.evaluation_limit
                    || !phase.seconds.is_finite()
                    || phase.seconds < 0.0
                    || !phase.seconds_limit.is_finite()
                    || phase.seconds_limit < 0.0
                {
                    errors.push(format!(
                        "cut {}: invalid phase outcomes or admission limit",
                        cut.cut
                    ));
                }
            }
            let reserve = search.cuts.len() > 1;
            let exploration_limit = if reserve {
                search.evaluations.div_ceil(2)
            } else {
                search.evaluations
            };
            let exploration_seconds = search.seconds as f64 / if reserve { 2.0 } else { 1.0 };
            let comparison_limit = if reserve {
                search.evaluations.saturating_sub(x.completed_evaluations)
            } else {
                0
            };
            let comparison_seconds = if reserve {
                (search.seconds as f64 - x.seconds).max(0.0)
            } else {
                0.0
            };
            if x.evaluation_limit != exploration_limit
                || x.seconds_limit != exploration_seconds
                || y.evaluation_limit != comparison_limit
                || (y.seconds_limit - comparison_seconds).abs() > 1e-9
                || cut.completed_evaluations > search.evaluations
                || x.completed_evaluations.checked_add(y.completed_evaluations)
                    != Some(cut.completed_evaluations)
                || x.successful_measurements
                    .checked_add(y.successful_measurements)
                    != Some(cut.successful_measurements)
                || x.failed_measurements.checked_add(y.failed_measurements)
                    != Some(cut.failed_or_unavailable_measurements)
                || !a.active_seconds.is_finite()
                || a.active_seconds < 0.0
                || (a.active_seconds - x.seconds - y.seconds).abs() > 1e-9
                || a.deadline_overrun != (a.active_seconds > search.seconds as f64)
            {
                errors.push(format!(
                    "cut {}: total budget/phase accounting mismatch",
                    cut.cut
                ));
            }
            let exploration_stop_valid = match x.stopped_by.as_str() {
                crate::search::STOPPED_BY_EVALUATIONS => {
                    x.completed_evaluations == x.evaluation_limit
                }
                crate::search::STOPPED_BY_SECONDS => {
                    x.seconds >= x.seconds_limit && x.completed_evaluations < x.evaluation_limit
                }
                crate::search::STOPPED_BY_EXHAUSTED => x.completed_evaluations < x.evaluation_limit,
                _ => false,
            };
            if !exploration_stop_valid {
                errors.push(format!("cut {}: exploration stop reason mismatch", cut.cut));
            }
            let expected_stop = if !reserve {
                x.stopped_by.as_str()
            } else if a.active_seconds >= search.seconds as f64 {
                crate::search::STOPPED_BY_SECONDS
            } else if cut.completed_evaluations >= search.evaluations {
                crate::search::STOPPED_BY_EVALUATIONS
            } else {
                crate::search::STOPPED_BY_POOL
            };
            if cut.stopped_by != expected_stop
                || (reserve && y.stopped_by != cut.stopped_by)
                || (!reserve && *y != Phase::not_requested())
            {
                errors.push(format!("cut {}: phase stop reason mismatch", cut.cut));
            }
            if coverage
                .iter()
                .filter(|p| p.measured_cuts.contains(&cut.cut))
                .count() as u64
                != cut.successful_measurements
                || coverage
                    .iter()
                    .filter(|p| p.failed_cuts.contains(&cut.cut))
                    .count() as u64
                    != cut.failed_or_unavailable_measurements
            {
                errors.push(format!("cut {}: coverage/outcome mismatch", cut.cut));
            }
        }
        if let Some(best) = &search.best {
            if best.cuts_measured != expected.len() as u64
                || best.program_sha256 != search.selected_program_sha256
                || crate::sha256_hex(best.genome.as_bytes()) != best.program_sha256
                || !coverage.iter().any(|p| {
                    p.program_sha256 == best.program_sha256
                        && p.measured_cuts.len() == expected.len()
                })
            {
                errors.push("selected program lacks complete bound cut coverage".into());
            }
        }
    }
}

/// The committed schema, parsed.
pub fn schema() -> serde_json::Value {
    serde_json::from_str(RECEIPT_SCHEMA_JSON).expect("committed receipt schema parses")
}

/// The schema schemars derives from [`Receipt`].
pub fn derived_schema() -> serde_json::Value {
    let mut value = schemars::schema_for!(Receipt).to_value();
    value["title"] = RECEIPT_SCHEMA_ID.into();
    value["$id"] =
        "https://github.com/veox-ai/JopeDime/schemas/jope.pipeline-receipt.v3.schema.json".into();
    value["properties"]["schema"]["const"] = RECEIPT_SCHEMA_ID.into();
    value["properties"]["policy"]["enum"] = serde_json::json!(["auto", "skip", "require"]);
    value["$defs"]["Stage"]["properties"]["status"]["enum"] = serde_json::json!(STATUSES);
    let stops = serde_json::json!([
        "evaluations",
        "seconds",
        "search_space_exhausted",
        "comparison_pool_exhausted"
    ]);
    value["$defs"]["CutSearch"]["properties"]["stopped_by"]["enum"] = stops.clone();
    let mut phase_stops = stops.as_array().unwrap().clone();
    phase_stops.push(serde_json::json!("not_requested"));
    value["$defs"]["Phase"]["properties"]["stopped_by"]["enum"] = phase_stops.into();
    // serde(default) keeps v1 deserialization compatible. Schemars' `required`
    // annotation removes null from these Options but does not undo that default.
    for (definition, names) in [
        ("CutSearch", &["accounting"][..]),
        (
            "Search",
            &["budget_policy", "compared_cuts", "coverage", "seeding"][..],
        ),
    ] {
        let required = value["$defs"][definition]["required"]
            .as_array_mut()
            .unwrap();
        for name in names {
            let name = serde_json::json!(name);
            if !required.contains(&name) {
                required.push(name);
            }
        }
    }
    // Close objects consistently, including optional nested definitions.
    fn close(value: &mut serde_json::Value) {
        match value {
            serde_json::Value::Object(m) => {
                if m.contains_key("properties") {
                    m.insert("additionalProperties".into(), false.into());
                }
                for child in m.values_mut() {
                    close(child);
                }
            }
            serde_json::Value::Array(a) => {
                for child in a {
                    close(child);
                }
            }
            _ => {}
        }
    }
    close(&mut value);
    value
}

/// Validate one receipt document: the committed schema, then the Rust type, then
/// the invariants above.
pub fn validate_value(value: &serde_json::Value) -> Result<Receipt, Vec<String>> {
    let schema = if value["schema"] == LEGACY_SCHEMA_ID {
        serde_json::from_str(LEGACY_SCHEMA_JSON).expect("legacy schema parses")
    } else if value["schema"] == V2_SCHEMA_ID {
        serde_json::from_str(V2_SCHEMA_JSON).expect("v2 schema parses")
    } else {
        schema()
    };
    let errors = jope_datasets::schema::validate(&schema, value);
    if !errors.is_empty() {
        return Err(errors);
    }
    let receipt: Receipt =
        serde_json::from_value(value.clone()).map_err(|error| vec![error.to_string()])?;
    let errors = receipt.invariant_errors();
    if errors.is_empty() {
        Ok(receipt)
    } else {
        Err(errors)
    }
}

/// Read and validate a receipt file.
pub fn read(path: &std::path::Path) -> anyhow::Result<Receipt> {
    let bytes =
        std::fs::read(path).map_err(|error| anyhow::anyhow!("read {}: {error}", path.display()))?;
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|error| anyhow::anyhow!("parse {}: {error}", path.display()))?;
    validate_value(&value)
        .map_err(|errors| anyhow::anyhow!("{}: {}", path.display(), errors.join("; ")))
}
