//! Deterministic self-play dataset, wave, and GP policy.
//!
//! This module is intentionally free of Python and filesystem side effects. Callers
//! materialize files and compute their hashes, then pass those identities here for
//! fail-closed validation and deterministic policy decisions.

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

pub const MICRO_TASK_SIZE: usize = 64;
/// The two-task wave shape. Retained as the *legacy* size: it is what every cohort written
/// before cohorts declared their task set has, and what `micro_wave_size` returns for two
/// tasks. New code should call `micro_wave_size` rather than assume this constant.
pub const MICRO_WAVE_SIZE: usize = 128;
pub const STABLE_ANCHOR_SLOTS: usize = 16;
pub const LEAST_OBSERVED_SLOTS: usize = 32;
pub const HARD_UNCERTAIN_SLOTS: usize = 16;

/// Datasets in a micro-wave that lanes `tasks`.
///
/// The per-task structure stays frozen — every task contributes exactly `MICRO_TASK_SIZE`
/// rows split 16 anchor / 32 least-observed / 16 hard-uncertain — so widening the wave is
/// purely additive and touches no RNG modulus. `GP_SEED_SLOTS` is unaffected because
/// `export_gp_seeds` filters by `(task, dataset_hash)`: its 64-slot contract is per-dataset,
/// not per-wave.
pub fn micro_wave_size(tasks: &[String]) -> usize {
    tasks.len() * MICRO_TASK_SIZE
}

pub const WAVE_SIZE: usize = 256;
pub const LOCAL_ELITE_SLOTS: usize = 32;
pub const CHECKPOINT_PROPOSAL_SLOTS: usize = 64;
pub const COMMON_WINNER_SLOTS: usize = 96;
pub const OFFSPRING_SLOTS: usize = 64;
pub const DL_SLOT_RUNGS: [usize; 5] = [8, 16, 32, 48, 64];

pub const GP_SEED_SLOTS: usize = 64;
pub const GP_INJECTION_SLOTS: usize = 8;

const REQUIRED_ENCODING_NAMES: [&str; 3] =
    ["hyperion3train128dim.vs", "train3.vs", "train_target.vs"];

const VOLATILE_RECIPE_KEYS: [&str; 10] = [
    "seed",
    "random_seed",
    "generator_seed",
    "row_count",
    "rows",
    "n_rows",
    "n_train_rows",
    "n_test_rows",
    "train_rows",
    "test_rows",
];

const IDENTITY_RECIPE_KEYS: [&str; 9] = [
    "dataset_id",
    "dataset_hash",
    "id",
    "name",
    "created_at",
    "generated_at",
    "source_path",
    "train_path",
    "test_path",
];

fn sha256_hex(bytes: impl AsRef<[u8]>) -> String {
    hex::encode(Sha256::digest(bytes.as_ref()))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn canonical_value(value: &Value) -> Value {
    match value {
        Value::Object(values) => Value::Object(
            values
                .iter()
                .map(|(key, value)| (key.clone(), canonical_value(value)))
                .collect::<BTreeMap<_, _>>()
                .into_iter()
                .collect(),
        ),
        Value::Array(values) => Value::Array(values.iter().map(canonical_value).collect()),
        value => value.clone(),
    }
}

pub fn canonical_json_sha256(value: &Value) -> Result<String> {
    let canonical = canonical_value(value);
    Ok(sha256_hex(
        serde_json::to_vec(&canonical).context("serialize canonical JSON")?,
    ))
}

fn value_truthy(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => false,
        Some(Value::Bool(value)) => *value,
        Some(Value::Number(value)) => value.as_f64().map(|number| number != 0.0).unwrap_or(true),
        Some(Value::String(value)) => !value.is_empty(),
        Some(Value::Array(value)) => !value.is_empty(),
        Some(Value::Object(value)) => !value.is_empty(),
    }
}

fn stable_recipe(value: &Value) -> Value {
    match value {
        Value::Object(values) => {
            let stable = values
                .iter()
                .filter_map(|(key, value)| {
                    let normalized = key.to_ascii_lowercase();
                    let excluded = VOLATILE_RECIPE_KEYS.contains(&normalized.as_str())
                        || IDENTITY_RECIPE_KEYS.contains(&normalized.as_str())
                        || normalized.ends_with("_seed")
                        || normalized.contains("row_count");
                    (!excluded).then(|| (key.clone(), stable_recipe(value)))
                })
                .collect::<BTreeMap<_, _>>();
            Value::Object(stable.into_iter().collect())
        }
        Value::Array(values) => Value::Array(values.iter().map(stable_recipe).collect()),
        value => value.clone(),
    }
}

fn has_recipe_signal(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::String(value) => !value.is_empty() && value != "synthetic",
        Value::Array(values) => values.iter().any(has_recipe_signal),
        Value::Object(values) => values.values().any(has_recipe_signal),
        Value::Bool(_) | Value::Number(_) => true,
    }
}

/// Derive a stable source family, ignoring synthetic seed and row-count noise.
pub fn source_family(meta: &Value, dataset_id: &str) -> Result<String> {
    let object = meta
        .as_object()
        .context("dataset metadata must be a JSON object")?;
    let synthetic = value_truthy(object.get("synthetic_origin"))
        || object.get("origin").and_then(Value::as_str) == Some("synthetic")
        || dataset_id.starts_with("synth_");
    if synthetic {
        let mut recipe = object
            .get("generator_recipe")
            .filter(|value| value_truthy(Some(value)))
            .cloned()
            .unwrap_or_else(|| {
                let mut fallback = Map::new();
                for key in ["generator_name", "generator_version", "target_generation"] {
                    fallback.insert(
                        key.to_owned(),
                        object.get(key).cloned().unwrap_or(Value::Null),
                    );
                }
                Value::Object(fallback)
            });
        recipe = stable_recipe(&recipe);
        if !has_recipe_signal(&recipe) {
            recipe = stable_recipe(meta);
        }
        ensure!(
            has_recipe_signal(&recipe),
            "{dataset_id}: synthetic dataset has no stable generator recipe"
        );
        return Ok(format!("synthetic:{}", canonical_json_sha256(&recipe)?));
    }

    let family = if let Some(acquisition) = object.get("acquisition").and_then(Value::as_object) {
        json_object([
            (
                "source_path",
                acquisition
                    .get("source_path")
                    .cloned()
                    .unwrap_or(Value::Null),
            ),
            (
                "raw_sha256",
                acquisition
                    .get("raw_sha256")
                    .cloned()
                    .unwrap_or(Value::Null),
            ),
            (
                "dedupe_fingerprint",
                acquisition
                    .get("dedupe_fingerprint")
                    .cloned()
                    .unwrap_or(Value::Null),
            ),
        ])
    } else {
        json_object([
            (
                "dataset_name",
                object
                    .get("dataset_name")
                    .filter(|value| value_truthy(Some(value)))
                    .or_else(|| object.get("name"))
                    .cloned()
                    .unwrap_or(Value::Null),
            ),
            (
                "source",
                object
                    .get("source")
                    .filter(|value| value_truthy(Some(value)))
                    .or_else(|| object.get("origin"))
                    .cloned()
                    .unwrap_or(Value::Null),
            ),
            (
                "dedupe_fingerprint",
                object
                    .get("dedupe_fingerprint")
                    .cloned()
                    .unwrap_or(Value::Null),
            ),
        ])
    };
    let family = if family
        .as_object()
        .expect("json_object returns an object")
        .values()
        .any(|value| value_truthy(Some(value)))
    {
        family
    } else {
        json_object([("dataset_id", Value::String(dataset_id.to_owned()))])
    };
    Ok(format!("real:{}", canonical_json_sha256(&family)?))
}

fn json_object<const N: usize>(values: [(&str, Value); N]) -> Value {
    Value::Object(
        values
            .into_iter()
            .map(|(key, value)| (key.to_owned(), value))
            .collect(),
    )
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct DatasetIdentity {
    pub task: String,
    pub dataset_hash: String,
    pub train_sha256: String,
    pub test_sha256: String,
}

impl DatasetIdentity {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            matches!(
                self.task.as_str(),
                "regression" | "binary" | "timeseries" | "multiclass"
            ),
            "invalid task: {}",
            self.task
        );
        ensure!(!self.dataset_hash.is_empty(), "dataset hash is absent");
        ensure!(is_sha256(&self.train_sha256), "invalid train SHA-256");
        ensure!(is_sha256(&self.test_sha256), "invalid test SHA-256");
        Ok(())
    }
}

/// Fail closed when an authoritative dataset and its mirror disagree.
pub fn validate_mirror_identity(
    authoritative: &DatasetIdentity,
    mirror: &DatasetIdentity,
) -> Result<()> {
    authoritative.validate()?;
    mirror.validate()?;
    ensure!(
        authoritative.task == mirror.task,
        "mirror task identity conflict"
    );
    ensure!(
        authoritative.dataset_hash == mirror.dataset_hash,
        "mirror dataset identity conflict"
    );
    ensure!(
        authoritative.train_sha256 == mirror.train_sha256,
        "mirror train content conflict"
    );
    ensure!(
        authoritative.test_sha256 == mirror.test_sha256,
        "mirror test content conflict"
    );
    Ok(())
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CacheIdentity {
    pub dataset: DatasetIdentity,
    pub dataset_id: String,
    pub source_family: String,
    pub encoder_protocol_sha256: String,
    pub encoding_hashes: BTreeMap<String, String>,
}

impl CacheIdentity {
    pub fn validate(&self) -> Result<()> {
        self.dataset.validate()?;
        ensure!(!self.dataset_id.is_empty(), "cache dataset id is absent");
        ensure!(
            !self.source_family.is_empty(),
            "cache source family is absent"
        );
        ensure!(
            is_sha256(&self.encoder_protocol_sha256),
            "invalid cache encoder protocol SHA-256"
        );
        for required in REQUIRED_ENCODING_NAMES {
            let digest = self
                .encoding_hashes
                .get(required)
                .with_context(|| format!("cache encoding is absent: {required}"))?;
            ensure!(
                is_sha256(digest),
                "invalid cache encoding SHA-256: {required}"
            );
        }
        ensure!(
            self.encoding_hashes
                .values()
                .all(|digest| is_sha256(digest)),
            "invalid cache encoding SHA-256"
        );
        Ok(())
    }
}

pub fn validate_cache_identity(expected: &CacheIdentity, actual: &CacheIdentity) -> Result<()> {
    expected.validate()?;
    actual.validate()?;
    ensure!(expected == actual, "expanded cache identity drift");
    Ok(())
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CatalogEntryIdentity {
    pub dataset: DatasetIdentity,
    pub source_family: String,
    pub cache_manifest_sha256: String,
    pub encoder_protocol_sha256: String,
    pub context_protocol_sha256: String,
    pub context_sha256: String,
    pub encoding_hashes: BTreeMap<String, String>,
    pub catalog_index: usize,
    pub split: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CatalogIdentity {
    pub row_count: usize,
    pub task_counts: BTreeMap<String, usize>,
    pub data_sha256: String,
    pub entries: Vec<CatalogEntryIdentity>,
}

impl CatalogIdentity {
    /// Validate identities, hashes, task counts, and family-disjoint splits.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.row_count == self.entries.len(),
            "catalog row count drift"
        );
        ensure!(is_sha256(&self.data_sha256), "invalid catalog data SHA-256");
        let mut identities = HashSet::with_capacity(self.entries.len());
        let mut contexts = HashSet::with_capacity(self.entries.len());
        let mut family_splits: HashMap<&str, &str> = HashMap::new();
        let mut counts = BTreeMap::<String, usize>::new();
        for (index, entry) in self.entries.iter().enumerate() {
            entry.dataset.validate()?;
            ensure!(
                entry.catalog_index == index,
                "catalog index drift at row {index}"
            );
            ensure!(
                identities.insert((
                    entry.dataset.task.as_str(),
                    entry.dataset.dataset_hash.as_str()
                )),
                "catalog repeats a task/dataset identity"
            );
            ensure!(
                is_sha256(&entry.cache_manifest_sha256),
                "invalid cache manifest SHA-256"
            );
            ensure!(
                is_sha256(&entry.encoder_protocol_sha256)
                    && is_sha256(&entry.context_protocol_sha256)
                    && is_sha256(&entry.context_sha256),
                "invalid catalog context identity"
            );
            ensure!(
                contexts.insert(entry.context_sha256.as_str()),
                "catalog repeats a context identity"
            );
            for required in REQUIRED_ENCODING_NAMES {
                ensure!(
                    entry
                        .encoding_hashes
                        .get(required)
                        .is_some_and(|digest| is_sha256(digest)),
                    "catalog encoding is absent or invalid: {required}"
                );
            }
            ensure!(
                matches!(entry.split.as_str(), "train" | "validation"),
                "invalid catalog split: {}",
                entry.split
            );
            if let Some(prior) =
                family_splits.insert(entry.source_family.as_str(), entry.split.as_str())
            {
                ensure!(
                    prior == entry.split,
                    "source family crosses training and validation splits"
                );
            }
            *counts.entry(entry.dataset.task.clone()).or_default() += 1;
        }
        ensure!(counts == self.task_counts, "catalog task counts drift");
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct MicroDataset {
    pub task: String,
    pub dataset_hash: String,
    #[serde(default)]
    pub stable_anchor: bool,
    #[serde(default)]
    pub golden42: bool,
    pub current_catalog: Option<bool>,
    pub observation_count: Option<u64>,
    pub uncertainty: Option<f64>,
    pub hardness: Option<f64>,
    pub incumbent_regret: Option<f64>,
    /// Head score spread measured live from the corpus: `best - p90`, in the
    /// task's own metric. `None` means "never measured" (no observations).
    ///
    /// This is the learnability signal. When the top decile of genomes on a
    /// dataset all score within noise of the winner, "which genome wins here"
    /// carries no gradient for the policy: the dataset is converged and one
    /// more evaluation teaches nothing. Wide spread is the opposite -- the
    /// ranking is real and the dataset is worth a cohort slot.
    #[serde(default)]
    pub score_spread: Option<f64>,
    /// Distinct waves this dataset has been evaluated in SINCE the wave that
    /// last improved its all-time best score. `None` means no wave history.
    #[serde(default)]
    pub stale_waves: Option<u32>,
}

/// Breadth-over-depth admission policy for the micro-cohort.
///
/// Measured 2026-07-28 on waves 161-170: 459 of 1,280 cohort slots (35.9%) and
/// 28,718 of 75,336 observations (38.1%) went to datasets that ALREADY carried
/// >= 128 observations before the window, while 2,290 activated regression
/// > datasets and 169 binary datasets had never been touched at all. Every cohort
/// > slot costs exactly the same cluster time (11 manifests == 11 SCQ jobs per
/// > dataset per wave, verified: 128 * 11 == 1,408 manifests in every wave), so a
/// > slot spent on a converged dataset is a slot NOT spent on an unseen one.
///
/// Every rule here DEGRADES rather than fails. `select_micro_cohort` is on the
/// only path that produces strict waves; if it returns an error the supply
/// supervisor cannot build a wave, the queue drains and the cluster idles. So
/// each lane exhausts its preferred pool, then its fallback pool, and only ever
/// shrinks the *quality* of a pick -- never the count.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct MicroAdmission {
    /// Hard per-dataset evaluation cap. At or above this many observations a
    /// dataset stops competing for cohort slots.
    pub eval_cap: u64,
    /// Minimum observations before convergence can be asserted at all. Below
    /// this a dataset is simply under-measured, not converged.
    pub retire_min_observations: u64,
    /// Retire once this many waves have been evaluated without the all-time
    /// best score improving.
    pub retire_stale_waves: u32,
    /// Retire once the head spread (`best - p90`) falls to or below this. Zero
    /// disables the spread half of the retirement rule.
    pub retire_spread_floor: f64,
    /// Parole rotation period, in waves. Each capped or retired dataset becomes
    /// re-admissible on roughly one wave in `parole_period`, deterministically
    /// keyed on the wave id -- so retirement can never be permanent and a new
    /// genome family always gets a chance to overturn an old best.
    pub parole_period: u32,
    /// Slots per task reserved inside `hard_uncertain` for paroled datasets.
    /// This is the explicit, bounded reversibility budget.
    pub parole_slots: usize,
    /// Observation-count band width for `least_observed`. Datasets inside one
    /// band are ordered by score spread, so breadth still wins outright while
    /// ties (hundreds of binary datasets sit at exactly 29 observations) are
    /// broken toward datasets that still have something to teach.
    pub least_observed_band: u64,
}

impl Default for MicroAdmission {
    fn default() -> Self {
        Self {
            eval_cap: 128,
            retire_min_observations: 64,
            retire_stale_waves: 3,
            retire_spread_floor: 1e-4,
            parole_period: 64,
            parole_slots: 4,
            least_observed_band: 16,
        }
    }
}

impl MicroAdmission {
    /// The pre-breadth policy: no cap, no retirement, no spread ordering.
    /// Kept so the change is reversible in one flag without a redeploy.
    ///
    /// "Unbounded" lifts only the per-dataset caps; it creates no channel or task.
    /// [`select_micro_cohort`] still returns exactly [`MICRO_WAVE_SIZE`] datasets per
    /// wave, so the work a wave admits stays fixed (`unbounded_policy_admits_everything`).
    pub fn unbounded() -> Self {
        Self {
            eval_cap: u64::MAX,
            retire_min_observations: u64::MAX,
            retire_stale_waves: u32::MAX,
            retire_spread_floor: -1.0,
            parole_period: 1,
            parole_slots: 0,
            least_observed_band: 1,
        }
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.eval_cap > 0,
            "micro admission eval cap must be positive"
        );
        ensure!(
            self.parole_period > 0,
            "micro admission parole period must be positive"
        );
        ensure!(
            self.parole_slots <= HARD_UNCERTAIN_SLOTS,
            "micro admission parole slots exceed the hard-uncertain lane"
        );
        ensure!(
            self.least_observed_band > 0,
            "micro admission least-observed band must be positive"
        );
        ensure!(
            self.retire_spread_floor.is_finite(),
            "micro admission retire spread floor is not finite"
        );
        Ok(())
    }
}

/// Per-dataset admission verdict, derived fresh from the live corpus on every
/// wave. Nothing is persisted, so a dataset that improves -- or that a growing
/// corpus re-scores -- recovers on the very next wave with no bookkeeping.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Admission {
    /// Under the cap and not converged: competes for every lane.
    Open,
    /// At or above the cap.
    Capped,
    /// Under the cap but converged (no improvement, or no measurable spread).
    Retired,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MicroLane {
    StableAnchor,
    LeastObserved,
    HardUncertain,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct SelectedMicroDataset {
    pub dataset: MicroDataset,
    pub micro_lane: MicroLane,
}

fn deterministic_tie(wave_id: &str, task: &str, lane: &str, dataset_hash: &str) -> [u8; 32] {
    Sha256::digest(format!("{wave_id}|{task}|{lane}|{dataset_hash}").as_bytes()).into()
}

fn optional_truthy_number(value: Option<f64>) -> Option<f64> {
    value.filter(|value| *value != 0.0)
}

/// Live observation count for a dataset: the catalog's embedded reading when it
/// carries one, otherwise the corpus roll-up. The live catalog embeds none
/// (measured 2026-07-28: 0 of 19,951 entries carry `observation_count`), so in
/// production this always resolves through the corpus map.
fn observation_count(
    row: &MicroDataset,
    task: &str,
    observations: &HashMap<(String, String), u64>,
) -> u64 {
    row.observation_count
        .filter(|count| *count != 0)
        .or_else(|| {
            observations
                .get(&(task.to_owned(), row.dataset_hash.clone()))
                .copied()
        })
        .unwrap_or(0)
}

/// Deterministic parole: is this dataset's turn to be re-tested on this wave?
///
/// Keyed on the wave id, so the rotation advances every wave and every excluded
/// dataset comes up for re-entry within `parole_period` waves. Retirement is
/// therefore a throttle, never a gate: a genome family invented tomorrow will
/// still be measured against today's converged winners.
fn paroled(wave_id: &str, task: &str, dataset_hash: &str, period: u32) -> bool {
    if period <= 1 {
        return true;
    }
    let digest = deterministic_tie(wave_id, task, "parole", dataset_hash);
    let draw = u32::from_be_bytes([digest[0], digest[1], digest[2], digest[3]]);
    draw.is_multiple_of(period)
}

/// Band a dataset by observation count so that breadth wins outright but ties
/// inside a band are broken by learnability. Never-observed datasets get their
/// own band 0 and can never be displaced by a spread argument.
fn observation_band(count: u64, band: u64) -> u64 {
    if count == 0 {
        0
    } else {
        1 + count / band.max(1)
    }
}

/// Sort key for the head-spread signal, descending, with unmeasured last.
/// f64 has no `Ord`, so spread is quantised to a monotone integer key.
fn spread_rank(spread: Option<f64>) -> i64 {
    match spread {
        Some(value) if value.is_finite() => -((value.max(0.0) * 1e9) as i64),
        _ => i64::MAX,
    }
}

/// Select exactly 64 datasets per participating task using the 16/32/16
/// policy, under the breadth-over-depth admission rules in [`MicroAdmission`].
///
/// The legacy regression and binary tasks always participate (an empty legacy
/// pool is an error). Future tasks remain out of this fixed 128-dataset
/// protocol until their Stage-5 (D9) cohort policy exists; a partially admitted
/// future-task catalog must not stop legacy supply.
pub fn select_micro_cohort(
    entries: &[MicroDataset],
    observations: &HashMap<(String, String), u64>,
    wave_id: &str,
    admission: &MicroAdmission,
    active_optional_tasks: &[String],
) -> Result<Vec<SelectedMicroDataset>> {
    ensure!(!wave_id.is_empty(), "wave id is absent");
    admission.validate()?;
    ensure!(
        active_optional_tasks.iter().all(|task| {
            crate::selfplay_expansion::TASKS.contains(&task.as_str())
                && !matches!(task.as_str(), "regression" | "binary")
        }),
        "active optional task list may only name multiclass/timeseries"
    );
    let mut selected_all = Vec::with_capacity(MICRO_WAVE_SIZE);
    for task in crate::selfplay_expansion::TASKS {
        let pool = entries
            .iter()
            .filter(|row| row.task == task)
            .cloned()
            .collect::<Vec<_>>();
        // Regression and binary are mandatory -- they are the frozen backbone, and a
        // short pool there is a real defect that must surface. Multiclass and
        // timeseries join ONLY when explicitly activated by the operator AND their
        // pool can fill a whole lane. Explicit activation is the epoch-06 lane-01
        // correction to the original implicit opt-in: wave 546 was planned as a
        // 192-entry cohort the moment the catalog crossed 64 multiclass rows, and
        // the pinned proposer bundle -- still two-task -- rejected it 17 times and
        // idled the whole cluster behind the one-wave-in-flight rule. Task
        // membership must never toggle on mutable pool state alone; it is an
        // operator decision made after the downstream path is proven end to end.
        let mandatory = matches!(task, "regression" | "binary");
        if !mandatory
            && (!active_optional_tasks.iter().any(|active| active == task)
                || pool.len() < MICRO_TASK_SIZE)
        {
            continue;
        }
        ensure!(
            pool.len() >= MICRO_TASK_SIZE,
            "{task}: a micro-wave requires at least {MICRO_TASK_SIZE} datasets"
        );
        ensure!(
            pool.iter()
                .map(|row| row.dataset_hash.as_str())
                .collect::<HashSet<_>>()
                .len()
                == pool.len(),
            "{task}: duplicate dataset identity in micro-wave pool"
        );
        ensure!(
            pool.iter().all(|row| [
                row.uncertainty,
                row.hardness,
                row.incumbent_regret,
                row.score_spread
            ]
            .into_iter()
            .flatten()
            .all(f64::is_finite)),
            "{task}: non-finite hardness signal"
        );

        // ---- admission verdicts, recomputed from live signals each wave ----
        let mut verdict = HashMap::<&str, Admission>::with_capacity(pool.len());
        let mut counts = HashMap::<&str, u64>::with_capacity(pool.len());
        for row in &pool {
            let count = observation_count(row, task, observations);
            counts.insert(row.dataset_hash.as_str(), count);
            let converged = count >= admission.retire_min_observations
                && (row.stale_waves.unwrap_or(0) >= admission.retire_stale_waves
                    || row
                        .score_spread
                        .is_some_and(|spread| spread <= admission.retire_spread_floor));
            verdict.insert(
                row.dataset_hash.as_str(),
                if count >= admission.eval_cap {
                    Admission::Capped
                } else if converged {
                    Admission::Retired
                } else {
                    Admission::Open
                },
            );
        }
        let open =
            |row: &MicroDataset| verdict.get(row.dataset_hash.as_str()) == Some(&Admission::Open);
        let on_parole = |row: &MicroDataset| {
            !open(row) && paroled(wave_id, task, &row.dataset_hash, admission.parole_period)
        };
        let count_of =
            |row: &MicroDataset| counts.get(row.dataset_hash.as_str()).copied().unwrap_or(0);

        // ---- lane 1: stable anchors -----------------------------------------
        //
        // The anchor pool is the pre-expansion catalog core. It is 100% at or
        // above the cap today (measured: 1,561 of 1,568 eligible regression
        // anchors and 242 of 242 binary anchors carry >= 128 observations), so
        // an unconditional cap here would make every wave unplannable. Instead
        // the lane prefers open anchors, then spends any remainder on the
        // GLOBAL open pool -- which is exactly "return the slot to an unseen
        // dataset" -- and only falls back to capped anchors if a task has no
        // open datasets left at all.
        let mut anchors = Vec::with_capacity(STABLE_ANCHOR_SLOTS);
        let mut used = HashSet::<String>::with_capacity(MICRO_TASK_SIZE);
        let is_anchor = |row: &MicroDataset| {
            row.stable_anchor || row.golden42 || row.current_catalog.unwrap_or(true)
        };
        for stage in 0..4 {
            if anchors.len() == STABLE_ANCHOR_SLOTS {
                break;
            }
            let mut stage_pool = pool
                .iter()
                .filter(|row| !used.contains(&row.dataset_hash))
                .filter(|row| match stage {
                    // preferred: an anchor that is still under the cap
                    0 => is_anchor(row) && open(row),
                    // reallocation: give the slot to any unseen/immature dataset
                    1 => open(row),
                    // fall back to the anchor pool even though it is capped
                    2 => is_anchor(row),
                    // last resort: ANY dataset. `pool.len() >= MICRO_TASK_SIZE`
                    // is checked above, so this stage can always fill and the
                    // lane can never fail. The pre-WS-3 selector had no such
                    // stage: a catalog whose entries all carry
                    // `current_catalog: false` has an EMPTY anchor pool and made
                    // every wave unplannable, which idles the whole cluster.
                    _ => true,
                })
                .cloned()
                .collect::<Vec<_>>();
            stage_pool.sort_by_key(|row| {
                (
                    // stage 1 hands the slot to the least-observed dataset in
                    // the task; stages 0 and 2 keep the historical hash draw.
                    if stage == 1 { count_of(row) } else { 0 },
                    deterministic_tie(wave_id, task, "anchor", &row.dataset_hash),
                )
            });
            for row in stage_pool {
                if anchors.len() == STABLE_ANCHOR_SLOTS {
                    break;
                }
                used.insert(row.dataset_hash.clone());
                anchors.push(row);
            }
        }
        ensure!(
            anchors.len() == STABLE_ANCHOR_SLOTS,
            "{task}: insufficient stable anchors"
        );

        // ---- lane 2: least observed, spread-aware ---------------------------
        //
        // Ordering is (observation band ASC, head spread DESC, deterministic
        // tie). Breadth still wins outright -- a never-observed dataset is in
        // band 0 and no spread argument can displace it -- but the huge ties
        // inside a band (measured: 40+ binary datasets sit at exactly 29
        // observations) now resolve toward datasets whose ranking is still
        // learnable instead of resolving by hash.
        let mut least = Vec::with_capacity(LEAST_OBSERVED_SLOTS);
        for stage in 0..2 {
            if least.len() == LEAST_OBSERVED_SLOTS {
                break;
            }
            let mut stage_pool = pool
                .iter()
                .filter(|row| !used.contains(&row.dataset_hash))
                .filter(|row| if stage == 0 { open(row) } else { true })
                .cloned()
                .collect::<Vec<_>>();
            stage_pool.sort_by_key(|row| {
                (
                    observation_band(count_of(row), admission.least_observed_band),
                    spread_rank(row.score_spread),
                    deterministic_tie(wave_id, task, "least", &row.dataset_hash),
                )
            });
            for row in stage_pool {
                if least.len() == LEAST_OBSERVED_SLOTS {
                    break;
                }
                used.insert(row.dataset_hash.clone());
                least.push(row);
            }
        }
        ensure!(
            least.len() == LEAST_OBSERVED_SLOTS,
            "{task}: insufficient least-observed datasets"
        );

        // ---- lane 3: hard / uncertain, with a parole reserve -----------------
        //
        // The catalog's `uncertainty`/`hardness`/`incumbent_regret` columns are
        // null on all 19,951 live entries, so this lane's historical ordering
        // collapsed to the hash tie-break -- a uniform random draw over the
        // whole pool (verified: 0-1 of 32 datasets shared between consecutive
        // waves, 677 distinct datasets over 22 waves). Live head spread is the
        // signal those columns were meant to carry, so it is used whenever the
        // catalog supplies nothing.
        //
        // `parole_slots` of this lane are reserved for capped/retired datasets
        // whose turn has come round, widest spread first. That is the explicit
        // reversibility budget: a converged dataset keeps being re-tested
        // against new genome families, but on a bounded slot count instead of
        // the unbounded 35.9% of the cohort it takes today.
        let hard_signal = |row: &MicroDataset| {
            optional_truthy_number(row.uncertainty)
                .or_else(|| optional_truthy_number(row.hardness))
                .or_else(|| optional_truthy_number(row.incumbent_regret))
        };
        let order_hard = |left: &MicroDataset, right: &MicroDataset| {
            match (hard_signal(left), hard_signal(right)) {
                (Some(l), Some(r)) => r.total_cmp(&l),
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => {
                    spread_rank(left.score_spread).cmp(&spread_rank(right.score_spread))
                }
            }
            .then_with(|| {
                deterministic_tie(wave_id, task, "hard", &left.dataset_hash).cmp(
                    &deterministic_tie(wave_id, task, "hard", &right.dataset_hash),
                )
            })
        };
        let mut hard = Vec::with_capacity(HARD_UNCERTAIN_SLOTS);
        let parole_target = admission.parole_slots.min(HARD_UNCERTAIN_SLOTS);
        let mut parole = pool
            .iter()
            .filter(|row| !used.contains(&row.dataset_hash) && on_parole(row))
            .cloned()
            .collect::<Vec<_>>();
        parole.sort_by(|left, right| order_hard(left, right));
        for row in parole.into_iter().take(parole_target) {
            used.insert(row.dataset_hash.clone());
            hard.push(row);
        }
        for stage in 0..2 {
            if hard.len() == HARD_UNCERTAIN_SLOTS {
                break;
            }
            let mut stage_pool = pool
                .iter()
                .filter(|row| !used.contains(&row.dataset_hash))
                .filter(|row| if stage == 0 { open(row) } else { true })
                .cloned()
                .collect::<Vec<_>>();
            stage_pool.sort_by(|left, right| order_hard(left, right));
            for row in stage_pool {
                if hard.len() == HARD_UNCERTAIN_SLOTS {
                    break;
                }
                used.insert(row.dataset_hash.clone());
                hard.push(row);
            }
        }
        ensure!(
            hard.len() == HARD_UNCERTAIN_SLOTS,
            "{task}: insufficient hard datasets"
        );

        selected_all.extend(anchors.into_iter().map(|dataset| SelectedMicroDataset {
            dataset,
            micro_lane: MicroLane::StableAnchor,
        }));
        selected_all.extend(least.into_iter().map(|dataset| SelectedMicroDataset {
            dataset,
            micro_lane: MicroLane::LeastObserved,
        }));
        selected_all.extend(hard.into_iter().map(|dataset| SelectedMicroDataset {
            dataset,
            micro_lane: MicroLane::HardUncertain,
        }));
    }
    // The wave is however many lanes actually qualified, each exactly MICRO_TASK_SIZE.
    let laned_tasks = crate::selfplay_expansion::TASKS
        .iter()
        .filter(|task| selected_all.iter().any(|row| row.dataset.task == **task))
        .map(|task| (*task).to_owned())
        .collect::<Vec<_>>();
    let expected = micro_wave_size(&laned_tasks);
    ensure!(
        selected_all.len() == expected,
        "deterministic micro-cohort is incomplete"
    );
    ensure!(
        selected_all
            .iter()
            .map(|row| (row.dataset.task.as_str(), row.dataset.dataset_hash.as_str()))
            .collect::<HashSet<_>>()
            .len()
            == expected,
        "deterministic micro-cohort repeats a dataset identity"
    );
    Ok(selected_all)
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WaveSlotGroup {
    LocalElite,
    DlProposal,
    CommonWinner,
    Offspring,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateOrigin {
    LocalElite,
    RobustWinner,
    DlDirect,
    GpOffspring,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct WaveSlot {
    pub canonical_genome: String,
    pub semantic_sha256: String,
    pub slot_group: WaveSlotGroup,
    pub slot_index: usize,
}

/// Validate the legacy frozen 32/64/96/64 composition.
pub fn validate_wave_composition(slots: &[WaveSlot]) -> Result<()> {
    validate_wave_composition_with_dl(slots, CHECKPOINT_PROPOSAL_SLOTS)
}

/// Validate a 256-slot wave with adaptive direct-DL allocation.
///
/// The fixed 32 local elites and 96 robust winners leave 128 slots shared by
/// direct DL and same-wave GP offspring. A context may emit fewer direct
/// proposals than its task's rung; the unused slots transfer to offspring.
pub fn validate_wave_composition_with_dl(slots: &[WaveSlot], direct_dl_slots: usize) -> Result<()> {
    ensure!(
        direct_dl_slots <= *DL_SLOT_RUNGS.last().expect("nonempty rungs"),
        "direct-DL slots exceed the largest allocation rung"
    );
    ensure!(
        slots.len() == WAVE_SIZE,
        "wave must contain exactly 256 slots"
    );
    let mut semantics = HashSet::with_capacity(WAVE_SIZE);
    let mut indexes = BTreeMap::<WaveSlotGroup, BTreeSet<usize>>::new();
    for slot in slots {
        ensure!(
            sha256_hex(slot.canonical_genome.as_bytes()) == slot.semantic_sha256,
            "wave semantic identity drift"
        );
        ensure!(
            semantics.insert(slot.semantic_sha256.as_str()),
            "wave contains a duplicate semantic identity"
        );
        ensure!(
            indexes
                .entry(slot.slot_group)
                .or_default()
                .insert(slot.slot_index),
            "wave repeats an index within a slot group"
        );
    }
    for (group, required) in [
        (WaveSlotGroup::LocalElite, LOCAL_ELITE_SLOTS),
        (WaveSlotGroup::DlProposal, direct_dl_slots),
        (WaveSlotGroup::CommonWinner, COMMON_WINNER_SLOTS),
        (
            WaveSlotGroup::Offspring,
            CHECKPOINT_PROPOSAL_SLOTS + OFFSPRING_SLOTS - direct_dl_slots,
        ),
    ] {
        let actual = indexes.get(&group).cloned().unwrap_or_default();
        ensure!(
            actual == (0..required).collect(),
            "{group:?} slot composition drift"
        );
    }
    Ok(())
}

pub fn validate_dl_slot_rung(direct_dl_slots: usize) -> Result<()> {
    ensure!(
        DL_SLOT_RUNGS.contains(&direct_dl_slots),
        "direct-DL allocation must be one of 8,16,32,48,64"
    );
    Ok(())
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct GpProposal {
    pub task: String,
    pub dataset_hash: String,
    pub canonical_genome: String,
    pub semantic_sha256: String,
    pub model_checkpoint_sha256: String,
    pub slot: usize,
    pub score: Option<f64>,
    pub proposal_score: Option<f64>,
}

impl GpProposal {
    fn validate_identity(&self, checkpoint_sha256: &str) -> Result<()> {
        ensure!(
            self.model_checkpoint_sha256 == checkpoint_sha256,
            "GP seed proposal checkpoint provenance drift"
        );
        ensure!(
            sha256_hex(self.canonical_genome.as_bytes()) == self.semantic_sha256,
            "GP seed semantic identity drift"
        );
        ensure!(
            self.score.into_iter().all(f64::is_finite)
                && self.proposal_score.into_iter().all(f64::is_finite),
            "GP proposal has a non-finite score"
        );
        Ok(())
    }

    fn ranking_score(&self) -> f64 {
        self.score
            .or(self.proposal_score)
            .unwrap_or(f64::NEG_INFINITY)
    }
}

/// Validate and sort all 64 current-checkpoint proposals for one task/dataset.
pub fn export_gp_seeds(
    proposals: &[GpProposal],
    task: &str,
    dataset_hash: &str,
    checkpoint_sha256: &str,
) -> Result<Vec<GpProposal>> {
    ensure!(is_sha256(checkpoint_sha256), "invalid checkpoint SHA-256");
    let mut selected = proposals
        .iter()
        .filter(|row| row.task == task && row.dataset_hash == dataset_hash)
        .cloned()
        .collect::<Vec<_>>();
    for row in &selected {
        row.validate_identity(checkpoint_sha256)?;
    }
    selected.sort_by_key(|row| row.slot);
    ensure!(
        selected.len() == GP_SEED_SLOTS,
        "GP seed export requires all 64 unique proposal slots"
    );
    ensure!(
        selected
            .iter()
            .enumerate()
            .all(|(slot, row)| row.slot == slot),
        "GP seed export requires contiguous slots 0..63"
    );
    ensure!(
        selected
            .iter()
            .map(|row| row.semantic_sha256.as_str())
            .collect::<HashSet<_>>()
            .len()
            == GP_SEED_SLOTS,
        "GP seed export requires 64 unique semantics"
    );
    Ok(selected)
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct GpPopulationMember {
    pub canonical_genome: String,
    pub semantic_sha256: String,
    pub score: Option<f64>,
    #[serde(default)]
    pub dl_origin: bool,
    pub injected_checkpoint_sha256: Option<String>,
    pub ancestry: Option<GpAncestry>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct GpAncestry {
    pub operation: String,
    pub parents: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct GpReplacement {
    pub slot: usize,
    pub removed_semantic_sha256: String,
    pub injected_semantic_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct GpInjectionReceipt {
    pub checkpoint_sha256: String,
    pub population_before: usize,
    pub population_after: usize,
    pub elite_count: usize,
    pub requested: usize,
    pub injected: usize,
    pub replacements: Vec<GpReplacement>,
}

/// Replace up to eight lowest-scoring non-elites without growing the population.
pub fn inject_promoted_proposals(
    population: &[GpPopulationMember],
    proposals: &[GpProposal],
    checkpoint_sha256: &str,
    elite_count: usize,
) -> Result<(Vec<GpPopulationMember>, GpInjectionReceipt)> {
    ensure!(is_sha256(checkpoint_sha256), "invalid checkpoint SHA-256");
    ensure!(elite_count <= population.len(), "invalid GP elite count");
    ensure!(
        population
            .iter()
            .filter_map(|row| row.score)
            .all(f64::is_finite),
        "GP population has a non-finite score"
    );
    let existing = population
        .iter()
        .map(|row| row.semantic_sha256.as_str())
        .collect::<HashSet<_>>();
    let mut candidate_semantics = HashSet::new();
    let mut candidates = Vec::new();
    for proposal in proposals {
        if proposal.validate_identity(checkpoint_sha256).is_err()
            || existing.contains(proposal.semantic_sha256.as_str())
            || !candidate_semantics.insert(proposal.semantic_sha256.as_str())
        {
            continue;
        }
        candidates.push(proposal.clone());
    }
    candidates.sort_by(|left, right| {
        right
            .ranking_score()
            .total_cmp(&left.ranking_score())
            .then_with(|| left.slot.cmp(&right.slot))
            .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256))
    });
    candidates.truncate(GP_INJECTION_SLOTS.min(population.len().saturating_sub(elite_count)));

    let mut replaceable = (elite_count..population.len()).collect::<Vec<_>>();
    replaceable.sort_by(|left, right| {
        let left_score = population[*left].score.unwrap_or(f64::NEG_INFINITY);
        let right_score = population[*right].score.unwrap_or(f64::NEG_INFINITY);
        left_score
            .total_cmp(&right_score)
            .then_with(|| right.cmp(left))
    });
    replaceable.truncate(candidates.len());

    let mut members = population.to_vec();
    let mut replacements = Vec::with_capacity(candidates.len());
    for (index, proposal) in replaceable.into_iter().zip(candidates) {
        let removed = members[index].semantic_sha256.clone();
        members[index] = GpPopulationMember {
            canonical_genome: proposal.canonical_genome,
            semantic_sha256: proposal.semantic_sha256.clone(),
            score: proposal.score.or(proposal.proposal_score),
            dl_origin: true,
            injected_checkpoint_sha256: Some(checkpoint_sha256.to_owned()),
            ancestry: Some(GpAncestry {
                operation: "checkpoint_proposal_injection".to_owned(),
                parents: vec![proposal.semantic_sha256.clone()],
            }),
        };
        replacements.push(GpReplacement {
            slot: index,
            removed_semantic_sha256: removed,
            injected_semantic_sha256: proposal.semantic_sha256,
        });
    }
    let receipt = GpInjectionReceipt {
        checkpoint_sha256: checkpoint_sha256.to_owned(),
        population_before: population.len(),
        population_after: members.len(),
        elite_count,
        requested: GP_INJECTION_SLOTS,
        injected: replacements.len(),
        replacements,
    };
    ensure!(
        receipt.population_before == receipt.population_after,
        "GP injection changed the population budget"
    );
    Ok((members, receipt))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn digest(value: &str) -> String {
        sha256_hex(value.as_bytes())
    }

    fn micro(task: &str, index: usize) -> MicroDataset {
        MicroDataset {
            task: task.to_owned(),
            dataset_hash: format!("{index:016x}"),
            stable_anchor: index < 20,
            golden42: false,
            current_catalog: Some(false),
            observation_count: Some(index as u64),
            uncertainty: Some(index as f64),
            hardness: None,
            incumbent_regret: None,
            score_spread: None,
            stale_waves: None,
        }
    }

    /// A dataset carrying only live corpus signals, the way the planner builds
    /// them from `catalog + corpus` in production (the live catalog embeds no
    /// observation count and no hardness column at all).
    fn live(task: &str, index: usize, observations: u64, spread: f64, stale: u32) -> MicroDataset {
        MicroDataset {
            task: task.to_owned(),
            dataset_hash: format!("{index:016x}"),
            stable_anchor: false,
            golden42: false,
            current_catalog: Some(false),
            observation_count: Some(observations),
            uncertainty: None,
            hardness: None,
            incumbent_regret: None,
            score_spread: Some(spread),
            stale_waves: Some(stale),
        }
    }

    fn proposal(slot: usize, checkpoint: &str) -> GpProposal {
        let genome = format!("(pipe X (ridge lambda={slot}))");
        GpProposal {
            task: "regression".to_owned(),
            dataset_hash: "dataset".to_owned(),
            semantic_sha256: digest(&genome),
            canonical_genome: genome,
            model_checkpoint_sha256: checkpoint.to_owned(),
            slot,
            score: Some(slot as f64),
            proposal_score: None,
        }
    }

    #[test]
    fn synthetic_family_ignores_seed_and_row_count_but_not_recipe() {
        let first = json!({
            "origin": "synthetic",
            "generator_recipe": {
                "kind": "friedman",
                "noise": 0.2,
                "seed": 1,
                "n_rows": 1000
            }
        });
        let second = json!({
            "origin": "synthetic",
            "generator_recipe": {
                "kind": "friedman",
                "noise": 0.2,
                "seed": 999,
                "n_rows": 8000
            }
        });
        let different = json!({
            "origin": "synthetic",
            "generator_recipe": {"kind": "friedman", "noise": 0.4, "seed": 1}
        });
        assert_eq!(
            source_family(&first, "synth_1").unwrap(),
            source_family(&second, "synth_2").unwrap()
        );
        assert_ne!(
            source_family(&first, "synth_1").unwrap(),
            source_family(&different, "synth_3").unwrap()
        );
    }

    #[test]
    fn mirror_and_cache_identity_fail_closed() {
        let identity = DatasetIdentity {
            task: "regression".to_owned(),
            dataset_hash: "dataset".to_owned(),
            train_sha256: digest("train"),
            test_sha256: digest("test"),
        };
        assert!(validate_mirror_identity(&identity, &identity).is_ok());
        let mut conflict = identity.clone();
        conflict.test_sha256 = digest("changed");
        assert!(validate_mirror_identity(&identity, &conflict).is_err());

        let cache = CacheIdentity {
            dataset: identity,
            dataset_id: "dataset".to_owned(),
            source_family: "real:family".to_owned(),
            encoder_protocol_sha256: digest("encoder"),
            encoding_hashes: REQUIRED_ENCODING_NAMES
                .into_iter()
                .map(|name| (name.to_owned(), digest(name)))
                .collect(),
        };
        assert!(validate_cache_identity(&cache, &cache).is_ok());
        let mut changed = cache.clone();
        changed
            .encoding_hashes
            .insert("train_target.vs".to_owned(), digest("changed-target"));
        assert!(validate_cache_identity(&cache, &changed).is_err());
    }

    #[test]
    fn micro_cohort_is_deterministic_and_has_16_32_16_per_task() {
        let entries = ["regression", "binary"]
            .into_iter()
            .flat_map(|task| (0..80).map(move |index| micro(task, index)))
            .collect::<Vec<_>>();
        let policy = MicroAdmission::default();
        let first = select_micro_cohort(
            &entries,
            &HashMap::new(),
            "strict-wave-000001",
            &policy,
            &[],
        )
        .unwrap();
        let second = select_micro_cohort(
            &entries,
            &HashMap::new(),
            "strict-wave-000001",
            &policy,
            &[],
        )
        .unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), 128);
        for task in ["regression", "binary"] {
            let selected = first
                .iter()
                .filter(|row| row.dataset.task == task)
                .collect::<Vec<_>>();
            assert_eq!(selected.len(), 64);
            assert_eq!(
                selected
                    .iter()
                    .filter(|row| row.micro_lane == MicroLane::StableAnchor)
                    .count(),
                16
            );
            assert_eq!(
                selected
                    .iter()
                    .filter(|row| row.micro_lane == MicroLane::LeastObserved)
                    .count(),
                32
            );
            assert_eq!(
                selected
                    .iter()
                    .filter(|row| row.micro_lane == MicroLane::HardUncertain)
                    .count(),
                16
            );
        }
    }

    /// Activation is explicit AND floor-checked; a partial future task never blocks supply.
    ///
    /// Three properties in one fixture, learned from the wave-546 incident (a 192-entry
    /// cohort auto-planned the moment the catalog crossed 64 multiclass rows, while the
    /// pinned proposer was still two-task -- 17 identical failures, cluster idle):
    /// 1. an ACTIVATED task at or above the floor joins with a full 16/32/16 lane;
    /// 2. an activated task BELOW the floor (timeseries at 42) is quietly excluded
    ///    rather than raising -- it must never block supply;
    /// 3. without activation, the same pools produce a two-task 128 cohort no matter
    ///    how large the optional pools are -- pool size alone must never change shape.
    #[test]
    fn optional_tasks_join_only_when_activated_and_at_the_floor() {
        let entries = [
            ("regression", 80),
            ("binary", 80),
            ("timeseries", 42),
            ("multiclass", 80),
        ]
        .into_iter()
        .flat_map(|(task, count)| (0..count).map(move |index| micro(task, index)))
        .collect::<Vec<_>>();

        let activated = ["multiclass".to_owned(), "timeseries".to_owned()];
        let selected = select_micro_cohort(
            &entries,
            &HashMap::new(),
            "strict-wave-000002",
            &MicroAdmission::default(),
            &activated,
        )
        .unwrap();

        assert_eq!(selected.len(), 3 * MICRO_TASK_SIZE);
        for task in ["regression", "binary", "multiclass"] {
            assert_eq!(
                selected
                    .iter()
                    .filter(|row| row.dataset.task == task)
                    .count(),
                MICRO_TASK_SIZE,
                "{task} must fill exactly one lane"
            );
        }
        assert!(
            !selected.iter().any(|row| row.dataset.task == "timeseries"),
            "an activated task below the lane floor must not enter the cohort"
        );

        let unactivated = select_micro_cohort(
            &entries,
            &HashMap::new(),
            "strict-wave-000002",
            &MicroAdmission::default(),
            &[],
        )
        .unwrap();
        assert_eq!(
            unactivated.len(),
            2 * MICRO_TASK_SIZE,
            "without explicit activation the wave must stay two-task"
        );
        assert!(
            unactivated
                .iter()
                .all(|row| matches!(row.dataset.task.as_str(), "regression" | "binary")),
            "pool size alone must never widen the wave"
        );
    }

    /// A wide pool of unseen datasets plus a block of capped ones: the cap must
    /// keep every capped dataset out of the cohort except the bounded parole
    /// reserve, and the cohort must still be exactly 16/32/16 per task.
    #[test]
    fn evaluation_cap_keeps_mature_datasets_out_of_the_cohort() {
        let policy = MicroAdmission::default();
        let mut entries = Vec::new();
        for task in ["regression", "binary"] {
            // 400 datasets far past the cap, 400 that have never been touched.
            for index in 0..400 {
                entries.push(live(task, index, 2000 + index as u64, 0.5, 0));
            }
            for index in 400..800 {
                entries.push(live(task, index, 0, 0.0, 0));
            }
        }
        let cohort = select_micro_cohort(
            &entries,
            &HashMap::new(),
            "strict-wave-000042",
            &policy,
            &[],
        )
        .unwrap();
        assert_eq!(cohort.len(), MICRO_WAVE_SIZE);
        for task in ["regression", "binary"] {
            let rows = cohort
                .iter()
                .filter(|row| row.dataset.task == task)
                .collect::<Vec<_>>();
            assert_eq!(rows.len(), MICRO_TASK_SIZE);
            let capped = rows
                .iter()
                .filter(|row| row.dataset.observation_count.unwrap_or(0) >= policy.eval_cap)
                .count();
            // Only the parole reserve may hold capped datasets.
            assert!(
                capped <= policy.parole_slots,
                "{task}: {capped} capped datasets admitted, budget is {}",
                policy.parole_slots
            );
            // Anchor slots that would have gone to mature datasets are handed
            // to unseen ones instead.
            let unseen = rows
                .iter()
                .filter(|row| row.dataset.observation_count.unwrap_or(0) == 0)
                .count();
            assert!(
                unseen >= MICRO_TASK_SIZE - policy.parole_slots,
                "{task}: {unseen}"
            );
        }
    }

    /// Retirement must be a throttle, never a gate: every converged dataset is
    /// re-admissible within `parole_period` waves, with no stored state.
    #[test]
    fn retirement_is_reversible_through_parole_rotation() {
        let policy = MicroAdmission::default();
        let converged = live("regression", 7, 500, 0.0, 9);
        assert!(converged.score_spread.unwrap() <= policy.retire_spread_floor);
        let mut seen = 0;
        for wave in 0..(policy.parole_period * 4) {
            if paroled(
                &format!("strict-wave-{wave:06}"),
                "regression",
                &converged.dataset_hash,
                policy.parole_period,
            ) {
                seen += 1;
            }
        }
        assert!(
            seen > 0,
            "a retired dataset never came up for parole in {} waves",
            policy.parole_period * 4
        );
        // `parole_period == 1` (the unbounded policy) paroles everything.
        assert!(paroled("strict-wave-000001", "regression", "abc", 1));
    }

    /// Head spread orders `least_observed` inside an observation band, but can
    /// never displace a never-observed dataset.
    #[test]
    fn least_observed_prefers_wide_spread_inside_a_band() {
        let policy = MicroAdmission::default();
        let mut entries = Vec::new();
        for task in ["regression", "binary"] {
            // 8 untouched datasets: these must be admitted regardless of spread.
            for index in 0..8 {
                entries.push(live(task, index, 0, 0.0, 0));
            }
            // 200 datasets all sitting at exactly 29 observations, half of them
            // converged flat and half with a real ranking signal.
            for index in 8..208 {
                let spread = if index % 2 == 0 { 0.4 } else { 1e-9 };
                entries.push(live(task, index, 29, spread, 0));
            }
        }
        let cohort = select_micro_cohort(
            &entries,
            &HashMap::new(),
            "strict-wave-000043",
            &policy,
            &[],
        )
        .unwrap();
        for task in ["regression", "binary"] {
            let least = cohort
                .iter()
                .filter(|row| {
                    row.dataset.task == task && row.micro_lane == MicroLane::LeastObserved
                })
                .collect::<Vec<_>>();
            assert_eq!(least.len(), LEAST_OBSERVED_SLOTS);
            let flat = least
                .iter()
                .filter(|row| row.dataset.score_spread.unwrap_or(0.0) < 1e-6)
                .filter(|row| row.dataset.observation_count.unwrap_or(0) > 0)
                .count();
            assert_eq!(
                flat, 0,
                "{task}: converged-flat dataset admitted over a wide-spread peer"
            );
        }
        // Untouched datasets are never displaced: all 8 per task are in the cohort.
        for task in ["regression", "binary"] {
            let untouched = cohort
                .iter()
                .filter(|row| {
                    row.dataset.task == task && row.dataset.observation_count.unwrap_or(0) == 0
                })
                .count();
            assert_eq!(untouched, 8, "{task}");
        }
    }

    /// The failure mode that would idle the cluster: a task whose entire pool is
    /// past the cap. The planner must still return a complete cohort.
    #[test]
    fn a_fully_capped_pool_still_plans_a_complete_wave() {
        let policy = MicroAdmission::default();
        let mut entries = Vec::new();
        for task in ["regression", "binary"] {
            for index in 0..70 {
                let mut row = live(task, index, 5000, 0.0, 50);
                row.current_catalog = if index < 20 { None } else { Some(false) };
                entries.push(row);
            }
        }
        let cohort = select_micro_cohort(
            &entries,
            &HashMap::new(),
            "strict-wave-000044",
            &policy,
            &[],
        )
        .unwrap();
        assert_eq!(cohort.len(), MICRO_WAVE_SIZE);
        for task in ["regression", "binary"] {
            for (lane, want) in [
                (MicroLane::StableAnchor, STABLE_ANCHOR_SLOTS),
                (MicroLane::LeastObserved, LEAST_OBSERVED_SLOTS),
                (MicroLane::HardUncertain, HARD_UNCERTAIN_SLOTS),
            ] {
                assert_eq!(
                    cohort
                        .iter()
                        .filter(|row| row.dataset.task == task && row.micro_lane == lane)
                        .count(),
                    want,
                    "{task} {lane:?}"
                );
            }
        }
    }

    /// A catalog with NO anchor-pool members at all. The pre-WS-3 selector
    /// returned "insufficient stable anchors" here, which stops wave planning
    /// dead and drains the queue. The expansion driver writes
    /// `current_catalog: false` on every tranche it merges, so this is one
    /// merge away from being the live state, not a hypothetical.
    #[test]
    fn an_empty_anchor_pool_still_plans_a_complete_wave() {
        let policy = MicroAdmission::default();
        let mut entries = Vec::new();
        for task in ["regression", "binary"] {
            for index in 0..70 {
                // live() sets current_catalog = Some(false) and both flags false
                entries.push(live(task, index, 4000, 0.0, 40));
            }
        }
        assert!(entries
            .iter()
            .all(|row| !row.stable_anchor && !row.golden42 && row.current_catalog == Some(false)));
        let cohort = select_micro_cohort(
            &entries,
            &HashMap::new(),
            "strict-wave-000046",
            &policy,
            &[],
        )
        .unwrap();
        assert_eq!(cohort.len(), MICRO_WAVE_SIZE);
        for task in ["regression", "binary"] {
            assert_eq!(
                cohort
                    .iter()
                    .filter(
                        |row| row.dataset.task == task && row.micro_lane == MicroLane::StableAnchor
                    )
                    .count(),
                STABLE_ANCHOR_SLOTS,
                "{task}"
            );
        }
    }

    /// `MicroAdmission::unbounded` must reproduce the pre-breadth behaviour so
    /// the change can be reverted with a flag instead of a redeploy.
    #[test]
    fn unbounded_policy_admits_everything() {
        let policy = MicroAdmission::unbounded();
        let mut entries = Vec::new();
        for task in ["regression", "binary"] {
            for index in 0..70 {
                entries.push(live(task, index, 9000, 0.0, 99));
            }
        }
        let cohort = select_micro_cohort(
            &entries,
            &HashMap::new(),
            "strict-wave-000045",
            &policy,
            &[],
        )
        .unwrap();
        assert_eq!(cohort.len(), MICRO_WAVE_SIZE);
    }

    #[test]
    fn wave_composition_requires_256_unique_fixed_slots() {
        let mut slots = wave_slots(CHECKPOINT_PROPOSAL_SLOTS);
        validate_wave_composition(&slots).unwrap();
        slots[255].slot_index = 0;
        assert!(validate_wave_composition(&slots).is_err());
    }

    fn wave_slots(direct_dl_slots: usize) -> Vec<WaveSlot> {
        let mut slots = Vec::new();
        for (group, count) in [
            (WaveSlotGroup::LocalElite, LOCAL_ELITE_SLOTS),
            (WaveSlotGroup::DlProposal, direct_dl_slots),
            (WaveSlotGroup::CommonWinner, COMMON_WINNER_SLOTS),
            (
                WaveSlotGroup::Offspring,
                CHECKPOINT_PROPOSAL_SLOTS + OFFSPRING_SLOTS - direct_dl_slots,
            ),
        ] {
            for slot_index in 0..count {
                let genome = format!("({group:?}-{slot_index})");
                slots.push(WaveSlot {
                    semantic_sha256: digest(&genome),
                    canonical_genome: genome,
                    slot_group: group,
                    slot_index,
                });
            }
        }
        slots
    }

    #[test]
    fn adaptive_wave_composition_accepts_every_rung_and_context_deficiency() {
        for direct_dl_slots in DL_SLOT_RUNGS {
            validate_dl_slot_rung(direct_dl_slots).unwrap();
            validate_wave_composition_with_dl(&wave_slots(direct_dl_slots), direct_dl_slots)
                .unwrap();
        }
        validate_wave_composition_with_dl(&wave_slots(5), 5).unwrap();
        assert!(validate_dl_slot_rung(5).is_err());
    }

    #[test]
    fn gp_export_requires_all_64_current_checkpoint_slots() {
        let checkpoint = digest("checkpoint");
        let mut proposals = (0..64)
            .rev()
            .map(|slot| proposal(slot, &checkpoint))
            .collect::<Vec<_>>();
        let exported = export_gp_seeds(&proposals, "regression", "dataset", &checkpoint).unwrap();
        assert_eq!(
            exported.iter().map(|row| row.slot).collect::<Vec<_>>(),
            (0..64).collect::<Vec<_>>()
        );
        proposals.pop();
        assert!(export_gp_seeds(&proposals, "regression", "dataset", &checkpoint).is_err());
    }

    #[test]
    fn gp_injection_replaces_eight_lowest_non_elites() {
        let checkpoint = digest("checkpoint");
        let population = (0..20)
            .map(|index| {
                let genome = format!("(population-{index})");
                GpPopulationMember {
                    semantic_sha256: digest(&genome),
                    canonical_genome: genome,
                    score: Some(index as f64),
                    dl_origin: false,
                    injected_checkpoint_sha256: None,
                    ancestry: None,
                }
            })
            .collect::<Vec<_>>();
        let proposals = (0..64)
            .map(|slot| proposal(slot, &checkpoint))
            .collect::<Vec<_>>();
        let (updated, receipt) =
            inject_promoted_proposals(&population, &proposals, &checkpoint, 3).unwrap();
        assert_eq!(updated.len(), population.len());
        assert_eq!(receipt.injected, 8);
        assert!(updated[..3]
            .iter()
            .zip(&population[..3])
            .all(|(after, before)| after.semantic_sha256 == before.semantic_sha256));
        assert_eq!(
            receipt
                .replacements
                .iter()
                .map(|row| row.slot)
                .collect::<BTreeSet<_>>(),
            (3..11).collect()
        );
        assert!(receipt
            .replacements
            .iter()
            .all(|row| updated[row.slot].dl_origin));
    }

    #[test]
    fn catalog_rejects_family_split_leakage() {
        let identity = |task: &str, dataset_hash: &str| DatasetIdentity {
            task: task.to_owned(),
            dataset_hash: dataset_hash.to_owned(),
            train_sha256: digest(&format!("{dataset_hash}-train")),
            test_sha256: digest(&format!("{dataset_hash}-test")),
        };
        let entry = |index, task: &str, dataset_hash: &str, split: &str| CatalogEntryIdentity {
            dataset: identity(task, dataset_hash),
            source_family: "same-family".to_owned(),
            cache_manifest_sha256: digest(&format!("{dataset_hash}-manifest")),
            encoder_protocol_sha256: digest("encoder"),
            context_protocol_sha256: digest("context-protocol"),
            context_sha256: digest(&format!("{dataset_hash}-context")),
            encoding_hashes: REQUIRED_ENCODING_NAMES
                .into_iter()
                .map(|name| (name.to_owned(), digest(&format!("{dataset_hash}-{name}"))))
                .collect(),
            catalog_index: index,
            split: split.to_owned(),
        };
        let catalog = CatalogIdentity {
            row_count: 2,
            task_counts: BTreeMap::from([("regression".to_owned(), 2)]),
            data_sha256: digest("data"),
            entries: vec![
                entry(0, "regression", "a", "train"),
                entry(1, "regression", "b", "validation"),
            ],
        };
        assert!(catalog.validate().is_err());
    }
}
