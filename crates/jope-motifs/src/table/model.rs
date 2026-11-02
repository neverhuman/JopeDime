use super::*;

/// Lowercase hex sha256 of the domain-separated expansion text.
pub fn motif_id(expansion: &str) -> String {
    let mut bytes = EXPANSION_DOMAIN.to_vec();
    bytes.extend_from_slice(expansion.as_bytes());
    sha256_hex(&bytes)
}

/// The reserved prefix followed by the first 16 hex digits of the id.
pub fn motif_name(id: &str) -> String {
    format!("{NAME_PREFIX}{}", id.get(..16).unwrap_or(id))
}

/// A name that carries the reserved prefix and collides with no operator, head, target
/// stage, slot class or member name.
pub fn name_is_reserved(name: &str) -> bool {
    name.starts_with(NAME_PREFIX)
        && registry::operator(name).is_none()
        && !is_head_op(name)
        && !is_target_op(name)
        && stage_class(name).is_none()
        && members::find(name).is_none()
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// A sealed private table of mined fragments and their source provenance.
pub struct MotifTable {
    /// The format identifier, currently [`TABLE_SCHEMA`].
    pub schema: String,
    /// Lowercase SHA-256 integrity digest of the canonical table body.
    pub table_sha256: String,
    /// The engine, source, mining settings, counts and fragment records.
    pub body: TableBody,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// The content covered by a [`MotifTable`] digest.
pub struct TableBody {
    /// Registry and emitter identity used to derive fragment signatures.
    pub engine: EngineBinding,
    /// Identity and counts of the qualified source set.
    pub source: SourceBinding,
    /// Parameters used to select fragments.
    pub parameters: MineParameters,
    /// Counts for every qualified input disposition.
    pub inputs: InputCounts,
    /// Fragment definitions in deterministic table order.
    pub motifs: Vec<MotifDef>,
    /// Recipe sha256 to canonical program, exactly the recipes that provenance references.
    pub hosts: BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// The engine registry and emitter identity bound into a mined table.
pub struct EngineBinding {
    /// SHA-256 of the current operator registry.
    pub registry_sha256: String,
    /// Version string of the native artifact emitter.
    pub emitter_version: String,
    /// Schema identifier of the bound engine artifact.
    pub bound_artifact_schema: String,
}

impl EngineBinding {
    /// The engine this build links.
    pub fn current() -> Self {
        Self {
            registry_sha256: registry::registry_hash().to_owned(),
            emitter_version: transpile::EMITTER_VERSION.to_owned(),
            bound_artifact_schema: ARTIFACT_SCHEMA.to_owned(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// Identity and counts of the qualified source set used for mining.
pub struct SourceBinding {
    /// Schema identifier of the qualified source.
    pub schema: String,
    /// SHA-256 identity of the source read.
    pub sha256: String,
    /// Number of qualified programs in the source.
    pub programs: u32,
    /// Number of source records excluded before qualification.
    pub excluded_at_source: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// Selection thresholds and optional family labels used by [`crate::mine::mine`].
pub struct MineParameters {
    /// Minimum distinct datasets supporting a fragment.
    pub min_datasets: u32,
    /// Minimum distinct families; values above one require a family map.
    pub min_families: u32,
    /// Maximum fragment length in stages, from two through six.
    pub max_stages: u32,
    /// SHA-256 of the family map used, when one was supplied.
    pub family_map_sha256: Option<String>,
}

impl Default for MineParameters {
    fn default() -> Self {
        Self {
            min_datasets: 2,
            min_families: 1,
            max_stages: 6,
            family_map_sha256: None,
        }
    }
}

impl MineParameters {
    /// `min_datasets >= 1`, `min_families >= 1` (above 1 only with a family map) and
    /// `2 <= max_stages <= 6`.
    pub fn check(&self) -> Result<()> {
        let bad = Code("invalid_parameters");
        ensure!(self.min_datasets >= 1 && self.min_families >= 1, bad);
        ensure!((2..=6).contains(&self.max_stages), bad);
        ensure!(
            self.min_families == 1 || self.family_map_sha256.is_some(),
            bad
        );
        ensure!(self.family_map_sha256.as_deref().is_none_or(is_hex64), bad);
        Ok(())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// Counts of qualified inputs by their mining disposition.
pub struct InputCounts {
    /// Total number of qualified inputs considered.
    pub qualified: u32,
    /// Inputs classified as usable hosts.
    pub used: u32,
    /// Inputs excluded because optimization could have observed test labels.
    pub excluded_adaptive: u32,
    /// Inputs that were not pipeline programs.
    pub skipped_form: u32,
    /// Inputs containing the reserved fragment-reference prefix.
    pub skipped_reserved: u32,
    /// Inputs refused by the required input/transform slot contract.
    pub skipped_slots: u32,
    /// Inputs refused by the task-bound grammar and classification schema.
    pub skipped_unsupported: u32,
    /// Otherwise usable hosts dropped because their stages lack a supported type.
    pub dropped_untyped: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// One fragment, with its expansion, signature, support and source occurrences.
pub struct MotifDef {
    /// Lowercase SHA-256 of the domain-separated expansion text.
    pub id: String,
    /// Reserved reference name derived from the id.
    pub name: String,
    /// Canonical stage text used when expanding the reference.
    pub expansion: String,
    /// Registry-derived signature of the expanded stages.
    pub signature: SignatureV1,
    /// Current lifecycle state; mined fragments remain quarantined.
    pub lifecycle: Lifecycle,
    /// Recorded promotion gates; this build keeps every gate not run.
    pub promotion: Promotion,
    /// Presence counts recomputed from the qualified occurrences.
    pub support: Support,
    /// Every source occurrence used to derive this fragment.
    pub provenance: Vec<Occurrence>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Lifecycle label stored for a fragment.
pub enum Lifecycle {
    /// Mined for private review; not eligible for product use.
    Quarantined,
    /// Reserved schema state; promotion is unavailable in this build.
    Promoted,
}

/// The step-7 list of the product plan: new operators need ablation, semantic and backend
/// coverage and versioned vocabulary compatibility; fragments also need expanded runtime,
/// cross-family usefulness and replay. None of them is run here.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Promotion {
    /// Whether a matched baseline ablation has been performed.
    pub ablation: Gate,
    /// Whether semantic and backend coverage has been demonstrated.
    pub semantic_backend_coverage: Gate,
    /// Whether compatibility across versioned vocabularies has been checked.
    pub versioned_vocabulary_compatibility: Gate,
    /// Whether the expanded fragment has passed runtime evaluation.
    pub expanded_runtime: Gate,
    /// Whether usefulness has been shown across families.
    pub cross_family_usefulness: Gate,
    /// Whether recorded outcomes can be replayed exactly.
    pub replay: Gate,
}

/// Backward-compatible name for a promotion check state.
///
/// `Gate` in `jope-learning` represents an environment switch; this type represents whether a
/// promotion requirement has been run. Keep the public alias while giving the canonical enum a
/// domain-specific name.
pub type Gate = PromotionCheck;

impl Promotion {
    /// Every requirement recorded as not run.
    pub fn required_not_run() -> Self {
        Self {
            ablation: PromotionCheck::RequiredNotRun,
            semantic_backend_coverage: PromotionCheck::RequiredNotRun,
            versioned_vocabulary_compatibility: PromotionCheck::RequiredNotRun,
            expanded_runtime: PromotionCheck::RequiredNotRun,
            cross_family_usefulness: PromotionCheck::RequiredNotRun,
            replay: PromotionCheck::RequiredNotRun,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// State of one required promotion check.
pub enum PromotionCheck {
    /// The check has not been run; this is the only state emitted here.
    RequiredNotRun,
}

/// Presence among qualified programs, recomputable from provenance alone. It is not a
/// measure of usefulness.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Support {
    /// Number of distinct datasets containing the fragment.
    pub datasets: u32,
    /// Number of distinct labelled families, when a family map was supplied.
    pub families: Option<u32>,
    /// Number of qualified records containing the fragment.
    pub records: u32,
    /// Number of distinct recipes containing the fragment.
    pub recipes: u32,
    /// Sorted tasks represented by the occurrences.
    pub tasks: Vec<TaskLabel>,
    /// Whether every host recipe is in the known seed set.
    pub every_host_is_a_known_seed: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// One occurrence of a fragment in a qualified host program.
pub struct Occurrence {
    /// Digest of the qualified source record.
    pub record_sha256: String,
    /// Digest of the comparison identity associated with the record.
    pub comparison_sha256: String,
    /// Digest of the source dataset.
    pub dataset_sha256: String,
    /// Digest identifying the dataset cut.
    pub cut_sha256: String,
    /// Digest of the host's canonical program text.
    pub recipe_sha256: String,
    /// Task under which the host was qualified.
    pub task: TaskLabel,
    /// Absolute index of the first fragment stage in the host's stage list; it always lies
    /// inside the host's transform body.
    pub stage_offset: u32,
    /// Family label from the supplied family map, when available.
    pub family: Option<String>,
    /// Provenance of the labels used to qualify this occurrence.
    pub label: LabelOrigin,
}

/// Registry-derived signature of one expanded fragment.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignatureV1 {
    /// `transform_run`: rows in, rows out, with the (possibly transformed) target visible.
    pub slot: String,
    /// Expanded complexity; a reference earns no credit.
    pub stages: u32,
    /// Stage operator names in expansion order.
    pub ops: Vec<String>,
    /// Semantic registry keys corresponding to each stage.
    pub semantic_keys: Vec<String>,
    /// Whether any stage fits parameters from data.
    pub fitted: bool,
    /// Whether any stage consumes labels during fitting.
    pub requires_labels: bool,
    /// Conjunction of the registry task support of every stage.
    pub declared_tasks: TaskMask,
    /// Sorted set of the occurrence tasks.
    pub observed_tasks: Vec<TaskLabel>,
    /// The registry shape rule of every stage, as JSON text.
    pub shapes: Vec<String>,
    /// Whether each stage appends its input columns.
    pub appends: Vec<bool>,
    /// Sorted union of the registry tags.
    pub tags: Vec<String>,
    /// Recorded fact only; not a promotion precondition.
    pub all_registry_generatable: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// Task support declared by all stages in a fragment signature.
pub struct TaskMask {
    /// Every stage supports regression.
    pub regression: bool,
    /// Every stage supports binary classification.
    pub binary: bool,
    /// Every stage supports multiclass classification.
    pub multiclass: bool,
}

/// Motif order: datasets descending, stages descending, id ascending.
pub fn motif_order(a: &MotifDef, b: &MotifDef) -> Ordering {
    b.support
        .datasets
        .cmp(&a.support.datasets)
        .then(b.signature.stages.cmp(&a.signature.stages))
        .then_with(|| a.id.cmp(&b.id))
}

/// Provenance order: dataset, cut, comparison, record, offset.
pub fn occurrence_order(a: &Occurrence, b: &Occurrence) -> Ordering {
    (
        a.dataset_sha256.as_str(),
        a.cut_sha256.as_str(),
        a.comparison_sha256.as_str(),
        a.record_sha256.as_str(),
        a.stage_offset,
    )
        .cmp(&(
            b.dataset_sha256.as_str(),
            b.cut_sha256.as_str(),
            b.comparison_sha256.as_str(),
            b.record_sha256.as_str(),
            b.stage_offset,
        ))
}

/// Support recomputed from provenance and the host texts.
pub fn support_of(
    provenance: &[Occurrence],
    hosts: &BTreeMap<String, String>,
    with_families: bool,
) -> Result<Support> {
    let datasets: BTreeSet<&str> = provenance
        .iter()
        .map(|o| o.dataset_sha256.as_str())
        .collect();
    let records: BTreeSet<&str> = provenance
        .iter()
        .map(|o| o.record_sha256.as_str())
        .collect();
    let recipes: BTreeSet<&str> = provenance
        .iter()
        .map(|o| o.recipe_sha256.as_str())
        .collect();
    let tasks: BTreeSet<TaskLabel> = provenance.iter().map(|o| o.task).collect();
    let families = if with_families {
        let labels: BTreeSet<&str> = provenance
            .iter()
            .filter_map(|o| o.family.as_deref())
            .collect();
        Some(u32::try_from(labels.len())?)
    } else {
        None
    };
    let seeds = fragment::known_seeds();
    let every_host_is_a_known_seed = !recipes.is_empty()
        && recipes
            .iter()
            .all(|recipe| hosts.get(*recipe).is_some_and(|host| seeds.contains(host)));
    Ok(Support {
        datasets: u32::try_from(datasets.len())?,
        families,
        records: u32::try_from(records.len())?,
        recipes: u32::try_from(recipes.len())?,
        tasks: tasks.into_iter().collect(),
        every_host_is_a_known_seed,
    })
}
