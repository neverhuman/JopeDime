//! The private fragment table: exact types, identities, the integrity digest and the V0 checks.
//!
//! Every struct refuses unknown fields, maps are ordered and nothing is a float, so the bytes
//! of a table body are deterministic. The digest is integrity, not authenticity: anyone can
//! recompute it.

use crate::fragment::{self, HostBody};
use crate::source::{is_family_label, LabelOrigin, TaskLabel};
use crate::{is_hex64, sha256_hex, Code, NAME_PREFIX};
use anyhow::{bail, ensure, Result};
use feat_math::advanced_v3::ARTIFACT_SCHEMA;
use feat_math::heads::is_head_op;
use feat_math::kernel::pipeline::{stage_class, PipelineSlots};
use feat_math::target_transform::is_target_op;
use feat_math::{members, registry, transpile, GenomeIr};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

/// Private label of the table format. There is no schema file for it.
pub const TABLE_SCHEMA: &str = "jope.motif-table/v1";
const EXPANSION_DOMAIN: &[u8] = b"jope-motif-expansion/v1\0";
const TABLE_DOMAIN: &[u8] = b"jope-motif-table/v1\0";

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

impl Promotion {
    /// Every requirement recorded as not run.
    pub fn required_not_run() -> Self {
        Self {
            ablation: Gate::RequiredNotRun,
            semantic_backend_coverage: Gate::RequiredNotRun,
            versioned_vocabulary_compatibility: Gate::RequiredNotRun,
            expanded_runtime: Gate::RequiredNotRun,
            cross_family_usefulness: Gate::RequiredNotRun,
            replay: Gate::RequiredNotRun,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// State of one required promotion check.
pub enum Gate {
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

impl TableBody {
    /// `sha256(TABLE_DOMAIN || serde_json(body))`, lowercase hex.
    pub fn digest(&self) -> Result<String> {
        let mut bytes = TABLE_DOMAIN.to_vec();
        bytes.extend_from_slice(&serde_json::to_vec(self)?);
        Ok(sha256_hex(&bytes))
    }
}

impl MotifTable {
    /// Wrap a body with the schema label and its digest.
    pub fn seal(body: TableBody) -> Result<Self> {
        Ok(Self {
            schema: TABLE_SCHEMA.to_owned(),
            table_sha256: body.digest()?,
            body,
        })
    }

    /// Parse table bytes. Unknown fields at any level and a foreign schema label are refused.
    pub fn load(bytes: &[u8]) -> Result<Self> {
        let table: Self = serde_json::from_slice(bytes).map_err(|_| Code("table_schema"))?;
        ensure!(table.schema == TABLE_SCHEMA, Code("table_schema"));
        Ok(table)
    }

    /// Pretty JSON bytes of the table.
    pub fn to_json(&self) -> Result<Vec<u8>> {
        Ok(serde_json::to_vec_pretty(self)?)
    }

    /// V0: the table is internally consistent. This does not show that its hosts were
    /// qualified; the table is not authenticated.
    pub fn verify(&self) -> Result<()> {
        ensure!(self.schema == TABLE_SCHEMA, Code("table_schema"));
        ensure!(
            self.body.digest()? == self.table_sha256,
            Code("table_digest_mismatch")
        );
        ensure!(
            self.body.engine == EngineBinding::current(),
            Code("engine_drift")
        );
        let parameters = &self.body.parameters;
        parameters.check()?;
        verify_inputs(&self.body)?;
        let max_stages = usize::try_from(parameters.max_stages)?;
        let with_families = parameters.family_map_sha256.is_some();

        let mut hosts: BTreeMap<&str, HostBody> = BTreeMap::new();
        for (recipe, text) in &self.body.hosts {
            hosts.insert(recipe.as_str(), verify_host(recipe, text)?);
        }

        let mut texts = Vec::with_capacity(self.body.motifs.len());
        for motif in &self.body.motifs {
            texts.push(verify_definition(motif, max_stages)?);
        }

        let mut referenced: BTreeSet<&str> = BTreeSet::new();
        for (motif, stage_texts) in self.body.motifs.iter().zip(&texts) {
            verify_provenance(motif, stage_texts, &hosts, with_families)?;
            referenced.extend(motif.provenance.iter().map(|o| o.recipe_sha256.as_str()));
        }
        ensure!(
            hosts.keys().copied().collect::<BTreeSet<&str>>() == referenced,
            Code("provenance_inconsistent")
        );

        for motif in &self.body.motifs {
            let support = support_of(&motif.provenance, &self.body.hosts, with_families)?;
            ensure!(support == motif.support, Code("support_mismatch"));
            let families_pass =
                !with_families || support.families.unwrap_or(0) >= parameters.min_families;
            ensure!(
                support.datasets >= parameters.min_datasets && families_pass,
                Code("support_mismatch")
            );
        }

        for motif in &self.body.motifs {
            let stages = fragment::expansion_stages(&motif.expansion)
                .map_err(|_| Code("signature_drift"))?;
            let signature = fragment::signature(&stages, &motif.provenance)
                .map_err(|_| Code("signature_drift"))?;
            ensure!(signature == motif.signature, Code("signature_drift"));
        }

        verify_order(&self.body.motifs, &texts)
    }
}

/// V0: the recorded input counts and source binding agree with each other. Every qualified
/// program lands in exactly one class, the source count equals the qualified count, and the
/// source is labelled and bound to a 64-hex digest. This is internal consistency only: a
/// table that moves a program from one class to another still passes, and only a check
/// against the source records themselves can show the counts are true.
fn verify_inputs(body: &TableBody) -> Result<()> {
    let bad = Code("provenance_inconsistent");
    let inputs = &body.inputs;
    let classified: u64 = [
        inputs.used,
        inputs.excluded_adaptive,
        inputs.skipped_form,
        inputs.skipped_reserved,
        inputs.skipped_slots,
        inputs.skipped_unsupported,
    ]
    .into_iter()
    .map(u64::from)
    .sum();
    ensure!(classified == u64::from(inputs.qualified), bad);
    ensure!(body.source.programs == inputs.qualified, bad);
    ensure!(
        !body.source.schema.is_empty() && is_hex64(&body.source.sha256),
        bad
    );
    Ok(())
}

/// V0 step 5: the key is the digest of the text, which is a canonical, reference-free,
/// slotted pipeline.
fn verify_host(recipe: &str, text: &str) -> Result<HostBody> {
    let bad = Code("provenance_inconsistent");
    ensure!(
        is_hex64(recipe) && sha256_hex(text.as_bytes()) == recipe,
        bad
    );
    ensure!(!text.contains(NAME_PREFIX), bad);
    let ir = GenomeIr::parse(text).map_err(|_| bad)?;
    ensure!(ir.canonical() == text, bad);
    let GenomeIr::Pipeline(expr) = &ir else {
        bail!(bad)
    };
    let slots = PipelineSlots::from_pipeline(expr).map_err(|_| bad)?;
    let start = usize::from(slots.target.is_some()) + usize::from(slots.input.is_some());
    Ok(HostBody {
        texts: fragment::stage_texts(&expr.stages),
        body: start..start + slots.transforms.len(),
    })
}

/// V0 step 6: identity, name, canonical expansion, lifecycle and gates. Returns the stage
/// texts of the expansion.
fn verify_definition(motif: &MotifDef, max_stages: usize) -> Result<Vec<String>> {
    ensure!(
        motif.id == motif_id(&motif.expansion),
        Code("fragment_id_mismatch")
    );
    ensure!(
        motif.name == motif_name(&motif.id) && name_is_reserved(&motif.name),
        Code("fragment_name_mismatch")
    );
    let stages = fragment::expansion_stages(&motif.expansion)
        .map_err(|_| Code("expansion_not_canonical"))?;
    ensure!(
        fragment::join(&stages) == motif.expansion && (2..=max_stages).contains(&stages.len()),
        Code("expansion_not_canonical")
    );
    ensure!(
        !motif.expansion.contains(NAME_PREFIX),
        Code("nested_reference")
    );
    // No promotion evidence format exists yet; the gate type admits only "required_not_run".
    ensure!(
        motif.lifecycle == Lifecycle::Quarantined,
        Code("promotion_evidence_missing")
    );
    ensure!(
        motif.promotion == Promotion::required_not_run(),
        Code("promotion_evidence_missing")
    );
    Ok(fragment::stage_texts(&stages))
}

/// V0 step 7 for one motif: every occurrence names a recorded host whose body holds the
/// expansion at the recorded offset.
fn verify_provenance(
    motif: &MotifDef,
    texts: &[String],
    hosts: &BTreeMap<&str, HostBody>,
    with_families: bool,
) -> Result<()> {
    let bad = Code("provenance_inconsistent");
    ensure!(!motif.provenance.is_empty(), bad);
    for occurrence in &motif.provenance {
        let host = hosts.get(occurrence.recipe_sha256.as_str()).ok_or(bad)?;
        let start = usize::try_from(occurrence.stage_offset).map_err(|_| bad)?;
        let end = start.checked_add(texts.len()).ok_or(bad)?;
        ensure!(host.body.start <= start && end <= host.body.end, bad);
        ensure!(host.texts.get(start..end) == Some(texts), bad);
        ensure!(occurrence.family.is_some() == with_families, bad);
        ensure!(
            occurrence.family.as_deref().is_none_or(is_family_label),
            bad
        );
        for digest in [
            &occurrence.dataset_sha256,
            &occurrence.cut_sha256,
            &occurrence.comparison_sha256,
            &occurrence.record_sha256,
        ] {
            ensure!(is_hex64(digest), bad);
        }
        ensure!(
            occurrence.label.burn_sha256.as_deref().is_none_or(is_hex64),
            bad
        );
        ensure!(!occurrence.label.is_adaptive(), bad);
    }
    Ok(())
}

/// V0 step 10: specified order, unique ids and no fragment that is a contiguous sub-run of
/// another with an identical dataset set.
fn verify_order(motifs: &[MotifDef], texts: &[Vec<String>]) -> Result<()> {
    let bad = Code("table_order");
    for pair in motifs.windows(2) {
        ensure!(motif_order(&pair[0], &pair[1]) == Ordering::Less, bad);
    }
    for motif in motifs {
        for pair in motif.provenance.windows(2) {
            ensure!(occurrence_order(&pair[0], &pair[1]) == Ordering::Less, bad);
        }
    }
    let ids: BTreeSet<&str> = motifs.iter().map(|m| m.id.as_str()).collect();
    ensure!(ids.len() == motifs.len(), bad);
    let datasets: Vec<BTreeSet<&str>> = motifs.iter().map(dataset_set).collect();
    for (inner, inner_datasets) in texts.iter().zip(&datasets) {
        for (outer, outer_datasets) in texts.iter().zip(&datasets) {
            let closed_away =
                fragment::is_sub_run(inner, outer) && inner_datasets == outer_datasets;
            ensure!(!closed_away, bad);
        }
    }
    Ok(())
}

/// The distinct dataset digests of one motif's provenance.
pub fn dataset_set(motif: &MotifDef) -> BTreeSet<&str> {
    motif
        .provenance
        .iter()
        .map(|o| o.dataset_sha256.as_str())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code_of;
    use crate::fixture;
    use serde_json::Value;

    fn reseal(table: &mut MotifTable) {
        table.table_sha256 = table.body.digest().expect("digest");
    }

    fn verify_code(table: &MotifTable) -> Option<&'static str> {
        table.verify().err().and_then(|error| code_of(&error))
    }

    #[test]
    fn motif_id_is_domain_separated_sha256_of_expansion() {
        let expansion = "(cheb append=true degree=3) (pair_prod append=true topk=24)";
        let id = motif_id(expansion);
        let mut framed = b"jope-motif-expansion/v1\0".to_vec();
        framed.extend_from_slice(expansion.as_bytes());
        assert_eq!(id, sha256_hex(&framed));
        assert_ne!(id, sha256_hex(expansion.as_bytes()));
        assert!(is_hex64(&id));
        let name = motif_name(&id);
        assert_eq!(name, format!("motif_{}", &id[..16]));
        assert_eq!(name.len(), NAME_PREFIX.len() + 16);
        assert_ne!(motif_id(expansion), motif_id("(standardize) (sigmoid)"));
    }

    #[test]
    fn names_use_reserved_prefix_and_collide_with_no_vocabulary() {
        let table = fixture::table();
        assert!(!table.body.motifs.is_empty());
        for motif in &table.body.motifs {
            assert!(motif.name.starts_with(NAME_PREFIX));
            assert!(name_is_reserved(&motif.name));
            assert!(registry::operator(&motif.name).is_none());
            assert!(!is_head_op(&motif.name));
            assert!(!is_target_op(&motif.name));
            assert!(stage_class(&motif.name).is_none());
            assert!(members::find(&motif.name).is_none());
            for op in &motif.signature.ops {
                assert!(!motif.name.contains(op.as_str()));
            }
        }
        for descriptor in registry::operator_registry() {
            assert!(!name_is_reserved(descriptor.name));
        }
        for member in &members::CORE_V1 {
            assert!(!name_is_reserved(member.name));
        }
    }

    #[test]
    fn table_digest_detects_any_body_edit() {
        let table = fixture::table();
        assert!(table.verify().is_ok());
        let edits: [fn(&mut TableBody); 5] = [
            |body| body.inputs.qualified += 1,
            |body| body.motifs[0].support.records += 1,
            |body| body.motifs[0].provenance[0].stage_offset += 1,
            |body| body.parameters.min_datasets += 1,
            |body| {
                let first = body.hosts.keys().next().cloned().expect("a host");
                body.hosts.insert(first, String::from("(pipe X (ridge))"));
            },
        ];
        for edit in edits {
            let mut edited = table.clone();
            edit(&mut edited.body);
            assert_eq!(verify_code(&edited), Some("table_digest_mismatch"));
        }
        let mut relabelled = table.clone();
        relabelled.schema = String::from("jope.motif-table/v2");
        assert_eq!(verify_code(&relabelled), Some("table_schema"));
    }

    #[test]
    fn unknown_fields_are_refused_at_every_level() {
        let table = fixture::table();
        let bytes = table.to_json().expect("serialize");
        assert_eq!(MotifTable::load(&bytes).expect("round trip"), table);
        let base: Value = serde_json::from_slice(&bytes).expect("json");
        let paths: [&[&str]; 12] = [
            &[],
            &["body"],
            &["body", "engine"],
            &["body", "source"],
            &["body", "parameters"],
            &["body", "inputs"],
            &["body", "motifs", "0"],
            &["body", "motifs", "0", "signature"],
            &["body", "motifs", "0", "signature", "declared_tasks"],
            &["body", "motifs", "0", "promotion"],
            &["body", "motifs", "0", "support"],
            &["body", "motifs", "0", "provenance", "0", "label"],
        ];
        for path in paths {
            let mut value = base.clone();
            let mut node = &mut value;
            for key in path {
                node = match key.parse::<usize>() {
                    Ok(index) => &mut node[index],
                    Err(_) => &mut node[*key],
                };
            }
            node.as_object_mut()
                .expect("an object")
                .insert(String::from("unexpected"), Value::Bool(true));
            let bytes = serde_json::to_vec(&value).expect("serialize");
            let error = MotifTable::load(&bytes).expect_err("unknown field refused");
            assert_eq!(code_of(&error), Some("table_schema"), "at {path:?}");
        }
        let mut occurrence = base.clone();
        occurrence["body"]["motifs"][0]["provenance"][0]
            .as_object_mut()
            .expect("an object")
            .insert(String::from("unexpected"), Value::Bool(true));
        let bytes = serde_json::to_vec(&occurrence).expect("serialize");
        assert!(MotifTable::load(&bytes).is_err());
        let mut gate = base;
        gate["body"]["motifs"][0]["promotion"]["ablation"] = Value::from("passed");
        let bytes = serde_json::to_vec(&gate).expect("serialize");
        assert!(MotifTable::load(&bytes).is_err());
    }

    #[test]
    fn verify_rejects_nested_noncanonical_or_renamed_entries() {
        let table = fixture::table();

        let mut renamed = table.clone();
        renamed.body.motifs[0].name = String::from("motif_0000000000000000");
        reseal(&mut renamed);
        assert_eq!(verify_code(&renamed), Some("fragment_name_mismatch"));

        let mut foreign_id = table.clone();
        foreign_id.body.motifs[0].id = motif_id("(standardize) (sigmoid)");
        reseal(&mut foreign_id);
        assert_eq!(verify_code(&foreign_id), Some("fragment_id_mismatch"));

        let set_expansion = |expansion: &str| {
            let mut edited = table.clone();
            let motif = &mut edited.body.motifs[0];
            motif.expansion = expansion.to_owned();
            motif.id = motif_id(expansion);
            motif.name = motif_name(&motif.id);
            reseal(&mut edited);
            edited
        };
        // Parameters out of canonical order.
        let noncanonical =
            set_expansion("(cheb degree=3 append=true) (pair_prod append=true topk=24)");
        assert_eq!(verify_code(&noncanonical), Some("expansion_not_canonical"));
        let single = set_expansion("(cheb append=true degree=3)");
        assert_eq!(verify_code(&single), Some("expansion_not_canonical"));
        let nested = set_expansion("(motif_0123456789abcdef) (pair_prod append=true topk=24)");
        assert_eq!(verify_code(&nested), Some("nested_reference"));
    }

    #[test]
    fn hand_written_table_without_provenance_is_refused() {
        let table = fixture::table();
        let mut bare = table.clone();
        for motif in &mut bare.body.motifs {
            motif.provenance.clear();
        }
        bare.body.hosts.clear();
        reseal(&mut bare);
        assert_eq!(verify_code(&bare), Some("provenance_inconsistent"));

        let mut empty_first = table;
        empty_first.body.motifs[0].provenance.clear();
        reseal(&mut empty_first);
        assert_eq!(verify_code(&empty_first), Some("provenance_inconsistent"));
    }

    #[test]
    fn verify_rejects_inconsistent_provenance() {
        let table = fixture::table();

        let mut missing_host = table.clone();
        missing_host.body.motifs[0].provenance[0].recipe_sha256 = fixture::digest("absent");
        reseal(&mut missing_host);
        assert_eq!(verify_code(&missing_host), Some("provenance_inconsistent"));

        let mut shifted = table.clone();
        shifted.body.motifs[0].provenance[0].stage_offset += 1;
        reseal(&mut shifted);
        assert_eq!(verify_code(&shifted), Some("provenance_inconsistent"));

        let mut extra = table.clone();
        let text = fixture::canonical("(pipe X (standardize) (ridge))");
        extra.body.hosts.insert(sha256_hex(text.as_bytes()), text);
        reseal(&mut extra);
        assert_eq!(verify_code(&extra), Some("provenance_inconsistent"));

        let mut mislabelled = table.clone();
        let key = mislabelled
            .body
            .hosts
            .keys()
            .next()
            .cloned()
            .expect("a host");
        mislabelled
            .body
            .hosts
            .insert(key, fixture::canonical("(pipe X (ridge))"));
        reseal(&mut mislabelled);
        assert_eq!(verify_code(&mislabelled), Some("provenance_inconsistent"));

        let mut adaptive = table.clone();
        adaptive.body.motifs[0].provenance[0].label = fixture::adaptive_label();
        reseal(&mut adaptive);
        assert_eq!(verify_code(&adaptive), Some("provenance_inconsistent"));

        let mut family = table;
        family.body.motifs[0].provenance[0].family = Some(String::from(fixture::FAMILY_A));
        reseal(&mut family);
        assert_eq!(verify_code(&family), Some("provenance_inconsistent"));
    }

    #[test]
    fn verify_rejects_inconsistent_input_counts_and_source() {
        let table = fixture::table();
        assert!(table.verify().is_ok());
        let edits: [fn(&mut TableBody); 6] = [
            // Under-reporting the excluded adaptive rows.
            |body| body.inputs.excluded_adaptive -= 1,
            |body| body.inputs.used += 1,
            |body| body.inputs.qualified += 1,
            |body| body.source.programs += 1,
            |body| body.source.sha256 = String::from("abc"),
            |body| body.source.schema.clear(),
        ];
        for edit in edits {
            let mut edited = table.clone();
            edit(&mut edited.body);
            reseal(&mut edited);
            assert_eq!(verify_code(&edited), Some("provenance_inconsistent"));
        }
    }

    #[test]
    fn verify_recomputes_support_and_signature() {
        let table = fixture::table();

        let mut datasets = table.clone();
        datasets.body.motifs[0].support.datasets += 1;
        reseal(&mut datasets);
        assert_eq!(verify_code(&datasets), Some("support_mismatch"));

        let mut seeds = table.clone();
        seeds.body.motifs[0].support.every_host_is_a_known_seed ^= true;
        reseal(&mut seeds);
        assert_eq!(verify_code(&seeds), Some("support_mismatch"));

        let mut filtered = table.clone();
        filtered.body.parameters.min_datasets = 50;
        reseal(&mut filtered);
        assert_eq!(verify_code(&filtered), Some("support_mismatch"));

        let mut tags = table.clone();
        tags.body.motifs[0]
            .signature
            .tags
            .push(String::from("extra"));
        reseal(&mut tags);
        assert_eq!(verify_code(&tags), Some("signature_drift"));

        let mut observed = table.clone();
        observed.body.motifs[0].signature.observed_tasks.clear();
        reseal(&mut observed);
        assert_eq!(verify_code(&observed), Some("signature_drift"));

        // Order is recomputed as well.
        let mut swapped = table;
        assert!(swapped.body.motifs.len() >= 2);
        swapped.body.motifs.swap(0, 1);
        reseal(&mut swapped);
        assert_eq!(verify_code(&swapped), Some("table_order"));
    }

    #[test]
    fn promoted_entries_are_refused_in_this_build() {
        let mut table = fixture::table();
        table.body.motifs[0].lifecycle = Lifecycle::Promoted;
        reseal(&mut table);
        assert_eq!(verify_code(&table), Some("promotion_evidence_missing"));
    }

    #[test]
    fn engine_binding_mismatch_is_reported_as_drift() {
        let table = fixture::table();
        assert_eq!(table.body.engine, EngineBinding::current());
        let edits: [fn(&mut EngineBinding); 3] = [
            |engine| engine.registry_sha256 = "0".repeat(64),
            |engine| engine.emitter_version = String::from("0.0.0"),
            |engine| engine.bound_artifact_schema = String::from("battle-artifact-v0"),
        ];
        for edit in edits {
            let mut drifted = table.clone();
            edit(&mut drifted.body.engine);
            reseal(&mut drifted);
            assert_eq!(verify_code(&drifted), Some("engine_drift"));
        }
    }
}
