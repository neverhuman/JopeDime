//! The eligible-State reader: the qualified records of a learning State, joined to their
//! exact-identity pointers and admitted with exact checks.
//!
//! Authentication belongs to the learning crate. Its qualified-record reader
//! (`jope_learning::eligibility::freeze_teaching`) lists a record only after it has checked
//! both hash-chained ledgers, found an accepted ledger event for the record's exact comparison
//! identity, and recomputed the stored qualification byte for byte from retained,
//! content-addressed evidence: the preoptimization freeze and its ledger receipt, the five
//! fold predictions, the worker receipts, the train-only baseline and both scores. It takes no
//! lock and writes nothing. Every other pointer is excluded there and only counted.
//!
//! This module reaches that reader through [`EligibleState`], a projection of exactly the
//! fields it needs, so the join and the checks below are tested without a learning State. The
//! adapter for the learning State (`learning`) implements it field for field. It refuses with
//! `state_unavailable` a root that holds no State: a missing root, or one without the tip of
//! the HWM chain, which every qualified record needs. So a wrong root never looks like an
//! empty State.
//!
//! Every entry is then checked again here. Any failure refuses the whole read with a fixed
//! code, and nothing missing is ever read as eligible:
//!
//! - the set is read twice, with the pointers in between, and both reads must be equal bit
//!   for bit, so that a NaN read twice is the same read (`input_set_changed`); a reader error,
//!   an unknown set label, a malformed set digest or an exclusion count that does not fit is
//!   `state_unavailable`;
//! - the entry carries the measured-admission label and a complete comparison identity: a task
//!   the scorer measures; 64-hex evaluator, vocabulary, dataset and cut digests; a protocol;
//!   and a train-side context under the scorer's numerical policy, with development labels,
//!   neither sealed nor a benchmark score, naming a 64-hex checkpoint or none. It was measured
//!   under the known admission guard, its five fold losses recount and average to the entry's
//!   figures, its mean gain clears the guard's margin, and its inner context is complete or
//!   absent (`record_unqualified`);
//! - test-role labels need a burn receipt digest and the receipt itself, read back from the
//!   State: a `jope.label-burn/v1` receipt for the entry's dataset and, outside an inner
//!   partition, its cut, relabelling test as development (`label_burn_missing`). Development
//!   and validation labels carry neither, and every other role is refused
//!   (`label_origin_refused`);
//! - exactly one pointer holds the entry's comparison identity and record
//!   (`input_join_failed`);
//! - that pointer is an exact-identity pointer to a structurally valid, unsealed, replayable
//!   record with a comparison context; its identity, task, dataset, cut and program digest are
//!   the entry's, and the frozen comparison is the qualified one (`record_unauthenticated`);
//! - the program then passes the admission checks of [`QualifiedProgram`]: canonical text
//!   whose sha256 is the recorded program digest.
//!
//! Projected shapes, all in `crates/jope-learning/src` on `main` (f03b1eb85):
//! `eligibility.rs` lines 125-150 (`Qualification`), 159-165 (`TeachingManifest`), 168-201
//! (`GuardPolicy`), 287-323 (`task`: the comparison requirements, lines 292-322 re-checked in
//! full here), 506-518 (the label burn receipt) and 1117-1174 (`freeze_teaching`); `state.rs`
//! lines 13-20 (`PointerKey`), 198-206 (`State::read_blob`) and 317-368 (`State::pointers`);
//! `record.rs` lines 19-59 (`ComparisonContext`, `HwmIdentity`) and 153-260 (`HwmRecord` and
//! its validation); `lib.rs` (the State layout). The scorer's numerical policy is imported from
//! `jope_pipeline::evaluation_score::NUMERICAL_POLICY`. A later revision of those shapes
//! changes this projection and the adapter, nothing else.
//!
//! Where the projection differs from the source API, for whoever integrates it:
//!
//! - Renamed: `QualifiedEntry::mean_loss` is `Qualification::candidate_loss`;
//!   `QualifiedRead::excluded` (u64) is `TeachingManifest::excluded_diagnostic` (usize);
//!   `QualifiedRead::schema` is owned where the manifest's is `&'static str`; `dataset_sha256`
//!   and `cut_sha256` are `dataset_hash` and `cut_id` (record.rs:39,42; state.rs:15-16).
//! - Not carried: the manifest's `exclusions`, whose reasons can quote paths and identities,
//!   so only their count is kept; and the qualification's `freeze.freeze_sha256`,
//!   `targets_sha256`, `baseline_predictions_sha256`, `fold_predictions_sha256`,
//!   `measurement` and the context's `class_order`. `freeze_teaching` has already recomputed
//!   every one of them.
//! - Kept in step by hand: `GuardPolicy::current`, `improves` and `admits` are private to the
//!   learning crate, so [`GuardFields::current`] and the recount below restate them. Any change
//!   to the guard refuses every read (`record_unqualified`) until they are updated, and so does
//!   a change to the numerical policy.
//! - Stricter than the source: `freeze_teaching` has no join step. Here exactly one pointer
//!   must hold the entry's identity and record, so a legacy pointer (no identity in its key)
//!   holding the same identity and record as the exact pointer refuses the whole read with
//!   `input_join_failed`. The burn receipt blob is read back and checked, which nothing
//!   upstream does after the freeze. The adapter refuses a root without the HWM chain tip,
//!   where `State::resolve` accepts any path.

use crate::source::{
    Admission, LabelOrigin, QualifiedProgram, QualifiedSet, QualifiedSource, TaskLabel,
};
use crate::{is_hex64, Code, Coded};
use anyhow::{bail, ensure, Result};
use jope_pipeline::evaluation_score::NUMERICAL_POLICY;
use std::path::Path;

/// Label of the qualified record set (`TeachingManifest::schema`, eligibility.rs:1134).
pub const SET_SCHEMA: &str = "jope.qualified-teaching-freeze/v1";
/// Label of one measured admission (`Qualification::schema`, eligibility.rs:36).
pub const ENTRY_SCHEMA: &str = "jope.measured-teaching/v1";
/// Label of the admission guard (`GuardPolicy::current`, eligibility.rs:180-190).
pub const GUARD_SCHEMA: &str = "jope.teaching-guard/v1";
/// Label of a label burn receipt (eligibility.rs:506-510).
pub const BURN_SCHEMA: &str = "jope.label-burn/v1";

// The comparison requirements of the measured-admission scorer (eligibility.rs:292-301); the
// numerical policy is the scorer's own, imported above.
const STAGE: &str = "train_oof";
const AGGREGATION: &str = "mean_fold_loss";
const RECIPE_ENCODING: &str = "native-canonical-dsl-sha256/v1";
const LABEL_ROLE: &str = "development";
/// Folds of every measured admission (`GuardPolicy::folds`).
const FOLDS: usize = 5;

/// The comparison context of an entry (`ComparisonContext`, record.rs:19-30): the fields the
/// measured-admission scorer requires. `class_order` is not carried: the scorer checks it
/// against the frozen panel when the freeze is written (eligibility.rs:211-222).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContextFields {
    /// Numerical policy identifier used by the measured comparison.
    pub numerical_policy: String,
    /// Evaluation stage, expected to be the training out-of-fold stage.
    pub stage: String,
    /// Aggregation rule for the fold losses.
    pub aggregation: String,
    /// Label role attached to the comparison context.
    pub label_role: String,
    /// Encoding used to hash the recipe.
    pub recipe_encoding: String,
    /// Whether this context is sealed from optimization.
    pub sealed: bool,
    /// Whether the context was accepted by the benchmark-validity gate.
    pub benchmark_valid: bool,
    /// Proposal checkpoint lineage; `None` when no proposal checkpoint applies.
    pub checkpoint_sha256: Option<String>,
}

/// The admission guard an entry was measured under (`GuardPolicy`, eligibility.rs:170-190).
/// Equality is bit for bit on the margins (see `same_bits`).
#[derive(Clone, Debug)]
pub struct GuardFields {
    /// Guard schema identifier.
    pub schema: String,
    /// Number of folds used by the guard.
    pub folds: usize,
    /// Minimum folds that must improve over the baseline.
    pub minimum_improved_folds: usize,
    /// Rule used to aggregate fold losses.
    pub aggregation: String,
    /// Train-only baseline definition.
    pub baseline: String,
    /// Absolute minimum loss improvement.
    pub absolute_gain: f64,
    /// Relative minimum loss improvement.
    pub relative_gain: f64,
}

impl GuardFields {
    /// The only guard the learning crate admits under (`GuardPolicy::current`).
    pub fn current() -> Self {
        Self {
            schema: GUARD_SCHEMA.to_owned(),
            folds: FOLDS,
            minimum_improved_folds: 4,
            aggregation: AGGREGATION.to_owned(),
            baseline: String::from("training_mean_or_class_frequency"),
            absolute_gain: 1e-12,
            relative_gain: 1e-6,
        }
    }

    /// `GuardPolicy::improves` (eligibility.rs:191-195): both losses finite and the gain above
    /// the absolute plus relative margin.
    fn improves(&self, loss: f64, baseline: f64) -> bool {
        loss.is_finite()
            && baseline.is_finite()
            && baseline - loss > self.absolute_gain + self.relative_gain * baseline.abs()
    }
}

/// Bit-for-bit equality of two losses or margins: a NaN equals itself and `0.0` differs from
/// `-0.0`. Two reads of the same bytes are therefore equal whatever they hold, so a changed
/// read is always `input_set_changed` and an inadmissible value is always refused by the
/// per-entry checks, never mistaken for a change between reads.
fn same_bits(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits()
}

impl PartialEq for GuardFields {
    fn eq(&self, other: &Self) -> bool {
        // Destructured without `..`, so a new field cannot be left out of the comparison.
        let Self {
            schema,
            folds,
            minimum_improved_folds,
            aggregation,
            baseline,
            absolute_gain,
            relative_gain,
        } = self;
        *schema == other.schema
            && *folds == other.folds
            && *minimum_improved_folds == other.minimum_improved_folds
            && *aggregation == other.aggregation
            && *baseline == other.baseline
            && same_bits(*absolute_gain, other.absolute_gain)
            && same_bits(*relative_gain, other.relative_gain)
    }
}

impl Eq for GuardFields {}

/// The label burn receipt a test-origin entry names, as the adapter read it back from the
/// State's content-addressed blobs (`jope.label-burn/v1`, written by the freeze,
/// eligibility.rs:506-518). Nothing in the learning crate reads the receipt again after the
/// freeze, so its presence is established only here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BurnReceipt {
    /// Label-burn receipt schema identifier.
    pub schema: String,
    /// Digest of the dataset whose labels were burned.
    pub dataset_sha256: String,
    /// `cut_id`: the split whose test labels were burned.
    pub cut_sha256: String,
    /// Label role before the burn.
    pub original_role: String,
    /// Label role after the burn.
    pub effective_role: String,
}

/// One entry of the qualified set (`Qualification`, eligibility.rs:125-150). It carries no
/// program text: that is only in the pointer's record. Equality is bit for bit on every loss
/// (see `same_bits`).
#[derive(Clone, Debug)]
pub struct QualifiedEntry {
    /// Measured teaching entry schema identifier.
    pub schema: String,
    /// `freeze.comparison_sha256`: the comparison the preoptimization freeze was pinned to.
    pub freeze_comparison_sha256: String,
    /// `comparison.sha256()`, recomputed by the adapter from the entry's comparison identity.
    pub comparison_sha256: String,
    /// `comparison.task`, `comparison.dataset_hash` and `comparison.cut_id`.
    pub task: String,
    /// Digest of the dataset used by the comparison.
    pub dataset_sha256: String,
    /// Digest identifying the evaluated cut.
    pub cut_sha256: String,
    /// `comparison.evaluator_sha256`, `comparison.vocabulary_sha256` and `comparison.protocol`.
    pub evaluator_sha256: String,
    /// Digest of the vocabulary used by the measured comparison.
    pub vocabulary_sha256: String,
    /// Protocol identifier used by the measured comparison.
    pub protocol: String,
    /// `comparison.comparison`; `None` for a legacy identity, which is never eligible.
    pub context: Option<ContextFields>,
    /// Digest of the qualified record.
    pub record_sha256: String,
    /// Digest of the program text in the record.
    pub program_sha256: String,
    /// Exact guard contract used for qualification.
    pub guard: GuardFields,
    /// Effective label role used by the comparison.
    pub label_role: String,
    /// Original label role before any burn.
    pub original_label_role: String,
    /// Digest of the referenced label-burn receipt, when present.
    pub label_burn_sha256: Option<String>,
    /// The receipt `label_burn_sha256` names, read back from the State; `None` when there is no
    /// digest or when its blob is missing, unreadable or not a receipt.
    pub label_burn: Option<BurnReceipt>,
    /// Mean fold loss of the recorded program.
    pub mean_loss: f64,
    /// Mean fold loss of the train-only baseline.
    pub baseline_loss: f64,
    /// Number of folds where the recorded program improved over the baseline.
    pub improved_folds: usize,
    /// `(program loss, baseline loss)` per fold, in fold order.
    pub fold_losses: Vec<(f64, f64)>,
    /// Outer context fold, when the comparison used a context fold.
    pub context_outer_fold: Option<u8>,
    /// Digest of rows used to fit the context, when recorded.
    pub context_fitting_rows_sha256: Option<String>,
}

impl PartialEq for QualifiedEntry {
    fn eq(&self, other: &Self) -> bool {
        // Destructured without `..`, so a new field cannot be left out of the comparison.
        let Self {
            schema,
            freeze_comparison_sha256,
            comparison_sha256,
            task,
            dataset_sha256,
            cut_sha256,
            evaluator_sha256,
            vocabulary_sha256,
            protocol,
            context,
            record_sha256,
            program_sha256,
            guard,
            label_role,
            original_label_role,
            label_burn_sha256,
            label_burn,
            mean_loss,
            baseline_loss,
            improved_folds,
            fold_losses,
            context_outer_fold,
            context_fitting_rows_sha256,
        } = self;
        *schema == other.schema
            && *freeze_comparison_sha256 == other.freeze_comparison_sha256
            && *comparison_sha256 == other.comparison_sha256
            && *task == other.task
            && *dataset_sha256 == other.dataset_sha256
            && *cut_sha256 == other.cut_sha256
            && *evaluator_sha256 == other.evaluator_sha256
            && *vocabulary_sha256 == other.vocabulary_sha256
            && *protocol == other.protocol
            && *context == other.context
            && *record_sha256 == other.record_sha256
            && *program_sha256 == other.program_sha256
            && *guard == other.guard
            && *label_role == other.label_role
            && *original_label_role == other.original_label_role
            && *label_burn_sha256 == other.label_burn_sha256
            && *label_burn == other.label_burn
            && same_bits(*mean_loss, other.mean_loss)
            && same_bits(*baseline_loss, other.baseline_loss)
            && *improved_folds == other.improved_folds
            && fold_losses.len() == other.fold_losses.len()
            && fold_losses
                .iter()
                .zip(&other.fold_losses)
                .all(|(a, b)| same_bits(a.0, b.0) && same_bits(a.1, b.1))
            && *context_outer_fold == other.context_outer_fold
            && *context_fitting_rows_sha256 == other.context_fitting_rows_sha256
    }
}

impl Eq for QualifiedEntry {}

/// One read of the qualified set (`TeachingManifest`, eligibility.rs:159-165) with the sha256
/// of its canonical JSON (`canon::canonical_sha256`). Exclusion reasons are not carried: they
/// can quote paths and identities, so only their count is. Two reads are compared bit for
/// bit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QualifiedRead {
    /// Qualified-set schema identifier.
    pub schema: String,
    /// Digest of the canonical qualified-set JSON.
    pub sha256: String,
    /// `excluded_diagnostic`.
    pub excluded: u64,
    /// Qualified entries in the source set.
    pub entries: Vec<QualifiedEntry>,
}

/// A pointer key (`PointerKey`, state.rs:13-20).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PointerKeyFields {
    /// Task label from the State pointer key.
    pub task: String,
    /// Dataset digest from the State pointer key.
    pub dataset_sha256: String,
    /// Cut digest from the State pointer key.
    pub cut_sha256: String,
    /// `None` for a legacy pointer, which is diagnostic only.
    pub identity_sha256: Option<String>,
}

/// One pointer of the State and the fields of its record (`State::pointers`, state.rs:317-368;
/// `HwmRecord`, record.rs:153-182).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PointerRecord {
    /// Key used to address this pointer in the State.
    pub key: PointerKeyFields,
    /// `identity.task`, `identity.dataset_hash` and `identity.cut_id`.
    pub task: String,
    /// Dataset digest from the record identity.
    pub dataset_sha256: String,
    /// Cut digest from the record identity.
    pub cut_sha256: String,
    /// Digest of the record identity.
    pub identity_sha256: String,
    /// `identity.comparison.is_some()`.
    pub has_comparison: bool,
    /// `provenance.record_sha256`.
    pub record_sha256: String,
    /// The program text of the record.
    pub program: String,
    /// The recorded sha256 of the program text.
    pub program_sha256: String,
    /// Whether the record was sealed.
    pub sealed: bool,
    /// Whether the record passed benchmark validity checks.
    pub benchmark_valid: bool,
    /// Whether the record supports deterministic replay.
    pub replayable: bool,
    /// `HwmRecord::validate` passed (record.rs:186-260): among other checks, the identity
    /// digest hashes the identity and the program digest hashes the program text.
    pub valid: bool,
}

/// A readable learning State. Implementations read only: no lock, no write.
pub trait EligibleState {
    /// One read of the qualified record set (`freeze_teaching`). A broken ledger or any read
    /// failure is an error, never an empty set.
    fn qualified_read(&self) -> Result<QualifiedRead>;
    /// Every pointer of the State with its record (`State::pointers`), legacy ones included.
    fn pointer_records(&self) -> Result<Vec<PointerRecord>>;
}

/// Opens the State at an explicit root, or at the product's resolved default.
pub(crate) type StateOpener = Box<dyn Fn(Option<&Path>) -> Result<Box<dyn EligibleState>>>;

/// An opener with no reader behind it, for tests: every open is refused with
/// `state_unavailable`, never read as an empty State.
#[cfg(test)]
pub(crate) fn no_reader(_state: Option<&Path>) -> Result<Box<dyn EligibleState>> {
    bail!(Code("state_unavailable"))
}

/// The qualified programs of one State.
pub struct StateTeaching<'a> {
    state: &'a dyn EligibleState,
}

impl<'a> StateTeaching<'a> {
    /// Create a read-only adapter for reading qualified training records.
    pub fn new(state: &'a dyn EligibleState) -> Self {
        Self { state }
    }
}

impl QualifiedSource for StateTeaching<'_> {
    fn qualified(&self) -> Result<QualifiedSet> {
        let first = self.state.qualified_read().coded("state_unavailable")?;
        let pointers = self.state.pointer_records().coded("state_unavailable")?;
        let second = self.state.qualified_read().coded("state_unavailable")?;
        ensure!(first == second, Code("input_set_changed"));
        ensure!(
            first.schema == SET_SCHEMA && is_hex64(&first.sha256),
            Code("state_unavailable")
        );
        let excluded = u32::try_from(first.excluded).map_err(|_| Code("state_unavailable"))?;
        let mut programs = Vec::with_capacity(first.entries.len());
        for entry in &first.entries {
            require_qualified(entry)?;
            let label = label_origin(entry)?;
            let pointer = joined(entry, &pointers)?;
            require_authenticated(entry, pointer)?;
            programs.push(QualifiedProgram::admit(Admission {
                program: &pointer.program,
                recipe_sha256: &entry.program_sha256,
                task: &entry.task,
                dataset_sha256: &entry.dataset_sha256,
                cut_sha256: &entry.cut_sha256,
                comparison_sha256: &entry.comparison_sha256,
                record_sha256: &entry.record_sha256,
                label,
            })?);
        }
        QualifiedSet::new(programs, &first.schema, &first.sha256, excluded)
    }
}

/// The entry is a measured admission of a complete comparison identity under the known guard
/// (`record_unqualified`). The identity checks are those of the scorer's `task`
/// (eligibility.rs:287-323): the context (292-301), identity completeness (302-309), the
/// checkpoint (310-316) and a task the scorer measures (317-322).
fn require_qualified(entry: &QualifiedEntry) -> Result<()> {
    let refused = Code("record_unqualified");
    ensure!(entry.schema == ENTRY_SCHEMA, refused);
    let context = entry.context.as_ref().ok_or(refused)?;
    ensure!(
        context.numerical_policy == NUMERICAL_POLICY
            && context.stage == STAGE
            && context.aggregation == AGGREGATION
            && context.label_role == LABEL_ROLE
            && context.recipe_encoding == RECIPE_ENCODING
            && !context.sealed
            && !context.benchmark_valid,
        refused
    );
    ensure!(
        is_hex64(&entry.evaluator_sha256)
            && is_hex64(&entry.vocabulary_sha256)
            && is_hex64(&entry.dataset_sha256)
            && is_hex64(&entry.cut_sha256)
            && !entry.protocol.trim().is_empty(),
        refused
    );
    ensure!(
        context.checkpoint_sha256.as_deref().is_none_or(is_hex64),
        refused
    );
    ensure!(TaskLabel::parse(&entry.task).is_ok(), refused);
    ensure!(entry.label_role == LABEL_ROLE, refused);
    ensure!(entry.guard == GuardFields::current(), refused);
    ensure!(guard_admits(entry), refused);
    let inner = match (
        entry.context_outer_fold,
        entry.context_fitting_rows_sha256.as_deref(),
    ) {
        (None, None) => true,
        (Some(fold), Some(rows)) => usize::from(fold) < FOLDS && is_hex64(rows),
        _ => false,
    };
    ensure!(inner, refused);
    Ok(())
}

/// `GuardPolicy::admits` (eligibility.rs:196-200), recomputed from the fold losses: five
/// folds, the improved-fold count recounted and at least the guard's minimum, both means
/// recomputed bit for bit (the scorer divides the fold sum by 5, eligibility.rs:955-964), and
/// the mean gain above the margin.
fn guard_admits(entry: &QualifiedEntry) -> bool {
    let guard = &entry.guard;
    let losses = &entry.fold_losses;
    let improved = losses
        .iter()
        .filter(|(loss, baseline)| guard.improves(*loss, *baseline))
        .count();
    let mean = |pick: fn(&(f64, f64)) -> f64| losses.iter().map(pick).sum::<f64>() / 5.0;
    losses.len() == guard.folds
        && improved == entry.improved_folds
        && improved >= guard.minimum_improved_folds
        && mean(|pair| pair.0).to_bits() == entry.mean_loss.to_bits()
        && mean(|pair| pair.1).to_bits() == entry.baseline_loss.to_bits()
        && guard.improves(entry.mean_loss, entry.baseline_loss)
}

/// Where the labels came from (`effective_label_role`, eligibility.rs:277-285, and the burn
/// receipt, eligibility.rs:506-518): optimization-visible test labels were relabelled
/// development and must name their burn receipt, which must be in the State and burn this
/// entry's labels. Development and validation labels name none.
fn label_origin(entry: &QualifiedEntry) -> Result<LabelOrigin> {
    let burn = entry.label_burn_sha256.as_deref();
    match entry.original_label_role.as_str() {
        "test" => {
            let missing = Code("label_burn_missing");
            ensure!(burn.is_some_and(is_hex64), missing);
            let receipt = entry.label_burn.as_ref().ok_or(missing)?;
            ensure!(burns_entry_labels(entry, receipt), missing);
        }
        "development" | "validation" => ensure!(
            burn.is_none() && entry.label_burn.is_none(),
            Code("label_origin_refused")
        ),
        _ => bail!(Code("label_origin_refused")),
    }
    Ok(LabelOrigin {
        original_role: entry.original_label_role.clone(),
        burn_sha256: entry.label_burn_sha256.clone(),
    })
}

/// The receipt burns this entry's labels: the test labels of its dataset, relabelled
/// development. Outside an inner partition the receipt names the entry's cut. Inside one, the
/// entry's cut is the partition's own identity, while the receipt names the outer split the
/// partition was derived from (eligibility.rs:443-446 and 506-513; inner.rs:106-131). The
/// entry does not carry that split, so there the receipt's cut must only be a digest.
fn burns_entry_labels(entry: &QualifiedEntry, receipt: &BurnReceipt) -> bool {
    let cut = match entry.context_outer_fold {
        None => receipt.cut_sha256 == entry.cut_sha256,
        Some(_) => is_hex64(&receipt.cut_sha256),
    };
    receipt.schema == BURN_SCHEMA
        && receipt.dataset_sha256 == entry.dataset_sha256
        && cut
        && receipt.original_role == "test"
        && receipt.effective_role == LABEL_ROLE
}

/// The one pointer that holds the entry's comparison identity and record
/// (`input_join_failed`).
fn joined<'p>(entry: &QualifiedEntry, pointers: &'p [PointerRecord]) -> Result<&'p PointerRecord> {
    let mut holders = pointers.iter().filter(|pointer| {
        pointer.identity_sha256 == entry.comparison_sha256
            && pointer.record_sha256 == entry.record_sha256
    });
    let pointer = holders.next().ok_or(Code("input_join_failed"))?;
    ensure!(holders.next().is_none(), Code("input_join_failed"));
    Ok(pointer)
}

/// The joined pointer is the exact, current, valid record the entry qualified
/// (`record_unauthenticated`).
fn require_authenticated(entry: &QualifiedEntry, pointer: &PointerRecord) -> Result<()> {
    let refused = Code("record_unauthenticated");
    let key = &pointer.key;
    let exact = key.task == pointer.task
        && key.dataset_sha256 == pointer.dataset_sha256
        && key.cut_sha256 == pointer.cut_sha256
        && key.identity_sha256.as_deref() == Some(pointer.identity_sha256.as_str());
    ensure!(pointer.valid && exact && pointer.has_comparison, refused);
    ensure!(
        !pointer.sealed && !pointer.benchmark_valid && pointer.replayable,
        refused
    );
    ensure!(
        is_hex64(&entry.comparison_sha256)
            && entry.freeze_comparison_sha256 == entry.comparison_sha256,
        refused
    );
    ensure!(
        entry.task == pointer.task
            && entry.dataset_sha256 == pointer.dataset_sha256
            && entry.cut_sha256 == pointer.cut_sha256,
        refused
    );
    ensure!(entry.program_sha256 == pointer.program_sha256, refused);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code_of;
    use crate::fixture::{self, FakeState};
    use crate::mine::mine;
    use crate::source::TaskLabel;
    use crate::table::MineParameters;

    fn read_code(state: &FakeState) -> Option<&'static str> {
        StateTeaching::new(state)
            .qualified()
            .err()
            .and_then(|error| code_of(&error))
    }

    /// A State of the small fixture lines, with one entry and its pointer edited.
    fn edited(edit: impl Fn(&mut QualifiedEntry, &mut PointerRecord)) -> FakeState {
        let mut state = FakeState::of(&fixture::small_lines());
        edit(&mut state.read.entries[0], &mut state.pointers[0]);
        state
    }

    #[test]
    fn read_admits_every_joined_entry_with_its_provenance() {
        let lines = fixture::lines();
        let state = FakeState::of(&lines);
        let set = StateTeaching::new(&state).qualified().expect("reads");
        assert_eq!(state.reads(), 2);
        assert_eq!(set.source_schema(), SET_SCHEMA);
        assert_eq!(set.source_sha256(), state.read.sha256);
        assert_eq!(set.excluded_at_source(), 0);
        assert_eq!(set.programs().len(), lines.len());
        for (read, line) in set.programs().iter().zip(&lines) {
            assert_eq!(read.program(), line.program());
            assert_eq!(read.recipe_sha256(), line.recipe_sha256());
            assert_eq!(read.task(), line.task());
            assert_eq!(read.dataset_sha256(), line.dataset_sha256());
            assert_eq!(read.cut_sha256(), line.cut_sha256());
            assert_eq!(read.comparison_sha256(), line.comparison_sha256());
            assert_eq!(read.record_sha256(), line.record_sha256());
            assert_eq!(read.label(), line.label());
        }
        // The programs read from a State mine exactly as the admitted lines do.
        let table = mine(&set, None, MineParameters::default()).expect("mines");
        assert_eq!(table.body.motifs, fixture::table().body.motifs);
        assert_eq!(table.body.source.schema, SET_SCHEMA);
    }

    #[test]
    fn empty_state_yields_an_empty_set_and_carries_its_exclusions() {
        let mut state = FakeState::of(&[]);
        state.read.excluded = 3;
        let set = StateTeaching::new(&state).qualified().expect("reads");
        assert!(set.programs().is_empty());
        assert_eq!(set.excluded_at_source(), 3);

        state.read.excluded = u64::from(u32::MAX) + 1;
        assert_eq!(read_code(&state), Some("state_unavailable"));
    }

    #[test]
    fn reader_failure_unknown_set_or_missing_reader_is_state_unavailable() {
        let mut broken = FakeState::of(&fixture::small_lines());
        broken.broken = true;
        assert_eq!(read_code(&broken), Some("state_unavailable"));

        let mut foreign = FakeState::of(&fixture::small_lines());
        foreign.read.schema = String::from("jope.qualified-teaching-freeze/v2");
        assert_eq!(read_code(&foreign), Some("state_unavailable"));

        let mut digest = FakeState::of(&fixture::small_lines());
        digest.read.sha256 = String::from("abc");
        assert_eq!(read_code(&digest), Some("state_unavailable"));

        let error = no_reader(None).err().expect("no reader is linked");
        assert_eq!(code_of(&error), Some("state_unavailable"));
        let error = no_reader(Some(Path::new("/state"))).err().expect("refused");
        assert_eq!(code_of(&error), Some("state_unavailable"));
    }

    #[test]
    fn second_read_change_is_input_set_changed() {
        let mut state = FakeState::of(&fixture::small_lines());
        let mut later = state.read.clone();
        later.entries.pop();
        state.later = Some(later);
        assert_eq!(read_code(&state), Some("input_set_changed"));
        assert_eq!(state.reads(), 2);

        let mut state = FakeState::of(&fixture::small_lines());
        let mut later = state.read.clone();
        later.sha256 = fixture::digest("state-read/changed");
        state.later = Some(later);
        assert_eq!(read_code(&state), Some("input_set_changed"));

        // Reads are compared bit for bit. A NaN read twice is the same read, and the entry is
        // then refused on its own terms, never as a change between reads.
        let mut nan = FakeState::of(&fixture::small_lines());
        nan.read.entries[0].fold_losses[0] = (f64::NAN, 1.0);
        nan.read.entries[1].guard.relative_gain = f64::NAN;
        assert_eq!(nan.read, nan.read.clone());
        assert_eq!(read_code(&nan), Some("record_unqualified"));
        assert_eq!(nan.reads(), 2);
        // A loss whose sign bit alone changed is a changed read, although 0.0 == -0.0.
        let mut signed = FakeState::of(&fixture::small_lines());
        signed.read.entries[0].fold_losses[0].0 = 0.0;
        let mut later = signed.read.clone();
        later.entries[0].fold_losses[0].0 = -0.0;
        assert_ne!(signed.read, later);
        signed.later = Some(later);
        assert_eq!(read_code(&signed), Some("input_set_changed"));
    }

    #[test]
    fn entry_join_rejects_identity_program_or_record_mismatch() {
        // No pointer holds the pair, or two do.
        let mut missing = FakeState::of(&fixture::small_lines());
        missing.pointers.remove(0);
        assert_eq!(read_code(&missing), Some("input_join_failed"));
        let mut doubled = FakeState::of(&fixture::small_lines());
        let mut legacy = doubled.pointers[0].clone();
        legacy.key.identity_sha256 = None;
        doubled.pointers.push(legacy);
        assert_eq!(read_code(&doubled), Some("input_join_failed"));
        let record = edited(|entry, _| entry.record_sha256 = fixture::digest("record/other"));
        assert_eq!(read_code(&record), Some("input_join_failed"));

        // The one holder is not the record the entry qualified.
        let cases: [fn(&mut QualifiedEntry, &mut PointerRecord); 8] = [
            |_, pointer| pointer.key.identity_sha256 = None,
            |_, pointer| pointer.key.cut_sha256 = fixture::digest("cut/other"),
            |_, pointer| pointer.valid = false,
            |_, pointer| pointer.has_comparison = false,
            |_, pointer| pointer.program_sha256 = fixture::digest("program/other"),
            |entry, _| entry.freeze_comparison_sha256 = fixture::digest("comparison/other"),
            |entry, _| entry.task = String::from("binary"),
            |entry, _| entry.dataset_sha256 = fixture::dataset("d9"),
        ];
        for (index, case) in cases.into_iter().enumerate() {
            assert_eq!(
                read_code(&edited(case)),
                Some("record_unauthenticated"),
                "case {index}"
            );
        }

        // The pointer's text does not hash to the recorded program digest.
        let text = edited(|_, pointer| {
            pointer.program = fixture::v3("(standardize) (rank)", "elasticnet");
        });
        assert_eq!(read_code(&text), Some("recipe_hash_mismatch"));
    }

    #[test]
    fn sealed_benchmark_or_unreplayable_pointer_is_refused() {
        let cases: [fn(&mut QualifiedEntry, &mut PointerRecord); 3] = [
            |_, pointer| pointer.sealed = true,
            |_, pointer| pointer.benchmark_valid = true,
            |_, pointer| pointer.replayable = false,
        ];
        for (index, case) in cases.into_iter().enumerate() {
            assert_eq!(
                read_code(&edited(case)),
                Some("record_unauthenticated"),
                "case {index}"
            );
        }
    }

    #[test]
    fn unqualified_entries_are_refused() {
        let cases: [fn(&mut QualifiedEntry, &mut PointerRecord); 25] = [
            |entry, _| entry.schema = String::from("jope.measured-teaching/v0"),
            |entry, _| entry.context = None,
            |entry, _| entry.context.as_mut().expect("context").sealed = true,
            |entry, _| entry.context.as_mut().expect("context").benchmark_valid = true,
            |entry, _| entry.context.as_mut().expect("context").stage = String::from("test"),
            |entry, _| entry.context.as_mut().expect("context").label_role = String::from("test"),
            |entry, _| entry.label_role = String::from("train"),
            |entry, _| entry.guard.minimum_improved_folds = 3,
            |entry, _| entry.guard.relative_gain = 0.0,
            // Three of five folds improve: the guard needs four.
            |entry, _| {
                entry.fold_losses[0] = (1.0, 1.0);
                entry.fold_losses[1] = (1.0, 1.0);
                entry.improved_folds = 3;
                entry.mean_loss = entry.fold_losses.iter().map(|pair| pair.0).sum::<f64>() / 5.0;
            },
            |entry, _| entry.improved_folds = 4,
            |entry, _| entry.mean_loss = 0.25,
            |entry, _| entry.baseline_loss = 2.0,
            |entry, _| {
                entry.fold_losses.pop();
            },
            |entry, _| entry.fold_losses[0] = (f64::NAN, 1.0),
            |entry, _| entry.context_outer_fold = Some(0),
            |entry, _| {
                entry.context_outer_fold = Some(5);
                entry.context_fitting_rows_sha256 = Some(fixture::digest("rows"));
            },
            // The rest of the scorer's comparison requirements (eligibility.rs:292-322).
            |entry, _| {
                entry.context.as_mut().expect("context").numerical_policy =
                    String::from("asm-panel-f32-v1")
            },
            |entry, _| {
                entry.context.as_mut().expect("context").checkpoint_sha256 = Some("abc".into())
            },
            |entry, _| entry.evaluator_sha256 = String::from("unbound"),
            |entry, _| entry.vocabulary_sha256 = String::from("unbound"),
            |entry, _| entry.protocol = String::from("  "),
            |entry, _| entry.task = String::from("timeseries"),
            |entry, _| entry.dataset_sha256 = String::from("abc"),
            |entry, _| entry.cut_sha256 = entry.cut_sha256.to_uppercase(),
        ];
        for (index, case) in cases.into_iter().enumerate() {
            assert_eq!(
                read_code(&edited(case)),
                Some("record_unqualified"),
                "case {index}"
            );
        }
        // A complete inner context is admitted, and so is a named checkpoint.
        let inner = edited(|entry, _| {
            entry.context_outer_fold = Some(4);
            entry.context_fitting_rows_sha256 = Some(fixture::digest("rows"));
        });
        assert!(StateTeaching::new(&inner).qualified().is_ok());
        let checkpoint = edited(|entry, _| {
            entry.context.as_mut().expect("context").checkpoint_sha256 =
                Some(fixture::digest("checkpoint"));
        });
        assert!(StateTeaching::new(&checkpoint).qualified().is_ok());
    }

    #[test]
    fn test_labels_require_a_burn_receipt_the_state_holds() {
        let lines = fixture::small_lines();
        let burn = fixture::digest("burn/state");
        let named = Some(burn.as_str());
        let receipt = fixture::burn_receipt(&lines[0]);
        let label = |role: &str, burn: Option<&str>, receipt: Option<&BurnReceipt>| {
            let mut state = FakeState::of(&lines);
            let entry = &mut state.read.entries[0];
            entry.original_label_role = role.to_owned();
            entry.label_burn_sha256 = burn.map(str::to_owned);
            entry.label_burn = receipt.cloned();
            state
        };
        assert_eq!(
            read_code(&label("test", None, None)),
            Some("label_burn_missing")
        );
        assert_eq!(
            read_code(&label("test", Some("not-a-digest"), Some(&receipt))),
            Some("label_burn_missing")
        );
        // The digest is recorded, but its blob was deleted from the State (or is unreadable,
        // or is not a receipt): the adapter reads no receipt back.
        assert_eq!(
            read_code(&label("test", named, None)),
            Some("label_burn_missing")
        );
        // A receipt that burns other labels, or none.
        let others: [fn(&mut BurnReceipt); 6] = [
            |other| other.schema = String::from("jope.label-burn/v0"),
            |other| other.dataset_sha256 = fixture::dataset("d9"),
            |other| other.cut_sha256 = fixture::digest("cut/other"),
            |other| other.cut_sha256 = String::from("abc"),
            |other| other.original_role = String::from("validation"),
            |other| other.effective_role = String::from("test"),
        ];
        for (index, edit) in others.into_iter().enumerate() {
            let mut other = receipt.clone();
            edit(&mut other);
            assert_eq!(
                read_code(&label("test", named, Some(&other))),
                Some("label_burn_missing"),
                "case {index}"
            );
        }
        for role in ["validation", "development"] {
            for (burn, receipt) in [
                (named, None),
                (named, Some(&receipt)),
                (None, Some(&receipt)),
            ] {
                assert_eq!(
                    read_code(&label(role, burn, receipt)),
                    Some("label_origin_refused"),
                    "{role}"
                );
            }
        }
        for role in ["sealed", "meta_test", "train", ""] {
            assert_eq!(
                read_code(&label(role, None, None)),
                Some("label_origin_refused"),
                "{role}"
            );
        }
        for (role, expected, held) in [("test", named, Some(&receipt)), ("development", None, None)]
        {
            let set = StateTeaching::new(&label(role, expected, held))
                .qualified()
                .expect("admitted");
            let origin = set.programs()[0].label();
            assert_eq!(origin.original_role, role);
            assert_eq!(origin.burn_sha256.as_deref(), expected);
        }

        // Inside an inner partition the receipt names the outer split, which the entry does
        // not carry: any digest is accepted there, a malformed one is not.
        let mut outer = receipt.clone();
        outer.cut_sha256 = fixture::digest("cut/outer");
        let mut inner = label("test", named, Some(&outer));
        inner.read.entries[0].context_outer_fold = Some(2);
        inner.read.entries[0].context_fitting_rows_sha256 = Some(fixture::digest("rows"));
        assert!(StateTeaching::new(&inner).qualified().is_ok());
        let mut malformed = inner.clone();
        if let Some(held) = malformed.read.entries[0].label_burn.as_mut() {
            held.cut_sha256 = String::from("abc");
        }
        assert_eq!(read_code(&malformed), Some("label_burn_missing"));
    }

    #[test]
    fn adaptive_label_provenance_is_carried_into_programs() {
        let small = fixture::v3(fixture::SMALL, "elasticnet");
        let lines = vec![
            fixture::qualified_with(
                &small,
                TaskLabel::Regression,
                "d1",
                "a01",
                fixture::development_label(),
            ),
            fixture::qualified_with(
                &small,
                TaskLabel::Regression,
                "d2",
                "a02",
                fixture::adaptive_label(),
            ),
        ];
        let state = FakeState::of(&lines);
        let set = StateTeaching::new(&state).qualified().expect("reads");
        assert_eq!(set.programs()[0].label(), &fixture::development_label());
        assert!(!set.programs()[0].label().is_adaptive());
        assert_eq!(set.programs()[1].label(), &fixture::adaptive_label());
        assert!(set.programs()[1].label().is_adaptive());
        // Burned test labels are excluded from mining by default and counted.
        let table = mine(&set, None, MineParameters::default()).expect("mines");
        assert_eq!(table.body.inputs.excluded_adaptive, 1);
        assert!(table.body.motifs.is_empty());
    }

    #[test]
    fn guard_mirrors_the_learning_policy() {
        let guard = GuardFields::current();
        assert_eq!(guard.schema, GUARD_SCHEMA);
        assert_eq!(guard.folds, 5);
        assert_eq!(guard.minimum_improved_folds, 4);
        assert!(guard.improves(0.5, 1.0));
        assert!(!guard.improves(1.0, 1.0));
        assert!(!guard.improves(1.0 - 1e-9, 1.0));
        assert!(!guard.improves(f64::INFINITY, 1.0));
        let (entry, _) = fixture::state_entry(&fixture::small_lines()[0]);
        assert!(guard_admits(&entry));
    }
}
