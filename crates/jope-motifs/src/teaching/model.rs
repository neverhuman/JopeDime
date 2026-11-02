use super::*;

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
    pub(super) fn improves(&self, loss: f64, baseline: f64) -> bool {
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
