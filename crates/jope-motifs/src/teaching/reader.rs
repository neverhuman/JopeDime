use super::*;

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
pub(crate) fn guard_admits(entry: &QualifiedEntry) -> bool {
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
