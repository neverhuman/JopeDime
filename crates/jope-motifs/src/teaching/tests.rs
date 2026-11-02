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
        |entry, _| entry.context.as_mut().expect("context").checkpoint_sha256 = Some("abc".into()),
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
    for (role, expected, held) in [("test", named, Some(&receipt)), ("development", None, None)] {
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
