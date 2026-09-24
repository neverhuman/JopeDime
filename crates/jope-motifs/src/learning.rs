//! The learning State as an [`EligibleState`]: `jope_learning::eligibility::freeze_teaching`
//! and `State::pointers`, projected field for field. Neither takes a lock or writes: the
//! ledgers are opened with the pure `Ledger::open`, and `freeze_teaching` recomputes every
//! qualification from retained blobs only (`measured(.., live = false)`).
//!
//! Reference only: this module needs the eligibility stack (the training63 integration patch,
//! applied after PR 83), which is not on this branch. It is compiled only with the
//! `learning-state` feature, which is off by default; enable it once that stack has landed.
//! The shapes it reads are listed in [`crate::teaching`].

use crate::teaching::{
    BurnReceipt, ContextFields, EligibleState, GuardFields, PointerKeyFields, PointerRecord,
    QualifiedEntry, QualifiedRead,
};
use crate::{is_hex64, Code, Coded};
use anyhow::{ensure, Result};
use jope_learning::canon::canonical_sha256;
use jope_learning::eligibility::{freeze_teaching, Qualification};
use jope_learning::record::HwmRecord;
use jope_learning::state::{PointerKey, State};
use serde::Deserialize;
use std::path::Path;

/// A learning State, read only.
struct LearningState(State);

/// The tip of the State's HWM chain (lib.rs:10). Every qualified record needs an accepted
/// HWM ledger event (eligibility.rs:1085-1096, 1112-1118), and a chain with events has this
/// file, so a root without it holds nothing that could ever be admitted.
const CHAIN_TIP: &str = "hwm/ledger/HEAD.json";

/// Open the State at `state`, or at the product's resolved default (`State::resolve`,
/// state.rs:85-91). The root must be an existing directory holding the HWM chain tip. A
/// missing root, a file, or a directory that holds no State (a mistyped `--state`, say) is
/// `state_unavailable`, never an empty State.
pub(crate) fn open(state: Option<&Path>) -> Result<Box<dyn EligibleState>> {
    let resolved = State::resolve(state.map(Path::to_path_buf)).coded("state_unavailable")?;
    let root = resolved.root();
    ensure!(
        root.is_dir() && root.join(CHAIN_TIP).is_file(),
        Code("state_unavailable")
    );
    Ok(Box::new(LearningState(resolved)))
}

impl EligibleState for LearningState {
    fn qualified_read(&self) -> Result<QualifiedRead> {
        let manifest = freeze_teaching(&self.0)?;
        Ok(QualifiedRead {
            schema: manifest.schema.to_owned(),
            sha256: canonical_sha256(&manifest)?,
            excluded: u64::try_from(manifest.excluded_diagnostic)?,
            entries: manifest
                .entries
                .iter()
                .map(|qualification| entry(&self.0, qualification))
                .collect::<Result<Vec<_>>>()?,
        })
    }

    fn pointer_records(&self) -> Result<Vec<PointerRecord>> {
        Ok(self
            .0
            .pointers()?
            .iter()
            .map(|(key, record)| pointer(key, record))
            .collect())
    }
}

/// One qualification (eligibility.rs:125-150), with its comparison digest recomputed and its
/// burn receipt, if it names one, read back from the State.
fn entry(state: &State, qualification: &Qualification) -> Result<QualifiedEntry> {
    let identity = &qualification.comparison;
    let guard = &qualification.guard;
    Ok(QualifiedEntry {
        schema: qualification.schema.clone(),
        freeze_comparison_sha256: qualification.freeze.comparison_sha256.clone(),
        comparison_sha256: identity.sha256()?,
        task: identity.task.clone(),
        dataset_sha256: identity.dataset_hash.clone(),
        cut_sha256: identity.cut_id.clone(),
        evaluator_sha256: identity.evaluator_sha256.clone(),
        vocabulary_sha256: identity.vocabulary_sha256.clone(),
        protocol: identity.protocol.clone(),
        context: identity.comparison.as_ref().map(|context| ContextFields {
            numerical_policy: context.numerical_policy.clone(),
            stage: context.stage.clone(),
            aggregation: context.aggregation.clone(),
            label_role: context.label_role.clone(),
            recipe_encoding: context.recipe_encoding.clone(),
            sealed: context.sealed,
            benchmark_valid: context.benchmark_valid,
            checkpoint_sha256: context.checkpoint_sha256.clone(),
        }),
        record_sha256: qualification.record_sha256.clone(),
        program_sha256: qualification.program_sha256.clone(),
        guard: GuardFields {
            schema: guard.schema.clone(),
            folds: guard.folds,
            minimum_improved_folds: guard.minimum_improved_folds,
            aggregation: guard.aggregation.clone(),
            baseline: guard.baseline.clone(),
            absolute_gain: guard.absolute_gain,
            relative_gain: guard.relative_gain,
        },
        label_role: qualification.label_role.clone(),
        original_label_role: qualification.original_label_role.clone(),
        label_burn_sha256: qualification.label_burn_sha256.clone(),
        label_burn: qualification
            .label_burn_sha256
            .as_deref()
            .and_then(|sha256| burn_receipt(state, sha256)),
        mean_loss: qualification.candidate_loss,
        baseline_loss: qualification.baseline_loss,
        improved_folds: qualification.improved_folds,
        fold_losses: qualification.fold_losses.clone(),
        context_outer_fold: qualification.context_outer_fold,
        context_fitting_rows_sha256: qualification.context_fitting_rows_sha256.clone(),
    })
}

/// A label burn receipt exactly as the freeze writes it (eligibility.rs:506-510).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LabelBurn {
    schema: String,
    dataset_sha256: String,
    cut_id: String,
    panel_sha256: String,
    development_sha256: String,
    original_role: String,
    effective_role: String,
}

/// The burn receipt a qualification names, read back from the State's blobs
/// (`State::read_blob`, state.rs:198-206, which checks the blob's digest). `None` when the
/// digest is malformed or when the blob is absent, unreadable or not a receipt; the reader
/// then refuses the whole read with `label_burn_missing`. Nothing in the learning crate reads
/// the receipt again after the freeze, so a deleted receipt is caught only here.
fn burn_receipt(state: &State, sha256: &str) -> Option<BurnReceipt> {
    if !is_hex64(sha256) {
        return None;
    }
    let bytes = state.read_blob(sha256).ok().flatten()?;
    let receipt: LabelBurn = serde_json::from_slice(&bytes).ok()?;
    let digests = is_hex64(&receipt.panel_sha256) && is_hex64(&receipt.development_sha256);
    digests.then(|| BurnReceipt {
        schema: receipt.schema,
        dataset_sha256: receipt.dataset_sha256,
        cut_sha256: receipt.cut_id,
        original_role: receipt.original_role,
        effective_role: receipt.effective_role,
    })
}

/// One pointer and its record (state.rs:13-38; record.rs:153-260).
fn pointer(key: &PointerKey, record: &HwmRecord) -> PointerRecord {
    PointerRecord {
        key: PointerKeyFields {
            task: key.task.clone(),
            dataset_sha256: key.dataset_hash.clone(),
            cut_sha256: key.cut_id.clone(),
            identity_sha256: key.identity_sha256.clone(),
        },
        task: record.identity.task.clone(),
        dataset_sha256: record.identity.dataset_hash.clone(),
        cut_sha256: record.identity.cut_id.clone(),
        identity_sha256: record.identity_sha256.clone(),
        has_comparison: record.identity.comparison.is_some(),
        record_sha256: record.provenance.record_sha256.clone(),
        program: record.genome.dsl.clone(),
        program_sha256: record.genome.program_sha256.clone(),
        sealed: record.sealed,
        benchmark_valid: record.benchmark_valid,
        replayable: record.replayable,
        valid: record.validate().is_ok(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code_of;
    use crate::source::QualifiedSource;
    use crate::teaching::{StateTeaching, SET_SCHEMA};
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    /// Every directory and file under `root`, with each file's bytes.
    fn snapshot(root: &Path) -> BTreeMap<PathBuf, Option<Vec<u8>>> {
        let mut out = BTreeMap::new();
        let mut pending = vec![root.to_path_buf()];
        while let Some(directory) = pending.pop() {
            for entry in std::fs::read_dir(&directory).expect("listing") {
                let path = entry.expect("entry").path();
                if path.is_dir() {
                    pending.push(path.clone());
                    out.insert(path, None);
                } else {
                    let bytes = std::fs::read(&path).expect("read");
                    out.insert(path, Some(bytes));
                }
            }
        }
        out
    }

    #[test]
    fn learning_state_read_writes_nothing() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let root = directory.path().join("state");
        std::fs::create_dir(&root).expect("state root");
        std::fs::write(root.join("marker"), b"kept").expect("marker");
        let before = snapshot(directory.path());
        assert_eq!(before.len(), 2);

        // The reader itself, on a State with nothing in it: an empty set, and nothing written.
        let state = LearningState(State::at(&root));
        let read = state.qualified_read().expect("reads");
        assert_eq!(read.schema, SET_SCHEMA);
        assert!(read.entries.is_empty());
        assert_eq!(read.excluded, 0);
        assert!(state.pointer_records().expect("pointers").is_empty());
        let set = StateTeaching::new(&state)
            .qualified()
            .expect("an empty set");
        assert!(set.programs().is_empty());
        assert_eq!(snapshot(directory.path()), before);

        // The opener refuses that root: it has no HWM chain tip, so it holds no State.
        let error = open(Some(root.as_path())).err().expect("refused");
        assert_eq!(code_of(&error), Some("state_unavailable"));
        assert_eq!(snapshot(directory.path()), before);
    }

    #[test]
    fn a_root_that_holds_no_state_is_unavailable_never_empty() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let missing = directory.path().join("absent");
        let error = open(Some(missing.as_path())).err().expect("refused");
        assert_eq!(code_of(&error), Some("state_unavailable"));
        assert!(missing.symlink_metadata().is_err());

        let file = directory.path().join("file");
        std::fs::write(&file, b"x").expect("file");
        let error = open(Some(file.as_path())).err().expect("refused");
        assert_eq!(code_of(&error), Some("state_unavailable"));

        // An existing directory that is not a State, such as a mistyped `--state`.
        let unrelated = directory.path().join("home");
        std::fs::create_dir_all(unrelated.join("hwm")).expect("unrelated directory");
        std::fs::write(unrelated.join("notes.txt"), b"kept").expect("unrelated file");
        let before = snapshot(directory.path());
        let error = open(Some(unrelated.as_path())).err().expect("refused");
        assert_eq!(code_of(&error), Some("state_unavailable"));
        assert_eq!(snapshot(directory.path()), before);

        // A chain tip that is not a ledger opens, and then every read is refused: a broken
        // chain is an error, never an empty set. Nothing is written either way.
        let tip = unrelated.join(CHAIN_TIP);
        std::fs::create_dir_all(tip.parent().expect("ledger directory")).expect("ledger");
        std::fs::write(&tip, b"{}").expect("chain tip");
        let before = snapshot(directory.path());
        let state = open(Some(unrelated.as_path())).expect("opens");
        assert!(state.qualified_read().is_err());
        let error = StateTeaching::new(state.as_ref())
            .qualified()
            .err()
            .expect("refused");
        assert_eq!(code_of(&error), Some("state_unavailable"));
        assert_eq!(snapshot(directory.path()), before);
    }
}
