//! Export only the actual committed checkpoint's complete private outbox.

use super::{
    capture::{Binding, CaptureState, CHECKPOINT_VERSION},
    *,
};
use crate::store::{RecordKey, Store};
use jope_api::ids::RunId;
use std::fs;

/// Materialize a complete, known committed run without clobbering an output.
/// Identity attestations still do not establish external corpus/model qualification.
pub fn materialize_committed(store: &Store, run: &RunId, output: &Path) -> Result<usize> {
    materialize(store, run, output, false)
}

#[cfg(test)]
pub(crate) fn materialize_controls(store: &Store, run: &RunId, output: &Path) -> Result<usize> {
    materialize(store, run, output, true)
}

fn materialize(store: &Store, run: &RunId, output: &Path, allow_synthetic: bool) -> Result<usize> {
    let stored = store
        .get(&RecordKey::Run(run.clone()))?
        .context("capture run missing")?;
    let checkpoint = &stored.data["checkpoint"];
    ensure!(
        checkpoint["version"].as_u64() == Some(u64::from(CHECKPOINT_VERSION)),
        "not a research checkpoint"
    );
    ensure!(
        !checkpoint["completed"].is_null(),
        "discovery capture is not complete"
    );
    let _: jope_api::CompletionReason = serde_json::from_value(checkpoint["completed"].clone())?;
    ensure!(
        checkpoint["pending"]
            .as_array()
            .context("pending inventory missing")?
            .is_empty(),
        "capture still has pending GP rounds"
    );
    let state: CaptureState = serde_json::from_value(checkpoint["capture"].clone())?;
    state.validate_storage(store)?;
    ensure!(
        !state.rounds.is_empty() && !state.unknown(),
        "capture has an unknown/aborted outcome or no records; any committed prefix is inspection-only"
    );
    ensure!(
        checkpoint["next_round"].as_u64() == Some(state.rounds.len() as u64),
        "capture/GP round progress mismatch"
    );
    ensure!(
        state.rounds.len() <= state.config.max_rounds
            && state.bytes <= state.config.max_bytes
            && state.charged <= state.config.max_attempts,
        "capture bounds exceeded"
    );
    let binding: Binding = CaptureState::read(store, &state.binding)?;
    ensure!(
        state.roots().iter().all(|r| stored.blobs.contains(r)),
        "capture roots missing from committed run"
    );
    ensure!(
        binding.identity.run_id == run.to_string(),
        "capture run binding mismatch"
    );
    ensure!(
        allow_synthetic || binding.identity.evidence == RecordEvidence::Engine,
        "synthetic control is not training evidence"
    );
    let mut records = vec![];
    let mut expected_bytes = vec![];
    for (index, (number, round)) in state.rounds.iter().enumerate() {
        ensure!(*number == index as u64, "capture round gap");
        let reference = round
            .committed
            .as_ref()
            .context("uncommitted capture round")?;
        ensure!(
            reference.bytes < MAX_LINE_BYTES as u64,
            "capture line exceeds bound"
        );
        let bytes = store.read_blob(reference)?;
        let record: ResearchRound = serde_json::from_slice(&bytes)?;
        record.validate()?;
        ensure!(
            record.round == *number && record.capture_complete,
            "partial capture is inspection-only"
        );
        expected_bytes.extend_from_slice(&bytes);
        expected_bytes.push(b'\n');
        records.push(record);
    }
    ensure!(output.is_absolute(), "capture destination must be absolute");
    let parent = output
        .parent()
        .context("capture destination has no parent")?;
    ensure!(
        fs::symlink_metadata(parent)?.is_dir() && fs::canonicalize(parent)? == parent,
        "capture destination parent must be a canonical real directory"
    );
    let staging = parent.join(format!(".jope-research-{}.jsonl", ulid::Ulid::new()));
    let mut writer = RecordWriter::create(&staging)?;
    for record in &records {
        writer.append(record)?;
    }
    drop(writer);
    #[cfg(test)]
    capture::boundary("after_file_sync");
    ensure!(
        fs::read(&staging)? == expected_bytes,
        "outbox encoding changed during materialization"
    );
    // hard_link is atomic and refuses an existing destination. The staging
    // evidence is deliberately retained, including after uncertain publication.
    fs::hard_link(&staging, output)?;
    #[cfg(test)]
    capture::boundary("after_file_publish");
    File::open(parent)?.sync_all()?;
    Ok(records.len())
}
