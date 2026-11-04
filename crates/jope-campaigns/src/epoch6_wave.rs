//! Durable, isolated end-to-end wave proofs for the Epoch 6 contracts.

use crate::epoch6::{
    ArtifactCas, ArtifactKind, ArtifactRefV1, CohortPlanV2, CohortSignalSnapshotV1,
    LaneActivationSetV1, ReplayProtocolV2, SupplyIntentStateV2, SupplyIntentV2, WaveCommitV2,
    COHORT_PLAN_SCHEMA, COHORT_SIGNAL_SNAPSHOT_SCHEMA, SUPPLY_INTENT_SCHEMA, WAVE_COMMIT_SCHEMA,
};
use crate::task::Task;
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

pub const SUPPLY_INTENT_HISTORY_SCHEMA: &str = "SupplyIntentHistory.v2";
pub const SYNTHETIC_MANIFEST_SCHEMA: &str = "Epoch6SyntheticManifest.v2";
pub const SYNTHETIC_TERMINAL_SCHEMA: &str = "Epoch6SyntheticTerminalReceipt.v2";
pub const WAVE_PROOF_SCHEMA: &str = "Epoch6WaveProof.v2";

mod canonical_storage;
use canonical_storage::{
    canonical_bytes, canonical_sha256, domain_sha256, durable_write_new, sync_directory,
};

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SupplyIntentTransitionV2 {
    pub schema: String,
    pub sequence: u64,
    pub intent_sha256: String,
    pub state: SupplyIntentStateV2,
    pub previous_entry_sha256: Option<String>,
    pub entry_sha256: String,
}

#[derive(Serialize)]
struct SupplyIntentTransitionBody<'a> {
    schema: &'a str,
    sequence: u64,
    intent_sha256: &'a str,
    state: SupplyIntentStateV2,
    previous_entry_sha256: Option<&'a str>,
}

impl SupplyIntentTransitionV2 {
    fn new(
        sequence: u64,
        intent_sha256: &str,
        state: SupplyIntentStateV2,
        previous_entry_sha256: Option<&str>,
    ) -> Result<Self> {
        let body = SupplyIntentTransitionBody {
            schema: SUPPLY_INTENT_HISTORY_SCHEMA,
            sequence,
            intent_sha256,
            state,
            previous_entry_sha256,
        };
        Ok(Self {
            schema: SUPPLY_INTENT_HISTORY_SCHEMA.to_owned(),
            sequence,
            intent_sha256: intent_sha256.to_owned(),
            state,
            previous_entry_sha256: previous_entry_sha256.map(str::to_owned),
            entry_sha256: canonical_sha256(&body)?,
        })
    }

    fn validate(&self) -> Result<()> {
        let expected = Self::new(
            self.sequence,
            &self.intent_sha256,
            self.state,
            self.previous_entry_sha256.as_deref(),
        )?;
        ensure!(*self == expected, "supply-intent transition digest drift");
        Ok(())
    }
}

#[derive(Debug)]
pub struct SupplyIntentHistoryV2 {
    path: PathBuf,
    intent_sha256: String,
}

impl SupplyIntentHistoryV2 {
    pub fn create(root: &Path, intent: &SupplyIntentV2) -> Result<Self> {
        intent.validate()?;
        ensure!(
            intent.state == SupplyIntentStateV2::Planned,
            "supply history must begin at Planned"
        );
        let intent_sha256 = intent.identity()?;
        let path = root.join(format!("{intent_sha256}.jsonl"));
        let entry =
            SupplyIntentTransitionV2::new(0, &intent_sha256, SupplyIntentStateV2::Planned, None)?;
        let mut bytes = canonical_bytes(&entry)?;
        bytes.push(b'\n');
        durable_write_new(&path, &bytes, 0o600)?;
        Ok(Self {
            path,
            intent_sha256,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn read(&self) -> Result<Vec<SupplyIntentTransitionV2>> {
        let mut bytes = Vec::new();
        File::open(&self.path)?.read_to_end(&mut bytes)?;
        let text = std::str::from_utf8(&bytes).context("supply history is not UTF-8")?;
        ensure!(
            text.ends_with('\n'),
            "supply history has a partial final line"
        );
        let entries: Vec<SupplyIntentTransitionV2> = text
            .lines()
            .map(|line| serde_json::from_str(line).context("decode supply history entry"))
            .collect::<Result<_>>()?;
        ensure!(!entries.is_empty(), "supply history is empty");
        for (index, entry) in entries.iter().enumerate() {
            entry.validate()?;
            ensure!(
                entry.sequence == index as u64 && entry.intent_sha256 == self.intent_sha256,
                "supply history sequence or identity drift"
            );
            if index == 0 {
                ensure!(
                    entry.state == SupplyIntentStateV2::Planned
                        && entry.previous_entry_sha256.is_none(),
                    "supply history genesis drift"
                );
            } else {
                ensure!(
                    entry.previous_entry_sha256.as_deref()
                        == Some(entries[index - 1].entry_sha256.as_str())
                        && entries[index - 1].state.can_transition_to(entry.state),
                    "supply history chain or state transition drift"
                );
            }
        }
        Ok(entries)
    }

    pub fn transition(&self, intent: &mut SupplyIntentV2, next: SupplyIntentStateV2) -> Result<()> {
        ensure!(
            intent.identity()? == self.intent_sha256,
            "supply history/intent identity mismatch"
        );
        let mut file = OpenOptions::new()
            .read(true)
            .append(true)
            .open(&self.path)?;
        // A failed lock is refused before history/intent changes, dropping and closing `file`.
        ensure!(file.lock().is_ok(), "lock supply-intent history failed");
        let result = (|| -> Result<()> {
            let entries = self.read()?;
            let previous = entries.last().unwrap();
            ensure!(
                previous.state == intent.state,
                "in-memory supply state is not the durable state"
            );
            let mut transitioned = intent.clone();
            transitioned.transition(next)?;
            let entry = SupplyIntentTransitionV2::new(
                entries.len() as u64,
                &self.intent_sha256,
                next,
                Some(&previous.entry_sha256),
            )?;
            let mut bytes = canonical_bytes(&entry)?;
            bytes.push(b'\n');
            file.write_all(&bytes)?;
            file.sync_all()?;
            sync_directory(self.path.parent().unwrap())?;
            *intent = transitioned;
            Ok(())
        })();
        // Unlock errors leave `result` unchanged; return closes `file`, releasing any remaining
        // lock when the final open-file-description reference closes.
        let _ = file.unlock();
        result
    }
}

mod synthetic;
pub use synthetic::{
    execute_synthetic_wave, Epoch6WaveProofV2, SyntheticManifestV2, SyntheticTerminalReceiptV2,
};
