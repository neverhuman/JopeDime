//! Durable, isolated end-to-end wave proofs for the Epoch 6 contracts.

use crate::epoch6::{
    ArtifactCas, ArtifactKind, ArtifactRefV1, CohortPlanV2, CohortSignalSnapshotV1,
    LaneActivationSetV1, ReplayProtocolV2, SupplyIntentStateV2, SupplyIntentV2, WaveCommitV2,
    COHORT_PLAN_SCHEMA, COHORT_SIGNAL_SNAPSHOT_SCHEMA, SUPPLY_INTENT_SCHEMA, WAVE_COMMIT_SCHEMA,
};
use crate::task::Task;
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

pub const SUPPLY_INTENT_HISTORY_SCHEMA: &str = "SupplyIntentHistory.v2";
pub const SYNTHETIC_MANIFEST_SCHEMA: &str = "Epoch6SyntheticManifest.v2";
pub const SYNTHETIC_TERMINAL_SCHEMA: &str = "Epoch6SyntheticTerminalReceipt.v2";
pub const WAVE_PROOF_SCHEMA: &str = "Epoch6WaveProof.v2";

fn canonical_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(value)?)
}

fn canonical_sha256<T: Serialize>(value: &T) -> Result<String> {
    Ok(hex::encode(Sha256::digest(canonical_bytes(value)?)))
}

fn domain_sha256(domain: &str, fields: impl Serialize) -> Result<String> {
    canonical_sha256(&serde_json::json!({"domain": domain, "fields": fields}))
}

fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}

fn durable_write_new(path: &Path, bytes: &[u8], mode: u32) -> Result<()> {
    let parent = path.parent().context("durable path has no parent")?;
    fs::create_dir_all(parent)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(mode)
        .open(path)
        .with_context(|| format!("create {}", path.display()))?;
    file.write_all(bytes)?;
    file.sync_all()?;
    fs::set_permissions(path, fs::Permissions::from_mode(mode))?;
    sync_directory(parent)
}

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
        // SAFETY: `file` owns the live FD through append/sync; FD/LOCK_EX use libc's c_int ABI.
        // flock borrows it without pointers or ownership transfer. Nonzero results are refused
        // before history/intent changes, dropping and closing `file` on that error path.
        let lock_result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) };
        ensure!(lock_result == 0, "lock supply-intent history failed");
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
        // SAFETY: `file` owns the live FD after either outcome; FD/LOCK_UN use libc's c_int ABI.
        // flock only borrows it. Unlock errors leave `result` unchanged; return closes `file`,
        // releasing any remaining lock when the final open-file-description reference closes.
        unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_UN) };
        result
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SyntheticManifestV2 {
    pub schema: String,
    pub epoch: String,
    pub wave: u64,
    pub slot: u32,
    pub task: Task,
    pub dataset_sha256: String,
    pub item_seed_sha256: String,
    pub evaluator: ArtifactRefV1,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SyntheticTerminalReceiptV2 {
    pub schema: String,
    pub slot: u32,
    pub manifest_sha256: String,
    pub evaluator_object_sha256: String,
    pub output_sha256: String,
    pub terminal_state: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Epoch6WaveProofV2 {
    pub schema: String,
    pub epoch: String,
    pub wave: u64,
    pub manifest_count: u32,
    pub task_counts: BTreeMap<Task, u32>,
    pub replay_protocol_sha256: String,
    pub cohort_plan_sha256: String,
    pub evaluator_object_sha256: String,
    pub manifest_set_sha256: String,
    pub queue_reservation_sha256: String,
    pub terminal_receipt_root_sha256: String,
    pub keeper_acknowledgement_sha256: String,
    pub ingest_root_sha256: String,
    pub primary_supply_history_root_sha256: String,
    pub superseded_supply_history_root_sha256: String,
}

impl Epoch6WaveProofV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.schema == WAVE_PROOF_SCHEMA, "wave proof schema drift");
        ensure!(
            matches!(self.manifest_count, 128 | 192 | 256),
            "wave proof count is not admitted"
        );
        ensure!(
            self.task_counts.values().sum::<u32>() == self.manifest_count,
            "wave proof task counts drift"
        );
        for digest in [
            &self.replay_protocol_sha256,
            &self.cohort_plan_sha256,
            &self.evaluator_object_sha256,
            &self.manifest_set_sha256,
            &self.queue_reservation_sha256,
            &self.terminal_receipt_root_sha256,
            &self.keeper_acknowledgement_sha256,
            &self.ingest_root_sha256,
            &self.primary_supply_history_root_sha256,
            &self.superseded_supply_history_root_sha256,
        ] {
            ensure!(
                digest.len() == 64
                    && digest
                        .bytes()
                        .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()),
                "wave proof contains an invalid digest"
            );
        }
        Ok(())
    }

    pub fn semantic_root(&self) -> Result<String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

fn history_root(history: &SupplyIntentHistoryV2) -> Result<String> {
    canonical_sha256(&history.read()?)
}

#[allow(clippy::too_many_arguments)]
pub fn execute_synthetic_wave(
    root: &Path,
    protocol: &ReplayProtocolV2,
    activation: &LaneActivationSetV1,
    epoch: &str,
    wave: u64,
    manifest_count: u32,
    completion_order: &[u32],
) -> Result<Epoch6WaveProofV2> {
    ensure!(
        root.is_absolute() && !root.exists(),
        "wave proof root must be new and absolute"
    );
    ensure!(
        matches!(manifest_count, 128 | 192 | 256),
        "unsupported wave size"
    );
    protocol.validate()?;
    activation.validate()?;
    let active_tasks: Vec<_> = protocol
        .task_order
        .iter()
        .copied()
        .filter(|task| activation.is_active(*task, epoch, wave))
        .collect();
    ensure!(!active_tasks.is_empty(), "wave has no active tasks");
    let expected_order: BTreeSet<_> = (0..manifest_count).collect();
    ensure!(
        completion_order.iter().copied().collect::<BTreeSet<_>>() == expected_order
            && completion_order.len() == manifest_count as usize,
        "completion order is not an exact manifest permutation"
    );
    fs::create_dir_all(root)?;
    sync_directory(root.parent().unwrap())?;

    let protocol_sha256 = protocol.digest()?;
    let activation_sha256 = canonical_sha256(activation)?;
    let catalog_sha256 = domain_sha256("epoch6-synthetic-catalog", (epoch, wave))?;
    let observation_sha256 = domain_sha256("epoch6-synthetic-observations", (epoch, wave))?;
    let signal = CohortSignalSnapshotV1 {
        schema: COHORT_SIGNAL_SNAPSHOT_SCHEMA.to_owned(),
        catalog_sha256: catalog_sha256.clone(),
        observations_sha256: observation_sha256.clone(),
        stale_wave_input_sha256: domain_sha256("epoch6-stale-input", (epoch, wave))?,
        hardness_input_sha256: domain_sha256("epoch6-hardness-input", (epoch, wave))?,
    };
    let cohort = CohortPlanV2 {
        schema: COHORT_PLAN_SCHEMA.to_owned(),
        activation_sha256: activation_sha256.clone(),
        replay_protocol_sha256: protocol_sha256.clone(),
        catalog_sha256: catalog_sha256.clone(),
        observation_sha256: observation_sha256.clone(),
        signal_snapshot_sha256: signal.digest()?,
    };
    cohort.validate()?;
    let cohort_sha256 = canonical_sha256(&cohort)?;

    let supply_root = root.join("supply-history");
    fs::create_dir_all(&supply_root)?;
    let make_intent = |proposal: &str| -> Result<SupplyIntentV2> {
        Ok(SupplyIntentV2 {
            schema: SUPPLY_INTENT_SCHEMA.to_owned(),
            checkpoint_sha256: domain_sha256("epoch6-checkpoint", epoch)?,
            cohort_sha256: cohort_sha256.clone(),
            catalog_sha256: catalog_sha256.clone(),
            replay_protocol_sha256: protocol_sha256.clone(),
            task_activation_sha256: activation_sha256.clone(),
            proposal_identity_sha256: domain_sha256("epoch6-proposal", proposal)?,
            seed_namespace_sha256: domain_sha256("epoch6-seed-namespace", (epoch, wave))?,
            expected_manifest_count: manifest_count,
            admission_budget: manifest_count,
            state: SupplyIntentStateV2::Planned,
        })
    };
    let mut primary_intent = make_intent("primary")?;
    let mut alternate_intent = make_intent("alternate")?;
    let primary_history = SupplyIntentHistoryV2::create(&supply_root, &primary_intent)?;
    let alternate_history = SupplyIntentHistoryV2::create(&supply_root, &alternate_intent)?;
    alternate_history.transition(&mut alternate_intent, SupplyIntentStateV2::Superseded)?;
    primary_history.transition(&mut primary_intent, SupplyIntentStateV2::Proposing)?;

    let cas = ArtifactCas::new(root.join("artifact-cas"));
    let evaluator_bytes = canonical_bytes(&serde_json::json!({
        "schema": "Epoch6SyntheticEvaluator.v1",
        "protocol_sha256": protocol.evaluator_protocol_sha256,
    }))?;
    let evaluator = cas.publish_bytes(
        ArtifactKind::Evaluator,
        &protocol.evaluator_protocol_sha256,
        &evaluator_bytes,
    )?;
    let mut manifests = Vec::with_capacity(manifest_count as usize);
    let mut task_counts = BTreeMap::new();
    for slot in 0..manifest_count {
        let task = active_tasks[slot as usize % active_tasks.len()];
        *task_counts.entry(task).or_insert(0) += 1;
        let dataset_sha256 = domain_sha256("epoch6-synthetic-dataset", (task, slot))?;
        let operator = &protocol.operator_vocabulary
            [slot as usize % protocol.operator_vocabulary.len()]
        .operator;
        let item_seed = protocol.derive_item_seed(
            wave,
            task,
            &dataset_sha256,
            "synthetic-wave",
            slot % protocol.candidate_slots.slots_per_dataset,
            operator,
            u64::from(slot),
        )?;
        manifests.push(SyntheticManifestV2 {
            schema: SYNTHETIC_MANIFEST_SCHEMA.to_owned(),
            epoch: epoch.to_owned(),
            wave,
            slot,
            task,
            dataset_sha256,
            item_seed_sha256: item_seed.to_string(),
            evaluator: evaluator.clone(),
        });
    }
    primary_history.transition(&mut primary_intent, SupplyIntentStateV2::Built)?;
    let manifest_set_sha256 = canonical_sha256(&manifests)?;
    let wave_commit = WaveCommitV2 {
        schema: WAVE_COMMIT_SCHEMA.to_owned(),
        epoch: epoch.to_owned(),
        wave,
        cohort_plan_sha256: cohort_sha256.clone(),
        task_counts: task_counts.clone(),
        manifest_count,
        manifest_set_sha256: manifest_set_sha256.clone(),
    };
    wave_commit.validate()?;

    let finalizer = root.join("reservation-publication.finalizer");
    durable_write_new(&finalizer, primary_intent.identity()?.as_bytes(), 0o444)?;
    let queue_reservation_sha256 = domain_sha256(
        "epoch6-queue-reservation",
        (&manifest_set_sha256, primary_intent.identity()?),
    )?;
    primary_history.transition(&mut primary_intent, SupplyIntentStateV2::Published)?;

    let mut terminal_by_slot = BTreeMap::new();
    for slot in completion_order {
        let manifest = &manifests[*slot as usize];
        let private_evaluator = root.join(format!("jobs/{slot:06}/private/evaluator"));
        cas.materialize(&manifest.evaluator, &private_evaluator)?;
        let manifest_sha256 = canonical_sha256(manifest)?;
        let output = canonical_bytes(&serde_json::json!({
            "domain": "epoch6-synthetic-output",
            "manifest_sha256": manifest_sha256,
            "evaluator_object_sha256": evaluator.object_sha256,
        }))?;
        let output_sha256 = hex::encode(Sha256::digest(&output));
        let output_path = root.join(format!("jobs/{slot:06}/output.json"));
        durable_write_new(&output_path, &output, 0o444)?;
        let receipt = SyntheticTerminalReceiptV2 {
            schema: SYNTHETIC_TERMINAL_SCHEMA.to_owned(),
            slot: *slot,
            manifest_sha256,
            evaluator_object_sha256: evaluator.object_sha256.clone(),
            output_sha256,
            terminal_state: "completed".to_owned(),
        };
        let receipt_path = root.join(format!("jobs/{slot:06}/terminal.json"));
        durable_write_new(&receipt_path, &canonical_bytes(&receipt)?, 0o444)?;
        let progress_path = root.join(format!("jobs/{slot:06}/progress.json"));
        durable_write_new(
            &progress_path,
            &canonical_bytes(&serde_json::json!({"slot": slot, "terminal": true}))?,
            0o444,
        )?;
        terminal_by_slot.insert(*slot, receipt);
    }
    ensure!(
        terminal_by_slot.len() == manifest_count as usize,
        "terminal receipt set is incomplete"
    );
    let terminal_receipts: Vec<_> = terminal_by_slot.into_values().collect();
    let terminal_receipt_root_sha256 = canonical_sha256(&terminal_receipts)?;
    primary_history.transition(&mut primary_intent, SupplyIntentStateV2::KeeperAcked)?;
    let keeper_acknowledgement_sha256 = domain_sha256(
        "epoch6-keeper-acknowledgement",
        (&primary_intent.identity()?, &terminal_receipt_root_sha256),
    )?;
    let ingest_root_sha256 = domain_sha256(
        "epoch6-ingest",
        (
            &terminal_receipt_root_sha256,
            &keeper_acknowledgement_sha256,
        ),
    )?;
    let proof = Epoch6WaveProofV2 {
        schema: WAVE_PROOF_SCHEMA.to_owned(),
        epoch: epoch.to_owned(),
        wave,
        manifest_count,
        task_counts,
        replay_protocol_sha256: protocol_sha256,
        cohort_plan_sha256: cohort_sha256,
        evaluator_object_sha256: evaluator.object_sha256,
        manifest_set_sha256,
        queue_reservation_sha256,
        terminal_receipt_root_sha256,
        keeper_acknowledgement_sha256,
        ingest_root_sha256,
        primary_supply_history_root_sha256: history_root(&primary_history)?,
        superseded_supply_history_root_sha256: history_root(&alternate_history)?,
    };
    proof.validate()?;
    Ok(proof)
}
