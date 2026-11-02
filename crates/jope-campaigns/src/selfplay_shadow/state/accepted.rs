use super::*;

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EncodingAdoptionLedgerV2 {
    #[serde(default)]
    pub successful: BTreeMap<String, EncodingAdoptionReceiptV2>,
    #[serde(default)]
    pub latest_failures: BTreeMap<String, EncodingAdoptionReceiptV2>,
}

impl EncodingAdoptionLedgerV2 {
    fn key(receipt: &EncodingAdoptionReceiptV2) -> String {
        format!(
            "{}:{}",
            receipt.dataset.dataset_id, receipt.dataset.content_sha256
        )
    }

    pub fn validate(&self) -> Result<()> {
        for (key, receipt) in &self.successful {
            receipt.validate()?;
            ensure!(
                receipt.succeeded(),
                "successful adoption contains a failure"
            );
            ensure!(*key == Self::key(receipt), "successful adoption key drift");
        }
        for (key, receipt) in &self.latest_failures {
            receipt.validate()?;
            ensure!(!receipt.succeeded(), "failure ledger contains a success");
            ensure!(*key == Self::key(receipt), "encoding failure key drift");
            ensure!(
                !self.successful.contains_key(key),
                "successful dataset cannot retain a failure disposition"
            );
        }
        Ok(())
    }

    pub fn adopt(&mut self, receipt: EncodingAdoptionReceiptV2) -> Result<bool> {
        receipt.validate()?;
        let key = Self::key(&receipt);
        if let Some(success) = self.successful.get(&key) {
            ensure!(
                success == &receipt || !receipt.succeeded(),
                "a successful dataset adoption cannot be replaced"
            );
            return Ok(false);
        }
        if receipt.succeeded() {
            self.latest_failures.remove(&key);
            self.successful.insert(key, receipt);
            self.validate()?;
            return Ok(true);
        }
        if let Some(previous) = self.latest_failures.get(&key) {
            ensure!(
                receipt.attempt_number >= previous.attempt_number,
                "encoding failure attempts cannot move backward"
            );
            if receipt == *previous {
                return Ok(false);
            }
        }
        self.latest_failures.insert(key, receipt);
        self.validate()?;
        Ok(true)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AcceptedFront {
    pub task: Task,
    pub generation: u64,
    pub checkpoint_sha256: Sha256Digest,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_checkpoint_sha256: Option<Sha256Digest>,
    pub promotion_evidence_sha256: Sha256Digest,
    pub promoted_at_ms: u64,
}

impl AcceptedFront {
    pub fn validate(&self, expected_task: Task) -> Result<()> {
        ensure!(self.task == expected_task, "accepted task branch mismatch");
        ensure!(self.promoted_at_ms > 0, "promotion timestamp is absent");
        if self.generation == 0 {
            ensure!(
                self.parent_checkpoint_sha256.is_none(),
                "generation zero cannot have an accepted parent"
            );
        } else {
            ensure!(
                self.parent_checkpoint_sha256.is_some(),
                "accepted child is missing its parent"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
/// The schema-v2 accepted-checkpoint pointers for binary and regression tasks.
///
/// New task families require a later schema; this type intentionally keeps only
/// the two fronts represented by the current wire format.
pub struct AcceptedByTaskV2 {
    /// Wire-format schema identifier.
    pub schema: String,
    /// Accepted checkpoint and promotion metadata for the binary task.
    pub binary: AcceptedFront,
    /// Accepted checkpoint and promotion metadata for the regression task.
    pub regression: AcceptedFront,
}

impl AcceptedByTaskV2 {
    /// Builds and validates a schema-v2 pointer from its two supported fronts.
    pub fn new(binary: AcceptedFront, regression: AcceptedFront) -> Result<Self> {
        let value = Self {
            schema: ACCEPTED_BY_TASK_SCHEMA.to_owned(),
            binary,
            regression,
        };
        value.validate()?;
        Ok(value)
    }

    /// Checks the schema identifier and each task-specific accepted front.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == ACCEPTED_BY_TASK_SCHEMA,
            "accepted pointer schema is not v2"
        );
        self.binary.validate(Task::Binary)?;
        self.regression.validate(Task::Regression)?;
        Ok(())
    }

    /// Returns the accepted front for a binary or regression task.
    ///
    /// # Panics
    /// Panics for timeseries and multiclass tasks because schema v2 has no
    /// corresponding accepted-front fields. Use a newer schema for those tasks.
    pub fn front(&self, task: Task) -> &AcceptedFront {
        match task {
            Task::Binary => &self.binary,
            Task::Regression => &self.regression,
            // AcceptedByTask.v2 is a two-front wire schema; new-task fronts
            // arrive with a future schema revision. No caller can reach this
            // today (no new-task anchors exist), and reaching it would mean a
            // routing bug, so fail loudly rather than mislabel a front.
            Task::Timeseries | Task::Multiclass => {
                panic!("AcceptedByTask.v2 has no {task} front")
            }
        }
    }

    /// Replaces one supported task front if its current checkpoint matches the
    /// expected parent, recording the child and its evidence.
    pub fn compare_and_swap(
        &mut self,
        task: Task,
        expected_parent: &Sha256Digest,
        child_checkpoint: Sha256Digest,
        evidence_sha256: Sha256Digest,
        promoted_at_ms: u64,
    ) -> Result<()> {
        self.validate()?;
        let front = match task {
            Task::Binary => &mut self.binary,
            Task::Regression => &mut self.regression,
            Task::Timeseries | Task::Multiclass => {
                bail!("AcceptedByTask.v2 has no {task} front to swap")
            }
        };
        ensure!(
            &front.checkpoint_sha256 == expected_parent,
            "accepted task pointer changed before compare-and-swap"
        );
        ensure!(
            child_checkpoint != *expected_parent,
            "accepted child equals its parent"
        );
        *front = AcceptedFront {
            task,
            generation: front.generation + 1,
            checkpoint_sha256: child_checkpoint,
            parent_checkpoint_sha256: Some(expected_parent.clone()),
            promotion_evidence_sha256: evidence_sha256,
            promoted_at_ms,
        };
        self.validate()
    }
}
