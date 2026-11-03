use super::*;

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EncodingDatasetIdentity {
    pub dataset_id: String,
    pub content_sha256: Sha256Digest,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EncodingChunkV2 {
    pub schema: String,
    pub chunk_sha256: Sha256Digest,
    pub encoder_identity_sha256: Sha256Digest,
    pub attempt_number: u32,
    pub datasets: Vec<EncodingDatasetIdentity>,
    pub preemptible: bool,
    pub reserved_cpu_cores: usize,
    pub required_node: String,
    pub project: String,
}

impl EncodingChunkV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == ENCODING_CHUNK_SCHEMA,
            "encoding chunk schema is not v2"
        );
        ensure!(
            !self.datasets.is_empty() && self.datasets.len() <= super::super::ENCODING_CHUNK_SIZE,
            "encoding chunk must contain 1..=256 datasets"
        );
        ensure!(self.attempt_number > 0, "encoding attempt number is zero");
        ensure!(
            self.preemptible
                && self.reserved_cpu_cores == 4
                && self.required_node == "xbabe3"
                && self.project == "jope-selfplay-breadth-encode",
            "encoding chunk resource reservation drift"
        );
        let unique = self
            .datasets
            .iter()
            .collect::<std::collections::BTreeSet<_>>();
        ensure!(
            unique.len() == self.datasets.len(),
            "encoding chunk contains duplicate dataset identities"
        );
        let identity = self
            .datasets
            .iter()
            .map(|dataset| format!("{}:{}", dataset.dataset_id, dataset.content_sha256.as_str()))
            .collect::<Vec<_>>()
            .join("|");
        ensure!(
            self.chunk_sha256
                == Sha256Digest::of_bytes(
                    format!(
                        "breadth-encode-v2|{}|{identity}",
                        self.encoder_identity_sha256
                    )
                    .as_bytes()
                ),
            "encoding chunk content identity mismatch"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum EncodingAdoptionDisposition {
    Succeeded {
        encoded_train_sha256: Sha256Digest,
        encoded_test_sha256: Sha256Digest,
    },
    Failed {
        error_class: String,
    },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EncodingAdoptionReceiptV2 {
    pub schema: String,
    pub dataset: EncodingDatasetIdentity,
    pub attempt_number: u32,
    pub job_identity: String,
    pub encoder_identity_sha256: Sha256Digest,
    pub disposition: EncodingAdoptionDisposition,
    pub recorded_at_ms: u64,
}

impl EncodingAdoptionReceiptV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == ENCODING_ADOPTION_RECEIPT_SCHEMA,
            "encoding adoption receipt schema is not v2"
        );
        ensure!(
            !self.dataset.dataset_id.trim().is_empty(),
            "encoding receipt dataset id is empty"
        );
        ensure!(self.attempt_number > 0, "encoding attempt number is zero");
        ensure!(
            !self.job_identity.trim().is_empty(),
            "encoding job identity is empty"
        );
        ensure!(
            self.recorded_at_ms > 0,
            "encoding receipt timestamp is absent"
        );
        if let EncodingAdoptionDisposition::Failed { error_class } = &self.disposition {
            ensure!(
                !error_class.trim().is_empty(),
                "failed encoding receipt requires an error class"
            );
        }
        Ok(())
    }

    pub fn succeeded(&self) -> bool {
        matches!(
            self.disposition,
            EncodingAdoptionDisposition::Succeeded { .. }
        )
    }
}
