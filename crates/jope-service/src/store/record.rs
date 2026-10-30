use jope_api::{
    ids::{ArtifactId, DatasetId, RunId},
    RunState,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Explicit content references are the only links the store interprets as blobs.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlobRef {
    pub sha256: [u8; 32],
    pub bytes: u64,
}

/// `data` remains the caller's last committed checkpoint; buffered work is not durable.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunRecovery {
    pub state: RunState,
    pub interrupted: bool,
    pub revision: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Record {
    pub data: Value,
    pub blobs: Vec<BlobRef>,
    pub run: Option<RunRecovery>,
}

impl Record {
    pub fn metadata(data: Value, blobs: Vec<BlobRef>) -> Self {
        Self {
            data,
            blobs,
            run: None,
        }
    }

    pub fn run(data: Value, blobs: Vec<BlobRef>, state: RunState, revision: u64) -> Self {
        Self {
            data,
            blobs,
            run: Some(RunRecovery {
                state,
                interrupted: false,
                revision,
            }),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Table {
    Runs,
    Events,
    Evaluations,
    Commands,
    Datasets,
    Artifacts,
}

impl Table {
    pub const ALL: [Self; 6] = [
        Self::Runs,
        Self::Events,
        Self::Evaluations,
        Self::Commands,
        Self::Datasets,
        Self::Artifacts,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Runs => "runs",
            Self::Events => "events",
            Self::Evaluations => "evaluations",
            Self::Commands => "commands",
            Self::Datasets => "datasets",
            Self::Artifacts => "artifacts",
        }
    }
}

#[derive(Clone, Debug)]
pub enum RecordKey {
    Run(RunId),
    Event {
        run: RunId,
        sequence: u64,
    },
    Evaluation {
        run: RunId,
        number: u64,
    },
    Command {
        principal: String,
        idempotency_key: String,
    },
    Dataset(DatasetId),
    Artifact(ArtifactId),
}

impl RecordKey {
    pub const fn table(&self) -> Table {
        match self {
            Self::Run(_) => Table::Runs,
            Self::Event { .. } => Table::Events,
            Self::Evaluation { .. } => Table::Evaluations,
            Self::Command { .. } => Table::Commands,
            Self::Dataset(_) => Table::Datasets,
            Self::Artifact(_) => Table::Artifacts,
        }
    }

    pub(super) fn bytes(&self) -> super::Result<Vec<u8>> {
        Ok(match self {
            Self::Run(id) => id.to_string().into_bytes(),
            Self::Dataset(id) => id.to_string().into_bytes(),
            Self::Artifact(id) => id.to_string().into_bytes(),
            Self::Event { run, sequence }
            | Self::Evaluation {
                run,
                number: sequence,
            } => {
                let mut key = run.to_string().into_bytes();
                key.push(0);
                key.extend_from_slice(&sequence.to_be_bytes());
                key
            }
            Self::Command {
                principal,
                idempotency_key,
            } => {
                if principal.is_empty()
                    || !(1..=128).contains(&idempotency_key.len())
                    || !idempotency_key.bytes().all(|v| (32..=126).contains(&v))
                {
                    return Err(super::StoreError::Invalid(
                        "invalid principal/idempotency key",
                    ));
                }
                let length = u32::try_from(principal.len())
                    .map_err(|_| super::StoreError::Invalid("principal too long"))?;
                let mut key = length.to_be_bytes().to_vec();
                key.extend_from_slice(principal.as_bytes());
                key.extend_from_slice(idempotency_key.as_bytes());
                key
            }
        })
    }
}
