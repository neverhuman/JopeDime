use jain_smartcluster_fabric::{
    sha256, valid_identifier, Digest32, DomainSignature, SignatureAuthority, WorkerAcknowledgement,
};
use serde::{Deserialize, Serialize};

use crate::{
    EventIdentity, ExecutionError, ExecutionReceipt, NodeControlFrame, NodeEvent,
    NodeRuntimeIdentity, ProtocolError, PublishedArtifact, SignedStorageFragmentReceipt,
    SignedStorageTelemetry, MAX_STREAM_CHUNK_BYTES,
};

const ASSIGNMENT_ACCEPTED_DOMAIN: &[u8] = b"jain.node.assignment-accepted.v1\0";
const EVENT_DOMAIN: &[u8] = b"jain.node.event.v1\0";
const TERMINAL_DOMAIN: &[u8] = b"jain.node.terminal.v1\0";
const STORAGE_TELEMETRY_FRAME_DOMAIN: &[u8] = b"jain.node.storage-telemetry-frame.v1\0";
const STORAGE_FRAGMENT_RECEIPT_FRAME_DOMAIN: &[u8] =
    b"jain.node.storage-fragment-receipt-frame.v1\0";
const MAX_EVENT_MESSAGE_BYTES: usize = 1024;
const MAX_EVENT_ARTIFACTS: usize = 128;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeFailureCode {
    UnsupportedHost,
    InvalidWorkUnit,
    InvalidStaging,
    NetworkPolicyUnavailable,
    DurabilityUnavailable,
    Cancelled,
    CancellationStateInvalid,
    DeadlineExceeded,
    MemoryLimitExceeded,
    NonZeroExit(i32),
    LogBudgetExceeded,
    StreamProtocolViolation,
    ClockUnavailable,
    StaleAuthority,
    AssignmentStateInvalid,
    IsolationUnavailable,
    StorageUnavailable,
    NetworkUnavailable,
    ContainerRuntimeUnavailable,
    OutboundControlUnavailable,
}

impl From<&ExecutionError> for NodeFailureCode {
    fn from(error: &ExecutionError) -> Self {
        match error {
            ExecutionError::UnsupportedHost => Self::UnsupportedHost,
            ExecutionError::InvalidWorkUnit => Self::InvalidWorkUnit,
            ExecutionError::InvalidStaging => Self::InvalidStaging,
            ExecutionError::NetworkPolicyUnavailable => Self::NetworkPolicyUnavailable,
            ExecutionError::DurabilityUnavailable => Self::DurabilityUnavailable,
            ExecutionError::Cancelled => Self::Cancelled,
            ExecutionError::CancellationState => Self::CancellationStateInvalid,
            ExecutionError::DeadlineExceeded => Self::DeadlineExceeded,
            ExecutionError::MemoryLimitExceeded => Self::MemoryLimitExceeded,
            ExecutionError::NonZeroExit(code) => Self::NonZeroExit(*code),
            ExecutionError::LogBudgetExceeded => Self::LogBudgetExceeded,
            ExecutionError::StreamProtocolViolation => Self::StreamProtocolViolation,
            ExecutionError::ClockUnavailable => Self::ClockUnavailable,
            ExecutionError::StaleAuthority => Self::StaleAuthority,
            ExecutionError::Assignment(_) => Self::AssignmentStateInvalid,
            ExecutionError::IsolationBroker(_) => Self::IsolationUnavailable,
            ExecutionError::Storage(_) => Self::StorageUnavailable,
            ExecutionError::Network(_) => Self::NetworkUnavailable,
            ExecutionError::Docker(_) => Self::ContainerRuntimeUnavailable,
            ExecutionError::OutboundControlUnavailable => Self::OutboundControlUnavailable,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeTerminalOutcome {
    Success(ExecutionReceipt),
    Failure {
        code: NodeFailureCode,
        detail_digest: Digest32,
    },
}

impl NodeTerminalOutcome {
    pub fn failure(error: &ExecutionError) -> Self {
        Self::Failure {
            code: NodeFailureCode::from(error),
            detail_digest: sha256(error.to_string().as_bytes()),
        }
    }
}

mod frame;
mod validation;
pub(crate) use validation::*;

#[cfg(test)]
#[path = "outbound/tests.rs"]
mod tests;
