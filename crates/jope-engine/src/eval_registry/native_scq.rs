//! Closed native scheduler proof. Legacy namespace/funding routes stay held.
#[path = "native_scq/policy.rs"]
pub(crate) mod policy;
use super::{service::ServiceInputs, ProcessIdentity, RequestProvenance, VerifiedFit};
use anyhow::{ensure, Context, Result};
use jain_smartcluster::{
    AttachTrustedFitChildRequest, ClientFrame, FencedAttempt, FrameCodec, GetRequest,
    IdempotencyKey, JobId, JobRecord, JobState, ProtocolVersion, Request, RequestEnvelope,
    Response, ServerFrame, TrustedFitChildPlacement, Validate, Workload,
};
use policy::{hash, InstalledPolicy};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::os::unix::fs::MetadataExt;
use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NativeRequest {
    pub job_id: JobId,
    pub candidate_index: u32,
    pub descriptor_sha256: String,
    pub deadline_monotonic_ns: u64,
    pub destination: PathBuf,
}
/// Deliberately not deserializable or constructible by any client.
pub(crate) struct NativeAdmission {
    policy: Arc<InstalledPolicy>,
    peer: ProcessIdentity,
    target_leader: ProcessIdentity,
    authority_leader: ProcessIdentity,
    pub(crate) target: FencedAttempt,
    pub(crate) authority: FencedAttempt,
    pub(crate) task_sha256: String,
    pub(crate) descriptor_sha256: String,
    pub(crate) candidate_index: u32,
    pub(crate) task_key: String,
    pub(crate) slot_key: String,
    pub(crate) fit_key: String,
    pub(crate) deadline_monotonic_ns: u64,
    pub(crate) task_deadline_ms: u64,
    pub(crate) task_budget: u32,
    pub(crate) run: String,
    pub(crate) generation: u64,
    pub(crate) task_id: String,
    pub(crate) placement: Option<TrustedFitChildPlacement>,
}
mod admission;
mod auth;
mod witness;

pub(crate) use admission::{verify_placement, verify_request};
#[cfg(test)]
pub(super) use auth::inspected_process;
pub(crate) use auth::{active_process, authenticated_exchange};
pub(super) use auth::{
    attempt_path, cgroup, current_job, exchange, live, monotonic_ns, native_leader, service,
    verify_ancestry,
};
#[cfg(test)]
pub(super) use auth::{inspected_process, process_parent};
pub(crate) use witness::verify_witness_event;

#[cfg(test)]
mod tests;
