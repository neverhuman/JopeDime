//! DIME-owned recovery and campaign status contracts.
//!
//! Infrastructure attempts are immutable terminal receipts.  A retry appends
//! one receipt for the attempt that just ended, then launches a fresh job
//! identity carrying the same `scientific_payload`.  Intentional cancellation
//! is a separate, short-lived, hash-bound authority; scheduler state alone is
//! never cancellation authority.

use crate::selfplay_runtime::{atomic_json, sha256_bytes};
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

pub const DIME_INFRASTRUCTURE_ATTEMPT_SCHEMA: &str = "DimeInfrastructureAttempt.v1";
pub const DIME_DRAIN_INTENT_SCHEMA: &str = "DimeDrainIntent.v1";
pub const DIME_RECOVERY_PROJECTION_SCHEMA: &str = "DimeRecoveryProjection.v1";
pub const JOPE_DLGP_BUDGET_CAMPAIGN_STATUS_SCHEMA: &str = "JopeDlgpBudgetCampaignStatus.v1";

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_sha256(value: &str, field: &str) -> Result<()> {
    ensure!(is_sha256(value), "{field} is not a lowercase SHA-256");
    Ok(())
}

fn canonical_sha256<T: Serialize>(value: &T) -> Result<String> {
    Ok(sha256_bytes(&serde_json::to_vec(value)?))
}

mod attempts;
mod budget_campaign;
mod drain;
mod recovery;
#[cfg(test)]
mod tests;

pub use attempts::{
    append_infrastructure_attempt, digest_label, ensure_same_scientific_payload,
    infrastructure_attempts, latest_attempt_receipt_sha256, latest_infrastructure_attempt,
    read_validated_attempt, DimeInfrastructureAttemptV1, DimeNamedDigestV1, DimeRoleV1,
    DimeScientificPayloadV1, DimeTerminalClassV1, DimeTerminalEvidenceV1,
};
pub use budget_campaign::{
    retry_backoff_ms, DlgpActiveJobV1, DlgpCampaignGateStateV1, JopeDlgpBudgetCampaignStatusV1,
};
pub use drain::{matching_drain_intent, write_drain_intent, DimeDrainIntentV1};
pub use recovery::{
    load_or_rebuild_recovery_projection, rebuild_recovery_projection, recovery_projection_path,
    save_recovery_projection, terminal_classification, DimeRecoveryProjectionV1,
};
