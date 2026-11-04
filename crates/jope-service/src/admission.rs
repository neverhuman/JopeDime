//! Optional, process-wide dataset admission policy.
//!
//! An embedding executable may install one policy before it starts an engine. Every dataset
//! registration (`POST /v1/datasets`: CSV stream, inline text or local path) then asks it with
//! the SHA-256 of the dataset's raw bytes before anything is committed. With no policy installed
//! (the default for every other consumer of this crate) every dataset is admitted.
//!
//! The policy runs in the engine process. A background engine started from the same executable
//! installs the same policy, so local HTTP clients cannot register around it.

use std::sync::OnceLock;

/// Admit (`Ok`) or refuse (`Err(detail)`) a dataset by the lowercase hex SHA-256 of its bytes.
/// The detail is returned to the client as the problem's `detail`.
pub type DatasetPolicy = fn(&str) -> Result<(), String>;

static POLICY: OnceLock<DatasetPolicy> = OnceLock::new();

/// Install the policy once for this process. Returns `false` if one was already installed;
/// the first policy stays in force.
pub fn set_dataset_policy(policy: DatasetPolicy) -> bool {
    POLICY.set(policy).is_ok()
}

/// Ask the installed policy, if any.
pub fn admit_dataset(sha256_hex: &str) -> Result<(), String> {
    POLICY.get().map_or(Ok(()), |policy| policy(sha256_hex))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn without_a_policy_every_dataset_is_admitted() {
        // No test in this crate installs a policy: the default must stay permissive.
        assert!(admit_dataset(&"0".repeat(64)).is_ok());
    }
}
