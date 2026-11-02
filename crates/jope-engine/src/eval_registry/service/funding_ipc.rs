//! Separate pinned cross-executable funding transport. The ordinary fit client
//! retains its same-executable rule and original 64 KiB frame bound.
#[allow(unused_imports)]
use super::*;
use crate::eval_registry::{funding as f, funding_config::InstalledFunding};

const MAX_FUNDING_BYTES: u64 = 16 * 1024 * 1024;
const FUNDING_LOCK_MIN_SLEEP_NS: u64 = 1_000_000;
const FUNDING_LOCK_MAX_SLEEP_NS: u64 = 25_000_000;

mod client;
mod protocol;
mod server;

pub use client::funding_request;
pub(super) use client::funding_request_installed;
pub(super) use protocol::funding_admission_installed;
pub use protocol::{funding_admission, FundingRequest, VerifiedFundingAdmission};
pub(super) use server::{lock_with_deadline, serve_request};
