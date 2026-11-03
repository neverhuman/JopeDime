use super::campaign_setup::scheduling::{cohort_win, qualifying_winner, shard_members, strict_win};
use super::campaign_setup::worker::is_sha256_hex;
use super::*;
use tempfile::tempdir;

include!("tests/initial.rs");
include!("tests/remaining.rs");
