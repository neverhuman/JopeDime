//! Local, nonqualifying custody-v2 observations. No S3 client or retention executor.
pub mod cost;
pub mod reader;

use crate::canon::{canonical_json, is_hex64, sha256_hex};
use anyhow::{ensure, Context, Result};
use jope_records::custody_archive::{self, Blob, MAX_MEMBERS, MAX_PAYLOAD};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    os::unix::fs::{DirBuilderExt, OpenOptionsExt},
    path::Path,
    sync::Arc,
};

/// Configured upper or lower bound for max manifest.
include!("custody/classification.rs");
include!("custody/verification.rs");
include!("custody/packing.rs");
include!("custody/dry_run.rs");
