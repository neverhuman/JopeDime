use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

include!("timeseries/types_and_metrics.rs");
include!("timeseries/reductions.rs");

mod metadata;
use metadata::{number_at, validate_manifest_spec, validate_meta};

include!("timeseries/tests.rs");
