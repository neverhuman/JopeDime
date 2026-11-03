//! Immutable teaching corpus. Scores remain explicitly adaptive to each cut.
use crate::{
    encoding::Encoding,
    grammar::{State, Vocab},
    manifest::*,
    prepare::Prepared,
};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

mod records;
pub use records::{ContextRow, Group, Index, Record, Records};

pub const SEQUENCE: usize = 128;
pub const RECORD_BYTES: usize = 532;

mod build;
mod ledger;
mod token_cache;

pub use build::{build, build_observed, build_observed_cached, CorpusSources};
use ledger::LedgerCache;
pub use ledger::ScanCacheStats;
pub(crate) use ledger::{scan_ledger, ScanInputs, Selected};
pub use token_cache::read_bound_vocabulary;
pub(crate) use token_cache::TokenCache;
