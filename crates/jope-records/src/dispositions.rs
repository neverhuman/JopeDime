//! `verify-dispositions`: scoped finding dispositions against hash-bound evidence (W12-T02).
//!
//! The v1 register stays an immutable open inventory. Dispositions live in a separate
//! append-only record whose closures are accepted only when stage-appropriate evidence exists,
//! is byte-identical to its recorded digest and names the same finding, stage and source
//! commit. A digest authenticates bytes, not the truth of their producer; independent review of
//! the evidence remains required. Evidence whose bytes carry the public conformance fixtures'
//! synthetic identity (W12-T07) may back only source-trace and synthetic-fixture evidence,
//! whatever kind it declares.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::py::{self, PosixPath};
use crate::pyjson::{Dict, Value};
use crate::{findings, require, synthetic, Error, Result};

mod evidence;
mod model;
mod repository;
mod validation;

pub(super) use evidence::{check_tests, load_evidence};
pub use model::*;
pub use repository::verify;
pub(super) use validation::validate_record;
