//! W10-T09: an independent recomputation of the transfer KPI from the plan, the hash-pinned
//! labels and the frozen predictions.
//!
//! The oracle below reads only plan, freeze and report JSON and the assets they pin. It never
//! links `jope_bench::panel::transfer`'s scoring code: it recomputes the training-mean baseline
//! and each arm's RMSE per search, the five-audit mean gain and the outer gain per
//! dataset/seed/outer unit, their per-unit minimum, and the equal-weight means over units within
//! datasets, datasets within families and families. Missing or failed pairs, zero or nonfinite
//! baselines and invalid values leave the score incomplete; nothing becomes a zero gain. Given a
//! reporter output it lists every unit, family, loss and headline disagreement. Release and
//! isolation evidence is the reporter's responsibility and is not assessed here. The crate's
//! public types and entry points are used only to build the cross-implementation fixture.

use jope_bench::panel::transfer::{
    self, Arm, Asset, Dataset, Freeze, Limits, Plan, Prediction, Release, Row, SelectionReceipt,
    Specification,
};
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

mod values;
use values::*;
mod score;
use score::{score, Score, Unit};
mod compare;
use compare::compare;
#[path = "transfer_independent/oracle_tests.rs"]
mod oracle_tests;
#[path = "transfer_independent/reporter_tests.rs"]
mod reporter_tests;
