//! Direct, fitted `BoundModel` to C++17 lowering.
//!
//! This is deliberately separate from the frozen V1 emitter.  The generated
//! program owns no files and performs no parsing: a persistent host supplies a
//! typed row-major table.  Numeric cells carry their parsed `f64`; categorical
//! cells carry the fitted vocabulary index (`-1` means unknown).  The parallel
//! missing bitmap distinguishes an unknown category from a missing category.
//!
//! The emitter revision, `FITTED_CPP_REVISION`, names what the emitted code
//! computes. Every source carries it in its first line (`emitter-revision=`),
//! and the typed-table codec and the fitted-model container in their
//! `emitter_revision` field. Revision 3 (kernel2 X3) checks each fitted column
//! index a stage reads against the row it reads, and answers status 8 instead of
//! reading out of range (see `emit_abi`); divides the mean and the standard
//! deviation of an empty row by one, as the runtime does; pads a stage's empty
//! output row to `[0.0]` on the legacy-predictor paths, as `ops::finish` does;
//! and encodes a `piecewise_linear` interval no wider than 1e-15 as a step, as
//! the runtime does. Everything else computes revision 2's bits.

use super::calibration::CalibrationModel;
use super::heads::{ExtraNode, V3Head};
use super::input::{
    CategoricalMode, FittedColumn, FittedRowOrder, HistoricalTargetState, SemanticKey,
};
use super::pipeline::{BoundModel, BoundPredictor};
use super::schema::{ColumnKind, RowSemantics};
use super::transforms::{FittedTableOrder, GroupTail, SemanticKey as RowSemanticKey, V3Op};
use super::types::{text_hash, TaskKind};
use crate::boost2::BoostNode;
use crate::gpr::KernelPrim;
use crate::heads::ConditionalRidge;
use crate::heads::{FittedHead, GbmNode};
use crate::ops::FittedOp;
use crate::portfolio::FittedMemberState;
use crate::target_transform::FittedTarget;
use anyhow::{bail, Context, Result};
use serde::Serialize;

#[derive(Clone, Debug)]
pub struct FittedCppArtifact {
    pub source: String,
    pub source_sha256: String,
    pub codec: String,
    pub codec_sha256: String,
}

#[derive(Serialize)]
struct Codec<'a> {
    schema: &'static str,
    emitter_revision: u32,
    schema_sha256: &'a str,
    vocabulary_sha256: &'a str,
    columns: Vec<CodecColumn<'a>>,
    categorical_unknown: i32,
    missing_bitmap: &'static str,
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum CodecColumn<'a> {
    Numeric,
    Categorical { vocabulary: &'a [String] },
}

include!("fitted_cpp/admission.rs");

include!("fitted_cpp/emit_model.rs");

include!("fitted_cpp/literals_runtime.rs");

include!("fitted_cpp/row_transforms.rs");
include!("fitted_cpp/operation_transforms.rs");

include!("fitted_cpp/heads.rs");

include!("fitted_cpp/portfolio.rs");

include!("fitted_cpp/prediction_abi_tests.rs");
