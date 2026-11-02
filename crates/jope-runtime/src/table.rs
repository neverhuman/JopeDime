//! Local table ingress (SPECIFICATION.md §13.1).
//!
//! The raw-cell encoding ([`encode_cell`], [`raw_table`]) and the fitting-input digest moved here
//! unchanged from `jope_pipeline::prepared_data`, which re-exports them: the appliance hands
//! `feat_math` exactly the cells the frozen panel does.
//!
//! The CSV contract, stated once:
//!
//! * The first record is the header. Names must be non-empty and unique; a UTF-8 byte-order mark
//!   is ignored. Every record has the header's width.
//! * An empty cell (after trimming ASCII whitespace) is missing; it is the only missing marker.
//!   Literal `NA`, `null` or `?` are values, which keeps categories injective.
//! * The target column is named explicitly and never used as a feature. A missing target fails.
//! * A feature column is numeric when every present cell parses as a finite number, otherwise
//!   categorical. Columns keep their header order. Constant and all-missing columns are kept; the
//!   input stage's median/indicator policy handles them.
//! * `auto` task: exactly two distinct target labels is binary; otherwise all-numeric targets are
//!   regression and anything else is multiclass. Integer-coded classes with more than two labels
//!   need an explicit `--task multiclass`.
//! * Class order is numeric when every label is a number, otherwise byte-lexicographic; for binary
//!   tasks the second class is the positive one.
//! * Prediction input matches feature columns by name in any order. A column the model does not
//!   know is refused, except the training target's name, which is ignored and reported.
//! * *Auxiliary* columns named by a regression objective (prices, unit costs, exposures) must be
//!   numeric and present in every training row. They score held-out predictions only: they are
//!   never features, and prediction input may carry them (they are ignored and reported).
use crate::limits::Limits;
use crate::task::{TaskChoice, TaskKind};
use crate::Failure;
use anyhow::{ensure, Context, Result};
use feat_math::advanced_v3::{ColumnSpec, DataSchema, RawTable, RowSemantics};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

pub use feat_math::advanced_v3::ColumnKind;
/// A column's name and kind, as a fitted file records it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ColumnRecord {
    pub name: String,
    pub kind: ColumnKind,
}

mod ingress;
mod training;

#[cfg(test)]
mod tests;

pub use ingress::{
    encode_cell, fitting_input_sha256, iid_schema, parse_csv, raw_table, read_csv, CsvTable,
    CATEGORY_PREFIX, FITTING_INPUT, RAW_CELL_ENCODING,
};
pub use training::{prediction, prediction_ignoring, training, training_with_auxiliary, Training};
