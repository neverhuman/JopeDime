//! W9-T03/T04 raw-input and hostile-string contract of the emitted public backends.
//!
//! Every case runs through the actual emitted source (Python, Rust, JavaScript and the
//! C++17/C++20 library behind its declared Python typed-table host) and is compared
//! with the native fitted predictor on the same raw rows. Fixtures are fitted once
//! through the registered Rust fitter and only read afterwards; no backend refits.
//! This is fixture proof of the emitted boundary, not a released-package qualification.
use super::super::data::raw_table;
use super::super::methods::{fit_rust, rust_methods, FittedRustArtifact, RustFitRequest};
use super::*;
use feat_math::advanced_v3::{DataSchema, RowSemantics};
use jope_pipeline::experiment::{Column, ColumnKind as Kind};
use serde_json::{json, Value};
use std::path::Path;
use std::process::Command;

#[path = "inference_contract_tests/contracts.rs"]
mod contracts;
#[path = "inference_contract_tests/fixtures.rs"]
mod fixtures;
#[path = "inference_contract_tests/runner.rs"]
mod runner;

pub(super) use fixtures::{case, hostile_categories};
pub(crate) use fixtures::{
    columns, fit, fit_method, hostile_fixture, raw_cases, Case, HOSTILE_CATEGORIES,
};
pub(super) use runner::{assert_tallies, scratch};
pub(crate) use runner::{compare, divergences, execute};
