//! Registry-derived conversion witnesses. The historical corpus stays read-only.

use feat_math::kernel::capabilities::Manifest;
use feat_math::kernel::expression_recipe::{emit_cpp17 as emit_expression_cpp17, PROFILE};
use feat_math::kernel::Program;
use feat_math::registry::{generatable_operators, operator_registry};
use feat_math::transpile::{emit, emit_cpp17, supports, supports_cpp17, EmitOptions};
use feat_math::{fit_predict, GenomeIr, Pcg32};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
use std::fs;
use std::path::Path;
use std::process::Command;

const TRAINING_ROWS: usize = 64;

const HISTORICAL: &str = include_str!(
    "../fixtures/compat_freeze/historical_v1/objects/sha256/b5e66d14bdde73e4ff7d69ad65d6e7a8b0772ebc7c532f6a60bb775f4c2ef5c3.json"
);
const EMITTED: &str = include_str!("fixtures/compat_freeze/emitted_bytes.v1.json");
const MODELS: &str = include_str!("fixtures/compat_freeze/model_serde.v1.json");

mod common;
use common::*;

include!("conversion_wire_matrix/frozen_registry.rs");
include!("conversion_wire_matrix/wire_recipes.rs");
include!("conversion_wire_matrix/seeded_recipes.rs");
include!("conversion_wire_matrix/genome_forms.rs");
