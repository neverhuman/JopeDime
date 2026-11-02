//! Repeated emission and real compilation of bounded scalar GP trees.
use feat_math::kernel::numerical_policy::NumericalPolicy;
use feat_math::kernel::scalar_program::{
    ScalarCarrier, ScalarExpressionProgram, ScalarFunction, ScalarLanguage, ScalarNode,
};
use feat_math::math_expr::{BinOp, UnOp, Var};
use feat_math::Pcg32;
use std::{fs, path::Path, process::Command};

const LANGUAGES: [ScalarLanguage; 3] = [
    ScalarLanguage::Python,
    ScalarLanguage::Rust,
    ScalarLanguage::Cpp17,
];
const SEED: u64 = 0x4750_454d_4954;
const WIDTH: usize = 3;
const LAMBDA: f64 = 0.02;
const ABS_TOLERANCE: f64 = 1e-12;
const REL_TOLERANCE: f64 = 1e-12;

include!("gp_emitter_determinism/corpus.rs");
include!("gp_emitter_determinism/process.rs");
include!("gp_emitter_determinism/differential.rs");
include!("gp_emitter_determinism/comparisons.rs");
