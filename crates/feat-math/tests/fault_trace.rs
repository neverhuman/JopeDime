//! Fault reporting of the scalar evaluator (kernel2 WP-1.1 to WP-1.3).
//!
//! - The fault classes: bits, names and severities are append-only (WP-1.1).
//! - The fault tables of WP-1.2 and WP-1.8, row by row, plus a case for every class, for the
//!   boundaries of each guard and for the order in which faults accumulate, through the
//!   interpreter and the compiled form. Values are compared with the legacy entry points, and
//!   with recorded bits where the value is free of libm.
//! - On every (expression, row) of the compatibility corpus, and every element of every row, every
//!   traced form (the interpreter's row, strict row and element calls; the compiled row and
//!   element calls; the compiled map) returns the legacy value bit for bit (0.0 for a legacy
//!   error, as the stage fallback computes it); a Hard fault is raised exactly when the legacy
//!   call returns an error, with the class of that error's kind; and the compiled form reports
//!   exactly the interpreter's faults (WP-1.2, WP-1.3).
//! - The same on 10,000 random expressions from chained edits, on every corpus row (WP-1.3).
//!
//! Inputs come from PR 01's compatibility corpus, `tests/fixtures/compat_freeze/
//! mathexpr_values.v1.json`: its rows and its expressions. The values recorded there are never
//! used as expectations here, because they depend on the host's libm; every check compares two
//! evaluations made by this build.
//!
//! Two ignored probes compare two builds rather than two paths of one build. Run them in release,
//! where inlining is on, with `cargo test --release --locked -p feat-math --test fault_trace --
//! --ignored --nocapture`:
//!
//! - `legacy_value_digest` prints one sha256 over every result the legacy entry points return on
//!   the corpus: the interpreter and the compiled form, in row, element and map context. Two
//!   builds return the same legacy values exactly when they print the same digest.
//! - `legacy_eval_row_throughput` times the legacy entry points on the corpus.

use feat_math::math_expr::{
    CompiledMathExpr, ExpressionScope, FaultClass, FaultSet, FaultSeverity, MathEvalError, MathExpr,
};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::hint::black_box;
use std::time::Instant;

mod cases;
mod compare;
mod corpus;
mod probes;

use cases::{all_cases, Case, POS_ZERO};
use compare::{check_every_form, At, Tallies};
use corpus::{corpus, Corpus};
