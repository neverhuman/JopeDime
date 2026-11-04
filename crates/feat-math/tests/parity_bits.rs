//! Bit-exact prediction parity between the runtime and three emitted programs (kernel2 X0).
//!
//! Every program in [`PROGRAMS`] is fitted by the runtime (`feat_math::fit`, then
//! `Model::predict`), which is the reference. Three lanes must reproduce every prediction bit for
//! bit, compared with `f64::to_bits`, so the sign of zero counts and a NaN prediction is always
//! reported:
//!
//! * `rust`: the emitted Rust program, compiled with `rustc -O` as `golden.rs` does;
//! * `python`: the emitted Python program, run with `python3`;
//! * `cpp17-fitted`: `emit_fitted_cpp17` of `fit_legacy_numeric_bound`, compiled with
//!   `-std=c++17 -O2 -ffp-contract=off -fno-fast-math` together with [`DRIVER_CPP`], a small
//!   driver this test writes. The driver reads the input bits from a file at run time, so no
//!   prediction can be folded at compile time, and prints each prediction's bits.
//!
//! The programs execute every `MathExpr` function, row variable and operator and reach the
//! division guard (checked by `parity_bits_programs_execute_every_function_variable_and_operator`),
//! in closed-form programs and in `map`/`derive` stages. The prediction rows are ordinary values plus the scalar boundary
//! corpus in four row shapes (the value in every column, or in one column with ordinary values in
//! the others) and rows that mix the two signed zeros. An empty-width stage is not reachable:
//! every stage output is padded to at least one column (`ops.rs`, `finish`) and both emitted CSV
//! readers skip blank lines, so a row without features cannot be expressed.
//!
//! Non-finite inputs. The runtime's closed-form predict evaluates raw inputs, while every
//! emitted program zero-fills a non-finite input at its boundary (the CSV readers and the fitted
//! C++ `encode_row`). That single input-contract difference is recorded once, by
//! `sym_input_contract`; the other closed-form programs receive finite rows only, and the
//! pipeline programs, which sanitize their inputs in every lane, receive every row.
//!
//! Known divergences at the base commit are listed in `fixtures/parity_allowlist.v1.json` with
//! the fix item that removes them. A lane fails on any divergence that is not listed and on any
//! listed entry of its language that no longer reproduces.
//!
//! With `JOPE_PARITY_STRICT=1` a missing `rustc`, `python3` or C++ compiler (`CXX`, default
//! `g++`) fails its lane. Without it the lane prints a `parity_bits SKIP` line and returns. The
//! `cpp17-fitted` lane treats a compiler other than g++ 13.3, the one its allowlist entries were
//! recorded with, the same way ([`FITTED_CXX_VERSION`]).
//!
//! Batching is over inputs: each lane runs each program once over its whole input set. Programs
//! are compiled one per process, concurrently, because every artifact is a standalone program
//! with its own `main` (Rust and Python) or its own `extern "C"` entry points (C++); merging them
//! would mean rewriting emitted text, and one Python process per batch would need a new Python
//! driver.
//!
//! Scratch files live under `TMPDIR` when it is set, otherwise under Cargo's target temporary
//! directory, in a directory removed at the end of the test (kept, and its path printed, with
//! `JOPE_PARITY_KEEP_SCRATCH=1`).
//!
//! The boundary corpus is the shared `tests/fixtures/math_boundary/v1.json` ([`BOUNDARY_V1`],
//! read with `include_str!`), the one `compat_freeze` checks. The allowlist entries were recorded
//! against its bytes, so their sha256 is pinned here too ([`BOUNDARY_V1_SHA256`]): a corpus
//! change fails this test as well as `compat_freeze`. Row ids are built from entry names.

use feat_math::cxx::CxxStd;
use feat_math::dsl::ParamValue;
use feat_math::kernel::Program as KernelProgram;
use feat_math::math_expr::{BinOp, Expr, Func, MathExpr, UnOp, Var};
use feat_math::transpile::{emit, supports, EmitOptions};
use feat_math::{emit_fitted_cpp17, fit, fit_legacy_numeric_bound, GenomeIr, Pcg32};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// Which rows a program is evaluated on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Inputs {
    /// Ordinary rows, every finite boundary row and the mixed signed-zero rows.
    Finite,
    /// [`Inputs::Finite`] plus every non-finite boundary row.
    All,
    /// Ordinary rows plus each non-finite boundary value in column 0.
    NonFiniteColumnZero,
}

struct Program {
    name: &'static str,
    source: &'static str,
    inputs: Inputs,
}

/// Every function argument depends on an input column, so no compiler can fold a transcendental
/// call on a constant (g++ folds with MPFR, the runtime calls the platform libm).
const PROGRAMS: &[Program] = &[
    Program {
        name: "sym_sqrt_sin_cos",
        source: "(sym raw \"sqrt(c0) + sin(c1) - cos(c2)\")",
        inputs: Inputs::Finite,
    },
    Program {
        name: "sym_tanh_sigmoid_log1p",
        source: "(sym raw \"tanh(c0) * sigmoid(c1) + log1p(c2)\")",
        inputs: Inputs::Finite,
    },
    Program {
        name: "sym_exp_log_div",
        source: "(sym raw \"exp(c0) / (c1 * c1 + 1) - log(c2)\")",
        inputs: Inputs::Finite,
    },
    // c1 / c0 reaches the division guard: the division_guard rows put 1e-12 and its neighbours in
    // column 0, where a quotient (±60 after the prediction clamp) and the guard's 0 (1.5 with the
    // fill) differ, so a lane that guards with <= instead of < diverges.
    Program {
        name: "sym_div_guard",
        source: "(sym raw \"c1 / c0 + c2\")",
        inputs: Inputs::Finite,
    },
    Program {
        name: "sym_asinh_atan_erf",
        source: "(sym raw \"asinh(c0) + atan(c1) * erf(c2)\")",
        inputs: Inputs::Finite,
    },
    Program {
        name: "sym_sign_abs_floor_round",
        source: "(sym raw \"sign(c0) * abs(c1) + floor(c1) - round(c2)\")",
        inputs: Inputs::Finite,
    },
    Program {
        name: "sym_min_max_step_clip",
        source: "(sym raw \"min(c0, c1) - max(c1, c2) + step(c0, c2) + clip(c0, c1, c2)\")",
        inputs: Inputs::Finite,
    },
    Program {
        name: "sym_pow_neg",
        source: "(sym raw \"-(abs(c0) ^ c1) + abs(c2) ^ c0\")",
        inputs: Inputs::Finite,
    },
    Program {
        name: "sym_row_summaries",
        source: "(sym raw \"row_sum - row_l1 + row_l2 + row_mean * row_std + row_min - row_max\")",
        inputs: Inputs::Finite,
    },
    // sign(v ^ -1) is -1, 0 or 1 and turns a signed zero into an infinity of the same sign, so
    // these two programs expose the sign of a zero intermediate; the weights 1, 3, 9 and 27 name
    // which term moved.
    Program {
        name: "sym_zero_sign_row_reductions",
        source: "(sym raw \"sign(row_sum ^ (-1)) + 3 * sign(row_mean ^ (-1)) + 9 * sign(row_min ^ (-1)) + 27 * sign(row_max ^ (-1))\")",
        inputs: Inputs::Finite,
    },
    Program {
        name: "sym_zero_sign_min_max",
        source: "(sym raw \"sign(min(c0, c1) ^ (-1)) + 3 * sign(max(c0, c1) ^ (-1)) + 9 * sign(min(c1, c2) ^ (-1)) + 27 * sign(max(c1, c2) ^ (-1))\")",
        inputs: Inputs::Finite,
    },
    Program {
        name: "sym_affine_exp_log1p_cos",
        source: "(sym affine \"exp(c0 / 4) - log1p(abs(c1)) + cos(c2)\")",
        inputs: Inputs::Finite,
    },
    Program {
        name: "sym_input_contract",
        source: "(sym raw \"c0 + c1 + c2 + 1\")",
        inputs: Inputs::NonFiniteColumnZero,
    },
    Program {
        name: "pipe_map_elementwise",
        source: "(pipe X (map expr=\"sqrt(abs(x)) * sign(x) + sin(x) * cos(c0)\") (ridge))",
        inputs: Inputs::All,
    },
    Program {
        name: "pipe_derive_row_summaries",
        source: "(pipe X (derive expr=\"row_sum + row_l1 - row_l2\" append=true) (derive expr=\"row_min * row_max + row_mean - row_std\" append=true) (ridge))",
        inputs: Inputs::All,
    },
    Program {
        name: "pipe_map_derive_elasticnet",
        source: "(pipe X (map expr=\"clip(x, row_min, row_max) + step(x, row_mean)\" append=true) (derive expr=\"erf(c0) + asinh(c1) - atan(c2) + tanh(c3)\" append=true) (elasticnet alpha=0.01 l1_ratio=0.5 sweeps=30))",
        inputs: Inputs::All,
    },
    Program {
        name: "pipe_derive_exp_log_pow",
        source: "(pipe X (derive expr=\"exp(c0) - log(abs(c1)) + log1p(abs(c2)) + abs(c0) ^ c1\" append=true) (ridge))",
        inputs: Inputs::All,
    },
    // The division guard inside a stage, where no prediction clamp hides the quotient.
    Program {
        name: "pipe_derive_div_guard",
        source: "(pipe X (derive expr=\"c1 / c0\" append=true) (ridge))",
        inputs: Inputs::All,
    },
    Program {
        name: "pipe_derive_zero_sign",
        source: "(pipe X (derive expr=\"sign(row_sum ^ (-1)) + 3 * sign(row_mean ^ (-1))\" append=true) (ridge))",
        inputs: Inputs::All,
    },
];

/// Feature columns of every training and prediction row.
const WIDTH: usize = 3;
const TRAIN_ROWS: usize = 64;
const ORDINARY_ROWS: usize = 8;
/// The other columns of a row that carries a boundary value in one column.
const FILL: [f64; WIDTH] = [0.75, -1.25, 1.5];
/// Concurrent compile-and-run jobs per lane.
const MAX_JOBS: usize = 8;

/// The shared scalar boundary corpus.
const BOUNDARY_V1: &str = include_str!("fixtures/math_boundary/v1.json");

/// sha256 of the [`BOUNDARY_V1`] bytes the allowlist entries were recorded against.
const BOUNDARY_V1_SHA256: &str = "ed1dfe9c31edc68402f1c7c69a51ec25814475c69b3c8c887fe385d1f2cf423a";

const ALLOWLIST: &str = include_str!("fixtures/parity_allowlist.v1.json");
const ALLOWLIST_SCHEMA: &str = "feat-math.parity-allowlist.v1";
const ALLOWLIST_STATUS: &str = "pre-existing-at-base";
/// The `cpp17-fitted` entries were recorded with g++ 13.3, and which signed zero `std::fmin` and
/// `std::fmax` return on a tie depends on the compiler's operand order. The lane therefore
/// requires the compiler version to contain this, the rule the product's fitted C++ path applies
/// (`advanced_v3::fitted_cpp_runtime`: "fitted model compilation requires g++ 13.3").
const FITTED_CXX_VERSION: &str = "13.3";

/// Reads `rows columns` and then one hexadecimal bit pattern per value from the file named on
/// its command line, and prints one bit pattern per prediction.
const DRIVER_CPP: &str = r#"#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

extern "C" int jope_fitted_predict(const double* values, std::size_t values_len,
                                   const std::uint8_t* missing, std::size_t missing_len,
                                   std::size_t rows, std::size_t columns, double* output,
                                   std::size_t output_len) noexcept;

int main(int argc, char** argv) {
  if (argc != 2) return 90;
  std::FILE* input = std::fopen(argv[1], "r");
  if (input == nullptr) return 91;
  unsigned long long rows = 0;
  unsigned long long columns = 0;
  if (std::fscanf(input, "%llu %llu", &rows, &columns) != 2) return 92;
  std::vector<double> values(static_cast<std::size_t>(rows * columns));
  for (double& value : values) {
    unsigned long long bits = 0;
    if (std::fscanf(input, "%llx", &bits) != 1) return 93;
    const std::uint64_t exact = static_cast<std::uint64_t>(bits);
    std::memcpy(&value, &exact, sizeof value);
  }
  std::fclose(input);
  std::vector<std::uint8_t> missing(values.size(), 0);
  std::vector<double> output(static_cast<std::size_t>(rows), 0.0);
  const int status = jope_fitted_predict(values.data(), values.size(), missing.data(),
                                         missing.size(), static_cast<std::size_t>(rows),
                                         static_cast<std::size_t>(columns), output.data(),
                                         output.size());
  if (status != 0) return 100 + status;
  for (const double value : output) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof bits);
    std::printf("%016llx\n", static_cast<unsigned long long>(bits));
  }
  return 0;
}
"#;

// ------------------------------------------------------------------------------------------------
// Inputs
// ------------------------------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Corpus {
    schema: String,
    note: String,
    groups: Vec<String>,
    entries: Vec<CorpusEntry>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CorpusEntry {
    name: String,
    group: String,
    bits: String,
    value_repr: String,
}

struct InputRow {
    id: String,
    values: Vec<f64>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lane {
    Rust,
    Python,
    Cpp17Fitted,
}

impl Lane {
    const ALL: [Lane; 3] = [Lane::Rust, Lane::Python, Lane::Cpp17Fitted];

    fn id(self) -> &'static str {
        match self {
            Lane::Rust => "rust",
            Lane::Python => "python",
            Lane::Cpp17Fitted => "cpp17-fitted",
        }
    }

    fn tool(self) -> String {
        match self {
            Lane::Rust => "rustc".to_owned(),
            Lane::Python => "python3".to_owned(),
            Lane::Cpp17Fitted => feat_math::cxx::compiler(),
        }
    }
}
struct Shared {
    train_csv: PathBuf,
    features_csv: BTreeMap<&'static str, PathBuf>,
    features_bits: BTreeMap<&'static str, PathBuf>,
    train_x: Vec<Vec<f64>>,
    train_y: Vec<f64>,
}
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Divergence {
    program: String,
    language: String,
    input: String,
    runtime_bits: u64,
    emitted_bits: u64,
}

impl Divergence {
    fn describe(&self) -> String {
        format!(
            "{{\"program\": \"{}\", \"language\": \"{}\", \"input\": \"{}\", \"runtime_bits\": \"{}\", \"emitted_bits\": \"{}\"}}  (runtime {:?}, emitted {:?})",
            self.program,
            self.language,
            self.input,
            hex(self.runtime_bits),
            hex(self.emitted_bits),
            f64::from_bits(self.runtime_bits),
            f64::from_bits(self.emitted_bits)
        )
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Allowlist {
    schema: String,
    note: String,
    entries: Vec<AllowlistEntry>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AllowlistEntry {
    program: String,
    language: String,
    input: String,
    runtime_bits: String,
    emitted_bits: String,
    status: String,
    fix_item: String,
    reason: String,
}

#[path = "parity_bits/allowlist.rs"]
mod allowlist;
#[path = "parity_bits/input.rs"]
mod input;
#[path = "parity_bits/programs.rs"]
mod programs;
#[path = "parity_bits/runner.rs"]
mod runner;
pub(crate) use allowlist::*;
pub(crate) use input::*;
pub(crate) use programs::*;
pub(crate) use runner::*;
