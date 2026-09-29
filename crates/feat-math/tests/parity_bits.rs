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
//! The boundary corpus is an exact local copy of `tests/fixtures/math_boundary/v1.json` from
//! kernel2 PR 01 ([`BOUNDARY_V1`], checked against that file's sha256). After this branch is
//! integrated with PR 01 the constant becomes `include_str!("fixtures/math_boundary/v1.json")`;
//! row ids are built from entry names, so no allowlist key changes.

use feat_math::cxx::CxxStd;
use feat_math::dsl::ParamValue;
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

/// Exact copy of kernel2 PR 01's `crates/feat-math/tests/fixtures/math_boundary/v1.json`.
const BOUNDARY_V1: &str = r#"{
  "schema": "feat-math.math-boundary.v1",
  "note": "Scalar boundary inputs shared by the Rust tests and the emitted-artifact harnesses (workplan WP-0.3). bits is the IEEE-754 binary64 pattern; value_repr is Rust's shortest round-trip form and parses back to exactly bits, except that NaN payloads live only in bits. Neighbours are exact: bits plus or minus one. A value that the workplan lists in two groups appears once, under the first of them. Generated by crates/feat-math/tests/compat_freeze.rs, which also proves the file equals its definition.",
  "groups": ["specials", "division_guard", "log_guard", "log1p_guard", "unity_small_integers", "rounding", "exp_sigmoid_clamps", "magnitude_clamp", "trig_erf_mid"],
  "entries": [
    {"name": "pos_zero", "group": "specials", "bits": "0x0000000000000000", "value_repr": "0.0"},
    {"name": "neg_zero", "group": "specials", "bits": "0x8000000000000000", "value_repr": "-0.0"},
    {"name": "pos_inf", "group": "specials", "bits": "0x7ff0000000000000", "value_repr": "inf"},
    {"name": "neg_inf", "group": "specials", "bits": "0xfff0000000000000", "value_repr": "-inf"},
    {"name": "nan_quiet", "group": "specials", "bits": "0x7ff8000000000000", "value_repr": "NaN"},
    {"name": "nan_payload", "group": "specials", "bits": "0x7ff8000000000001", "value_repr": "NaN"},
    {"name": "div_guard_pos", "group": "division_guard", "bits": "0x3d719799812dea11", "value_repr": "1e-12"},
    {"name": "div_guard_pos_toward_zero", "group": "division_guard", "bits": "0x3d719799812dea10", "value_repr": "9.999999999999998e-13"},
    {"name": "div_guard_pos_away_from_zero", "group": "division_guard", "bits": "0x3d719799812dea12", "value_repr": "1.0000000000000002e-12"},
    {"name": "div_guard_neg", "group": "division_guard", "bits": "0xbd719799812dea11", "value_repr": "-1e-12"},
    {"name": "div_guard_neg_toward_zero", "group": "division_guard", "bits": "0xbd719799812dea10", "value_repr": "-9.999999999999998e-13"},
    {"name": "div_guard_neg_away_from_zero", "group": "division_guard", "bits": "0xbd719799812dea12", "value_repr": "-1.0000000000000002e-12"},
    {"name": "log_guard", "group": "log_guard", "bits": "0x01a56e1fc2f8f359", "value_repr": "1e-300"},
    {"name": "log_guard_toward_zero", "group": "log_guard", "bits": "0x01a56e1fc2f8f358", "value_repr": "9.999999999999999e-301"},
    {"name": "log_guard_away_from_zero", "group": "log_guard", "bits": "0x01a56e1fc2f8f35a", "value_repr": "1.0000000000000002e-300"},
    {"name": "min_subnormal", "group": "log_guard", "bits": "0x0000000000000001", "value_repr": "5e-324"},
    {"name": "min_normal", "group": "log_guard", "bits": "0x0010000000000000", "value_repr": "2.2250738585072014e-308"},
    {"name": "neg_one", "group": "log1p_guard", "bits": "0xbff0000000000000", "value_repr": "-1.0"},
    {"name": "neg_one_toward_zero", "group": "log1p_guard", "bits": "0xbfefffffffffffff", "value_repr": "-0.9999999999999999"},
    {"name": "neg_one_two_ulps_toward_zero", "group": "log1p_guard", "bits": "0xbfeffffffffffffe", "value_repr": "-0.9999999999999998"},
    {"name": "neg_one_away_from_zero", "group": "log1p_guard", "bits": "0xbff0000000000001", "value_repr": "-1.0000000000000002"},
    {"name": "neg_half", "group": "log1p_guard", "bits": "0xbfe0000000000000", "value_repr": "-0.5"},
    {"name": "pos_half", "group": "log1p_guard", "bits": "0x3fe0000000000000", "value_repr": "0.5"},
    {"name": "pos_one", "group": "unity_small_integers", "bits": "0x3ff0000000000000", "value_repr": "1.0"},
    {"name": "pos_two", "group": "unity_small_integers", "bits": "0x4000000000000000", "value_repr": "2.0"},
    {"name": "neg_two", "group": "unity_small_integers", "bits": "0xc000000000000000", "value_repr": "-2.0"},
    {"name": "pos_three", "group": "unity_small_integers", "bits": "0x4008000000000000", "value_repr": "3.0"},
    {"name": "neg_three", "group": "unity_small_integers", "bits": "0xc008000000000000", "value_repr": "-3.0"},
    {"name": "pos_one_and_half", "group": "rounding", "bits": "0x3ff8000000000000", "value_repr": "1.5"},
    {"name": "neg_one_and_half", "group": "rounding", "bits": "0xbff8000000000000", "value_repr": "-1.5"},
    {"name": "pos_two_and_half", "group": "rounding", "bits": "0x4004000000000000", "value_repr": "2.5"},
    {"name": "below_half", "group": "rounding", "bits": "0x3fdfffffffffffff", "value_repr": "0.49999999999999994"},
    {"name": "two_pow_52_plus_one", "group": "rounding", "bits": "0x4330000000000001", "value_repr": "4503599627370497.0"},
    {"name": "pos_27", "group": "exp_sigmoid_clamps", "bits": "0x403b000000000000", "value_repr": "27.0"},
    {"name": "neg_27", "group": "exp_sigmoid_clamps", "bits": "0xc03b000000000000", "value_repr": "-27.0"},
    {"name": "ln_1e12", "group": "exp_sigmoid_clamps", "bits": "0x403ba18a998fffa0", "value_repr": "27.631021115928547"},
    {"name": "pos_27_64", "group": "exp_sigmoid_clamps", "bits": "0x403ba3d70a3d70a4", "value_repr": "27.64"},
    {"name": "pos_28", "group": "exp_sigmoid_clamps", "bits": "0x403c000000000000", "value_repr": "28.0"},
    {"name": "neg_28", "group": "exp_sigmoid_clamps", "bits": "0xc03c000000000000", "value_repr": "-28.0"},
    {"name": "pos_30", "group": "exp_sigmoid_clamps", "bits": "0x403e000000000000", "value_repr": "30.0"},
    {"name": "neg_30", "group": "exp_sigmoid_clamps", "bits": "0xc03e000000000000", "value_repr": "-30.0"},
    {"name": "pos_31", "group": "exp_sigmoid_clamps", "bits": "0x403f000000000000", "value_repr": "31.0"},
    {"name": "neg_31", "group": "exp_sigmoid_clamps", "bits": "0xc03f000000000000", "value_repr": "-31.0"},
    {"name": "max_abs", "group": "magnitude_clamp", "bits": "0x426d1a94a2000000", "value_repr": "1000000000000.0"},
    {"name": "max_abs_toward_zero", "group": "magnitude_clamp", "bits": "0x426d1a94a1ffffff", "value_repr": "999999999999.9999"},
    {"name": "max_abs_away_from_zero", "group": "magnitude_clamp", "bits": "0x426d1a94a2000001", "value_repr": "1000000000000.0001"},
    {"name": "neg_max_abs", "group": "magnitude_clamp", "bits": "0xc26d1a94a2000000", "value_repr": "-1000000000000.0"},
    {"name": "neg_max_abs_toward_zero", "group": "magnitude_clamp", "bits": "0xc26d1a94a1ffffff", "value_repr": "-999999999999.9999"},
    {"name": "neg_max_abs_away_from_zero", "group": "magnitude_clamp", "bits": "0xc26d1a94a2000001", "value_repr": "-1000000000000.0001"},
    {"name": "pos_1e154", "group": "magnitude_clamp", "bits": "0x5fe7dddf6b095ff1", "value_repr": "1e154"},
    {"name": "neg_1e154", "group": "magnitude_clamp", "bits": "0xdfe7dddf6b095ff1", "value_repr": "-1e154"},
    {"name": "sqrt_max", "group": "magnitude_clamp", "bits": "0x5fefffffffffffff", "value_repr": "1.3407807929942596e154"},
    {"name": "neg_sqrt_max", "group": "magnitude_clamp", "bits": "0xdfefffffffffffff", "value_repr": "-1.3407807929942596e154"},
    {"name": "pos_1_34e154", "group": "magnitude_clamp", "bits": "0x5feffb3abd82527b", "value_repr": "1.34e154"},
    {"name": "neg_1_34e154", "group": "magnitude_clamp", "bits": "0xdfeffb3abd82527b", "value_repr": "-1.34e154"},
    {"name": "pos_1_35e154", "group": "magnitude_clamp", "bits": "0x5ff01c2a01d98729", "value_repr": "1.35e154"},
    {"name": "neg_1_35e154", "group": "magnitude_clamp", "bits": "0xdff01c2a01d98729", "value_repr": "-1.35e154"},
    {"name": "pos_1e155", "group": "magnitude_clamp", "bits": "0x601dd55745cbb7ed", "value_repr": "1e155"},
    {"name": "neg_1e155", "group": "magnitude_clamp", "bits": "0xe01dd55745cbb7ed", "value_repr": "-1e155"},
    {"name": "max_finite", "group": "magnitude_clamp", "bits": "0x7fefffffffffffff", "value_repr": "1.7976931348623157e308"},
    {"name": "neg_max_finite", "group": "magnitude_clamp", "bits": "0xffefffffffffffff", "value_repr": "-1.7976931348623157e308"},
    {"name": "tenth", "group": "trig_erf_mid", "bits": "0x3fb999999999999a", "value_repr": "0.1"},
    {"name": "tiny", "group": "trig_erf_mid", "bits": "0x3e45798ee2308c3a", "value_repr": "1e-8"},
    {"name": "six", "group": "trig_erf_mid", "bits": "0x4018000000000000", "value_repr": "6.0"},
    {"name": "pi", "group": "trig_erf_mid", "bits": "0x400921fb54442d18", "value_repr": "3.141592653589793"},
    {"name": "neg_pi", "group": "trig_erf_mid", "bits": "0xc00921fb54442d18", "value_repr": "-3.141592653589793"}
  ]
}
"#;

/// sha256 of PR 01's `tests/fixtures/math_boundary/v1.json`.
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

fn parse_bits(text: &str) -> Option<u64> {
    let digits = text.strip_prefix("0x")?;
    if digits.len() != 16 || !digits.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    u64::from_str_radix(digits, 16).ok()
}

fn hex(bits: u64) -> String {
    format!("0x{bits:016x}")
}

/// `(name, value)` for every corpus entry, in file order.
fn boundary_values() -> Vec<(String, f64)> {
    let corpus: Corpus = serde_json::from_str(BOUNDARY_V1).expect("boundary corpus parses");
    corpus
        .entries
        .into_iter()
        .map(|entry| {
            let bits = parse_bits(&entry.bits)
                .unwrap_or_else(|| panic!("{}: bad bits {}", entry.name, entry.bits));
            (entry.name, f64::from_bits(bits))
        })
        .collect()
}

struct InputRow {
    id: String,
    values: Vec<f64>,
}

fn ordinary_rows() -> Vec<InputRow> {
    let mut rng = Pcg32::new(20_260_930, 11);
    (0..ORDINARY_ROWS)
        .map(|index| InputRow {
            id: format!("ordinary:{index:02}"),
            values: (0..WIDTH).map(|_| rng.next_f64() * 6.0 - 3.0).collect(),
        })
        .collect()
}

/// The value in every column (`all:`), or in one column with [`FILL`] in the others (`c0:` ..).
fn shaped_rows(name: &str, value: f64) -> Vec<InputRow> {
    let mut rows = vec![InputRow {
        id: format!("all:{name}"),
        values: vec![value; WIDTH],
    }];
    for column in 0..WIDTH {
        let mut values = FILL.to_vec();
        values[column] = value;
        rows.push(InputRow {
            id: format!("c{column}:{name}"),
            values,
        });
    }
    rows
}

/// Rows whose entries are all zeros of mixed sign: which zero a sum, a minimum or a maximum
/// returns depends on the order its implementation visits them in.
fn mixed_zero_rows() -> Vec<InputRow> {
    let (p, n) = (0.0_f64, -0.0_f64);
    [
        ("zeros:pos_pos_neg", [p, p, n]),
        ("zeros:neg_neg_pos", [n, n, p]),
        ("zeros:pos_neg_pos", [p, n, p]),
        ("zeros:neg_pos_neg", [n, p, n]),
    ]
    .into_iter()
    .map(|(id, values)| InputRow {
        id: id.to_owned(),
        values: values.to_vec(),
    })
    .collect()
}

fn input_rows(inputs: Inputs) -> Vec<InputRow> {
    let corpus = boundary_values();
    let mut rows = ordinary_rows();
    match inputs {
        Inputs::Finite | Inputs::All => {
            for (name, value) in &corpus {
                if value.is_finite() || inputs == Inputs::All {
                    rows.extend(shaped_rows(name, *value));
                }
            }
            rows.extend(mixed_zero_rows());
        }
        Inputs::NonFiniteColumnZero => {
            for (name, value) in corpus.iter().filter(|(_, value)| !value.is_finite()) {
                rows.extend(
                    shaped_rows(name, *value)
                        .into_iter()
                        .filter(|row| row.id.starts_with("c0:")),
                );
            }
        }
    }
    let ids: BTreeSet<&str> = rows.iter().map(|row| row.id.as_str()).collect();
    assert_eq!(ids.len(), rows.len(), "duplicate input row id");
    rows
}

/// Finite training rows (every lane fits on them; the fitted C++ binding requires finite data).
fn training_data() -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = Pcg32::new(20_260_929, 7);
    let mut x = Vec::with_capacity(TRAIN_ROWS);
    let mut y = Vec::with_capacity(TRAIN_ROWS);
    for _ in 0..TRAIN_ROWS {
        let row: Vec<f64> = (0..WIDTH).map(|_| rng.next_f64() * 4.0 - 2.0).collect();
        let target =
            1.3 * row[0] - 0.7 * row[1] + 0.4 * (2.0 * row[2]).sin() + 0.1 * rng.gaussian();
        x.push(row);
        y.push(target);
    }
    (x, y)
}

fn csv_line(values: &[f64]) -> String {
    values
        .iter()
        .map(|value| format!("{value:?}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn write_train_csv(path: &Path, x: &[Vec<f64>], y: &[f64]) {
    let mut text = String::new();
    for (row, target) in x.iter().zip(y) {
        let mut cells = row.clone();
        cells.push(*target);
        text.push_str(&csv_line(&cells));
        text.push('\n');
    }
    fs::write(path, text).unwrap();
}

fn write_features_csv(path: &Path, rows: &[InputRow]) {
    let mut text = String::new();
    for row in rows {
        text.push_str(&csv_line(&row.values));
        text.push('\n');
    }
    fs::write(path, text).unwrap();
}

fn write_features_bits(path: &Path, rows: &[InputRow]) {
    let mut text = format!("{} {WIDTH}\n", rows.len());
    for row in rows {
        for value in &row.values {
            text.push_str(&format!("{:016x}\n", value.to_bits()));
        }
    }
    fs::write(path, text).unwrap();
}

// ------------------------------------------------------------------------------------------------
// Lanes
// ------------------------------------------------------------------------------------------------

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

fn strict() -> bool {
    std::env::var_os("JOPE_PARITY_STRICT").is_some_and(|value| value == "1")
}

/// The lane's tool version, or `None` after a loud skip (a failure under `JOPE_PARITY_STRICT=1`)
/// when the tool is missing or, for `cpp17-fitted`, is not the compiler the allowlist was recorded
/// with ([`FITTED_CXX_VERSION`]).
fn toolchain(lane: Lane) -> Option<String> {
    let version = match lane {
        Lane::Cpp17Fitted => feat_math::cxx::version(),
        Lane::Rust | Lane::Python => {
            Command::new(lane.tool())
                .arg("--version")
                .output()
                .map(|output| {
                    String::from_utf8_lossy(&output.stdout)
                        .lines()
                        .next()
                        .unwrap_or_default()
                        .to_owned()
                })
        }
    };
    match version {
        Ok(version) if lane == Lane::Cpp17Fitted && !version.contains(FITTED_CXX_VERSION) => {
            assert!(
                !strict(),
                "parity_bits: the {} allowlist entries were recorded with g++ {FITTED_CXX_VERSION}, \
                 and {:?} reports {version:?}; the comparison would report compiler differences \
                 as divergences, so JOPE_PARITY_STRICT=1 needs g++ {FITTED_CXX_VERSION} (CXX)",
                lane.id(),
                lane.tool()
            );
            eprintln!(
                "parity_bits SKIP: {} lane, {:?} reports {version:?}, not the g++ \
                 {FITTED_CXX_VERSION} its allowlist entries were recorded with ({} programs not \
                 compared)",
                lane.id(),
                lane.tool(),
                PROGRAMS.len()
            );
            None
        }
        Ok(version) => Some(version),
        Err(error) if error.kind() == ErrorKind::NotFound => {
            assert!(
                !strict(),
                "parity_bits: {} tool {:?} is unavailable and JOPE_PARITY_STRICT=1",
                lane.id(),
                lane.tool()
            );
            eprintln!(
                "parity_bits SKIP: {} lane, tool {:?} is unavailable ({} programs not compared)",
                lane.id(),
                lane.tool(),
                PROGRAMS.len()
            );
            None
        }
        Err(error) => panic!(
            "parity_bits: cannot start {} tool {:?}: {error}",
            lane.id(),
            lane.tool()
        ),
    }
}

/// `TMPDIR` when it is set, otherwise Cargo's per-target temporary directory; never an implicit
/// system default.
fn scratch_root() -> PathBuf {
    let root = match std::env::var_os("TMPDIR") {
        Some(dir) if !dir.is_empty() => PathBuf::from(dir),
        _ => PathBuf::from(env!("CARGO_TARGET_TMPDIR")),
    };
    fs::create_dir_all(&root).unwrap();
    root
}

fn run_checked(what: &str, command: &mut Command) -> Output {
    let output = command
        .output()
        .unwrap_or_else(|error| panic!("{what}: cannot start: {error}"));
    assert!(
        output.status.success(),
        "{what} failed with {}:\n{}\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

fn parse_decimal_predictions(what: &str, text: &str) -> Vec<u64> {
    text.lines()
        .map(|line| {
            line.trim()
                .parse::<f64>()
                .unwrap_or_else(|error| panic!("{what}: prediction {line:?}: {error}"))
                .to_bits()
        })
        .collect()
}

struct Shared {
    train_csv: PathBuf,
    features_csv: BTreeMap<&'static str, PathBuf>,
    features_bits: BTreeMap<&'static str, PathBuf>,
    train_x: Vec<Vec<f64>>,
    train_y: Vec<f64>,
}

fn input_key(inputs: Inputs) -> &'static str {
    match inputs {
        Inputs::Finite => "finite",
        Inputs::All => "all",
        Inputs::NonFiniteColumnZero => "nonfinite-c0",
    }
}

/// One program's predictions in `lane`, as bit patterns in input-row order.
fn lane_predictions(
    lane: Lane,
    program: &Program,
    genome: &GenomeIr,
    rows: usize,
    shared: &Shared,
    dir: &Path,
) -> Vec<u64> {
    fs::create_dir_all(dir).unwrap();
    let key = input_key(program.inputs);
    let what = format!("{} {}", lane.id(), program.name);
    let bits = match lane {
        Lane::Rust => {
            let artifact = emit(genome, &EmitOptions::default()).unwrap();
            let source = dir.join("model.rs");
            let binary = dir.join("model_bin");
            let predictions = dir.join("predictions.txt");
            fs::write(&source, &artifact.rust).unwrap();
            run_checked(
                &format!("{what}: rustc"),
                Command::new("rustc")
                    .arg("-O")
                    .arg("-o")
                    .arg(&binary)
                    .arg(&source)
                    .env("TMPDIR", dir),
            );
            run_checked(
                &format!("{what}: run"),
                Command::new(&binary)
                    .arg(&shared.train_csv)
                    .arg(&shared.features_csv[key])
                    .arg(&predictions),
            );
            parse_decimal_predictions(&what, &fs::read_to_string(&predictions).unwrap())
        }
        Lane::Python => {
            let artifact = emit(genome, &EmitOptions::default()).unwrap();
            let source = dir.join("model.py");
            let predictions = dir.join("predictions.txt");
            fs::write(&source, &artifact.python).unwrap();
            run_checked(
                &format!("{what}: python3"),
                Command::new("python3")
                    .arg(&source)
                    .arg(&shared.train_csv)
                    .arg(&shared.features_csv[key])
                    .arg(&predictions)
                    .env("PYTHONDONTWRITEBYTECODE", "1")
                    .env("TMPDIR", dir),
            );
            parse_decimal_predictions(&what, &fs::read_to_string(&predictions).unwrap())
        }
        Lane::Cpp17Fitted => {
            let bound = fit_legacy_numeric_bound(genome, &shared.train_x, &shared.train_y)
                .unwrap_or_else(|error| panic!("{what}: bind: {error:#}"));
            let artifact =
                emit_fitted_cpp17(&bound).unwrap_or_else(|error| panic!("{what}: emit: {error:#}"));
            let model = dir.join("model.cpp");
            let driver = dir.join("driver.cpp");
            let binary = dir.join("model_bin");
            fs::write(&model, &artifact.source).unwrap();
            fs::write(&driver, DRIVER_CPP).unwrap();
            run_checked(
                &format!("{what}: {}", lane.tool()),
                feat_math::cxx::command(Some(CxxStd::Cxx17))
                    .args([
                        "-O2",
                        "-ffp-contract=off",
                        "-fno-fast-math",
                        "-Wall",
                        "-Wextra",
                        "-Werror",
                    ])
                    .arg(&model)
                    .arg(&driver)
                    .arg("-o")
                    .arg(&binary)
                    .env("TMPDIR", dir),
            );
            let output = run_checked(
                &format!("{what}: run"),
                Command::new(&binary).arg(&shared.features_bits[key]),
            );
            String::from_utf8(output.stdout)
                .unwrap()
                .lines()
                .map(|line| {
                    u64::from_str_radix(line.trim(), 16)
                        .unwrap_or_else(|error| panic!("{what}: prediction {line:?}: {error}"))
                })
                .collect()
        }
    };
    assert_eq!(
        bits.len(),
        rows,
        "{what}: {} predictions for {rows} rows",
        bits.len()
    );
    bits
}

/// Run `work` over `items` on up to [`MAX_JOBS`] threads; results keep the items' order.
fn parallel_map<T: Sync, R: Send>(items: &[T], work: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let jobs = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .clamp(1, MAX_JOBS)
        .min(items.len().max(1));
    let next = AtomicUsize::new(0);
    let slots: Vec<Mutex<Option<R>>> = items.iter().map(|_| Mutex::new(None)).collect();
    std::thread::scope(|scope| {
        for _ in 0..jobs {
            scope.spawn(|| loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                let Some(item) = items.get(index) else {
                    break;
                };
                let result = work(item);
                *slots[index].lock().unwrap() = Some(result);
            });
        }
    });
    slots
        .into_iter()
        .map(|slot| slot.into_inner().unwrap().expect("every item ran"))
        .collect()
}

// ------------------------------------------------------------------------------------------------
// Divergences and the allowlist
// ------------------------------------------------------------------------------------------------

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

/// Every allowlist entry as a [`Divergence`], after checking the file's shape: known programs,
/// lanes and input rows, exact bit patterns that differ (or involve a NaN), the one allowed
/// status, a fix item, a reason, and no duplicate.
fn allowlist() -> Vec<Divergence> {
    let list: Allowlist = serde_json::from_str(ALLOWLIST).expect("parity_allowlist.v1.json parses");
    assert_eq!(list.schema, ALLOWLIST_SCHEMA);
    assert!(!list.note.trim().is_empty(), "the allowlist needs its note");
    let mut seen = BTreeSet::new();
    let mut divergences = Vec::new();
    for entry in list.entries {
        let label = format!("{} {} {}", entry.program, entry.language, entry.input);
        let program = PROGRAMS
            .iter()
            .find(|program| program.name == entry.program)
            .unwrap_or_else(|| panic!("{label}: unknown program"));
        assert!(
            Lane::ALL.iter().any(|lane| lane.id() == entry.language),
            "{label}: unknown language"
        );
        assert!(
            input_rows(program.inputs)
                .iter()
                .any(|row| row.id == entry.input),
            "{label}: the program is not evaluated on this input"
        );
        let runtime_bits =
            parse_bits(&entry.runtime_bits).unwrap_or_else(|| panic!("{label}: runtime_bits"));
        let emitted_bits =
            parse_bits(&entry.emitted_bits).unwrap_or_else(|| panic!("{label}: emitted_bits"));
        assert!(
            runtime_bits != emitted_bits
                || f64::from_bits(runtime_bits).is_nan()
                || f64::from_bits(emitted_bits).is_nan(),
            "{label}: identical bit patterns are not a divergence"
        );
        assert_eq!(entry.status, ALLOWLIST_STATUS, "{label}: status");
        assert!(
            entry.fix_item.starts_with('X')
                && entry.fix_item.len() > 1
                && entry.fix_item[1..].bytes().all(|b| b.is_ascii_digit()),
            "{label}: fix_item must name a plan item such as X6"
        );
        assert!(!entry.reason.trim().is_empty(), "{label}: reason");
        assert!(
            seen.insert((
                entry.program.clone(),
                entry.language.clone(),
                entry.input.clone()
            )),
            "{label}: duplicate entry"
        );
        divergences.push(Divergence {
            program: entry.program,
            language: entry.language,
            input: entry.input,
            runtime_bits,
            emitted_bits,
        });
    }
    divergences
}

/// Compare one lane against the runtime on every program and check the result against the
/// allowlist entries of that lane.
fn check_lane(lane: Lane) {
    let Some(version) = toolchain(lane) else {
        return;
    };
    let expected = allowlist();
    let (train_x, train_y) = training_data();
    let mut scratch = tempfile::Builder::new()
        .prefix(&format!("parity-bits-{}-", lane.id()))
        .tempdir_in(scratch_root())
        .unwrap();
    let keep = std::env::var_os("JOPE_PARITY_KEEP_SCRATCH").is_some_and(|value| value == "1");
    if keep {
        // Also keeps the directory when an assertion below panics.
        scratch.disable_cleanup(true);
        eprintln!(
            "parity_bits[{}]: artifacts and inputs kept under {}",
            lane.id(),
            scratch.path().display()
        );
    }
    let root = scratch.path();
    let train_csv = root.join("train.csv");
    write_train_csv(&train_csv, &train_x, &train_y);
    let mut features_csv = BTreeMap::new();
    let mut features_bits = BTreeMap::new();
    for inputs in [Inputs::Finite, Inputs::All, Inputs::NonFiniteColumnZero] {
        let key = input_key(inputs);
        let rows = input_rows(inputs);
        let csv = root.join(format!("features-{key}.csv"));
        let bits = root.join(format!("features-{key}.bits"));
        write_features_csv(&csv, &rows);
        write_features_bits(&bits, &rows);
        features_csv.insert(key, csv);
        features_bits.insert(key, bits);
    }
    let shared = Shared {
        train_csv,
        features_csv,
        features_bits,
        train_x,
        train_y,
    };

    let per_program = parallel_map(PROGRAMS, |program| {
        let genome = GenomeIr::parse(program.source)
            .unwrap_or_else(|error| panic!("{}: {error:#}", program.name));
        let rows = input_rows(program.inputs);
        let features: Vec<Vec<f64>> = rows.iter().map(|row| row.values.clone()).collect();
        let reference: Vec<u64> = fit(&genome, &shared.train_x, &shared.train_y)
            .unwrap_or_else(|error| panic!("{}: runtime fit: {error:#}", program.name))
            .predict(&features)
            .into_iter()
            .map(f64::to_bits)
            .collect();
        let emitted = lane_predictions(
            lane,
            program,
            &genome,
            rows.len(),
            &shared,
            &root.join(program.name),
        );
        let mut divergences = Vec::new();
        for ((row, runtime_bits), emitted_bits) in rows.iter().zip(&reference).zip(&emitted) {
            let nan =
                f64::from_bits(*runtime_bits).is_nan() || f64::from_bits(*emitted_bits).is_nan();
            if runtime_bits != emitted_bits || nan {
                divergences.push(Divergence {
                    program: program.name.to_owned(),
                    language: lane.id().to_owned(),
                    input: row.id.clone(),
                    runtime_bits: *runtime_bits,
                    emitted_bits: *emitted_bits,
                });
            }
        }
        (rows.len(), divergences)
    });
    let compared: usize = per_program.iter().map(|(rows, _)| rows).sum();
    let found: BTreeSet<Divergence> = per_program
        .into_iter()
        .flat_map(|(_, divergences)| divergences)
        .collect();
    let listed: BTreeSet<Divergence> = expected
        .into_iter()
        .filter(|entry| entry.language == lane.id())
        .collect();
    let unexpected: Vec<&Divergence> = found.difference(&listed).collect();
    let stale: Vec<&Divergence> = listed.difference(&found).collect();
    eprintln!(
        "parity_bits[{}] {version}: {} programs, {compared} predictions, {} divergences \
         ({} allowlisted, {} unexpected, {} stale)",
        lane.id(),
        PROGRAMS.len(),
        found.len(),
        found.intersection(&listed).count(),
        unexpected.len(),
        stale.len()
    );
    let mut report = String::new();
    for divergence in &unexpected {
        report.push_str(&format!("  unexpected: {}\n", divergence.describe()));
    }
    for divergence in &stale {
        report.push_str(&format!(
            "  stale (no longer reproduces): {}\n",
            divergence.describe()
        ));
    }
    assert!(
        report.is_empty(),
        "parity_bits[{}]: the runtime and the emitted program disagree beyond the allowlist \
         (fixtures/parity_allowlist.v1.json):\n{report}",
        lane.id()
    );
    if keep {
        // `close` removes the directory whatever `disable_cleanup` says; `keep` leaves it.
        let _ = scratch.keep();
    } else {
        scratch.close().unwrap();
    }
}

#[test]
fn parity_bits_runtime_matches_emitted_rust() {
    check_lane(Lane::Rust);
}

#[test]
fn parity_bits_runtime_matches_emitted_python() {
    check_lane(Lane::Python);
}

#[test]
fn parity_bits_runtime_matches_fitted_cpp17() {
    check_lane(Lane::Cpp17Fitted);
}

// ------------------------------------------------------------------------------------------------
// Coverage, corpus and allowlist checks
// ------------------------------------------------------------------------------------------------

/// Exhaustive, so a new function fails to compile here until a program executes it.
fn func_name(func: Func) -> &'static str {
    match func {
        Func::Abs => "abs",
        Func::Sqrt => "sqrt",
        Func::Log1p => "log1p",
        Func::Sin => "sin",
        Func::Cos => "cos",
        Func::Tanh => "tanh",
        Func::Sigmoid => "sigmoid",
        Func::Min => "min",
        Func::Max => "max",
        Func::Clip => "clip",
        Func::Step => "step",
        Func::Exp => "exp",
        Func::Log => "log",
        Func::Asinh => "asinh",
        Func::Atan => "atan",
        Func::Erf => "erf",
        Func::Sign => "sign",
        Func::Floor => "floor",
        Func::Round => "round",
    }
}

const FUNCS: [Func; 19] = [
    Func::Abs,
    Func::Sqrt,
    Func::Log1p,
    Func::Sin,
    Func::Cos,
    Func::Tanh,
    Func::Sigmoid,
    Func::Min,
    Func::Max,
    Func::Clip,
    Func::Step,
    Func::Exp,
    Func::Log,
    Func::Asinh,
    Func::Atan,
    Func::Erf,
    Func::Sign,
    Func::Floor,
    Func::Round,
];

fn var_name(var: Var) -> &'static str {
    match var {
        Var::X => "x",
        Var::Col(_) => "col",
        Var::RowMean => "row_mean",
        Var::RowSum => "row_sum",
        Var::RowL1 => "row_l1",
        Var::RowL2 => "row_l2",
        Var::RowMin => "row_min",
        Var::RowMax => "row_max",
        Var::RowStd => "row_std",
    }
}

const VARS: [Var; 9] = [
    Var::X,
    Var::Col(0),
    Var::RowMean,
    Var::RowSum,
    Var::RowL1,
    Var::RowL2,
    Var::RowMin,
    Var::RowMax,
    Var::RowStd,
];

fn binop_name(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::Pow => "^",
    }
}

const BINOPS: [BinOp; 5] = [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div, BinOp::Pow];

fn unop_name(op: UnOp) -> &'static str {
    match op {
        UnOp::Neg => "neg",
    }
}

fn walk(expr: &Expr, seen: &mut BTreeSet<&'static str>) {
    match expr {
        Expr::Const(_) => {}
        Expr::Var(var) => {
            seen.insert(var_name(*var));
        }
        Expr::Unary(op, child) => {
            seen.insert(unop_name(*op));
            walk(child, seen);
        }
        Expr::Binary(op, left, right) => {
            seen.insert(binop_name(*op));
            walk(left, seen);
            walk(right, seen);
        }
        Expr::Call(func, args) => {
            seen.insert(func_name(*func));
            for arg in args {
                walk(arg, seen);
            }
        }
    }
}

/// Every expression a program evaluates: its closed form, or each `map`/`derive` stage.
fn expressions(program: &Program) -> Vec<MathExpr> {
    match GenomeIr::parse(program.source).unwrap() {
        GenomeIr::Symbolic { expr, .. } => vec![expr],
        GenomeIr::Pipeline(dsl) => dsl
            .stages
            .iter()
            .filter(|stage| matches!(stage.op.as_str(), "map" | "derive"))
            .map(|stage| match stage.params.get("expr") {
                Some(ParamValue::Str(source)) => MathExpr::parse(source).unwrap(),
                other => panic!("{}: expression parameter {other:?}", program.name),
            })
            .collect(),
        GenomeIr::Portfolio(_) => panic!("{}: portfolio programs are not compared", program.name),
    }
}

#[test]
fn parity_bits_programs_execute_every_function_variable_and_operator() {
    let mut seen = BTreeSet::new();
    let mut names = BTreeSet::new();
    for program in PROGRAMS {
        assert!(
            names.insert(program.name),
            "{}: duplicate name",
            program.name
        );
        let genome = GenomeIr::parse(program.source).unwrap();
        supports(&genome).unwrap_or_else(|missing| panic!("{}: {missing:?}", program.name));
        for expr in expressions(program) {
            walk(expr.ast(), &mut seen);
        }
    }
    let required: BTreeSet<&str> = FUNCS
        .iter()
        .map(|f| func_name(*f))
        .chain(VARS.iter().map(|v| var_name(*v)))
        .chain(BINOPS.iter().map(|op| binop_name(*op)))
        .chain([unop_name(UnOp::Neg)])
        .collect();
    assert_eq!(required.len(), 19 + 9 + 5 + 1);
    let missing: Vec<&&str> = required.difference(&seen).collect();
    assert!(missing.is_empty(), "no program executes {missing:?}");
    // Both expression contexts and both closed-form modes are exercised.
    let sources: Vec<&str> = PROGRAMS.iter().map(|program| program.source).collect();
    for context in ["(map expr=", "(derive expr=", "(sym raw ", "(sym affine "] {
        assert!(
            sources.iter().any(|source| source.contains(context)),
            "no program uses {context}"
        );
    }
    // Every input policy is used, and every lane sees the boundary rows.
    for inputs in [Inputs::Finite, Inputs::All, Inputs::NonFiniteColumnZero] {
        assert!(PROGRAMS.iter().any(|program| program.inputs == inputs));
    }
    let finite = input_rows(Inputs::Finite);
    let all = input_rows(Inputs::All);
    assert!(finite
        .iter()
        .all(|row| row.values.iter().all(|v| v.is_finite())));
    assert!(all.iter().any(|row| row.values.iter().any(|v| v.is_nan())));
    assert!(all.len() > finite.len());
    // The division_guard rows reach the guard of `/`: with the value in column 0, 1e-12 divides,
    // and the next value toward zero is guarded to 0 exactly like a zero denominator.
    let program = PROGRAMS
        .iter()
        .find(|program| program.name == "sym_div_guard")
        .unwrap();
    let (train_x, train_y) = training_data();
    let genome = GenomeIr::parse(program.source).unwrap();
    let model = fit(&genome, &train_x, &train_y).unwrap();
    let rows = input_rows(program.inputs);
    let predict = |id: &str| {
        let row = rows
            .iter()
            .find(|row| row.id == id)
            .unwrap_or_else(|| panic!("no input row {id}"));
        model.predict(&vec![row.values.clone()])[0].to_bits()
    };
    for sign in ["pos", "neg"] {
        let zero = predict(&format!("c0:{sign}_zero"));
        assert_eq!(
            predict(&format!("c0:div_guard_{sign}_toward_zero")),
            zero,
            "{sign}: one step toward zero from 1e-12 takes the guard"
        );
        assert_ne!(
            predict(&format!("c0:div_guard_{sign}")),
            zero,
            "{sign}: 1e-12 divides"
        );
    }
}

#[test]
fn parity_bits_local_boundary_copy_is_the_shared_corpus() {
    let digest = format!("{:x}", Sha256::digest(BOUNDARY_V1.as_bytes()));
    assert_eq!(digest, BOUNDARY_V1_SHA256);
    let corpus: Corpus = serde_json::from_str(BOUNDARY_V1).unwrap();
    assert_eq!(corpus.schema, "feat-math.math-boundary.v1");
    assert!(!corpus.note.is_empty());
    let mut names = BTreeSet::new();
    for entry in &corpus.entries {
        assert!(
            names.insert(entry.name.as_str()),
            "duplicate {}",
            entry.name
        );
        assert!(corpus.groups.contains(&entry.group), "{}", entry.name);
        let bits = parse_bits(&entry.bits).unwrap();
        let value = f64::from_bits(bits);
        if value.is_nan() {
            assert_eq!(entry.value_repr, "NaN");
        } else {
            assert_eq!(format!("{value:?}"), entry.value_repr, "{}", entry.name);
        }
    }
    for group in &corpus.groups {
        assert!(
            corpus.entries.iter().any(|entry| &entry.group == group),
            "empty group {group}"
        );
    }
    // The rows the lanes compare carry every entry and both signed zeros in every position.
    let all = input_rows(Inputs::All);
    for entry in &corpus.entries {
        assert!(all
            .iter()
            .any(|row| row.id == format!("all:{}", entry.name)));
    }
    assert!(all.iter().any(|row| row
        .values
        .iter()
        .all(|v| v.to_bits() == (-0.0f64).to_bits())));
}

#[test]
fn parity_bits_allowlist_is_well_formed_and_justified() {
    // allowlist() checks every entry's shape and justification fields.
    let entries = allowlist();
    let text: serde_json::Value = serde_json::from_str(ALLOWLIST).unwrap();
    assert_eq!(
        text["entries"].as_array().map(Vec::len),
        Some(entries.len())
    );
}
