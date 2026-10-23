//! Deterministic token counting for the tokens/sec metric, plus an exact
//! static token *estimator* (`estimate_tokens`) that predicts the emitted
//! Python/Rust token counts without running the emitters.
//!
//! Counting: identifiers/keywords, numeric literals, string literals, and
//! maximal operator/punctuation glyphs count 1 token each; words inside
//! comments and docstrings count 1 each (comments are required content).
//! Whitespace counts 0, so the metric cannot be inflated by formatting.
//!
//! Estimation: the emitted artifacts are section-additive (sections are joined
//! by whitespace separators, and no token spans a whitespace boundary), so the
//! total token count decomposes into
//!
//!   moddoc(plan) + head_base[kind] + Σ op_section[kind] + Σ repeat_calls[kind]
//!   + Σ helper[h in helper-union closure] + exact doc-word and literal deltas.
//!
//! Every constant in that decomposition is **self-calibrated**: on first use a
//! one-time builder emits ~130 tiny genomes through the real emitters, then
//! solves set-intersection equations for the per-helper costs and diffs for
//! the per-op / per-head costs. No hand-maintained token numbers exist here;
//! the only replicated knowledge is the *shape* of the variable fragments
//! (module doc, per-stage formula sections, f64 literals, kernel-term
//! literals), which are rebuilt verbatim and counted with the same lexer.
//! Drift in either emitter is caught by `tests/tokens.rs`, which asserts exact
//! equality against real emissions on every golden fixture plus hundreds of
//! random genomes.

use crate::mathexpr::{fmt_f64, MathAst};
use crate::spec::{self, Helper, Plan, PlannedHead, PlannedOp};
use crate::transpile::{EmitOptions, TokenCount};
use crate::GenomeIr;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

#[derive(Clone, Copy, PartialEq)]
pub enum Lang {
    Python,
    Rust,
    Cpp,
    C,
    JavaScript,
    Go,
    TypeScript,
    CudaCpp,
}

pub fn count_tokens(src: &str, lang: Lang) -> TokenCount {
    let mut code_tokens = 0usize;
    let mut doc_tokens = 0usize;
    let bytes = src.len();
    let b = src.as_bytes();
    let mut i = 0usize;
    while i < b.len() {
        let c = b[i] as char;
        if c.is_whitespace() {
            i += 1;
        } else if c == '#' && lang == Lang::Python {
            let end = line_end(b, i);
            doc_tokens += word_count(&src[i + 1..end]);
            i = end;
        } else if matches!(
            lang,
            Lang::Rust
                | Lang::Cpp
                | Lang::C
                | Lang::JavaScript
                | Lang::Go
                | Lang::TypeScript
                | Lang::CudaCpp
        ) && c == '/'
            && i + 1 < b.len()
            && b[i + 1] == b'/'
        {
            let end = line_end(b, i);
            doc_tokens += word_count(src[i..end].trim_start_matches(['/', '!']));
            i = end;
        } else if lang == Lang::Python
            && i + 2 < b.len()
            && (b.get(i..i + 3) == Some(b"\"\"\"") || b.get(i..i + 3) == Some(b"'''"))
        {
            let quote = &src[i..i + 3];
            let rest = &src[i + 3..];
            if let Some(end) = rest.find(quote) {
                doc_tokens += word_count(&rest[..end]);
                i += 3 + end + 3;
            } else {
                doc_tokens += word_count(rest);
                i = src.len();
            }
        } else if c == '`' && lang == Lang::Go {
            i = src[i + 1..].find('`').map_or(src.len(), |end| i + end + 2);
            code_tokens += 1;
        } else if c == '"'
            || (c == '\''
                && matches!(
                    lang,
                    Lang::Python | Lang::JavaScript | Lang::Go | Lang::TypeScript
                ))
        {
            // string literal = 1 code token
            let q = b[i];
            let mut j = i + 1;
            while j < b.len() {
                if b[j] == b'\\' {
                    j += 2;
                    continue;
                }
                if b[j] == q {
                    break;
                }
                j += 1;
            }
            code_tokens += 1;
            i = (j + 1).min(b.len());
        } else if c.is_alphanumeric() || c == '_' {
            let mut j = i;
            while j < b.len() && ((b[j] as char).is_alphanumeric() || b[j] == b'_' || b[j] == b'.')
            {
                j += 1;
            }
            code_tokens += 1;
            i = j;
        } else {
            // maximal operator glyph (up to 3 chars)
            let three = src.get(i..(i + 3).min(src.len())).unwrap_or("");
            let two = src.get(i..(i + 2).min(src.len())).unwrap_or("");
            const OPS3: [&str; 6] = ["**=", "//=", ">>=", "<<=", "...", "..="];
            const OPS2: [&str; 20] = [
                "**", "//", "<<", ">>", "<=", ">=", "==", "!=", "+=", "-=", "*=", "/=", "%=", "&&",
                "||", "->", "=>", "::", "&=", "|=",
            ];
            if OPS3.contains(&three) {
                i += 3;
            } else if OPS2.contains(&two) {
                i += 2;
            } else {
                i += 1;
            }
            code_tokens += 1;
        }
    }
    TokenCount {
        code_tokens,
        doc_tokens,
        bytes,
    }
}

fn line_end(b: &[u8], from: usize) -> usize {
    let mut i = from;
    while i < b.len() && b[i] != b'\n' {
        i += 1;
    }
    i
}

fn word_count(s: &str) -> usize {
    s.split_whitespace()
        .filter(|w| w.chars().any(|c| c.is_alphanumeric()))
        .count()
}

// ===========================================================================
//                        static token estimator (X6)
// ===========================================================================

/// Predicted total (code + doc) token counts of the two emitted artifacts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct TokenEstimate {
    pub py: usize,
    pub rs: usize,
}

/// Exact static estimate of the emitted token counts for `genome` under
/// `EmitOptions::default()`, computed without emitting. Panics on genomes the
/// emitter does not support (same coverage as `supports()`); use
/// [`try_estimate_tokens`] for a fallible variant or custom options.
pub fn estimate_tokens(genome: &GenomeIr) -> TokenEstimate {
    try_estimate_tokens(genome, &EmitOptions::default())
        .expect("estimate_tokens: unsupported genome (emitters would reject it too)")
}

/// Fallible estimator; `Err` lists the unsupported genome elements, exactly
/// like `supports()`.
pub fn try_estimate_tokens(
    genome: &GenomeIr,
    opts: &EmitOptions,
) -> Result<TokenEstimate, Vec<String>> {
    if let Err(error) = crate::transpile::validate_math_kernel(opts) {
        return Err(vec![error.to_string()]);
    }
    let plan = spec::build_plan(genome)?;
    // Fast-runtime artifacts restructure the emitted text (wrapper method,
    // header lines); like portfolio genomes they are estimated by emitting
    // and counting — exact by construction, and the profile is opt-in and
    // rare enough that the static estimator's speed does not matter.
    if opts.fast_runtime {
        let emitted = match crate::transpile::emit(genome, opts) {
            Ok(e) => e,
            Err(err) => return Err(vec![format!("{err}")]),
        };
        return Ok(TokenEstimate {
            py: emitted.py_tokens.code_tokens + emitted.py_tokens.doc_tokens,
            rs: emitted.rs_tokens.code_tokens + emitted.rs_tokens.doc_tokens,
        });
    }
    // Portfolio heads: the section set depends on the resolved member roster,
    // so the estimate is computed by emitting and counting — exact by
    // construction, and portfolio genomes are far too rare for the static
    // estimator's speed to matter.
    if let Plan::Pipeline { head, .. } = &plan {
        if matches!(head, spec::PlannedHead::Portfolio(_)) {
            let emitted = match crate::transpile::emit(genome, opts) {
                Ok(e) => e,
                Err(err) => return Err(vec![format!("{err}")]),
            };
            return Ok(TokenEstimate {
                py: emitted.py_tokens.code_tokens + emitted.py_tokens.doc_tokens,
                rs: emitted.rs_tokens.code_tokens + emitted.rs_tokens.doc_tokens,
            });
        }
    }
    let cal = calibration();
    let mut t = moddoc_pair(&plan, opts);
    match &plan {
        // NOTE: the estimator has no calibrated costs for the `(target ...)`
        // pseudo-stage yet; target plans estimate as if the target section were
        // absent (jope-validate strips the leading target before estimating).
        Plan::Pipeline { stages, head, .. } => {
            // Head: calibrated base + exact doc-word and literal deltas.
            let hk = spec::head_fn_name(head);
            t += cal.head_base[hk];
            let dw = word_count(&format!("This head is {}", spec::head_doc(head))) as i64;
            t += Pair::both(dw);
            let slots = head_f64_slots(head);
            for (i, v) in slots.iter().enumerate() {
                let d = lit_tokens(*v) - cal.head_ref_lit[hk][i];
                t += cal.head_mult[hk][i] * d;
            }
            if let PlannedHead::GprKernel { spec: ks, .. } = head {
                let py_lit = code_tok(&py_kernel_terms(ks), Lang::Python);
                let rs_lit = code_tok(&rs_kernel_terms(ks), Lang::Rust);
                t.py += cal.gprk_mult.py * (py_lit - cal.gprk_ref_lit.py);
                t.rs += cal.gprk_mult.rs * (rs_lit - cal.gprk_ref_lit.rs);
            }
            // Stages: first occurrence of an op kind carries its section (with
            // the first instance's doc words); repeats carry call lines only;
            // formula stages are reconstructed verbatim per stage.
            let mut seen: Vec<&'static str> = Vec::new();
            let mut has_formula = false;
            for (i, stage) in stages.iter().enumerate() {
                let n = i + 1;
                match &stage.op {
                    PlannedOp::MapExpr { source, expr } => {
                        has_formula = true;
                        t.py += total_tok(&py_map_src(n, source, expr), Lang::Python);
                        t.rs += total_tok(&rs_map_src(n, source, expr), Lang::Rust);
                        t += cal.map_calls;
                    }
                    PlannedOp::DeriveExpr { source, expr } => {
                        has_formula = true;
                        t.py += total_tok(&py_derive_src(n, source, expr), Lang::Python);
                        t.rs += total_tok(&rs_derive_src(n, source, expr), Lang::Rust);
                        t += cal.derive_calls;
                    }
                    op => {
                        let name = spec::op_fn_name(op);
                        if !seen.contains(&name) {
                            seen.push(name);
                            t += cal.op_sec[name];
                            let dw =
                                word_count(&format!("This transform {}", spec::op_doc(op))) as i64;
                            t += Pair::both(dw);
                        } else {
                            t += cal.op_calls[name];
                        }
                        if let Some(v) = op_f64_slot(op) {
                            t += cal.op_mult[name] * (lit_tokens(v) - cal.op_ref_lit[name]);
                        }
                    }
                }
            }
            if has_formula {
                // The Rust emitter adds `#![allow(unused_parens)]` whenever a
                // formula stage exists (measured as ~0 for Python).
                t += cal.formula_attr;
            }
            for h in spec::helper_set(&plan) {
                if let Some(c) = cal.helpers.get(&h) {
                    t += *c;
                }
            }
        }
        Plan::Symbolic {
            source,
            expr,
            affine,
        } => {
            t += cal.sym_base[*affine as usize];
            t.py += total_tok(&py_expression_fn(source, expr), Lang::Python);
            t.rs += total_tok(&rs_expression_fn(source, expr), Lang::Rust);
            for h in spec::helper_set(&plan) {
                if let Some(c) = cal.helpers.get(&h) {
                    t += *c;
                }
            }
        }
    }
    Ok(TokenEstimate {
        py: t.py.max(0) as usize,
        rs: t.rs.max(0) as usize,
    })
}

// ------------------------------ arithmetic ---------------------------------

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Pair {
    py: i64,
    rs: i64,
}

impl Pair {
    fn both(v: i64) -> Pair {
        Pair { py: v, rs: v }
    }
}

impl std::ops::Add for Pair {
    type Output = Pair;
    fn add(self, o: Pair) -> Pair {
        Pair {
            py: self.py + o.py,
            rs: self.rs + o.rs,
        }
    }
}
impl std::ops::Sub for Pair {
    type Output = Pair;
    fn sub(self, o: Pair) -> Pair {
        Pair {
            py: self.py - o.py,
            rs: self.rs - o.rs,
        }
    }
}
impl std::ops::Mul<i64> for Pair {
    type Output = Pair;
    fn mul(self, k: i64) -> Pair {
        Pair {
            py: self.py * k,
            rs: self.rs * k,
        }
    }
}
impl std::ops::AddAssign for Pair {
    fn add_assign(&mut self, o: Pair) {
        *self = *self + o;
    }
}
impl std::ops::SubAssign for Pair {
    fn sub_assign(&mut self, o: Pair) {
        *self = *self - o;
    }
}

fn total_tok(s: &str, lang: Lang) -> i64 {
    let c = count_tokens(s, lang);
    (c.code_tokens + c.doc_tokens) as i64
}

fn code_tok(s: &str, lang: Lang) -> i64 {
    count_tokens(s, lang).code_tokens as i64
}

/// Token count of an f64 literal as both emitters format it (`fmt_f64`); the
/// string is identical in both languages, so a single value suffices
/// (e.g. `0.05` -> 1, `-0.5` -> 2, `1e-05` -> 3).
fn lit_tokens(v: f64) -> i64 {
    code_tok(&fmt_f64(v), Lang::Python)
}

// ------------------------- variable-fragment replicas -----------------------
// Byte-exact replicas of the emitters' *variable* fragments. Each mirrors the
// named private function; tests/tokens.rs asserts full-artifact equality on
// every golden fixture and hundreds of random genomes, so any drift between
// an emitter and its replica fails loudly.

/// Token count of the module doc in both artifacts, mirroring
/// `emit_py::docstring` / `emit_rs::module_doc` without building the strings:
/// the module doc lexes to doc words only (the `"""` pair and `//!` markers
/// carry no code tokens), and greedy wrapping never splits or merges words, so
/// the count is the word count of the unwrapped title + paragraphs plus the
/// fixed usage/build lines. Appending the title's trailing '.' can never
/// change its word count (it carries no alphanumeric of its own).
fn moddoc_pair(plan: &Plan, opts: &EmitOptions) -> Pair {
    let (title, paragraphs) = spec::doc_paragraphs(plan, opts);
    let mut words = word_count(&title) as i64;
    for p in &paragraphs {
        words += word_count(p) as i64;
    }
    let py_usage =
        word_count("Usage: python3 model.py train.csv test_features.csv predictions.csv") as i64;
    let rs_frame = word_count("Build: rustc -O model.rs") as i64
        + word_count("Usage: ./model train.csv test_features.csv predictions.csv") as i64;
    Pair {
        py: words + py_usage,
        rs: words + rs_frame,
    }
}

/// Replica of `emit_py::map_src`.
fn py_map_src(n: usize, source: &str, expr: &MathAst) -> String {
    let features = expr.features();
    let mut lines: Vec<String> = Vec::new();
    lines.push(format!("def elementwise_formula{n}(x, row):"));
    lines.push(format!(
        "    \"\"\"f(x, row) = {source}, with guarded arithmetic; x is one cell of row.\"\"\""
    ));
    if let Some(maxc) = features.max_col {
        lines.push(format!("    if len(row) <= {maxc}:"));
        lines.push("        return 0.0".to_string());
    }
    if features.funcs.contains(&crate::mathexpr::MathFunc::Clip) {
        lines.push("    try:".to_string());
        lines.push(format!("        value = {}", expr.to_python()));
        lines.push("    except ArithmeticError:".to_string());
        lines.push("        return 0.0".to_string());
    } else {
        lines.push(format!("    value = {}", expr.to_python()));
    }
    lines.push("    if math.isfinite(value):".to_string());
    lines.push("        return _clamp(value, -1e12, 1e12)".to_string());
    lines.push("    return 0.0".to_string());
    lines.push(String::new());
    lines.push(String::new());
    lines.push(format!("def elementwise_formula{n}_apply(x, append):"));
    lines.push(format!(
        "    \"\"\"Stage {n}: map every cell through elementwise_formula{n}.\"\"\""
    ));
    lines.push(format!(
        "    out = [[elementwise_formula{n}(v, row) for v in row] for row in x]"
    ));
    lines.push("    return _finish_stage(x, out, append)".to_string());
    lines.join("\n")
}

/// Replica of `emit_py::derive_src`.
fn py_derive_src(n: usize, source: &str, expr: &MathAst) -> String {
    let features = expr.features();
    let mut lines: Vec<String> = Vec::new();
    lines.push(format!("def derived_column{n}(row):"));
    lines.push(format!(
        "    \"\"\"g(row) = {source}, with guarded arithmetic.\"\"\""
    ));
    if features.uses_x {
        lines.push(
            "    # The formula names the element variable x, which has no meaning for a whole"
                .to_string(),
        );
        lines.push("    # row, so every row maps to 0.0 (mirroring the reference).".to_string());
        lines.push("    return 0.0".to_string());
    } else {
        if let Some(maxc) = features.max_col {
            lines.push(format!("    if len(row) <= {maxc}:"));
            lines.push("        return 0.0".to_string());
        }
        if features.funcs.contains(&crate::mathexpr::MathFunc::Clip) {
            lines.push("    try:".to_string());
            lines.push(format!("        value = {}", expr.to_python()));
            lines.push("    except ArithmeticError:".to_string());
            lines.push("        return 0.0".to_string());
        } else {
            lines.push(format!("    value = {}", expr.to_python()));
        }
        lines.push("    if math.isfinite(value):".to_string());
        lines.push("        return _clamp(value, -1e12, 1e12)".to_string());
        lines.push("    return 0.0".to_string());
    }
    lines.push(String::new());
    lines.push(String::new());
    lines.push(format!("def derived_column{n}_apply(x, append):"));
    lines.push(format!(
        "    \"\"\"Stage {n}: reduce every row to the single column derived_column{n}(row).\"\"\""
    ));
    lines.push(format!("    out = [[derived_column{n}(row)] for row in x]"));
    lines.push("    return _finish_stage(x, out, append)".to_string());
    lines.join("\n")
}

/// Replica of `emit_py::expression_fn`.
fn py_expression_fn(source: &str, expr: &MathAst) -> String {
    let features = expr.features();
    let mut lines: Vec<String> = Vec::new();
    lines.push("def expression_value(row):".to_string());
    lines.push(format!(
        "    \"\"\"Evaluate f(row) = {source} with guarded arithmetic.\"\"\""
    ));
    if features.uses_x {
        lines.push(
            "    # The formula names the element variable x, which has no meaning for a whole"
                .to_string(),
        );
        lines.push("    # row, so every row maps to 0.0 (mirroring the reference).".to_string());
        lines.push("    return 0.0".to_string());
        return lines.join("\n");
    }
    if let Some(maxc) = features.max_col {
        lines.push(format!("    if len(row) <= {maxc}:"));
        lines.push("        return 0.0".to_string());
    }
    if features.funcs.contains(&crate::mathexpr::MathFunc::Clip) {
        lines.push("    try:".to_string());
        lines.push(format!("        value = {}", expr.to_python()));
        lines.push("    except ArithmeticError:".to_string());
        lines.push("        return 0.0".to_string());
    } else {
        lines.push(format!("    value = {}", expr.to_python()));
    }
    lines.push("    if math.isfinite(value):".to_string());
    lines.push("        return _clamp(value, -1e12, 1e12)".to_string());
    lines.push("    return 0.0".to_string());
    lines.join("\n")
}

/// Replica of `emit_rs::map_src`.
fn rs_map_src(n: usize, source: &str, expr: &MathAst) -> String {
    let features = expr.features();
    let mut lines: Vec<String> = Vec::new();
    lines.push(format!(
        "/// f(x, row) = {source}, with guarded arithmetic; x is one cell of row."
    ));
    lines.push(format!(
        "fn elementwise_formula{n}(x: f64, row: &[f64]) -> f64 {{"
    ));
    if let Some(maxc) = features.max_col {
        lines.push(format!("    if row.len() <= {maxc} {{"));
        lines.push("        return 0.0;".to_string());
        lines.push("    }".to_string());
    } else if features.vars.is_empty()
        || features
            .vars
            .iter()
            .all(|v| matches!(v, crate::mathexpr::MathVar::X))
    {
        lines.push("    let _ = row;".to_string());
    }
    if !features.uses_x {
        lines.push("    let _ = x;".to_string());
    }
    if features.funcs.contains(&crate::mathexpr::MathFunc::Clip) {
        lines.push("    let mut math_failed = false;".to_string());
        lines.push(format!("    let value = {};", expr.to_rust()));
        lines.push("    if math_failed { return 0.0; }".to_string());
    } else {
        lines.push(format!("    let value = {};", expr.to_rust()));
    }
    lines.push("    if value.is_finite() {".to_string());
    lines.push("        value.clamp(-1e12, 1e12)".to_string());
    lines.push("    } else {".to_string());
    lines.push("        0.0".to_string());
    lines.push("    }".to_string());
    lines.push("}".to_string());
    lines.push(String::new());
    lines.push(format!(
        "/// Stage {n}: map every cell through elementwise_formula{n}."
    ));
    lines.push(format!(
        "fn elementwise_formula{n}_apply(x: &[Vec<f64>], append: bool) -> Vec<Vec<f64>> {{"
    ));
    lines.push("    let mut out = Vec::new();".to_string());
    lines.push("    for row in x {".to_string());
    lines.push(format!(
        "        out.push(row.iter().map(|v| elementwise_formula{n}(*v, row)).collect());"
    ));
    lines.push("    }".to_string());
    lines.push("    finish_stage(x, out, append)".to_string());
    lines.push("}".to_string());
    lines.join("\n")
}

/// Replica of `emit_rs::derive_src`.
fn rs_derive_src(n: usize, source: &str, expr: &MathAst) -> String {
    let features = expr.features();
    let mut lines: Vec<String> = Vec::new();
    lines.push(format!("/// g(row) = {source}, with guarded arithmetic."));
    lines.push(format!("fn derived_column{n}(row: &[f64]) -> f64 {{"));
    if features.uses_x {
        lines.push(
            "    // The formula names the element variable x, which has no meaning for a whole"
                .to_string(),
        );
        lines.push("    // row, so every row maps to 0.0 (mirroring the reference).".to_string());
        lines.push("    let _ = row;".to_string());
        lines.push("    0.0".to_string());
    } else {
        if let Some(maxc) = features.max_col {
            lines.push(format!("    if row.len() <= {maxc} {{"));
            lines.push("        return 0.0;".to_string());
            lines.push("    }".to_string());
        } else if features.vars.is_empty() {
            lines.push("    let _ = row;".to_string());
        }
        if features.funcs.contains(&crate::mathexpr::MathFunc::Clip) {
            lines.push("    let mut math_failed = false;".to_string());
            lines.push(format!("    let value = {};", expr.to_rust()));
            lines.push("    if math_failed { return 0.0; }".to_string());
        } else {
            lines.push(format!("    let value = {};", expr.to_rust()));
        }
        lines.push("    if value.is_finite() {".to_string());
        lines.push("        value.clamp(-1e12, 1e12)".to_string());
        lines.push("    } else {".to_string());
        lines.push("        0.0".to_string());
        lines.push("    }".to_string());
    }
    lines.push("}".to_string());
    lines.push(String::new());
    lines.push(format!(
        "/// Stage {n}: reduce every row to the single column derived_column{n}(row)."
    ));
    lines.push(format!(
        "fn derived_column{n}_apply(x: &[Vec<f64>], append: bool) -> Vec<Vec<f64>> {{"
    ));
    lines.push(format!(
        "    let out: Vec<Vec<f64>> = x.iter().map(|row| vec![derived_column{n}(row)]).collect();"
    ));
    lines.push("    finish_stage(x, out, append)".to_string());
    lines.push("}".to_string());
    lines.join("\n")
}

/// Replica of `emit_rs::expression_fn`.
fn rs_expression_fn(source: &str, expr: &MathAst) -> String {
    let features = expr.features();
    let mut lines: Vec<String> = Vec::new();
    lines.push(format!(
        "/// Evaluate f(row) = {source} with guarded arithmetic."
    ));
    lines.push("fn expression_value(row: &[f64]) -> f64 {".to_string());
    if features.uses_x {
        lines.push(
            "    // The formula names the element variable x, which has no meaning for a whole"
                .to_string(),
        );
        lines.push("    // row, so every row maps to 0.0 (mirroring the reference).".to_string());
        lines.push("    let _ = row;".to_string());
        lines.push("    0.0".to_string());
        lines.push("}".to_string());
        return lines.join("\n");
    }
    if let Some(maxc) = features.max_col {
        lines.push(format!("    if row.len() <= {maxc} {{"));
        lines.push("        return 0.0;".to_string());
        lines.push("    }".to_string());
    }
    if features.max_col.is_none() && features.vars.is_empty() {
        lines.push("    let _ = row;".to_string());
    }
    if features.funcs.contains(&crate::mathexpr::MathFunc::Clip) {
        lines.push("    let mut math_failed = false;".to_string());
        lines.push(format!("    let value = {};", expr.to_rust()));
        lines.push("    if math_failed { return 0.0; }".to_string());
    } else {
        lines.push(format!("    let value = {};", expr.to_rust()));
    }
    lines.push("    if value.is_finite() {".to_string());
    lines.push("        value.clamp(-1e12, 1e12)".to_string());
    lines.push("    } else {".to_string());
    lines.push("        0.0".to_string());
    lines.push("    }".to_string());
    lines.push("}".to_string());
    lines.join("\n")
}

/// Replica of `emit_py::py_kernel_terms`.
fn py_kernel_terms(spec: &crate::gpr::KernelSpec) -> String {
    let terms: Vec<String> = spec
        .iter()
        .map(|term| {
            let factors: Vec<String> = term
                .iter()
                .map(|p| format!("\"{}\"", spec::kernel_prim_tag(*p)))
                .collect();
            format!("[{}]", factors.join(", "))
        })
        .collect();
    format!("[{}]", terms.join(", "))
}

/// Replica of `emit_rs::rs_kernel_terms`.
fn rs_kernel_terms(spec: &crate::gpr::KernelSpec) -> String {
    let terms: Vec<String> = spec
        .iter()
        .map(|term| {
            let factors: Vec<String> = term
                .iter()
                .map(|p| format!("\"{}\"", spec::kernel_prim_tag(*p)))
                .collect();
            format!("vec![{}]", factors.join(", "))
        })
        .collect();
    format!("[{}]", terms.join(", "))
}

// ----------------------------- literal slots --------------------------------
// The f64 literals each planned op/head bakes into its Model-class call lines,
// in call order (usize / u64 / bool literals are always exactly 1 token and
// need no adjustment). Mirrors `op_calls` / `head_calls` in both emitters.

fn op_f64_slot(op: &PlannedOp) -> Option<f64> {
    match op {
        PlannedOp::WinsorClip { q } => Some(*q),
        PlannedOp::Symlog { c } => Some(*c),
        PlannedOp::Impute { value, .. } => Some(*value),
        PlannedOp::RffCos { gamma, .. } => Some(*gamma),
        PlannedOp::RffSin { gamma, .. } => Some(*gamma),
        PlannedOp::LandmarkDist { gamma, .. } => Some(*gamma),
        PlannedOp::TargetBinEncode { prior, .. } => Some(*prior),
        PlannedOp::Step { t } => Some(*t),
        PlannedOp::Nystrom { gamma, .. } => Some(*gamma),
        _ => None,
    }
}

fn head_f64_slots(head: &PlannedHead) -> Vec<f64> {
    match head {
        PlannedHead::Ridge { lambda_frac } => vec![*lambda_frac],
        PlannedHead::ElasticNet {
            alpha, l1_ratio, ..
        } => vec![*alpha, *l1_ratio],
        PlannedHead::Gbm(g) => vec![g.lr, g.lambda],
        PlannedHead::Blend { lambda_frac, gbm } => vec![*lambda_frac, gbm.lr, gbm.lambda],
        PlannedHead::HistGbm2 {
            lr,
            row_subsample,
            col_subsample,
            huber_delta,
            ..
        } => {
            vec![*lr, *row_subsample, *col_subsample, *huber_delta]
        }
        PlannedHead::Ksmooth { gamma, .. } => vec![*gamma],
        PlannedHead::RidgeGcv
        | PlannedHead::Krr { .. }
        | PlannedHead::Gpr { .. }
        | PlannedHead::GprKernel { .. }
        | PlannedHead::Stack
        // Portfolio heads are estimated by emission (see try_estimate_tokens).
        | PlannedHead::Portfolio(_) => vec![],
    }
}

// ------------------------------- calibration --------------------------------

struct Calibration {
    /// Per-helper (code+doc) cost per language. Helpers that only ever occur
    /// alongside a single op/head (Probit, Normalize, UpperBound, Dot,
    /// CholeskySolve, JacobiEigen, SoftThreshold, TreeCore, MeanOf, Rmse,
    /// RSquared, Sanitize, FinishStage) are folded into that component and
    /// absent here.
    helpers: BTreeMap<Helper, Pair>,
    /// Rust-only `#![allow(unused_parens)]` attribute added when any formula
    /// stage exists (Python component is measured and asserted 0).
    formula_attr: Pair,
    /// Per-head cost net of calibrated helpers AND of the head's doc words
    /// ("This head is ..."), which the estimator recomputes from the actual
    /// planned head.
    head_base: BTreeMap<&'static str, Pair>,
    head_mult: BTreeMap<&'static str, Vec<Pair>>,
    head_ref_lit: BTreeMap<&'static str, Vec<i64>>,
    gprk_mult: Pair,
    gprk_ref_lit: Pair,
    /// First-occurrence op cost (section + stage lines) net of calibrated
    /// helpers AND of the op's doc words ("This transform ...").
    op_sec: BTreeMap<&'static str, Pair>,
    op_calls: BTreeMap<&'static str, Pair>,
    op_mult: BTreeMap<&'static str, Pair>,
    op_ref_lit: BTreeMap<&'static str, i64>,
    map_calls: Pair,
    derive_calls: Pair,
    /// Indexed by `affine as usize`; net of the expression section and helpers.
    sym_base: [Pair; 2],
}

static CALIBRATION: OnceLock<Calibration> = OnceLock::new();

fn calibration() -> &'static Calibration {
    CALIBRATION.get_or_init(build_calibration)
}

/// Helpers whose costs are calibrated individually (all others are folded).
/// `RngGaussian` carries the joint cost of {RngGaussian, Dot} and `MidRank`
/// the joint cost of {MidRank, UpperBound}; each pair provably co-occurs in
/// every reachable helper closure.
const CAL_HELPERS: [Helper; 38] = [
    Helper::Clamp,
    Helper::Fmin,
    Helper::Fmax,
    Helper::SafeSin,
    Helper::SafeCos,
    Helper::PowGuard,
    Helper::MathDiv,
    Helper::MathLog1p,
    Helper::MathSigmoid,
    Helper::MathClip,
    Helper::Asinh,
    Helper::MathExp,
    Helper::MathLog,
    Helper::Erf,
    Helper::Sign,
    Helper::MathFloor,
    Helper::RoundHalfAway,
    Helper::RowMean,
    Helper::RowSum,
    Helper::RowL1,
    Helper::RowL2,
    Helper::RowMin,
    Helper::RowMax,
    Helper::RowStd,
    Helper::Pcg32,
    Helper::RngUniform,
    Helper::RngRange,
    Helper::RngGaussian,
    Helper::StageRng,
    Helper::SortedColumn,
    Helper::Quantile,
    Helper::LowerBound,
    Helper::MidRank,
    Helper::QuantileEdges,
    Helper::DftMags,
    Helper::Pearson,
    Helper::RbfRow,
    Helper::StdFit,
];

const B0: &str = "(pipe X (ridge))";

/// Memoizing measurement engine: `t(genome)` = emitted (code+doc) totals minus
/// the reconstructed module doc, per language; `hs(genome)` = helper closure.
struct Bench {
    cache: BTreeMap<String, (Pair, BTreeSet<Helper>)>,
}

impl Bench {
    fn new() -> Bench {
        Bench {
            cache: BTreeMap::new(),
        }
    }

    fn entry(&mut self, src: &str) -> (Pair, BTreeSet<Helper>) {
        if let Some(v) = self.cache.get(src) {
            return v.clone();
        }
        let genome = GenomeIr::parse(src)
            .unwrap_or_else(|e| panic!("calibration genome failed to parse: {src}: {e}"));
        let opts = EmitOptions::default();
        let emitted = crate::transpile::emit(&genome, &opts)
            .unwrap_or_else(|e| panic!("calibration genome failed to emit: {src}: {e}"));
        let plan = spec::build_plan(&genome)
            .unwrap_or_else(|e| panic!("calibration genome failed to plan: {src}: {e:?}"));
        let md = moddoc_pair(&plan, &opts);
        let t = Pair {
            py: (emitted.py_tokens.code_tokens + emitted.py_tokens.doc_tokens) as i64 - md.py,
            rs: (emitted.rs_tokens.code_tokens + emitted.rs_tokens.doc_tokens) as i64 - md.rs,
        };
        let hs = spec::helper_set(&plan);
        self.cache.insert(src.to_string(), (t, hs.clone()));
        (t, hs)
    }

    fn t(&mut self, src: &str) -> Pair {
        self.entry(src).0
    }

    fn hs(&mut self, src: &str) -> BTreeSet<Helper> {
        self.entry(src).1
    }

    fn plan(&mut self, src: &str) -> Plan {
        let genome = GenomeIr::parse(src).unwrap();
        spec::build_plan(&genome).unwrap()
    }
}

/// Set-intersection solve: `T(a) + T(b) - T(ab) - T(base)` cancels every
/// section and leaves `H[(closure(a) ∩ closure(b)) \ shared]`; subtracting the
/// already-solved helpers isolates `target`. The coefficient bookkeeping is
/// done numerically from the real helper closures, and the step asserts that
/// `target` is the only unsolved helper left with a nonzero coefficient.
fn solve_intersection(
    bench: &mut Bench,
    solved: &BTreeMap<Helper, Pair>,
    target: Helper,
    a: &str,
    b: &str,
    ab: &str,
    base: &str,
) -> Pair {
    let mut m = bench.t(a) + bench.t(b) - bench.t(ab) - bench.t(base);
    let (ha, hb, hab, hbase) = (bench.hs(a), bench.hs(b), bench.hs(ab), bench.hs(base));
    for h in CAL_HELPERS {
        let c = ha.contains(&h) as i64 + hb.contains(&h) as i64
            - hab.contains(&h) as i64
            - hbase.contains(&h) as i64;
        if h == target {
            assert_eq!(
                c, 1,
                "calibration: target {h:?} coefficient != 1 in {a} / {b}"
            );
        } else if c != 0 {
            let known = solved
                .get(&h)
                .unwrap_or_else(|| panic!("calibration: {h:?} needed before {target:?}"));
            m -= *known * c;
        }
    }
    m
}

/// Formula-stage solve: a map stage is never deduped, so the delta between a
/// two-copy and a one-copy genome is exactly (section + call lines); the delta
/// between the one-copy genome and the base is (attr + section + call lines +
/// new helpers). first - marginal - attr isolates the expression's helpers.
fn solve_expr_helpers(
    bench: &mut Bench,
    solved: &BTreeMap<Helper, Pair>,
    attr: Pair,
    target: Helper,
    expr: &str,
) -> Pair {
    let g1 = format!("(pipe X (map expr=\"{expr}\") (ridge))");
    let g2 = format!("(pipe X (map expr=\"{expr}\") (map expr=\"{expr}\") (ridge))");
    let mut m = bench.t(&g1) * 2 - bench.t(&g2) - bench.t(B0) - attr;
    let (h1, h0) = (bench.hs(&g1), bench.hs(B0));
    for h in CAL_HELPERS {
        let c = h1.contains(&h) as i64 - h0.contains(&h) as i64;
        if h == target {
            assert_eq!(
                c, 1,
                "calibration: target {h:?} coefficient != 1 for expr {expr}"
            );
        } else if c != 0 {
            let known = solved
                .get(&h)
                .unwrap_or_else(|| panic!("calibration: {h:?} needed before {target:?}"));
            m -= *known * c;
        }
    }
    m
}

fn build_calibration() -> Calibration {
    let bench = &mut Bench::new();
    let mut helpers: BTreeMap<Helper, Pair> = BTreeMap::new();

    // ---- per-helper costs via intersection equations (order matters) ----
    let steps: [(Helper, &str, &str, &str, &str); 17] = [
        (
            Helper::Clamp,
            "(pipe X (cheb) (histgbm))",
            "(pipe X (map expr=\"x\") (histgbm))",
            "(pipe X (cheb) (map expr=\"x\") (histgbm))",
            "(pipe X (histgbm))",
        ),
        (
            Helper::Quantile,
            "(pipe X (row_stats) (ridge))",
            "(pipe X (winsor_clip) (ridge))",
            "(pipe X (row_stats) (winsor_clip) (ridge))",
            B0,
        ),
        (
            Helper::SortedColumn,
            "(pipe X (winsor_clip) (ridge))",
            "(pipe X (robust_scale) (ridge))",
            "(pipe X (winsor_clip) (robust_scale) (ridge))",
            B0,
        ),
        (
            Helper::LowerBound,
            "(pipe X (quantile_bins) (ridge))",
            "(pipe X (rank) (ridge))",
            "(pipe X (quantile_bins) (rank) (ridge))",
            B0,
        ),
        (
            Helper::MidRank,
            "(pipe X (rank) (ridge))",
            "(pipe X (blom) (ridge))",
            "(pipe X (rank) (blom) (ridge))",
            B0,
        ),
        (
            Helper::QuantileEdges,
            "(pipe X (quantile_bins) (ridge))",
            "(pipe X (bin_onehot) (ridge))",
            "(pipe X (quantile_bins) (bin_onehot) (ridge))",
            B0,
        ),
        (
            Helper::DftMags,
            "(pipe X (fft_mag) (ridge))",
            "(pipe X (row_spectral) (ridge))",
            "(pipe X (fft_mag) (row_spectral) (ridge))",
            B0,
        ),
        (
            Helper::Pearson,
            "(pipe X (pair_prod) (ridge))",
            "(pipe X (row_diff_stats) (ridge))",
            "(pipe X (pair_prod) (row_diff_stats) (ridge))",
            B0,
        ),
        (
            Helper::Pcg32,
            "(pipe X (nystrom) (ridge))",
            "(pipe X (krr))",
            "(pipe X (nystrom) (krr))",
            B0,
        ),
        (
            Helper::StdFit,
            "(pipe X (standardize) (histgbm))",
            "(pipe X (ksmooth))",
            "(pipe X (standardize) (ksmooth))",
            "(pipe X (histgbm))",
        ),
        (
            Helper::StageRng,
            "(pipe X (rand_proj) (ridge))",
            "(pipe X (nystrom) (ridge))",
            "(pipe X (rand_proj) (nystrom) (ridge))",
            B0,
        ),
        (
            Helper::RbfRow,
            "(pipe X (landmark_dist) (ridge))",
            "(pipe X (nystrom) (ridge))",
            "(pipe X (landmark_dist) (nystrom) (ridge))",
            B0,
        ),
        (
            Helper::RngUniform,
            "(pipe X (rand_proj) (ridge))",
            "(pipe X (histgbm2))",
            "(pipe X (rand_proj) (histgbm2))",
            B0,
        ),
        (
            Helper::RngRange,
            "(pipe X (rff_cos) (ridge))",
            "(pipe X (landmark_dist) (ridge))",
            "(pipe X (rff_cos) (landmark_dist) (ridge))",
            B0,
        ),
        (
            Helper::RngGaussian,
            "(pipe X (rand_proj) (ridge))",
            "(pipe X (rff_cos) (ridge))",
            "(pipe X (rand_proj) (rff_cos) (ridge))",
            B0,
        ),
        (
            Helper::SafeCos,
            "(pipe X (rff_cos) (ridge))",
            "(pipe X (map expr=\"cos(x)\") (ridge))",
            "(pipe X (rff_cos) (map expr=\"cos(x)\") (ridge))",
            B0,
        ),
        (
            Helper::SafeSin,
            "(pipe X (rff_sin) (ridge))",
            "(pipe X (map expr=\"sin(x)\") (ridge))",
            "(pipe X (rff_sin) (map expr=\"sin(x)\") (ridge))",
            B0,
        ),
    ];
    for (target, a, b, ab, base) in steps {
        let v = solve_intersection(bench, &helpers, target, a, b, ab, base);
        helpers.insert(target, v);
    }

    // Rust formula attribute (`#![allow(unused_parens)]`): first map minus its
    // own duplication marginal; the Python component must measure exactly 0.
    let map_x1 = "(pipe X (map expr=\"x\") (ridge))";
    let map_x2 = "(pipe X (map expr=\"x\") (map expr=\"x\") (ridge))";
    let formula_attr = bench.t(map_x1) * 2 - bench.t(map_x2) - bench.t(B0);
    assert_eq!(
        formula_attr.py, 0,
        "calibration: unexpected Python formula attribute"
    );
    assert!(
        formula_attr.rs >= 0,
        "calibration: negative Rust formula attribute"
    );

    // Expression-driven helpers, isolated through never-deduped map stages.
    let expr_steps: [(Helper, &str); 21] = [
        (Helper::MathDiv, "x/2"),
        (Helper::PowGuard, "x^2"),
        (Helper::Fmax, "sqrt(x)"),
        (Helper::Fmin, "min(x, 1)"),
        (Helper::MathSigmoid, "sigmoid(x)"),
        (Helper::MathLog1p, "log1p(x)"),
        (Helper::MathClip, "clip(x, 0, 1)"),
        (Helper::Asinh, "asinh(x)"),
        (Helper::MathExp, "exp(x)"),
        (Helper::MathLog, "log(x)"),
        (Helper::Erf, "erf(x)"),
        (Helper::Sign, "sign(x)"),
        (Helper::MathFloor, "floor(x)"),
        (Helper::RoundHalfAway, "round(x)"),
        (Helper::RowMean, "x + row_mean"),
        (Helper::RowSum, "x + row_sum"),
        (Helper::RowL1, "x + row_l1"),
        (Helper::RowL2, "x + row_l2"),
        (Helper::RowMin, "x + row_min"),
        (Helper::RowMax, "x + row_max"),
        (Helper::RowStd, "x + row_std"),
    ];
    for (target, expr) in expr_steps {
        let v = solve_expr_helpers(bench, &helpers, formula_attr, target, expr);
        helpers.insert(target, v);
    }

    // Net-of-helpers measurement: subtract every calibrated helper the genome's
    // closure contains, leaving sections + folded helpers only.
    let h_sum = |bench: &mut Bench, src: &str, helpers: &BTreeMap<Helper, Pair>| -> Pair {
        let hs = bench.hs(src);
        let mut s = Pair::default();
        for h in &hs {
            if let Some(c) = helpers.get(h) {
                s += *c;
            }
        }
        s
    };
    let n_of = |bench: &mut Bench, src: &str, helpers: &BTreeMap<Helper, Pair>| -> Pair {
        let t = bench.t(src);
        t - h_sum(bench, src, helpers)
    };

    // ---- heads ----
    let head_refs: [&str; 11] = [
        B0,
        "(pipe X (ridge_gcv))",
        "(pipe X (elasticnet))",
        "(pipe X (histgbm))",
        "(pipe X (ensemble))",
        "(pipe X (histgbm2))",
        "(pipe X (krr))",
        "(pipe X (gpr))",
        "(pipe X (gpr kernel=rq))",
        "(pipe X (ksmooth))",
        "(pipe X (stack))",
    ];
    let mut head_base: BTreeMap<&'static str, Pair> = BTreeMap::new();
    let mut head_ref_lit: BTreeMap<&'static str, Vec<i64>> = BTreeMap::new();
    let mut gprk_ref_lit = Pair::default();
    for src in head_refs {
        let plan = bench.plan(src);
        let Plan::Pipeline { head, .. } = &plan else {
            unreachable!()
        };
        let key = spec::head_fn_name(head);
        let dw = word_count(&format!("This head is {}", spec::head_doc(head))) as i64;
        let base = n_of(bench, src, &helpers) - Pair::both(dw);
        head_base.insert(key, base);
        head_ref_lit.insert(
            key,
            head_f64_slots(head)
                .iter()
                .map(|v| lit_tokens(*v))
                .collect(),
        );
        if let PlannedHead::GprKernel { spec: ks, .. } = head {
            gprk_ref_lit = Pair {
                py: code_tok(&py_kernel_terms(ks), Lang::Python),
                rs: code_tok(&rs_kernel_terms(ks), Lang::Rust),
            };
        }
    }

    // Head f64-slot multiplicities: vary one slot to a differently-tokenized
    // literal and divide the (doc-adjusted) delta by the literal-token delta.
    let mut head_mult: BTreeMap<&'static str, Vec<Pair>> = BTreeMap::new();
    let head_mult_specs: [(&str, &str, usize, &str); 8] = [
        (
            "ridge_regression",
            B0,
            0,
            "(pipe X (ridge lambda_frac=0.00001))",
        ),
        (
            "elastic_net",
            "(pipe X (elasticnet))",
            0,
            "(pipe X (elasticnet alpha=0.00001))",
        ),
        (
            "elastic_net",
            "(pipe X (elasticnet))",
            1,
            "(pipe X (elasticnet l1_ratio=0.00001))",
        ),
        (
            "gradient_boosted_trees",
            "(pipe X (histgbm))",
            0,
            "(pipe X (histgbm lr=0.00001))",
        ),
        (
            "gradient_boosted_trees",
            "(pipe X (histgbm))",
            1,
            "(pipe X (histgbm lambda=0.00001))",
        ),
        (
            "blended_regression",
            "(pipe X (ensemble))",
            0,
            "(pipe X (ensemble lambda_frac=0.00001))",
        ),
        (
            "blended_regression",
            "(pipe X (ensemble))",
            1,
            "(pipe X (ensemble lr=0.00001))",
        ),
        (
            "blended_regression",
            "(pipe X (ensemble))",
            2,
            "(pipe X (ensemble lambda=0.00001))",
        ),
    ];
    let slot_mult = |bench: &mut Bench, ref_src: &str, var_src: &str, slot: usize| -> Pair {
        let ref_plan = bench.plan(ref_src);
        let var_plan = bench.plan(var_src);
        let (Plan::Pipeline { head: rh, .. }, Plan::Pipeline { head: vh, .. }) =
            (&ref_plan, &var_plan)
        else {
            unreachable!()
        };
        let dlit = lit_tokens(head_f64_slots(vh)[slot]) - lit_tokens(head_f64_slots(rh)[slot]);
        assert!(dlit != 0, "calibration: literal delta is 0 for {var_src}");
        let ddoc = word_count(&format!("This head is {}", spec::head_doc(vh))) as i64
            - word_count(&format!("This head is {}", spec::head_doc(rh))) as i64;
        let m = bench.t(var_src) - bench.t(ref_src) - Pair::both(ddoc);
        assert!(
            m.py % dlit == 0 && m.rs % dlit == 0,
            "calibration: non-integer head mult"
        );
        Pair {
            py: m.py / dlit,
            rs: m.rs / dlit,
        }
    };
    for (key, ref_src, slot, var_src) in head_mult_specs {
        let m = slot_mult(bench, ref_src, var_src, slot);
        let e = head_mult.entry(key).or_default();
        while e.len() <= slot {
            e.push(Pair::default());
        }
        e[slot] = m;
    }
    // histgbm2: lr is measurable; row/col_subsample are clamped to [0.5, 1.0],
    // where fmt_f64 is always a single token, so their multiplier (identical
    // call-line position to lr) can never produce a nonzero adjustment; the
    // huber delta is measured on a loss=huber pair.
    {
        let m_lr = slot_mult(
            bench,
            "(pipe X (histgbm2))",
            "(pipe X (histgbm2 lr=0.00001))",
            0,
        );
        let m_delta = slot_mult(
            bench,
            "(pipe X (histgbm2 loss=huber delta=0.5))",
            "(pipe X (histgbm2 loss=huber delta=0.00001))",
            3,
        );
        head_mult.insert("leafwise_boosted_trees", vec![m_lr, m_lr, m_lr, m_delta]);
    }
    {
        let m = slot_mult(
            bench,
            "(pipe X (ksmooth))",
            "(pipe X (ksmooth gamma=0.00001))",
            0,
        );
        head_mult.insert("nadaraya_watson_regression", vec![m]);
    }
    for key in [
        "gcv_ridge_regression",
        "kernel_ridge_regression",
        "gaussian_process_ard_regression",
        "gaussian_process_composite_kernel_regression",
        "greedy_ensemble_selection",
    ] {
        head_mult.insert(key, vec![]);
    }

    // Composite-GP kernel-terms literal multiplicity (fit + predict call).
    let gprk_mult = {
        let g1 = "(pipe X (gpr kernel=rq))";
        let g2 = "(pipe X (gpr kernel=rq+linear))";
        let p1 = bench.plan(g1);
        let p2 = bench.plan(g2);
        let get = |p: &Plan| -> (Pair, i64) {
            let Plan::Pipeline { head, .. } = p else {
                unreachable!()
            };
            let PlannedHead::GprKernel { spec: ks, .. } = head else {
                unreachable!()
            };
            let lits = Pair {
                py: code_tok(&py_kernel_terms(ks), Lang::Python),
                rs: code_tok(&rs_kernel_terms(ks), Lang::Rust),
            };
            (
                lits,
                word_count(&format!("This head is {}", spec::head_doc(head))) as i64,
            )
        };
        let (l1, d1) = get(&p1);
        let (l2, d2) = get(&p2);
        let m = bench.t(g2) - bench.t(g1) - Pair::both(d2 - d1);
        let dl = l2 - l1;
        assert!(
            dl.py != 0 && dl.rs != 0,
            "calibration: kernel literal delta is 0"
        );
        assert!(
            m.py % dl.py == 0 && m.rs % dl.rs == 0,
            "calibration: non-integer gprk mult"
        );
        Pair {
            py: m.py / dl.py,
            rs: m.rs / dl.rs,
        }
    };

    // ---- ops ----
    let op_refs: [&str; 34] = [
        "symlog",
        "select_corr",
        "impute",
        "pair_ratio",
        "spline",
        "standardize",
        "robust_scale",
        "rank",
        "blom",
        "quantile_bins",
        "bin_onehot",
        "hinge",
        "winsor_clip",
        "cheb",
        "poly_interact",
        "rff_cos",
        "rff_sin",
        "rand_proj",
        "landmark_dist",
        "oblique_hinge",
        "row_stats",
        "pair_prod",
        "target_bin_encode",
        "sigmoid",
        "step",
        "norm_row",
        "polar",
        "lag",
        "roll",
        "fft_mag",
        "row_spectral",
        "row_diff_stats",
        "svd_project",
        "nystrom",
    ];
    let mut op_sec: BTreeMap<&'static str, Pair> = BTreeMap::new();
    let mut op_calls: BTreeMap<&'static str, Pair> = BTreeMap::new();
    let mut op_ref_lit: BTreeMap<&'static str, i64> = BTreeMap::new();
    let n_base = n_of(bench, B0, &helpers);
    for op in op_refs {
        let g1 = format!("(pipe X ({op}) (ridge))");
        let g2 = format!("(pipe X ({op}) ({op}) (ridge))");
        let plan = bench.plan(&g1);
        let Plan::Pipeline { stages, .. } = &plan else {
            unreachable!()
        };
        let planned = &stages[0].op;
        let key = spec::op_fn_name(planned);
        let dw = word_count(&format!("This transform {}", spec::op_doc(planned))) as i64;
        let sec = n_of(bench, &g1, &helpers) - n_base - Pair::both(dw);
        op_sec.insert(key, sec);
        op_calls.insert(key, bench.t(&g2) - bench.t(&g1));
        if let Some(v) = op_f64_slot(planned) {
            op_ref_lit.insert(key, lit_tokens(v));
        }
    }

    // Op f64-slot multiplicities (each of these ops has exactly one f64 slot).
    let mut op_mult: BTreeMap<&'static str, Pair> = BTreeMap::new();
    let op_mult_specs: [(&str, &str); 9] = [
        ("symlog", "(pipe X (symlog c=0.00001) (ridge))"),
        ("impute", "(pipe X (impute value=0.00001) (ridge))"),
        ("winsor_clip", "(pipe X (winsor_clip q=0.00001) (ridge))"),
        ("rff_cos", "(pipe X (rff_cos gamma=0.00001) (ridge))"),
        ("rff_sin", "(pipe X (rff_sin gamma=0.00001) (ridge))"),
        (
            "landmark_dist",
            "(pipe X (landmark_dist gamma=0.00001) (ridge))",
        ),
        (
            "target_bin_encode",
            "(pipe X (target_bin_encode prior=0.00001) (ridge))",
        ),
        ("step", "(pipe X (step t=-1.5) (ridge))"),
        ("nystrom", "(pipe X (nystrom gamma=0.00001) (ridge))"),
    ];
    for (op, var_src) in op_mult_specs {
        let ref_src = format!("(pipe X ({op}) (ridge))");
        let ref_plan = bench.plan(&ref_src);
        let var_plan = bench.plan(var_src);
        let (Plan::Pipeline { stages: rs_, .. }, Plan::Pipeline { stages: vs_, .. }) =
            (&ref_plan, &var_plan)
        else {
            unreachable!()
        };
        let key = spec::op_fn_name(&rs_[0].op);
        let dlit = lit_tokens(op_f64_slot(&vs_[0].op).unwrap())
            - lit_tokens(op_f64_slot(&rs_[0].op).unwrap());
        assert!(
            dlit != 0,
            "calibration: op literal delta is 0 for {var_src}"
        );
        let ddoc = word_count(&format!("This transform {}", spec::op_doc(&vs_[0].op))) as i64
            - word_count(&format!("This transform {}", spec::op_doc(&rs_[0].op))) as i64;
        let m = bench.t(var_src) - bench.t(&ref_src) - Pair::both(ddoc);
        assert!(
            m.py % dlit == 0 && m.rs % dlit == 0,
            "calibration: non-integer op mult"
        );
        op_mult.insert(
            key,
            Pair {
                py: m.py / dlit,
                rs: m.rs / dlit,
            },
        );
    }

    // ---- formula stage call lines ----
    // Marginal of a duplicated formula stage minus its reconstructed section
    // leaves exactly the Model-class call lines it adds.
    let map_calls = {
        let marg = bench.t(map_x2) - bench.t(map_x1);
        let expr = MathAst::parse("x").unwrap();
        marg - Pair {
            py: total_tok(&py_map_src(2, "x", &expr), Lang::Python),
            rs: total_tok(&rs_map_src(2, "x", &expr), Lang::Rust),
        }
    };
    let derive_calls = {
        let g1 = "(pipe X (derive expr=\"c0\") (ridge))";
        let g2 = "(pipe X (derive expr=\"c0\") (derive expr=\"c0\") (ridge))";
        let marg = bench.t(g2) - bench.t(g1);
        let expr = MathAst::parse("c0").unwrap();
        marg - Pair {
            py: total_tok(&py_derive_src(2, "c0", &expr), Lang::Python),
            rs: total_tok(&rs_derive_src(2, "c0", &expr), Lang::Rust),
        }
    };

    // ---- symbolic bases ----
    let mut sym_base = [Pair::default(); 2];
    for (affine, src) in [(false, "(sym raw \"c0\")"), (true, "(sym affine \"c0\")")] {
        let plan = bench.plan(src);
        let Plan::Symbolic { source, expr, .. } = &plan else {
            unreachable!()
        };
        let sec = Pair {
            py: total_tok(&py_expression_fn(source, expr), Lang::Python),
            rs: total_tok(&rs_expression_fn(source, expr), Lang::Rust),
        };
        sym_base[affine as usize] = n_of(bench, src, &helpers) - sec;
    }

    Calibration {
        helpers,
        formula_attr,
        head_base,
        head_mult,
        head_ref_lit,
        gprk_mult,
        gprk_ref_lit,
        op_sec,
        op_calls,
        op_mult,
        op_ref_lit,
        map_calls,
        derive_calls,
        sym_base,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counting_is_deterministic_and_whitespace_immune() {
        let a = count_tokens("x = foo(1, 2.5)  # add two numbers", Lang::Python);
        let b_ = count_tokens("x   =   foo(1,   2.5)  # add two numbers", Lang::Python);
        assert_eq!(a.code_tokens, b_.code_tokens);
        assert_eq!(a.doc_tokens, 3);
        assert!(a.code_tokens >= 8);
    }

    #[test]
    fn javascript_comments_and_quoted_strings_are_counted_exactly() {
        let count = count_tokens(
            "// two words\nconst s = 'a//b'; const t = \"x\";",
            Lang::JavaScript,
        );
        assert_eq!(count.doc_tokens, 2);
        assert_eq!(count.code_tokens, 10);
        assert_eq!(count.bytes, 45);
    }

    #[test]
    fn literal_token_costs() {
        assert_eq!(lit_tokens(0.05), 1);
        assert_eq!(lit_tokens(-0.5), 2);
        assert_eq!(lit_tokens(0.00001), 3); // "1e-05"
        assert_eq!(lit_tokens(10.0), 1);
    }
}
