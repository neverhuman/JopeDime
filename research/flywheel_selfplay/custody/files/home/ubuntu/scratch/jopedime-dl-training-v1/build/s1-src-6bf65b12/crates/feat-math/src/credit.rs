//! Token-credit accounting for the invention run.
//!
//! Every genome the search *evaluates* is a complete, working algorithm we could have shipped as
//! full dual-language source (the emitters produce it on demand). The ledger therefore credits each
//! fresh evaluation with the estimated token count of its FULL emitted Python + Rust artifacts, and
//! tracks a live tokens/sec rate. Estimation uses a lazily-built per-op token table (probe genomes
//! emitted once per op — emission is pure string assembly, microseconds each); genuine improvements
//! are additionally credited with their REAL emitted token counts.

use crate::dsl::DslExpr;
use crate::transpile::{emit, EmitOptions};
use crate::GenomeIr;
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

/// Fall-through cost for an op whose probe emission failed (roughly one helper block).
const DEFAULT_OP_TOKENS: f64 = 900.0;
/// Sliding window for the live rate, in seconds.
const RATE_WINDOW_SECS: f64 = 2.0;

pub struct TokenLedger {
    /// op name -> estimated (python + rust, code + doc) tokens contributed by that stage.
    op_cost: HashMap<String, f64>,
    /// tokens of the minimal `(pipe X (ridge))` program (boilerplate + ridge head).
    skeleton: f64,
    /// tokens of `(pipe X (<head>))` per head (full program with that head).
    head_total: HashMap<String, f64>,
    total: f64,
    started: Instant,
    window: VecDeque<(Instant, f64)>,
}

fn emitted_tokens(dsl: &str) -> Option<f64> {
    let genome = GenomeIr::parse(dsl).ok()?;
    let e = emit(&genome, &EmitOptions::default()).ok()?;
    Some(
        (e.py_tokens.code_tokens
            + e.py_tokens.doc_tokens
            + e.rs_tokens.code_tokens
            + e.rs_tokens.doc_tokens) as f64,
    )
}

impl TokenLedger {
    pub fn new() -> Self {
        let skeleton = emitted_tokens("(pipe X (ridge))").unwrap_or(4000.0);
        TokenLedger {
            op_cost: HashMap::new(),
            skeleton,
            head_total: HashMap::new(),
            total: 0.0,
            started: Instant::now(),
            window: VecDeque::new(),
        }
    }

    fn op_cost(&mut self, op: &str) -> f64 {
        if let Some(c) = self.op_cost.get(op) {
            return *c;
        }
        let cost = emitted_tokens(&format!("(pipe X ({op}) (ridge))"))
            .map(|t| (t - self.skeleton).max(120.0))
            .unwrap_or(DEFAULT_OP_TOKENS);
        self.op_cost.insert(op.to_string(), cost);
        cost
    }

    fn head_cost(&mut self, head: &str) -> f64 {
        if let Some(c) = self.head_total.get(head) {
            return *c;
        }
        let total = emitted_tokens(&format!("(pipe X ({head}))")).unwrap_or(self.skeleton);
        self.head_total.insert(head.to_string(), total);
        total
    }

    /// Estimated full-artifact tokens (Python + Rust) for a genome, from the per-op table.
    pub fn estimate(&mut self, dsl: &DslExpr, head_ops: &[&str]) -> f64 {
        let mut tokens = 0.0;
        let mut have_head = false;
        if let Some(last) = dsl.stages.last() {
            if head_ops.contains(&last.op.as_str()) {
                tokens += self.head_cost(&last.op.clone());
                have_head = true;
            }
        }
        if !have_head {
            tokens += self.skeleton; // implicit ridge head
        }
        let n = dsl.stages.len();
        for (i, st) in dsl.stages.iter().enumerate() {
            if have_head && i + 1 == n {
                continue; // already counted as the head program
            }
            tokens += self.op_cost(&st.op.clone());
        }
        tokens
    }

    /// Credit one evaluated genome; returns the tokens credited.
    pub fn credit(&mut self, dsl: &DslExpr, head_ops: &[&str]) -> f64 {
        let t = self.estimate(dsl, head_ops);
        self.add(t);
        t
    }

    /// Add raw tokens (e.g. the REAL counts of an emitted improvement artifact).
    pub fn add(&mut self, tokens: f64) {
        self.total += tokens;
        let now = Instant::now();
        self.window.push_back((now, self.total));
        while let Some(front) = self.window.front() {
            if now.duration_since(front.0).as_secs_f64() > RATE_WINDOW_SECS {
                self.window.pop_front();
            } else {
                break;
            }
        }
    }

    pub fn total(&self) -> f64 {
        self.total
    }

    /// Live tokens/sec over the sliding window (lifetime average until the window fills).
    pub fn rate(&self) -> f64 {
        if self.window.len() >= 2 {
            let (t0, v0) = *self.window.front().unwrap();
            let (t1, v1) = *self.window.back().unwrap();
            let dt = t1.duration_since(t0).as_secs_f64();
            if dt > 0.05 {
                return (v1 - v0) / dt;
            }
        }
        let dt = self.started.elapsed().as_secs_f64();
        if dt > 0.0 {
            self.total / dt
        } else {
            0.0
        }
    }
}

impl Default for TokenLedger {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsl::parse_dsl;

    #[test]
    fn ledger_estimates_scale_with_pipeline_size() {
        let mut ledger = TokenLedger::new();
        let heads = ["ridge", "histgbm"];
        let small = parse_dsl("(pipe X (ridge))").unwrap();
        let big =
            parse_dsl("(pipe X (winsor_clip q=0.1) (cheb degree=6) (robust_scale) (histgbm))")
                .unwrap();
        let ts = ledger.estimate(&small, &heads);
        let tb = ledger.estimate(&big, &heads);
        assert!(ts > 1000.0, "skeleton should be substantial: {ts}");
        assert!(
            tb > ts,
            "bigger pipeline must credit more tokens: {tb} <= {ts}"
        );
        ledger.credit(&big, &heads);
        assert!(ledger.total() > 0.0);
    }
}
