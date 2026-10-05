//! Protected-semantics e-graph simplification (B20).
//!
//! Staged: `v2/mod.rs` declares this module only under `#[cfg(test)]` until B19 merges and the
//! search move is wired. Nothing outside its tests calls it.
//!
//! A small in-crate e-graph: union-find, a hashcons keyed by literal bits (so `0.0` and `-0.0`
//! are different nodes), and a rebuild back to congruence after every write phase (Willsey et
//! al., POPL 2021). It saturates under a fixed rewrite set in which every rule is exact under
//! the engine's protected semantics: on every input row whose cells, and every literal, are not
//! signaling NaNs, the rewritten tree gives the same error status, the same root value bits
//! (signed zero included) and the same fault set.
//! - Signaling NaNs are outside that domain. R1, R2, R3 and R10 delete the IEEE arithmetic step
//!   that quiets one, while `-` and `abs` pass it through. The evaluator's `powf` gives 1.0 for
//!   `pow(qNaN, ±0)` and `pow(1, qNaN)` but NaN for a signaling operand, which `safe` maps to 0.0
//!   with NONFINITE_SANITIZED. Text parsing never produces a signaling NaN; raw-bit rows and the
//!   binary codec can, and the search re-evaluates every proposed child natively.
//! - Inside the domain, NaN bits cannot reach an output, because every operation's non-NaN
//!   result ignores them and `safe` maps a non-finite root to 0.0.
//!
//! Extraction picks the smallest equivalent tree (nodes, then depth, then its printed form), so
//! the result never depends on e-class numbering. Limits are counts only (e-nodes and
//! iterations), checked before every application, so the bound is hard. When saturation
//! completes the result does not depend on rule order either; when a limit stops it, the partial
//! write phase can. A tree is replaced only when the result is strictly smaller, or as small and
//! shallower.
//!
//! The rules (proof notes per rule in the tests):
//! - R1 `x*1`, `1*x` → `x`; R2 `x/1` → `x`;
//! - R3 `x + (-0)`, `(-0) + x`, `x - 0` → `x`;
//! - R4 `-(-x)` → `x`; R5 `|-x|` → `|x|`, `||x||` → `|x|`;
//! - R7 `x - y` ↔ `x + (-y)`, `x - (-y)` → `x + y`;
//! - R8 `(-a)*(-b)` → `a*b`, `(-a)*b` → `-(a*b)`, `a*(-b)` → `-(a*b)`;
//! - R9 `x*2` → `x + x`; R10 `x*(-1)` → `-x`, `x/(-1)` → `-x`;
//! - R11 `x/c` → `x*(1/c)` for `c = ±2^k` with `-39 ≤ k ≤ 1023` (the protected division's
//!   threshold is 1e-12, and 2^-40 is below it);
//! - constant folding of `+ - *` and of `/` with a divisor of magnitude at least 1e-12, to finite
//!   results only, with the evaluator's own arithmetic; and of `abs` of any literal, which is
//!   exact because it only clears the sign bit.
//!
//! Pinned as outside the exact domain: R1, R2, R3 and R10 on signaling-NaN rows under `pow`.
//! Excluded, each pinned by a counterexample test: `x + 0.0`, `x - (-0.0)`, `x*0`, `x - x`, `x/x`,
//! reassociation, distribution, `a - b` → `-(b - a)`, operand commutation (on an error row it
//! changes which operand's faults are raised before the Hard one), `(-a)/b` ↔ `-(a/b)` (the
//! protected branch returns +0.0 for both), `x^2` ↔ `x*x` (`powf` precision is unspecified), and
//! any folding of libm-backed functions.

use super::*;
use std::collections::{BTreeMap, BTreeSet};

type Id = usize;

/// The most saturation rounds.
const MAX_ITERATIONS: usize = 8;
/// The hard e-node cap: eight times the input's nodes, at most 512.
const NODE_CAP_PER_INPUT_NODE: usize = 8;
const NODE_CAP_MAX: usize = 512;

const ADD: u8 = 0;
const SUB: u8 = 1;
const MUL: u8 = 2;
const DIV: u8 = 3;
const POW: u8 = 4;

fn tag(op: BinOp) -> u8 {
    match op {
        BinOp::Add => ADD,
        BinOp::Sub => SUB,
        BinOp::Mul => MUL,
        BinOp::Div => DIV,
        BinOp::Pow => POW,
    }
}

fn op(tag: u8) -> BinOp {
    [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div, BinOp::Pow][tag as usize]
}

/// An e-node: an operator over e-class ids. Literals are keyed by their bits.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Node {
    Const(u64),
    Var(Var),
    Neg(Id),
    Bin(u8, Id, Id),
    Call(Func, Vec<Id>),
}

/// A right-hand side over existing e-classes.
#[derive(Clone, Debug)]
enum Term {
    Class(Id),
    Literal(f64),
    Neg(Box<Term>),
    Bin(u8, Box<Term>, Box<Term>),
    Call(Func, Vec<Term>),
}

/// Why saturation stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Saturation {
    /// A round added nothing.
    Saturated,
    IterationLimit,
    /// The next application would have passed the e-node cap.
    NodeLimit,
}

/// The rules, in their fixed application order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Rule {
    MulOne,
    DivOne,
    AddNegZero,
    SubZero,
    NegNeg,
    AbsNeg,
    AbsAbs,
    SubToAddNeg,
    AddNegToSub,
    SubNegToAdd,
    MulNegNeg,
    MulNegLeft,
    MulNegRight,
    MulTwo,
    MulMinusOne,
    DivMinusOne,
    DivPowerOfTwo,
    Fold,
}

const RULES: [Rule; 18] = [
    Rule::MulOne,
    Rule::DivOne,
    Rule::AddNegZero,
    Rule::SubZero,
    Rule::NegNeg,
    Rule::AbsNeg,
    Rule::AbsAbs,
    Rule::SubToAddNeg,
    Rule::AddNegToSub,
    Rule::SubNegToAdd,
    Rule::MulNegNeg,
    Rule::MulNegLeft,
    Rule::MulNegRight,
    Rule::MulTwo,
    Rule::MulMinusOne,
    Rule::DivMinusOne,
    Rule::DivPowerOfTwo,
    Rule::Fold,
];

#[derive(Default)]
struct EGraph {
    parent: Vec<Id>,
    classes: Vec<BTreeSet<Node>>,
    memo: BTreeMap<Node, Id>,
}

impl EGraph {
    fn find(&self, mut id: Id) -> Id {
        while self.parent[id] != id {
            id = self.parent[id];
        }
        id
    }

    fn canonical(&self, node: &Node) -> Node {
        match node {
            Node::Neg(a) => Node::Neg(self.find(*a)),
            Node::Bin(t, a, b) => Node::Bin(*t, self.find(*a), self.find(*b)),
            Node::Call(f, args) => Node::Call(*f, args.iter().map(|a| self.find(*a)).collect()),
            other => other.clone(),
        }
    }

    /// The number of distinct e-nodes.
    fn enodes(&self) -> usize {
        self.memo.len()
    }

    fn add(&mut self, node: Node) -> Id {
        let node = self.canonical(&node);
        if let Some(&id) = self.memo.get(&node) {
            return self.find(id);
        }
        let id = self.parent.len();
        self.parent.push(id);
        self.classes.push(BTreeSet::from([node.clone()]));
        self.memo.insert(node, id);
        id
    }

    /// Merges two classes; the lower id stays the root.
    fn union(&mut self, a: Id, b: Id) -> bool {
        let (a, b) = (self.find(a), self.find(b));
        if a == b {
            return false;
        }
        let (root, child) = (a.min(b), a.max(b));
        self.parent[child] = root;
        let moved = std::mem::take(&mut self.classes[child]);
        self.classes[root].extend(moved);
        true
    }

    /// Restores congruence: re-canonicalizes every node and merges classes that now hold the
    /// same node, until nothing changes.
    fn roots(&self) -> Vec<Id> {
        (0..self.classes.len())
            .filter(|&id| self.find(id) == id)
            .collect()
    }

    fn rebuild(&mut self) {
        loop {
            let mut memo: BTreeMap<Node, Id> = BTreeMap::new();
            let mut merges = Vec::new();
            for id in self.roots() {
                let nodes: BTreeSet<Node> =
                    self.classes[id].iter().map(|n| self.canonical(n)).collect();
                for node in &nodes {
                    match memo.get(node) {
                        Some(&other) if self.find(other) != id => merges.push((other, id)),
                        Some(_) => {}
                        None => {
                            memo.insert(node.clone(), id);
                        }
                    }
                }
                self.classes[id] = nodes;
            }
            let mut changed = false;
            for (a, b) in merges {
                changed |= self.union(a, b);
            }
            if !changed {
                self.memo = memo.into_iter().map(|(n, id)| (n, self.find(id))).collect();
                return;
            }
        }
    }

    fn insert(&mut self, term: &Term) -> Id {
        match term {
            Term::Class(id) => self.find(*id),
            Term::Literal(v) => {
                if v.is_sign_negative() {
                    let inner = self.add(Node::Const((-v).to_bits()));
                    self.add(Node::Neg(inner))
                } else {
                    self.add(Node::Const(v.to_bits()))
                }
            }
            Term::Neg(a) => {
                let a = self.insert(a);
                self.add(Node::Neg(a))
            }
            Term::Bin(t, a, b) => {
                let (a, b) = (self.insert(a), self.insert(b));
                self.add(Node::Bin(*t, a, b))
            }
            Term::Call(f, args) => {
                let args = args.iter().map(|a| self.insert(a)).collect();
                self.add(Node::Call(*f, args))
            }
        }
    }

    /// The literal value a class holds: a constant, or the negation of one.
    fn literal(&self, id: Id) -> Option<f64> {
        let id = self.find(id);
        let mut found = None;
        for node in &self.classes[id] {
            match node {
                Node::Const(bits) => return Some(f64::from_bits(*bits)),
                Node::Neg(inner) if found.is_none() => {
                    found = self.classes[self.find(*inner)]
                        .iter()
                        .find_map(|n| match n {
                            Node::Const(bits) => Some(-f64::from_bits(*bits)),
                            _ => None,
                        });
                }
                _ => {}
            }
        }
        found
    }

    fn negations(&self, id: Id) -> Vec<Id> {
        self.classes[self.find(id)]
            .iter()
            .filter_map(|n| match n {
                Node::Neg(a) => Some(*a),
                _ => None,
            })
            .collect()
    }

    fn holds_abs(&self, id: Id) -> bool {
        self.classes[self.find(id)]
            .iter()
            .any(|n| matches!(n, Node::Call(Func::Abs, _)))
    }
}

fn is(value: Option<f64>, expected: f64) -> bool {
    value.is_some_and(|v| v.to_bits() == expected.to_bits())
}

/// `c = ±2^k` with `-39 ≤ k ≤ 1023`, `|c| ≠ 1`: dividing by `c` is multiplying by the exact `1/c`,
/// and `|c|` clears the protected division's 1e-12 threshold.
fn exact_reciprocal(c: f64) -> Option<f64> {
    let bits = c.abs().to_bits();
    let exponent = (bits >> 52) as i64;
    let mantissa = bits & ((1u64 << 52) - 1);
    let k = exponent - 1023;
    (mantissa == 0 && (-39..=1023).contains(&k) && k != 0).then(|| 1.0 / c)
}

/// The right-hand sides `rule` gives for `node`.
fn matches(eg: &EGraph, rule: Rule, node: &Node) -> Vec<Term> {
    let lit = |id: Id| eg.literal(id);
    match (rule, node) {
        (Rule::MulOne, Node::Bin(MUL, x, y)) => {
            let mut out = Vec::new();
            if is(lit(*y), 1.0) {
                out.push(Term::Class(*x));
            }
            if is(lit(*x), 1.0) {
                out.push(Term::Class(*y));
            }
            out
        }
        (Rule::DivOne, Node::Bin(DIV, x, y)) if is(lit(*y), 1.0) => vec![Term::Class(*x)],
        (Rule::AddNegZero, Node::Bin(ADD, x, y)) => {
            let mut out = Vec::new();
            if is(lit(*y), -0.0) {
                out.push(Term::Class(*x));
            }
            if is(lit(*x), -0.0) {
                out.push(Term::Class(*y));
            }
            out
        }
        (Rule::SubZero, Node::Bin(SUB, x, y)) if is(lit(*y), 0.0) => vec![Term::Class(*x)],
        (Rule::NegNeg, Node::Neg(x)) => eg.negations(*x).into_iter().map(Term::Class).collect(),
        (Rule::AbsNeg, Node::Call(Func::Abs, args)) => eg
            .negations(args[0])
            .into_iter()
            .map(|a| Term::Call(Func::Abs, vec![Term::Class(a)]))
            .collect(),
        (Rule::AbsAbs, Node::Call(Func::Abs, args)) if eg.holds_abs(args[0]) => {
            vec![Term::Class(args[0])]
        }
        (Rule::SubToAddNeg, Node::Bin(SUB, x, y)) => vec![Term::Bin(
            ADD,
            Box::new(Term::Class(*x)),
            Box::new(Term::Neg(Box::new(Term::Class(*y)))),
        )],
        (Rule::AddNegToSub, Node::Bin(ADD, x, y)) => eg
            .negations(*y)
            .into_iter()
            .map(|b| Term::Bin(SUB, Box::new(Term::Class(*x)), Box::new(Term::Class(b))))
            .collect(),
        (Rule::SubNegToAdd, Node::Bin(SUB, x, y)) => eg
            .negations(*y)
            .into_iter()
            .map(|b| Term::Bin(ADD, Box::new(Term::Class(*x)), Box::new(Term::Class(b))))
            .collect(),
        (Rule::MulNegNeg, Node::Bin(MUL, x, y)) => {
            let mut out = Vec::new();
            for a in eg.negations(*x) {
                for b in eg.negations(*y) {
                    out.push(Term::Bin(
                        MUL,
                        Box::new(Term::Class(a)),
                        Box::new(Term::Class(b)),
                    ));
                }
            }
            out
        }
        (Rule::MulNegLeft, Node::Bin(MUL, x, y)) => eg
            .negations(*x)
            .into_iter()
            .map(|a| {
                Term::Neg(Box::new(Term::Bin(
                    MUL,
                    Box::new(Term::Class(a)),
                    Box::new(Term::Class(*y)),
                )))
            })
            .collect(),
        (Rule::MulNegRight, Node::Bin(MUL, x, y)) => eg
            .negations(*y)
            .into_iter()
            .map(|b| {
                Term::Neg(Box::new(Term::Bin(
                    MUL,
                    Box::new(Term::Class(*x)),
                    Box::new(Term::Class(b)),
                )))
            })
            .collect(),
        (Rule::MulTwo, Node::Bin(MUL, x, y)) if is(lit(*y), 2.0) => {
            vec![Term::Bin(
                ADD,
                Box::new(Term::Class(*x)),
                Box::new(Term::Class(*x)),
            )]
        }
        (Rule::MulMinusOne, Node::Bin(MUL, x, y)) if is(lit(*y), -1.0) => {
            vec![Term::Neg(Box::new(Term::Class(*x)))]
        }
        (Rule::DivMinusOne, Node::Bin(DIV, x, y)) if is(lit(*y), -1.0) => {
            vec![Term::Neg(Box::new(Term::Class(*x)))]
        }
        (Rule::DivPowerOfTwo, Node::Bin(DIV, x, y)) => match lit(*y).and_then(exact_reciprocal) {
            Some(r) if r.abs() != 1.0 => {
                vec![Term::Bin(
                    MUL,
                    Box::new(Term::Class(*x)),
                    Box::new(Term::Literal(r)),
                )]
            }
            _ => Vec::new(),
        },
        (Rule::Fold, Node::Bin(t, x, y)) if *t != POW => match (lit(*x), lit(*y)) {
            (Some(a), Some(b)) if *t != DIV || b.abs() >= 1e-12 => {
                let v = compute_binary(op(*t), a, b);
                if v.is_finite() {
                    vec![Term::Literal(v)]
                } else {
                    Vec::new()
                }
            }
            _ => Vec::new(),
        },
        (Rule::Fold, Node::Call(Func::Abs, args)) => match lit(args[0]) {
            Some(a) => vec![Term::Literal(a.abs())],
            None => Vec::new(),
        },
        _ => Vec::new(),
    }
}

/// The result of [`simplify`].
#[derive(Clone, Debug)]
struct Simplified {
    expression: MathExpr,
    changed: bool,
    stop: Saturation,
    iterations: usize,
    enodes: usize,
    applications: usize,
}

/// The saturation limits: rounds, and the hard e-node cap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Limits {
    iterations: usize,
    enodes: usize,
}

impl Limits {
    /// The production limits for an input of `nodes` nodes.
    fn for_input(nodes: usize) -> Self {
        Self {
            iterations: MAX_ITERATIONS,
            enodes: (NODE_CAP_PER_INPUT_NODE * nodes).clamp(16, NODE_CAP_MAX),
        }
    }
}

/// `expression` simplified under the exact rules, applied in `order`, within `limits`.
fn simplify_with(expression: &MathExpr, order: &[Rule], limits: Limits) -> Simplified {
    let mut eg = EGraph::default();
    let root = add_expr(&mut eg, &expression.root);
    let cap = limits.enodes;
    let mut stop = Saturation::IterationLimit;
    let mut applications = 0usize;
    let mut iterations = 0usize;
    'rounds: for _ in 0..limits.iterations {
        iterations += 1;
        // Read phase: every match, in rule order, then class id, then node order.
        let mut pending: Vec<(Id, Term)> = Vec::new();
        let roots = eg.roots();
        for &rule in order {
            for &id in &roots {
                for node in &eg.classes[id] {
                    for rhs in matches(&eg, rule, node) {
                        pending.push((id, rhs));
                    }
                }
            }
        }
        // Write phase, checking the cap before every application.
        let (before_nodes, before_classes) = (eg.enodes(), live_classes(&eg));
        for (id, rhs) in pending {
            if eg.enodes() + term_size(&rhs) > cap {
                stop = Saturation::NodeLimit;
                eg.rebuild();
                break 'rounds;
            }
            let new = eg.insert(&rhs);
            eg.union(id, new);
            applications += 1;
        }
        eg.rebuild();
        if eg.enodes() == before_nodes && live_classes(&eg) == before_classes {
            stop = Saturation::Saturated;
            break;
        }
    }
    let enodes = eg.enodes();
    let best = extract(&eg);
    let original = (count_nodes(&expression.root), depth(&expression.root));
    let candidate = best
        .get(&eg.find(root))
        .and_then(|(_, _, _, tree)| MathExpr::from_ast(tree.clone()).ok());
    let (expression, changed) = match candidate {
        Some(tree) => {
            let size = (count_nodes(&tree.root), depth(&tree.root));
            if size < original {
                (tree, true)
            } else {
                (expression.clone(), false)
            }
        }
        None => (expression.clone(), false),
    };
    Simplified {
        expression,
        changed,
        stop,
        iterations,
        enodes,
        applications,
    }
}

fn simplify(expression: &MathExpr) -> Simplified {
    simplify_with(
        expression,
        &RULES,
        Limits::for_input(count_nodes(&expression.root)),
    )
}

fn live_classes(eg: &EGraph) -> usize {
    eg.roots().len()
}

/// An upper bound on the e-nodes a right-hand side can add.
fn term_size(term: &Term) -> usize {
    match term {
        Term::Class(_) => 0,
        Term::Literal(_) => 2,
        Term::Neg(a) => 1 + term_size(a),
        Term::Bin(_, a, b) => 1 + term_size(a) + term_size(b),
        Term::Call(_, args) => 1 + args.iter().map(term_size).sum::<usize>(),
    }
}

fn add_expr(eg: &mut EGraph, expr: &Expr) -> Id {
    match expr {
        Expr::Const(v) => eg.add(Node::Const(v.to_bits())),
        Expr::Var(v) => eg.add(Node::Var(*v)),
        Expr::Unary(UnOp::Neg, a) => {
            let a = add_expr(eg, a);
            eg.add(Node::Neg(a))
        }
        Expr::Binary(o, a, b) => {
            let (a, b) = (add_expr(eg, a), add_expr(eg, b));
            eg.add(Node::Bin(tag(*o), a, b))
        }
        Expr::Call(f, args) => {
            let args = args.iter().map(|a| add_expr(eg, a)).collect();
            eg.add(Node::Call(*f, args))
        }
    }
}

type Best = BTreeMap<Id, (usize, usize, String, Expr)>;

/// The smallest tree of every class (nodes, then depth, then its printed form), by a fixed-point
/// pass over the classes.
fn extract(eg: &EGraph) -> Best {
    let mut best: Best = BTreeMap::new();
    let roots = eg.roots();
    loop {
        let mut changed = false;
        for &id in &roots {
            for node in &eg.classes[id] {
                let Some(tree) = build(eg, &best, node) else {
                    continue;
                };
                let key = (count_nodes(&tree), depth(&tree), format!("{tree:?}"));
                let better = best
                    .get(&id)
                    .is_none_or(|(n, d, s, _)| (key.0, key.1, &key.2) < (*n, *d, s));
                if better {
                    best.insert(id, (key.0, key.1, key.2, tree));
                    changed = true;
                }
            }
        }
        if !changed {
            return best;
        }
    }
}

fn build(eg: &EGraph, best: &Best, node: &Node) -> Option<Expr> {
    let child = |id: &Id| best.get(&eg.find(*id)).map(|(_, _, _, tree)| tree.clone());
    Some(match node {
        Node::Const(bits) => Expr::Const(f64::from_bits(*bits)),
        Node::Var(v) => Expr::Var(*v),
        Node::Neg(a) => Expr::Unary(UnOp::Neg, Box::new(child(a)?)),
        Node::Bin(t, a, b) => Expr::Binary(op(*t), Box::new(child(a)?), Box::new(child(b)?)),
        Node::Call(f, args) => Expr::Call(*f, args.iter().map(child).collect::<Option<Vec<_>>>()?),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Pcg32;

    fn source(text: &str) -> String {
        simplify(&MathExpr::parse(text).unwrap())
            .expression
            .to_source()
            .unwrap()
    }

    fn printed(text: &str) -> String {
        MathExpr::parse(text).unwrap().to_source().unwrap()
    }

    /// The boundary corpus: signed zeros, infinities, NaN payloads, subnormals, the protected
    /// division threshold's neighbours, 2^-39 and 2^-40, and the large end.
    fn corpus() -> Vec<f64> {
        let threshold = 1e-12f64;
        vec![
            0.0,
            -0.0,
            1.0,
            -1.0,
            2.0,
            -2.5,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            f64::from_bits(0x7ff8_0000_0000_0001),
            f64::from_bits(0xfff8_0000_0000_0000),
            f64::MIN_POSITIVE,
            -f64::MIN_POSITIVE,
            5e-324,
            -5e-324,
            1e-300,
            -1e-300,
            threshold,
            threshold.next_up(),
            threshold.next_down(),
            2f64.powi(-39),
            2f64.powi(-40),
            1e200,
            -1e200,
            1e308,
            f64::MAX,
            0.1,
            3.0,
        ]
    }

    fn rows() -> Vec<Vec<f64>> {
        let values = corpus();
        let mut rows = Vec::new();
        for (i, a) in values.iter().enumerate() {
            rows.push(vec![
                *a,
                values[(i * 7 + 3) % values.len()],
                values[(i * 11 + 5) % values.len()],
            ]);
            rows.push(vec![*a, *a, -*a]);
        }
        let mut rng = Pcg32::new(0xB20, 1);
        for _ in 0..400 {
            let bits =
                |rng: &mut Pcg32| (u64::from(rng.next_u32()) << 32) | u64::from(rng.next_u32());
            rows.push(vec![
                f64::from_bits(bits(&mut rng)),
                f64::from_bits(bits(&mut rng)),
                f64::from_bits(bits(&mut rng)),
            ]);
        }
        rows.push(vec![]);
        rows
    }

    /// Same error status (in the compat shape the value is 0.0 with a Hard fault), the same root
    /// value bits and the same fault set, on every row.
    fn assert_exact(original: &MathExpr, simplified: &MathExpr) {
        assert_exact_on(original, simplified, &rows());
    }

    fn is_signaling(v: f64) -> bool {
        v.is_nan() && v.to_bits() & (1 << 51) == 0
    }

    /// [`assert_exact`] on the documented domain only: rows holding a signaling NaN are skipped.
    fn assert_exact_quiet(original: &MathExpr, simplified: &MathExpr) {
        let quiet: Vec<Vec<f64>> = rows()
            .into_iter()
            .filter(|row| !row.iter().copied().any(is_signaling))
            .collect();
        assert_exact_on(original, simplified, &quiet);
    }

    fn assert_exact_on(original: &MathExpr, simplified: &MathExpr, rows: &[Vec<f64>]) {
        let (a, b) = (original.compile().unwrap(), simplified.compile().unwrap());
        for row in rows {
            let (va, fa) = a.eval_row_traced(row);
            let (vb, fb) = b.eval_row_traced(row);
            assert_eq!(
                (va.to_bits(), fa),
                (vb.to_bits(), fb),
                "{:?} vs {:?} on {row:?}",
                original.to_source(),
                simplified.to_source()
            );
        }
    }

    #[test]
    fn each_rule_simplifies_its_pattern_exactly() {
        let cases = [
            ("(c0*1.0)", "c0"),               // R1
            ("(1.0*c0)", "c0"),               // R1
            ("(c0/1.0)", "c0"),               // R2
            ("(c0+(-0.0))", "c0"),            // R3: x + (-0) = x, including x = ±0
            ("((-0.0)+c0)", "c0"),            // R3
            ("(c0-0.0)", "c0"),               // R3: x - (+0) = x
            ("(-(-(c0)))", "c0"),             // R4: sign-bit flips
            ("abs((-(c0)))", "abs(c0)"),      // R5
            ("abs(abs(c0))", "abs(c0)"),      // R5
            ("(c0-(-(c1)))", "(c0+c1)"),      // R7: a - (-b) = a + b
            ("(c0+(-(c1)))", "(c0-c1)"),      // R7: a + (-b) = a - b
            ("((-(c0))*(-(c1)))", "(c0*c1)"), // R8: RNE is sign-symmetric
            ("((-(c0))*c1)", "(-(c0*c1))"),   // R8: the same node count, shallower? no: kept
            ("(c0*(-1.0))", "(-c0)"),         // R10
            ("(c0/(-1.0))", "(-c0)"),         // R10
            ("((2.0*3.0)+c0)", "(6.0+c0)"),   // folding
            ("(c0+(1.5-1.5))", "(c0+0.0)"),   // folding to +0.0; x + 0.0 itself stays
            ("abs((-2.5))*c0", "(2.5*c0)"),   // folding abs of a literal
            ("((c0*1.0)/1.0)+(-0.0)", "c0"),  // chains
        ];
        for (input, expected) in cases {
            let original = MathExpr::parse(input).unwrap();
            let result = simplify(&original);
            let expected_source = if expected == "(-(c0*c1))" {
                // (-a)*b and -(a*b) both take five nodes at depth 3; the original stays.
                printed(input)
            } else {
                printed(expected)
            };
            assert_eq!(
                result.expression.to_source().unwrap(),
                expected_source,
                "{input}"
            );
            assert_exact(&original, &result.expression);
        }
    }

    #[test]
    fn power_of_two_division_joins_the_reciprocal_product() {
        // R11 keeps the node count, so extraction keeps the original, but both are one class.
        for (c, reciprocal) in [(0.25, 4.0), (-8.0, -0.125), (2f64.powi(-39), 2f64.powi(39))] {
            let mut eg = EGraph::default();
            let x = eg.add(Node::Var(Var::Col(0)));
            let c_id = eg.insert(&Term::Literal(c));
            let division = eg.add(Node::Bin(DIV, x, c_id));
            let r_id = eg.insert(&Term::Literal(reciprocal));
            let product = eg.add(Node::Bin(MUL, x, r_id));
            for rhs in matches(&eg, Rule::DivPowerOfTwo, &Node::Bin(DIV, x, c_id)) {
                let new = eg.insert(&rhs);
                eg.union(division, new);
            }
            eg.rebuild();
            assert_eq!(eg.find(division), eg.find(product), "{c}");
            let original = MathExpr::from_ast(Expr::Binary(
                BinOp::Div,
                Box::new(Expr::Var(Var::Col(0))),
                Box::new(literal(c)),
            ))
            .unwrap();
            let rewritten = MathExpr::from_ast(Expr::Binary(
                BinOp::Mul,
                Box::new(Expr::Var(Var::Col(0))),
                Box::new(literal(reciprocal)),
            ))
            .unwrap();
            assert_exact(&original, &rewritten);
        }
        // 2^-40 is under the protected threshold, 3 is not a power of two, and 2^-1074 has no
        // finite reciprocal: none is rewritten.
        for c in [2f64.powi(-40), 3.0, 5e-324, 1.0, -1.0] {
            assert!(exact_reciprocal(c).is_none_or(|r| r.abs() == 1.0), "{c}");
        }
    }

    #[test]
    fn excluded_rewrites_are_wrong_and_never_applied() {
        let differs = |a: &str, b: &str, row: &[f64]| {
            let (va, fa) = MathExpr::parse(a).unwrap().eval_row_traced(row);
            let (vb, fb) = MathExpr::parse(b).unwrap().eval_row_traced(row);
            (va.to_bits(), fa) != (vb.to_bits(), fb)
        };
        // x + 0.0 and x - (-0.0) turn -0.0 into +0.0.
        assert!(differs("(c0+0.0)", "c0", &[-0.0]));
        assert!(differs("(c0-(-0.0))", "c0", &[-0.0]));
        // x*0 keeps the sign (and inf*0 is NaN, sanitized with a fault).
        assert!(differs("(c0*0.0)", "0.0", &[-1.0]));
        assert!(differs("(c0*0.0)", "0.0", &[f64::INFINITY]));
        // x - x and x / x at infinity and zero.
        assert!(differs("(c0-c0)", "0.0", &[f64::INFINITY]));
        assert!(differs("(c0/c0)", "1.0", &[0.0]));
        // Reassociation: 2^53 + 1 rounds to 2^53.
        assert!(differs(
            "((c0+c1)+c2)",
            "(c0+(c1+c2))",
            &[2f64.powi(53), 1.0, -(2f64.powi(53))]
        ));
        // Distribution: 3 (1 + 2^-53) rounds differently from 3 + 3·2^-53.
        assert!(differs(
            "(c0*(c1+c2))",
            "((c0*c1)+(c0*c2))",
            &[3.0, 1.0, 2f64.powi(-53)]
        ));
        // a - b versus -(b - a) at a = b.
        assert!(differs("(c0-c1)", "(-(c1-c0))", &[1.0, 1.0]));
        // The protected branch returns +0.0 for (-a)/b, but -(a/b) is -0.0.
        assert!(differs("((-(c0))/c1)", "(-(c0/c1))", &[1.0, 1e-13]));
        // Commutation changes which operand's faults precede a Hard fault (column out of range).
        assert!(differs("(log(c0)+c9)", "(c9+log(c0))", &[-1.0]));
        // No exact rule reaches the wrong target. x - (-0.0) may become x + 0.0, which IEEE
        // defines as the same operation, but never x.
        for (text, forbidden) in [
            ("(c0+0.0)", "c0"),
            ("(c0-(-0.0))", "c0"),
            ("(c0*0.0)", "0.0"),
            ("(c0-c0)", "0.0"),
            ("(c0/c0)", "1.0"),
        ] {
            let original = MathExpr::parse(text).unwrap();
            let result = simplify(&original);
            assert_ne!(
                result.expression.to_source().unwrap(),
                printed(forbidden),
                "{text}"
            );
            assert_exact(&original, &result.expression);
        }
        assert_eq!(source("(c0-(-0.0))"), printed("(c0+0.0)"));
        // Where no exact rule applies at all, the tree is unchanged.
        for text in [
            "(c0+0.0)",
            "(c0*0.0)",
            "(c0-c0)",
            "(c0/c0)",
            "((c0+c1)+c2)",
            "(c0*(c1+c2))",
            "((-(c0))/c1)",
            "(c0^2.0)",
            "min(c0,c1)",
            "exp(log(c0))",
            "(c0/(2.0^(-40.0)))",
        ] {
            assert_eq!(source(text), printed(text), "{text}");
        }
    }

    /// Random trees biased toward the rules' patterns, every one simplified exactly.
    fn random_tree(rng: &mut Pcg32, depth: usize) -> Expr {
        let leaf = |rng: &mut Pcg32| -> Expr {
            match rng.next_u32() % 5 {
                0 | 1 => Expr::Var(Var::Col(rng.next_u32() as usize % 3)),
                _ => literal(
                    [0.0, -0.0, 1.0, -1.0, 2.0, 0.5, -0.25, 3.0, 2f64.powi(-40)]
                        [rng.next_u32() as usize % 9],
                ),
            }
        };
        if depth == 0 || rng.next_u32().is_multiple_of(5) {
            return leaf(rng);
        }
        match rng.next_u32() % 6 {
            0 => Expr::Unary(UnOp::Neg, Box::new(random_tree(rng, depth - 1))),
            1 => Expr::Call(Func::Abs, vec![random_tree(rng, depth - 1)]),
            2 => Expr::Call(
                [Func::Sin, Func::Sqrt, Func::Log, Func::Tanh][rng.next_u32() as usize % 4],
                vec![random_tree(rng, depth - 1)],
            ),
            _ => Expr::Binary(
                [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div][rng.next_u32() as usize % 4],
                Box::new(random_tree(rng, depth - 1)),
                Box::new(random_tree(rng, depth - 1)),
            ),
        }
    }

    #[test]
    fn random_trees_simplify_exactly_and_never_grow() {
        let mut rng = Pcg32::new(0xB20, 2);
        let (mut changed, mut checked) = (0usize, 0usize);
        for _ in 0..300 {
            let Ok(original) = MathExpr::from_ast(random_tree(&mut rng, 4)) else {
                continue;
            };
            let result = simplify(&original);
            assert!(count_nodes(&result.expression.root) <= count_nodes(&original.root));
            assert!(result.enodes <= NODE_CAP_MAX);
            assert_exact(&original, &result.expression);
            changed += usize::from(result.changed);
            checked += 1;
        }
        assert!(
            checked > 250 && changed > 50,
            "{checked} checked, {changed} changed"
        );
    }

    #[test]
    fn saturation_is_bounded_deterministic_and_independent_of_rule_order() {
        let text = "((((-(c0))*(-(c1)))*1.0)-(-((c2/(-1.0)))))+((-(-(c0)))/0.5)";
        let original = MathExpr::parse(text).unwrap();
        let first = simplify(&original);
        let again = simplify(&original);
        assert_eq!(
            first.expression.to_source().unwrap(),
            again.expression.to_source().unwrap()
        );
        assert_eq!(
            (first.enodes, first.applications, first.iterations),
            (again.enodes, again.applications, again.iterations)
        );
        // Rule-order independence is claimed for a completed saturation, which this is.
        let limits = Limits::for_input(count_nodes(&original.root));
        assert_eq!(first.stop, Saturation::Saturated);
        let mut reversed = RULES.to_vec();
        reversed.reverse();
        let permuted = simplify_with(&original, &reversed, limits);
        assert_eq!(permuted.stop, Saturation::Saturated);
        assert_eq!(
            permuted.expression.to_source().unwrap(),
            first.expression.to_source().unwrap()
        );
        assert!(first.changed);
        assert_exact(&original, &first.expression);
        assert!(
            first.enodes <= limits.enodes,
            "{} > {}",
            first.enodes,
            limits.enodes
        );
        // A long chain of x - y rewrites saturates well inside its cap.
        let wide =
            "((((c0-c1)-c2)-(c0-c2))-((c1-c0)-(c2-c1)))-(((c0-c1)-(c1-c2))-((c2-c0)-(c0-c1)))";
        let wide = MathExpr::parse(wide).unwrap();
        let result = simplify(&wide);
        assert_eq!(result.stop, Saturation::Saturated);
        assert!(result.enodes <= Limits::for_input(count_nodes(&wide.root)).enodes);
        assert_exact(&wide, &result.expression);
    }

    #[test]
    fn hard_limits_stop_saturation_with_an_exact_rebuilt_result() {
        let original =
            MathExpr::parse("((((-(c0))*(-(c1)))*1.0)-(-((c2/(-1.0)))))+((-(-(c0)))/0.5)").unwrap();
        let full = simplify(&original);
        assert_eq!(full.stop, Saturation::Saturated);
        // A tight e-node cap stops the second round part way; the rebuilt graph still extracts
        // an exact tree, never larger than the input.
        let capped = simplify_with(
            &original,
            &RULES,
            Limits {
                iterations: MAX_ITERATIONS,
                enodes: 29,
            },
        );
        assert_eq!(capped.stop, Saturation::NodeLimit, "{capped:?}");
        assert!(capped.enodes <= 29, "{capped:?}");
        assert!(capped.applications < full.applications, "{capped:?}");
        assert!(count_nodes(&capped.expression.root) <= count_nodes(&original.root));
        assert_exact(&original, &capped.expression);
        // One round of a longer saturation.
        let one = simplify_with(
            &original,
            &RULES,
            Limits {
                iterations: 1,
                enodes: NODE_CAP_MAX,
            },
        );
        assert_eq!((one.stop, one.iterations), (Saturation::IterationLimit, 1));
        assert_exact(&original, &one.expression);
        // Both limited runs are deterministic.
        let again = simplify_with(
            &original,
            &RULES,
            Limits {
                iterations: MAX_ITERATIONS,
                enodes: 29,
            },
        );
        assert_eq!(
            (again.enodes, again.applications, again.iterations),
            (capped.enodes, capped.applications, capped.iterations)
        );
    }

    #[test]
    fn signaling_nan_rows_are_outside_the_exact_domain() {
        let snan = f64::from_bits(0x7ff0_0000_0000_0001);
        assert!(is_signaling(snan));
        for input in [
            "((c0*1.0)^0.0)",
            "((c0/1.0)^0.0)",
            "((c0-0.0)^0.0)",
            "((c0+(-0.0))^0.0)",
            "((c0*(-1.0))^0.0)",
            "((c0/(-1.0))^0.0)",
            "(1.0^(c0*1.0))",
        ] {
            let original = MathExpr::parse(input).unwrap();
            let simplified = simplify(&original).expression;
            assert_ne!(simplified, original, "{input}: the rule fires");
            let (a, b) = (original.compile().unwrap(), simplified.compile().unwrap());
            // A quiet NaN stays exact.
            let (qa, qb) = (
                a.eval_row_traced(&[f64::NAN]),
                b.eval_row_traced(&[f64::NAN]),
            );
            assert_eq!((qa.0.to_bits(), qa.1), (qb.0.to_bits(), qb.1), "{input}");
            // A signaling NaN: the deleted step quieted it, and pow tells the two apart.
            let (sa, sb) = (a.eval_row_traced(&[snan]), b.eval_row_traced(&[snan]));
            assert_eq!(
                (sa.0.to_bits(), sa.1),
                (1.0f64.to_bits(), FaultSet::NONFINITE_INPUT),
                "{input}"
            );
            assert_eq!(
                (sb.0.to_bits(), sb.1),
                (
                    0.0f64.to_bits(),
                    FaultSet::NONFINITE_INPUT.union(FaultSet::NONFINITE_SANITIZED)
                ),
                "{input}"
            );
        }
    }

    /// Random trees over every operator and the selector, clamp and exponential functions.
    fn random_tree_wide(rng: &mut Pcg32, depth: usize) -> Expr {
        let leaf = |rng: &mut Pcg32| -> Expr {
            match rng.next_u32() % 5 {
                0 | 1 => Expr::Var(Var::Col(rng.next_u32() as usize % 3)),
                _ => literal(
                    [
                        0.0,
                        -0.0,
                        1.0,
                        -1.0,
                        2.0,
                        0.5,
                        -0.25,
                        3.0,
                        9.0,
                        2f64.powi(-40),
                    ][rng.next_u32() as usize % 10],
                ),
            }
        };
        if depth == 0 || rng.next_u32().is_multiple_of(5) {
            return leaf(rng);
        }
        let child = |rng: &mut Pcg32| random_tree_wide(rng, depth - 1);
        match rng.next_u32() % 9 {
            0 => Expr::Unary(UnOp::Neg, Box::new(child(rng))),
            1 => Expr::Call(Func::Abs, vec![child(rng)]),
            2 => {
                let func = [Func::Exp, Func::Sigmoid, Func::Step, Func::Sin, Func::Log]
                    [rng.next_u32() as usize % 5];
                let arity = (1..=3)
                    .find(|n| validate_arity(func, *n).is_ok())
                    .expect("an arity");
                Expr::Call(func, (0..arity).map(|_| child(rng)).collect())
            }
            3 => Expr::Call(
                [Func::Min, Func::Max][rng.next_u32() as usize % 2],
                vec![child(rng), child(rng)],
            ),
            4 => Expr::Call(Func::Clip, vec![child(rng), child(rng), child(rng)]),
            5 => Expr::Binary(BinOp::Pow, Box::new(child(rng)), Box::new(child(rng))),
            _ => Expr::Binary(
                [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div][rng.next_u32() as usize % 4],
                Box::new(child(rng)),
                Box::new(child(rng)),
            ),
        }
    }

    #[test]
    fn random_trees_over_every_operation_simplify_exactly_on_the_domain() {
        let mut rng = Pcg32::new(0xB20, 3);
        let (mut changed, mut checked) = (0usize, 0usize);
        for _ in 0..300 {
            let Ok(original) = MathExpr::from_ast(random_tree_wide(&mut rng, 4)) else {
                continue;
            };
            let result = simplify(&original);
            assert!(count_nodes(&result.expression.root) <= count_nodes(&original.root));
            assert_exact_quiet(&original, &result.expression);
            changed += usize::from(result.changed);
            checked += 1;
        }
        assert!(
            checked > 200 && changed > 30,
            "{checked} checked, {changed} changed"
        );
    }
}
