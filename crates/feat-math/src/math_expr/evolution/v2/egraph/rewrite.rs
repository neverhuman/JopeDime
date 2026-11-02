use crate::math_expr::evaluator::compute_binary;

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

/// The right-hand sides `rule` gives for `node`. An `abs` of any other arity than one matches
/// nothing.
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
        (Rule::AbsNeg, Node::Call(Func::Abs, args)) if args.len() == 1 => eg
            .negations(args[0])
            .into_iter()
            .map(|a| Term::Call(Func::Abs, vec![Term::Class(a)]))
            .collect(),
        (Rule::AbsAbs, Node::Call(Func::Abs, args)) if args.len() == 1 && eg.holds_abs(args[0]) => {
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
        (Rule::Fold, Node::Call(Func::Abs, args)) if args.len() == 1 => match lit(args[0]) {
            Some(a) => vec![Term::Literal(a.abs())],
            None => Vec::new(),
        },
        _ => Vec::new(),
    }
}
