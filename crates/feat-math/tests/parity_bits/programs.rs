use super::*;

/// Exhaustive, so a new function fails to compile here until a program executes it.
pub(super) fn func_name(func: Func) -> &'static str {
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

pub(super) fn var_name(var: Var) -> &'static str {
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

pub(super) fn binop_name(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::Pow => "^",
    }
}

const BINOPS: [BinOp; 5] = [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div, BinOp::Pow];

pub(super) fn unop_name(op: UnOp) -> &'static str {
    match op {
        UnOp::Neg => "neg",
    }
}

pub(super) fn walk(expr: &Expr, seen: &mut BTreeSet<&'static str>) {
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
pub(super) fn expressions(program: &Program) -> Vec<MathExpr> {
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
        model.predict(&vec![row.values.clone()]).unwrap()[0].to_bits()
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

/// The lanes read the shared corpus on the bytes the allowlist entries were recorded against, so a
/// corpus change is noticed here as well as in `compat_freeze`, and every entry reaches the rows
/// the lanes compare.
#[test]
fn parity_bits_reads_the_shared_boundary_corpus_the_allowlist_was_recorded_on() {
    let digest = format!("{:x}", Sha256::digest(BOUNDARY_V1.as_bytes()));
    assert_eq!(
        digest, BOUNDARY_V1_SHA256,
        "tests/fixtures/math_boundary/v1.json changed; the parity allowlist was recorded against \
         sha256 {BOUNDARY_V1_SHA256}. The corpus is versioned: add math_boundary/v2.json instead \
         of editing v1."
    );
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
