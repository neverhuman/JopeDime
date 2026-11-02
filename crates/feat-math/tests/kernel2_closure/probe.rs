use super::*;

#[derive(Clone, Debug, Default)]
pub(super) struct ProbeEvidence {
    pub(super) checked: usize,
    pub(super) observed: FaultSet,
    pub(super) errors: Vec<String>,
}

pub(super) fn boundary_values() -> Vec<f64> {
    let data: Value = serde_json::from_str(BOUNDARY).unwrap();
    let values: Vec<_> = data["entries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| {
            let bits = entry["bits"].as_str().unwrap().strip_prefix("0x").unwrap();
            f64::from_bits(u64::from_str_radix(bits, 16).unwrap())
        })
        .collect();
    assert_eq!(values.len(), 66);
    values
}

pub(super) fn probe_source(row: &feat_math::kernel::capabilities::ElementRow) -> (String, usize) {
    match row.kind {
        "function" => {
            let arity = match row.name {
                "clip" => 3,
                "min" | "max" | "step" => 2,
                _ => 1,
            };
            (
                format!("{}({})", row.name, ["c0", "c1", "c2"][..arity].join(",")),
                arity,
            )
        }
        "binary" => (format!("(c0{}c1)", row.name), 2),
        "unary" => ("(-c0)".to_owned(), 1),
        "variable" => (
            match row.name {
                "col_under_8" => "c7".to_owned(),
                "col_8_plus" => "c8".to_owned(),
                other => other.to_owned(),
            },
            0,
        ),
        _ => unreachable!(),
    }
}

pub(super) fn each_tuple(
    values: &[f64],
    arity: usize,
    row: &mut Vec<f64>,
    visit: &mut impl FnMut(&[f64]),
) {
    if row.len() == arity {
        visit(row);
        return;
    }
    for &value in values {
        row.push(value);
        each_tuple(values, arity, row, visit);
        row.pop();
    }
}

pub(super) fn each_probe_row(kind: &str, name: &str, arity: usize, visit: &mut impl FnMut(&[f64])) {
    if kind != "variable" {
        each_tuple(&boundary_values(), arity, &mut Vec::new(), visit);
    } else if name == "x" {
        for value in boundary_values() {
            visit(&[value]);
        }
    } else {
        for width in 0..=9 {
            let mut row: Vec<f64> = (1..=width).map(|n| n as f64).collect();
            visit(&row);
            if width > 0 {
                for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                    row[width - 1] = value;
                    visit(&row);
                }
            }
        }
    }
}

pub(super) fn probe_evidence(row: &feat_math::kernel::capabilities::ElementRow) -> ProbeEvidence {
    let (source, arity) = probe_source(row);
    let expression = MathExpr::parse(&source).unwrap_or_else(|error| panic!("{source}: {error:#}"));
    assert_eq!(expression.to_source().unwrap(), source);
    let compiled = expression.compile().unwrap();
    let mut out = ProbeEvidence::default();
    each_probe_row(row.kind, row.name, arity, &mut |cells| {
        let interpreted = expression.eval_row_traced(cells);
        let compiled_row = compiled.eval_row_traced(cells);
        if (interpreted.0.to_bits() != compiled_row.0.to_bits() || interpreted.1 != compiled_row.1)
            && out.errors.len() < 5
        {
            out.errors.push("row evaluator mismatch".to_owned());
        }
        out.observed = out.observed.union(interpreted.1);
        out.checked += 1;
        let x = cells.first().copied().unwrap_or(0.0);
        let interpreted = expression.eval_element_traced(x, cells);
        let compiled_element = compiled.eval_element_traced(x, cells);
        if (interpreted.0.to_bits() != compiled_element.0.to_bits()
            || interpreted.1 != compiled_element.1)
            && out.errors.len() < 5
        {
            out.errors.push("element evaluator mismatch".to_owned());
        }
        out.observed = out.observed.union(interpreted.1);
        out.checked += 1;
        let (mapped, faults) = compiled.map_row_traced(cells);
        if mapped.len() != cells.len() || faults.len() != cells.len() {
            if out.errors.len() < 5 {
                out.errors.push("map width mismatch".to_owned());
            }
        } else {
            for (index, &value) in cells.iter().enumerate() {
                let expected = expression.eval_element_traced(value, cells);
                if (mapped[index].to_bits() != expected.0.to_bits() || faults[index] != expected.1)
                    && out.errors.len() < 5
                {
                    out.errors.push("map cell mismatch".to_owned());
                }
                out.observed = out.observed.union(faults[index]);
            }
        }
        out.checked += 1;
    });
    out
}

pub(super) fn probes() -> &'static BTreeMap<(&'static str, &'static str), ProbeEvidence> {
    static PROBES: OnceLock<BTreeMap<(&'static str, &'static str), ProbeEvidence>> =
        OnceLock::new();
    PROBES.get_or_init(|| {
        Manifest::current()
            .expression
            .iter()
            .map(|row| ((row.kind, row.name), probe_evidence(row)))
            .collect()
    })
}

pub(super) fn fault_problem(declared: FaultSet, evidence: &ProbeEvidence) -> Option<String> {
    let allowed = FaultSet::NONFINITE_SANITIZED
        .union(FaultSet::MAGNITUDE_CLAMPED)
        .union(FaultSet::NONFINITE_INPUT);
    let missing = declared.bits() & !evidence.observed.bits();
    let unexpected = evidence.observed.bits() & !(declared.bits() | allowed.bits());
    if missing != 0 {
        Some(format!("declared fault class not raised: {missing:#x}"))
    } else if unexpected != 0 {
        Some(format!("undeclared fault class raised: {unexpected:#x}"))
    } else if evidence.observed.intersects(FaultSet::COMPILE_FAILED) {
        Some("well-formed probe raised COMPILE_FAILED".to_owned())
    } else if !evidence.errors.is_empty() {
        Some(evidence.errors.join("; "))
    } else {
        None
    }
}

pub(super) fn clause_c(
    manifest: &Manifest,
    profile: &SearchProfile,
    report: &mut ClosureReport,
) -> usize {
    let mut selected = BTreeSet::new();
    for element in profile.elements() {
        let pair = match element.kind {
            ElementKind::Function => Some(("function", element.name)),
            ElementKind::Binary => Some(("binary", element.name)),
            ElementKind::Variable => Some(("variable", element.name)),
            ElementKind::Policy if element.name == "columns" => Some(("variable", "col_under_8")),
            ElementKind::Policy if element.name == "negation" => Some(("unary", "neg")),
            _ => None,
        };
        if let Some(pair) = pair {
            selected.insert(pair);
        }
    }
    let mut checked = 0;
    for (kind, name) in selected {
        let row = manifest
            .expression
            .iter()
            .find(|row| row.kind == kind && row.name == name);
        let evidence = probes().get(&(kind, name));
        match (row, evidence) {
            (Some(row), Some(evidence)) => {
                checked += evidence.checked;
                if let Some(problem) = fault_problem(row.capability.fault_classes, evidence) {
                    report.fail(profile, name, "(c)", problem);
                }
            }
            _ => report.fail(profile, name, "(c)", "missing expression row or probe"),
        }
    }
    checked
}

pub(super) fn manifest_c(manifest: &Manifest, report: &mut ClosureReport) -> usize {
    let mut checked = 0;
    for row in &manifest.expression {
        if let Some(evidence) = probes().get(&(row.kind, row.name)) {
            checked += evidence.checked;
            if let Some(problem) = fault_problem(row.capability.fault_classes, evidence) {
                report.violations.push(ClosureViolation {
                    profile: "manifest".to_owned(),
                    element: row.name.to_owned(),
                    clause: "(c)",
                    reason: problem,
                });
            }
        } else {
            report.violations.push(ClosureViolation {
                profile: "manifest".to_owned(),
                element: row.name.to_owned(),
                clause: "(c)",
                reason: "missing probe".to_owned(),
            });
        }
    }
    checked
}
