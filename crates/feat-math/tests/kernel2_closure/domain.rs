use super::*;

pub(super) fn domain_points(domain: &ParamDomain) -> Vec<Option<ParamValue>> {
    let values = match domain {
        ParamDomain::IntRange { lo, hi } => vec![
            Some(ParamValue::Int(*lo)),
            Some(ParamValue::Int((lo + hi) / 2)),
            Some(ParamValue::Int(*hi)),
        ],
        ParamDomain::FloatRange { lo, hi } => vec![
            Some(ParamValue::Float(*lo)),
            Some(ParamValue::Float((lo + hi) / 2.0)),
            Some(ParamValue::Float(*hi)),
        ],
        ParamDomain::IntChoice(items) => items
            .iter()
            .copied()
            .map(ParamValue::Int)
            .map(Some)
            .collect(),
        ParamDomain::FloatChoice(items) => items
            .iter()
            .copied()
            .map(ParamValue::Float)
            .map(Some)
            .collect(),
        ParamDomain::TextChoice(items) => items
            .iter()
            .map(|item| match item {
                TextChoice::Value(value) => Some(ParamValue::Str((*value).to_owned())),
                TextChoice::Omit => None,
            })
            .collect(),
        ParamDomain::Flag { .. } => vec![None, Some(ParamValue::Bool(true))],
    };
    let mut unique = Vec::new();
    for value in values {
        if !unique.contains(&value) {
            unique.push(value);
        }
    }
    unique
}

pub(super) fn program_fixed_point(program: &GenomeIr) -> bool {
    let Ok(reparsed) = GenomeIr::parse(&program.canonical()) else {
        return false;
    };
    let before = Program::import_legacy(program).to_bytes();
    let after = Program::import_legacy(&reparsed).to_bytes();
    before.is_ok() && before == after
}

pub(super) fn stage_program(
    name: &str,
    params: BTreeMap<String, ParamValue>,
    head: bool,
) -> GenomeIr {
    let stage = Stage {
        op: name.to_owned(),
        params,
    };
    let stages = if head {
        vec![stage]
    } else {
        vec![stage, Stage::new("ridge")]
    };
    GenomeIr::Pipeline(DslExpr { stages })
}

pub(super) fn expression_program(source: &str, carrier: &str) -> GenomeIr {
    if carrier == "symbolic" {
        return GenomeIr::parse(&format!("(sym raw \"{source}\")")).unwrap();
    }
    let mut params = BTreeMap::new();
    params.insert("expr".to_owned(), ParamValue::Str(source.to_owned()));
    GenomeIr::Pipeline(DslExpr {
        stages: vec![
            Stage {
                op: carrier.to_owned(),
                params,
            },
            Stage::new("ridge"),
        ],
    })
}

pub(super) fn clause_d(
    manifest: &Manifest,
    profile: &SearchProfile,
    report: &mut ClosureReport,
) -> usize {
    let mut checked = 0;
    for element in profile.elements() {
        if !matches!(element.kind, ElementKind::Stage | ElementKind::Head) {
            continue;
        }
        let Some(domains) = profile.domains.get(element.name) else {
            report.fail(profile, element.name, "(d)", "missing birth domain");
            continue;
        };
        let choices: Vec<_> = domains
            .iter()
            .map(|(key, domain)| ((*key).to_owned(), domain_points(domain)))
            .collect();
        let mut base = BTreeMap::new();
        for (key, points) in &choices {
            if let Some(Some(value)) = points.first() {
                base.insert(key.clone(), value.clone());
            }
        }
        let mut programs = vec![stage_program(
            element.name,
            base.clone(),
            element.kind == ElementKind::Head,
        )];
        for (key, points) in &choices {
            for point in points.iter().skip(1) {
                let mut params = base.clone();
                if let Some(value) = point {
                    params.insert(key.clone(), value.clone());
                } else {
                    params.remove(key);
                }
                programs.push(stage_program(
                    element.name,
                    params,
                    element.kind == ElementKind::Head,
                ));
            }
        }
        for program in programs {
            checked += 1;
            if !program_fixed_point(&program) {
                report.fail(
                    profile,
                    element.name,
                    "(d)",
                    format!("canonical program loses bytes: {}", program.canonical()),
                );
            }
        }
    }
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
    for (kind, name) in selected {
        let Some(row) = manifest
            .expression
            .iter()
            .find(|row| row.kind == kind && row.name == name)
        else {
            report.fail(profile, name, "(d)", "missing expression row");
            continue;
        };
        let (source, _) = probe_source(row);
        checked += 1;
        if MathExpr::parse(&source)
            .ok()
            .and_then(|expr| expr.to_source().ok())
            .as_deref()
            != Some(source.as_str())
        {
            report.fail(profile, name, "(d)", "expression text is not a fixed point");
        }
        for carrier in ["map", "derive", "symbolic"] {
            if name == "x" && carrier != "map" {
                continue;
            }
            let program = expression_program(&source, carrier);
            checked += 1;
            if !program_fixed_point(&program) {
                report.fail(
                    profile,
                    name,
                    "(d)",
                    format!("canonical expression carrier loses bytes: {carrier}"),
                );
            }
        }
    }
    if let Some(expr) = &profile.expr {
        for &bin in &expr.birth_constants {
            let source = match expr.negation {
                Negation::ConstantsOnly if bin < 0.0 => format!("(-{:?})", -bin),
                Negation::ConstantsOnly => format!("{bin:?}"),
            };
            checked += 1;
            if MathExpr::parse(&source)
                .ok()
                .and_then(|value| value.to_source().ok())
                .as_deref()
                != Some(&source)
            {
                report.fail(
                    profile,
                    "constants",
                    "(d)",
                    format!("constant bin is not a fixed point: {source}"),
                );
            }
        }
    }
    checked
}
