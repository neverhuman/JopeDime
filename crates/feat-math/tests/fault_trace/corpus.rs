use super::*;

/// One corpus row: its label and its cells.
struct Row {
    label: String,
    cells: Vec<f64>,
}

/// PR 01's corpus: every row and every expression source.
pub(crate) struct Corpus {
    pub(crate) rows: Vec<Row>,
    pub(crate) expressions: Vec<String>,
}

pub(crate) fn corpus() -> Corpus {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/compat_freeze/mathexpr_values.v1.json"
    );
    let text = std::fs::read_to_string(path).unwrap_or_else(|error| panic!("{path}: {error}"));
    let file: serde_json::Value =
        serde_json::from_str(&text).expect("mathexpr_values.v1.json parses");
    let rows: Vec<Row> = file["rows"]
        .as_array()
        .expect("rows")
        .iter()
        .map(|row| Row {
            label: row["label"].as_str().expect("row label").to_string(),
            // The empty row is recorded as "": no cells, not one empty token.
            cells: row["values"]
                .as_str()
                .expect("row values")
                .split(' ')
                .filter(|token| !token.is_empty())
                .map(|token| {
                    f64::from_bits(
                        u64::from_str_radix(token, 16)
                            .unwrap_or_else(|error| panic!("{token}: {error}")),
                    )
                })
                .collect(),
        })
        .collect();
    let expressions: Vec<String> = file["expressions"]
        .as_array()
        .expect("expressions")
        .iter()
        .map(|expression| expression["source"].as_str().expect("source").to_string())
        .collect();
    assert!(
        rows.iter().any(|row| row.cells.is_empty()) && !expressions.is_empty(),
        "the corpus has an empty row and at least one expression"
    );
    Corpus { rows, expressions }
}

#[test]
fn every_traced_form_equals_the_legacy_values_on_the_compat_corpus() {
    let corpus = corpus();
    let mut tallies = Tallies::default();
    for source in &corpus.expressions {
        let expr = MathExpr::parse(source).expect("corpus expression parses");
        let compiled = expr.compile().expect("corpus expression compiles");
        for row in &corpus.rows {
            check_every_form(
                At::new(source, &row.label),
                &expr,
                &compiled,
                &row.cells,
                &mut tallies,
            );
        }
    }
    eprintln!(
        "PROPERTY expressions={} rows={}",
        corpus.expressions.len(),
        corpus.rows.len()
    );
    eprintln!("PROPERTY row context: {}", tallies.rows);
    eprintln!("PROPERTY element context: {}", tallies.elements);
    assert!(tallies.rows.evaluations > 0 && tallies.elements.evaluations > 0);
}

// ---------------------------------------------------------------------------------------------
// Random expressions (WP-1.3)
// ---------------------------------------------------------------------------------------------

/// A recombination donor with every row summary, a negation, `x` and column reads: the edit
/// operator's leaves draw only row_mean among the summaries, and it never negates, so without
/// this the summary cache and most reductions would never meet the compiled form.
const DONOR: &str =
    "row_sum - row_l1 + row_l2 * row_min / row_max ^ row_std + row_mean * (-x) - c0 / c2 + sqrt(c1)";

/// Chained random expressions for `seeds`, each checked on every corpus row through every traced
/// form. Each seed edits the parent with `mutate_v1`, or recombines it with `DONOR` through
/// `crossover_v1` at every third seed, within a scope that changes every 10 seeds (element or row
/// context) and every 100 seeds (0, 1, 2, 3 or 8 columns); a parent outside the new scope
/// restarts from `0.1`.
fn random_expressions_agree(seeds: std::ops::Range<u64>) {
    let corpus = corpus();
    let donor = MathExpr::parse(DONOR).expect("the donor parses");
    let restart = MathExpr::parse("0.1").expect("the restart parses");
    let mut parent = restart.clone();
    let mut distinct = BTreeSet::new();
    let mut changed = 0usize;
    let mut tallies = Tallies::default();
    for seed in seeds.clone() {
        let scope = ExpressionScope {
            columns: [0, 1, 2, 3, 8][(seed / 100 % 5) as usize],
            elementwise: seed / 10 % 2 == 0,
            max_nodes: 64,
            max_depth: 16,
        };
        if scope.validate(&parent).is_err() {
            parent = restart.clone();
        }
        let edit = if seed % 3 == 0 {
            parent.crossover_v1(&donor, seed, scope)
        } else {
            parent.mutate_v1(seed, scope)
        }
        .unwrap_or_else(|error| panic!("seed {seed}: {error}"));
        changed += usize::from(edit.changed);
        let expr = edit.expression;
        let compiled = expr
            .compile()
            .unwrap_or_else(|error| panic!("seed {seed}: {error}"));
        let ast = format!("{:?}", expr.ast());
        let name = format!("seed {seed}, {ast}");
        for row in &corpus.rows {
            check_every_form(
                At::new(&name, &row.label),
                &expr,
                &compiled,
                &row.cells,
                &mut tallies,
            );
        }
        distinct.insert(ast);
        parent = expr;
    }
    eprintln!(
        "DIFFERENTIAL seeds={seeds:?} expressions={} changed={changed} distinct={} rows={}",
        seeds.end - seeds.start,
        distinct.len(),
        corpus.rows.len()
    );
    eprintln!("DIFFERENTIAL row context: {}", tallies.rows);
    eprintln!("DIFFERENTIAL element context: {}", tallies.elements);
}

#[test]
fn random_expressions_agree_across_forms_seeds_0_to_2499() {
    random_expressions_agree(0..2_500);
}

#[test]
fn random_expressions_agree_across_forms_seeds_2500_to_4999() {
    random_expressions_agree(2_500..5_000);
}

#[test]
fn random_expressions_agree_across_forms_seeds_5000_to_7499() {
    random_expressions_agree(5_000..7_500);
}

#[test]
fn random_expressions_agree_across_forms_seeds_7500_to_9999() {
    random_expressions_agree(7_500..10_000);
}
