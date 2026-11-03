// Extracted from inference_contract_tests.rs; test behavior is unchanged.
use super::*;

/// Present category strings that must stay data in every backend. Several imitate
/// shell, Python, JavaScript and Rust syntax, template markers or the missing sentinel.
const HOSTILE_JS_BOUNDARY_FIXTURE: &str = concat!(
    "'); require('child_",
    "process').ex",
    "ecSync('touch PWNED'); ('"
);

pub(crate) const HOSTILE_CATEGORIES: [&str; 18] = [
    "NA",
    "null",
    "",
    "na",
    "line\n\"quoted\"\u{2028}",
    "__PREDICT____REGRESSION____CLASSES_JSON__",
    "\"; import os; os.system('touch PWNED') #",
    HOSTILE_JS_BOUNDARY_FIXTURE,
    "\\\"}]); panic!(\"x\"); //",
    "\u{0}MISSING",
    "jope.category.v1:\"NA\"",
    "\u{1f642} emoji",
    "\u{7f}\u{1b}[31m",
    "*/ /*",
    "${HOME}",
    "$(touch PWNED)",
    "`touch PWNED`",
    "\\u0041",
];

pub(crate) fn columns() -> Vec<Column> {
    vec![
        Column {
            name: "amount".into(),
            kind: Kind::Numeric,
        },
        Column {
            name: "country".into(),
            kind: Kind::Categorical,
        },
    ]
}

/// Fit one registered recipe on explicit synthetic rows. `amount_missing_everywhere`
/// makes the numeric training column entirely null.
pub(crate) fn fit(
    task: Task,
    family: &str,
    categories: &[Option<String>],
    repeats: &dyn Fn(usize) -> usize,
    amount_missing_everywhere: bool,
) -> Result<FittedRustArtifact> {
    let method = rust_methods(task)?
        .into_iter()
        .find(|method| method.family == family)
        .context("family")?;
    fit_method(task, method, categories, repeats, amount_missing_everywhere)
}

pub(crate) fn fit_method(
    task: Task,
    method: super::super::methods::RustMethod,
    categories: &[Option<String>],
    repeats: &dyn Fn(usize) -> usize,
    amount_missing_everywhere: bool,
) -> Result<FittedRustArtifact> {
    let mut rows = Vec::new();
    let mut targets = Vec::new();
    let mut index = 0usize;
    for (position, category) in categories.iter().enumerate() {
        for _ in 0..repeats(position) {
            let amount = if amount_missing_everywhere || index.is_multiple_of(5) {
                None
            } else {
                Some(format!("{}", index as f64 * 0.75 - 4.0))
            };
            rows.push(vec![amount, category.clone()]);
            targets.push(match task {
                Task::Regression => (index % 7) as f64 * 1.5 + position as f64 * 0.25,
                Task::Binary => ((position + index) % 2) as f64,
                _ => (index % 3) as f64,
            });
            index += 1;
        }
    }
    let classes: Vec<String> = match task {
        Task::Regression => vec![],
        Task::Binary => vec![
            "no\"; os.system('touch PWNED')".into(),
            "yes\u{2028}".into(),
        ],
        _ => vec!["zebra".into(), "`touch PWNED`".into(), "${x}".into()],
    };
    let request = RustFitRequest {
        task,
        schema: DataSchema {
            columns: vec![
                ColumnSpec::numeric("amount"),
                ColumnSpec::categorical("country"),
            ],
            row_semantics: RowSemantics::Iid,
        },
        classes,
        train: raw_table(&columns(), &rows)?,
        targets,
        predict: raw_table(&columns(), &[vec![Some("1".into()), None]])?,
        method,
    };
    Ok(fit_rust(&request)?.artifact)
}

pub(crate) fn hostile_categories() -> Vec<Option<String>> {
    let mut categories: Vec<Option<String>> = vec![None];
    categories.extend(
        HOSTILE_CATEGORIES
            .iter()
            .map(|value| Some((*value).to_owned())),
    );
    categories
}

pub(crate) fn hostile_fixture(task: Task, family: &str) -> FittedRustArtifact {
    fit(task, family, &hostile_categories(), &|_| 3, false).unwrap()
}

pub(crate) fn row(values: &[Option<&str>]) -> Value {
    Value::Array(
        values
            .iter()
            .map(|value| value.map_or(Value::Null, |text| Value::String(text.into())))
            .collect(),
    )
}

/// Raw JSON cases. `None` typed rows mean the case is not representable through the
/// typed Rust/native API (JSON type confusion); every dynamic backend must refuse it.
pub(crate) struct Case {
    pub name: String,
    pub text: String,
    pub typed: Option<Vec<Vec<Option<String>>>>,
    pub expect_refusal: bool,
}

pub(crate) fn typed(value: &Value) -> Option<Vec<Vec<Option<String>>>> {
    value
        .as_array()?
        .iter()
        .map(|row| {
            row.as_array()?
                .iter()
                .map(|cell| match cell {
                    Value::Null => Some(None),
                    Value::String(text) => Some(Some(text.clone())),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()
        })
        .collect()
}

pub(crate) fn case(name: &str, rows: Value, refuse: bool) -> Case {
    Case {
        name: name.into(),
        text: serde_json::to_string(&rows).unwrap(),
        typed: typed(&rows),
        expect_refusal: refuse,
    }
}

pub(crate) fn raw_cases(categories: &[&str]) -> Vec<Case> {
    let n = |amount: &str| row(&[Some(amount), Some("NA")]);
    let mut cases = vec![
        case("all-missing-row", json!([[null, null]]), false),
        case(
            "all-missing-column-batch",
            json!([[null, "NA"], [null, null], [null, "unseen"]]),
            false,
        ),
        case(
            "unseen-category",
            json!([["5", "never-seen-category"]]),
            false,
        ),
        case(
            "present-null-like-strings-are-not-missing",
            json!([
                ["3.5", "NA"],
                ["3.5", "null"],
                ["3.5", ""],
                ["3.5", "na"],
                ["3.5", null]
            ]),
            false,
        ),
        case(
            "numeric-lexical-forms",
            Value::Array(
                [
                    " 3.5\t",
                    "\u{a0}2",
                    "+1e2",
                    ".5",
                    "5.",
                    "-0.0",
                    "1e-320",
                    "1E3",
                    "\u{2028}3\u{2029}",
                    "\u{3000}4",
                    "0000012",
                    "-.5e-3",
                    "+.5",
                    "1.e5",
                    "4.9e-324",
                    "1.7976931348623157e308",
                    "-1e308",
                    "1e-400",
                ]
                .iter()
                .map(|value| n(value))
                .collect(),
            ),
            false,
        ),
        case(
            "hostile-present-categories",
            Value::Array(
                categories
                    .iter()
                    .map(|value| row(&[Some("1.25"), Some(value)]))
                    .collect(),
            ),
            false,
        ),
        // Deterministic negative security case: the child-process payload is categorical data.
        // The contract runner sends it through each backend and checks that no PWNED sentinel
        // appears, while this valid category remains accepted by the typed inference boundary.
        case(
            "negative-hostile-js-child-process-exec-is-inert-category-data",
            Value::Array(vec![row(&[
                Some("1.25"),
                Some(HOSTILE_JS_BOUNDARY_FIXTURE),
            ])]),
            false,
        ),
        case("long-category", json!([["2", "x".repeat(100_000)]]), false),
        case(
            "cardinality-fifty-rows",
            Value::Array(
                (0..50)
                    .map(|i| {
                        row(&[
                            if i % 4 == 0 {
                                None
                            } else {
                                Some(["1", "-2.5", "7e1"][i % 3])
                            },
                            if i % 6 == 0 {
                                None
                            } else {
                                Some(categories[i % categories.len()])
                            },
                        ])
                    })
                    .collect(),
            ),
            false,
        ),
        case("empty-input", json!([]), false),
    ];
    for (name, value) in [
        ("non-number", "abc"),
        ("nan", "NaN"),
        ("lower-nan", "nan"),
        ("inf", "inf"),
        ("negative-infinity", "-Infinity"),
        ("overflow", "1e400"),
        ("empty-string-numeric", ""),
        ("blank-numeric", "   "),
        ("underscore", "1_000"),
        ("hex", "0x10"),
        ("comma-decimal", "1,5"),
        ("arabic-indic-digit", "\u{661}"),
        ("dangling-exponent", "1e"),
        ("bare-exponent", "e5"),
        ("double-sign", "+-1"),
        ("two-points", "1.2.3"),
        ("unit-separator-padding", "\u{1c}5"),
        ("zero-width-space", "\u{200b}5"),
    ] {
        cases.push(case(
            &format!("malformed-numeric-{name}"),
            json!([n(value)]),
            true,
        ));
    }
    cases.extend([
        case("narrow-row", json!([["1"]]), true),
        case("extra-column", json!([["1", "NA", "extra"]]), true),
        case("empty-row", json!([[]]), true),
        case(
            "one-malformed-row-in-batch-has-no-partial-output",
            json!([["1", "NA"], ["bad", "NA"], ["2", "NA"]]),
            true,
        ),
        case("json-number-for-numeric", json!([[5, "NA"]]), true),
        case("json-number-for-category", json!([["5", 7]]), true),
        case("boolean-numeric", json!([[true, "NA"]]), true),
        case("boolean-category", json!([[null, false]]), true),
        case("nested-cell", json!([[["5"], "NA"]]), true),
        case(
            "named-record-row",
            json!([{"amount": "5", "country": "NA"}]),
            true,
        ),
        case("string-row", json!(["5,NA"]), true),
        case("object-batch", json!({"rows": []}), true),
        case("null-batch", json!(null), true),
        case("null-row", json!([null]), true),
    ]);
    // Not representable in serde_json::Value or Rust strings: a lone UTF-16 surrogate.
    cases.push(Case {
        name: "lone-surrogate-category".into(),
        text: "[[\"5\",\"\\ud800\"]]".into(),
        typed: None,
        expect_refusal: true,
    });
    cases
}
