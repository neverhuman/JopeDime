use super::*;
use serde_json::{json, Value};

fn param(name: &str, kind: &str, bins: &[Value], required: bool) -> ParamSpec {
    ParamSpec {
        name: name.to_owned(),
        kind: kind.to_owned(),
        bins: bins.to_vec(),
        default: Value::Null,
        required,
    }
}

fn op(name: &str, role: &str, tasks: &[&str], params: Vec<ParamSpec>) -> OpSpec {
    OpSpec {
        name: name.to_owned(),
        role: role.to_owned(),
        params,
        allow_append: role == "transform",
        generate: true,
        takes_expr: false,
        tasks: tasks.iter().map(|task| (*task).to_owned()).collect(),
    }
}

fn spec(
    engine: &str,
    version: &str,
    transforms: Vec<(&str, OpSpec)>,
    heads: Vec<(&str, OpSpec)>,
) -> GrammarSpec {
    GrammarSpec {
        engine: engine.to_owned(),
        version: version.to_owned(),
        max_transforms: 6,
        transforms: transforms
            .into_iter()
            .map(|(name, op)| (name.to_owned(), op))
            .collect(),
        heads: heads
            .into_iter()
            .map(|(name, op)| (name.to_owned(), op))
            .collect(),
        expr: None,
        max_expr_stages: 3,
    }
}

#[test]
fn builder_sorts_ops_but_preserves_engine_and_task_order() {
    let first = spec(
        "first",
        "v1",
        vec![
            ("z_transform", op("z_transform", "transform", &[], vec![])),
            ("a_transform", op("a_transform", "transform", &[], vec![])),
        ],
        vec![
            (
                "ridge",
                op("ridge", "head", &["regression", "binary"], vec![]),
            ),
            (
                "tree",
                op("tree", "head", &["timeseries", "binary"], vec![]),
            ),
        ],
    );
    let second = spec(
        "second",
        "v2",
        vec![],
        vec![("logistic", op("logistic", "head", &["multiclass"], vec![]))],
    );
    let vocab = build_action_vocab(&[first, second]).unwrap();

    assert_eq!(
        vocab.tokens[..3]
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        vec!["PAD", "BOS", "EOS"]
    );
    assert_eq!(vocab.engines, vec!["first".to_owned(), "second".to_owned()]);
    assert!(vocab.index["OP=first:a_transform"] < vocab.index["OP=first:z_transform"]);
    assert_eq!(
        vocab.tasks,
        vec![
            "regression".to_owned(),
            "binary".to_owned(),
            "timeseries".to_owned(),
            "multiclass".to_owned()
        ]
    );
    assert_eq!(vocab.version, "first:v1+second:v2");
    assert_eq!(
        vocab.head_tasks[&vocab.index["OP=first:tree"]],
        vec!["timeseries".to_owned(), "binary".to_owned()]
    );
    let serialized = serde_json::to_value(&vocab).unwrap();
    assert_eq!(serialized["tokens"][0], "PAD");
    assert_eq!(serialized["version"], "first:v1+second:v2");
    assert!(serialized.get("op_id_by_name").is_none());
}

#[test]
fn parameter_tokens_match_python_scalar_spelling_and_escaping() {
    assert_eq!(format_param_token(&json!(true)), "true");
    assert_eq!(format_param_token(&json!(2)), "2");
    assert_eq!(format_param_token(&json!(1.0)), "1.0");
    assert_eq!(format_param_token(&json!(1.25)), "1.25");
    assert_eq!(format_param_token(&json!(0.00001)), "1e-05");
    assert_eq!(format_param_token(&json!(0.0001)), "0.0001");
    assert_eq!(format_param_token(&json!("plain-1")), "plain-1");
    assert_eq!(format_param_token(&json!("two words")), "\"two words\"");
    assert_eq!(format_param_token(&json!("a\\b\"c")), "\"a\\\\b\\\"c\"");
}

#[test]
fn append_only_engine_and_task_growth_keeps_the_old_prefix() {
    let base_spec = spec(
        "base",
        "v1",
        vec![("scale", op("scale", "transform", &[], vec![]))],
        vec![("ridge", op("ridge", "head", &["regression"], vec![]))],
    );
    let base = build_action_vocab(&[base_spec]).unwrap();
    let extra = spec(
        "extra",
        "v2",
        vec![
            ("z", op("z", "transform", &[], vec![])),
            ("a", op("a", "transform", &[], vec![])),
        ],
        vec![("classifier", op("classifier", "head", &["binary"], vec![]))],
    );
    let extended = extend_action_vocab_append_only(&base, &extra).unwrap();
    assert_eq!(&extended.tokens[..base.tokens.len()], &base.tokens[..]);
    assert_eq!(extended.version, "base:v1+extra:v2");
    assert!(extended.index["OP=extra:z"] < extended.index["OP=extra:a"]);
    assert!(extend_action_vocab_append_only(&extended, &extra)
        .unwrap_err()
        .to_string()
        .contains("engine already present"));

    let with_tasks = append_missing_task_tokens(&extended).unwrap();
    assert_eq!(
        &with_tasks.tokens[..extended.tokens.len()],
        &extended.tokens[..]
    );
    assert_eq!(
        with_tasks.tasks,
        vec![
            "regression".to_owned(),
            "binary".to_owned(),
            "timeseries".to_owned(),
            "multiclass".to_owned()
        ]
    );
    assert_eq!(
        append_missing_task_tokens(&with_tasks).unwrap().tokens,
        with_tasks.tokens
    );
}

#[test]
fn reconciliation_grants_only_declared_nonexcluded_canonical_tasks() {
    let grammar = spec(
        "engine",
        "v1",
        vec![],
        vec![
            (
                "ridge",
                op(
                    "ridge",
                    "head",
                    &["regression", "binary", "timeseries"],
                    vec![],
                ),
            ),
            (
                "logistic",
                op("logistic", "head", &["binary", "multiclass"], vec![]),
            ),
        ],
    );
    let base = build_action_vocab(&[spec(
        "engine",
        "v1",
        vec![],
        vec![
            ("ridge", op("ridge", "head", &["regression"], vec![])),
            ("logistic", op("logistic", "head", &["binary"], vec![])),
        ],
    )])
    .unwrap();
    let before = base.tokens.clone();
    let granted = reconcile_head_tasks_from_grammar(
        &base,
        &grammar,
        &[
            "multiclass".into(),
            "timeseries".into(),
            "binary".into(),
            "binary".into(),
        ],
        &["logistic".into()],
    )
    .unwrap();
    assert_eq!(granted.tokens, before);
    assert_eq!(
        granted.head_tasks[&base.index["OP=engine:ridge"]],
        vec![
            "regression".to_owned(),
            "binary".to_owned(),
            "timeseries".to_owned()
        ]
    );
    assert_eq!(
        granted.head_tasks[&base.index["OP=engine:logistic"]],
        vec!["binary".to_owned()]
    );
    assert!(reconcile_head_tasks_from_grammar(&base, &grammar, &["other".into()], &[]).is_err());
}

#[test]
fn compatibility_ops_follow_requested_order_and_refuse_duplicates() {
    let grammar = spec(
        "engine",
        "v1",
        vec![
            ("append", op("append", "transform", &[], vec![])),
            (
                "compat",
                op(
                    "compat",
                    "transform",
                    &[],
                    vec![param("mode", "str", &[json!("fast")], false)],
                ),
            ),
        ],
        vec![("ridge", op("ridge", "head", &["regression"], vec![]))],
    );
    let base = build_action_vocab(&[spec(
        "engine",
        "v1",
        vec![],
        vec![("ridge", op("ridge", "head", &["regression"], vec![]))],
    )])
    .unwrap();
    let grown = extend_engine_ops_append_only(&base, &grammar, &["compat".into(), "append".into()])
        .unwrap();
    assert_eq!(&grown.tokens[..base.tokens.len()], &base.tokens[..]);
    assert!(grown.index["OP=engine:compat"] < grown.index["OP=engine:append"]);
    assert_eq!(grown.version, "engine:v1+engine:compat");
    assert!(extend_engine_ops_append_only(&grown, &grammar, &["compat".into()]).is_err());
}
