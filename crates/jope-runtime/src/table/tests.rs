use super::*;

fn limits() -> Limits {
    Limits::default()
}

#[test]
fn missing_cells_categories_and_kinds_follow_the_contract() {
    let csv = parse_csv(
        "\u{feff}x, color ,y\n1.5,NA,3\n,red,4\n2, ,5\n".as_bytes(),
        &limits(),
    )
    .unwrap();
    assert_eq!(csv.headers, ["x", "color", "y"]);
    assert_eq!(csv.rows[1][0], None);
    assert_eq!(csv.rows[2][1], None);
    let training = training(&csv, "y", TaskChoice::Auto, &limits()).unwrap();
    assert_eq!(training.kind, TaskKind::Regression);
    assert_eq!(training.schema.columns[0].kind, ColumnKind::Numeric);
    assert_eq!(training.schema.columns[1].kind, ColumnKind::Categorical);
    assert_eq!(training.targets, [3.0, 4.0, 5.0]);
    assert_eq!(
        training.table.rows[0][1],
        format!("{CATEGORY_PREFIX}\"NA\"")
    );
    assert_eq!(training.table.rows[2][1], "");
}

#[test]
fn tasks_and_class_order() {
    let binary = parse_csv(b"a,y\n1,yes\n2,no\n3,yes\n", &limits()).unwrap();
    let t = training(&binary, "y", TaskChoice::Auto, &limits()).unwrap();
    assert_eq!(
        (t.kind, t.classes.clone()),
        (TaskKind::Binary, vec!["no".into(), "yes".into()])
    );
    assert_eq!(t.targets, [1.0, 0.0, 1.0]);
    let numeric = parse_csv(b"a,y\n1,10\n2,2\n3,9\n4,2\n", &limits()).unwrap();
    let t = training(
        &numeric,
        "y",
        TaskChoice::Fixed(TaskKind::Multiclass),
        &limits(),
    )
    .unwrap();
    assert_eq!(t.classes, ["2", "9", "10"]);
    let t = training(&numeric, "y", TaskChoice::Auto, &limits()).unwrap();
    assert_eq!(t.kind, TaskKind::Regression);
    let labels = parse_csv(b"a,y\n1,cat\n2,dog\n3,eel\n", &limits()).unwrap();
    assert_eq!(
        training(&labels, "y", TaskChoice::Auto, &limits())
            .unwrap()
            .kind,
        TaskKind::Multiclass
    );
    assert!(training(&labels, "y", TaskChoice::Fixed(TaskKind::Binary), &limits()).is_err());
    assert!(training(
        &labels,
        "y",
        TaskChoice::Fixed(TaskKind::Regression),
        &limits()
    )
    .is_err());
}

#[test]
fn malformed_input_is_refused_with_an_input_failure() {
    let refuse = |text: &str| {
        let error = parse_csv(text.as_bytes(), &limits())
            .and_then(|csv| training(&csv, "y", TaskChoice::Auto, &limits()).map(|_| ()))
            .unwrap_err();
        assert_eq!(
            crate::error::kind_of(&error),
            crate::FailureKind::Input,
            "{text}: {error:#}"
        );
    };
    refuse("a,a,y\n1,2,3\n");
    refuse("a,,y\n1,2,3\n");
    refuse("a,y\n1,2\n3\n");
    refuse("a,b\n1,2\n");
    refuse("a,y\n1,\n");
    refuse("y\n1\n2\n");
    refuse("a,y\n");
    let small = Limits {
        max_rows: 1,
        ..Limits::default()
    };
    assert!(parse_csv(b"a,y\n1,2\n3,4\n", &small).is_err());
}

#[test]
fn prediction_columns_match_by_name_and_refuse_unknown_ones() {
    let csv = parse_csv(b"a,b,y\n1,x,3\n2,y,4\n", &limits()).unwrap();
    let t = training(&csv, "y", TaskChoice::Auto, &limits()).unwrap();
    let reordered = parse_csv(b"b,y,a\nx,0,1\n", &limits()).unwrap();
    let (table, ignored) = prediction(&reordered, &t.schema, "y").unwrap();
    assert_eq!(table.headers, ["a", "b"]);
    assert_eq!(ignored, ["y"]);
    let extra = parse_csv(b"a,b,id\n1,x,7\n", &limits()).unwrap();
    assert!(prediction(&extra, &t.schema, "y").is_err());
    let missing = parse_csv(b"a\n1\n", &limits()).unwrap();
    assert!(prediction(&missing, &t.schema, "y").is_err());
    let bad_number = parse_csv(b"a,b\nseven,x\n", &limits()).unwrap();
    assert!(prediction(&bad_number, &t.schema, "y").is_err());
}

#[test]
fn objective_columns_are_numeric_never_features_and_ignored_at_prediction() {
    let regression = || TaskChoice::Fixed(TaskKind::Regression);
    let wanted = |name: &str| -> BTreeSet<String> { [name.to_owned()].into() };
    let csv = parse_csv(b"a,price,y\n1,10,3\n2,12.5,4\n", &limits()).unwrap();
    let t = training_with_auxiliary(&csv, "y", regression(), &limits(), &wanted("price")).unwrap();
    assert_eq!(t.table.headers, ["a"]);
    assert_eq!(t.auxiliary["price"], [10.0, 12.5]);
    let input = parse_csv(b"price,a\n9,1\n", &limits()).unwrap();
    let (table, ignored) =
        prediction_ignoring(&input, &t.schema, "y", &["price".to_owned()]).unwrap();
    assert_eq!(table.headers, ["a"]);
    assert_eq!(ignored, ["price"]);
    assert!(prediction(&input, &t.schema, "y").is_err());
    for (text, column) in [
        (&b"a,price,y\n1,,3\n2,1,4\n"[..], "price"),
        (b"a,price,y\n1,cheap,3\n2,1,4\n", "price"),
        (b"a,y\n1,3\n2,4\n", "price"),
        (b"a,y\n1,3\n2,4\n", "y"),
        (b"price,y\n1,3\n2,4\n", "price"),
    ] {
        let csv = parse_csv(text, &limits()).unwrap();
        assert!(
            training_with_auxiliary(&csv, "y", regression(), &limits(), &wanted(column)).is_err(),
            "{column}"
        );
    }
}
