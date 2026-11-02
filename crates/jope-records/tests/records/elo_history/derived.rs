use super::*;

fn plot(change: fn(&mut Vec<Dict>)) -> impl FnOnce(Vec<u8>) -> Vec<u8> {
    move |data| {
        let text = String::from_utf8(data).unwrap();
        let mut rows = pycsv::dict_reader(&text, Newlines::Universal).unwrap().rows;
        change(&mut rows);
        let fieldnames: Vec<String> = rows[0].keys().map(str::to_owned).collect();
        let mut out = String::new();
        pycsv::write_row(&mut out, &fieldnames);
        for row in &rows {
            let cells: Vec<String> = fieldnames
                .iter()
                .map(|f| row.get(f).map_or_else(String::new, pycsv::cell))
                .collect();
            pycsv::write_row(&mut out, &cells);
        }
        out.into_bytes()
    }
}

#[test]
fn plot_rows_must_repeat_reproducible_snapshots() {
    let document = record();
    let mut histories: HashMap<Key, &Value> = HashMap::new();
    for h in document.get("histories").and_then(Value::as_list).unwrap() {
        histories.insert(Key::of(h.get("id").unwrap()).unwrap(), h);
    }
    let view = document
        .get("derived_views")
        .and_then(Value::as_list)
        .unwrap()[0]
        .clone();
    let copy = Copy::new(document.clone());
    assert_eq!(
        derived_view(&copy.root, &view, &histories, Zstd::Builtin).unwrap(),
        661
    ); // positive control
    #[allow(clippy::ptr_arg)] // one signature for every plot change; `drop_row` pops
    fn raise_rating(rows: &mut Vec<Dict>) {
        rows[0].insert("elo", "1999.9".into());
    }
    fn drop_row(rows: &mut Vec<Dict>) {
        rows.pop();
    }
    type Change = fn(&mut Vec<Dict>);
    type Case = (
        &'static str,
        Option<Change>,
        Option<&'static str>,
        &'static str,
    );
    let cases: [Case; 3] = [
        (
            "raised rating",
            Some(raise_rating as Change),
            None,
            "differs from its snapshot",
        ),
        (
            "dropped row",
            Some(drop_row as Change),
            None,
            "only partly plotted",
        ),
        (
            "irrecoverable source",
            None,
            Some("H-legacy-round1-internal-elo"),
            "not backed by a reproducible history",
        ),
    ];
    for (label, change, source, reason) in cases {
        let mut changed = view.clone();
        let copy = Copy::new(document.clone());
        if let Some(change) = change {
            rewrite(&copy.root, at(&mut changed, "table"), plot(change));
        }
        if let Some(source) = source {
            set(
                at(&mut changed, "snapshots"),
                "same-cut-v1-deep",
                source.into(),
            );
        }
        let result = derived_view(&copy.root, &changed, &histories, Zstd::Builtin);
        assert!(result.is_err(), "{label}: accepted");
        assert_refused(result, reason);
    }
}
