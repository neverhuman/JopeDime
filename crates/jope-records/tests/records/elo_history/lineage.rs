use super::*;

const A100: &str = "L-lineage-b-a100-recovery";
const AUDIT: &str = "L-native-audit-108-blocks";

#[test]
fn continuity_only_through_retained_edges() {
    let document = record();
    let repo = repo();
    let path = |granted: Value| {
        granted
            .get("path")
            .and_then(Value::as_list)
            .unwrap()
            .clone()
    };
    let granted = path(continuity(&repo, &document, A100, "6000", "60000", Zstd::Builtin).unwrap());
    assert_eq!(granted[0], Value::from("60000"));
    assert_eq!(granted.last(), Some(&Value::from("6000")));
    assert_eq!(granted.len(), 19); // one block-initialization edge per block
    let audit = path(
        continuity(
            &repo,
            &document,
            AUDIT,
            "runs/continuous-learning-v2-20260909/cycle-000000/train-2",
            "runs/continuous-learning-v2-20260909/cycle-000001/train-2",
            Zstd::Builtin,
        )
        .unwrap(),
    );
    assert_eq!(audit.len(), 2);
    let refusals = [
        ((A100, "3000", "60000"), "3000 is not a retained checkpoint"),
        ((A100, "4000", "4477"), "missing lineage edge above 4477"),
        (
            (A100, "6000", "70000"),
            "70000 is not a retained checkpoint",
        ),
        (
            ("L-lineage-a-xbabe2", "a", "b"),
            "irrecoverable lineage supports no continuity",
        ),
        (("L-unknown", "a", "b"), "unknown lineage"),
    ];
    for ((lineage, ancestor, descendant), reason) in refusals {
        assert_refused(
            continuity(
                &repo,
                &document,
                lineage,
                ancestor,
                descendant,
                Zstd::Builtin,
            ),
            reason,
        );
    }
    for lineage in [A100, AUDIT, "L-lineage-a-xbabe2"] {
        assert_refused(
            exposure(&document, lineage),
            "exposure edges are not retained",
        );
    }
}

type Mutate = fn(&mut Value);

#[test]
fn recorded_lineage_facts_must_match_the_table() {
    let mutations: [(&str, Mutate, &str); 4] = [
        (
            "inflated retained edges",
            |d| {
                let expected = at(lineage(d, A100), "expected");
                set(expected, "retained_parent_edges", Value::Int(78));
                set(expected, "unretained_parent_edges", Value::Int(0));
            },
            "recorded counts differ",
        ),
        (
            "hidden missing edge",
            |d| set(lineage(d, A100), "missing_edges", Value::List(vec![])),
            "recorded missing edges differ",
        ),
        (
            "table without columns",
            |d| drop(pop(lineage(d, AUDIT), "columns")),
            "lineage columns",
        ),
        (
            "irrecoverable table-less lineage relabelled",
            |d| {
                set(
                    lineage(d, "L-lineage-a-xbabe2"),
                    "disposition",
                    elo_history::REPRODUCIBLE.into(),
                )
            },
            "is irrecoverable and names what is missing",
        ),
    ];
    for (label, mutate, reason) in mutations {
        let mut document = fast(record());
        at(&mut document, "histories")
            .as_list_mut()
            .unwrap()
            .retain(|history| {
                history.get("id") != Some(&Value::from("H-render-elo-ranking-static"))
            });
        mutate(&mut document);
        let result = Copy::new(document).verify().map(|_| ());
        assert!(result.is_err(), "{label}: accepted");
        assert_refused(result, reason);
    }
}

#[test]
fn inconsistent_or_ambiguous_tables_are_refused() {
    let header = "state,parent,step,parent_step,cycle,corpus\n";
    let tables = [
        (
            "inconsistent node",
            format!("{header}s1,p,1,0,0,c\ns1,q,1,0,0,c\n"),
            "recorded inconsistently",
        ),
        (
            "duplicate label",
            format!("{header}s1,p,1,0,0,c\ns2,s1,1,1,0,c\n"),
            "labels are not unique",
        ),
    ];
    for (label, text, reason) in tables {
        let directory = scratch();
        std::fs::write(directory.path().join("table.csv"), &text).unwrap();
        let entry = json(&format!(
            r#"{{"id": "L-test", "columns": {{"state": "state", "parent": "parent", "label": "step",
                    "parent_label": "parent_step", "block": "cycle", "corpus": "corpus"}},
                    "table": {{"path": "table.csv", "sha256": "{}"}}}}"#,
            py::sha256_hex(text.as_bytes())
        ));
        let result = lineage_graph(directory.path(), &entry, Zstd::Builtin).map(|_| ());
        assert!(result.is_err(), "{label}: accepted");
        assert_refused(result, reason);
    }
}
