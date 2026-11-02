use super::*;
use crate::scores::parse_scores;

fn row(dataset: &str, method: &str, value: &str, status: &str) -> String {
    valid_row(dataset, method, value, status, "True")
}

fn valid_row(dataset: &str, method: &str, value: &str, status: &str, valid: &str) -> String {
    format!("t,{dataset},{method},public-development,rmse,{value},{status},{valid},k,,,,\n")
}

fn inputs<'a>(
    ladder: Ladder,
    names: &'a BTreeMap<String, String>,
    pending: &'a BTreeMap<String, usize>,
) -> Inputs<'a> {
    Inputs {
        track_id: "t",
        version: Some("v1"),
        ladder,
        scores_sha256: "s",
        corrections: None,
        method_names: names,
        pending,
        total_datasets: 3,
        direction: Direction::Minimize,
    }
}

/// W12-T04: a second delivery of one rankable (dataset, method) cell
/// cannot replace the first or add a match; the ladder refuses it, as the
/// independent replay (`jope_records::elo_history::parse_scores`) does, and
/// before any validity filtering.
#[test]
fn a_duplicate_cell_delivery_is_refused_not_silently_replaced() {
    let header = || {
        let mut text = crate::scores::COLUMNS.join(",");
        text.push('\n');
        text
    };
    let names = BTreeMap::new();
    let pending = BTreeMap::new();
    // Positive control: one delivery per cell.
    let mut single = header();
    single += &row("d1", "a", "1.0", "ok");
    single += &row("d1", "b", "2.0", "ok");
    let board = build_leaderboard(
        &parse_scores(&single).unwrap(),
        inputs(Ladder::Published, &names, &pending),
    )
    .unwrap();
    assert_eq!(board.rows[0].method_id, "a");
    // A second delivery of (d1, a) that would flip the match.
    let mut duplicate = single.clone();
    duplicate += &row("d1", "a", "3.0", "ok");
    for ladder in [Ladder::Published, Ladder::BenchmarkValid] {
        let refused = build_leaderboard(
            &parse_scores(&duplicate).unwrap(),
            inputs(ladder, &names, &pending),
        );
        assert!(
            matches!(&refused, Err(e) if e.to_string().contains("duplicate delivery")),
            "a duplicate (dataset, method) cell was silently replaced: {:?}",
            refused.map(|b| b
                .rows
                .iter()
                .map(|r| r.method_id.clone())
                .collect::<Vec<_>>())
        );
    }
    // Also when the duplicate is a benchmark-invalid row the valid ladder would drop.
    let mut invalid_duplicate = single;
    invalid_duplicate += &valid_row("d1", "a", "3.0", "ok", "False");
    let refused = build_leaderboard(
        &parse_scores(&invalid_duplicate).unwrap(),
        inputs(Ladder::BenchmarkValid, &names, &pending),
    );
    assert!(
        matches!(&refused, Err(e) if e.to_string().contains("duplicate delivery")),
        "a duplicate delivery hidden by the validity filter was accepted"
    );
}

#[test]
fn a_maximize_track_ranks_the_largest_value_first() {
    // The same rows as the reference-shape test, read as "higher wins": b now beats a on d1
    // and c on d3, a still takes the d2 tie by id.
    let mut text = crate::scores::COLUMNS.join(",");
    text.push('\n');
    text += &row("d1", "a", "1.0", "ok");
    text += &row("d1", "b", "2.0", "ok");
    text += &row("d2", "a", "1.0", "ok");
    text += &row("d2", "b", "1.0", "ok");
    text += &row("d3", "c", "0.5", "ok");
    text += &row("d3", "b", "0.7", "ok");
    let scores = parse_scores(&text).unwrap();
    let names = BTreeMap::new();
    let pending = BTreeMap::new();
    let board = build_leaderboard(
        &scores,
        Inputs {
            direction: Direction::Maximize,
            ..inputs(Ladder::Published, &names, &pending)
        },
    )
    .unwrap();
    let ids: Vec<&str> = board.rows.iter().map(|r| r.method_id.as_str()).collect();
    assert_eq!(ids[0], "b", "{ids:?}");
    let b = &board.rows[0];
    assert_eq!(b.dataset_wins, 2);
    assert_eq!(b.average_rank, "1.333");
    let a = board.rows.iter().find(|r| r.method_id == "a").unwrap();
    assert_eq!(a.dataset_wins, 1, "the d2 tie goes to a by id");
    assert_eq!(a.average_rank, "1.5");
    let c = board.rows.iter().find(|r| r.method_id == "c").unwrap();
    assert_eq!(c.dataset_wins, 0);
    assert_eq!(c.average_rank, "2.0");
    // Under minimize the same rows put c first (the reference-shape test); the fit is the
    // same code with the outcomes flipped.
    let minimized =
        build_leaderboard(&scores, inputs(Ladder::Published, &names, &pending)).unwrap();
    assert_eq!(minimized.rows[0].method_id, "c");
}

#[test]
fn ladder_matches_the_python_reference_shape() {
    let mut text = crate::scores::COLUMNS.join(",");
    text.push('\n');
    text += &row("d1", "a", "1.0", "ok");
    text += &row("d1", "b", "2.0", "ok");
    text += &row("d2", "a", "1.0", "ok");
    text += &row("d2", "b", "1.0", "ok");
    text += &row("d2", "c", "", "missing");
    text += &row("d3", "c", "0.5", "ok");
    text += &row("d3", "b", "0.7", "ok");
    let scores = parse_scores(&text).unwrap();
    let names = BTreeMap::from([("a".to_owned(), "Method A".to_owned())]);
    let pending = BTreeMap::from([("c".to_owned(), 1)]);
    let board = build_leaderboard(&scores, inputs(Ladder::Published, &names, &pending)).unwrap();
    assert_eq!(board.rankable_rows, 6);
    assert_eq!(board.datasets_scored, 3);
    assert_eq!(board.excluded_rows, 0);
    assert_eq!(board.scores_sha256, "s");
    assert!(board.rows.iter().all(|r| r.benchmark_valid == "True"));
    let ids: Vec<&str> = board.rows.iter().map(|r| r.method_id.as_str()).collect();
    assert_eq!(
        ids,
        ["c", "a", "b"],
        "c only ever beat b; a beat b once and tied once"
    );
    let a = &board.rows[1];
    assert_eq!(a.method, "Method A");
    assert_eq!(a.dataset_wins, 2, "a wins d1 and the d2 tie by id");
    assert_eq!(a.datasets_scored, 2);
    assert_eq!(a.full_coverage, "False");
    assert_eq!(a.full_coverage_rank, "");
    assert_eq!(a.coverage_pct, "66.7");
    assert_eq!(a.average_rank, "1.0");
    assert_eq!(a.coverage_status, "2/3");
    let b = &board.rows[2];
    assert_eq!(b.full_coverage, "True");
    assert_eq!(b.full_coverage_rank, "1");
    assert_eq!(b.average_rank, "2.0");
    assert_eq!(b.coverage_pct, "100.0");
    let c = &board.rows[0];
    assert_eq!(c.pending_cells, 1);
    assert_eq!(c.coverage_status, "1/3 · 1 pending");
    assert_eq!(c.dataset_wins, 1);
    let csv = to_csv(&board.rows).unwrap();
    assert!(csv.starts_with("elo_rank,full_coverage_rank,method_id,method,elo,average_rank,dataset_wins,datasets_scored,datasets_total,coverage_pct,full_coverage,pending_cells,coverage_status,protocols,benchmark_valid,scope\n"));
    assert_eq!(parse_csv(&csv).unwrap(), board.rows);
    let mean: f64 = board.ratings.values().sum::<f64>() / 3.0;
    assert!((mean - 1500.0).abs() < 1e-9);
    let snapshot = Snapshot::new(&board).unwrap();
    assert_eq!(snapshot.schema, SNAPSHOT_SCHEMA);
    assert_eq!(snapshot.ladder, Ladder::Published);
    assert_eq!(snapshot.leaderboard_sha256, sha256_hex(csv.as_bytes()));
    let text = serde_json::to_string(&snapshot).unwrap();
    assert!(text.contains("\"ladder\":\"published\""), "{text}");
    assert_eq!(serde_json::from_str::<Snapshot>(&text).unwrap(), snapshot);
}

#[test]
fn benchmark_valid_ladder_drops_invalid_rows_and_published_ladder_marks_them() {
    // Method `a` wins d1 with a test-adaptive row; `b` is the honest full-coverage entry.
    let mut text = crate::scores::COLUMNS.join(",");
    text.push('\n');
    text += &valid_row("d1", "a", "0.1", "ok", "False");
    text += &row("d1", "b", "0.5", "ok");
    text += &row("d2", "a", "0.4", "ok");
    text += &row("d2", "b", "0.5", "ok");
    text += &row("d3", "b", "0.5", "ok");
    text += &valid_row("d3", "c", "0.2", "ok", "maybe");
    let scores = parse_scores(&text).unwrap();
    let names = BTreeMap::new();
    let pending = BTreeMap::new();
    let error =
        build_leaderboard(&scores, inputs(Ladder::Published, &names, &pending)).unwrap_err();
    assert!(
        error.to_string().contains("neither True nor False"),
        "{error}"
    );
    let scores: Vec<ScoreRow> = scores.into_iter().filter(|r| r.method_id != "c").collect();

    let published =
        build_leaderboard(&scores, inputs(Ladder::Published, &names, &pending)).unwrap();
    assert_eq!(published.rankable_rows, 5);
    assert_eq!(published.excluded_rows, 0);
    let a = published.rows.iter().find(|r| r.method_id == "a").unwrap();
    assert_eq!(a.datasets_scored, 2);
    assert_eq!(
        a.benchmark_valid, "False",
        "one invalid row marks the method's published standing"
    );
    assert_eq!(
        published.rows[0].method_id, "a",
        "a wins both datasets as published"
    );
    let b = published.rows.iter().find(|r| r.method_id == "b").unwrap();
    assert_eq!(b.benchmark_valid, "True");
    assert_eq!(published.rows[0].scope, Ladder::Published.scope());

    let valid =
        build_leaderboard(&scores, inputs(Ladder::BenchmarkValid, &names, &pending)).unwrap();
    assert_eq!(valid.rankable_rows, 4);
    assert_eq!(valid.excluded_rows, 1);
    assert_eq!(valid.ladder, Ladder::BenchmarkValid);
    let a = valid.rows.iter().find(|r| r.method_id == "a").unwrap();
    assert_eq!(
        a.datasets_scored, 1,
        "the invalid d1 row never entered the fit"
    );
    assert_eq!(a.full_coverage, "False");
    assert_eq!(a.benchmark_valid, "True");
    assert_eq!(a.dataset_wins, 1, "a still wins d2 honestly");
    assert!(valid.rows.iter().all(|r| r.benchmark_valid == "True"));
    assert_eq!(valid.rows[0].scope, Ladder::BenchmarkValid.scope());
    let snapshot = Snapshot::new(&valid).unwrap();
    assert_eq!(snapshot.excluded_rows, 1);
    assert!(serde_json::to_string(&snapshot)
        .unwrap()
        .contains("\"ladder\":\"benchmark-valid\""));
}

#[test]
fn pending_counts_only_unresolved_cells() {
    let ledger = "{\"method_id\":\"x\",\"resolved\":false}\n{\"method_id\":\"x\",\"resolved\":true}\n\n{\"method_id\":\"y\",\"resolved\":false}\n";
    let pending = pending_by_method(ledger).unwrap();
    assert_eq!(
        pending,
        BTreeMap::from([("x".to_owned(), 1), ("y".to_owned(), 1)])
    );
    assert!(pending_by_method("{\"method_id\":\"x\"}\n").is_err());
}

#[test]
fn results_resolution_prefers_versions_then_flat() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let base = root.join("results/tr");
    fs::create_dir_all(base.join("v1")).unwrap();
    fs::create_dir_all(base.join("v2")).unwrap();
    fs::create_dir_all(base.join("empty")).unwrap();
    assert!(resolve_results(root, "tr", None).is_err());
    fs::write(base.join("scores.csv"), "").unwrap();
    assert_eq!(
        resolve_results(root, "tr", None).unwrap(),
        ResultsLocation {
            dir: base.clone(),
            version: None
        }
    );
    fs::write(base.join("v1/scores.csv"), "").unwrap();
    crate::io::write_gz(&base.join("v2/scores.csv.gz"), b"").unwrap();
    assert_eq!(list_versions(root, "tr").unwrap(), ["v1", "v2"]);
    assert_eq!(
        resolve_results(root, "tr", None)
            .unwrap()
            .version
            .as_deref(),
        Some("v2")
    );
    assert_eq!(
        resolve_results(root, "tr", Some("v1")).unwrap().dir,
        base.join("v1")
    );
    assert!(resolve_results(root, "tr", Some("empty"))
        .unwrap_err()
        .to_string()
        .contains("versions present"));
}
