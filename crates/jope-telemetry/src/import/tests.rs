use super::*;
use crate::gp::EditOp;
use crate::train;

fn tmpfile(dir: &tempfile::TempDir, name: &str, body: &str) -> std::path::PathBuf {
    let p = dir.path().join(name);
    std::fs::write(&p, body).unwrap();
    p
}

#[test]
fn engine_rows_map_and_validation_is_deduplicated() {
    let d = tempfile::tempdir().unwrap();
    let row = |step: u64, vstep: u64| {
        format!(
            r#"{{"schema":"parallel.neural_training.v1","step":{step},"epoch":0.5,"loss":2.0,"sequence_nll":1.5,"gradient_norm":0.7,"gradient_clipped":false,"step_seconds":0.25,"validation":{{"sequence_nll":1.1,"evaluated_step":{vstep},"source_families":3}}}}"#
        )
    };
    let body = format!("{}\n{}\n{{\"torn", row(1, 1), row(2, 1));
    let p = tmpfile(&d, "state.jsonl", &body);
    let out = engine_state_jsonl(&p).unwrap();
    let wall: Vec<_> = out.iter().filter(|s| s.metric == "step.wall").collect();
    assert_eq!(wall.len(), 2);
    assert_eq!(wall[0].value, 250.0);
    assert_eq!(
        out.iter()
            .filter(|s| s.metric == "val.loss.component")
            .count(),
        1
    );
    assert!(out
        .iter()
        .all(|s| s.src.source == "import" && s.src.file_sha256.len() == 64));
    let clipped = out.iter().find(|s| s.metric == "optim.clipped").unwrap();
    assert_eq!(
        (clipped.value, clipped.metric_id),
        (0.0, Some(train::GRAD_CLIPPED))
    );
}

#[test]
fn parse_utc_known_values() {
    // Reference values from Python's calendar.timegm.
    assert_eq!(parse_utc("1970-01-01T00:00:00Z"), Some(0));
    assert_eq!(parse_utc("2026-10-24T05:52:15Z"), Some(1_792_821_135));
    assert_eq!(parse_utc("2000-02-29T12:00:00.5Z"), Some(951_825_600));
    assert_eq!(parse_utc("2026-10-24 05:52:15Z"), None);
    assert_eq!(parse_utc("2026-13-01T00:00:00Z"), None);
}

#[test]
fn hwm_consistency_flags_beats_from_an_earlier_run() {
    let src = || SourceRef {
        source: "import".into(),
        file_name: "x".into(),
        file_sha256: "0".into(),
    };
    let beat = |cid: &str, t: &str| HwmBeat {
        schema: HWM_SCHEMA.into(),
        context_sha256: cid.into(),
        program_sha256: "p".into(),
        operator: EditOp::EditParameter,
        parents: vec![],
        previous_best_fitness: None,
        new_best_fitness: None,
        fitness_improvement: None,
        cv_gate_passed: None,
        cv_test_agreed: None,
        updated_at_utc: Some(t.into()),
        missing_identity_fields: vec![],
        src: src(),
    };
    let ret = Retention {
        schema: RETENTION_SCHEMA.into(),
        round: 1,
        created_at_utc: Some("2026-10-24T05:52:15Z".into()),
        wall_s: Some(183.26),
        counts: [("new_hwm_count".to_owned(), 0u64)].into(),
        src: src(),
    };
    let beats = [
        beat("a", "2026-10-24T00:59:18Z"),
        beat("b", "2026-10-24T01:03:35Z"),
    ];
    let c = hwm_consistency(1, &beats, &ret, 60);
    assert_eq!((c.files, c.in_round_window, c.stale.len()), (2, 0, 2));
    assert_eq!(
        c.count_matches,
        Some(true),
        "0 in-window beats == new_hwm_count 0"
    );
    let fresh = [beat("c", "2026-10-24T05:50:00Z")];
    let c = hwm_consistency(1, &fresh, &ret, 60);
    assert_eq!((c.in_round_window, c.count_matches), (1, Some(false)));
}

#[test]
fn hwm_lists_missing_identity_and_drops_dsl() {
    let d = tempfile::tempdir().unwrap();
    let p = tmpfile(
        &d,
        "c.json",
        r#"{"schema":"jope.hwm/v1","context_sha256":"aa","program_sha256":"bb","dsl":"(pipe X (a) (b))","operator":"edit_replace_head","parents":["cc"],"previous_best_fitness":-1.0,"new_best_fitness":-1.1,"fitness_improvement":0.1,"cv_gate_passed":true}"#,
    );
    let h = hwm_file(&p).unwrap();
    assert_eq!(h.operator, EditOp::EditReplaceHead);
    assert_eq!(h.missing_identity_fields.len(), HWM_IDENTITY_FIELDS.len());
    assert!(!serde_json::to_string(&h).unwrap().contains("pipe X"));
}

#[test]
fn retained_scores_summary_has_no_provenance_claim() {
    let d = tempfile::tempdir().unwrap();
    let rows = [
        r#"{"schema":"jope.score/v1","dsl":"(pipe X (a q=1) (h))","fitness":-0.5,"cv_gate_passed":true,"operator":"edit_parameter","retention_tier":"top_k_winner"}"#,
        r#"{"schema":"jope.score/v1","dsl":"(pipe X (a q=2) (h))","fitness":-0.4,"cv_gate_passed":true,"operator":"edit_parameter","retention_tier":"top_k_winner"}"#,
        r#"{"schema":"jope.score/v1","dsl":"(pipe X (b) (h))","fitness":100.0,"cv_gate_passed":false,"operator":"edit_insert_stage","retention_tier":"hard_negative"}"#,
    ];
    let p = tmpfile(&d, "s.jsonl", &rows.join("\n"));
    let g = retained_scores(&p, 1, Some(4)).unwrap();
    assert_eq!(g.pop_size, 3);
    assert_eq!(g.status.gate_passed, 2);
    assert_eq!(
        g.fitness_passers.max,
        Some(-0.4),
        "sentinel 100.0 stays out of the quantiles"
    );
    assert!(g.provenance_population.novel.is_none());
    assert_eq!(g.ops_distinct, 3);
    assert_eq!(g.ops_unused, Some(1));
}
