//! Rust counterparts for the data shaping and display helpers used by `jope.tui`.
//!
//! The Python TUI remains the active importer for now. This module keeps its helper behavior
//! available to Rust callers while that UI is still Python-owned.

mod code;
mod filesystem;
mod formatting;
mod rows;

pub use code::{
    clip_code, code_for_row, code_path, genome_stages, head_from_genome, inline_code,
    python_algorithm_preview, read_code, rust_algorithm_preview, split_transforms_head,
    stage_params, CODE_PREVIEW_CHARS, HEAD_OPS,
};
pub use filesystem::{
    dir_size, file_age, first_existing, gpu_states, json_objects_from_log, mount_label, read_json,
    read_jsonl, tail_lines,
};
pub use formatting::{
    fmt_age, fmt_float, fmt_int, fmt_pct, fmt_plain_float, freshness_style, load_style,
    ranking_style, temp_style, to_float, to_int, truncate, win_style,
};
pub use rows::{
    bar, candidate_key, code_token_total, eval_delta, evaluation_maps, freshest_age,
    gene_stream_entries, latest_algorithm_row, latest_rank_status, matching_eval, meter,
    metric_values, row_genome, spark, trend_arrow, CandidateKey, EvaluationMaps, GeneEntry,
    GenomeStage,
};

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn metrics_skip_non_numeric_and_use_fallback_only_when_needed() {
        let rows = vec![
            json!({"loss": "bad", "alternate": 2.5}),
            json!({"loss": 1, "alternate": 8}),
            json!({"loss": null, "alternate": true}),
            json!({"alternate": "3"}),
        ];
        assert_eq!(
            metric_values(&rows, "loss", Some("alternate")),
            vec![2.5, 1.0, 1.0]
        );
    }

    #[test]
    fn genome_stages_preserve_parameter_order_and_select_a_known_head() {
        let row = json!({"genome": "(pipe (scale center=true) (ridge alpha=0.1))"});
        let (transforms, head) = split_transforms_head(row.as_object().unwrap());
        assert_eq!(transforms.len(), 1);
        assert_eq!(transforms[0].0, "scale");
        assert_eq!(transforms[0].1, vec![("center".into(), "true".into())]);
        assert_eq!(head.0, "ridge");
    }

    #[test]
    fn entry_selection_prefers_dataset_specific_matches_and_deduplicates() {
        let proposals = vec![json!({"genome":"(ridge)","dataset_hash":"d"})];
        let audit = vec![json!({"genome":"(ridge)","dataset_hash":"d"})];
        let evaluations = vec![
            json!({"genome":"(ridge)","dataset_hash":"other","score":1}),
            json!({"genome":"(ridge)","dataset_hash":"d","score":2}),
        ];
        let entries = gene_stream_entries(&proposals, &audit, &evaluations, 10);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].0, "audit");
        assert_eq!(entries[0].2.as_ref().unwrap()["score"], 2);
        assert_eq!(entries[1].0, "gene");
    }

    #[test]
    fn formatting_helpers_keep_thresholds_and_clipping_rules() {
        assert_eq!(spark(&[], 5, "loss"), "loss: waiting");
        assert_eq!(trend_arrow(&[1.0, 1.1]), "▲");
        assert_eq!(meter(50.0, 100.0, 4), "[██░░]");
        assert_eq!(load_style(90.0), "bold red");
        assert_eq!(fmt_age(None), "—");
        assert_eq!(fmt_int(&json!(1234567)), "1,234,567");
        assert_eq!(truncate("abcdef", 5), "ab...");
    }

    #[test]
    fn file_readers_skip_bad_jsonl_and_keep_the_requested_tail() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("events.jsonl");
        std::fs::write(&path, "{\"n\":1}\nnot-json\n\n{\"n\":2}\n").unwrap();
        assert_eq!(read_jsonl(&path, 1), vec![json!({"n":2})]);
        assert_eq!(read_jsonl(&path, 0).len(), 2);
        assert_eq!(json_objects_from_log(&path).len(), 2);
        assert_eq!(tail_lines(&path, 1), vec!["{\"n\":2}"]);
        assert_eq!(read_json(&dir.path().join("missing.json")), json!({}));
    }

    #[test]
    fn inline_code_precedes_preview_and_clipping_uses_character_count() {
        let row = json!({"python_code":"inline source", "genome":"(scale center=true) (ridge alpha=0.1)"});
        let (code, origin) = code_for_row(row.as_object().unwrap(), "python", &[]);
        assert_eq!(code, "inline source");
        assert_eq!(origin, "inline");
        let clipped = clip_code(&"é".repeat(CODE_PREVIEW_CHARS + 1));
        assert_eq!(
            clipped.chars().count(),
            CODE_PREVIEW_CHARS + "\n# ... clipped ...".chars().count()
        );
    }

    #[test]
    fn directory_size_uses_binary_units() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("two-bytes"), b"ab").unwrap();
        assert_eq!(dir_size(dir.path()), "2.0 B");
    }
}
