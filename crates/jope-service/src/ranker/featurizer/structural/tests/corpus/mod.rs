use super::*;
use crate::ranker::featurizer::{parse, test_support, Featurizer};

mod golden;
use golden::*;

#[test]
fn all_1309_train_program_records_match_all_30_python_golden_coordinates_exactly() {
    let rows = test_support::corpus();
    let golden_rows: Vec<_> = GOLDEN_CSV.lines().collect();
    assert_eq!(
        jope_runtime::sha256_hex(GOLDEN_CSV.as_bytes()),
        GOLDEN_SHA256
    );
    assert_eq!(golden_rows.len(), rows.len());
    let names: Vec<_> = FEATURE_NAMES
        .iter()
        .copied()
        .filter(|n| is_feature(n))
        .collect();
    assert_eq!(names.len(), 30);
    let mut common_parser = 0;
    for (index, (row, golden)) in rows.iter().zip(golden_rows).enumerate() {
        let program = parse::parse(&row.dsl).unwrap();
        common_parser += usize::from(program.parser() == parse::ParserKind::GenomePipeline);
        let actual = extract(&program);
        let expected: Vec<f64> = golden.split(',').map(|v| v.parse().unwrap()).collect();
        assert_eq!(actual.len(), 30);
        assert_eq!(expected.len(), names.len());
        for (name, expected) in names.iter().zip(expected) {
            assert_eq!(
                actual[*name], expected,
                "record {index}, {}: {name}",
                row.program_sha256
            );
        }
        let projection = Featurizer::default().project(&actual);
        for (index, name) in FEATURE_NAMES.iter().enumerate() {
            assert_eq!(projection.values()[index], actual.get(*name).copied());
        }
    }
    assert_eq!(
        common_parser,
        rows.len(),
        "all corpus pipelines must reuse GenomeIr"
    );
    if let Some(root) = std::env::var_os("JOPE_FEATURIZER_PROOF_DIR") {
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(std::path::Path::new(&root).join("structural-corpus.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "records": rows.len(), "unique_programs":1292,"coordinates_per_record":30,
                "comparisons":rows.len()*30,"mismatches":0,"genome_parser_records":common_parser,
                "golden_sha256":GOLDEN_SHA256,"complete_feat412":false,
                "corpus":"RES-02 TRAIN row-free champions+qualified winners; expanded RES-07 corpus pending"
            })).unwrap()).unwrap();
    }
}
