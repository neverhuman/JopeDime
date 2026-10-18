use jope_pipeline::{
    search,
    seeds::{
        self, SeedAncestryV2, SeedDrawV2, SeedGenome, SeedSetV2, SeedSourceV2, SeedsFile,
        SEEDS_SCHEMA_ID, SEEDS_V2_SCHEMA_ID,
    },
};
use std::collections::BTreeMap;

fn draw(id: &str, genome: &str, source: SeedSourceV2) -> SeedDrawV2 {
    SeedDrawV2 {
        draw_id: id.into(),
        genome: genome.into(),
        program_sha256: jope_pipeline::sha256_hex(genome.as_bytes()),
        source,
        ancestry: Some(SeedAncestryV2 {
            parent_families: vec!["parent-1".into()],
            dataset_families: vec!["dataset-1".into()],
            canonical_program_sha256: vec![jope_pipeline::sha256_hex(genome.as_bytes())],
        }),
    }
}
fn set(draws: Vec<SeedDrawV2>, cap: usize) -> SeedSetV2 {
    SeedSetV2 {
        schema: SEEDS_V2_SCHEMA_ID.into(),
        source_snapshot_sha256: "a".repeat(64),
        cap,
        requested_by_source: BTreeMap::from([(SeedSourceV2::OtherLineageRetrieval, draws.len())]),
        draws,
    }
}

#[test]
fn legacy_v1_cap_remains_four_and_schema_identity_is_unchanged() {
    assert_eq!(SEEDS_SCHEMA_ID, "jope.pipeline-seeds/v1");
    let genomes = ["standardize", "robust_scale", "rank", "blom", "quantile"];
    let offered: Vec<_> = genomes
        .iter()
        .map(|stage| {
            (
                format!("(pipe X ({stage}) (ridge))"),
                seeds::FROM_SEEDS_FILE,
            )
        })
        .collect();
    assert_eq!(seeds::admit(&offered).unwrap().seeds.len(), search::ELITES);
    let legacy = SeedsFile::new(
        None,
        vec![SeedGenome {
            genome: "(pipe X (standardize) (ridge))".into(),
            program_sha256: jope_pipeline::sha256_hex(b"(pipe X (standardize) (ridge))"),
            bank_entry_sha256: None,
            receipt_sha256: None,
        }],
    );
    assert_eq!(legacy.schema, SEEDS_SCHEMA_ID);
}

#[test]
fn v2_cap_is_explicit_and_source_denominators_survive_dedup() {
    let a = "(pipe X (standardize) (ridge))";
    let b = "(pipe X (robust_scale) (ridge))";
    let set = set(
        vec![
            draw("d1", a, SeedSourceV2::OtherLineageRetrieval),
            draw("d2", a, SeedSourceV2::NeuralProposal),
            draw("d3", b, SeedSourceV2::NeuralProposal),
        ],
        1,
    );
    let result = seeds::admit_v2(&set).unwrap();
    assert_eq!(
        result.offered_by_source[&SeedSourceV2::OtherLineageRetrieval],
        1
    );
    assert_eq!(result.offered_by_source[&SeedSourceV2::NeuralProposal], 2);
    assert_eq!(
        result.realized_by_source[&SeedSourceV2::OtherLineageRetrieval],
        1
    );
    assert_eq!(result.rejected_duplicates, 1);
    assert_eq!(result.surplus, 1);
    assert_eq!(result.seeds.len(), 1);
    assert_eq!(
        result.provenance[0].source,
        SeedSourceV2::OtherLineageRetrieval
    );
    assert_eq!(result.provenance[0].draw_id, "d1");
}

#[test]
fn v2_refuses_missing_explicit_cap_bad_digest_and_duplicate_draw_identity() {
    let mut value = set(
        vec![draw(
            "same",
            "(pipe X (standardize) (ridge))",
            SeedSourceV2::OtherLineageRetrieval,
        )],
        2,
    );
    value.cap = 0;
    assert!(value.check().is_err());
    value.cap = 2;
    value.draws[0].program_sha256 = "0".repeat(64);
    assert!(value.check().is_err());
    value.draws[0].program_sha256 = jope_pipeline::sha256_hex(value.draws[0].genome.as_bytes());
    value.draws.push(value.draws[0].clone());
    assert!(value.check().is_err());
}

#[test]
fn transfer_draws_require_complete_ancestry_and_a_declared_source_mixture() {
    let mut value = set(
        vec![draw(
            "transfer",
            "(pipe X (standardize) (ridge))",
            SeedSourceV2::OtherLineageRetrieval,
        )],
        2,
    );
    value.draws[0].ancestry = None;
    assert!(value.check().is_err());
    value.draws[0].ancestry = Some(SeedAncestryV2::default());
    assert!(value.check().is_err());
    value.requested_by_source.clear();
    assert!(value.check().is_err());
}

#[test]
fn v2_json_round_trip_preserves_origin_and_ancestry() {
    let mut seed = draw(
        "retrieval-draw",
        "(pipe X (standardize) (ridge))",
        SeedSourceV2::OtherLineageRetrieval,
    );
    seed.ancestry = Some(SeedAncestryV2 {
        parent_families: vec!["family-2".into()],
        dataset_families: vec!["dataset-family-2".into()],
        canonical_program_sha256: vec![seed.program_sha256.clone()],
    });
    let value = set(vec![seed], 7);
    assert_eq!(
        serde_json::from_slice::<SeedSetV2>(&serde_json::to_vec(&value).unwrap()).unwrap(),
        value
    );
}
