//! Typed `#[serde(deny_unknown_fields)]` verification of `models/dope-compat.json` (`DV-00`)
//! against `DopeEncoder` constants, `evaluator_identity.v1.json`, `unified-icl-regbin-v2.json`,
//! `Cargo.toml`, and `token_layout.json`.

use dope_core::mlc_v3::{MLC_V3_EXPECTED_PARAMETERS, MLC_V3_EXPECTED_TENSORS};
use starforge::dope_encoder::{
    prepare_table_with_ids, FitScope, Head, PrepTask, TrainingInput, DOPE512_RECIPE,
    DOPE_PACKET_V1_SCHEMA, DOPE_RESEARCH_VIEW_629_SCHEMA, DOPE_ULTRA_PACKET_859_DIM,
    DOPE_ULTRA_PACKET_859_SCHEMA, DOPE_WEIGHTS_SHA256,
};

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct DopeCompatManifest {
    schema: String,
    engine_release_tag: String,
    engine_source_commit: String,
    engine_tag_object: String,
    preparation_source: String,
    checkpoint_identity: String,
    standalone_weights_sha256: String,
    released_tensors: usize,
    released_parameters: usize,
    active_encoder_tensors: usize,
    active_encoder_parameters: usize,
    dope512_recipe: String,
    evaluator_cargo_lock_subgraph_sha256: String,
    evaluator_identity_digest: String,
    fused_serving_composite_sha256: String,
    fused_bundle_dope_safetensors_sha256: String,
    research_schemas: Vec<String>,
    research_859: DopeCompatResearch859,
    contracts: DopeCompatContracts,
    exposure: DopeCompatExposure,
    consumers: DopeCompatConsumers,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct DopeCompatResearch859 {
    schema: String,
    dim: usize,
    opt_in: bool,
    serving_adopted: bool,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct DopeCompatContracts {
    ranker_history: String,
    h256_token_layout: DopeCompatTokenLayout,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct DopeCompatTokenLayout {
    schema: String,
    path: String,
    sha256: String,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct DopeCompatExposure {
    upstream_schema: String,
    dl_val91_exposure_class: String,
    dl_val91_family_overlap: String,
    dl_val91_view_overlap: String,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct DopeCompatConsumers {
    engine_auto: EngineAutoConsumer,
    service_explicit_learned: ServiceExplicitLearnedConsumer,
    appliance_companion: ApplianceCompanionConsumer,
    carbon_cells: CarbonCellsConsumer,
    unified_icl_regbin_v2: UnifiedIclRegbinV2Consumer,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct EngineAutoConsumer {
    dope: String,
    guidance: String,
    uses_dope_weights: bool,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ServiceExplicitLearnedConsumer {
    guidance: String,
    bundles: Vec<String>,
    landmark_mode: String,
    tasks: Vec<String>,
    fused_composite_sha256: String,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ApplianceCompanionConsumer {
    product: String,
    policy_context: String,
    consumer_modes: Vec<String>,
    standalone_weights_sha256: String,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct CarbonCellsConsumer {
    bundle: String,
    free_tier: String,
    pro_enterprise_tier: String,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct UnifiedIclRegbinV2Consumer {
    bundle: String,
    manifest: String,
    tasks: Vec<String>,
    dope_git_rev: String,
    dope_safetensors_sha256: String,
}

#[test]
fn dope_compat_manifest_matches_pinned_v130_and_encoder_constants() {
    let raw = include_str!("../../../../models/dope-compat.json");
    let compat: DopeCompatManifest = serde_json::from_str(raw).unwrap();
    let eval_id: serde_json::Value = serde_json::from_str(include_str!(
        "../../../feat-math/tests/fixtures/compat_freeze/evaluator_identity.v1.json"
    ))
    .unwrap();
    let uicl: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../models/bundles/unified-icl-regbin-v2.json"
    ))
    .unwrap();
    let r882: serde_json::Value =
        serde_json::from_str(include_str!("../../../../models/bundles/r882.json")).unwrap();
    let r882_dope_sha = r882["files"]
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["path"].as_str() == Some("dope.safetensors"))
        .and_then(|f| f["sha256"].as_str())
        .unwrap();
    let tok_bytes = include_bytes!(
        "../../../../research/flywheel_selfplay/h473_ultra_packet_retrieval_and_audits/token_layout.json"
    );

    assert_eq!(compat.schema, "jope.dope-compat/v1");
    assert_eq!(compat.engine_release_tag, "dope-engine-v1.3.0");
    assert_eq!(
        compat.engine_source_commit,
        "cd35d9d0f0a772bea0e1ae2352eb8eb7d34ad09b"
    );
    assert_eq!(
        compat.engine_tag_object,
        "64d83580b520bd39af16168bbffd943647c9a48a"
    );
    assert_eq!(compat.preparation_source, compat.engine_source_commit);
    assert_eq!(compat.checkpoint_identity, "ReleasedV1");
    assert_eq!(compat.standalone_weights_sha256, DOPE_WEIGHTS_SHA256);
    assert_eq!(compat.released_tensors, MLC_V3_EXPECTED_TENSORS);
    assert_eq!(compat.released_parameters, MLC_V3_EXPECTED_PARAMETERS);
    assert_eq!(compat.active_encoder_tensors, 40);
    assert_eq!(compat.active_encoder_parameters, 353_095);
    assert_eq!(compat.dope512_recipe, DOPE512_RECIPE);
    assert_eq!(
        compat.evaluator_cargo_lock_subgraph_sha256,
        eval_id["cargo_lock_sha256"].as_str().unwrap()
    );
    assert_eq!(
        compat.evaluator_identity_digest,
        eval_id["identity_digest"].as_str().unwrap()
    );
    assert_eq!(compat.fused_bundle_dope_safetensors_sha256, r882_dope_sha);
    assert_eq!(
        compat.research_schemas,
        vec![
            DOPE_RESEARCH_VIEW_629_SCHEMA.to_owned(),
            DOPE_PACKET_V1_SCHEMA.to_owned(),
            DOPE_ULTRA_PACKET_859_SCHEMA.to_owned(),
        ]
    );
    assert_eq!(compat.research_859.schema, DOPE_ULTRA_PACKET_859_SCHEMA);
    assert_eq!(compat.research_859.dim, DOPE_ULTRA_PACKET_859_DIM);
    assert!(compat.research_859.opt_in);
    assert!(!compat.research_859.serving_adopted);
    assert_eq!(
        compat.contracts.ranker_history,
        "jope.ranker-history-contract/v1"
    );
    assert_eq!(
        compat.contracts.h256_token_layout.schema,
        "jope.h256-token-layout/v1"
    );
    assert_eq!(
        compat.contracts.h256_token_layout.path,
        "research/flywheel_selfplay/h473_ultra_packet_retrieval_and_audits/token_layout.json"
    );
    assert_eq!(
        compat.contracts.h256_token_layout.sha256,
        jope_runtime::sha256_hex(tok_bytes)
    );
    assert_eq!(
        compat.exposure.upstream_schema,
        "dope.exposure-inventory/v1"
    );
    assert_eq!(compat.exposure.dl_val91_exposure_class, "in_corpus");
    assert_eq!(compat.exposure.dl_val91_family_overlap, "10/19");
    assert_eq!(compat.exposure.dl_val91_view_overlap, "50/91");

    assert_eq!(compat.consumers.engine_auto.dope, "none");
    assert_eq!(compat.consumers.engine_auto.guidance, "heuristic");
    assert!(!compat.consumers.engine_auto.uses_dope_weights);
    assert_eq!(
        compat.consumers.service_explicit_learned.guidance,
        "Guidance::Learned"
    );
    assert_eq!(
        compat.consumers.service_explicit_learned.bundles,
        vec!["r782", "r882"]
    );
    assert_eq!(
        compat.consumers.service_explicit_learned.landmark_mode,
        "full"
    );
    assert_eq!(
        compat.consumers.service_explicit_learned.tasks,
        vec!["regression"]
    );
    assert_eq!(
        compat
            .consumers
            .service_explicit_learned
            .fused_composite_sha256,
        compat.fused_serving_composite_sha256
    );
    assert_eq!(compat.consumers.appliance_companion.product, "Dope512");
    assert_eq!(
        compat.consumers.appliance_companion.policy_context,
        "JopeContext512"
    );
    assert_eq!(
        compat.consumers.appliance_companion.consumer_modes,
        vec!["encoding-only", "jope-dl"]
    );
    assert_eq!(
        compat
            .consumers
            .appliance_companion
            .standalone_weights_sha256,
        DOPE_WEIGHTS_SHA256
    );
    assert_eq!(compat.consumers.carbon_cells.bundle, "r882");
    assert_eq!(compat.consumers.carbon_cells.free_tier, "wave0_seed_only");
    assert_eq!(
        compat.consumers.carbon_cells.pro_enterprise_tier,
        "multi_wave_learned"
    );
    assert_eq!(
        compat.consumers.unified_icl_regbin_v2.bundle,
        "unified-icl-regbin-v2"
    );
    assert_eq!(
        compat.consumers.unified_icl_regbin_v2.manifest,
        "models/bundles/unified-icl-regbin-v2.json"
    );
    assert_eq!(
        compat.consumers.unified_icl_regbin_v2.tasks,
        vec!["regression", "binary"]
    );
    assert_eq!(
        compat.consumers.unified_icl_regbin_v2.dope_git_rev,
        uicl["dope_pin"]["git_rev"].as_str().unwrap()
    );
    assert_eq!(
        compat
            .consumers
            .unified_icl_regbin_v2
            .dope_safetensors_sha256,
        uicl["dope_pin"]["weights_sha256"].as_str().unwrap()
    );

    let x = (0..12).map(|i| vec![i as f64, 1.0]).collect::<Vec<_>>();
    let y = (0..12).map(|i| (2 * i) as f64).collect::<Vec<_>>();
    let ids = (0..12).collect::<Vec<u64>>();
    let prepared = prepare_table_with_ids(&x, &y, PrepTask::Regression, ids.clone()).unwrap();
    let input = TrainingInput::numeric(x, y, ids, Head::Regression);
    let fit_scope = FitScope::from_prepared(&input, &prepared).unwrap();
    assert_eq!(fit_scope.preparation_source, compat.preparation_source);
    assert!(include_str!("../../Cargo.toml").contains(&compat.engine_source_commit));
}
