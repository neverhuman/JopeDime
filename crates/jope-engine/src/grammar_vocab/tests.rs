use super::*;

#[test]
fn learned_bins_union_respects_expression_skip_and_cap() {
    let spec = json!({"engine":"featmath","transforms":{"scale":{"params":[
            {"name":"mode","kind":"str","bins":["rank"]},
            {"name":"expr","kind":"expr","bins":["x"]}
        ]}},"heads":{}});
    let observed = json!({"scale":{"mode":["rank","robust","clip"],"expr":["y"]}});
    let merged = merge_learned_bins(&spec, &observed, 2);
    assert_eq!(
        merged["transforms"]["scale"]["params"][0]["bins"],
        json!(["rank", "robust"])
    );
    assert_eq!(
        merged["transforms"]["scale"]["params"][1]["bins"],
        json!(["x"])
    );
}

#[test]
fn observed_tokens_filter_engine_and_invalid_genomes_and_rank_by_frequency() {
    let spec = json!({"engine":"featmath","max_transforms":2,
            "transforms":{"scale":{"params":[{"name":"mode","kind":"str"}]}},
            "heads":{"ridge":{"params":[]}}});
    let rows = vec![
        json!({"engine":"featmath","genome":"(pipe X (scale mode=robust) (ridge))"}),
        json!({"engine":"featmath","genome":"(pipe X (scale mode=rank) (ridge))"}),
        json!({"engine":"featmath","genome":"(pipe X (scale mode=robust) (ridge))"}),
        json!({"engine":"newmath","genome":"(pipe X (scale mode=wrong) (ridge))"}),
        json!({"engine":"featmath","genome":"not a pipeline"}),
    ];
    assert_eq!(
        observed_param_tokens(&rows, &spec, 2),
        json!({"scale":{"mode":["robust","rank"]}})
    );
}

#[test]
fn artifact_contains_family_list_and_engine_snapshots() {
    let spec = json!({"engine":"featmath","transforms":{},"heads":{}});
    let artifact = build_active_vocab_v3(&[], std::slice::from_ref(&spec));
    assert_eq!(artifact["syntax"], "active_multi_engine_v3");
    assert_eq!(artifact["families"].as_array().unwrap().len(), 21);
    assert_eq!(artifact["engines"]["featmath"], spec);
}

#[test]
fn legacy_snapshots_restore_source_tasks_and_generation_flags() {
    let vocab =
        json!({"engines":{"featmath":{"engine":"featmath","transforms":{"gpr":{}},"heads":{}}}});
    let sources = [
        json!({"engine":"featmath","transforms":{"gpr":{"tasks":["regression"],"generate":false}},"heads":{}}),
    ];
    let restored = grammars_from_vocab_v3(&vocab, &sources);
    assert_eq!(
        restored["featmath"]["transforms"]["gpr"]["tasks"],
        json!(["regression"])
    );
    assert_eq!(restored["featmath"]["transforms"]["gpr"]["generate"], false);
}
