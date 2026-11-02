use super::*;
#[test]
fn already_completed_gp_selects_bound_parent_without_launching_training() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let corpus = root.join("corpus");
    let frozen = root.join("runs/frozen/checkpoints/step-3000");
    let output = root.join("runs/cycle/auxiliary");
    let mut groups = Vec::new();
    for i in 0..8 {
        groups.push(crate::corpus::Group {
            context: i,
            family: format!("family-{i}"),
            partition: "meta_validation".into(),
            records: (i * 8..i * 8 + 8).collect(),
            max_length: 4,
        });
    }
    let index = Index {
        schema: "parallel.teaching_corpus.v1".into(),
        qualification: true,
        vocabulary_sha256: "vocab".into(),
        context_index_sha256: "context".into(),
        records_sha256: "records".into(),
        record_bytes: 0,
        records: 64,
        legal_sets: vec![],
        contexts: vec![],
        labels_sha256: None,
        groups,
    };
    let panel = training::validation_panel(&index, 8, SEED).unwrap();
    let mut counts = [0usize; 8];
    for &id in panel.iter().flatten() {
        counts[id / 8] += 1;
    }
    assert_eq!(counts, [256; 8]);
    assert_eq!(panel, training::validation_panel(&index, 8, SEED).unwrap());
    let panel_sha = hash(&serde_json::to_vec(&panel).unwrap());
    write_json(&corpus.join("index.json"), &index).unwrap();
    fs::create_dir_all(&frozen).unwrap();
    let mut state = json!({"step":3000,"sampler_id":training::SAMPLER_ID,"corpus_sha256":file_hash(&corpus.join("index.json")).unwrap(),"validation":{"sequence_nll":1.0,"evaluated_step":3000,"panel_sha256":panel_sha,"sampler_id":training::SAMPLER_ID,"family_balanced":true}});
    for (name, key) in [
        ("model.ot", "model_sha256"),
        ("optimizer.ot", "optimizer_sha256"),
        ("optimizer.json", "optimizer_metadata_sha256"),
        ("config.json", "config_sha256"),
    ] {
        fs::write(frozen.join(name), name).unwrap();
        state[key] = file_hash(&frozen.join(name)).unwrap().into();
    }
    write_json(&frozen.join("state.json"), &state).unwrap();
    let selected = Auxiliary::start(
        root,
        &corpus,
        &frozen,
        &output,
        Options::for_cycle(3000),
        false,
    )
    .unwrap()
    .finish()
    .unwrap();
    assert_eq!(selected.checkpoint, frozen.strip_prefix(root).unwrap());
    assert!(!output.join("train/run.json").exists());
    assert_eq!(
        load_selection(root, &output).unwrap().state_sha256,
        selected.state_sha256
    );
    fs::write(frozen.join("model.ot"), b"changed weights").unwrap();
    assert!(load_selection(root, &output).is_err());
}

#[test]
fn plateau_requires_an_epoch_and_counts_only_fixed_panel_checks() {
    let options = Options::for_cycle(3000);
    let mut p = Plateau {
        best: 1.,
        significant: 1.,
        stale: 0,
        minimum_steps: 400,
        options,
    };
    assert!(!p.observe(100, 1.1).unwrap());
    assert!(!p.observe(300, 1.2).unwrap());
    assert_eq!(p.stale, 0);
    assert!(p.observe(600, 0.9995).unwrap());
    assert_eq!(p.stale, 1);
    assert!(p.observe(900, 0.9985).unwrap());
    assert_eq!(p.stale, 0); // cumulative improvement clears the0.1% threshold.
    for step in [1200, 1500, 1800] {
        p.observe(step, 1.).unwrap();
    }
    assert!(p.stopped(1800));
    assert!(p.observe(2100, 0.98).unwrap());
    assert!(!p.stopped(2100));
    assert!(p.observe(2400, f64::NAN).is_err());
}
