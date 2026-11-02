use super::*;
#[test]
fn omitted_overlay_preserves_original_task_serialization() {
    let t = task();
    let b = serde_json::to_value(&t).unwrap();
    assert!(b.get("preparation_overlay").is_none());
    let decoded: GenomeTask = serde_json::from_value(b.clone()).unwrap();
    assert!(decoded.preparation_overlay.is_none());
    assert_eq!(serde_json::to_value(decoded).unwrap(), b);
}
fn task() -> GenomeTask {
    let a = Artifact {
        path: "a.json".into(),
        sha256: "a".repeat(64),
    };
    GenomeTask {
        preparation_overlay: None,
        schema: "jope.native_genome_task.v1".into(),
        stage: GenomeEvaluationStage::Direct,
        candidate: a.clone(),
        output: "out".into(),
        vocabulary: a.clone(),
        admission_policy: a,
        qualification: false,
        evaluations: 1,
        seconds: 30,
        rng_seed: 7,
        direct_parent: None,
    }
}
#[test]
fn continuation_keeps_original_parent_until_measured_descendant_improves_it() {
    let parent = Artifact {
        path: "direct/observations.json".into(),
        sha256: "a".repeat(64),
    };
    let child = Artifact {
        path: "continuation/observations.json".into(),
        sha256: "b".repeat(64),
    };
    let mut rows = vec![
        (
            parent.clone(),
            0,
            json!({"runtime_seconds":2.,"failure":null,"metrics":{"rmse":1.}}),
        ),
        (
            child.clone(),
            0,
            json!({"runtime_seconds":3.,"failure":"timeout","metrics":null}),
        ),
        (
            child.clone(),
            1,
            json!({"runtime_seconds":4.,"failure":null,"metrics":{"rmse":2.}}),
        ),
    ];
    let (best, cost, failures) = summarize_outcomes(&rows).unwrap();
    assert_eq!(best.unwrap().1, parent);
    assert_eq!(cost, 9.);
    assert_eq!(failures, 1);
    rows.push((
        child.clone(),
        2,
        json!({"runtime_seconds":5.,"failure":null,"metrics":{"rmse":0.5}}),
    ));
    let (best, cost, failures) = summarize_outcomes(&rows).unwrap();
    assert_eq!(best.unwrap(), (0.5, child, 2));
    assert_eq!(cost, 14.);
    assert_eq!(failures, 1);
}
#[test]
fn all_failed_outcomes_do_not_invent_a_quality_target() {
    let a = Artifact {
        path: "original.json".into(),
        sha256: "a".repeat(64),
    };
    let rows = vec![(
        a,
        0,
        json!({"runtime_seconds":1.5,"failure":"invalid","metrics":null}),
    )];
    let (best, cost, failed) = summarize_outcomes(&rows).unwrap();
    assert!(best.is_none());
    assert_eq!(cost, 1.5);
    assert_eq!(failed, 1);
}
#[test]
fn exact_direct_and_original_parent_contract_cannot_be_relabelled() {
    let mut t = task();
    assert!(t.validate().is_ok());
    t.evaluations = 2;
    assert!(t.validate().is_err());
    t.stage = GenomeEvaluationStage::Continuation;
    assert!(t.validate().is_err());
    t.direct_parent = Some(t.candidate.clone());
    assert!(t.validate().is_ok());
    t.stage = GenomeEvaluationStage::Direct;
    assert!(t.validate().is_err());
}
#[test]
fn sealed_partition_and_wrong_program_rejected_before_fit() {
    let tmp = tempfile::tempdir().unwrap();
    let mut t = task();
    let c = json!({"schema":"jope.native_candidate.v1","run":"r","generation":0,"arm":"model","matched_slot":0,"context":{"dataset":"d","family":"f","cut":0,"partition":"meta_test","path":"x","sha256":"a".repeat(64),"prepared":"p","prepared_sha256":"a".repeat(64)},"genome":"(pipe X (ridge))","genome_sha256":hash(b"(pipe X (ridge))"),"proposal":t.candidate,"parent_checkpoint":t.candidate,"sampling_seed":1,"origin_capture":t.candidate,"origin_row":0,"generic_draw":null});
    let bytes = serde_json::to_vec(&c).unwrap();
    fs::write(tmp.path().join("a.json"), &bytes).unwrap();
    t.candidate.sha256 = hash(&bytes);
    assert!(t
        .candidate(tmp.path())
        .err()
        .unwrap()
        .to_string()
        .contains("sealed partition"));
}
fn preparation(clean: bool) -> Vec<u8> {
    serde_json::to_vec(&json!({"schema":"parallel.prepared.v1","dataset":"d","family":"f",
            "task":"regression","meta_partition":"meta_train","seed":1,"family_partition_frozen":true,
            "clean_evaluation":clean,"clean_status":"fixture","original_train_sha256":"0".repeat(64),
            "original_test_sha256":"1".repeat(64),"selected_train_indices":[0],
            "selected_test_indices":[],"target_last":true,"rows_total":1,"cuts":[]}))
        .unwrap()
}
/// A direct task whose candidate binds `prepared` by digest. Its
/// vocabulary is deliberately unbound, so a candidate that passes the
/// scientific boundary stops at the vocabulary.
fn task_binding(root: &Path, prepared: &[u8]) -> GenomeTask {
    let mut t = task();
    fs::create_dir_all(root.join("p")).unwrap();
    fs::write(root.join("p/prepared.json"), prepared).unwrap();
    let c = json!({"schema":"jope.native_candidate.v1","run":"r","generation":0,"arm":"model","matched_slot":0,"context":{"dataset":"d","family":"f","cut":0,"partition":"meta_train","path":"x","sha256":"a".repeat(64),"prepared":"p","prepared_sha256":hash(prepared)},"genome":"(pipe X (ridge))","genome_sha256":hash(b"(pipe X (ridge))"),"proposal":t.candidate,"parent_checkpoint":t.candidate,"sampling_seed":1,"origin_capture":t.candidate,"origin_row":0,"generic_draw":null});
    let bytes = serde_json::to_vec(&c).unwrap();
    fs::write(root.join("a.json"), &bytes).unwrap();
    t.candidate.sha256 = hash(&bytes);
    t
}
/// Replace `target` with `bytes` the first time a receiver reports it verified.
fn replace_on_verification(
    target: &Path,
    bytes: Vec<u8>,
) -> crate::manifest::verification_hook::Guard {
    let target = fs::canonicalize(target).unwrap();
    let mut done = false;
    crate::manifest::verification_hook::install(move |path| {
        if !done && fs::canonicalize(path).ok().as_deref() == Some(target.as_path()) {
            done = true;
            let staged = target.with_extension("replacement");
            fs::write(&staged, &bytes).unwrap();
            fs::rename(&staged, &target).unwrap();
        }
    })
}
fn refusal(t: &GenomeTask, root: &Path) -> String {
    match t.candidate(root) {
        Ok(_) => "accepted".into(),
        Err(e) => format!("{e:#}"),
    }
}
/// W1 route R8 (native_genome.rs:195-200 before the fix): the candidate's
/// scientific boundary is decided on exactly the preparation bytes whose
/// digest the candidate binds.
#[test]
fn the_candidate_boundary_is_decided_on_the_verified_preparation() {
    const BOUNDARY: &str = "candidate scientific boundary changed";
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let prepared = root.join("p/prepared.json");
    // A bound preparation that is not a clean evaluation is refused, also
    // when a clean preparation replaces it after its digest check.
    let t = task_binding(root, &preparation(false));
    assert!(refusal(&t, root).contains(BOUNDARY));
    {
        let _hook = replace_on_verification(&prepared, preparation(true));
        let refused = refusal(&t, root);
        assert!(
            refused.contains(BOUNDARY),
            "the boundary was decided on a preparation other than the verified one: {refused}"
        );
    }
    // Positive control: a bound clean preparation passes the boundary (and
    // stops at the unbound vocabulary), also when replaced after its check.
    let t = task_binding(root, &preparation(true));
    let stopped = refusal(&t, root);
    assert!(
        !stopped.contains(BOUNDARY) && stopped.contains("native genome artifact"),
        "{stopped}"
    );
    let _hook = replace_on_verification(&prepared, preparation(false));
    let stopped = refusal(&t, root);
    assert!(
        !stopped.contains(BOUNDARY),
        "the boundary was decided on a preparation other than the verified one: {stopped}"
    );
}
fn referencing(root: &Path, target: &str) -> Vec<u8> {
    let digest = hash(&fs::read(root.join(target)).unwrap());
    serde_json::to_vec(&json!({"input":{"path":target,"sha256":digest}})).unwrap()
}
/// W1 route R8 (required_inputs :484-490 before the fix): the input
/// closure follows the nested references of exactly the verified bytes.
#[test]
fn the_input_closure_follows_only_the_verified_bytes() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().canonicalize().unwrap();
    fs::write(root.join("x.txt"), b"bound dependency").unwrap();
    fs::write(root.join("y.txt"), b"other dependency").unwrap();
    let bound = referencing(&root, "x.txt");
    let other = referencing(&root, "y.txt");
    fs::write(root.join("a.json"), &bound).unwrap();
    let a = Artifact {
        path: "a.json".into(),
        sha256: hash(&bound),
    };
    let follows = |c: &BTreeMap<PathBuf, String>| {
        (
            c.contains_key(Path::new("x.txt")),
            c.contains_key(Path::new("y.txt")),
        )
    };
    // Positive control.
    let closure = artifact_closure(&root, vec![a.clone()]).unwrap();
    assert_eq!(follows(&closure), (true, false));
    // Negative control: other bytes than the bound ones are refused.
    fs::write(root.join("a.json"), &other).unwrap();
    let refused = format!(
        "{:#}",
        artifact_closure(&root, vec![a.clone()]).unwrap_err()
    );
    assert!(refused.contains("native genome input changed"), "{refused}");
    fs::write(root.join("a.json"), &bound).unwrap();
    // The input is replaced right after its digest check.
    let _hook = replace_on_verification(&root.join("a.json"), other);
    let raced = artifact_closure(&root, vec![a]).unwrap();
    assert_eq!(
        follows(&raced),
        (true, false),
        "the closure followed references of bytes other than the verified ones: {raced:?}"
    );
}
/// W1 route R8 (required_inputs :493-516 before the fix): the closure
/// records the preparation the candidate bound, and its original
/// observation source is read from exactly those bytes.
#[test]
fn the_prepared_closure_is_the_preparation_the_candidate_bound() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().canonicalize().unwrap();
    fs::create_dir_all(root.join("p")).unwrap();
    fs::write(root.join("s1.csv"), b"x,y\n1,2\n").unwrap();
    fs::write(root.join("s2.csv"), b"x,y\n3,4\n").unwrap();
    let with_source = |source: &str| {
        let mut v: Value = serde_json::from_slice(&preparation(true)).unwrap();
        v["storage"] = json!({"source":source,
                "source_sha256":hash(&fs::read(root.join(source)).unwrap()),
                "feature_columns":["x"],"target_column":"y","original_training_row_map":[0]});
        serde_json::to_vec(&v).unwrap()
    };
    let bound = with_source("s1.csv");
    let other = with_source("s2.csv");
    let prepared = root.join("p/prepared.json");
    fs::write(&prepared, &bound).unwrap();
    let context: ContextRow = serde_json::from_value(json!({"dataset":"d","family":"f",
            "cut":0,"partition":"meta_train","path":"x","sha256":"a".repeat(64),"prepared":"p",
            "prepared_sha256":hash(&bound)}))
    .unwrap();
    let closure = || {
        let mut out = BTreeMap::new();
        prepared_closure(&root, &context, &mut out).map(|()| out)
    };
    // Positive control.
    let out = closure().unwrap();
    assert_eq!(out[Path::new("p/prepared.json")], hash(&bound));
    assert!(out.contains_key(Path::new("s1.csv")) && !out.contains_key(Path::new("s2.csv")));
    // The candidate's check (`GenomeTask::candidate`) passes, and the
    // preparation is replaced right after it.
    let _hook = replace_on_verification(&prepared, other);
    read_verified(&prepared, &context.prepared_sha256, "candidate preparation").unwrap();
    let raced = closure().map_err(|e| format!("{e:#}"));
    assert!(
        matches!(&raced, Err(e) if e.contains("native genome input changed")),
        "the closure recorded a preparation other than the one the candidate bound: {raced:?}"
    );
}
#[test]
fn large_corpus_metadata_is_explicitly_typed_and_hash_bound() {
    let tmp = tempfile::tempdir().unwrap();
    let bytes = serde_json::to_vec(&json!({"schema":"parallel.teaching_shards.v1","shards":[],"retained_metadata":"x".repeat(17 * 1024 * 1024)})).unwrap();
    fs::write(tmp.path().join("index.json"), &bytes).unwrap();
    let mut artifact = Artifact {
        path: "index.json".into(),
        sha256: hash(&bytes),
    };
    assert!(load::<Value>(tmp.path(), &artifact).is_err());
    assert_eq!(
        load_corpus_index(tmp.path(), &artifact).unwrap()["schema"],
        "parallel.teaching_shards.v1"
    );
    assert!(load_dataset_receipt(tmp.path(), &artifact).is_err());
    artifact.sha256 = "0".repeat(64);
    assert!(load_corpus_index(tmp.path(), &artifact).is_err());
}
#[test]
fn plain_seeded_continuation_admits_only_measured_children() {
    assert_eq!(CONTINUATION_PROTOCOL, "plain_seeded_continuation_v1");
    assert_eq!(MINIMUM_COMPLETED_CHILDREN, 1);
    assert_eq!(completed_children(&[]).unwrap(), 0);
    assert_eq!(training_admission(0), REFUSED_NO_COMPLETED_CHILD);
    let failed = vec![
        json!({"runtime_seconds":30.,"failure":"candidate_or_view_timeout","metrics":null}),
        json!({"runtime_seconds":1.,"failure":"invalid","metrics":null}),
    ];
    assert_eq!(completed_children(&failed).unwrap(), 0);
    assert_eq!(training_admission(0), REFUSED_NO_COMPLETED_CHILD);
    let mut measured = failed.clone();
    measured.push(json!({"runtime_seconds":2.,"failure":null,"metrics":{"rmse":0.25}}));
    assert_eq!(completed_children(&measured).unwrap(), 1);
    assert_eq!(training_admission(1), TRAINING_ADMITTED);
    let absent = json!({"runtime_seconds":2.,"failure":null,"metrics":{"rmse":null}});
    assert!(completed_children(&[absent]).is_err());
    let negative = json!({"runtime_seconds":2.,"failure":null,"metrics":{"rmse":-1.}});
    assert!(completed_children(&[negative]).is_err());
}
