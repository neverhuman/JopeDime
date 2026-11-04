use super::*;

struct Custody {
    _temp: tempfile::TempDir,
    study: PathBuf,
    repository: PathBuf,
}

fn write(path: &Path, bytes: &[u8]) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, bytes).unwrap();
}

fn write_json(path: &Path, value: &Value) {
    write(path, &serde_json::to_vec_pretty(value).unwrap());
}

fn custody() -> Custody {
    let temp = tempfile::tempdir().unwrap();
    let study = temp.path().join("study-final");
    let repository = temp.path().join("repository");
    let source = "d".repeat(40);
    let object = |name: &str, content: &str| -> Value {
        let path = study.join(name);
        write(&path, content.as_bytes());
        json!({"path":path,"sha256":sha256_hex(content.as_bytes())})
    };
    let raw_a = object("raw-a.json", "{\"rows\":[1]}");
    let raw_b = object("raw-b.json", "{\"rows\":[2]}");
    let folds: Vec<Value> = (0..2)
        .map(|k| {
            json!([
                object(&format!("gp-a-fold-{k}.json"), &format!("fold {k}")),
                ["1"]
            ])
        })
        .collect();
    write_json(
        &study.join("retained-fitting-inputs.json"),
        &json!({"schema":FITTING_INPUTS_SCHEMA,"panel_sha256":digest("panel"),
            "configuration_sha256":digest("configuration"),"projection_source_sha":source,"seed":0,
            "assignments":{"a":{"path":"/splits/a.json","sha256":digest("split a")},
                "b":{"path":"/splits/b.json","sha256":digest("split b")}},
            "inputs":[{"id":"a","raw":raw_a,"gp":{"folds":folds}},{"id":"b","raw":raw_b,"gp":null}]}),
    );
    write_json(
        &study.join("plan.json"),
        &json!({"schema":HISTORICAL_PLAN_SCHEMA}),
    );
    write_json(&study.join("plan-command.json"), &json!(["jope", "plan"]));
    let deviation = json!({"schema":"jope.core-ip-protocol-deviation/v1","stage":"pre-window input staging",
        "source_sha":source,"finding":"outer labels parsed before projection",
        "outer_scoring_labels_parsed":true,"sealed_target_tables_opened":false,
        "outer_scoring_executed":false,"campaign_held_looks_consumed":false,
        "worker_payload_contains_outer_targets":false,"original_result_reclassified":false});
    write_json(&study.join("protocol-deviation.json"), &deviation);
    write(&study.join("run.stderr"), b"exit-code 1\n");
    let window = json!({"experiment_sha256":digest("window"),"started_unix_ms":1000,"deadline_unix_ms":2000});
    write_json(&study.join("journal/window.json"), &window);
    write_json(
        &study.join("journal/launch.json"),
        &json!({"exit_code":1,"original_window":window}),
    );
    write(&study.join("jope"), b"binary");
    let closeout = json!({"source_sha":source,"window":window,
        "status":"controller_failed_before_cuda_qualification","failure":"collision",
        "gpu_qualification_executed":false,"direct_requests":0,"full_gp_arms":0,"prediction_batches":0,
        "files":{study.join("plan.json").display().to_string():sha256_hex(&fs::read(study.join("plan.json")).unwrap()),
            study.join("run.stderr").display().to_string():sha256_hex(b"exit-code 1\n")}});
    write_json(&study.join("closeout.json"), &closeout);
    let data = repository.join("docs/internal/core-ip/data");
    write_json(&data.join("gpu-protocol-deviation.json"), &deviation);
    write_json(&data.join("gpu-closeout.json"), &closeout);
    write_json(
        &data.join("gpu-study-manifest.json"),
        &json!({"source_sha":source,"binary":{"sha256":sha256_hex(b"binary")},
            "plan":{"sha256":sha256_hex(&fs::read(study.join("plan.json")).unwrap())}}),
    );
    write_json(&data.join("gpu-plan-command.json"), &json!(["jope"]));
    write_json(
        &data.join("planner-local-proof.json"),
        &json!({"positive_plan_exit":0,"execution_started":false}),
    );
    Custody {
        _temp: temp,
        study,
        repository,
    }
}

#[test]
fn derivation_cross_checks_the_retained_evidence_and_is_deterministic() {
    let c = custody();
    let d = derive_disposition(&c.study, &c.repository).unwrap();
    let bytes = d.encode().unwrap();
    assert_eq!(
        derive_disposition(&c.study, &c.repository)
            .unwrap()
            .encode()
            .unwrap(),
        bytes
    );
    assert_eq!(HistoricalDisposition::open(&bytes).unwrap(), d);
    assert_eq!(d.input_closure.members.len(), 2);
    assert_eq!(d.input_closure.members[0].gp_fold_sha256.len(), 2);
    // manifest, 2 raw inputs, 2 folds, v2 plan, v3 local proof.
    assert_eq!(d.affected_descendants.len(), 7);
    // The corrected protocol rejects exactly this retained closure by replay.
    let p = CorrectedProtocol::derive(&d).unwrap();
    let retained = fs::read(c.study.join("retained-fitting-inputs.json")).unwrap();
    rejected(verify_closure(&p, &retained), "historical input closure");

    type Break = fn(&Custody);
    let breaks: [(&str, Break, &str); 6] = [
        (
            "missing controller stderr",
            |c| fs::remove_file(c.study.join("run.stderr")).unwrap(),
            "absent",
        ),
        (
            "changed retained projected input",
            |c| fs::write(c.study.join("raw-a.json"), b"{\"rows\":[9]}").unwrap(),
            "differs from the manifest",
        ),
        (
            "second staging source",
            |c| {
                let path = c.study.join("closeout.json");
                let mut v: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
                v["source_sha"] = json!("e".repeat(40));
                write_json(&path, &v);
            },
            "more than one staging source",
        ),
        (
            "second window",
            |c| {
                let path = c.study.join("journal/launch.json");
                let mut v: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
                v["original_window"]["started_unix_ms"] = json!(999);
                write_json(&path, &v);
            },
            "more than one execution window",
        ),
        (
            "published deviation differs",
            |c| {
                let path = c
                    .repository
                    .join("docs/internal/core-ip/data/gpu-protocol-deviation.json");
                let mut v: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
                v["outer_scoring_labels_parsed"] = json!(false);
                write_json(&path, &v);
            },
            "differ",
        ),
        (
            "symlinked plan",
            |c| {
                let plan = c.study.join("plan.json");
                let moved = c.study.parent().unwrap().join("plan-elsewhere.json");
                fs::rename(&plan, &moved).unwrap();
                std::os::unix::fs::symlink(&moved, &plan).unwrap();
            },
            "symlink",
        ),
    ];
    for (name, damage, needle) in breaks {
        let c = custody();
        damage(&c);
        let error = format!(
            "{:#}",
            derive_disposition(&c.study, &c.repository).expect_err(name)
        );
        assert!(
            error.contains(needle),
            "{name}: expected {needle:?}: {error}"
        );
    }
}

/// W1 follow-up 2d: consumers open the embedded pair, which is exactly the
/// committed release; the digest-level refusal covers every staging product
/// and nothing else.
#[test]
fn the_embedded_pair_is_the_committed_release_and_refuses_only_staging_products() {
    let (d, p) = committed();
    let (embedded_d, embedded_p) = jope_datasets::staging_disposition::committed().unwrap();
    assert_eq!((&embedded_d, &embedded_p), (&d, &p));
    assert_eq!(embedded_p.encode().unwrap(), PROTOCOL);
    let products = d.staging_products();
    for product in &products {
        let error = refuse_historical_objects(&p, [("object".to_string(), product.as_str())])
            .unwrap_err()
            .to_string();
        assert!(error.contains("historical staging path"), "{error}");
    }
    // Label-free split memberships are panel inputs, not staging products.
    let memberships = d
        .input_closure
        .members
        .iter()
        .map(|m| ("membership".to_string(), m.assignment_sha256.as_str()));
    assert_eq!(
        refuse_historical_objects(&p, memberships).unwrap(),
        d.input_closure.members.len()
    );
    let fresh = fresh(3, "2d");
    assert_eq!(
        refuse_historical_objects(&p, fresh.iter().map(|f| ("fresh".to_string(), f.as_str())))
            .unwrap(),
        3
    );
    assert_eq!(products.len(), p.forbidden_closure_sha256.len());
}
