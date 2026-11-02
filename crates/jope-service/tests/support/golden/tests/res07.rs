#[test]
fn res07_fixture_inventory_loads_every_safetensors_shape() {
    let explicit_root = env::var_os(GOLDEN_DIR_ENV);
    // The default inventory is generated and fetched from custody, never committed, so the
    // general parity-strict flag must not require it; an explicit root or this flag does.
    let strict = env::var("JOPE_GOLDEN_STRICT").is_ok_and(|value| value == "1");
    let root = explicit_root.as_ref().map_or_else(
        || Path::new(env!("CARGO_MANIFEST_DIR")).join("../../research/flywheel_selfplay/golden"),
        std::path::PathBuf::from,
    );
    let index_path = root.join("index.json");
    let index_bytes = match fs::read(&index_path) {
        Ok(bytes) => bytes,
        Err(error)
            if error.kind() == std::io::ErrorKind::NotFound
                && explicit_root.is_none()
                && !strict =>
        {
            eprintln!(
                    "skipping RES-07 fixture inventory smoke: default inventory is absent at {}; set {GOLDEN_DIR_ENV} to run it",
                    index_path.display()
                );
            return;
        }
        Err(error) => panic!(
            "RES-07 fixture inventory is unavailable at {}: {error}",
            index_path.display()
        ),
    };
    let index: serde_json::Value =
        serde_json::from_slice(&index_bytes).expect("RES-07 index.json must be valid JSON");
    assert_eq!(index["schema"], "jope.golden-fixtures/v1");

    let files = index["files"]
        .as_object()
        .expect("RES-07 index.json must contain a files object");
    let fixtures = files
        .iter()
        .filter(|(name, _)| name.ends_with(".safetensors"))
        .collect::<Vec<_>>();
    assert!(
        !fixtures.is_empty(),
        "RES-07 index lists no SafeTensors fixtures"
    );

    for (name, metadata) in fixtures {
        let tensor_metadata = metadata["tensors"]
            .as_object()
            .unwrap_or_else(|| panic!("RES-07 inventory entry `{name}` has no tensor schema"));
        let schema = tensor_metadata
            .iter()
            .map(|(tensor, spec)| {
                let shape = spec["shape"]
                    .as_array()
                    .unwrap_or_else(|| panic!("RES-07 tensor `{tensor}` in `{name}` has no shape"))
                    .iter()
                    .map(|dimension| {
                        usize::try_from(
                            dimension
                                .as_u64()
                                .expect("tensor dimensions must be unsigned"),
                        )
                        .expect("tensor dimension must fit usize")
                    })
                    .collect::<Vec<_>>();
                TensorSpec::new(tensor, shape)
            })
            .collect::<Vec<_>>();
        let fixture_name = name
            .strip_suffix(".safetensors")
            .expect("filtered SafeTensors name must have its extension");
        let fixture = load_named(fixture_name, &schema)
            .unwrap_or_else(|error| panic!("RES-07 fixture `{name}` failed validation: {error}"))
            .unwrap_or_else(|| panic!("RES-07 fixture `{name}` was skipped"));

        for (tensor, spec) in schema.iter().map(|spec| (&spec.name, &spec.shape)) {
            assert_eq!(
                fixture.tensors[tensor].dims(),
                spec,
                "shape mismatch in `{name}` tensor `{tensor}`"
            );
        }
    }
}

#[test]
fn rnk07_context_schema_accepts_complete_res07_intermediates() {
    // Independent fixture contract from golden/reference.py::intermediates.
    // These zeros exercise file/schema mechanics, not numerical parity.
    let tensors = vec![
        ("v256", vec![1, 256]),
        ("col", vec![1, 64, 17]),
        ("dope629", vec![1, 629]),
        ("lm48", vec![1, 48]),
        ("lm48_z", vec![1, 48]),
        ("tokens", vec![1, 10, 32]),
        ("mask", vec![1, 10]),
    ];
    let directory = tempdir().unwrap();
    let write = |name: &str, tensors: &[(&str, Vec<usize>)]| {
        let buffers = tensors
            .iter()
            .map(|(_, shape)| vec![0_u8; shape.iter().product::<usize>() * 4])
            .collect::<Vec<_>>();
        let views = tensors
            .iter()
            .zip(&buffers)
            .map(|((name, shape), bytes)| {
                (
                    *name,
                    TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        fs::write(
            directory.path().join(format!("{name}.safetensors")),
            serialize(views, None).unwrap(),
        )
        .unwrap();
    };
    write("context", &tensors);
    let schema = rnk07_output_schema();
    let fixture = load_from_dir(directory.path(), "context", &schema, true)
        .unwrap()
        .unwrap();
    assert_eq!(fixture.tensors.len(), 7);
    assert_eq!(fixture.values_f32["lm48_z"].len(), 48);

    let mut missing = tensors.clone();
    missing.retain(|(name, _)| *name != "lm48_z");
    write("missing", &missing);
    assert!(load_from_dir(directory.path(), "missing", &schema, true).is_err());

    let mut extra = tensors.clone();
    extra.push(("unexpected", vec![1]));
    write("extra", &extra);
    assert!(load_from_dir(directory.path(), "extra", &schema, true)
        .unwrap_err()
        .to_string()
        .contains("tensor names"));

    let mut wrong_shape = tensors;
    wrong_shape
        .iter_mut()
        .find(|(name, _)| *name == "mask")
        .unwrap()
        .1 = vec![1, 9];
    write("wrong-shape", &wrong_shape);
    assert!(
        load_from_dir(directory.path(), "wrong-shape", &schema, true)
            .unwrap_err()
            .to_string()
            .contains("expected [1, 10]")
    );
}

fn rnk07_output_schema() -> [TensorSpec; 7] {
    // RES-07 context files carry every intermediate, not only Stage-1b outputs.
    [
        TensorSpec::new("v256", vec![1, 256]),
        TensorSpec::new("col", vec![1, 64, 17]),
        TensorSpec::new("dope629", vec![1, 629]),
        TensorSpec::new("lm48", vec![1, 48]),
        TensorSpec::new("lm48_z", vec![1, 48]),
        TensorSpec::new("tokens", vec![1, 10, 32]),
        TensorSpec::new("mask", vec![1, 10]),
    ]
}

#[test]
fn rnk07_loader_seam_rejects_invalid_fixture_paths() {
    let error = load_named("../outside", &rnk07_output_schema()).unwrap_err();
    assert!(error.to_string().contains("invalid golden fixture name"));
}

#[test]
#[ignore = "RES-07 fixtures not qualified; enable only after qualified tensors are published"]
fn rnk07_fixture_schema_smoke_after_res07_qualification() {
    let context =
        env::var("JOPE_RNK07_GOLDEN_CONTEXT").expect("select a qualified RES-07 training context");
    let fixture_name = format!("contexts/{context}");
    let fixture = load_named(&fixture_name, &rnk07_output_schema())
        .expect("qualified RES-07 fixture must satisfy the documented output schema")
        .expect("fixture-dependent check remains ignored until RES-07 qualification");

    assert_eq!(fixture.tensors["tokens"].dims(), [1, 10, 32]);
    assert_eq!(fixture.tensors["mask"].dims(), [1, 10]);
}

#[test]
fn rnk07_stage1b_module_is_available_for_synthetic_tests() {
    assert_eq!(stage1b::TOKEN_COUNT, 10);
    assert_eq!(stage1b::TOKEN_WIDTH, 32);
}
