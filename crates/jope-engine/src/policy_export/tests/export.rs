use super::*;

#[test]
fn export_is_bit_exact_and_binds_the_state() {
    let scratch = scratch();
    let config = tiny();
    let store = policy_store(&config);
    // Values a lossy writer would change: signed zero, the smallest subnormal, the smallest
    // normal and the largest finite float.
    let special = [-0.0f32, f32::from_bits(1), f32::MIN_POSITIVE, f32::MAX];
    let mut values = vec![0.25f32; config.width as usize];
    values[..4].copy_from_slice(&special);
    tch::no_grad(|| {
        let mut bias = store.variables()["ctx_proj.bias"].shallow_clone();
        bias.copy_(&Tensor::from_slice(values.as_slice()));
    });
    let tensors = named(&store);
    let dir = scratch.path().join("checkpoint");
    let state_sha256 = write_checkpoint(&dir, &tensors, &config, 61600);
    let first = scratch.path().join("first");
    let first_receipt = scratch.path().join("first.receipt.json");
    let summary = export_on_cpu(&request(&dir, &first, &first_receipt, &state_sha256)).unwrap();
    let receipt: Value = read_json(&first_receipt).unwrap();
    assert_eq!(receipt["schema"], RECEIPT_SCHEMA);
    assert!(receipt["relative_change"].is_null());
    assert!(summary["relative_change_total"].is_null());
    let checkpoint = &receipt["checkpoint"];
    assert_eq!(checkpoint["accepted_identity"], false);
    assert_eq!(checkpoint["state_sha256"], state_sha256.as_str());
    assert_eq!(
        checkpoint["expected_state_sha256"],
        checkpoint["state_sha256"]
    );
    // The exporter's recomputation, the state's claim and the training code's own hash of
    // the store are three routes to one value.
    let trained = crate::training::parameter_hash(&store).unwrap();
    let state: Value = read_json(&dir.join("state.json")).unwrap();
    assert_eq!(checkpoint["parameter_sha256_recomputed"], trained.as_str());
    assert_eq!(state["parameter_sha256"], trained.as_str());
    let policy = &receipt["policy"];
    let entries = policy["tensors"].as_array().unwrap();
    assert_eq!(policy["tensor_count"], store.variables().len());
    assert_eq!(entries.len(), store.variables().len());
    let names = entries
        .iter()
        .map(|e| e["name"].as_str().unwrap())
        .collect::<Vec<_>>();
    let mut sorted = names.clone();
    sorted.sort_unstable();
    assert_eq!(names, sorted);
    assert!(entries.iter().all(|e| e["dtype"] == "F32"));
    assert_eq!(
        policy["inventory_sha256"],
        hash(&serde_json::to_vec(entries).unwrap()).as_str()
    );
    // Every tensor reads back bit-exact through tch, including the special values.
    let output = first.join(POLICY_OUTPUT);
    let reread = Tensor::read_safetensors(&output)
        .unwrap()
        .into_iter()
        .collect::<BTreeMap<_, _>>();
    assert_eq!(reread.len(), tensors.len());
    for (name, source) in &tensors {
        assert_eq!(
            raw_bytes(&reread[name]).unwrap(),
            raw_bytes(source).unwrap(),
            "{name}"
        );
    }
    let bias = raw_bytes(&reread["ctx_proj.bias"]).unwrap();
    for (i, value) in special.iter().enumerate() {
        assert_eq!(bias[4 * i..4 * i + 4], value.to_le_bytes());
    }
    // Like tch's own writer, no `__metadata__` entry is written.
    let (_, header) = safetensors_header(&fs::read(&output).unwrap()).unwrap();
    assert!(!header.contains_key("__metadata__"));
    assert_eq!(parse_safetensors(&output).unwrap().len(), tensors.len());
    assert_eq!(summary["policy_sha256"], policy["output"]["sha256"]);
    assert_eq!(
        policy["output"]["sha256"],
        file_hash(&output).unwrap().as_str()
    );
    // Determinism: a second export of the same checkpoint writes the same bytes.
    let second = scratch.path().join("second");
    export_on_cpu(&request(
        &dir,
        &second,
        &scratch.path().join("second.receipt.json"),
        &state_sha256,
    ))
    .unwrap();
    assert_eq!(
        fs::read(&output).unwrap(),
        fs::read(second.join(POLICY_OUTPUT)).unwrap()
    );
    assert_no_partials(&first);
    assert_no_partials(scratch.path());
}

#[test]
fn export_refuses_identity_mismatch() {
    let scratch = scratch();
    let config = tiny();
    let store = policy_store(&config);
    let dir = scratch.path().join("checkpoint");
    let state_sha256 = write_checkpoint(&dir, &named(&store), &config, 61600);
    let good: Value = read_json(&dir.join("state.json")).unwrap();
    let output = scratch.path().join("out");
    let receipt = scratch.path().join("receipt.json");
    let attempt = |expect: &str| export_on_cpu(&request(&dir, &output, &receipt, expect));
    let nothing_written = || {
        assert!(!receipt.exists());
        assert!(!output.join(POLICY_OUTPUT).exists());
        assert_no_partials(&output);
        assert_no_partials(scratch.path());
    };
    // (a) another identity, (b) the rejected identity, (c) an uppercase pin.
    refused(
        attempt(&hash(b"other state")),
        "is not the expected identity",
    );
    nothing_written();
    refused(attempt(REJECTED_STATE_SHA256), "rejected");
    nothing_written();
    refused(attempt(&state_sha256.to_ascii_uppercase()), "lowercase");
    nothing_written();
    // (d)-(i): one state.json field at a time, re-pinned so only that property fails.
    type StateEdit<'a> = &'a dyn Fn(&mut Value);
    let cases: [(StateEdit<'_>, &str); 6] = [
        (
            &|s: &mut Value| s["step"] = json!(REJECTED_STEP),
            "rejected checkpoint step",
        ),
        (
            &|s: &mut Value| s["model_sha256"] = json!(hash(b"other model")),
            "model.ot sha256",
        ),
        (
            &|s: &mut Value| s["config_sha256"] = json!(hash(b"other config")),
            "config.json sha256",
        ),
        (
            &|s: &mut Value| {
                s.as_object_mut().unwrap().remove("autocast_cache_lifetime");
            },
            "autocast_cache_lifetime",
        ),
        (
            &|s: &mut Value| s["parameter_sha256"] = json!(hash(b"other parameters")),
            "parameter hash recomputed",
        ),
        (
            &|s: &mut Value| s["schema"] = json!("parallel.checkpoint.v0"),
            "schema",
        ),
    ];
    for (edit, needle) in cases {
        let mut state = good.clone();
        edit(&mut state);
        let pinned = write_state(&dir, &state);
        refused(attempt(&pinned), needle);
        nothing_written();
    }
    let restored = write_state(&dir, &good);
    assert_eq!(restored, state_sha256);
    refused(require_cpu_only(None), "CPU-only");
    refused(require_cpu_only(Some("0")), "CPU-only");
    require_cpu_only(Some("")).unwrap();
}

#[test]
fn export_refuses_architecture_drift() {
    let scratch = scratch();
    let config = tiny();
    let store = policy_store(&config);
    let base = named(&store);
    let w = config.width;
    let bytes_of = |name: &str| raw_bytes(&tensor_of(&base, name)).unwrap();
    let removed = base
        .iter()
        .filter(|(n, _)| n != "reward_encoder.4.bias")
        .map(|(n, t)| (n.clone(), t.shallow_clone()))
        .collect::<Vec<_>>();
    let mut extra = shallow(&base);
    extra.push((
        "extra.weight".into(),
        Tensor::zeros([1], (Kind::Float, Device::Cpu)),
    ));
    let reshaped = replaced(
        &base,
        "ctx_proj.weight",
        Tensor::from_data_size(&bytes_of("ctx_proj.weight"), &[w * w], Kind::Float),
    );
    let widened = bytes_of("ctx_proj.bias")
        .chunks_exact(4)
        .flat_map(|b| f64::from(f32::from_le_bytes([b[0], b[1], b[2], b[3]])).to_le_bytes())
        .collect::<Vec<u8>>();
    let double = replaced(
        &base,
        "ctx_proj.bias",
        Tensor::from_data_size(&widened, &[w], Kind::Double),
    );
    let mut poisoned = bytes_of("token_emb.weight");
    poisoned[4..8].copy_from_slice(&f32::NAN.to_le_bytes());
    let nan = replaced(
        &base,
        "token_emb.weight",
        Tensor::from_data_size(
            &poisoned,
            &tensor_of(&base, "token_emb.weight").size(),
            Kind::Float,
        ),
    );
    for (i, (tensors, needle)) in [
        (removed, "missing [\"reward_encoder.4.bias\"]"),
        (extra, "unexpected [\"extra.weight\"]"),
        (reshaped, "has shape"),
        (double, "expected F32"),
        (nan, "non-finite"),
    ]
    .into_iter()
    .enumerate()
    {
        let dir = scratch.path().join(format!("case-{i}"));
        let pinned = write_checkpoint(&dir, &tensors, &config, 61600);
        let output = scratch.path().join(format!("out-{i}"));
        let receipt = scratch.path().join(format!("receipt-{i}.json"));
        refused(
            export_on_cpu(&request(&dir, &output, &receipt, &pinned)),
            needle,
        );
        assert!(!receipt.exists() && !output.join(POLICY_OUTPUT).exists());
        assert_no_partials(&output);
    }
    assert_no_partials(scratch.path());
}

#[test]
fn export_never_overwrites_or_leaves_partials() {
    let scratch = scratch();
    let config = tiny();
    let store = policy_store(&config);
    let dir = scratch.path().join("checkpoint");
    let state_sha256 = write_checkpoint(&dir, &named(&store), &config, 61600);
    // An existing policy file is refused and left unchanged.
    let output = scratch.path().join("out");
    fs::create_dir_all(&output).unwrap();
    fs::write(output.join(POLICY_OUTPUT), b"existing").unwrap();
    let receipt = scratch.path().join("receipt.json");
    refused(
        export_on_cpu(&request(&dir, &output, &receipt, &state_sha256)),
        "refusing to overwrite",
    );
    assert_eq!(fs::read(output.join(POLICY_OUTPUT)).unwrap(), b"existing");
    assert!(!receipt.exists());
    assert_no_partials(&output);
    // An existing receipt is refused before any write.
    let fresh = scratch.path().join("fresh");
    let existing = scratch.path().join("existing.receipt.json");
    fs::write(&existing, b"{}").unwrap();
    refused(
        export_on_cpu(&request(&dir, &fresh, &existing, &state_sha256)),
        "refusing to overwrite",
    );
    assert!(!fresh.exists());
    assert_eq!(fs::read(&existing).unwrap(), b"{}");
    // Output or receipt under /tmp, or relative, is refused before anything is read.
    refused(
        export_on_cpu(&request(
            &dir,
            Path::new("/tmp/policy-export"),
            &receipt,
            &state_sha256,
        )),
        "never /tmp",
    );
    refused(
        export_on_cpu(&request(
            &dir,
            &fresh,
            Path::new("relative.receipt.json"),
            &state_sha256,
        )),
        "never /tmp",
    );
    // A receipt that could not be written after the policy file is refused before it.
    let unwritten = scratch.path().join("unwritten");
    let missing = scratch.path().join("missing").join("receipt.json");
    refused(
        export_on_cpu(&request(&dir, &unwritten, &missing, &state_sha256)),
        "receipt directory does not exist",
    );
    assert!(!unwritten.join(POLICY_OUTPUT).exists() && !missing.exists());
    refused(
        export_on_cpu(&request(&dir, &unwritten, &unwritten, &state_sha256)),
        "receipt must not name the policy output",
    );
    assert!(!unwritten.exists());
    fs::create_dir_all(&unwritten).unwrap();
    refused(
        export_on_cpu(&request(
            &dir,
            &unwritten,
            &unwritten.join(POLICY_OUTPUT),
            &state_sha256,
        )),
        "receipt must not name the policy output",
    );
    assert!(!unwritten.join(POLICY_OUTPUT).exists());
    assert_no_partials(&unwritten);
    assert!(!fresh.exists());
    assert_no_partials(scratch.path());
}

#[test]
fn export_removes_the_weights_when_the_receipt_cannot_be_written() {
    let scratch = scratch();
    let out = scratch.path().join("out");
    fs::create_dir(&out).unwrap();
    let weights = out.join(POLICY_OUTPUT);
    let receipt = scratch.path().join("receipt.json");
    // A receipt write that left no receipt: the weights it would bind are removed.
    fs::write(&weights, b"weights").unwrap();
    let removed = after_receipt_failure(anyhow::anyhow!("write failed"), &receipt, &weights);
    assert!(
        format!("{removed:#}").contains("removed the unreceipted"),
        "{removed:#}"
    );
    assert!(!weights.exists());
    // This call's own receipt could not be undone: the weights it binds are kept.
    fs::write(&weights, b"weights").unwrap();
    let kept = after_receipt_failure(
        unpublish(
            std::io::Error::other("sync failed"),
            "sync".into(),
            &receipt,
        ),
        &receipt,
        &weights,
    );
    assert!(
        format!("{kept:#}").contains("is still published after a failed write; kept"),
        "{kept:#}"
    );
    assert!(weights.exists());
    // A marker for another path is not this receipt: the weights are removed.
    let other = unpublish(
        std::io::Error::other("sync failed"),
        "sync".into(),
        &scratch.path().join("elsewhere"),
    );
    let removed = after_receipt_failure(other, &receipt, &weights);
    assert!(
        format!("{removed:#}").contains("removed the unreceipted"),
        "{removed:#}"
    );
    assert!(!weights.exists());
}
