use super::*;

#[test]
fn export_refuses_stale_temporaries_before_any_write() {
    let scratch = scratch();
    let config = tiny();
    let store = policy_store(&config);
    let dir = scratch.path().join("checkpoint");
    let state_sha256 = write_checkpoint(&dir, &named(&store), &config, 61600);
    let output = scratch.path().join("out");
    let receipt = scratch.path().join("receipt.json");
    // A wrong pinned identity: only the pre-check can refuse with the temporary message;
    // any later refusal (write_and_verify's own check) would name the identity instead.
    let wrong = hash(b"other state");
    // A stale receipt temporary (here a directory) is refused before the weights exist.
    let stale_receipt = receipt_partial(&receipt);
    fs::create_dir(&stale_receipt).unwrap();
    refused(
        export_on_cpu(&request(&dir, &output, &receipt, &wrong)),
        "refusing to replace an existing temporary",
    );
    assert!(!output.exists() && stale_receipt.is_dir());
    fs::remove_dir(&stale_receipt).unwrap();
    // So is a stale weights temporary in an existing output directory, left untouched.
    fs::create_dir(&output).unwrap();
    let stale_weights = weights_partial(&output.join(POLICY_OUTPUT)).unwrap();
    fs::write(&stale_weights, b"stale").unwrap();
    refused(
        export_on_cpu(&request(&dir, &output, &receipt, &wrong)),
        "refusing to replace an existing temporary",
    );
    assert_eq!(fs::read(&stale_weights).unwrap(), b"stale");
    assert!(!output.join(POLICY_OUTPUT).exists() && !receipt.exists());
    // Without them the export succeeds and leaves no temporary behind.
    fs::remove_file(&stale_weights).unwrap();
    export_on_cpu(&request(&dir, &output, &receipt, &state_sha256)).unwrap();
    assert!(output.join(POLICY_OUTPUT).exists() && receipt.exists());
    assert_no_partials(&output);
    assert_no_partials(scratch.path());
}

#[test]
fn export_refuses_a_dangling_symlink_at_the_receipt_or_policy_path() {
    let scratch = scratch();
    let config = tiny();
    let store = policy_store(&config);
    let dir = scratch.path().join("checkpoint");
    write_checkpoint(&dir, &named(&store), &config, 61600);
    // A wrong pinned identity: only the pre-check can refuse with "refusing to overwrite";
    // any later refusal would name the identity instead.
    let wrong = hash(b"other state");
    let nowhere = scratch.path().join("nowhere").join("target");
    // A dangling link at the receipt path is refused before the weights are written.
    let output = scratch.path().join("out");
    let receipt = scratch.path().join("receipt.json");
    std::os::unix::fs::symlink(&nowhere, &receipt).unwrap();
    refused(
        export_on_cpu(&request(&dir, &output, &receipt, &wrong)),
        "refusing to overwrite",
    );
    assert!(!output.exists());
    // So is a dangling link at the policy path, before anything is written.
    let other = scratch.path().join("other");
    fs::create_dir(&other).unwrap();
    std::os::unix::fs::symlink(&nowhere, other.join(POLICY_OUTPUT)).unwrap();
    let fresh = scratch.path().join("fresh.receipt.json");
    refused(
        export_on_cpu(&request(&dir, &other, &fresh, &wrong)),
        "refusing to overwrite",
    );
    assert!(!fresh.exists() && !nowhere.exists());
    assert_no_partials(&other);
    assert_no_partials(scratch.path());
}

#[test]
fn export_refuses_a_receipt_that_aliases_the_policy_file_through_a_symlink() {
    let scratch = scratch();
    let config = tiny();
    let store = policy_store(&config);
    let dir = scratch.path().join("checkpoint");
    write_checkpoint(&dir, &named(&store), &config, 61600);
    // A wrong pinned identity: only the pre-check can refuse with the alias message.
    let wrong = hash(b"other state");
    let real = scratch.path().join("real");
    fs::create_dir(&real).unwrap();
    let alias = scratch.path().join("alias");
    std::os::unix::fs::symlink(&real, &alias).unwrap();
    for (output, receipt) in [
        (&real, alias.join(POLICY_OUTPUT)),
        (&alias, real.join(POLICY_OUTPUT)),
    ] {
        refused(
            export_on_cpu(&request(&dir, output, &receipt, &wrong)),
            "receipt must not name the policy output",
        );
    }
    assert!(!real.join(POLICY_OUTPUT).exists());
    assert_no_partials(&real);
    // A receipt naming a directory the export would create above its output is refused
    // before that directory is created.
    let new = scratch.path().join("new");
    refused(
        export_on_cpu(&request(&dir, &new.join("deeper"), &new, &wrong)),
        "receipt must not name the policy output",
    );
    assert!(!new.exists());
}

#[test]
fn export_refuses_tmp_through_parent_components_and_symlinks() {
    let scratch = scratch();
    let config = tiny();
    let store = policy_store(&config);
    let dir = scratch.path().join("checkpoint");
    let state_sha256 = write_checkpoint(&dir, &named(&store), &config, 61600);
    let output = scratch.path().join("out");
    let receipt = scratch.path().join("receipt.json");
    // Only a link is created here; nothing is ever written under /tmp.
    let link = scratch.path().join("tmp-link");
    std::os::unix::fs::symlink("/tmp", &link).unwrap();
    let (link_output, link_receipt) = (link.join("out"), link.join("receipt.json"));
    // `<scratch>/../../…/tmp/policy-export`, which resolves to /tmp/policy-export.
    let mut dotted = scratch.path().to_path_buf();
    for _ in scratch.path().components() {
        dotted.push("..");
    }
    dotted.push("tmp/policy-export");
    // A `..` across a directory that does not exist yet resolves only when written: refused
    // by the component rule alone.
    let crossing = scratch.path().join("absent").join("..").join("out2");
    // A dangling link to a /tmp directory that does not exist (it is never created), above
    // the output or as the output directory itself.
    let nowhere = format!("/tmp/jope-export-test-absent-{}", std::process::id());
    let dangling = scratch.path().join("dangling-link");
    std::os::unix::fs::symlink(&nowhere, &dangling).unwrap();
    let below_dangling = dangling.join("out");
    for (out, rcpt) in [
        (&dotted, &receipt),
        (&link_output, &receipt),
        (&output, &link_receipt),
        (&crossing, &receipt),
        (&below_dangling, &receipt),
        (&dangling, &receipt),
    ] {
        refused(
            export_on_cpu(&request(&dir, out, rcpt, &state_sha256)),
            "never /tmp",
        );
    }
    assert!(!output.exists() && !receipt.exists());
    assert!(!scratch.path().join("absent").exists() && !scratch.path().join("out2").exists());
    assert!(!Path::new(&nowhere).exists());
    assert_no_partials(scratch.path());
}

#[test]
fn relative_change_per_tensor() {
    let scratch = scratch();
    let config = tiny();
    let store = policy_store(&config);
    let parent_tensors = named(&store);
    let parent_dir = scratch.path().join("parent");
    write_checkpoint(&parent_dir, &parent_tensors, &config, 61000);
    // The child scales ctx_proj.weight by 1.5 in float32; ctx_proj.bias is unchanged.
    let scaled = raw_bytes(&tensor_of(&parent_tensors, "ctx_proj.weight"))
        .unwrap()
        .chunks_exact(4)
        .flat_map(|b| (f32::from_le_bytes([b[0], b[1], b[2], b[3]]) * 1.5).to_le_bytes())
        .collect::<Vec<u8>>();
    let child_tensors = replaced(
        &parent_tensors,
        "ctx_proj.weight",
        Tensor::from_data_size(&scaled, &[config.width, config.width], Kind::Float),
    );
    let child_dir = scratch.path().join("child");
    let child_state = write_checkpoint(&child_dir, &child_tensors, &config, 61600);
    let parent = load_checkpoint(&parent_dir, None).unwrap();
    let child = load_checkpoint(&child_dir, None).unwrap();
    let change = relative_change(&parent, &child).unwrap();
    assert_eq!(
        change["parent"]["state_sha256"],
        parent.state_sha256.as_str()
    );
    let rows = change["tensors"].as_array().unwrap();
    assert_eq!(rows.len(), child.tensors.len());
    let (mut parent_squares, mut change_squares) = (0f64, 0f64);
    for row in rows {
        let name = row["name"].as_str().unwrap();
        let parent_norm = row["parent_norm"].as_f64().unwrap();
        let change_norm = row["change_norm"].as_f64().unwrap();
        parent_squares += parent_norm * parent_norm;
        change_squares += change_norm * change_norm;
        if name == "ctx_proj.weight" {
            let weights = &parent
                .tensors
                .iter()
                .find(|t| t.name == name)
                .unwrap()
                .bytes;
            let nonzero = f32_values(weights).filter(|v| *v != 0.0).count();
            assert_eq!(row["changed_values"], nonzero);
            let relative = row["relative_change"].as_f64().unwrap();
            // 0.5 up to the float32 rounding of 1.5·θ, and exactly the f64 ratio of the
            // stored values.
            assert!((relative - 0.5).abs() < 1e-6, "{relative}");
            let (mut p, mut d) = (0f64, 0f64);
            for (before, after) in f32_values(weights).zip(f32_values(&scaled)) {
                p += f64::from(before) * f64::from(before);
                d += (f64::from(after) - f64::from(before)).powi(2);
            }
            assert!((relative - d.sqrt() / p.sqrt()).abs() < 1e-12);
        } else {
            assert_eq!(row["change_norm"], 0.0f64, "{name}");
            assert_eq!(row["changed_values"], 0u64, "{name}");
            // Zero-initialized tensors (the loss log-variances, attention and norm biases)
            // have no parent norm, so their relative change is undefined.
            if parent_norm == 0.0 {
                assert!(row["relative_change"].is_null(), "{name}");
            } else {
                assert_eq!(row["relative_change"], 0.0f64, "{name}");
            }
        }
        if name.starts_with("loss_log_vars.") {
            assert_eq!(parent_norm, 0.0);
            assert!(row["relative_change"].is_null());
        }
    }
    let total = &change["total"];
    let total_parent = total["parent_norm"].as_f64().unwrap();
    let total_change = total["change_norm"].as_f64().unwrap();
    assert!((total_parent - parent_squares.sqrt()).abs() <= 1e-12 * total_parent.max(1.0));
    assert!((total_change - change_squares.sqrt()).abs() <= 1e-12 * total_change.max(1.0));
    // A parent of another configuration is refused; a parent equal to the child is all zero.
    let other = Config {
        layers: 2,
        ..tiny()
    };
    let other_dir = scratch.path().join("other");
    write_checkpoint(&other_dir, &named(&policy_store(&other)), &other, 61000);
    let other = load_checkpoint(&other_dir, None).unwrap();
    refused(
        relative_change(&other, &child),
        "parent checkpoint has another configuration",
    );
    // The export path: `--parent` copies the same diagnostic into the receipt and the
    // summary; a parent of another configuration is refused before any write.
    let exported = |parent: &Path, name: &str| {
        let out = scratch.path().join(format!("out-{name}"));
        let receipt = scratch.path().join(format!("{name}.receipt.json"));
        let result = export_on_cpu(&ExportRequest {
            parent: Some(parent.to_path_buf()),
            ..request(&child_dir, &out, &receipt, &child_state)
        });
        (result, out, receipt)
    };
    let (summary, _, receipt) = exported(&parent_dir, "with-parent");
    let summary = summary.unwrap();
    let receipt: Value = read_json(&receipt).unwrap();
    assert_eq!(receipt["relative_change"], change);
    assert_eq!(summary["relative_change_total"], change["total"]);
    assert_eq!(
        summary["relative_change_total"],
        receipt["relative_change"]["total"]
    );
    // H3: a parent equal to the child reports a zero total change through the export.
    let (summary, _, _) = exported(&child_dir, "self-parent");
    let summary = summary.unwrap();
    assert_eq!(summary["relative_change_total"]["change_norm"], 0.0f64);
    let (result, out, receipt) = exported(&other_dir, "other-parent");
    refused(result, "parent checkpoint has another configuration");
    assert!(!receipt.exists() && !out.join(POLICY_OUTPUT).exists());
    assert_no_partials(&out);
    assert_no_partials(scratch.path());
    let same = relative_change(&child, &child).unwrap();
    assert_eq!(same["total"]["change_norm"], 0.0f64);
    for row in same["tensors"].as_array().unwrap() {
        assert_eq!(row["change_norm"], 0.0f64);
        assert_eq!(row["changed_values"], 0u64);
    }
}

/// Anchor rule A1 (hard) and A2 (recorded): exporting the step-60000 checkpoint must
/// reproduce the accepted export tensor for tensor, and should reproduce its bytes.
#[test]
#[ignore = "needs JOPE_EXPORT_ANCHOR_CHECKPOINT (the step-60000 checkpoint directory)"]
fn anchor_step_60000_reproduces_the_accepted_export() {
    let checkpoint = PathBuf::from(
        std::env::var_os("JOPE_EXPORT_ANCHOR_CHECKPOINT")
            .expect("set JOPE_EXPORT_ANCHOR_CHECKPOINT to run this ignored evidence test"),
    );
    let scratch = scratch();
    let output = scratch.path().join("anchor");
    let receipt_path = scratch.path().join("anchor.receipt.json");
    export_on_cpu(&request(
        &checkpoint,
        &output,
        &receipt_path,
        ACCEPTED_STATE_SHA256,
    ))
    .unwrap();
    let receipt: Value = read_json(&receipt_path).unwrap();
    let policy = &receipt["policy"];
    // A1: tensor-identical to the accepted export.
    assert_eq!(receipt["checkpoint"]["accepted_identity"], true);
    assert_eq!(
        policy["inventory_sha256"],
        "dcc5629e4c6c921d610037432d04609c109804a75a68f98fffd33bf51b2c02fa"
    );
    assert_eq!(policy["tensor_count"], 136u64);
    assert_eq!(policy["total_tensor_bytes"], 111_185_976u64);
    assert_eq!(
        receipt["checkpoint"]["parameter_sha256_recomputed"],
        "9c7057e7f6a9e348f030da73f03522977b8c08915d3a221d7d5ab2e4213fcd92"
    );
    assert_eq!(
        policy["source"]["sha256"],
        "053df4f53b84d2ebe7c14fbf907a149760af3d11de4a57b5ff89be8c2d3e8129"
    );
    assert_eq!(policy["source"]["file_bytes"], 111_218_463u64);
    if let Some(accepted) = std::env::var_os("JOPE_EXPORT_ACCEPTED_RECEIPT") {
        let accepted: Value = read_json(Path::new(&accepted)).unwrap();
        assert_eq!(policy["tensors"], accepted["policy"]["tensors"]);
    }
    // A2: recorded, not asserted.
    let byte_identical = policy["output"]["sha256"]
        == "4b4b5a40f03c9226b4b15856fd35b90127c8b6a4d1435b815484cdd050744f1e"
        && policy["output"]["file_bytes"] == 111_199_088u64;
    println!(
            "anchor A2: policy.safetensors sha256 {} ({} bytes); byte_identical_to_accepted={byte_identical}",
            policy["output"]["sha256"], policy["output"]["file_bytes"]
        );
}
