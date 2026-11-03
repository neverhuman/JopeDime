use super::*;

fn clear_git_environment(command: &mut Command) {
    for (key, _) in std::env::vars_os().filter(|(key, _)| key.to_string_lossy().starts_with("GIT_"))
    {
        command.env_remove(key);
    }
}

#[test]
fn original_fixture_union_and_hashes_are_preserved() {
    let index = fixture("fixture-index.json");
    assert_eq!(index["scenario_count"], 69);
    assert_eq!(index["original_test_ids"].as_array().unwrap().len(), 28);
    let mut ids = BTreeSet::new();
    for group in ["budget", "classify", "cost", "pack", "schedule"] {
        for case in fixture(&format!("{group}.json"))["cases"]
            .as_array()
            .unwrap()
        {
            assert!(ids.insert(case["id"].as_str().unwrap().to_owned()));
        }
    }
    assert_eq!(
        ids,
        index["scenario_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_owned())
            .collect()
    );
    for (name, hash) in index["file_sha256"].as_object().unwrap() {
        assert_eq!(
            sha256_hex(&fs::read(fixtures().join(name)).unwrap()),
            hash.as_str().unwrap()
        );
    }
}

#[test]
fn dry_run_and_fresh_process_completion_authenticate_all_files() {
    let temp = tempfile::tempdir().unwrap();
    let destination = temp.path().join("custody");
    let members = vec![member("v1"), member("v2")];
    let rows = members.iter().map(|m| m.row.clone()).collect::<Vec<_>>();
    let budgets = json!({"new_version_bytes":0,"raw_hot_bytes":0,"puts":0});
    let result = custody::dry_run(
        &destination,
        &DryRunRequest {
            rows: &rows,
            facts: &[],
            budgets: &budgets,
            members: &members,
            input_sha: None,
            pricing: None,
        },
        &mut |_, _| Ok(()),
    )
    .unwrap();
    let pin = result["receipt_sha256"].as_str().unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_jope_custody"))
        .args(["verify-complete", "--directory"])
        .arg(&destination)
        .args(["--receipt-sha", pin])
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let verified: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(verified["complete"], true);
    assert_eq!(verified["remote_calls"], 0);
    assert_eq!(verified["readback"]["physical_refs"], 2);
    assert_eq!(verified["readback"]["verified_blobs"], 1);
    fs::write(destination.join("object-ledger.json"), b"[]").unwrap();
    assert!(custody::verify_complete(&destination, pin).is_err());
}

#[test]
fn all_publication_write_and_sync_failures_refuse_completed_readers() {
    for name in [
        "object-ledger.json",
        "pack",
        "manifest.json",
        "receipt.json",
    ] {
        for step in [
            PersistStep::Write,
            PersistStep::FileSync,
            PersistStep::ParentSync,
        ] {
            let temp = tempfile::tempdir().unwrap();
            let destination = temp.path().join("custody");
            let members = [member("v1")];
            let rows = [row("v1")];
            let budgets = json!({});
            let result = custody::dry_run(
                &destination,
                &DryRunRequest {
                    rows: &rows,
                    facts: &[],
                    budgets: &budgets,
                    members: &members,
                    input_sha: None,
                    pricing: None,
                },
                &mut |p, s| {
                    let hit = if name == "pack" {
                        p.extension().is_some_and(|x| x == "gz")
                    } else {
                        p.file_name().is_some_and(|x| x == name)
                    };
                    if hit && s == step {
                        bail!("injected persistence error");
                    }
                    Ok(())
                },
            );
            assert!(result.is_err(), "{name}/{step:?}");
            assert!(destination.join("ABORTED").exists());
            let pin = fs::read(destination.join("receipt.json"))
                .map(|b| sha256_hex(&b))
                .unwrap_or_else(|_| "a".repeat(64));
            assert!(
                custody::verify_complete(&destination, &pin).is_err(),
                "{name}/{step:?}"
            );
        }
    }
}

#[test]
fn failed_read_attempts_remain_counted_and_unknown() {
    let mut r = reader(2);
    let mut fake = Fake::new();
    fake.fail_get = true;
    assert!(r.capture(&mut fake, &row("v1")).is_err());
    assert!(r.capture(&mut fake, &row("v1")).is_err());
    assert!(r.capture(&mut fake, &row("v1")).is_err());
    assert_eq!(fake.calls.len(), 2);
    let receipt = r.receipt();
    assert_eq!(receipt["physical_request_count"], 2);
    assert!(receipt["http_requests"]
        .as_array()
        .unwrap()
        .iter()
        .all(|r| r["outcome"].is_null()));
}

#[test]
fn newly_created_root_and_pack_directory_parent_failures_refuse_success() {
    for directory in ["custody", "pack"] {
        let temp = tempfile::tempdir().unwrap();
        let destination = temp.path().join("custody");
        let rows = [row("v1")];
        let members = [member("v1")];
        let budgets = json!({});
        let result = custody::dry_run(
            &destination,
            &DryRunRequest {
                rows: &rows,
                facts: &[],
                budgets: &budgets,
                members: &members,
                input_sha: None,
                pricing: None,
            },
            &mut |path, step| {
                if path.file_name().is_some_and(|n| n == directory)
                    && step == PersistStep::ParentSync
                {
                    bail!("injected directory entry fsync failure");
                }
                Ok(())
            },
        );
        assert!(result.is_err());
        assert!(!destination.join("receipt.json").exists());
        assert!(custody::verify_complete(&destination, &"a".repeat(64)).is_err());
    }
}

#[test]
fn valid_pax_compatibility_and_entire_gzip_tar_tails_are_checked() {
    let index = fixture("hardening/index.json");
    let blob = jope_records::custody_archive::Blob {
        name: index["member_name"].as_str().unwrap().into(),
        sha256: index["member_sha256"].as_str().unwrap().into(),
        bytes: index["member_bytes"].as_u64().unwrap(),
    };
    for (name, entry) in index["archives"].as_object().unwrap() {
        let raw = fs::read(fixtures().join("hardening").join(name)).unwrap();
        assert_eq!(sha256_hex(&raw), entry["sha256"].as_str().unwrap());
        let result = jope_records::custody_archive::verify(
            raw.as_slice(),
            entry["sha256"].as_str().unwrap(),
            std::slice::from_ref(&blob),
        );
        assert_eq!(
            result.is_ok(),
            entry["accept"].as_bool().unwrap(),
            "{name}: {result:?}"
        );
    }
}

#[test]
fn archive_round_trips_cross_block_and_record_size_boundaries() {
    use jope_records::custody_archive::{encode, verify, Blob};

    for size in [
        0, 1, 511, 512, 513, 8192, 8704, 8705, 9215, 9216, 9728, 10239, 10240, 10241, 17409, 19968,
        20479, 20480,
    ] {
        for count in 1..=3 {
            let mut bodies = BTreeMap::new();
            for i in 0..count {
                let bytes = (size + i * 513) % 20481;
                let body = Arc::<[u8]>::from(vec![(i + 1) as u8; bytes]);
                bodies.insert(sha256_hex(&body), body);
            }
            assert_eq!(bodies.len(), count);
            let blobs: Vec<_> = bodies
                .iter()
                .map(|(sha, body)| Blob {
                    name: format!("members/{sha}"),
                    sha256: sha.clone(),
                    bytes: body.len() as u64,
                })
                .collect();
            let encoded = encode(&bodies).unwrap();
            verify(encoded.as_slice(), &sha256_hex(&encoded), &blobs)
                .unwrap_or_else(|error| panic!("size={size} count={count}: {error}"));
        }
    }
}

#[test]
fn external_gnu_ustar_and_pax_record_padding_verify() {
    use jope_records::custody_archive::{verify, Blob};

    let index = fixture("hardening/gnu-layout-index.json");
    for archive in index["archives"].as_array().unwrap() {
        let raw = fs::read(
            fixtures()
                .join("hardening")
                .join(archive["file"].as_str().unwrap()),
        )
        .unwrap();
        assert_eq!(raw.len() as u64, archive["bytes"].as_u64().unwrap());
        assert_eq!(sha256_hex(&raw), archive["sha256"].as_str().unwrap());
        let blobs: Vec<_> = archive["blobs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|blob| Blob {
                name: blob["name"].as_str().unwrap().into(),
                sha256: blob["sha256"].as_str().unwrap().into(),
                bytes: blob["bytes"].as_u64().unwrap(),
            })
            .collect();
        verify(raw.as_slice(), archive["sha256"].as_str().unwrap(), &blobs)
            .unwrap_or_else(|error| panic!("{}: {error}", archive["file"]));
    }
}

#[test]
fn absent_interrupted_or_forged_completion_receipts_refuse() {
    let temp = tempfile::tempdir().unwrap();
    let dir = temp.path().join("incomplete");
    fs::create_dir(&dir).unwrap();
    assert!(custody::verify_complete(&dir, &"a".repeat(64)).is_err());
    fs::write(dir.join("receipt.json"), b"{\"complete\":true").unwrap();
    assert!(custody::verify_complete(&dir, &sha256_hex(b"{\"complete\":true")).is_err());
}

#[test]
fn cost_overflow_is_refused_and_missing_hold_evidence_keeps_bytes() {
    let case = fixture("cost.json")["cases"][0]["input"].clone();
    let mut rows = case["rows"].as_array().unwrap().clone();
    rows[0]["bytes"] = json!(u64::MAX);
    rows.push(rows[0].clone());
    assert!(
        custody::cost::report(&rows, &[], &case["rates"], "2026-10-13T00:00:00+00:00").is_err()
    );
    let mut facts = fixture("classify.json")["cases"][2]["input"]["facts"].clone();
    facts.as_object_mut().unwrap().remove("retention_until");
    let item = fixture("classify.json")["cases"][2]["input"]["row"].clone();
    assert_eq!(
        custody::classify(&item, Some(&facts)).unwrap()["retention_class"],
        "keeper"
    );
}

#[test]
fn cli_dry_run_requires_input_body_pins_and_outside_checkout_output() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input.json");
    let body = temp.path().join("body");
    fs::write(&body, b"same bytes").unwrap();
    let raw=canonical_json(&json!({"rows":[row("v1")],"facts":[],"budgets":{"new_version_bytes":0,"raw_hot_bytes":0,"puts":0},
        "members":[{"row":row("v1"),"body_path":"body","body_sha256":sha256_hex(b"same bytes")}]})).unwrap();
    fs::write(&input, &raw).unwrap();
    let checkout = temp.path().join("checkout");
    fs::create_dir(&checkout).unwrap();
    let mut initialize = Command::new("git");
    initialize.args(["init", "--quiet"]).current_dir(&checkout);
    clear_git_environment(&mut initialize);
    let initialized = initialize.output().unwrap();
    assert!(
        initialized.status.success(),
        "could not initialize checkout fixture: {}",
        String::from_utf8_lossy(&initialized.stderr)
    );
    let execute = |sha: &str, out: &Path| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_jope_custody"));
        command
            .args(["dry-run", "--input"])
            .arg(&input)
            .args(["--input-sha", sha, "--output"])
            .arg(out)
            .current_dir(&checkout);
        clear_git_environment(&mut command);
        command.output().unwrap()
    };
    let bad = temp.path().join("bad");
    assert!(!execute(&"a".repeat(64), &bad).status.success());
    assert!(!bad.exists());
    let inside = checkout.join("target-forbidden-custody-output");
    let refused = execute(&sha256_hex(&raw), &inside);
    assert!(!refused.status.success());
    assert!(!inside.exists());
    let output = temp.path().join("good");
    let accepted = execute(&sha256_hex(&raw), &output);
    assert!(
        accepted.status.success(),
        "{}",
        String::from_utf8_lossy(&accepted.stderr)
    );
    let result: Value = serde_json::from_slice(&accepted.stdout).unwrap();
    assert_eq!(result["receipt"]["input_sha256"], sha256_hex(&raw));
    assert!(custody::verify_complete(&output, result["receipt_sha256"].as_str().unwrap()).is_ok());
}
