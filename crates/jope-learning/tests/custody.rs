//! Original custody-v2 behavioral oracle plus fresh durability/CLI boundary checks.
use anyhow::{bail, Result};
use jope_learning::{
    canon::{canonical_json, sha256_hex},
    custody::{
        self,
        reader::{GetResponse, ReadTransport, Reader},
        CapturedMember, DryRunRequest, PersistStep,
    },
};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    fs,
    io::{Cursor, Read},
    path::{Path, PathBuf},
    process::Command,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/custody_v2")
}
fn fixture(name: &str) -> Value {
    serde_json::from_slice(&fs::read(fixtures().join(name)).unwrap()).unwrap()
}
fn row(version: &str) -> Value {
    json!({"bucket":"bucket","key":"safe/objects/evaluation","version_id":version,"bytes":10,
        "storage_class":"STANDARD","is_latest":true,"etag":"not-a-sha","last_modified":"2026-10-13T00:00:00+00:00"})
}
fn member(version: &str) -> CapturedMember {
    CapturedMember {
        row: row(version),
        body: Arc::from(b"same bytes".as_slice()),
    }
}
fn reader(cap: usize) -> Reader {
    Reader::new(
        "bucket",
        "safe/objects/",
        cap,
        32 * 1024 * 1024,
        64 * 1024 * 1024,
        900.,
    )
    .unwrap()
}

struct Body {
    raw: Cursor<Vec<u8>>,
    read: Arc<AtomicUsize>,
    closed: Arc<AtomicBool>,
}
impl Read for Body {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        let n = self.raw.read(out)?;
        self.read.fetch_add(n, Ordering::SeqCst);
        Ok(n)
    }
}
impl Drop for Body {
    fn drop(&mut self) {
        self.closed.store(true, Ordering::SeqCst);
    }
}
struct Fake {
    pages: VecDeque<Value>,
    calls: Vec<Value>,
    version: String,
    bytes: Vec<u8>,
    read: Arc<AtomicUsize>,
    closed: Arc<AtomicBool>,
    fail_get: bool,
}
impl Fake {
    fn new() -> Self {
        Self {
            pages: VecDeque::new(),
            calls: Vec::new(),
            version: "v1".into(),
            bytes: b"same bytes".to_vec(),
            read: Arc::new(AtomicUsize::new(0)),
            closed: Arc::new(AtomicBool::new(false)),
            fail_get: false,
        }
    }
}
impl ReadTransport for Fake {
    fn list_versions(&mut self, params: &Value) -> Result<Value> {
        self.calls.push(params.clone());
        self.pages
            .pop_front()
            .ok_or_else(|| anyhow::anyhow!("no scripted page"))
    }
    fn get_version(&mut self, params: &Value) -> Result<GetResponse> {
        self.calls.push(params.clone());
        if self.fail_get {
            bail!("injected unknown send outcome");
        }
        Ok(GetResponse {
            version_id: self.version.clone(),
            content_length: self.bytes.len() as u64,
            body: Box::new(Body {
                raw: Cursor::new(self.bytes.clone()),
                read: self.read.clone(),
                closed: self.closed.clone(),
            }),
        })
    }
}
fn listing(version: &str) -> Value {
    json!({"Key":"safe/objects/evaluation","VersionId":version,"Size":10,"StorageClass":"STANDARD",
        "IsLatest":true,"LastModified":"2026-10-13T00:00:00+00:00","ETag":"not-a-sha"})
}

fn schedule(id: &str, input: &Value) -> Result<()> {
    match id {
        "gb02.schedule.returned_version_mismatch" => {
            let mut fake = Fake::new();
            fake.version = input["returned_version"].as_str().unwrap().into();
            let mut r = reader(10);
            assert!(r.capture(&mut fake, &row("v1")).is_err());
            assert_eq!(fake.read.load(Ordering::SeqCst), 0);
            assert!(fake.closed.load(Ordering::SeqCst));
            assert_eq!(r.receipt()["physical_request_count"], 1);
        }
        "gb02.schedule.mutations" => {
            for op in input["operations"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap())
                .chain(["GetObject"])
            {
                let output = Command::new(env!("CARGO_BIN_EXE_jope_custody"))
                    .arg(op)
                    .output()?;
                assert_eq!(output.status.code(), Some(2));
                assert!(output.stdout.is_empty());
                assert!(String::from_utf8(output.stderr)?.contains("unrecognized subcommand"));
            }
        }
        "gb02.schedule.scope_or_unversioned" => {
            for (field, value) in [
                ("bucket", "other"),
                ("key", "outside/evaluation"),
                ("version_id", "null"),
            ] {
                let mut r = reader(10);
                let mut fake = Fake::new();
                let mut item = row("v1");
                item[field] = json!(value);
                assert!(r.capture(&mut fake, &item).is_err());
                assert!(fake.calls.is_empty());
                assert_eq!(r.receipt()["physical_request_count"], 0);
            }
        }
        "gb02.schedule.two_pages"
        | "gb02.schedule.repeated_cursor"
        | "gb02.schedule.duplicate_version"
        | "gb02.schedule.request_cap" => {
            let versions = input["versions"].as_array().unwrap();
            let mut fake = Fake::new();
            for (i, v) in versions.iter().enumerate() {
                let mut page = json!({"Versions":[listing(v.as_str().unwrap())],"IsTruncated":i+1<versions.len()});
                if id == "gb02.schedule.two_pages" {
                    let fields = &input["pages"][i];
                    page["IsTruncated"] = fields["IsTruncated"].clone();
                    if !fields["NextKeyMarker"].is_null() {
                        page["NextKeyMarker"] = fields["NextKeyMarker"].clone();
                        page["NextVersionIdMarker"] = fields["NextVersionIdMarker"].clone();
                    }
                } else if id == "gb02.schedule.repeated_cursor" {
                    page["IsTruncated"] = json!(true);
                    page["NextKeyMarker"] = input["cursors"][i].clone();
                } else if i + 1 < versions.len() {
                    page["NextKeyMarker"] = json!("next");
                }
                fake.pages.push_back(page);
            }
            let mut r = reader(input["request_cap"].as_u64().unwrap_or(10) as usize);
            let result = r.census(&mut fake);
            if id == "gb02.schedule.two_pages" {
                let result = result?;
                assert_eq!(result["rows"].as_array().unwrap().len(), 2);
                assert_eq!(result["complete_listing"], true);
                assert_eq!(result["point_in_time_snapshot"], false);
                assert_eq!(fake.calls[1]["KeyMarker"], "next");
                assert_eq!(fake.calls[1]["VersionIdMarker"], "old");
            } else {
                assert!(result.is_err());
            }
            if id == "gb02.schedule.request_cap" {
                assert_eq!(fake.calls.len(), 1);
            }
        }
        "gb02.schedule.deadline_or_body_cap" => {
            assert!(Reader::new("bucket", "safe/objects/", 10, 100, 100, f64::INFINITY).is_err());
            let mut r = Reader::new("bucket", "safe/objects/", 10, 100, 1, 900.)?;
            let mut fake = Fake::new();
            assert!(r.capture(&mut fake, &row("v1")).is_err());
            assert!(fake.calls.is_empty());
            let mut r = Reader::new("bucket", "safe/objects/", 10, 100, 100, 1e-9)?;
            assert!(r.capture(&mut fake, &row("v1")).is_err());
            assert!(fake.calls.is_empty());
        }
        "gb02.schedule.capture" => {
            let mut r = reader(10);
            let mut fake = Fake::new();
            assert_eq!(r.capture(&mut fake, &row("v1"))?, b"same bytes");
            assert_eq!(r.receipt()["etag_is_sha256"], false);
            assert_eq!(r.receipt()["object_body_bytes"], 10);
            assert!(fake.closed.load(Ordering::SeqCst));
        }
        "gb02.schedule.manifest_replacement" => {
            let temp = tempfile::tempdir()?;
            let dir = temp.path().join("pack");
            custody::pack_local(&dir, &[member("v1")])?;
            let before = fs::read(dir.join("manifest.json"))?;
            fs::write(dir.join("manifest.json"), b"{}")?;
            let checked = custody::verify_manifest_bytes(&before, &dir)?;
            assert_eq!(checked["manifest_sha256"], sha256_hex(&before));
            assert_eq!(checked["remote_calls"], 0);
        }
        "gb02.schedule.fsync_failure" => {
            for step in [PersistStep::FileSync, PersistStep::ParentSync] {
                let temp = tempfile::tempdir()?;
                let dir = temp.path().join("pack");
                assert!(custody::pack_with(&dir, &[member("v1")], &mut |_, s| {
                    if s == step {
                        bail!("injected EIO");
                    }
                    Ok(())
                })
                .is_err());
                assert!(!dir.join("manifest.json").exists());
                assert!(dir.join("ABORTED").exists());
            }
        }
        "gb02.schedule.existing_output" => {
            let temp = tempfile::tempdir()?;
            fs::write(temp.path().join("sentinel"), b"keep")?;
            assert!(custody::pack_local(temp.path(), &[member("v1")]).is_err());
            assert_eq!(fs::read(temp.path().join("sentinel"))?, b"keep");
        }
        "gb02.schedule.oversized_manifest" => {
            let temp = tempfile::tempdir()?;
            let file = temp.path().join("manifest.json");
            let mut raw = b"{}".to_vec();
            raw.resize(custody::MAX_MANIFEST as usize + 2, b' ');
            fs::write(&file, raw)?;
            let error = custody::verify_saved(&file).unwrap_err();
            assert!(format!("{error:#}").contains("bounded manifest bytes"));
        }
        _ => bail!("unimplemented original scenario {id}"),
    }
    Ok(())
}

fn run_original(original: &str) {
    let mut executed = 0;
    for group in ["budget", "classify", "cost", "pack", "schedule"] {
        for case in fixture(&format!("{group}.json"))["cases"]
            .as_array()
            .unwrap()
        {
            if !case["original_test_ids"]
                .as_array()
                .unwrap()
                .iter()
                .any(|v| v == original)
            {
                continue;
            }
            let input = &case["input"];
            let id = case["id"].as_str().unwrap();
            let result = match group {
                "budget" => custody::budget_status(input),
                "classify" => custody::classify(&input["row"], Some(&input["facts"])),
                "cost" => custody::cost::report(
                    input["rows"].as_array().unwrap(),
                    input["requests"].as_array().unwrap(),
                    &input["rates"],
                    input["as_of"].as_str().unwrap(),
                ),
                "pack" => custody::verify_pack(
                    &fixtures().join(input["pack"].as_str().unwrap()),
                    &canonical_json(&input["manifest"]).unwrap(),
                ),
                "schedule" => {
                    schedule(id, input).unwrap();
                    executed += 1;
                    continue;
                }
                _ => unreachable!(),
            };
            if case["expected"]["outcome"] == "refuse" {
                let error = result.unwrap_err();
                assert!(
                    format!("{error:#}").contains(case["expected"]["reason"].as_str().unwrap()),
                    "{id}: {error:#}"
                );
            } else {
                let value = result.unwrap_or_else(|error| panic!("{id}: {error:#}"));
                for (key, expected) in case["expected"]["result"].as_object().unwrap() {
                    assert_eq!(&value[key], expected, "{id}/{key}");
                }
            }
            executed += 1;
        }
    }
    assert!(
        executed > 0,
        "original ID has no executed mapping: {original}"
    );
}

macro_rules! original {
    ($name:ident,$class:literal,$test:literal) => {
        #[test]
        fn $name() {
            run_original(concat!(
                "custody_v2.tests.test_retention.",
                $class,
                ".",
                $test
            ));
        }
    };
}
original!(
    retention_unknown,
    "RetentionTests",
    "test_unknown_facts_and_null_versions_keep_everything"
);
original!(
    retention_negative,
    "RetentionTests",
    "test_complete_negative_facts_are_only_a_non_applied_proposal"
);
original!(
    retention_keeper_facts,
    "RetentionTests",
    "test_each_keeper_reason_wins"
);
original!(
    retention_borrowed_facts,
    "RetentionTests",
    "test_keeper_facts_cannot_be_borrowed_from_another_version"
);
original!(
    retention_missing_facts,
    "RetentionTests",
    "test_missing_salt_or_hold_facts_cannot_release_bytes"
);
original!(
    retention_permanent,
    "RetentionTests",
    "test_noncurrent_unknown_and_small_artifacts_stay_forever"
);
original!(
    budget_boundaries,
    "RetentionTests",
    "test_alarm_at_80_percent_and_refusal_at_100_percent"
);
original!(
    budget_unknown_and_types,
    "RetentionTests",
    "test_missing_counters_and_boolean_counts_refuse"
);
original!(
    cost_original,
    "RetentionTests",
    "test_cost_uses_binary_gb_actual_request_counts_and_explicit_unknowns"
);
original!(
    pack_aliases,
    "PackTests",
    "test_deduplicated_pack_preserves_all_physical_refs_and_reopens"
);
original!(
    pack_existing_and_qualified,
    "PackTests",
    "test_existing_destination_and_qualified_manifest_refuse"
);
original!(
    pack_duplicate_and_short,
    "PackTests",
    "test_duplicate_physical_version_and_incomplete_body_refuse"
);
original!(
    pack_corruption,
    "PackTests",
    "test_corrupt_pack_bad_member_hash_and_unknown_refs_refuse"
);
original!(
    pack_ref_bounds,
    "PackTests",
    "test_excessive_reference_and_blob_rosters_refuse"
);
original!(
    pack_boolean_bytes,
    "PackTests",
    "test_boolean_reference_bytes_refuse"
);
original!(
    pack_manifest_bound,
    "PackTests",
    "test_oversized_manifest_refuses_before_json_or_pack_parsing"
);
original!(
    pack_name_bound,
    "PackTests",
    "test_invalid_pack_digest_refuses_before_pack_file_access"
);
original!(
    pack_payload_hash,
    "PackTests",
    "test_payload_corruption_refuses_even_if_outer_archive_sha_is_updated"
);
original!(
    pack_expansion_bound,
    "PackTests",
    "test_pax_metadata_expansion_refuses_before_tar_parsing"
);
original!(
    pack_manifest_identity,
    "PackTests",
    "test_readback_digest_binds_verified_manifest_despite_replacement"
);
original!(
    pack_fsync,
    "PackTests",
    "test_fsync_failure_never_returns_a_success_or_publishes_manifest"
);
original!(
    reader_unowned_and_mutation,
    "ReaderTests",
    "test_all_mutations_are_refused_before_any_fake_dispatch"
);
original!(
    reader_scope_and_version,
    "ReaderTests",
    "test_wrong_scope_and_unversioned_get_refuse_before_dispatch"
);
original!(
    reader_pagination,
    "ReaderTests",
    "test_full_pagination_and_every_version_is_retained"
);
original!(
    reader_cursor_identity_budget,
    "ReaderTests",
    "test_repeated_cursor_duplicate_version_and_request_cap_refuse"
);
original!(
    reader_deadline_body,
    "ReaderTests",
    "test_nonfinite_expired_and_body_budget_refuse"
);
original!(
    reader_capture,
    "ReaderTests",
    "test_capture_records_real_body_bytes_and_never_uses_etag_as_sha"
);
original!(
    reader_get_identity,
    "ReaderTests",
    "test_wrong_returned_version_refuses_without_consuming_body"
);

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
    let execute = |sha: &str, out: &Path| {
        Command::new(env!("CARGO_BIN_EXE_jope_custody"))
            .args(["dry-run", "--input"])
            .arg(&input)
            .args(["--input-sha", sha, "--output"])
            .arg(out)
            .current_dir(Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."))
            .output()
            .unwrap()
    };
    let bad = temp.path().join("bad");
    assert!(!execute(&"a".repeat(64), &bad).status.success());
    assert!(!bad.exists());
    let inside = Path::new(env!("CARGO_MANIFEST_DIR")).join("target-forbidden-custody-output");
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
