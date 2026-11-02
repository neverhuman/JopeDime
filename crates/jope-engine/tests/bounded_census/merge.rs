use super::*;

#[test]
fn merge_combines_pinned_documents_into_a_preflight_census() {
    let now = now_ms();
    let (_t, root) = tempdir();
    let (base, _) = collect::collect(
        &cpu_only(install(&root)),
        &with_commands(&FakeNvidia::default()),
        now,
    )
    .unwrap();
    let docs: Vec<HostCensus> = [2, 0, 1]
        .into_iter()
        .map(|i| synthetic(&base, i, now - 1_000 * i as u64))
        .collect();
    let pins: Vec<PinnedDocument> = docs
        .iter()
        .enumerate()
        .map(|(i, d)| pinned(&format!("doc-{i}"), d))
        .collect();
    let (merged, report) = collect::merge(&pins, policy(), now).unwrap();
    merged.validate().unwrap();
    assert_eq!(
        merged
            .hosts
            .iter()
            .map(|h| h.identity.host.as_str())
            .collect::<Vec<_>>(),
        ["fixture-host-0", "fixture-host-1", "fixture-host-2"]
    );
    assert_eq!(merged.captured_unix_ms, now);
    assert_eq!(
        report.census_sha256,
        sha(&serde_json::to_vec(&merged).unwrap())
    );
    assert!(!report.allocation_held);
    let expected: std::collections::BTreeMap<String, String> = docs
        .iter()
        .zip(&pins)
        .map(|(d, p)| (d.hosts[0].identity.host.clone(), p.sha256.clone()))
        .collect();
    assert_eq!(report.documents.len(), 3);
    for d in &report.documents {
        assert_eq!(expected[&d.host], d.sha256);
    }
    let hosts: Vec<_> = merged
        .hosts
        .iter()
        .map(|h| (h.identity.clone(), None))
        .collect();
    let receipt = preflight(&root, "merged", &plan_for(now, &hosts), &merged, now).unwrap();
    assert_eq!(receipt.hosts.len(), 3);
    assert_eq!(receipt.census_sha256, report.census_sha256);
}

#[test]
fn merge_refuses_tampered_stale_future_duplicate_and_mixed_documents() {
    let now = now_ms();
    let (_t, root) = tempdir();
    let (base, _) = collect::collect(
        &cpu_only(install(&root)),
        &with_commands(&FakeNvidia::default()),
        now,
    )
    .unwrap();
    let good = |i| pinned(&format!("doc-{i}"), &synthetic(&base, i, now));
    let refuse = |docs: Vec<PinnedDocument>, needle: &str| {
        let refused = err(collect::merge(&docs, policy(), now));
        assert!(refused.contains(needle), "{needle}: {refused}");
    };
    // Tampered in transit: the bytes no longer match the collector's printed digest.
    let mut tampered = good(1);
    tampered.bytes = String::from_utf8(tampered.bytes)
        .unwrap()
        .replace("fixture-host-1", "fixture-host-9")
        .into_bytes();
    refuse(vec![good(0), tampered], "was altered");
    let mut bad_pin = good(1);
    bad_pin.sha256 = "0".repeat(63);
    refuse(vec![good(0), bad_pin], "64 lowercase hex");
    // Stale and future telemetry.
    refuse(
        vec![good(0), pinned("stale", &synthetic(&base, 1, now - 60_001))],
        "stale",
    );
    refuse(
        vec![good(0), pinned("future", &synthetic(&base, 1, now + 2_001))],
        "future",
    );
    // Duplicates: the same document twice, or two documents sharing any identity.
    refuse(vec![good(0), good(0)], "given twice");
    for (what, edit) in [
        (
            "host",
            Box::new(|c: &mut HostCensus| c.hosts[0].identity.host = "fixture-host-0".into())
                as Box<dyn Fn(&mut HostCensus)>,
        ),
        (
            "machine id",
            Box::new(|c: &mut HostCensus| {
                c.hosts[0].identity.machine_id = synthetic(&base, 0, now).hosts[0]
                    .identity
                    .machine_id
                    .clone()
            }),
        ),
        (
            "boot id",
            Box::new(|c: &mut HostCensus| c.hosts[0].identity.boot_id = "fixture-boot-0".into()),
        ),
        (
            "address",
            Box::new(|c: &mut HostCensus| c.hosts[0].identity.address = "192.0.2.1".into()),
        ),
        (
            "certificate",
            Box::new(|c: &mut HostCensus| {
                c.hosts[0].identity.certificate_sha256 = sha(b"fixture-certificate-0")
            }),
        ),
    ] {
        let mut twin = synthetic(&base, 1, now);
        edit(&mut twin);
        refuse(
            vec![good(0), pinned("twin", &twin)],
            &format!("duplicates the {what}"),
        );
    }
    // Mixed or unsupported schema versions.
    let mut old = synthetic(&base, 1, now);
    old.schema = "jope.native_host_census.v0".into();
    refuse(vec![good(0), pinned("old", &old)], "mixed or unsupported");
    // A document is one host's observation.
    let mut two = synthetic(&base, 1, now);
    two.hosts.push(synthetic(&base, 2, now).hosts[0].clone());
    refuse(vec![good(0), pinned("two", &two)], "exactly one host");
    // Freshness policy bounds.
    for bad in [
        MergePolicy {
            max_age_ms: 0,
            clock_skew_ms: 0,
        },
        MergePolicy {
            max_age_ms: 60_000,
            clock_skew_ms: 10_001,
        },
    ] {
        assert!(collect::merge(&[good(0)], bad, now).is_err());
    }
    refuse(Vec::new(), "1..=64");
}
