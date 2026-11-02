const COMMITTED_RELEASE: &[u8] = include_bytes!("../releases/regression-10-v1/release.json");
const COMMITTED_REGISTRY: &[u8] =
    include_bytes!("../releases/regression-10-v1/family-registry.json");
const FROZEN_PANEL: &[u8] = include_bytes!("../../../benchmarks/panel.json");
const PREPARATION_IMPLEMENTATION: &[u8] = include_bytes!("../../../benchmarks/prepare.py");

/// W1-T01: receivers consume the release embedded in the library, which is
/// exactly the committed document and binds the committed registry.
#[test]
fn the_embedded_release_is_the_committed_document() {
    let embedded = jope_datasets::regression_release::committed().unwrap();
    assert_eq!(
        embedded,
        RegressionRelease::open(COMMITTED_RELEASE).unwrap()
    );
    assert_eq!(
        embedded.release_id,
        jope_datasets::regression_release::COMMITTED_RELEASE_ID
    );
    assert_eq!(
        embedded.family_registry_sha256,
        FamilyRegistry::open(COMMITTED_REGISTRY)
            .unwrap()
            .sha256()
            .unwrap()
    );
}

/// Membership proof for the committed release against the unchanged frozen
/// panel. Byte-level re-derivation from the staged custody objects is the
/// `jope-datasets regression-release verify` receipt, not this CI test.
#[test]
fn committed_ten_regression_release_matches_the_frozen_panel() {
    let release = RegressionRelease::open(COMMITTED_RELEASE).unwrap();
    assert_eq!(release.encode().unwrap(), COMMITTED_RELEASE);
    assert_eq!(
        release.release_id,
        "859ce6895f6ca2b016103465f44e4f84a3dd54dcce8b2c832ba15ec536454d4a"
    );
    let registry = FamilyRegistry::open(COMMITTED_REGISTRY).unwrap();
    assert_eq!(registry.encode().unwrap(), COMMITTED_REGISTRY);
    assert_eq!(registry.sha256().unwrap(), release.family_registry_sha256);
    check_against_panel(&release, FROZEN_PANEL).unwrap();
    assert_eq!(
        release.preparation.source_sha256,
        sha256_hex(PREPARATION_IMPLEMENTATION)
    );
    let members: Vec<String> = release.members.iter().map(|m| m.dataset.clone()).collect();
    assert_eq!(members, panel_regression_members(FROZEN_PANEL).unwrap());
    assert_eq!(
        members,
        [
            "197_cpu_act",
            "225_puma8NH",
            "230_machine_cpu",
            "522_pm10",
            "529_pollen",
            "579_fri_c0_250_5",
            "auto_insurance_price",
            "690_visualizing_galaxy",
            "712_chscase_geyser1",
            "503_wind"
        ]
    );
    let binary = release.exclusions.iter().filter(|e| e.task == "binary");
    let multiclass = release.exclusions.iter().filter(|e| e.task == "multiclass");
    assert_eq!((binary.count(), multiclass.count()), (10, 10));
    assert_eq!(release.derived_from.task_counts["timeseries"], 0);
    assert_eq!(
        registry.members(),
        members
            .iter()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>()
    );
    // Historical exposure is unknown, never unseen: every never-seen claim is blocked.
    for m in &members {
        let blockers = registry.never_seen_claim_blockers(m).unwrap();
        assert!(blockers
            .iter()
            .any(|b| b.starts_with("encoder exposure unknown")));
        assert!(blockers
            .iter()
            .any(|b| b.starts_with("checkpoint exposure unknown")));
    }
}

fn write_staged(root: &Path, f: &Fixture) {
    for dir in ["downloads", "metadata", "development", "splits", "sealed"] {
        std::fs::create_dir_all(root.join(dir)).unwrap();
    }
    for (id, s) in &f.staged {
        std::fs::write(root.join("downloads").join(sha256_hex(&s.raw)), &s.raw).unwrap();
        std::fs::write(
            root.join("metadata").join(format!("{id}.yaml")),
            &s.metadata,
        )
        .unwrap();
        for (dir, bytes) in [
            ("development", &s.development),
            ("splits", &s.split),
            ("sealed", &s.sealed),
        ] {
            std::fs::write(root.join(dir).join(format!("{id}.json")), bytes).unwrap();
        }
    }
}

#[test]
fn staged_loader_reads_only_exact_regular_objects_without_fallback() {
    let f = Fixture::new();
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("prepared");
    write_staged(&root, &f);
    // Classification members have no staged objects at all; they are never opened.
    let loaded = load_staged(&root, &f.panel_bytes()).unwrap();
    assert_eq!(loaded.len(), 10);
    for (id, s) in &loaded {
        assert_eq!(s.development, f.staged[id].development);
        assert_eq!(s.raw, f.staged[id].raw);
    }

    // A symlinked leaf is refused even when it points at identical bytes.
    let dev = root.join("development/reg1.json");
    let copy = dir.path().join("elsewhere.json");
    std::fs::rename(&dev, &copy).unwrap();
    std::os::unix::fs::symlink(&copy, &dev).unwrap();
    assert!(err_text(load_staged(&root, &f.panel_bytes())).contains("symlink"));
    std::fs::remove_file(&dev).unwrap();
    std::fs::rename(&copy, &dev).unwrap();

    // A missing exact object is not replaced by a `latest` or differently named copy.
    let raw = root
        .join("downloads")
        .join(sha256_hex(&f.staged["reg2"].raw));
    std::fs::rename(&raw, root.join("downloads/latest")).unwrap();
    assert!(err_text(load_staged(&root, &f.panel_bytes())).contains("absent"));
    std::fs::rename(root.join("downloads/latest"), &raw).unwrap();

    // A different version under the exact name loads but cannot derive a release.
    std::fs::write(&raw, b"new version under the old key").unwrap();
    let substituted = load_staged(&root, &f.panel_bytes()).unwrap();
    let result = derive(&ReleaseSources {
        panel: &f.panel_bytes(),
        preparation_source_path: "benchmarks/prepare.py",
        preparation_source: &f.preparation,
        registry: &f.registry,
        staged: &substituted,
    });
    assert!(err_text(result).contains("raw object differs"));

    // A symlinked partition directory is refused.
    let moved = dir.path().join("splits-real");
    std::fs::rename(root.join("splits"), &moved).unwrap();
    std::os::unix::fs::symlink(&moved, root.join("splits")).unwrap();
    assert!(err_text(load_staged(&root, &f.panel_bytes())).contains("symlink"));
}
