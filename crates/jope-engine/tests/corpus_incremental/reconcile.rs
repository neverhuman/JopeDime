use super::*;

#[test]
fn canceled_cache_commits_resume_without_adopting_partial_contributions() {
    for boundary in [
        "cache payload durable",
        "cache manifest durable",
        "cache directory durable",
        "cache entry renamed",
        "cache entry committed",
    ] {
        let f = Fixture::new();
        f.full("full", true);
        assert!(corpus::build_observed_cached(
            f.root(),
            jope_parallel::corpus::CorpusSources {
                contexts_path: &f.contexts(),
                vocab_path: &f.vocab(),
                observations: &f.observations()
            },
            &f.root().join("canceled"),
            true,
            &f.cache(),
            |operation, _, _| {
                if operation == boundary {
                    anyhow::bail!("injected cancellation");
                }
                Ok(())
            },
        )
        .is_err());
        assert!(!f.root().join("canceled").exists());
        let partial = f.cache().join(".partial-interrupted-process");
        fs::create_dir_all(&partial).unwrap();
        fs::write(partial.join("contribution.json"), b"{truncated").unwrap();
        let stats = f.cached("resumed", true).unwrap();
        let committed =
            usize::from(boundary == "cache entry renamed" || boundary == "cache entry committed");
        assert_eq!(stats.ledgers_reused, committed);
        assert_eq!(stats.ledgers_parsed, 3 - committed);
        f.equal("full", "resumed");
        assert!(partial.exists());
    }
}

#[cfg(unix)]
#[test]
fn sigkill_at_each_cache_durable_boundary_keeps_only_complete_entries() {
    use std::{os::unix::process::ExitStatusExt, process::Command};
    for boundary in [
        "cache payload durable",
        "cache manifest durable",
        "cache directory durable",
        "cache entry renamed",
        "cache entry committed",
    ] {
        let f = Fixture::new();
        f.full("full", true);
        let status = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "reconcile::cache_kill_child", "--ignored"])
            .env("JOPE_CORPUS_KILL_ROOT", f.root())
            .env("JOPE_CORPUS_KILL_BOUNDARY", boundary)
            .stdout(std::process::Stdio::null())
            .status()
            .unwrap();
        assert_eq!(
            status.signal(),
            Some(libc::SIGKILL),
            "child did not reach {boundary}"
        );
        assert!(!f.root().join("killed").exists());
        let stats = f.cached("resumed", true).unwrap();
        let committed =
            usize::from(boundary == "cache entry renamed" || boundary == "cache entry committed");
        assert_eq!(stats.ledgers_reused, committed);
        f.equal("full", "resumed");
    }
}

#[cfg(unix)]
#[test]
#[ignore = "subprocess fixture; parent verifies SIGKILL at each durable boundary"]
fn cache_kill_child() {
    let root = PathBuf::from(std::env::var_os("JOPE_CORPUS_KILL_ROOT").unwrap());
    let boundary = std::env::var("JOPE_CORPUS_KILL_BOUNDARY").unwrap();
    corpus::build_observed_cached(
        &root,
        jope_parallel::corpus::CorpusSources {
            contexts_path: &root.join("contexts.json"),
            vocab_path: &root.join("vocabulary.json"),
            observations: &[root.join("observations")],
        },
        &root.join("killed"),
        true,
        &root.join("runs/experiment/scan-cache"),
        |operation, _, _| {
            if operation == boundary {
                unsafe {
                    libc::kill(libc::getpid(), libc::SIGKILL);
                }
            }
            Ok(())
        },
    )
    .unwrap();
    panic!("kill boundary was never reached");
}

#[test]
#[ignore = "explicit local scan diagnostic; synthetic repeated ledgers, not deployment throughput"]
fn repeated_ledger_scan_diagnostic() {
    let f = Fixture::new();
    for family in ["family-a", "family-b", "family-c"] {
        let path = f.ledger(family);
        let rows: Vec<Value> = read_json(&path).unwrap();
        let expanded = (0..1000).flat_map(|_| rows.clone()).collect::<Vec<_>>();
        write_json(&path, &expanded).unwrap();
    }
    let source_bytes = ["family-a", "family-b", "family-c"]
        .iter()
        .map(|family| fs::metadata(f.ledger(family)).unwrap().len())
        .sum::<u64>();
    let start = std::time::Instant::now();
    f.full("full", true);
    let full_seconds = start.elapsed().as_secs_f64();
    let start = std::time::Instant::now();
    let cold = f.cached("cold", true).unwrap();
    let cold_seconds = start.elapsed().as_secs_f64();
    let start = std::time::Instant::now();
    let warm = f.cached("warm", true).unwrap();
    let warm_seconds = start.elapsed().as_secs_f64();
    f.equal("full", "cold");
    f.equal("full", "warm");
    assert_eq!(warm.ledgers_parsed, 0);
    assert_eq!(warm.ledgers_reused, 3);
    println!(
        "{}",
        json!({"diagnostic":"synthetic repeated ledgers; local build; all source bytes still SHA256-checked",
        "observations":39000,"selected_programs":24,"source_bytes":source_bytes,
        "uncached_seconds":full_seconds,"cold_cache_seconds":cold_seconds,"warm_cache_seconds":warm_seconds,
        "cold_stats":cold,"warm_stats":warm,"byte_identical":true})
    );
}

#[test]
fn ledger_directory_names_never_replace_row_identity_or_drop_provenance() {
    let f = Fixture::new();
    // This deliberately misleading source path must not suppress valid rows.
    let renamed = f.root().join("observations/not-the-dataset/cut-999");
    fs::create_dir_all(renamed.parent().unwrap()).unwrap();
    fs::rename(f.ledger("family-a").parent().unwrap(), &renamed).unwrap();
    f.full("full", true);
    let stats = f.cached("cold", true).unwrap();
    assert_eq!(stats.ledgers_parsed, 3);
    let stats = f.cached("warm", true).unwrap();
    assert_eq!(stats.ledgers_reused, 3);
    f.equal("full", "cold");
    f.equal("full", "warm");
    let provenance: Value = read_json(&f.root().join("warm/provenance.json")).unwrap();
    assert_eq!(provenance["counts"]["observations_seen"], 39);
    assert_eq!(provenance["selected_programs"], 24);
    assert!(provenance["counts"]["skipped_ledger"].is_null());
    assert!(
        !f.root().join("runs/corpus-progress.json").exists(),
        "cache builder wrote outside its private namespace"
    );
}

#[test]
fn cpu_cli_marks_verified_publication_completed() {
    let f = Fixture::new();
    let progress = f.root().join("private-progress.json");
    let output = f.root().join("cli-corpus");
    let status = std::process::Command::new(env!("CARGO_BIN_EXE_jope-parallel"))
        .arg("--root")
        .arg(f.root())
        .arg("build-corpus")
        .arg("--contexts")
        .arg(f.contexts())
        .arg("--vocabulary")
        .arg(f.vocab())
        .arg("--observations")
        .arg(&f.observations()[0])
        .arg("--output")
        .arg(&output)
        .arg("--scan-cache")
        .arg(f.root().join("runs/cli-test/scan-cache"))
        .arg("--progress")
        .arg(&progress)
        .args(["--qualification", "--label-policy", "reference_v1"])
        .status()
        .unwrap();
    assert!(status.success());
    let saved: Value = read_json(&progress).unwrap();
    assert_eq!(saved["phase"], "completed");
    assert_eq!(saved["complete"], true);
    assert!(output.join("provenance.json").is_file());
}

#[test]
#[ignore = "requires independently built immutable pre-cache baseline executable"]
fn independent_legacy_binary_has_identical_artifact_bytes() {
    let binary = PathBuf::from(
        std::env::var_os("JOPE_CORPUS_BASELINE_BIN").expect("baseline executable absent"),
    );
    let f = Fixture::new();
    for mode in ["all-train", "withheld"] {
        if mode == "withheld" {
            let mut contexts: Value = read_json(&f.contexts()).unwrap();
            contexts["encodings"][2]["partition"] = json!("meta_test");
            write_json(&f.contexts(), &contexts).unwrap();
        }
        let legacy = format!("legacy-{mode}");
        let warm = format!("warm-{mode}");
        let status = std::process::Command::new(&binary)
            .arg("--root")
            .arg(f.root())
            .arg("build-corpus")
            .arg("--contexts")
            .arg(f.contexts())
            .arg("--vocabulary")
            .arg(f.vocab())
            .arg("--observations")
            .arg(&f.observations()[0])
            .arg("--output")
            .arg(f.root().join(&legacy))
            .arg("--qualification")
            .status()
            .unwrap();
        assert!(status.success());
        f.cached(&format!("cold-{mode}"), true).unwrap();
        let stats = f.cached(&warm, true).unwrap();
        assert_eq!(stats.ledgers_reused, 3);
        assert_eq!(stats.ledgers_parsed, 0);
        f.equal(&legacy, &warm);
    }
}
