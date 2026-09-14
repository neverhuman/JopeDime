use jope_parallel::{corpus::{self, ScanCacheStats}, manifest::*};
use serde_json::{json, Value};
use std::{fs, path::{Path, PathBuf}};

struct Fixture { directory: tempfile::TempDir }
impl Fixture {
    fn new() -> Self {
        let this = Self { directory: tempfile::tempdir().unwrap() };
        let root = this.root();
        fs::write(this.vocab(), fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/grammar/featmath.json")).unwrap()).unwrap();
        let mut contexts = Vec::new();
        for family in ["family-a", "family-b", "family-c"] {
            let prepared = root.join(family);
            write_json(&prepared.join("prepared.json"), &json!({
                "schema":"parallel.prepared.v1","dataset":family,"family":family,"task":"regression",
                "meta_partition":"meta_train","seed":17,"clean_evaluation":true,"clean_status":"fixture",
                "original_train_sha256":"original","original_test_sha256":"sealed",
                "selected_train_indices":[0,1],"selected_test_indices":[],"target_last":true,"rows_total":2,
                "cuts":[{"index":0,"train_indices":[0],"validation_indices":[1],"training_sha256":"train",
                    "validation_sha256":"validation","preprocessing_sha256":"preprocessing"}]})).unwrap();
            let prepared_sha = file_hash(&prepared.join("prepared.json")).unwrap();
            let path = prepared.join("context.bin"); fs::write(&path, b"frozen context fixture").unwrap();
            write_json(&path.with_extension("json"), &json!({"schema":"parallel.encoding.v1","dataset":family,
                "family":family,"partition":"meta_train","cut":0,"prepared_sha256":prepared_sha,
                "training_sha256":"train","selected_training_rows_sha256":"rows","preprocessing_sha256":"preprocessing",
                "encoder_spec_sha256":"assembly","numerical_backend":"fixture","dataset_dimensions":1536,
                "target_dimensions":512,"output_sha256":file_hash(&path).unwrap()})).unwrap();
            contexts.push(json!({"dataset":family,"family":family,"cut":0,"partition":"meta_train",
                "path":path.strip_prefix(root).unwrap(),"sha256":file_hash(&path).unwrap(),
                "prepared":family,"prepared_sha256":prepared_sha}));
            let mut rows = Vec::new();
            for (n, transform) in ["standardize","robust_scale","norm_row","row_stats","rank","blom","sigmoid","step"].iter().enumerate() {
                let genome = format!("(pipe X ({transform}) (ridge))");
                rows.push(json!({"dataset":family,"family":family,"cut":0,"partition":"meta_train",
                    "training_sha256":"train","validation_sha256":"validation","genome":genome,
                    "program_sha256":hash(genome.as_bytes()),"metrics":{"rmse":n as f64+1.},
                    "runtime_seconds":0.1,"evaluation_bias":"first","failure":null,
                    "family_partition_frozen":true,"qualification":false}));
            }
            let mut duplicate = rows[0].clone(); duplicate["metrics"]["rmse"] = json!(99.0);
            duplicate["evaluation_bias"] = json!("duplicate must not win"); rows.push(duplicate);
            let mut failed = rows[0].clone(); failed["failure"] = json!("fixture failure"); rows.push(failed);
            let mut missing = rows[0].clone(); missing["dataset"] = json!("out-of-index"); rows.push(missing);
            let mut rejected = rows[0].clone();
            let genome = "(pipe X (rff_cos n=17 gamma=0.513 seed=7) (ridge))";
            rejected["genome"] = json!(genome); rejected["program_sha256"] = json!(hash(genome.as_bytes()));
            rows.push(rejected.clone()); rows.push(rejected);
            write_json(&this.ledger(family), &rows).unwrap();
            write_json(&this.ledger(family).with_file_name("search.json"), &json!({"schema":"parallel.search.v1"})).unwrap();
        }
        write_json(&this.contexts(), &json!({"schema":"parallel.context_index.v1","qualification":false,
            "assembly_sha256":"assembly","encodings":contexts})).unwrap();
        this
    }
    fn root(&self) -> &Path { self.directory.path() }
    fn vocab(&self) -> PathBuf { self.root().join("vocabulary.json") }
    fn contexts(&self) -> PathBuf { self.root().join("contexts.json") }
    fn observations(&self) -> Vec<PathBuf> { vec![self.root().join("observations")] }
    fn ledger(&self, family: &str) -> PathBuf { self.root().join("observations").join(family).join("observations.json") }
    fn cache(&self) -> PathBuf { self.root().join("runs/experiment/scan-cache") }
    fn full(&self, name: &str, qualification: bool) {
        corpus::build(self.root(), &self.contexts(), &self.vocab(), &self.observations(), &self.root().join(name), qualification).unwrap();
    }
    fn cached(&self, name: &str, qualification: bool) -> anyhow::Result<ScanCacheStats> {
        corpus::build_observed_cached(self.root(), &self.contexts(), &self.vocab(), &self.observations(),
            &self.root().join(name), qualification, &self.cache(), |_, _, _| Ok(()))
    }
    fn equal(&self, a: &str, b: &str) {
        for name in ["records.bin", "labels.jsonl", "index.json", "provenance.json", "vocabulary.json"] {
            assert_eq!(fs::read(self.root().join(a).join(name)).unwrap(), fs::read(self.root().join(b).join(name)).unwrap(), "changed {name}");
        }
    }
    fn entries(&self) -> Vec<PathBuf> {
        let mut paths = fs::read_dir(self.cache()).unwrap().map(|e| e.unwrap().path())
            .filter(|p| !p.file_name().unwrap().to_string_lossy().starts_with('.')).collect::<Vec<_>>();
        paths.sort(); paths
    }
}

#[test]
fn reuse_is_byte_identical_with_exact_counts_first_wins_and_no_source_writes() {
    let f = Fixture::new();
    let original = f.observations().into_iter().flat_map(|p| walkdir::WalkDir::new(p).into_iter().map(|e| e.unwrap().into_path()))
        .filter(|p| p.is_file()).map(|p| { let sha=file_hash(&p).unwrap(); (p,sha) }).collect::<Vec<_>>();
    f.full("full", true);
    let first = f.cached("first", true).unwrap();
    assert_eq!(first, ScanCacheStats { ledgers_hashed:3, ledgers_parsed:3, ledgers_reused:0, entries_committed:3 });
    let second = f.cached("reuse", true).unwrap();
    assert_eq!(second, ScanCacheStats { ledgers_hashed:3, ledgers_parsed:0, ledgers_reused:3, entries_committed:0 });
    f.equal("full", "first"); f.equal("full", "reuse");
    let provenance:Value = read_json(&f.root().join("reuse/provenance.json")).unwrap();
    assert_eq!(provenance["counts"]["observations_seen"], 39);
    assert_eq!(provenance["counts"]["no_matching_context"], 3);
    assert_eq!(provenance["counts"]["failed_program"], 3);
    assert_eq!(provenance["selected_programs"], 24);
    let labels=fs::read_to_string(f.root().join("reuse/labels.jsonl")).unwrap();
    assert!(!labels.contains("duplicate must not win"));
    for (path, sha) in original { assert_eq!(file_hash(&path).unwrap(), sha); }
}

#[test]
fn changed_same_size_ledger_and_added_removed_sources_are_reconciled() {
    let f=Fixture::new(); f.cached("initial",true).unwrap();
    let path=f.ledger("family-a"); let before=fs::read(&path).unwrap();
    let mut rows:Value=serde_json::from_slice(&before).unwrap(); rows[0]["metrics"]["rmse"]=json!(8.0);
    write_json(&path,&rows).unwrap(); assert_eq!(before.len(),fs::metadata(&path).unwrap().len() as usize);
    let stats=f.cached("changed",true).unwrap(); assert_eq!(stats.ledgers_parsed,1);assert_eq!(stats.ledgers_reused,2);
    f.full("changed-full",true);f.equal("changed-full","changed");
    assert_ne!(fs::read(f.root().join("initial/records.bin")).unwrap(),fs::read(f.root().join("changed/records.bin")).unwrap());
    fs::remove_dir_all(f.ledger("family-c").parent().unwrap()).unwrap();
    let stats=f.cached("removed",true).unwrap(); assert_eq!(stats.ledgers_reused,2);assert_eq!(stats.ledgers_parsed,0);
    f.full("removed-full",true);f.equal("removed-full","removed");
    let added=f.ledger("a-first");let mut rows:Value=read_json(&f.ledger("family-a")).unwrap();
    rows[0]["metrics"]["rmse"]=json!(0.5);rows[0]["evaluation_bias"]=json!("earlier ledger wins");
    write_json(&added,&rows).unwrap();write_json(&added.with_file_name("search.json"),&json!({})).unwrap();
    let stats=f.cached("added",true).unwrap();assert_eq!(stats.ledgers_reused,2);assert_eq!(stats.ledgers_parsed,1);
    f.full("added-full",true);f.equal("added-full","added");
    assert!(fs::read_to_string(f.root().join("added/labels.jsonl")).unwrap().contains("earlier ledger wins"));
}

#[test]
fn context_vocab_qualification_search_and_source_path_bind_cache_identity() {
    let f=Fixture::new();f.cached("initial",true).unwrap();
    // Semantically identical files with different exact identities cannot hit.
    for (i,path) in [f.contexts(),f.vocab()].into_iter().enumerate() {
        use std::io::Write;
        fs::OpenOptions::new().append(true).open(path).unwrap().write_all(b"\n").unwrap();
        let stats=f.cached(&format!("binding-{i}"),true).unwrap();assert_eq!(stats.ledgers_parsed,3);assert_eq!(stats.ledgers_reused,0);
        f.full(&format!("binding-full-{i}"),true);f.equal(&format!("binding-full-{i}"),&format!("binding-{i}"));
    }
    let stats=f.cached("clean",false).unwrap();assert_eq!(stats.ledgers_parsed,3);assert_eq!(stats.ledgers_reused,0);
    f.full("clean-full",false);f.equal("clean","clean-full");
    write_json(&f.ledger("family-a").with_file_name("search.json"),&json!({"changed":true})).unwrap();
    let stats=f.cached("search-changed",false).unwrap();assert_eq!(stats.ledgers_parsed,1);assert_eq!(stats.ledgers_reused,2);
    fs::rename(f.ledger("family-a").parent().unwrap(),f.ledger("renamed-family").parent().unwrap()).unwrap();
    let stats=f.cached("path-changed",false).unwrap();assert_eq!(stats.ledgers_parsed,1);assert_eq!(stats.ledgers_reused,2);
    f.full("path-changed-full",false);f.equal("path-changed-full","path-changed");
}

#[test]
fn altered_context_or_prepared_artifacts_are_rejected_before_cache_reuse() {
    for artifact in ["context.bin","prepared.json"] {
        let f=Fixture::new();f.cached("initial",true).unwrap();
        fs::write(f.root().join("family-a").join(artifact),b"changed").unwrap();
        assert!(f.cached("must-fail",true).is_err());
        assert!(!f.root().join("must-fail").exists());
    }
}

#[test]
fn cache_corruption_and_misbound_manifest_are_errors_not_silent_labels() {
    for mode in ["payload", "manifest", "missing"] {
        let f=Fixture::new();f.cached("initial",true).unwrap();let entry=&f.entries()[0];
        match mode {
            "payload" => {
                let p=entry.join("contribution.json");let mut rows:Value=read_json(&p).unwrap();
                rows["selected"]["0"]=json!({});write_json(&p,&rows).unwrap();
            },
            "manifest" => {
                let p=entry.join("manifest.json");let mut m:Value=read_json(&p).unwrap();
                m["binding"]["source_sha256"]=json!("wrong");write_json(&p,&m).unwrap();
            },
            _ => fs::remove_file(entry.join("contribution.json")).unwrap(),
        }
        assert!(f.cached("must-fail",true).is_err(), "accepted {mode}");
        assert!(!f.root().join("must-fail").exists());
    }
}

#[test]
fn duplicate_rows_still_validate_and_invalid_changed_scores_never_hit_cache() {
    let f=Fixture::new();f.cached("initial",true).unwrap();let path=f.ledger("family-a");
    let mut rows:Value=read_json(&path).unwrap();rows[8]["program_sha256"]=json!("invalid duplicate");
    write_json(&path,&rows).unwrap();assert!(f.cached("bad-duplicate",true).is_err());
    rows[8]["program_sha256"]=rows[0]["program_sha256"].clone();rows[0]["metrics"]["rmse"]=json!(-1.0);
    write_json(&path,&rows).unwrap();assert!(f.cached("bad-score",true).is_err());
}

#[cfg(unix)]
#[test]
fn cache_and_source_symlinks_and_overlapping_namespaces_are_rejected() {
    use std::os::unix::fs::symlink;
    let f=Fixture::new();fs::create_dir_all(f.root().join("elsewhere")).unwrap();
    fs::create_dir_all(f.root().join("runs/experiment")).unwrap();symlink(f.root().join("elsewhere"),f.cache()).unwrap();
    assert!(f.cached("linked-cache",true).is_err());fs::remove_file(f.cache()).unwrap();
    let source=f.ledger("family-a");fs::rename(&source,source.with_extension("original")).unwrap();
    symlink(source.with_extension("original"),&source).unwrap();assert!(f.cached("linked-source",true).is_err());
    fs::remove_file(&source).unwrap();fs::rename(source.with_extension("original"),source).unwrap();
    f.cached("initial",true).unwrap();let entry=f.entries()[0].clone();let payload=entry.join("contribution.json");
    fs::rename(&payload,payload.with_extension("original")).unwrap();symlink(payload.with_extension("original"),&payload).unwrap();
    assert!(f.cached("linked-payload",true).is_err());
    for cache in [f.root().join("runs/global"),f.root().join("runs/experiment/output/cache")] {
        assert!(corpus::build_observed_cached(f.root(),&f.contexts(),&f.vocab(),&f.observations(),
            &f.root().join("runs/experiment/output"),true,&cache,|_,_,_|Ok(())).is_err());
    }
}

#[test]
fn canceled_cache_commits_resume_without_adopting_partial_contributions() {
    for boundary in ["cache payload durable","cache manifest durable","cache directory durable","cache entry renamed","cache entry committed"] {
        let f=Fixture::new();f.full("full",true);
        assert!(corpus::build_observed_cached(f.root(),&f.contexts(),&f.vocab(),&f.observations(),
            &f.root().join("canceled"),true,&f.cache(),|operation,_,_| {
                if operation==boundary { anyhow::bail!("injected cancellation"); } Ok(())
            }).is_err());
        assert!(!f.root().join("canceled").exists());
        let partial=f.cache().join(".partial-interrupted-process");fs::create_dir_all(&partial).unwrap();
        fs::write(partial.join("contribution.json"),b"{truncated").unwrap();
        let stats=f.cached("resumed",true).unwrap();
        let committed=usize::from(boundary=="cache entry renamed" || boundary=="cache entry committed");
        assert_eq!(stats.ledgers_reused,committed);assert_eq!(stats.ledgers_parsed,3-committed);
        f.equal("full","resumed");assert!(partial.exists());
    }
}

#[cfg(unix)]
#[test]
fn sigkill_at_each_cache_durable_boundary_keeps_only_complete_entries() {
    use std::{os::unix::process::ExitStatusExt, process::Command};
    for boundary in ["cache payload durable","cache manifest durable","cache directory durable","cache entry renamed","cache entry committed"] {
        let f=Fixture::new();f.full("full",true);
        let status=Command::new(std::env::current_exe().unwrap()).args(["--exact","cache_kill_child","--ignored"])
            .env("JOPE_CORPUS_KILL_ROOT",f.root()).env("JOPE_CORPUS_KILL_BOUNDARY",boundary)
            .stdout(std::process::Stdio::null()).status().unwrap();
        assert_eq!(status.signal(),Some(libc::SIGKILL),"child did not reach {boundary}");
        assert!(!f.root().join("killed").exists());
        let stats=f.cached("resumed",true).unwrap();
        let committed=usize::from(boundary=="cache entry renamed" || boundary=="cache entry committed");
        assert_eq!(stats.ledgers_reused,committed);f.equal("full","resumed");
    }
}

#[cfg(unix)]
#[test]
#[ignore = "subprocess fixture; parent verifies SIGKILL at each durable boundary"]
fn cache_kill_child() {
    let root=PathBuf::from(std::env::var_os("JOPE_CORPUS_KILL_ROOT").unwrap());
    let boundary=std::env::var("JOPE_CORPUS_KILL_BOUNDARY").unwrap();
    corpus::build_observed_cached(&root,&root.join("contexts.json"),&root.join("vocabulary.json"),
        &[root.join("observations")],&root.join("killed"),true,&root.join("runs/experiment/scan-cache"),
        |operation,_,_| {
            if operation==boundary { unsafe { libc::kill(libc::getpid(),libc::SIGKILL); } }
            Ok(())
        }).unwrap();
    panic!("kill boundary was never reached");
}

#[test]
#[ignore = "explicit local scan diagnostic; synthetic repeated ledgers, not deployment throughput"]
fn repeated_ledger_scan_diagnostic() {
    let f=Fixture::new();
    for family in ["family-a","family-b","family-c"] {
        let path=f.ledger(family);let rows:Vec<Value>=read_json(&path).unwrap();
        let expanded=(0..1000).flat_map(|_|rows.clone()).collect::<Vec<_>>();
        write_json(&path,&expanded).unwrap();
    }
    let source_bytes=["family-a","family-b","family-c"].iter().map(|family|fs::metadata(f.ledger(family)).unwrap().len()).sum::<u64>();
    let start=std::time::Instant::now();f.full("full",true);let full_seconds=start.elapsed().as_secs_f64();
    let start=std::time::Instant::now();let cold=f.cached("cold",true).unwrap();let cold_seconds=start.elapsed().as_secs_f64();
    let start=std::time::Instant::now();let warm=f.cached("warm",true).unwrap();let warm_seconds=start.elapsed().as_secs_f64();
    f.equal("full","cold");f.equal("full","warm");
    assert_eq!(warm.ledgers_parsed,0);assert_eq!(warm.ledgers_reused,3);
    println!("{}",json!({"diagnostic":"synthetic repeated ledgers; local build; all source bytes still SHA256-checked",
        "observations":39000,"selected_programs":24,"source_bytes":source_bytes,
        "uncached_seconds":full_seconds,"cold_cache_seconds":cold_seconds,"warm_cache_seconds":warm_seconds,
        "cold_stats":cold,"warm_stats":warm,"byte_identical":true}));
}

#[test]
fn ledger_directory_names_never_replace_row_identity_or_drop_provenance() {
    let f=Fixture::new();
    // This deliberately misleading source path must not suppress valid rows.
    let renamed=f.root().join("observations/not-the-dataset/cut-999");
    fs::create_dir_all(renamed.parent().unwrap()).unwrap();
    fs::rename(f.ledger("family-a").parent().unwrap(),&renamed).unwrap();
    f.full("full",true);let stats=f.cached("cold",true).unwrap();assert_eq!(stats.ledgers_parsed,3);
    let stats=f.cached("warm",true).unwrap();assert_eq!(stats.ledgers_reused,3);f.equal("full","cold");f.equal("full","warm");
    let provenance:Value=read_json(&f.root().join("warm/provenance.json")).unwrap();
    assert_eq!(provenance["counts"]["observations_seen"],39);assert_eq!(provenance["selected_programs"],24);
    assert!(provenance["counts"]["skipped_ledger"].is_null());
    assert!(!f.root().join("runs/corpus-progress.json").exists(),"cache builder wrote outside its private namespace");
}

#[test]
#[ignore = "requires independently built immutable pre-cache baseline executable"]
fn independent_legacy_binary_has_identical_artifact_bytes() {
    let binary=PathBuf::from(std::env::var_os("JOPE_CORPUS_BASELINE_BIN").expect("baseline executable absent"));
    let f=Fixture::new();
    for mode in ["all-train", "withheld"] {
        if mode=="withheld" {
            let mut contexts:Value=read_json(&f.contexts()).unwrap();
            contexts["encodings"][2]["partition"]=json!("meta_test");write_json(&f.contexts(),&contexts).unwrap();
        }
        let legacy=format!("legacy-{mode}");let warm=format!("warm-{mode}");
        let status=std::process::Command::new(&binary).arg("--root").arg(f.root()).arg("build-corpus")
            .arg("--contexts").arg(f.contexts()).arg("--vocabulary").arg(f.vocab())
            .arg("--observations").arg(&f.observations()[0]).arg("--output").arg(f.root().join(&legacy))
            .arg("--qualification").status().unwrap();
        assert!(status.success());
        f.cached(&format!("cold-{mode}"),true).unwrap();let stats=f.cached(&warm,true).unwrap();
        assert_eq!(stats.ledgers_reused,3);assert_eq!(stats.ledgers_parsed,0);
        f.equal(&legacy,&warm);
    }
}
