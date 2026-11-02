use super::*;
use crate::{clock::Clock, emit::Language, run::Policy, search::Budget};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

fn dataset() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../datasets/test/pmlb-229-pwlinear"
    )
    .to_owned()
}

fn template(evaluations: usize) -> RunOptions {
    RunOptions {
        budget: Budget {
            evaluations,
            seconds: 600,
        },
        rng_seed: 7,
        languages: vec![Language::Rust],
        policy: Policy::Skip,
        ..RunOptions::new(dataset(), PathBuf::new())
    }
}

fn evolve_into(output: &Path, rounds: usize, control: bool, run: RunOptions) -> Result<Lineage> {
    evolve(&EvolveOptions {
        run,
        output: output.to_path_buf(),
        rounds,
        control,
        seeds: None,
    })
}

/// Every file under `root`, by relative path.
fn tree(root: &Path) -> std::collections::BTreeMap<String, Vec<u8>> {
    fn walk(root: &Path, dir: &Path, out: &mut std::collections::BTreeMap<String, Vec<u8>>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                walk(root, &path, out);
            } else {
                out.insert(
                    path.strip_prefix(root).unwrap().display().to_string(),
                    std::fs::read(&path).unwrap(),
                );
            }
        }
    }
    let mut out = std::collections::BTreeMap::new();
    walk(root, root, &mut out);
    out
}

/// Every entry under `root`, directories included, by relative path (a file with its bytes).
fn entries(root: &Path) -> Vec<(String, Vec<u8>)> {
    let mut out = Vec::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            let bytes = if path.is_dir() {
                pending.push(path.clone());
                Vec::new()
            } else {
                std::fs::read(&path).unwrap()
            };
            out.push((
                path.strip_prefix(root).unwrap().display().to_string(),
                bytes,
            ));
        }
    }
    out.sort();
    out
}

#[test]
fn rounds_chain_on_identical_cuts_never_get_worse_and_verify() {
    let temp = tempfile::tempdir().unwrap();
    let out = temp.path().join("lineage");
    let lineage = evolve_into(&out, 3, true, template(24)).unwrap();
    assert_eq!(lineage.rounds.len(), 3);
    assert_eq!(lineage.rounds[0].kind, UNSEEDED);
    assert_eq!(lineage.rounds[0].seed_origin, FROM_NONE);
    for pair in lineage.rounds.windows(2) {
        assert!(pair[1].validation_rmse <= pair[0].validation_rmse);
        assert_eq!(
            pair[1].parent_receipt_sha256.as_deref(),
            Some(pair[0].receipt_sha256.as_str())
        );
        assert_eq!(pair[1].seed_origin, FROM_PREVIOUS_ROUND);
        assert_ne!(pair[1].rng_seed, pair[0].rng_seed);
    }
    let control = lineage.control.as_ref().unwrap();
    assert_eq!(control.kind, CONTROL);
    assert!(control.seeds.is_empty() && control.parent_receipt_sha256.is_none());
    assert_eq!(control.rng_seed, lineage.rounds[2].rng_seed);
    // Round 2 starts from round 1's champion and ranked runners-up, nothing else.
    let first = evidence::verify_run(&out.join("round-001/receipt.json")).unwrap();
    let expected: Vec<String> = seeds_after(&first)
        .unwrap()
        .into_iter()
        .map(|c| c.genome)
        .filter(|g| g != search::BOOTSTRAP_GENOME)
        .collect();
    assert_eq!(
        lineage.rounds[1]
            .seeds
            .iter()
            .map(|s| s.genome.clone())
            .collect::<Vec<_>>(),
        expected
    );
    // Per-generation best-so-far never rises, and every round measured bred candidates.
    for round in &lineage.rounds {
        for cut in &round.cuts {
            let values: Vec<f64> = cut.best_so_far.iter().flatten().copied().collect();
            assert!(values.windows(2).all(|w| w[1] <= w[0]), "{values:?}");
        }
        assert!(round.generation_ge1_measured > 0);
    }
    assert_eq!(read_lineage(&out.join(LINEAGE_FILE)).unwrap(), lineage);
    let report = verify(&out.join(LINEAGE_FILE), true).unwrap();
    assert_eq!(report.rounds, 3);
    assert!(report.bred_candidates_in_every_round);
    // What the rounds found is reported per round, and agrees with the RMSEs.
    assert_eq!(
        report.champion_found_in_round,
        lineage
            .rounds
            .iter()
            .map(Round::champion_found_in_round)
            .collect::<Vec<_>>()
    );
    assert_eq!(report.improved_on_previous_round.len(), 2);
    for (index, improved) in report.improved_on_previous_round.iter().enumerate() {
        let (before, after) = (&lineage.rounds[index], &lineage.rounds[index + 1]);
        assert_eq!(*improved, after.validation_rmse < before.validation_rmse);
        // Never worse, so a round that did not improve tied its parent exactly.
        if !*improved {
            assert_eq!(after.validation_rmse, before.validation_rmse);
        }
    }
    assert_eq!(
        report.control_validation_rmse,
        Some(control.validation_rmse)
    );
    // The compute the control comparison rests on: the rounds' total against one round's.
    assert_eq!(
        report.rounds_completed_evaluations,
        lineage
            .rounds
            .iter()
            .map(|r| r.completed_evaluations)
            .sum::<u64>()
    );
    assert_eq!(
        report.control_completed_evaluations,
        Some(control.completed_evaluations)
    );
    assert!(report.rounds_completed_evaluations > control.completed_evaluations);
}

#[test]
fn a_tampered_lineage_round_receipt_or_seeds_file_fails_verification() {
    let temp = tempfile::tempdir().unwrap();
    let out = temp.path().join("lineage");
    evolve_into(&out, 2, false, template(16)).unwrap();
    let path = out.join(LINEAGE_FILE);
    verify(&path, false).unwrap();
    let expect_refused = |needle: &str| {
        let error = format!("{:#}", verify(&path, false).unwrap_err());
        assert!(error.contains(needle), "wanted {needle:?}: {error}");
    };

    let pristine = std::fs::read(&path).unwrap();
    let mut value: serde_json::Value = serde_json::from_slice(&pristine).unwrap();
    let rmse = value["rounds"][1]["validation_rmse"].as_f64().unwrap();
    value["rounds"][1]["validation_rmse"] = (rmse / 2.0).into();
    std::fs::write(&path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
    expect_refused("does not match what its receipt and evidence show");
    std::fs::write(&path, &pristine).unwrap();

    // The island model is lineage-wide: a lineage that claims another one than its rounds
    // were searched with is refused, not reported.
    for (field, claimed) in [("islands", 3u64), ("migration_interval", 1)] {
        let mut value: serde_json::Value = serde_json::from_slice(&pristine).unwrap();
        assert_ne!(value[field].as_u64(), Some(claimed));
        value[field] = claimed.into();
        std::fs::write(&path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
        expect_refused("another island model than the lineage's");
        std::fs::write(&path, &pristine).unwrap();
    }

    let seeds = out.join("round-002/seeds.json");
    let original = std::fs::read(&seeds).unwrap();
    std::fs::write(&seeds, [original.as_slice(), b" "].concat()).unwrap();
    expect_refused("seeds.json is not the file");
    std::fs::write(&seeds, &original).unwrap();

    let observations = out.join("round-001/search/cut-0/observations.json");
    let original = std::fs::read(&observations).unwrap();
    std::fs::write(&observations, [original.as_slice(), b" "].concat()).unwrap();
    expect_refused("round 1");
    std::fs::write(&observations, &original).unwrap();

    verify(&path, false).unwrap();
}

/// `--threads 1` and `--threads 8` write the same lineage: every round directory and
/// `lineage.json`, byte for byte, with the wall clock frozen (so the evaluation budget, never
/// the seconds bound, ends every phase; a binding seconds bound is timing-dependent).
#[test]
fn one_and_eight_threads_write_byte_identical_lineages() {
    let temp = tempfile::tempdir().unwrap();
    let write = |threads: usize, name: &str| {
        let out = temp.path().join(name);
        evolve_into(
            &out,
            2,
            true,
            RunOptions {
                threads,
                clock: Clock::Frozen,
                ..template(20)
            },
        )
        .unwrap();
        tree(&out)
    };
    let workers = crate::parallel::affinity_cpus()
        .unwrap()
        .min(std::thread::available_parallelism().map_or(1, std::num::NonZero::get))
        .min(8);
    if workers < 2 {
        eprintln!("skipped: this host admits one worker");
        return;
    }
    let one = write(1, "one");
    let many = write(workers, "many");
    assert_eq!(
        one.keys().collect::<Vec<_>>(),
        many.keys().collect::<Vec<_>>()
    );
    for (path, bytes) in &one {
        assert!(
            many[path] == *bytes,
            "{path} differs between 1 and {workers} threads"
        );
    }
    assert!(one.contains_key(LINEAGE_FILE) && one.contains_key("control/receipt.json"));
}

/// A stop request during round 2 (SIGTERM in the CLI) leaves round 2 without a receipt and
/// the lineage with round 1 only, which still verifies.
#[test]
fn a_stop_request_during_a_round_keeps_only_completed_rounds() {
    let temp = tempfile::tempdir().unwrap();
    let out = temp.path().join("lineage");
    let flag = Arc::new(AtomicBool::new(false));
    let watcher = {
        let flag = flag.clone();
        let prepared = out.join("round-002/prepared/prepared.json");
        std::thread::spawn(move || {
            while !prepared.exists() {
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            flag.store(true, Ordering::SeqCst);
        })
    };
    let error = evolve_into(
        &out,
        3,
        true,
        RunOptions {
            cancel: Some(flag),
            ..template(30)
        },
    )
    .unwrap_err();
    watcher.join().unwrap();
    assert!(crate::parallel::is_cancelled(&error), "{error:#}");
    let lineage = read_lineage(&out.join(LINEAGE_FILE)).unwrap();
    assert_eq!(lineage.rounds.len(), 1);
    assert!(lineage.control.is_none());
    assert!(out.join("round-001/receipt.json").exists());
    assert!(!out.join("round-002/receipt.json").exists());
    assert!(!out.join("round-003").exists() && !out.join(CONTROL_DIRECTORY).exists());
    verify(&out.join(LINEAGE_FILE), false).unwrap();
}

/// A round whose champion is one of its own seeds found nothing better, however many bred
/// candidates it measured; the report must say so rather than call it a discovery.
#[test]
fn a_round_whose_champion_is_its_seed_is_not_reported_as_a_discovery() {
    let temp = tempfile::tempdir().unwrap();
    // A strong pipeline for this dataset, against 3 bred candidates in generation 1.
    let genome =
        seeds::canonical_pipeline("(pipe X (rank append=true) (gpr m=400 steps=60))").unwrap();
    let file = temp.path().join("strong.json");
    seeds::write(
        &file,
        &seeds::SeedsFile::new(
            None,
            vec![seeds::SeedGenome {
                program_sha256: sha256_hex(genome.as_bytes()),
                genome: genome.clone(),
                bank_entry_sha256: None,
                receipt_sha256: None,
            }],
        ),
    )
    .unwrap();
    let out = temp.path().join("lineage");
    let lineage = evolve(&EvolveOptions {
        run: template(16),
        output: out.clone(),
        rounds: 1,
        control: false,
        seeds: Some(file),
    })
    .unwrap();
    let round = &lineage.rounds[0];
    assert_eq!(
        round.selected_genome, genome,
        "the seed was beaten after all"
    );
    assert!(round.generation_ge1_measured > 0);
    assert!(!round.champion_found_in_round());
    assert_eq!(round.champion_origin(), "carried from its seeds");
    // The bred-candidates check still holds (the search ran past generation 0), and the
    // report says, separately, that nothing it bred won.
    let report = serde_json::to_value(verify(&out.join(LINEAGE_FILE), true).unwrap()).unwrap();
    assert_eq!(report["bred_candidates_in_every_round"], true, "{report}");
    assert_eq!(
        report["champion_found_in_round"],
        serde_json::json!([false]),
        "{report}"
    );
    assert_eq!(report["improved_on_previous_round"], serde_json::json!([]));
    assert!(report.get("discovery_in_every_round").is_none(), "{report}");
}

#[test]
fn round_one_can_start_from_a_seeds_file() {
    let temp = tempfile::tempdir().unwrap();
    let genome = seeds::canonical_pipeline("(pipe X (standardize) (ridge_gcv))").unwrap();
    let file = temp.path().join("bank-export.json");
    let sha = seeds::write(
        &file,
        &seeds::SeedsFile::new(
            None,
            vec![seeds::SeedGenome {
                program_sha256: sha256_hex(genome.as_bytes()),
                genome: genome.clone(),
                bank_entry_sha256: None,
                receipt_sha256: None,
            }],
        ),
    )
    .unwrap();
    let out = temp.path().join("lineage");
    let lineage = evolve(&EvolveOptions {
        run: template(16),
        output: out.clone(),
        rounds: 1,
        control: false,
        seeds: Some(file),
    })
    .unwrap();
    assert_eq!(lineage.seeds_file_sha256.as_deref(), Some(sha.as_str()));
    let round = &lineage.rounds[0];
    assert_eq!(
        (round.kind.as_str(), round.seed_origin.as_str()),
        (SEEDED, FROM_SEEDS_FILE)
    );
    assert_eq!(round.seeds[0].genome, genome);
    verify(&out.join(LINEAGE_FILE), false).unwrap();
}

/// A copy of the committed regression fixture whose catalog row says `binary`.
fn binary_dataset(root: &Path) -> String {
    let source = Path::new(&dataset()).to_path_buf();
    let target = root.join("binary");
    std::fs::create_dir_all(&target).unwrap();
    for name in ["train.csv", "test.csv"] {
        std::fs::copy(source.join(name), target.join(name)).unwrap();
    }
    let mut row: serde_json::Value =
        serde_json::from_slice(&std::fs::read(source.join("metadata.json")).unwrap()).unwrap();
    row["task"] = "binary".into();
    row["target"]["type"] = "binary".into();
    std::fs::write(target.join("metadata.json"), row.to_string()).unwrap();
    target.display().to_string()
}

/// Every refusal a round could make is made before evolve writes anything: a seed that is
/// not a pipeline, an unresolvable or non-regression dataset, a budget that cannot re-measure
/// round 1's seeds or the seeds a later round carries. Nothing is left behind, so the
/// corrected command runs into the same --output.
#[test]
fn a_refused_lineage_writes_nothing_and_the_corrected_retry_runs() {
    let temp = tempfile::tempdir().unwrap();
    let out = temp.path().join("lineage");
    let seeds_of = |name: &str, genomes: &[&str]| {
        let path = temp.path().join(name);
        seeds::write(
            &path,
            &seeds::SeedsFile::new(
                None,
                genomes
                    .iter()
                    .map(|g| seeds::SeedGenome {
                        genome: (*g).to_owned(),
                        program_sha256: sha256_hex(g.as_bytes()),
                        bank_entry_sha256: None,
                        receipt_sha256: None,
                    })
                    .collect(),
            ),
        )
        .unwrap();
        path
    };
    let not_a_pipeline = seeds_of("sym.json", &["(sym affine \"x0\")"]);
    let four = seeds_of(
        "four.json",
        &[
            "(pipe X (standardize) (ridge))",
            "(pipe X (rank) (ridge))",
            "(pipe X (blom) (ridge))",
            "(pipe X (robust_scale) (ridge))",
        ],
    );
    let refused = |options: EvolveOptions, needle: &str| {
        let error = format!("{:#}", evolve(&options).unwrap_err());
        assert!(error.contains(needle), "wanted {needle:?}: {error}");
        assert!(
            !out.exists(),
            "a refusal ({needle}) left {} behind",
            out.display()
        );
    };
    let options = |run: RunOptions, rounds: usize, seeds: Option<PathBuf>| EvolveOptions {
        run,
        output: out.clone(),
        rounds,
        control: false,
        seeds,
    };
    refused(
        options(template(16), 2, Some(not_a_pipeline)),
        "seed \"(sym affine",
    );
    refused(
        options(
            RunOptions {
                dataset: "0000000000000000".to_owned(),
                ..template(16)
            },
            2,
            None,
        ),
        "not in the local dataset cache",
    );
    refused(
        options(
            RunOptions {
                dataset: binary_dataset(temp.path()),
                ..template(16)
            },
            2,
            None,
        ),
        "task binary is unsupported",
    );
    refused(
        options(template(4), 1, Some(four.clone())),
        "cannot re-measure 4 seed(s)",
    );
    refused(
        options(
            RunOptions {
                cuts: 2,
                ..template(8)
            },
            2,
            None,
        ),
        "every round after the first",
    );
    // The corrected command runs into the same directory.
    let lineage = evolve(&options(template(16), 2, Some(four))).unwrap();
    assert_eq!(lineage.rounds.len(), 2);
    verify(&out.join(LINEAGE_FILE), false).unwrap();
}

/// The PR #118 review's reproduction: `--rounds 2 --cuts 6` was refused by the preparation
/// inside round 1, after `round-001/` existed, so the corrected `--cuts 3` into the same
/// `--output` failed with "already holds a lineage". A cut count outside the partition (0 or
/// 6) and a dataset that resolves and parses but cannot be cut are now refused before
/// anything is written, and the corrected command runs.
#[test]
fn a_refused_cut_count_leaves_nothing_and_the_corrected_retry_runs() {
    let temp = tempfile::tempdir().unwrap();
    let out = temp.path().join("lineage");
    let refused = |run: RunOptions, needle: &str| {
        let error = format!("{:#}", evolve_into(&out, 2, false, run).unwrap_err());
        assert!(error.contains(needle), "wanted {needle:?}: {error}");
        assert!(
            !out.exists(),
            "a refusal ({needle}) left {} behind",
            out.display()
        );
    };
    for cuts in [0, crate::prepare::FOLDS + 1] {
        refused(
            RunOptions {
                cuts,
                ..template(16)
            },
            &format!("--cuts must be 1..={}, got {cuts}", crate::prepare::FOLDS),
        );
    }
    refused(
        RunOptions {
            dataset: crate::run::tests::constant_target_dataset(temp.path()),
            cuts: 3,
            ..template(16)
        },
        "held-out target is constant",
    );
    let lineage = evolve_into(
        &out,
        2,
        false,
        RunOptions {
            cuts: 3,
            ..template(16)
        },
    )
    .unwrap();
    assert_eq!(lineage.rounds.len(), 2);
    assert_eq!(lineage.cuts, vec![0, 1, 2]);
    verify(&out.join(LINEAGE_FILE), false).unwrap();
}

#[test]
fn an_existing_lineage_or_a_bad_round_count_is_refused_before_writing() {
    let temp = tempfile::tempdir().unwrap();
    let out = temp.path().join("lineage");
    for rounds in [0, MAX_ROUNDS + 1] {
        assert!(evolve_into(&out, rounds, false, template(4)).is_err());
        assert!(!out.exists());
    }
    let error = format!(
        "{:#}",
        evolve_into(
            &out,
            1,
            false,
            RunOptions {
                cpu_ceiling_millicores: Some(10),
                ..template(4)
            }
        )
        .unwrap_err()
    );
    assert!(error.contains("--cpu-ceiling"), "{error}");
    assert!(!out.exists());
    evolve_into(&out, 1, false, template(4)).unwrap();
    let error = format!(
        "{:#}",
        evolve_into(&out, 1, false, template(4)).unwrap_err()
    );
    assert!(error.contains("already holds a lineage"), "{error}");
}

/// The PR #118 verifier's reproduction: an `--output` holding a stale `control/` or
/// `round-002/` was refused only when that run was reached, after the earlier rounds had
/// written `lineage.json` (and round 2 its `seeds.json`), so the corrected retry was refused
/// as "already holds a lineage". Anything in `--output`, an unrelated file included, is now
/// refused before a byte is written, and so is an `--output` that is not a directory. An
/// empty directory is accepted: the same command runs once the stale entry is removed.
#[test]
fn an_objective_ranked_lineage_is_refused_before_writing() {
    let temp = tempfile::tempdir().unwrap();
    let out = temp.path().join("objective");
    let run = RunOptions {
        objective: Some(jope_runtime::fitness::Fitness::Mae {}),
        ..template(10)
    };
    let error = format!("{:#}", evolve_into(&out, 2, false, run).unwrap_err());
    assert!(
        error.contains("ranks and verifies its rounds by held-out RMSE"),
        "{error}"
    );
    assert!(!out.exists());
}

#[test]
fn a_non_empty_output_is_refused_before_writing_and_the_emptied_one_runs() {
    let temp = tempfile::tempdir().unwrap();
    for (name, stale, control, needle) in [
        (
            "stale-control",
            "control/receipt.json",
            true,
            "is not empty (control)",
        ),
        (
            "stale-round",
            "round-002/receipt.json",
            false,
            "is not empty (round-002)",
        ),
        ("unrelated", "notes.txt", false, "is not empty (notes.txt)"),
        (
            "old-lineage",
            LINEAGE_FILE,
            false,
            "already holds a lineage",
        ),
    ] {
        let out = temp.path().join(name);
        let path = out.join(stale);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"{}\n").unwrap();
        let before = entries(&out);
        let error = format!(
            "{:#}",
            evolve_into(&out, 2, control, template(10)).unwrap_err()
        );
        assert!(error.contains(needle), "{name}: wanted {needle:?}: {error}");
        assert_eq!(entries(&out), before, "{name}: the refusal wrote into it");
    }
    let file = temp.path().join("a-file");
    std::fs::write(&file, b"x").unwrap();
    let dangling = temp.path().join("dangling");
    let missing = temp.path().join("missing");
    std::os::unix::fs::symlink(&missing, &dangling).unwrap();
    for out in [&file, &dangling] {
        let error = format!("{:#}", evolve_into(out, 1, false, template(4)).unwrap_err());
        assert!(
            error.contains("not a directory to write a lineage into"),
            "{error}"
        );
    }
    assert_eq!(std::fs::read(&file).unwrap(), b"x");
    assert!(
        !missing.exists(),
        "evolve created the dangling link's target"
    );
    // Positive control: the stale control/ removed, the same command runs into the directory.
    let out = temp.path().join("stale-control");
    std::fs::remove_dir_all(out.join(CONTROL_DIRECTORY)).unwrap();
    assert!(entries(&out).is_empty());
    let lineage = evolve_into(&out, 2, true, template(10)).unwrap();
    assert_eq!(lineage.rounds.len(), 2);
    assert!(lineage.control.is_some());
    verify(&out.join(LINEAGE_FILE), false).unwrap();
}
