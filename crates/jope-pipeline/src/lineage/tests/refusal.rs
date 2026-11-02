use super::*;

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
