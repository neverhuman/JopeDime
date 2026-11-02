use super::*;

/// The sign test against the exact binomial tail computed in integers, and finite far past
/// the 1023 pairs where `2^n` leaves the range of an f64.
#[test]
pub(super) fn ablation_sign_test_matches_the_exact_binomial_tail() {
    assert_eq!(sign_test(0, 0), 1.0);
    assert_eq!(sign_test(5, 0), 0.0625);
    assert_eq!(sign_test(0, 6), 0.031_25);
    assert_eq!(sign_test(3, 3), 1.0);
    for n in 1u32..=60 {
        // C(n, k) for every k, exactly.
        let mut binomial = vec![1u128];
        for k in 1..=u128::from(n) {
            let next = binomial[k as usize - 1] * (u128::from(n) - k + 1) / k;
            binomial.push(next);
        }
        for wins in 0..=n {
            let losses = n - wins;
            let tail = wins.min(losses) as usize;
            let lower: u128 = binomial[..=tail].iter().sum();
            let exact = (2.0 * lower as f64 / 2f64.powi(n as i32)).min(1.0);
            let p = sign_test(u64::from(wins), u64::from(losses));
            assert!(
                (p - exact).abs() <= 1e-12 * exact,
                "n {n} wins {wins}: {p} against {exact}"
            );
            assert_eq!(p, sign_test(u64::from(losses), u64::from(wins)));
        }
    }
    assert_eq!(sign_test(1000, 1000), 1.0);
    let p = sign_test(1100, 900);
    assert!(p > 1e-6 && p < 1e-4, "{p}");
    assert!(sign_test(2000, 0).is_finite());
}

/// Every text the kernel2 arms add, the option names and help, the profile ids, the new
/// document keys and fixed values, and the refusal, passes the shipped-text lint.
#[test]
pub(super) fn ablation_kernel2_text_passes_the_shipped_text_lint() {
    use clap::Args as _;
    let command = AblationArgs::augment_args(clap::Command::new("ablation"));
    let mut texts: Vec<String> = Vec::new();
    for id in ["refine_top_k", "refine_trials"] {
        let arg = command
            .get_arguments()
            .find(|a| a.get_id().as_str() == id)
            .unwrap();
        texts.push(arg.get_long().unwrap().to_owned());
        for help in [arg.get_help(), arg.get_long_help()].into_iter().flatten() {
            texts.push(help.to_string());
        }
    }
    texts.extend(
        [
            Profile::Kernel2Baseline.id(),
            Profile::Kernel2Experimental.id(),
            PAIRED_DEFINITIONS,
            PAIRED_RULE,
            REFINEMENT_REFUSAL,
            SCRATCH_IN_USE,
        ]
        .map(str::to_owned),
    );
    fn keys(value: &serde_json::Value, out: &mut Vec<String>) {
        match value {
            serde_json::Value::Object(map) => {
                for (key, value) in map {
                    out.push(key.clone());
                    keys(value, out);
                }
            }
            serde_json::Value::Array(items) => items.iter().for_each(|v| keys(v, out)),
            _ => {}
        }
    }
    let budget = Budget {
        evaluations: KERNEL2_EVALUATIONS,
        seconds: SEARCH_SECONDS,
    };
    let record = Profile::Kernel2Baseline
        .record(budget, Some(SMALL_REFINEMENT))
        .unwrap();
    // The record's `search` and `selection` values are the route's existing engine and
    // selection ids (`kernel2::receipt::{ENGINE, SELECTION}`), which this harness reuses
    // and does not lint; the legacy record's keys predate the lint.
    texts.push(record.candidate_source.to_owned());
    keys(&serde_json::to_value(&record).unwrap(), &mut texts);
    let summary = paired(
        &["d0".to_owned()],
        &[
            paired_run("A", "d0", 1, 1.0, "p"),
            paired_run("B", "d0", 1, 0.5, "q"),
        ],
    )
    .unwrap();
    keys(&serde_json::to_value(&summary).unwrap(), &mut texts);
    let cut = Kernel2Cut {
        measurements: 1,
        objective_trials: 0,
        rejected_trials: 0,
        arm_counts: BTreeMap::new(),
    };
    keys(&serde_json::to_value(&cut).unwrap(), &mut texts);
    keys(
        &serde_json::to_value(paired_timing(&[])).unwrap(),
        &mut texts,
    );
    texts.extend(["kernel2", "paired"].map(str::to_owned));
    let legacy_keys = [
        "id",
        "search",
        "candidate_source",
        "candidates_per_round",
        "carried_per_round",
        "max_transforms",
        "threads",
        "islands",
    ];
    for text in texts.iter().filter(|t| !legacy_keys.contains(&t.as_str())) {
        let hits = feat_math::transpile::lint_shipped_text(text);
        assert!(hits.is_empty(), "{text:?}: {hits:?}");
    }
}

/// An entry of a directory tree: its kind, its modification time, and a file's bytes or a
/// link's target.
type Entry = (&'static str, std::time::SystemTime, Vec<u8>);

/// Every entry under `root`, symbolic links not followed.
pub(super) fn tree(root: &Path) -> BTreeMap<PathBuf, Entry> {
    let mut entries = BTreeMap::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            let metadata = std::fs::symlink_metadata(&path).unwrap();
            let kind = metadata.file_type();
            let (name, content) = if kind.is_symlink() {
                let target = std::fs::read_link(&path).unwrap();
                ("link", target.into_os_string().into_encoded_bytes())
            } else if kind.is_dir() {
                pending.push(path.clone());
                ("dir", Vec::new())
            } else {
                ("file", std::fs::read(&path).unwrap())
            };
            let relative = path.strip_prefix(root).unwrap().to_path_buf();
            entries.insert(relative, (name, metadata.modified().unwrap(), content));
        }
    }
    entries
}

/// The entries whose kind, modification time or content differs between two [`tree`]s, or
/// that only one of them has.
pub(super) fn changed_entries(
    before: &BTreeMap<PathBuf, Entry>,
    after: &BTreeMap<PathBuf, Entry>,
) -> Vec<PathBuf> {
    before
        .keys()
        .chain(after.keys().filter(|path| !before.contains_key(*path)))
        .filter(|path| before.get(*path) != after.get(*path))
        .cloned()
        .collect()
}

/// A comparison into a scratch directory that already holds its kernel2 searches is refused
/// before anything is prepared or searched (review finding PR0-R1): every entry the first
/// run left keeps its bytes and modification time and nothing is added, also when the fold
/// seed, the jobs or the other slot's profile differ. A planned kernel2 output that is a
/// dangling symbolic link is refused the same way, with nothing written.
#[test]
pub(super) fn ablation_kernel2_refuses_a_scratch_that_holds_its_searches() {
    let config = Config {
        datasets: vec![d5("pmlb-first-principles-supernovae-zg")],
        evaluations: 1,
        refinement: Some(Refinement {
            top_k: 0,
            trials_per_program: 0,
        }),
        jobs: 1,
        ..kernel2_config()
    };
    let dir = scratch();
    let first = run(&config, dir.path()).unwrap();
    let hash = first.document.body.datasets[0].dataset_hash.clone();
    let before = tree(dir.path());
    for slot in ["A", "B"] {
        let published = search_output(Path::new(""), slot, &hash, 6).join("search/cut-0");
        for name in ["search.json", "observations.json", "refinements.json"] {
            let path = published.join(name);
            assert!(
                before.get(&path).is_some_and(|entry| entry.0 == "file"),
                "{}",
                path.display()
            );
        }
    }
    assert!(before.contains_key(&Path::new("prepared").join(&hash).join("prepared.json")));

    for again in [
        config.clone(),
        Config {
            fold_seed: config.fold_seed + 1,
            ..config.clone()
        },
        Config {
            jobs: 2,
            ..config.clone()
        },
        Config {
            profile_a: Profile::Legacy,
            ..config.clone()
        },
        Config {
            profile_b: Profile::Kernel2Experimental,
            ..config.clone()
        },
    ] {
        let error = format!("{:#}", run(&again, dir.path()).unwrap_err());
        assert!(error.contains(SCRATCH_IN_USE), "{again:?}: {error}");
        let changed = changed_entries(&before, &tree(dir.path()));
        assert!(
            changed.is_empty(),
            "{again:?}: a refused rerun changed {changed:?}"
        );
    }

    #[cfg(unix)]
    {
        let dir = scratch();
        let output = search_output(dir.path(), "B", &hash, 6);
        std::fs::create_dir_all(output.parent().unwrap()).unwrap();
        std::os::unix::fs::symlink("missing", &output).unwrap();
        let before = tree(dir.path());
        let error = format!("{:#}", run(&config, dir.path()).unwrap_err());
        assert!(error.contains(SCRATCH_IN_USE), "{error}");
        let changed = changed_entries(&before, &tree(dir.path()));
        assert!(changed.is_empty(), "a refused run changed {changed:?}");
        assert!(std::fs::symlink_metadata(dir.path().join("prepared")).is_err());
    }
}

/// A paired refusal: its message, which must pass the shipped-text lint.
pub(super) fn paired_refusal(datasets: &[&str], runs: &[RunRecord]) -> String {
    let datasets: Vec<String> = datasets.iter().map(|d| (*d).to_owned()).collect();
    let error = match paired(&datasets, runs) {
        Ok(summary) => panic!("refusal expected, received {summary:?}"),
        Err(error) => format!("{error:#}"),
    };
    let hits = feat_math::transpile::lint_shipped_text(&error);
    assert!(hits.is_empty(), "{error:?}: {hits:?}");
    error
}

/// Review finding PR0-R2, `reject_duplicate_a_and_unpaired_b`: a repeated A run whose extra
/// pair is balanced by an unpaired B run is refused, not counted as two wins; so is the
/// mirror case on the B side.
#[test]
pub(super) fn ablation_paired_rejects_duplicate_a_and_unpaired_b() {
    let error = paired_refusal(
        &["d0"],
        &[
            paired_run("A", "d0", 1, 1.0, "p"),
            paired_run("A", "d0", 1, 1.0, "p"),
            paired_run("B", "d0", 1, 0.5, "q"),
            paired_run("B", "d0", 2, 99.0, "r"),
        ],
    );
    assert!(error.contains("two A runs"), "{error}");
    let error = paired_refusal(
        &["d0"],
        &[
            paired_run("A", "d0", 1, 1.0, "p"),
            paired_run("A", "d0", 2, 1.0, "p"),
            paired_run("B", "d0", 1, 0.5, "q"),
            paired_run("B", "d0", 1, 0.5, "q"),
        ],
    );
    assert!(error.contains("two B runs"), "{error}");
}

/// Review finding PR0-R2, `reject_unlisted_unpaired_a`: an A run on a dataset the
/// comparison does not list is refused rather than left out; so is a B run there, paired or
/// not, and a B run on a listed dataset without its A run.
#[test]
pub(super) fn ablation_paired_rejects_unlisted_unpaired_a() {
    let pair = [
        paired_run("A", "d0", 1, 1.0, "p"),
        paired_run("B", "d0", 1, 0.5, "q"),
    ];
    for extra in [
        vec![paired_run("A", "outside", 9, 2.0, "p")],
        vec![paired_run("B", "outside", 9, 2.0, "p")],
        vec![
            paired_run("A", "outside", 9, 2.0, "p"),
            paired_run("B", "outside", 9, 2.0, "p"),
        ],
    ] {
        let runs: Vec<RunRecord> = pair.iter().cloned().chain(extra).collect();
        let error = paired_refusal(&["d0"], &runs);
        assert!(error.contains("does not list"), "{error}");
    }
    let mut runs = pair.to_vec();
    runs.push(paired_run("B", "d0", 2, 0.5, "q"));
    let error = paired_refusal(&["d0"], &runs);
    assert!(error.contains("a B run needs exactly one A run"), "{error}");
}

/// The pairing refuses a run whose slot is neither A nor B, and a dataset listed twice.
#[test]
pub(super) fn ablation_paired_rejects_a_repeated_dataset_or_an_unknown_slot() {
    let pair = [
        paired_run("A", "d0", 1, 1.0, "p"),
        paired_run("B", "d0", 1, 0.5, "q"),
    ];
    let mut runs = pair.to_vec();
    runs.push(paired_run("C", "d0", 1, 0.5, "q"));
    let error = paired_refusal(&["d0"], &runs);
    assert!(error.contains("neither A nor B"), "{error}");
    let error = paired_refusal(&["d0", "d0"], &pair);
    assert!(error.contains("listed twice"), "{error}");
}

/// `execute` makes every refusal of the options before it creates a scratch directory
/// (review finding PR0-R3): for each refused combination neither the `--scratch` path, nor
/// any of its parents under the test directory, nor `--out` exists afterwards.
#[test]
pub(super) fn ablation_execute_refuses_before_creating_scratch() {
    let root = scratch();
    let base = AblationArgs {
        profile_a: Profile::Kernel2Baseline.id().to_owned(),
        profile_b: Profile::Kernel2Baseline.id().to_owned(),
        refine_top_k: Some(1),
        refine_trials: Some(1),
        datasets: vec![d5("pmlb-first-principles-supernovae-zg")],
        seeds: vec![6],
        evaluations: 1,
        cuts: 1,
        fold_seed: jope_pipeline::DEFAULT_SEED,
        jobs: 1,
        out: PathBuf::new(),
        scratch: None,
    };
    // The base passes every check, so each case that follows is refused for its one change.
    config(&base).unwrap().check().unwrap();
    let legacy = AblationArgs {
        profile_a: Profile::Legacy.id().to_owned(),
        profile_b: Profile::Legacy.id().to_owned(),
        refine_top_k: None,
        refine_trials: None,
        ..base.clone()
    };
    config(&legacy).unwrap().check().unwrap();
    let cases = [
        // A kernel2 profile without the allowance.
        (
            AblationArgs {
                refine_top_k: None,
                refine_trials: None,
                ..base.clone()
            },
            "--refine-top-k",
        ),
        (
            AblationArgs {
                profile_a: Profile::Legacy.id().to_owned(),
                refine_top_k: None,
                refine_trials: None,
                ..base.clone()
            },
            "--refine-trials",
        ),
        // The allowance with legacy only.
        (
            AblationArgs {
                refine_top_k: Some(0),
                refine_trials: Some(0),
                ..legacy.clone()
            },
            "legacy profile refuses",
        ),
        // More than one cut.
        (
            AblationArgs {
                cuts: 2,
                ..base.clone()
            },
            "--cuts 1",
        ),
        // More refined parents than the carried set.
        (
            AblationArgs {
                refine_top_k: Some(search::ELITES + 1),
                refine_trials: Some(0),
                ..base.clone()
            },
            "refinement allowance",
        ),
        // More trials than the cut's budget.
        (
            AblationArgs {
                refine_trials: Some(2),
                ..base.clone()
            },
            "refinement allowance",
        ),
        // A refinement option alone, refused by `config`.
        (
            AblationArgs {
                refine_trials: None,
                ..base.clone()
            },
            "--refine-trials",
        ),
        // The checks every profile makes.
        (
            AblationArgs {
                seeds: vec![6, 6],
                ..legacy.clone()
            },
            "a seed is listed twice",
        ),
        (
            AblationArgs {
                evaluations: search::MAX_EVALUATIONS + 1,
                ..legacy.clone()
            },
            "budget evaluations must be",
        ),
        (
            AblationArgs {
                cuts: prepare::FOLDS + 1,
                ..legacy.clone()
            },
            "--cuts must be",
        ),
        (
            AblationArgs {
                jobs: 0,
                ..legacy.clone()
            },
            "--jobs must be at least 1",
        ),
    ];
    for (index, (args, needle)) in cases.into_iter().enumerate() {
        let case = root.path().join(format!("case-{index}"));
        let args = AblationArgs {
            scratch: Some(case.join("scratch/nested")),
            out: case.join("out/ablation.json"),
            ..args
        };
        let error = format!("{:#}", execute(&args).unwrap_err());
        assert!(
            error.contains(needle),
            "case {index}: {needle:?} not in {error:?}"
        );
        assert!(
            std::fs::symlink_metadata(&case).is_err(),
            "case {index} ({error}) created {}",
            case.display()
        );
    }
    assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 0);
}

// Review 02: findings PR0-R4 (a used scratch) and PR0-R5 (a non-finite M1).

/// The configuration of the reviewer's PR0-R4 reproducer: kernel2-baseline A=A on
/// supernovae-zg, seed 6, fold seed 20260907, one evaluation, one cut, one job and no
/// refinement.
pub(super) fn shared_scratch_config() -> Config {
    Config {
        datasets: vec![d5("pmlb-first-principles-supernovae-zg")],
        evaluations: 1,
        fold_seed: 20_260_907,
        jobs: 1,
        refinement: Some(Refinement {
            top_k: 0,
            trials_per_program: 0,
        }),
        ..kernel2_config()
    }
}
