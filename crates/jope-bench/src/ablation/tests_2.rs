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

/// [`shared_scratch_config`] as command-line arguments.
pub(super) fn shared_scratch_args(scratch: PathBuf, out: PathBuf) -> AblationArgs {
    AblationArgs {
        profile_a: Profile::Kernel2Baseline.id().to_owned(),
        profile_b: Profile::Kernel2Baseline.id().to_owned(),
        refine_top_k: Some(0),
        refine_trials: Some(0),
        datasets: vec![d5("pmlb-first-principles-supernovae-zg")],
        seeds: vec![6],
        evaluations: 1,
        cuts: 1,
        fold_seed: 20_260_907,
        jobs: 1,
        out,
        scratch: Some(scratch),
    }
}

/// A comparison refused for its scratch directory ([`SCRATCH_NOT_UNUSED`]): its message,
/// which must pass the shipped-text lint with the path left out.
pub(super) fn scratch_refusal(config: &Config, scratch: &Path) -> String {
    scratch_refusal_saying(config, scratch, SCRATCH_NOT_UNUSED)
}

/// [`scratch_refusal`] with the text the message must contain.
pub(super) fn scratch_refusal_saying(config: &Config, scratch: &Path, needle: &str) -> String {
    let error = format!("{:#}", run(config, scratch).unwrap_err());
    assert!(error.contains(needle), "{needle:?} not in {error}");
    let text = error.replace(&scratch.display().to_string(), "<scratch>");
    let hits = feat_math::transpile::lint_shipped_text(&text);
    assert!(hits.is_empty(), "{text:?}: {hits:?}");
    error
}

/// Review finding PR0-R4, the reviewer's reproducer: a second comparison with a kernel2
/// profile into the same scratch directory, with a new search seed (7) and a new fold seed
/// (20260908), would rewrite the shared `prepared.json`. It is refused before anything is
/// written, also with a legacy slot A: every entry keeps its kind, modification time and
/// bytes, nothing is added, and the first document's `prepared_sha256` still names the file.
#[test]
pub(super) fn ablation_kernel2_refuses_a_used_scratch_with_a_new_fold_seed() {
    let config = shared_scratch_config();
    let dir = scratch();
    let first = run(&config, dir.path()).unwrap();
    let record = &first.document.body.datasets[0];
    let before = tree(dir.path());
    let again = Config {
        seeds: vec![7],
        fold_seed: 20_260_908,
        ..config.clone()
    };
    for again in [
        again.clone(),
        Config {
            profile_a: Profile::Legacy,
            ..again
        },
    ] {
        scratch_refusal_saying(&again, dir.path(), SCRATCH_HOLDS_KERNEL2);
        let changed = changed_entries(&before, &tree(dir.path()));
        assert!(
            changed.is_empty(),
            "{again:?}: a refused comparison changed {changed:?}"
        );
    }
    let prepared = dir
        .path()
        .join("prepared")
        .join(&record.dataset_hash)
        .join("prepared.json");
    assert_eq!(
        sha256_hex(&std::fs::read(prepared).unwrap()),
        record.prepared_sha256
    );
}

/// Each comparison with a kernel2 arm takes its own scratch directory: a new search seed with
/// the same fold seed and cuts, which would write the same partition, is refused too, with
/// nothing changed.
#[test]
pub(super) fn ablation_kernel2_refuses_a_used_scratch_with_the_same_fold_seed() {
    let config = shared_scratch_config();
    let dir = scratch();
    run(&config, dir.path()).unwrap();
    let before = tree(dir.path());
    scratch_refusal_saying(
        &Config {
            seeds: vec![7],
            ..config.clone()
        },
        dir.path(),
        SCRATCH_HOLDS_KERNEL2,
    );
    let changed = changed_entries(&before, &tree(dir.path()));
    assert!(
        changed.is_empty(),
        "a refused comparison changed {changed:?}"
    );
}

/// A scratch directory that is not unused is refused before anything is written, with
/// nothing changed: one holding only `prepared/`, one holding a stray file, a symbolic link
/// to an empty directory, and a path that is a regular file, from the library and from the
/// command line. Each case has its own test directory, whose whole tree is compared. A
/// symbolic link and a regular file are refused before the lock file is created; the other
/// two under the reservation, whose lock file is created and removed in the scratch
/// directory, so that directory's own modification time is the one change allowed.
#[test]
pub(super) fn ablation_kernel2_refuses_a_scratch_that_is_not_unused() {
    let config = shared_scratch_config();
    let refused = |needle: &dyn Fn(&Path) -> String, make: &dyn Fn(&Path) -> PathBuf| {
        let root = scratch();
        let path = make(root.path());
        let before = tree(root.path());
        scratch_refusal_saying(&config, &path, &needle(&path));
        let after = tree(root.path());
        let changed = changed_entries(&before, &after);
        let own = path.strip_prefix(root.path()).unwrap();
        let kind =
            |entries: &BTreeMap<PathBuf, Entry>, entry: &PathBuf| entries.get(entry).map(|e| e.0);
        assert!(
            changed.iter().all(|entry| entry.as_path() == own
                && kind(&before, entry) == Some("dir")
                && kind(&after, entry) == Some("dir")),
            "{}: a refused comparison changed {changed:?}",
            path.display()
        );
        assert!(std::fs::symlink_metadata(path.join(LOCK_FILE)).is_err());
    };
    let unused = |_: &Path| SCRATCH_NOT_UNUSED.to_owned();
    refused(&unused, &|root: &Path| {
        let path = root.join("scratch");
        std::fs::create_dir_all(path.join("prepared")).unwrap();
        path
    });
    refused(&unused, &|root: &Path| {
        let path = root.join("scratch");
        std::fs::create_dir(&path).unwrap();
        std::fs::write(path.join("stray.txt"), b"stray").unwrap();
        path
    });
    #[cfg(unix)]
    {
        refused(&unused, &|root: &Path| {
            std::fs::create_dir(root.join("empty")).unwrap();
            let path = root.join("scratch");
            std::os::unix::fs::symlink("empty", &path).unwrap();
            path
        });
    }
    refused(&unused, &|root: &Path| {
        let path = root.join("scratch");
        std::fs::write(&path, b"a file").unwrap();
        path
    });
    let root = scratch();
    let path = root.path().join("scratch");
    std::fs::write(&path, b"a file").unwrap();
    let before = tree(root.path());
    let args = shared_scratch_args(path, root.path().join("ablation.json"));
    assert!(execute(&args).is_err());
    let changed = changed_entries(&before, &tree(root.path()));
    assert!(changed.is_empty(), "a refused command changed {changed:?}");
}

/// An unused scratch directory is admitted: an existing empty directory, an absent path (the
/// comparison creates it), and the empty directory `execute` creates for `--scratch`.
#[test]
pub(super) fn ablation_kernel2_admits_an_unused_scratch() {
    let config = shared_scratch_config();
    let dir = scratch();
    let report = run(&config, dir.path()).unwrap();
    let hash = &report.document.body.datasets[0].dataset_hash;
    let prepared = Path::new("prepared").join(hash).join("prepared.json");
    assert!(dir.path().join(&prepared).is_file());
    let root = scratch();
    let absent = root.path().join("absent/nested");
    run(&config, &absent).unwrap();
    assert!(absent.join(&prepared).is_file());
    let root = scratch();
    let args = shared_scratch_args(
        root.path().join("scratch/nested"),
        root.path().join("ablation.json"),
    );
    execute(&args).unwrap();
    assert!(root.path().join("scratch/nested").join(&prepared).is_file());
    assert!(root.path().join("ablation.json").is_file());
}

/// Review finding PR0-R5: `paired()` refuses an M1 RMSE that is not finite, on either side,
/// before any bit-equality, ordering or arithmetic, naming the dataset, the seed and the
/// slot: equal NaNs, equal infinities of either sign, an infinity of either sign against a
/// finite value and NaN against a finite value, each also with the sides swapped, and a NaN
/// after a valid pair on another dataset.
#[test]
pub(super) fn ablation_paired_refuses_a_nonfinite_m1() {
    let (nan, inf) = (f64::NAN, f64::INFINITY);
    for (a, b, slot) in [
        (nan, nan, "A"),
        (inf, inf, "A"),
        (-inf, -inf, "A"),
        (1.0, inf, "B"),
        (inf, 1.0, "A"),
        (-inf, 1.0, "A"),
        (1.0, -inf, "B"),
        (nan, 1.0, "A"),
        (1.0, nan, "B"),
    ] {
        let error = paired_refusal(
            &["d0"],
            &[
                paired_run("A", "d0", 3, a, "p"),
                paired_run("B", "d0", 3, b, "q"),
            ],
        );
        assert!(
            error.contains(&format!("dataset d0 seed 3: the {slot} run's M1 RMSE is "))
                && error.contains("finite values"),
            "{a} against {b}: {error}"
        );
    }
    let error = paired_refusal(
        &["d0", "d1"],
        &[
            paired_run("A", "d0", 1, 1.0, "p"),
            paired_run("B", "d0", 1, 0.5, "q"),
            paired_run("A", "d1", 4, 1.0, "p"),
            paired_run("B", "d1", 4, nan, "q"),
        ],
    );
    assert!(
        error.contains("dataset d1 seed 4: the B run's M1 RMSE is NaN"),
        "{error}"
    );
}

/// Finite M1 RMSEs whose summary would leave the range of an f64 are refused, so every number
/// a returned summary holds is finite; the largest finite difference is still reported.
#[test]
pub(super) fn ablation_paired_refuses_a_summary_that_is_not_finite() {
    let max = f64::MAX;
    // One difference of f64::MAX minus -f64::MAX.
    let error = paired_refusal(
        &["d0"],
        &[
            paired_run("A", "d0", 1, -max, "p"),
            paired_run("B", "d0", 1, max, "q"),
        ],
    );
    assert!(
        error.contains("paired median difference is not finite"),
        "{error}"
    );
    // Two pairs at f64::MAX: the mean of B's two middle values leaves the range.
    let error = paired_refusal(
        &["d0"],
        &[
            paired_run("A", "d0", 1, 0.0, "p"),
            paired_run("B", "d0", 1, max, "q"),
            paired_run("A", "d0", 2, 0.0, "p"),
            paired_run("B", "d0", 2, max, "q"),
        ],
    );
    assert!(
        error.contains("paired median M1 RMSE of B is not finite"),
        "{error}"
    );
    let summary = paired(
        &["d0".to_owned()],
        &[
            paired_run("A", "d0", 1, 0.0, "p"),
            paired_run("B", "d0", 1, max, "q"),
        ],
    )
    .unwrap();
    let c = &summary.overall;
    assert_eq!(
        (c.losses, c.median_b, c.median_difference, c.mean_difference),
        (1, max, max, max)
    );
    assert_eq!(c.largest_gain.difference, max);
}

/// The independent reviewer's PR0-R5 regression test (review 02), adopted unchanged.
#[test]
pub(super) fn review02_public_paired_refuses_nonfinite_m1() {
    let mut accepted = Vec::new();
    for (name, a, b) in [
        ("same_nan", f64::NAN, f64::NAN),
        ("same_positive_infinity", f64::INFINITY, f64::INFINITY),
        (
            "same_negative_infinity",
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ),
        ("infinity_vs_finite", f64::INFINITY, 1.0),
        ("finite_vs_infinity", 1.0, f64::INFINITY),
        ("nan_vs_finite", f64::NAN, 1.0),
    ] {
        let runs = [
            paired_run("A", "d", 1, a, "p"),
            paired_run("B", "d", 1, b, "q"),
        ];
        match paired(&["d".to_owned()], &runs) {
            Ok(summary) => {
                println!(
                    "ACCEPTED_INVALID {name}: {}",
                    serde_json::to_string(&summary).unwrap()
                );
                accepted.push(name);
            }
            Err(error) => println!("REFUSED_INVALID {name}: {error:#}"),
        }
    }
    assert!(accepted.is_empty(), "nonfinite RMSE accepted: {accepted:?}");
}

// Review 03: findings PR0-R4 (reopened: a later legacy-only comparison) and PR0-R6 (concurrent
// comparisons on one scratch directory).

/// The reproducer's dataset compared legacy against legacy: seed `seed`, fold seed
/// `fold_seed`, one evaluation, one cut and one job.
pub(super) fn legacy_scratch_config(seed: u64, fold_seed: u64) -> Config {
    Config {
        profile_a: Profile::Legacy,
        profile_b: Profile::Legacy,
        seeds: vec![seed],
        fold_seed,
        refinement: None,
        ..shared_scratch_config()
    }
}

/// The names directly under `dir`.
pub(super) fn names(dir: &Path) -> BTreeSet<String> {
    std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect()
}

/// `names` as a set.
pub(super) fn set(names: &[&str]) -> BTreeSet<String> {
    names.iter().map(|name| (*name).to_owned()).collect()
}
