use super::*;

/// [`shared_scratch_config`] as command-line arguments.

pub(crate) fn shared_scratch_args(scratch: PathBuf, out: PathBuf) -> AblationArgs {
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
pub(crate) fn scratch_refusal(config: &Config, scratch: &Path) -> String {
    scratch_refusal_saying(config, scratch, SCRATCH_NOT_UNUSED)
}

/// [`scratch_refusal`] with the text the message must contain.
pub(crate) fn scratch_refusal_saying(config: &Config, scratch: &Path, needle: &str) -> String {
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
pub(crate) fn ablation_kernel2_refuses_a_used_scratch_with_a_new_fold_seed() {
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
pub(crate) fn ablation_kernel2_refuses_a_used_scratch_with_the_same_fold_seed() {
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
pub(crate) fn ablation_kernel2_refuses_a_scratch_that_is_not_unused() {
    let config = shared_scratch_config();
    let refused = |needle: &dyn Fn(&Path) -> String, make: &dyn Fn(&Path) -> PathBuf| {
        let root = scratch();
        let path = make(root.path());
        let before = tree(root.path());
        scratch_refusal_saying(&config, &path, &needle(&path));
        let after = tree(root.path());
        let changed = changed_entries(&before, &after);
        let own = path.strip_prefix(root.path()).unwrap();
        let kind = |entries: &BTreeMap<PathBuf, super::part_a::Entry>, entry: &PathBuf| {
            entries.get(entry).map(|e| e.0)
        };
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
pub(crate) fn ablation_kernel2_admits_an_unused_scratch() {
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
pub(crate) fn ablation_paired_refuses_a_nonfinite_m1() {
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
pub(crate) fn ablation_paired_refuses_a_summary_that_is_not_finite() {
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
pub(crate) fn review02_public_paired_refuses_nonfinite_m1() {
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
pub(crate) fn legacy_scratch_config(seed: u64, fold_seed: u64) -> Config {
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
pub(crate) fn names(dir: &Path) -> BTreeSet<String> {
    std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect()
}

/// `names` as a set.
pub(crate) fn set(names: &[&str]) -> BTreeSet<String> {
    names.iter().map(|name| (*name).to_owned()).collect()
}
