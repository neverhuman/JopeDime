use super::*;

#[test]
fn one_thread_and_eight_threads_write_byte_identical_runs() {
    let workers = workers_up_to_eight();
    if workers < 2 {
        eprintln!("skipped: this host admits one worker, so no parallel run can be compared");
        return;
    }
    let temp = tempfile::tempdir().unwrap();
    let write = |threads: usize, name: &str| {
        let output = temp.path().join(name);
        let receipt = run(&RunOptions {
            threads,
            clock: Clock::Frozen,
            cuts: 2,
            ..small(&output, 30)
        })
        .unwrap();
        (receipt, tree(&output))
    };
    let (one, one_tree) = write(1, "one");
    let (many, many_tree) = write(workers, "many");
    assert_eq!(
        one_tree.keys().collect::<Vec<_>>(),
        many_tree.keys().collect::<Vec<_>>()
    );
    for (path, bytes) in &one_tree {
        assert!(
            many_tree[path] == *bytes,
            "{path} differs between 1 and {workers} threads"
        );
    }
    assert_eq!(one, many);
    assert!(one_tree.contains_key("receipt.json"));
    assert!(one.search.cuts.iter().all(|c| c
        .accounting
        .as_ref()
        .unwrap()
        .comparison
        .completed_evaluations
        > 0));
    // The regime this comparison covers: no phase was ended by the seconds bound.
    assert!(one.search.cuts.iter().all(|c| {
        let accounting = c.accounting.as_ref().unwrap();
        c.stopped_by != search::STOPPED_BY_SECONDS
            && accounting.exploration.stopped_by != search::STOPPED_BY_SECONDS
            && accounting.comparison.stopped_by != search::STOPPED_BY_SECONDS
    }));
}

#[test]
fn workers_above_the_cpu_ceiling_or_the_affinity_mask_are_refused_first() {
    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("run");
    let cpus = crate::parallel::affinity_cpus().unwrap();
    let error = format!(
        "{:#}",
        run(&RunOptions {
            threads: cpus + 1,
            ..small(&output, 4)
        })
        .unwrap_err()
    );
    assert!(error.contains("sched_getaffinity"), "{error}");
    let error = format!(
        "{:#}",
        run(&RunOptions {
            cpu_ceiling_millicores: Some(999),
            ..small(&output, 4)
        })
        .unwrap_err()
    );
    assert!(error.contains("--cpu-ceiling 999"), "{error}");
    assert!(!output.exists(), "a refused run wrote {}", output.display());
}
