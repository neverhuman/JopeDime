use super::*;

#[test]
fn a_stale_latest_temporary_from_a_killed_writer_cannot_block_publication() {
    // A writer killed between creating latest.json's temporary and renaming it
    // leaves `latest.partial.<pid>`. The previous publication reused that
    // fixed name with create_new, so a later writer with the same process id
    // (a restarted container, a recycled pid) failed every checkpoint.
    let mut f = fixture();
    f.train_to(2);
    let pid = std::process::id();
    fs::write(f.run.join(format!("latest.partial.{pid}")), b"{\"torn\":").unwrap();
    fs::write(f.run.join(format!("initialization.partial.{pid}")), b"{").unwrap();
    let latest = f.commit(&mut checkpoint_commit::no_stop).unwrap();
    assert_eq!(latest["step"], 2);
    assert_eq!(f.load().unwrap().step, 2);
    // The same holds for the run's initialization record.
    replace_json(&f.run.join("initialization.json"), &json!({"kind":"fresh"})).unwrap();
    assert_eq!(
        read_json::<Value>(&f.run.join("initialization.json")).unwrap(),
        json!({"kind":"fresh"})
    );
    // Stale temporaries are left in place (inert, never read), not deleted.
    assert!(f.run.join(format!("latest.partial.{pid}")).exists());
}

#[test]
fn a_stop_between_any_two_publication_steps_never_exposes_a_partial_checkpoint() {
    for phase in PHASES {
        let mut f = fixture();
        f.train_to(2);
        f.commit(&mut checkpoint_commit::no_stop).unwrap();
        f.train_to(4);
        let error = f.commit(&mut stop_at(phase)).unwrap_err();
        assert!(
            error.to_string().contains("injected stop"),
            "{phase:?}: {error}"
        );
        let published = f.load().unwrap();
        if phase == Phase::LatestPublished {
            // The publication itself completed; only the caller was stopped.
            assert_eq!(published.step, 4, "{phase:?}");
            let journal = fs::read(f.journal.path()).unwrap();
            let r = consumption::reconcile(&journal, Some(4)).unwrap();
            assert!(r.commit_event_missing && r.committed_through == 4);
            continue;
        }
        assert_eq!(
            published.step, 2,
            "{phase:?}: an interrupted publication was exposed"
        );
        if matches!(phase, Phase::Renamed | Phase::LatestStaged) {
            // The complete but unpublished directory is never overwritten.
            assert!(
                f.commit(&mut checkpoint_commit::no_stop).is_err(),
                "{phase:?}"
            );
        }
        // Stopped-rank recovery quarantines the partial work without deleting it.
        checkpoint_recovery::reconcile(&f.root, &f.run).unwrap();
        let quarantine = fs::read_dir(f.run.join("checkpoint-recovery"))
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let receipt: Value = read_json(&quarantine.join("receipt.json")).unwrap();
        assert_eq!(receipt["completed"], true, "{phase:?}");
        assert_eq!(receipt["moves"].as_array().unwrap().len(), 1, "{phase:?}");
        assert_eq!(f.load().unwrap().step, 2, "{phase:?}");
        // The re-executed publication commits the same step.
        f.commit(&mut checkpoint_commit::no_stop).unwrap();
        assert_eq!(f.load().unwrap().step, 4, "{phase:?}");
    }
}

#[test]
fn commit_refuses_to_expose_an_incomplete_or_inconsistent_checkpoint() {
    type Edit = Box<dyn Fn(&mut Value)>;
    let cases: Vec<(&str, bool, Edit)> = vec![
        ("missing optimizer", true, Box::new(|_| {})),
        (
            "rng absent",
            false,
            Box::new(|s| drop(s.as_object_mut().unwrap().remove("rng"))),
        ),
        (
            "counters inconsistent",
            false,
            Box::new(|s| {
                let counts = &mut s["committed_exposure_by_rank"][0]["counts"];
                counts["program_presentations"] = json!(16);
                counts["comparison_groups"] = json!(2);
                counts["designated_broad_groups"] = json!(2);
            }),
        ),
        (
            "rank receipt unbound",
            false,
            Box::new(|s| s["rank_receipts"][0]["sha256"] = json!("e".repeat(64))),
        ),
        (
            "parameters differ from receipt",
            false,
            Box::new(|s| s["parameter_sha256"] = json!("f".repeat(64))),
        ),
        ("manifest step", false, Box::new(|s| s["step"] = json!(2))),
    ];
    // Positive control: the unedited manifest commits.
    let mut control = fixture();
    control.train_to(3);
    control
        .commit_with(&|_| {}, false, &mut checkpoint_commit::no_stop)
        .unwrap();
    for (case, omit_optimizer, edit) in cases {
        let mut f = fixture();
        f.train_to(2);
        f.commit(&mut checkpoint_commit::no_stop).unwrap();
        f.train_to(3);
        let latest_before = fs::read(f.run.join("latest.json")).unwrap();
        let result = f.commit_with(&*edit, omit_optimizer, &mut checkpoint_commit::no_stop);
        assert!(
            result.is_err(),
            "{case}: an invalid checkpoint was committed"
        );
        assert!(!f.checkpoint_dir(3).exists(), "{case}: exposed step-3");
        assert_eq!(
            fs::read(f.run.join("latest.json")).unwrap(),
            latest_before,
            "{case}: latest.json changed"
        );
        assert_eq!(f.load().unwrap().step, 2);
    }
}
