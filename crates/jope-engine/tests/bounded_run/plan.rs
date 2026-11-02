// ---------------------------------------------------------------- W7-T02 plan

struct Run {
    _t: tempfile::TempDir,
    root: PathBuf,
    dir: PathBuf,
    keys: Keys,
    contract: BoundedRunContract,
    sha: String,
}
fn planned(now: u64) -> Run {
    let (t, root) = tempdir();
    write_closure(&root);
    let k = keys(3);
    let c = contract(now, 3, &k);
    let dir = root.join("run");
    let report = controller::plan(&dir, &bytes(&c), now, None).unwrap();
    assert!(report.created);
    Run {
        _t: t,
        root,
        dir,
        sha: report.plan_sha256,
        keys: k,
        contract: c,
    }
}

#[test]
fn plan_freezes_canonical_contract_and_is_idempotent() {
    let now = now_ms();
    let r = planned(now);
    let frozen = fs::read(r.dir.join("plan.json")).unwrap();
    assert_eq!(frozen, r.contract.canonical_bytes().unwrap());
    assert_eq!(r.sha, sha(&frozen));
    let status = controller::status(&r.dir).unwrap();
    assert_eq!(status.journal_records, 1);
    assert_eq!(
        status.original_deadline_unix_ms,
        r.contract.original_deadline_unix_ms
    );
    assert!(status.ranks.iter().all(|x| x.state == RankState::Scheduled));
    let before = tree(&r.dir);
    let again = controller::plan(&r.dir, &bytes(&r.contract), now, None).unwrap();
    assert!(!again.created);
    assert_eq!(again.plan_sha256, r.sha);
    assert_eq!(again.review["dispatch_started"], false);
    assert_eq!(tree(&r.dir), before);
    let mut other = r.contract.clone();
    other.counters.planned_updates = 4;
    for w in other
        .work_census
        .iter_mut()
        .filter(|w| w.kind == WorkKind::Numerical)
    {
        w.updates = 4;
    }
    assert!(controller::plan(&r.dir, &bytes(&other), now, None)
        .unwrap_err()
        .to_string()
        .contains("different plan"));
    assert_eq!(tree(&r.dir), before);
}

fn refused(c: &BoundedRunContract, now: u64) -> (PlanRefusal, String) {
    let typed = match BoundedRunContract::parse(&bytes(c)) {
        Err(e) => e,
        Ok(parsed) => parsed.validate_at(now).unwrap_err(),
    };
    let (_t, root) = tempdir();
    let dir = root.join("run");
    let e = controller::plan(&dir, &bytes(c), now, None)
        .unwrap_err()
        .to_string();
    assert!(!dir.exists(), "a refused plan created its run directory");
    (typed, e)
}

#[test]
fn invalid_plans_fail_before_anything_is_created() {
    let now = now_ms();
    let k = keys(3);
    let base = contract(now, 3, &k);
    type Edit = Box<dyn Fn(&mut BoundedRunContract)>;
    let cases: Vec<(&str, Edit)> = vec![
        ("rank order", Box::new(|c| c.participants.swap(0, 1))),
        (
            "duplicate device",
            Box::new(|c| c.participants[1].device = c.participants[0].device.clone()),
        ),
        (
            "cpu participant with vram",
            Box::new(|c| c.participants[2].device = None),
        ),
        (
            "local rank gap",
            Box::new(|c| c.participants[1].local_rank = 1),
        ),
        (
            "grant for another rank",
            Box::new(|c| c.participants[1].grant = c.participants[0].grant.clone()),
        ),
        (
            "grant signature",
            Box::new(|c| c.participants[0].grant.body.expires_unix_ms -= 1),
        ),
        (
            "no verification work",
            Box::new(|c| {
                c.work_census
                    .retain(|w| !(w.global_rank == 2 && w.kind == WorkKind::Verification))
            }),
        ),
        (
            "no cleanup work",
            Box::new(|c| {
                c.work_census
                    .retain(|w| !(w.global_rank == 0 && w.kind == WorkKind::Cleanup))
            }),
        ),
        (
            "unequal synchronous updates",
            Box::new(|c| c.work_census[0].updates = 2),
        ),
        (
            "updates on non-numerical work",
            Box::new(|c| c.work_census[1].updates = 1),
        ),
        ("closure order", Box::new(|c| c.artifact_closure.reverse())),
        (
            "result destination collides with closure",
            Box::new(|c| c.participants[0].result_destination.path = "inputs/corpus.bin".into()),
        ),
        (
            "message limit",
            Box::new(|c| c.rendezvous.max_message_bytes = 16 * 1024 * 1024),
        ),
        ("attempt without predecessor", Box::new(|c| c.attempt = 2)),
        (
            "two hosts share a machine id",
            Box::new(|c| {
                c.participants[1].host.machine_id = c.participants[0].host.machine_id.clone()
            }),
        ),
        (
            "malformed machine id",
            Box::new(|c| c.participants[2].host.machine_id = "fixture-host-2".into()),
        ),
        (
            "shared participant key",
            Box::new(|c| c.participants[1].public_key = c.participants[0].public_key.clone()),
        ),
        (
            "grant outlives deadline",
            Box::new(|c| {
                c.participants[0].grant.body.expires_unix_ms = c.original_deadline_unix_ms + 1
            }),
        ),
    ];
    for (name, edit) in cases {
        let mut c = base.clone();
        edit(&mut c);
        if !matches!(name, "grant signature" | "grant for another rank") {
            resign(&mut c, &k);
        }
        let (typed, message) = refused(&c, now);
        assert!(
            matches!(typed, PlanRefusal::Invalid(_)),
            "{name}: expected invalid, got {typed:?}"
        );
        assert!(
            message.contains("invalid bounded-run plan"),
            "{name}: {message}"
        );
    }
}

#[test]
fn underfunded_plans_fail_before_dispatch() {
    let now = now_ms();
    let k = keys(3);
    let base = contract(now, 3, &k);
    type Edit = Box<dyn Fn(&mut BoundedRunContract)>;
    let cases: Vec<(&str, Edit)> = vec![
        (
            "grant below cpu ceiling",
            Box::new(|c| c.participants[0].grant.body.cpu_millicores = 3_999),
        ),
        (
            "grant below vram ceiling",
            Box::new(|c| c.participants[1].grant.body.gpu_vram_bytes = Some(GIB)),
        ),
        (
            "cpu work exceeds funded core-ms",
            Box::new(|c| c.work_census[0].cpu_core_ms = 3_000_000),
        ),
        (
            "wall census exceeds ceiling",
            Box::new(|c| c.work_census[0].wall_ms = 590_000),
        ),
        (
            "peak memory exceeds ceiling",
            Box::new(|c| c.work_census[0].peak_ram_bytes = 9 * GIB),
        ),
        (
            "grant window shorter than wall ceiling",
            Box::new(|c| {
                c.participants[2].grant.body.expires_unix_ms = c.not_before_unix_ms + 500_000
            }),
        ),
    ];
    for (name, edit) in cases {
        let mut c = base.clone();
        edit(&mut c);
        resign(&mut c, &k);
        let (typed, message) = refused(&c, now);
        assert!(
            matches!(typed, PlanRefusal::Underfunded(_)),
            "{name}: expected underfunded, got {typed:?}"
        );
        assert!(
            message.contains("underfunded bounded-run plan"),
            "{name}: {message}"
        );
    }
    // The remaining original window, not the nominal one, must fund the ceilings.
    let late = base.original_deadline_unix_ms - 500_000;
    let (typed, _) = refused(&base, late);
    assert!(matches!(typed, PlanRefusal::Underfunded(_)), "{typed:?}");
    let (typed, _) = refused(&base, base.original_deadline_unix_ms);
    assert!(matches!(typed, PlanRefusal::Invalid(_)), "{typed:?}");
}

#[test]
fn legacy_manifests_keep_legacy_semantics() {
    use jain_smartcluster::{JobId, ResourceRequest, Timestamp};
    use jope_parallel::native_graph::Artifact;
    let a = Artifact {
        path: "a.json".into(),
        sha256: "a".repeat(64),
    };
    let legacy = jope_parallel::native_loop::Config {
        schema: "jope.native_loop.v1".into(),
        run: "legacy".into(),
        accepted_checkpoint: a.clone(),
        initial_corpus: a.clone(),
        accepted_decision: a.clone(),
        scientific_policy: a.clone(),
        label_admission_policy: a,
        fit_policy: "policy.json".into(),
        run_approval: "/etc/jope-dime/approval.json".into(),
        resident_job: JobId::new(),
        workers: 48,
        cycles: 2,
        initial_optimizer_step: 60_000,
        predecessor_run: None,
        updates_per_block: 3_000,
        global_batch: 2_048,
        microbatch: 256,
        direct_seconds: 30,
        continuation_evaluations: 16,
        continuation_seconds: 60,
        cpu_resources: ResourceRequest::cpu_and_ram(1_000, GIB),
        task_deadline: Timestamp::from_unix_millis(u64::MAX),
        qualification: false,
        preparation_overlay: None,
        cuts: 1,
        cut_start: 0,
        materialization: None,
    };
    let legacy_bytes = serde_json::to_vec(&legacy).unwrap();
    // The old loader still reads its own manifest unchanged ...
    let reread: jope_parallel::native_loop::Config = serde_json::from_slice(&legacy_bytes).unwrap();
    assert_eq!(serde_json::to_vec(&reread).unwrap(), legacy_bytes);
    // ... is never reinterpreted as a bounded plan ...
    assert!(matches!(
        BoundedRunContract::parse(&legacy_bytes),
        Err(PlanRefusal::Legacy(_))
    ));
    let (_t, root) = tempdir();
    let e = controller::plan(&root.join("run"), &legacy_bytes, now_ms(), None).unwrap_err();
    assert!(e.to_string().contains("legacy fixed run contract"));
    assert!(!root.join("run").exists());
    // ... and a bounded contract is not an old manifest.
    let k = keys(3);
    let bounded = bytes(&contract(now_ms(), 3, &k));
    assert!(serde_json::from_slice::<jope_parallel::native_loop::Config>(&bounded).is_err());
}
