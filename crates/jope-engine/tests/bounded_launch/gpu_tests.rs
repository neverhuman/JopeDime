use super::*;

#[test]
fn smartcluster_allocates_a_gpu_rank_exactly_one_exclusive_gpu() {
    use jain_smartcluster::{GpuMode, GpuRequest};
    use jope_parallel::native_loop::bounded::controller::RankLaunch;
    let k = keys(3);
    let c = with_cpu_rank(&k);
    let (_t, root) = tempdir();
    let run_dir = root.join("run");
    let plan = controller::plan(&run_dir, &bytes(&c), now_ms(), None)
        .unwrap()
        .plan_sha256;
    let intent = now_ms() - 1000;
    let spec = |rank: u32| {
        fixture_supervisor(&root, &run_dir, &format!("fixture-host-{rank}"), rank).launch_spec(
            &RankLaunch {
                dispatch_intent_unix_ms: intent,
                contract: &c,
                plan_sha256: &plan,
                global_rank: rank,
                dispatch_id: &rendezvous::dispatch_id(&plan, c.attempt, rank).unwrap(),
            },
        )
    };
    // A GPU rank asks SmartCluster for one exclusive GPU with its planned VRAM ceiling;
    // SmartCluster chooses the device and the rank checks it is the planned one.
    let (_, gpu) = spec(1).unwrap();
    assert_eq!(
        gpu.deadline,
        Some(jain_smartcluster::Timestamp::from(
            intent + c.participants[1].ceilings.wall_ms
        ))
    );
    assert_eq!(
        gpu.resources.gpu,
        Some(GpuRequest {
            count: 1,
            vram_bytes: 20 * GIB,
            compute_share_per_mille: 1000,
            mode: GpuMode::Exclusive,
        })
    );
    // A GPU rank with no VRAM ceiling in its plan cannot be allocated a GPU: refused.
    let mut unbounded = c.clone();
    unbounded.participants[1].ceilings.gpu_vram_bytes = None;
    let e = fixture_supervisor(&root, &run_dir, "fixture-host-1", 11)
        .launch_spec(&RankLaunch {
            dispatch_intent_unix_ms: now_ms(),
            contract: &unbounded,
            plan_sha256: &plan,
            global_rank: 1,
            dispatch_id: &rendezvous::dispatch_id(&plan, c.attempt, 1).unwrap(),
        })
        .unwrap_err();
    assert!(e.to_string().contains("has no VRAM ceiling"), "{e}");
    // A CPU rank asks for none, and keeps its CPU and RAM ceilings.
    let (_, cpu) = spec(2).unwrap();
    assert_eq!(cpu.resources.gpu, None);
    assert_eq!(
        (cpu.resources.cpu, cpu.resources.ram),
        (
            jain_smartcluster::ResourceRequest::cpu_and_ram(4_000, 8 * GIB).cpu,
            jain_smartcluster::ResourceRequest::cpu_and_ram(4_000, 8 * GIB).ram
        )
    );
}
#[test]
fn a_smartcluster_gpu_index_must_translate_to_the_planned_uuid() {
    let k = keys(3);
    let c = with_cpu_rank(&k);
    let plan = digest(&c);
    // Rank 1 is planned on GPU-fixture-1; rank 2 is a CPU rank.
    let gpu = launch::launch_record(&c, &plan, 1).unwrap();
    let cpu = launch::launch_record(&c, &plan, 2).unwrap();
    let host = "0, GPU-fixture-0\n1, GPU-another-device\n2, GPU-fixture-1\n";
    // Positive control: SmartCluster allocated index 2, which is the planned GPU; the worker
    // pins that UUID for itself.
    let smi = Smi::listing(host);
    let resolved = launch::resolve_visibility(&gpu, Some(OsStr::new("2")), &smi).unwrap();
    assert_eq!(
        resolved,
        ResolvedVisibility::SchedulerIndex {
            index: 2,
            uuid: "GPU-fixture-1".into()
        }
    );
    assert_eq!(resolved.pinned(), Some("GPU-fixture-1"));
    assert_eq!(smi.calls.get(), 1);
    // The launcher's exact UUID needs no translation, and a CPU rank never runs nvidia-smi.
    let never = Smi::with(None);
    assert_eq!(
        launch::resolve_visibility(&gpu, Some(OsStr::new("GPU-fixture-1")), &never).unwrap(),
        ResolvedVisibility::PlannedUuid("GPU-fixture-1".into())
    );
    for hidden in ["-1", ""] {
        let resolved = launch::resolve_visibility(&cpu, Some(OsStr::new(hidden)), &never).unwrap();
        assert_eq!(
            (resolved.clone(), resolved.pinned()),
            (ResolvedVisibility::Cpu, None)
        );
    }
    assert!(launch::resolve_visibility(&cpu, Some(OsStr::new("2")), &never).is_err());
    assert_eq!(never.calls.get(), 0);
    // Everything else is refused before any device work.
    let refuse = |observed: Option<&str>, smi: &Smi, needle: &str| {
        let e = launch::resolve_visibility(&gpu, observed.map(OsStr::new), smi).unwrap_err();
        assert!(
            format!("{e:#}").contains(needle),
            "{observed:?}: {e:#} (expected `{needle}`)"
        );
    };
    // SmartCluster allocated another GPU than the planned one.
    refuse(Some("0"), &Smi::listing(host), "not rank 1's planned GPU");
    refuse(Some("1"), &Smi::listing(host), "not rank 1's planned GPU");
    refuse(
        Some("7"),
        &Smi::listing(host),
        "which nvidia-smi does not report",
    );
    for malformed in ["1,2", "2,2", "-1", " 2", "2 ", "+2", "0x2", "", "10000"] {
        refuse(
            Some(malformed),
            &Smi::listing(host),
            "is not exactly one SmartCluster GPU index",
        );
    }
    refuse(None, &Smi::listing(host), "is unset");
    refuse(Some("GPU-fixture-0"), &never, "differs from the planned");
    refuse(Some("2"), &Smi::with(None), "nvidia-smi is not installed");
    refuse(
        Some("2"),
        &Smi::with(Some((false, String::new()))),
        "nvidia-smi --query-gpu=index,uuid failed: NVML failure",
    );
    refuse(
        Some("2"),
        &Smi::listing("2 GPU-fixture-1\n"),
        "malformed nvidia-smi line",
    );
    refuse(
        Some("2"),
        &Smi::listing("2, [N/A]\n"),
        "malformed nvidia-smi line",
    );
    refuse(
        Some("2"),
        &Smi::listing("2, GPU-fixture-1\n2, GPU-another-device\n"),
        "reports GPU index 2 twice",
    );
    refuse(
        Some("2"),
        &Smi::listing("1, GPU-fixture-1\n2, GPU-fixture-1\n"),
        "reports GPU GPU-fixture-1 twice",
    );
}
