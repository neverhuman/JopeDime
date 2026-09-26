#![cfg(feature = "neural")]
//! W7-T03: communicator identity separated from the local CUDA device (CPU
//! only; no device is enumerated or selected here).
//!
//! The pre-change runtime called `cudaSetDevice(rank)`, so global rank 2 on a
//! one-GPU host selected a nonexistent CUDA index 2. That call is native and
//! needs a device, so it is recorded by source trace, not as a failing test.

use jope_parallel::training::topology::{
    bind, validate, Assignment, Binding, LocalDevices, Participant, SCHEMA,
};
use serde_json::json;

const PLAN: &str = "5555555555555555555555555555555555555555555555555555555555555555";

fn uuid(n: u8) -> String {
    format!("GPU-{0:02x}{0:02x}{0:02x}{0:02x}-{0:02x}{0:02x}-{0:02x}{0:02x}-{0:02x}{0:02x}-{0:02x}{0:02x}{0:02x}{0:02x}{0:02x}{0:02x}", n)
}

fn participant(rank: usize, host: &str, local: usize, index: usize, gpu: u8) -> Participant {
    Participant {
        global_rank: rank,
        host_id: host.into(),
        boot_id: format!("boot-{host}"),
        local_rank: local,
        device_index: index,
        gpu_uuid: uuid(gpu),
    }
}

/// One GPU on each of three hosts (the W7 target: xbabe1, xbabe2, xbabe3).
fn three_hosts() -> Assignment {
    Assignment {
        schema: SCHEMA.into(),
        plan_sha256: PLAN.into(),
        world: 3,
        participants: vec![
            participant(0, "xbabe2", 0, 0, 2),
            participant(1, "xbabe3", 0, 0, 3),
            participant(2, "xbabe1", 0, 0, 1),
        ],
    }
}

fn view(host: &str, visible: &[u8]) -> LocalDevices {
    LocalDevices {
        host_id: host.into(),
        boot_id: format!("boot-{host}"),
        visible: visible.iter().map(|&g| uuid(g)).collect(),
    }
}

#[test]
fn one_gpu_on_each_of_three_hosts_maps_global_ranks_to_local_index_zero() {
    let a = three_hosts();
    validate(&a).unwrap();
    for (rank, host, gpu) in [(0, "xbabe2", 2), (1, "xbabe3", 3), (2, "xbabe1", 1)] {
        let b = bind(&a, rank, 3, &view(host, &[gpu])).unwrap();
        assert_eq!(
            (b.global_rank(), b.world(), b.local_rank(), b.device_index()),
            (rank, 3, 0, 0),
            "global rank {rank} must use its host's only device"
        );
        assert_eq!(b.gpu_uuid(), Some(uuid(gpu).as_str()));
        let receipt = b.receipt();
        assert_eq!(receipt["binding"], "assigned");
        assert_eq!(receipt["plan_sha256"], PLAN);
    }
    // Global rank 2 does not imply CUDA index 2 (what cudaSetDevice(rank) did).
    assert_ne!(
        bind(&a, 2, 3, &view("xbabe1", &[1]))
            .unwrap()
            .device_index(),
        2
    );
}

#[test]
fn two_devices_on_one_host_bind_by_uuid_not_by_rank_order() {
    // Two workers on one host, each restricted to its own device, and a
    // second host whose worker sees one device.
    let a = Assignment {
        schema: SCHEMA.into(),
        plan_sha256: PLAN.into(),
        world: 3,
        participants: vec![
            participant(0, "h1", 1, 0, 11),
            participant(1, "h1", 0, 0, 10),
            participant(2, "h2", 0, 0, 20),
        ],
    };
    let b0 = bind(&a, 0, 3, &view("h1", &[11])).unwrap();
    let b1 = bind(&a, 1, 3, &view("h1", &[10])).unwrap();
    assert_eq!((b0.local_rank(), b1.local_rank()), (1, 0));
    assert_eq!(b0.gpu_uuid(), Some(uuid(11).as_str()));
    // A worker that can see both GPUs of its host does not own exactly one.
    assert!(bind(&a, 0, 3, &view("h1", &[11, 10])).is_err());
}

#[test]
fn invalid_assignments_are_refused_before_any_communicator() {
    type Edit = Box<dyn Fn(&mut Assignment)>;
    let cases: Vec<(&str, Edit)> = vec![
        (
            "duplicate global rank",
            Box::new(|a| a.participants[2].global_rank = 1),
        ),
        (
            "rank outside the world",
            Box::new(|a| a.participants[2].global_rank = 3),
        ),
        (
            "duplicate GPU UUID",
            Box::new(|a| a.participants[2].gpu_uuid = a.participants[0].gpu_uuid.clone()),
        ),
        (
            "participant count differs from world",
            Box::new(|a| {
                a.participants.pop();
            }),
        ),
        ("world differs from participants", Box::new(|a| a.world = 4)),
        (
            "local rank reused on one host",
            Box::new(|a| {
                a.participants[1].host_id = "xbabe2".into();
                a.participants[1].boot_id = "boot-xbabe2".into();
                a.participants[1].device_index = 1;
            }),
        ),
        (
            "local ranks not contiguous",
            Box::new(|a| a.participants[0].local_rank = 1),
        ),
        (
            "one host under two boots",
            Box::new(|a| {
                a.participants[1].host_id = "xbabe2".into();
                a.participants[1].boot_id = "other-boot".into();
                a.participants[1].device_index = 1;
                a.participants[1].local_rank = 1;
            }),
        ),
        (
            "malformed UUID",
            Box::new(|a| a.participants[0].gpu_uuid = "GPU-0".into()),
        ),
        (
            "missing host identity",
            Box::new(|a| a.participants[0].host_id.clear()),
        ),
        ("unknown schema", Box::new(|a| a.schema = "other".into())),
        (
            "malformed plan digest",
            Box::new(|a| a.plan_sha256 = "abc".into()),
        ),
    ];
    for (case, edit) in cases {
        let mut a = three_hosts();
        edit(&mut a);
        assert!(validate(&a).is_err(), "{case}: accepted");
        assert!(
            bind(&a, 0, a.world, &view("xbabe2", &[2])).is_err(),
            "{case}: bound"
        );
    }
}

#[test]
fn a_worker_whose_local_view_disagrees_is_refused() {
    let a = three_hosts();
    let cases: Vec<(&str, usize, usize, LocalDevices)> = vec![
        // The assigned index is not visible (no device, or index past the end).
        ("no visible device", 0, 3, view("xbabe2", &[])),
        (
            "assigned index not visible",
            0,
            3,
            LocalDevices {
                visible: vec![],
                ..view("xbabe2", &[])
            },
        ),
        // Mismatched visibility: a different GPU at the assigned index.
        ("wrong device at index", 0, 3, view("xbabe2", &[9])),
        // Another participant's GPU is visible to this worker.
        ("sees another rank's GPU", 0, 3, view("xbabe2", &[2, 3])),
        ("wrong host", 0, 3, view("xbabe3", &[2])),
        (
            "wrong boot",
            0,
            3,
            LocalDevices {
                boot_id: "rebooted".into(),
                ..view("xbabe2", &[2])
            },
        ),
        ("unassigned rank", 3, 3, view("xbabe2", &[2])),
        ("world differs", 0, 4, view("xbabe2", &[2])),
    ];
    for (case, rank, world, local) in cases {
        assert!(bind(&a, rank, world, &local).is_err(), "{case}: bound");
    }
    // Invalid local index in the assignment itself: index 1 on a one-GPU host.
    let mut one_gpu = three_hosts();
    one_gpu.participants[0].device_index = 1;
    validate(&one_gpu).unwrap();
    let error = bind(&one_gpu, 0, 3, &view("xbabe2", &[2])).unwrap_err();
    assert!(error.to_string().contains("not visible"), "{error}");
}

#[test]
fn the_legacy_single_host_mapping_is_explicit_and_labelled() {
    let b = Binding::legacy_single_host(2, 8).unwrap();
    assert_eq!(
        (b.global_rank(), b.device_index(), b.gpu_uuid()),
        (2, 2, None)
    );
    assert_eq!(b.receipt()["binding"], "legacy_single_host");
    assert!(b.receipt()["plan_sha256"].is_null());
    assert!(Binding::legacy_single_host(8, 8).is_err());
    assert!(Binding::legacy_single_host(0, 0).is_err());
    // The assignment round-trips through its serialized form and refuses
    // unknown fields.
    let a = three_hosts();
    let value = serde_json::to_value(&a).unwrap();
    assert_eq!(
        serde_json::from_value::<Assignment>(value.clone()).unwrap(),
        a
    );
    let mut extra = value;
    extra["participants"][0]["cuda_index"] = json!(0);
    assert!(serde_json::from_value::<Assignment>(extra).is_err());
}
