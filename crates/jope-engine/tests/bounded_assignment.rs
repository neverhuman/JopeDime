#![cfg(feature = "neural")]
//! W7-T02 × W7-T03 join (CPU only; no device is enumerated or selected): a frozen
//! bounded-run plan maps, by a pure function, to exactly the trainer Assignment that
//! `training::topology::validate` accepts, and a plan/assignment mismatch is refused.
//! Synthetic identities; not the three-host proof (W7-T07..T09).
#[path = "support/bounded_fixture.rs"]
mod fixture;
use fixture::*;
use jope_parallel::{
    native_loop::bounded::{
        assignment::{
            bound_worker, trainer_assignment, verify_trainer_assignment,
            worker_cuda_visible_devices, RESTRICTED_DEVICE_INDEX,
        },
        contract::{BoundedRunContract, DeviceAssignment},
        controller, launch,
    },
    training::{
        communicator_source,
        topology::{self, Assignment, LocalDevices, Participant, SCHEMA},
        verify_admitted_bootstrap, CommunicatorSource,
    },
};
use std::ffi::OsStr;

fn uuid(n: usize) -> String {
    format!("GPU-{n:08x}-0000-4000-8000-{n:012x}")
}
/// The fixture plan with nvidia-smi-form GPU UUIDs. `layout[i]` is the host of rank `i`;
/// ranks sharing a host get consecutive local ranks and distinct host GPU indices.
fn plan(layout: &[usize]) -> (BoundedRunContract, Keys) {
    let k = keys(layout.len());
    let mut c = contract(now_ms(), layout.len(), &k);
    let mut next_local = std::collections::BTreeMap::new();
    for (rank, host) in layout.iter().enumerate() {
        let local = next_local.entry(*host).or_insert(0u32);
        let p = &mut c.participants[rank];
        p.host = identity(*host);
        p.local_rank = *local;
        p.device = Some(DeviceAssignment {
            uuid: uuid(0x10 + rank),
            local_index: *local,
        });
        *local += 1;
    }
    resign(&mut c, &k);
    c.validate_structure().unwrap();
    (c, k)
}
fn expected(c: &BoundedRunContract) -> Assignment {
    Assignment {
        schema: SCHEMA.into(),
        plan_sha256: sha(&bytes(c)),
        world: c.participants.len(),
        participants: c
            .participants
            .iter()
            .map(|p| Participant {
                global_rank: p.global_rank as usize,
                host_id: p.host.machine_id.clone(),
                boot_id: p.host.boot_id.clone(),
                local_rank: p.local_rank as usize,
                device_index: 0,
                gpu_uuid: p.device.as_ref().unwrap().uuid.clone(),
            })
            .collect(),
    }
}
/// What a worker launched under the visibility contract sees.
fn restricted_view(c: &BoundedRunContract, rank: u32) -> LocalDevices {
    let p = &c.participants[rank as usize];
    LocalDevices {
        host_id: p.host.machine_id.clone(),
        boot_id: p.host.boot_id.clone(),
        visible: vec![worker_cuda_visible_devices(c, rank).unwrap()],
    }
}

#[path = "bounded_assignment/assignment_tests.rs"]
mod assignment_tests;
#[path = "bounded_assignment/selection_tests.rs"]
mod selection_tests;
#[path = "bounded_assignment/worker_tests.rs"]
mod worker_tests;
