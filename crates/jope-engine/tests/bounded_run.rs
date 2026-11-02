//! W7-T01/T02/T04 fixtures: typed host census preflight refusals, the versioned
//! bounded-run contract, Plan, read-only Status/Verify and idempotent Cancel, the
//! controller CLI, and authenticated rendezvous, full-group admission and result
//! authentication. In-process and same-host only: no host is contacted, no GPU state is
//! read and no numerical work is started. These are fixtures, not the three-host proof
//! (W7-T07).
#[path = "support/bounded_fixture.rs"]
mod fixture;
use fixture::*;
use jope_parallel::native_loop::bounded::{
    census::{preflight, HostCensus, OwnerKind, Ownership, RefusalReason},
    contract::{BoundedRunContract, PlanRefusal, WorkKind},
    controller::{self, Coordinator, Disposition, Liveness, Outcome, RankState},
    rendezvous::{self, Connected, GroupRefusal, Hello, HelloBody, RefusalCause, Rendezvous},
    transport::DeadlineSocket,
};
use std::{
    fs,
    os::unix::net::UnixStream,
    path::{Path, PathBuf},
    process::Command,
};

fn plan_sha(c: &BoundedRunContract) -> String {
    sha(&bytes(c))
}
fn reasons(c: &BoundedRunContract, census: &HostCensus, now: u64) -> Vec<RefusalReason> {
    preflight(c, &plan_sha(c), census, "census", now)
        .unwrap_err()
        .reasons
}

include!("bounded_run/preflight.rs");
include!("bounded_run/plan.rs");
include!("bounded_run/status.rs");
include!("bounded_run/cli.rs");
include!("bounded_run/rendezvous.rs");
include!("bounded_run/socket_rendezvous.rs");
include!("bounded_run/window.rs");
include!("bounded_run/training_partition.rs");
