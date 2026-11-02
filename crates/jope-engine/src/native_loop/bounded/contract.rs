//! `jope.native_bounded_run.v1`: a versioned bounded-run contract (W7-T02).
//!
//! The fixed `jope.native_loop.v1` contract ([`crate::native_loop::Config`]) keeps its exact
//! semantics; a legacy manifest offered here is refused as legacy, and a bounded-run
//! contract offered to the old loader fails its `deny_unknown_fields` parse. Nothing is
//! inferred from mutable flags: the frozen plan bytes are the only execution input.
//!
//! **Trainer join point (W7-T03).** A participant freezes `global_rank`, `world_size`
//! (= number of participants), `local_rank`, the host's `machine_id` and `boot_id`, and the
//! device `uuid`. `training::topology::Assignment` is derived from exactly these fields by
//! the pure `bounded::assignment::trainer_assignment` (`neural` builds), never from process
//! flags or `rank` alone. `DeviceAssignment::local_index` is the host-wide index the census
//! observed, not a CUDA ordinal: each worker sees only its own UUID, at ordinal 0.
//!
//! **Training partition (W7-T06).** A plan may select the execution partition its ranks train
//! under ([`TrainingSelection`]): partition v1, or the lossless v2 that three ranks need. The
//! record is derived for exactly the plan's world size, and the selection is admitted only
//! when the trainer's partition gate accepts it. A plan that selects none omits the field, so
//! every earlier plan keeps its exact bytes and digest. A later attempt cannot change it.
use super::{
    census::{validate_identity, HostIdentity},
    is_digest, is_name, is_relative, sign, verify, verifying_key,
};
use ed25519_dalek::SigningKey;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
};

type Check = std::result::Result<(), PlanRefusal>;
/// Whether validation checks the grant signatures: always for a plan, never for an authored
/// contract awaiting `grant sign`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Signatures {
    Verify,
    Skip,
}
macro_rules! need {
    ($cond:expr, $kind:ident, $($msg:tt)+) => {
        if !$cond {
            return Err(PlanRefusal::$kind(format!($($msg)+)));
        }
    };
}
fn invalid(e: impl std::fmt::Display) -> PlanRefusal {
    PlanRefusal::Invalid(format!("{e:#}"))
}

mod api;
mod lifecycle;
mod model;
mod validation;

pub use model::*;

pub fn sign_grant(authority: &SigningKey, body: GrantBody) -> anyhow::Result<SignedGrant> {
    Ok(SignedGrant {
        signature: sign(authority, GRANT_DOMAIN, &body)?,
        body,
    })
}
