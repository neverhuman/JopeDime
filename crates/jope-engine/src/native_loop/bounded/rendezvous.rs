//! Authenticated rendezvous and verified artifact/result exchange (W7-T04).
//!
//! Replaces the shared-filesystem assumption of the legacy same-allocation NCCL path
//! (`neural::Nccl::init` polls an unauthenticated `nccl.id` file). Here every message is
//! a bounded length-prefixed frame carrying a domain-separated ed25519 signature from a key
//! frozen in the plan, bound to run, attempt, plan digest, a per-open rendezvous identity
//! (which includes the coordinator boot), the participant's boot, grant and input closure.
//! The whole synchronous group is admitted, or refused, before any rank may start; a
//! refusal is terminal for the rendezvous and commits nothing.
//!
//! Transport-independent verification lives in [`Rendezvous`], [`verify_challenge`],
//! [`verify_admission`] and [`verify_result`]. [`gather`] / [`join`] run the handshake over
//! any [`ControlStream`] from an [`Acceptor`]: the cross-host TLS transport with pinned
//! certificate identity ([`super::transport`]), or a same-host Unix socket for fixtures
//! ([`gather_unix`] / [`join_unix`]). A peer's channel identity ([`PeerIdentity`]) is bound
//! to the rank it claims. The admission is delivered only by the controller after it is
//! journaled.
use super::{
    census::RuntimeIdentity,
    contract::{BoundedRunContract, ClosureEntry},
    digest, is_digest, sign,
    transport::DeadlineSocket,
    verify,
};
use anyhow::{ensure, Context, Result};
use ed25519_dalek::SigningKey;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs,
    io::{Read, Write},
    os::unix::{
        fs::OpenOptionsExt,
        net::{UnixListener, UnixStream},
    },
    path::Path,
    time::{Duration, Instant},
};

pub const CHALLENGE_SCHEMA: &str = "jope.native_rendezvous_challenge.v1";
pub const HELLO_SCHEMA: &str = "jope.native_rendezvous_hello.v1";
pub const ADMISSION_SCHEMA: &str = "jope.native_group_admission.v1";
pub const RESULT_SCHEMA: &str = "jope.native_rank_result.v1";
const CHALLENGE_DOMAIN: &[u8] = b"jope.native_rendezvous_challenge.signature.v1\0";
const HELLO_DOMAIN: &[u8] = b"jope.native_rendezvous_hello.signature.v1\0";
const ADMISSION_DOMAIN: &[u8] = b"jope.native_group_admission.signature.v1\0";
const RESULT_DOMAIN: &[u8] = b"jope.native_rank_result.signature.v1\0";
/// Opaque communicator bootstrap (for example an NCCL unique id) carried only by rank 0's
/// hello and redistributed only inside a signed full-group admission.
pub const MAX_BOOTSTRAP_BYTES: usize = 256;

mod protocol;
pub use protocol::*;
mod handshake;
pub use handshake::*;
mod verify;
pub use verify::*;
mod exchange;
pub use exchange::*;
