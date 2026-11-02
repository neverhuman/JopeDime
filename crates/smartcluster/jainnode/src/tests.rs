use std::{collections::BTreeMap, future::pending, path::PathBuf, sync::Mutex};

use ed25519_dalek::SigningKey;
use jain_smartcluster_fabric::{
    image_signed_bytes, projection_signed_bytes, ArtifactMount, DomainSignature, GpuRequirement,
    OutputDeclaration, ResourceLimits, SignatureAuthority, SignedWorkUnitProjection,
    VerifiedWorkUnit, WorkUnitProjection,
};
use tokio::sync::{mpsc, Notify};
use ulid::Ulid;

use super::*;
use crate::docker::{DockerError, DockerLogChunk, DockerLogEvent, DockerRunSpec, ExitStatus};

#[path = "dispatch_tests.rs"]
mod dispatch_tests;

include!("tests/support_mocks.rs");
include!("tests/support_helpers.rs");
include!("tests/execution.rs");
