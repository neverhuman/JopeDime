//! T2 pre-stage sentinel harness.
//!
//! PUB-06 tests scan a bounded real service run and its public outputs. The synthetic
//! projection cases remain useful scanner unit controls, separate from the runtime proof.

use axum::{
    body::{to_bytes, Body},
    http::{header, Method, Request, StatusCode},
};
use jope_api::{decimal::DecimalU64, ids::RunId, PublicEvent};
use jope_service::http::{
    auth::{BearerToken, LocalAuth},
    router,
};
use jope_service::public::{
    phrase::{CatalogEntry, EventDetails, Template},
    project::{AdmittedProposal, EvaluationOrdinal, ProblemDigest, Projector},
};
#[cfg(all(feature = "failpoints", debug_assertions))]
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Duration, Instant},
};
use tower::ServiceExt;

mod http;
mod real_run;
mod scanner;

use scanner::{scan_surfaces, Surface, SENTINELS};
