use super::{
    learned::AdmittedRanker,
    service::{self, BackendPlan},
    surrogate,
};
use crate::{
    dataset::register,
    engine::InProcessEngine,
    http::auth::{BearerToken, LocalAuth},
    ranker::cuda::release::load_r782_release,
};
use axum::{
    body::{to_bytes, Body},
    http::Request,
};
use jope_api::{error::ErrorCode, RunSpec};
use serde_json::{json, Value};
use std::{
    env,
    path::PathBuf,
    sync::atomic::Ordering,
    time::{Duration, Instant},
};
use tower::ServiceExt;

mod admission_tests;
mod surrogate_tests;
