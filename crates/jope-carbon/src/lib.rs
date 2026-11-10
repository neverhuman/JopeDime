//! Carbon serverless cell runtime, adaptive multi-class execution planner, authenticated state
//! sealer, and Enterprise/CI server foundation ([ADR-0007]).
//!
//! Feature-gated compilation keeps `carbon-edge` (`--no-default-features --features edge`)
//! physically separate from `jope-service`, `jope-runtime`, `feat-math`, and neural weights.
//!
//! [ADR-0007]: ../../docs/adr/ADR-0007-carbon-lambda-cells.md

pub mod cas;
#[cfg(feature = "edge")]
pub mod client;
#[cfg(feature = "edge")]
pub mod edge_router;
#[cfg(feature = "emu")]
pub mod emu;
#[cfg(feature = "engine")]
pub mod engine_worker;
pub mod error;
#[cfg(feature = "eval")]
pub mod eval_worker;
#[cfg(feature = "edge")]
pub mod lambda_runtime;
pub mod planner;
#[cfg(feature = "edge")]
pub mod presign;
pub mod role;
pub mod seal;
#[cfg(feature = "serve")]
pub mod serve;
pub mod spot_advisor;
pub mod tls;

#[cfg(feature = "edge")]
pub use cas::CarbonHttpCas;
pub use cas::{CarbonCasStore, CarbonFsCas};
#[cfg(feature = "edge")]
pub use client::{CarbonClientPreparedDataset, CarbonProbedEndpoint, CarbonThinClient};
#[cfg(feature = "edge")]
pub use edge_router::{
    CarbonEdgeConfig, CarbonEdgeRouter, CarbonEngineInvoker, CarbonEngineWireCommand,
    CarbonGatewayRequest, CarbonGatewayResponse, CarbonHttpEngineInvoker,
};
#[cfg(feature = "emu")]
pub use emu::{CarbonEmuConfig, CarbonEmuTelemetrySnapshot, CarbonLocalLambdaEmulator};
#[cfg(feature = "engine")]
pub use engine_worker::CarbonEngineWorker;
pub use error::{CarbonError, CarbonGuidance, CarbonResult, CARBON_DOCS_URL};
#[cfg(feature = "eval")]
pub use eval_worker::{
    CarbonEvalInvoker, CarbonEvalWireRequest, CarbonEvalWorker, CarbonHttpEvalInvoker,
};
#[cfg(feature = "edge")]
pub use lambda_runtime::{run_lambda_custom_runtime_loop, CarbonLambdaInvocation};
pub use planner::{
    estimate_working_set_mb, plan_carbon_execution, CarbonExecutionMode, CarbonPlanDecision,
    CarbonPlanInput, EvalMemoryClass, ENGINE_MEMORY_MB, LAMBDA_ARM64_USD_PER_GB_SEC,
    LAMBDA_USD_PER_INVOCATION,
};
#[cfg(feature = "edge")]
pub use presign::CarbonPresigner;
pub use role::{CarbonEffortTier, CarbonProductTier, CarbonRole};
#[cfg(feature = "edge")]
pub use seal::{CarbonSealKey, SealedCheckpointEnvelope};
#[cfg(feature = "serve")]
pub use serve::{
    CarbonEnterpriseServeConfig, CarbonEnterpriseServer, CarbonEnterpriseServerHandle,
};
pub use spot_advisor::{
    evaluate_spot_hybrid_gates, MonthlyWindowMetrics, SpotAdvisorRecommendation,
    SpotAdvisorThresholds, SpotGateVerdict,
};
#[cfg(feature = "edge")]
pub use tls::carbon_https_agent;
