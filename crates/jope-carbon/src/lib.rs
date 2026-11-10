//! Carbon serverless cell runtime, adaptive multi-class execution planner, authenticated state
//! sealer, and Enterprise/CI server foundation ([ADR-0007]).
//!
//! Feature-gated compilation keeps `carbon-edge` (`--no-default-features --features edge`)
//! physically separate from `jope-service`, `jope-runtime`, `feat-math`, and neural weights.
//!
//! [ADR-0007]: ../../docs/adr/ADR-0007-carbon-lambda-cells.md

pub mod error;
pub mod planner;
pub mod role;
pub mod seal;
pub mod spot_advisor;
pub mod tls;

pub use error::{CarbonError, CarbonGuidance, CarbonResult, CARBON_DOCS_URL};
pub use planner::{
    estimate_working_set_mb, plan_carbon_execution, CarbonExecutionMode, CarbonPlanDecision,
    CarbonPlanInput, EvalMemoryClass, ENGINE_MEMORY_MB, LAMBDA_ARM64_USD_PER_GB_SEC,
    LAMBDA_USD_PER_INVOCATION,
};
pub use role::{CarbonEffortTier, CarbonProductTier, CarbonRole};
#[cfg(feature = "edge")]
pub use seal::{CarbonSealKey, SealedCheckpointEnvelope};
pub use spot_advisor::{
    evaluate_spot_hybrid_gates, MonthlyWindowMetrics, SpotAdvisorRecommendation,
    SpotAdvisorThresholds, SpotGateVerdict,
};
#[cfg(feature = "edge")]
pub use tls::carbon_https_agent;
