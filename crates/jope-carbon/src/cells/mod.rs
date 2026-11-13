//! Isolated serverless Lambda cells (`edge`, `engine`, and `eval`) and the shared custom runtime loop.

#[cfg(feature = "edge")]
pub mod edge;
#[cfg(feature = "engine")]
pub mod engine;
#[cfg(feature = "edge")]
pub mod engine_invoker;
#[cfg(feature = "engine")]
pub mod eval;
#[cfg(feature = "edge")]
pub mod lambda_runtime;
