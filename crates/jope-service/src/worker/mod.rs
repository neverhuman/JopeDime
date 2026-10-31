//! Bounded worker processes, attempt fencing and the existing protocol-1 frame codec.

pub mod evaluate;
pub mod frame;
mod pool;
mod process;
mod protocol;
mod runtime;

pub use pool::{PoolConfig, WorkerPool};
pub use protocol::{Attempt, Limits, Request, Response, Scores, WorkerFailure};
pub use runtime::serve;

#[cfg(test)]
mod tests;
