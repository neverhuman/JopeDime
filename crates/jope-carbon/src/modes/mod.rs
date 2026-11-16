//! Execution modes, adaptive multi-class planner, 3-Gate Spot-Hybrid advisor, Enterprise VPC server,
//! and local multi-cell AWS Lambda + S3 CAS emulator (`ADR-0007`).

#[cfg(feature = "emu")]
pub mod emu;
#[cfg(feature = "emu")]
pub mod emu_aws_services;
pub mod planner;
#[cfg(feature = "serve")]
pub(crate) mod runtime_driver;
#[cfg(feature = "serve")]
pub mod serve;
pub mod spot_advisor;
