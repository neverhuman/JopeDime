//! Carbon thin client SDK (`client`), CLI run & benchmark harness (`runner`), 16-vector
//! live/CI adversarial security red-team suite (`redteam`), and declarative CloudFormation
//! packager/deployer (`deploy`).

pub mod client;
#[cfg(feature = "engine")]
pub mod deploy;
pub mod redteam;
pub mod runner;
#[cfg(feature = "emu")]
pub mod spec_report;
