#![deny(unsafe_code)]

pub mod actor;
pub mod auth;
pub mod capacity;
pub mod cgroup;
pub mod config;
pub mod control;
pub mod event_log;
pub mod executor;
pub mod fanout;
pub mod gpu;
pub mod health;
pub mod overload;
pub mod psi;
pub mod recovery;
pub mod scheduler;
pub mod server;
pub mod store;

pub mod resource_policy;

pub mod fit_child_policy;

/// Lowercase hex SHA-256 of a complete byte buffer.
pub fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(bytes))
}
