//! JopeDime v1 engine service library.
//!
//! This crate exposes the service-side components for configuration, dataset
//! registration and discovery, run coordination, persistence, HTTP handling,
//! worker execution, and telemetry. The modules below document the purpose of
//! each public subsystem.

/// Service configuration loading and validation.
pub mod config;
/// The local daemon's owner-scoped discovery record.
pub mod daemon_file;
/// Dataset registration, metadata, and loading support.
pub mod dataset;
/// Dataset discovery and candidate proposal workflows.
pub mod discover;
/// Engine lifecycle, locking, and in-process service entry points.
pub mod engine;
/// Durable service event records and event-log operations.
pub mod events;
/// Model and source artifact export support.
pub mod export;
/// Resource observation and admission controls for service work.
pub mod governor;
/// HTTP routes and request handlers.
pub mod http;
/// Idempotency keys and execution-result tracking.
pub mod idempotency;
/// Lexing and parsing support for the JopeDime program language.
pub mod lex;
/// Objective evaluation types and helpers.
pub mod objective;
/// Authenticated embedded proposal runtime and its appliance capability record.
pub mod proposal;
/// Public response projection and claim-level filtering.
pub mod public;
/// Learned ranking, feature extraction, and candidate admission.
pub mod ranker;
/// Research-record mode types and serialization support.
#[cfg(feature = "research")]
pub mod research;
/// Run orchestration and lifecycle state.
pub mod run;
/// Candidate sampling strategies.
pub mod sampler;
/// Persistent service state and storage interfaces.
pub mod store;
/// Service telemetry collection and reporting.
pub mod telemetry;
/// User-facing and diagnostic warning types.
pub mod warnings;
/// Isolated worker protocol, pool management, and execution runtime.
pub mod worker;

/// Test-only capture points used by the PUB-06 leak acceptance harness.
#[cfg(all(feature = "failpoints", debug_assertions))]
pub mod pub06_test {
    use std::sync::{Mutex, OnceLock};

    static LOGS: OnceLock<Mutex<Vec<String>>> = OnceLock::new();

    pub(crate) fn record_log(line: String) {
        LOGS.get_or_init(Default::default)
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .push(line);
    }

    /// Drain run-service diagnostics emitted by the test-only worker path.
    pub fn take_logs() -> Vec<String> {
        std::mem::take(
            &mut *LOGS
                .get_or_init(Default::default)
                .lock()
                .unwrap_or_else(|error| error.into_inner()),
        )
    }
}
