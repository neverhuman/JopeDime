//! Progress events + sinks. Events are emitted by every phase and consumed by the live TUI through
//! an in-process channel or by a lightweight human stderr sink.
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::sync::Mutex;

include!("progress/events.rs");
include!("progress/sinks.rs");
include!("progress/display.rs");
#[cfg(test)]
mod tests;
