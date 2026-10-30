//! Background-only output sinks.

pub mod jsonl;
#[cfg(feature = "parquet")]
pub mod parquet;
