//! A minimal Apache Parquet writer and reader for flat tables (the `gp_distilled/v1` table).
//!
//! No Parquet crate is in `Cargo.lock`, and adding one would change the ADR-0004 evaluator
//! identity, so this module writes the subset it needs by hand:
//! - one row group, one data page (v1) per column;
//! - physical types BOOLEAN, INT64, DOUBLE, BYTE_ARRAY (UTF8), each REQUIRED or OPTIONAL;
//! - PLAIN values; definition levels in the RLE/bit-packed hybrid (bit width 1);
//! - pages either UNCOMPRESSED or ZSTD (one zstd frame per page);
//! - the footer (`FileMetaData`) in the Thrift compact protocol, with key/value metadata.
//!
//! The reader decodes exactly that subset (plus RLE runs in definition levels) and refuses
//! anything else, so a table can be reconciled against its raw source (strategy §8.2 E3).

use anyhow::{bail, ensure, Context, Result};
use std::collections::BTreeMap;

const MAGIC: &[u8; 4] = b"PAR1";
/// Decoded page size cap for the reader (a hostile footer cannot request more).
const MAX_PAGE: usize = 1 << 30;

include!("parquet/types_statistics.rs");
include!("parquet/thrift_write.rs");
include!("parquet/writer.rs");
include!("parquet/thrift_read.rs");
include!("parquet/stream_metadata.rs");
include!("parquet/stream_reader.rs");
