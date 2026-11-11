//! Pure-Rust, WebAssembly-compatible client preparation crate for Carbon ([ADR-0007]):
//! deterministic `FIT` / `HOLDOUT` splitting, `FIT`-only landmark conditioning (`6,368`-byte
//! `JLMB` v1 wire format), and standalone `JMAT` v1 matrix encoding.
//!
//! [ADR-0007]: ../../docs/adr/ADR-0007-carbon-lambda-cells.md

pub mod conditioning;
pub mod cuen;
pub mod error;
pub mod jmat_lite;
pub mod landmarks;
pub mod split;

pub use conditioning::{FitLandmarkBundle, FIT_LANDMARK_WIRE_BYTES};
pub use cuen::{
    build_cudk_aad, build_cuen_aad, pack_cudk_v1, pack_cuen_v1, parse_cudk_v1, parse_cuen_v1,
    CudkParsedEnvelope, CuenParsedEnvelope, CuenPayloadKind, CUDK_WIRE_BYTES, CUEN_HEADER_BYTES,
    CUEN_NONCE_BYTES, CUEN_TAG_BYTES, P256_UNCOMPRESSED_PUBKEY_BYTES,
};
pub use error::{PrepError, PrepGuidance, PrepResult, PREP_DOCS_URL};
pub use jmat_lite::{
    encode_jmat_v1, split_and_encode_jmat_v1, PrepCell, PrepColumn, PrepColumnKind, PrepMatrix,
    PrepTable, SplitJmatArtifacts,
};
pub use split::{SplitPlan, DEFAULT_HOLDOUT_FRACTION, MAX_LANDMARK_FIT_ROWS};
