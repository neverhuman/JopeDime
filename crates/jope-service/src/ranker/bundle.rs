//! Bounded inputs for fused-ranker verification (§5.8).
//!
//! File hashes and the declared qualification matrix do not authenticate a
//! bundle. The loader must verify the release signature before using either.

use jope_api::error::ErrorCode;
use p256::{
    ecdsa::{signature::Verifier, Signature, VerifyingKey},
    pkcs8::{DecodePublicKey, EncodePublicKey},
};
pub use safetensors::Dtype;
use safetensors::SafeTensors;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fmt, io::Read};
use std::{collections::BTreeSet, fs::File, path::Path};

// Bounded source fragments preserve this module's existing API and item scope.
include!("bundle/source_part1.rs");
include!("bundle/source_part2.rs");
include!("bundle/source_part3.rs");
