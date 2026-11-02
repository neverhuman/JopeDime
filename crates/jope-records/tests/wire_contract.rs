//! Integration and conformance tests for `jope_records::wire_contract` (`jope.canonical-json/v1`, `I02`).

use jope_records::pyjson::{self, Value as PyValue};
use jope_records::wire_contract::{
    accept_measured_claim, parse_outcome_wire, CanonicalPayload, DomainSeparatedId, EnvelopeDomain,
    EnvelopeFrame, ExactF32, ExactF64, HexU64, InferenceEnvelopeAdapter, LegacyBridgeRecord,
    LegacyHashScheme, MeasuredEvidenceKind, MeasuredEvidenceProof, MissingReason,
    NativeEnvelopeAdapter, ParsedMeasuredClaim, ParsedOutcomeWire, SignatureEnvelope,
    StorageEnvelopeAdapter, VerifiedMeasuredOutcome, WireContractError,
    CANONICAL_JSON_ENVELOPE_SCHEMA, DOMAIN_SEPARATED_ID_FORMULA, SAFE_INT_MAX, SAFE_INT_MIN,
};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
}

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/wire-contract-v1")
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    let out = h.finalize();
    out.iter().map(|b| format!("{b:02x}")).collect()
}

#[path = "wire_contract/adapters.rs"]
mod adapters;
#[path = "wire_contract/canonical_vectors.rs"]
mod canonical_vectors;
#[path = "wire_contract/conformance.rs"]
mod conformance;
