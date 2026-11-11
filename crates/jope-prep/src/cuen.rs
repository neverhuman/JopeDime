//! Pure-Rust, WebAssembly-compatible binary wire framing and canonical AAD builder for Carbon
//! per-user zero-knowledge envelope encryption (`CUEN\x01` payload envelopes and `CUDK\x01`
//! `ECDH-ES+HKDF-SHA256+AES-256-GCM` wrapped `U_DEK` envelopes).
//!
//! Keeping the canonical byte layout and length-prefixed AAD construction in `jope-prep` guarantees
//! byte-for-byte interoperability between browser WebAssembly clients (calling WebCrypto
//! `SubtleCrypto` `ECDH P-256` / `HKDF` / `AES-GCM`) and native Rust clients / Lambda `engine`
//! cells (`jope-carbon::security::user_envelope`) without linking C libraries into `wasm32`.

use sha2::{Digest, Sha256};

use crate::error::{PrepError, PrepResult};

/// Magic prefix for a `CUEN` v1 per-user encrypted artifact envelope (`5` bytes).
pub const CUEN_V1_MAGIC: &[u8; 5] = b"CUEN\x01";
/// Fixed nonce length (`96` bits) for `AES-256-GCM` in `CUEN` v1 and `CUDK` v1.
pub const CUEN_NONCE_BYTES: usize = 12;
/// Fixed `AES-256-GCM` authentication tag length (`128` bits).
pub const CUEN_TAG_BYTES: usize = 16;
/// Fixed header size for `CUEN` v1 (`5` magic + `1` kind + `2` reserved + `12` nonce + `32` plaintext sha256 = `52` B).
pub const CUEN_HEADER_BYTES: usize = 52;

/// Magic prefix for a `CUDK` v1 wrapped per-user `U_DEK` envelope (`5` bytes).
pub const CUDK_V1_MAGIC: &[u8; 5] = b"CUDK\x01";
/// Uncompressed NIST P-256 SEC1 public point byte length (`0x04 || X || Y` = `65` B).
pub const P256_UNCOMPRESSED_PUBKEY_BYTES: usize = 65;
/// Wrapped `U_DEK` ciphertext + `AES-256-GCM` tag length (`32 + 16 = 48` B).
pub const CUDK_CIPHERTEXT_TAG_BYTES: usize = 48;
/// Total fixed wire length of a `CUDK` v1 wrapped `U_DEK` (`5 + 3 + 65 + 12 + 48 = 133` B).
pub const CUDK_WIRE_BYTES: usize = 133;

/// `CuenPayloadKind` is an enum discriminating the artifact slot of a `CUEN` v1 ciphertext so an
/// attacker or misconfigured gateway cannot swap `fit.jmat`, `holdout.jmat`, `JLMB` landmarks, or
/// `/predict` request/response payloads even within the same `(tenant_id, run_id)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum CuenPayloadKind {
    /// Encrypted `FIT` `JMAT` v1 training partition (`fit.jmat`).
    FitJmat = 1,
    /// Encrypted `HOLDOUT` `JMAT` v1 outer evaluation partition (`holdout.jmat`).
    HoldoutJmat = 2,
    /// Encrypted `6,368`-byte `JLMB` v1 landmark conditioning bundle.
    LandmarkJlmb = 3,
    /// Encrypted `/predict` feature matrix payload (`client -> engine`).
    PredictFeatures = 4,
    /// Encrypted `/predict` scalar output vector (`engine -> client`).
    PredictOutputs = 5,
}

impl CuenPayloadKind {
    /// Decode a wire discriminant byte into `CuenPayloadKind`.
    pub const fn from_u8(raw: u8) -> Option<Self> {
        match raw {
            1 => Some(Self::FitJmat),
            2 => Some(Self::HoldoutJmat),
            3 => Some(Self::LandmarkJlmb),
            4 => Some(Self::PredictFeatures),
            5 => Some(Self::PredictOutputs),
            _ => None,
        }
    }

    /// Canonical ASCII label bound into `HKDF` info and `AES-256-GCM` AAD.
    pub const fn domain_label(self) -> &'static str {
        match self {
            Self::FitJmat => "fit_jmat",
            Self::HoldoutJmat => "holdout_jmat",
            Self::LandmarkJlmb => "landmark_jlmb",
            Self::PredictFeatures => "predict_features",
            Self::PredictOutputs => "predict_outputs",
        }
    }
}

/// `CuenParsedEnvelope` is a struct borrowing the verified header fields and `ciphertext || tag`
/// slice from a `CUEN\x01` wire buffer prior to `AES-256-GCM` decryption.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CuenParsedEnvelope<'a> {
    pub payload_kind: CuenPayloadKind,
    pub nonce: [u8; CUEN_NONCE_BYTES],
    pub plaintext_sha256: [u8; 32],
    pub ciphertext_and_tag: &'a [u8],
}

/// Build the canonical length-prefixed `AES-256-GCM` AAD for a `CUEN\x01` payload envelope,
/// cryptographically binding `(magic, payload_kind, tenant_id, run_id, nonce, plaintext_sha256)`.
pub fn build_cuen_aad(
    payload_kind: CuenPayloadKind,
    tenant_id: &str,
    run_id: &str,
    nonce: &[u8; CUEN_NONCE_BYTES],
    plaintext_sha256: &[u8; 32],
) -> PrepResult<Vec<u8>> {
    if tenant_id.is_empty() || run_id.is_empty() {
        return Err(PrepError::InvalidUserEnvelope {
            reason: "tenant_id and run_id must be non-empty",
        });
    }
    let tenant_len =
        u32::try_from(tenant_id.len()).map_err(|_| PrepError::InvalidUserEnvelope {
            reason: "tenant_id length overflow",
        })?;
    let run_len = u32::try_from(run_id.len()).map_err(|_| PrepError::InvalidUserEnvelope {
        reason: "run_id length overflow",
    })?;
    let label = payload_kind.domain_label();
    let label_len = label.len() as u32;

    let mut aad = Vec::with_capacity(64 + tenant_id.len() + run_id.len() + label.len());
    aad.extend_from_slice(CUEN_V1_MAGIC);
    aad.push(payload_kind as u8);
    aad.extend_from_slice(&label_len.to_le_bytes());
    aad.extend_from_slice(label.as_bytes());
    aad.extend_from_slice(&tenant_len.to_le_bytes());
    aad.extend_from_slice(tenant_id.as_bytes());
    aad.extend_from_slice(&run_len.to_le_bytes());
    aad.extend_from_slice(run_id.as_bytes());
    aad.extend_from_slice(nonce);
    aad.extend_from_slice(plaintext_sha256);
    Ok(aad)
}

/// Pack a `CUEN\x01` wire envelope from header components and `ciphertext_and_tag` (`>= 16` bytes).
pub fn pack_cuen_v1(
    payload_kind: CuenPayloadKind,
    nonce: &[u8; CUEN_NONCE_BYTES],
    plaintext_sha256: &[u8; 32],
    ciphertext_and_tag: &[u8],
) -> PrepResult<Vec<u8>> {
    if ciphertext_and_tag.len() < CUEN_TAG_BYTES {
        return Err(PrepError::InvalidUserEnvelope {
            reason: "ciphertext_and_tag shorter than 16-byte GCM tag",
        });
    }
    let mut out = Vec::with_capacity(CUEN_HEADER_BYTES + ciphertext_and_tag.len());
    out.extend_from_slice(CUEN_V1_MAGIC);
    out.push(payload_kind as u8);
    out.extend_from_slice(&[0u8, 0u8]);
    out.extend_from_slice(nonce);
    out.extend_from_slice(plaintext_sha256);
    out.extend_from_slice(ciphertext_and_tag);
    Ok(out)
}

/// Parse and validate a `CUEN\x01` wire envelope header and enforce `expected_kind`.
pub fn parse_cuen_v1(
    wire: &[u8],
    expected_kind: CuenPayloadKind,
) -> PrepResult<CuenParsedEnvelope<'_>> {
    if wire.len() < CUEN_HEADER_BYTES + CUEN_TAG_BYTES {
        return Err(PrepError::InvalidUserEnvelope {
            reason: "CUEN envelope shorter than minimum header + tag size",
        });
    }
    if &wire[..5] != CUEN_V1_MAGIC {
        return Err(PrepError::InvalidUserEnvelope {
            reason: "invalid CUEN v1 magic prefix",
        });
    }
    let Some(payload_kind) = CuenPayloadKind::from_u8(wire[5]) else {
        return Err(PrepError::InvalidUserEnvelope {
            reason: "unknown CUEN payload kind discriminant",
        });
    };
    if payload_kind != expected_kind {
        return Err(PrepError::InvalidUserEnvelope {
            reason: "CUEN payload kind mismatch for target slot",
        });
    }
    if wire[6] != 0 || wire[7] != 0 {
        return Err(PrepError::InvalidUserEnvelope {
            reason: "non-zero reserved bytes in CUEN v1 header",
        });
    }
    let mut nonce = [0u8; CUEN_NONCE_BYTES];
    nonce.copy_from_slice(&wire[8..20]);
    let mut plaintext_sha256 = [0u8; 32];
    plaintext_sha256.copy_from_slice(&wire[20..52]);
    Ok(CuenParsedEnvelope {
        payload_kind,
        nonce,
        plaintext_sha256,
        ciphertext_and_tag: &wire[CUEN_HEADER_BYTES..],
    })
}

/// Compute SHA-256 of `plaintext` as a `[u8; 32]` array for `CUEN\x01` inner integrity binding.
pub fn sha256_bytes(plaintext: &[u8]) -> [u8; 32] {
    Sha256::digest(plaintext).into()
}

/// `CudkParsedEnvelope` is a struct holding the parsed `133`-byte `CUDK\x01` wrapped `U_DEK` fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CudkParsedEnvelope {
    pub ephemeral_pubkey: [u8; P256_UNCOMPRESSED_PUBKEY_BYTES],
    pub nonce: [u8; CUEN_NONCE_BYTES],
    pub wrapped_dek_and_tag: [u8; CUDK_CIPHERTEXT_TAG_BYTES],
}

/// Build the canonical length-prefixed AAD for wrapping or unwrapping a per-run `U_DEK` (`CUDK\x01`).
pub fn build_cudk_aad(
    tenant_id: &str,
    run_id: &str,
    recipient_pubkey: &[u8; P256_UNCOMPRESSED_PUBKEY_BYTES],
    ephemeral_pubkey: &[u8; P256_UNCOMPRESSED_PUBKEY_BYTES],
    nonce: &[u8; CUEN_NONCE_BYTES],
) -> PrepResult<Vec<u8>> {
    if tenant_id.is_empty() || run_id.is_empty() {
        return Err(PrepError::InvalidUserEnvelope {
            reason: "tenant_id and run_id must be non-empty",
        });
    }
    let tenant_len =
        u32::try_from(tenant_id.len()).map_err(|_| PrepError::InvalidUserEnvelope {
            reason: "tenant_id length overflow",
        })?;
    let run_len = u32::try_from(run_id.len()).map_err(|_| PrepError::InvalidUserEnvelope {
        reason: "run_id length overflow",
    })?;
    let mut aad = Vec::with_capacity(180 + tenant_id.len() + run_id.len());
    aad.extend_from_slice(CUDK_V1_MAGIC);
    aad.extend_from_slice(&tenant_len.to_le_bytes());
    aad.extend_from_slice(tenant_id.as_bytes());
    aad.extend_from_slice(&run_len.to_le_bytes());
    aad.extend_from_slice(run_id.as_bytes());
    aad.extend_from_slice(recipient_pubkey);
    aad.extend_from_slice(ephemeral_pubkey);
    aad.extend_from_slice(nonce);
    Ok(aad)
}

/// Pack a `133`-byte `CUDK\x01` wrapped `U_DEK` wire envelope.
pub fn pack_cudk_v1(
    ephemeral_pubkey: &[u8; P256_UNCOMPRESSED_PUBKEY_BYTES],
    nonce: &[u8; CUEN_NONCE_BYTES],
    wrapped_dek_and_tag: &[u8; CUDK_CIPHERTEXT_TAG_BYTES],
) -> PrepResult<[u8; CUDK_WIRE_BYTES]> {
    if ephemeral_pubkey[0] != 0x04 {
        return Err(PrepError::InvalidUserEnvelope {
            reason: "ephemeral P-256 public key must be uncompressed SEC1 (0x04)",
        });
    }
    let mut out = [0u8; CUDK_WIRE_BYTES];
    out[..5].copy_from_slice(CUDK_V1_MAGIC);
    out[5..8].copy_from_slice(&[0u8, 0u8, 0u8]);
    out[8..73].copy_from_slice(ephemeral_pubkey);
    out[73..85].copy_from_slice(nonce);
    out[85..133].copy_from_slice(wrapped_dek_and_tag);
    Ok(out)
}

/// Parse and validate a `133`-byte `CUDK\x01` wrapped `U_DEK` wire envelope.
pub fn parse_cudk_v1(wire: &[u8]) -> PrepResult<CudkParsedEnvelope> {
    if wire.len() != CUDK_WIRE_BYTES {
        return Err(PrepError::InvalidUserEnvelope {
            reason: "CUDK v1 envelope must be exactly 133 bytes",
        });
    }
    if &wire[..5] != CUDK_V1_MAGIC {
        return Err(PrepError::InvalidUserEnvelope {
            reason: "invalid CUDK v1 magic prefix",
        });
    }
    if wire[5] != 0 || wire[6] != 0 || wire[7] != 0 {
        return Err(PrepError::InvalidUserEnvelope {
            reason: "non-zero reserved bytes in CUDK v1 header",
        });
    }
    if wire[8] != 0x04 {
        return Err(PrepError::InvalidUserEnvelope {
            reason: "ephemeral P-256 public key must start with 0x04",
        });
    }
    let mut ephemeral_pubkey = [0u8; P256_UNCOMPRESSED_PUBKEY_BYTES];
    ephemeral_pubkey.copy_from_slice(&wire[8..73]);
    let mut nonce = [0u8; CUEN_NONCE_BYTES];
    nonce.copy_from_slice(&wire[73..85]);
    let mut wrapped_dek_and_tag = [0u8; CUDK_CIPHERTEXT_TAG_BYTES];
    wrapped_dek_and_tag.copy_from_slice(&wire[85..133]);
    Ok(CudkParsedEnvelope {
        ephemeral_pubkey,
        nonce,
        wrapped_dek_and_tag,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuen_and_cudk_framing_roundtrip_and_reject_slot_confusion() {
        let nonce = [7u8; CUEN_NONCE_BYTES];
        let pt_sha = sha256_bytes(b"hello fit matrix");
        let aad_a = build_cuen_aad(CuenPayloadKind::FitJmat, "ten_a", "run_1", &nonce, &pt_sha)
            .expect("aad_a");
        let aad_b = build_cuen_aad(
            CuenPayloadKind::HoldoutJmat,
            "ten_a",
            "run_1",
            &nonce,
            &pt_sha,
        )
        .expect("aad_b");
        assert_ne!(aad_a, aad_b);

        let fake_ct = [0xAAu8; 32];
        let wire = pack_cuen_v1(CuenPayloadKind::FitJmat, &nonce, &pt_sha, &fake_ct).expect("pack");
        let parsed = parse_cuen_v1(&wire, CuenPayloadKind::FitJmat).expect("parse fit");
        assert_eq!(parsed.nonce, nonce);
        assert_eq!(parsed.plaintext_sha256, pt_sha);
        assert_eq!(parsed.ciphertext_and_tag, &fake_ct);

        let slot_err = parse_cuen_v1(&wire, CuenPayloadKind::HoldoutJmat).unwrap_err();
        assert!(matches!(slot_err, PrepError::InvalidUserEnvelope { .. }));

        let mut pubk = [3u8; P256_UNCOMPRESSED_PUBKEY_BYTES];
        pubk[0] = 0x04;
        let wrapped = [9u8; CUDK_CIPHERTEXT_TAG_BYTES];
        let cudk_wire = pack_cudk_v1(&pubk, &nonce, &wrapped).expect("pack cudk");
        let cudk_parsed = parse_cudk_v1(&cudk_wire).expect("parse cudk");
        assert_eq!(cudk_parsed.ephemeral_pubkey, pubk);
        assert_eq!(cudk_parsed.nonce, nonce);
        assert_eq!(cudk_parsed.wrapped_dek_and_tag, wrapped);
    }
}
