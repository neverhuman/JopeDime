//! Zero-knowledge per-user envelope encryption (`CUEN\x01` + `CUDK\x01`) via `aws-lc-rs`
//! (`ECDH-ES` on NIST `P-256` + `HKDF-SHA256` + `AES-256-GCM`) with `Zeroizing` key material.
//!
//! Guarantees that `carbon-edge` and S3 CAS hold zero decryption capability over user `fit.jmat`,
//! `holdout.jmat`, `JLMB` landmarks, or `/predict` inputs/outputs, and `carbon-eval` receives
//! only a 60-second ephemeral `FitJmat` slot subkey (`RT-01`, `RT-07`).

#[cfg(feature = "edge")]
use aws_lc_rs::{
    aead::{Aad, LessSafeKey, Nonce, UnboundKey, AES_256_GCM},
    agreement::{agree, PrivateKey, UnparsedPublicKey, ECDH_P256},
    hkdf::{Salt, HKDF_SHA256},
    rand::fill as secure_random_fill,
};
#[cfg(feature = "edge")]
use base64::Engine as _;
#[cfg(feature = "edge")]
use jope_prep::cuen::{
    build_cudk_aad, build_cuen_aad, pack_cudk_v1, pack_cuen_v1, parse_cudk_v1, parse_cuen_v1,
    sha256_bytes, CuenPayloadKind, CUDK_CIPHERTEXT_TAG_BYTES, CUDK_WIRE_BYTES, CUEN_NONCE_BYTES,
    P256_UNCOMPRESSED_PUBKEY_BYTES,
};
#[cfg(feature = "edge")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "edge")]
use zeroize::Zeroizing;

#[cfg(feature = "edge")]
use crate::error::{CarbonError, CarbonResult};

#[cfg(feature = "edge")]
const CUDK_HKDF_SALT: &[u8] = b"jope.carbon.cudk.salt.v1";
#[cfg(feature = "edge")]
const CUEN_HKDF_SALT: &[u8] = b"jope.carbon.cuen.salt.v1";
/// Context label bound into `FitJmat` and `HoldoutJmat` `CUEN\x01` AAD at upload time; each
/// upload generates a unique 256-bit `U_DEK` which is then bound to `(tenant_id, run_id)` via `CUDK\x01`.
pub const CAS_BLOB_CUEN_SCOPE: &str = "cas_blob_v1";

/// `CarbonEngineKek` is a struct holding `carbon-engine`'s NIST P-256 `ECDH` key-encryption keypair.
/// `carbon-edge` only ever receives the 65-byte uncompressed public point (`pubkey_b64()`).
#[cfg(feature = "edge")]
pub struct CarbonEngineKek {
    private_key: PrivateKey,
    pubkey_uncompressed: [u8; P256_UNCOMPRESSED_PUBKEY_BYTES],
}

#[cfg(feature = "edge")]
impl CarbonEngineKek {
    /// Generate a fresh random P-256 `ECDH` keypair (used by ephemeral client response keys and `carbon emu`).
    pub fn generate() -> CarbonResult<Self> {
        let private_key =
            PrivateKey::generate(&ECDH_P256).map_err(|_| CarbonError::UserEnvelopeFailed {
                reason: "failed to generate P-256 ECDH keypair",
            })?;
        Self::from_loaded_private_key(private_key)
    }

    /// Load a deterministic P-256 `ECDH` private key from a 32-byte big-endian scalar.
    pub fn from_scalar_bytes(scalar: &[u8; 32]) -> CarbonResult<Self> {
        let private_key = PrivateKey::from_private_key(&ECDH_P256, scalar).map_err(|_| {
            CarbonError::UserEnvelopeFailed {
                reason: "invalid 32-byte P-256 private scalar",
            }
        })?;
        Self::from_loaded_private_key(private_key)
    }

    /// Load a P-256 `ECDH` private key from a 64-char hex scalar (`CARBON_ENGINE_KEK_HEX`).
    pub fn from_hex(hex_str: &str) -> CarbonResult<Self> {
        let decoded = Zeroizing::new(hex::decode(hex_str.trim()).map_err(|_| {
            CarbonError::UserEnvelopeFailed {
                reason: "CARBON_ENGINE_KEK_HEX must be valid 64-char hex",
            }
        })?);
        if decoded.len() != 32 {
            return Err(CarbonError::UserEnvelopeFailed {
                reason: "CARBON_ENGINE_KEK_HEX must decode to exactly 32 bytes",
            });
        }
        let mut arr = Zeroizing::new([0u8; 32]);
        arr.copy_from_slice(&decoded);
        Self::from_scalar_bytes(&arr)
    }

    fn from_loaded_private_key(private_key: PrivateKey) -> CarbonResult<Self> {
        let pub_key =
            private_key
                .compute_public_key()
                .map_err(|_| CarbonError::UserEnvelopeFailed {
                    reason: "failed to compute P-256 uncompressed public point",
                })?;
        let pub_slice = pub_key.as_ref();
        if pub_slice.len() != P256_UNCOMPRESSED_PUBKEY_BYTES || pub_slice[0] != 0x04 {
            return Err(CarbonError::UserEnvelopeFailed {
                reason: "unexpected P-256 SEC1 public key encoding",
            });
        }
        let mut pubkey_uncompressed = [0u8; P256_UNCOMPRESSED_PUBKEY_BYTES];
        pubkey_uncompressed.copy_from_slice(pub_slice);
        Ok(Self {
            private_key,
            pubkey_uncompressed,
        })
    }

    pub fn pubkey_bytes(&self) -> &[u8; P256_UNCOMPRESSED_PUBKEY_BYTES] {
        &self.pubkey_uncompressed
    }

    pub fn pubkey_b64(&self) -> String {
        base64::engine::general_purpose::STANDARD.encode(self.pubkey_uncompressed)
    }
}

/// Decode a base64-encoded 65-byte uncompressed P-256 public key (`0x04 || X || Y`).
#[cfg(feature = "edge")]
pub fn decode_p256_pubkey_b64(
    pubkey_b64: &str,
) -> CarbonResult<[u8; P256_UNCOMPRESSED_PUBKEY_BYTES]> {
    let raw = base64::engine::general_purpose::STANDARD
        .decode(pubkey_b64.trim())
        .map_err(|_| CarbonError::UserEnvelopeFailed {
            reason: "invalid base64 P-256 public key",
        })?;
    if raw.len() != P256_UNCOMPRESSED_PUBKEY_BYTES || raw[0] != 0x04 {
        return Err(CarbonError::UserEnvelopeFailed {
            reason: "P-256 public key must be 65-byte uncompressed SEC1 (0x04)",
        });
    }
    let mut out = [0u8; P256_UNCOMPRESSED_PUBKEY_BYTES];
    out.copy_from_slice(&raw);
    Ok(out)
}

/// Generate a fresh 256-bit per-run Data Encryption Key (`U_DEK`) in `Zeroizing` memory.
#[cfg(feature = "edge")]
pub fn generate_random_u_dek() -> CarbonResult<Zeroizing<[u8; 32]>> {
    let mut u_dek = Zeroizing::new([0u8; 32]);
    secure_random_fill(u_dek.as_mut()).map_err(|_| CarbonError::UserEnvelopeFailed {
        reason: "CSPRNG fill failed for U_DEK",
    })?;
    Ok(u_dek)
}

/// Wrap a 32-byte `u_dek` to `recipient_pubkey` (`65` B P-256) bound to `(tenant_id, run_id)`
/// using ephemeral `ECDH-P256 + HKDF-SHA256 + AES-256-GCM`, producing a `133`-byte `CUDK\x01` envelope.
#[cfg(feature = "edge")]
pub fn wrap_u_dek_to_pubkey(
    u_dek: &[u8; 32],
    recipient_pubkey: &[u8; P256_UNCOMPRESSED_PUBKEY_BYTES],
    tenant_id: &str,
    run_id: &str,
) -> CarbonResult<[u8; CUDK_WIRE_BYTES]> {
    let eph = CarbonEngineKek::generate()?;
    let mut nonce_bytes = [0u8; CUEN_NONCE_BYTES];
    secure_random_fill(&mut nonce_bytes).map_err(|_| CarbonError::UserEnvelopeFailed {
        reason: "CSPRNG fill failed for CUDK nonce",
    })?;
    let aad = build_cudk_aad(
        tenant_id,
        run_id,
        recipient_pubkey,
        eph.pubkey_bytes(),
        &nonce_bytes,
    )
    .map_err(|_| CarbonError::UserEnvelopeFailed {
        reason: "invalid CUDK AAD inputs",
    })?;

    let peer_pub = UnparsedPublicKey::new(&ECDH_P256, recipient_pubkey.as_slice());
    let wrapped_vec = agree(
        &eph.private_key,
        peer_pub,
        CarbonError::UserEnvelopeFailed {
            reason: "ECDH-P256 key agreement failed",
        },
        |shared_secret| {
            let salt = Salt::new(HKDF_SHA256, CUDK_HKDF_SALT);
            let prk = salt.extract(shared_secret);
            let aad_slice = [aad.as_slice()];
            let okm = prk.expand(&aad_slice, &AES_256_GCM).map_err(|_| {
                CarbonError::UserEnvelopeFailed {
                    reason: "HKDF-SHA256 expand failed for CUDK",
                }
            })?;
            let aead_key = LessSafeKey::new(UnboundKey::from(okm));
            let mut buf = Vec::with_capacity(CUDK_CIPHERTEXT_TAG_BYTES);
            buf.extend_from_slice(u_dek);
            aead_key
                .seal_in_place_append_tag(
                    Nonce::assume_unique_for_key(nonce_bytes),
                    Aad::from(&aad),
                    &mut buf,
                )
                .map_err(|_| CarbonError::UserEnvelopeFailed {
                    reason: "AES-256-GCM wrap failed for U_DEK",
                })?;
            Ok(buf)
        },
    )?;
    let mut wrapped_arr = [0u8; CUDK_CIPHERTEXT_TAG_BYTES];
    wrapped_arr.copy_from_slice(&wrapped_vec);
    pack_cudk_v1(eph.pubkey_bytes(), &nonce_bytes, &wrapped_arr).map_err(|_| {
        CarbonError::UserEnvelopeFailed {
            reason: "failed to pack CUDK v1 envelope",
        }
    })
}

/// Unwrap a `133`-byte `CUDK\x01` envelope using `kek` (`carbon-engine`'s P-256 KEK or client's
/// response KEK), refusing any tamper or `(tenant_id, run_id)` mismatch.
#[cfg(feature = "edge")]
pub fn unwrap_u_dek_with_kek(
    kek: &CarbonEngineKek,
    cudk_wire: &[u8],
    tenant_id: &str,
    run_id: &str,
) -> CarbonResult<Zeroizing<[u8; 32]>> {
    let parsed = parse_cudk_v1(cudk_wire).map_err(|_| CarbonError::UserEnvelopeFailed {
        reason: "invalid CUDK v1 wire header or length",
    })?;
    let aad = build_cudk_aad(
        tenant_id,
        run_id,
        kek.pubkey_bytes(),
        &parsed.ephemeral_pubkey,
        &parsed.nonce,
    )
    .map_err(|_| CarbonError::UserEnvelopeFailed {
        reason: "invalid CUDK AAD context",
    })?;

    let peer_pub = UnparsedPublicKey::new(&ECDH_P256, parsed.ephemeral_pubkey.as_slice());
    agree(
        &kek.private_key,
        peer_pub,
        CarbonError::UserEnvelopeFailed {
            reason: "ECDH-P256 unwrap key agreement failed",
        },
        |shared_secret| {
            let salt = Salt::new(HKDF_SHA256, CUDK_HKDF_SALT);
            let prk = salt.extract(shared_secret);
            let aad_slice = [aad.as_slice()];
            let okm = prk.expand(&aad_slice, &AES_256_GCM).map_err(|_| {
                CarbonError::UserEnvelopeFailed {
                    reason: "HKDF-SHA256 expand failed during CUDK unwrap",
                }
            })?;
            let aead_key = LessSafeKey::new(UnboundKey::from(okm));
            let mut buf = Zeroizing::new(parsed.wrapped_dek_and_tag.to_vec());
            let pt = aead_key
                .open_in_place(
                    Nonce::assume_unique_for_key(parsed.nonce),
                    Aad::from(&aad),
                    buf.as_mut_slice(),
                )
                .map_err(|_| CarbonError::UserEnvelopeFailed {
                    reason:
                        "CUDK AES-256-GCM tag verification failed (wrong key, tenant, or run_id)",
                })?;
            if pt.len() != 32 {
                return Err(CarbonError::UserEnvelopeFailed {
                    reason: "unwrapped U_DEK is not 32 bytes",
                });
            }
            let mut dek = Zeroizing::new([0u8; 32]);
            dek.copy_from_slice(pt);
            Ok(dek)
        },
    )
}

#[cfg(feature = "edge")]
fn build_slot_hkdf_info(kind: CuenPayloadKind, tenant_id: &str, scope_id: &str) -> Vec<u8> {
    let label = kind.domain_label();
    let mut info = Vec::with_capacity(32 + label.len() + tenant_id.len() + scope_id.len());
    info.extend_from_slice(b"jope.carbon.cuen.slot_key.v1\0");
    info.push(kind as u8);
    info.extend_from_slice(&(label.len() as u32).to_le_bytes());
    info.extend_from_slice(label.as_bytes());
    info.extend_from_slice(&(tenant_id.len() as u32).to_le_bytes());
    info.extend_from_slice(tenant_id.as_bytes());
    info.extend_from_slice(&(scope_id.len() as u32).to_le_bytes());
    info.extend_from_slice(scope_id.as_bytes());
    info
}

/// Derive a single-slot `AES-256-GCM` subkey (`32` B) from `u_dek` so `engine` can grant a
/// stateless `eval` shard temporary `FitJmat`-only decryption capability without ever sharing `U_DEK` (`RT-07`).
#[cfg(feature = "edge")]
pub fn derive_cuen_slot_subkey(
    u_dek: &[u8; 32],
    kind: CuenPayloadKind,
    tenant_id: &str,
    scope_id: &str,
) -> CarbonResult<Zeroizing<[u8; 32]>> {
    let salt = Salt::new(HKDF_SHA256, CUEN_HKDF_SALT);
    let prk = salt.extract(u_dek);
    let info = build_slot_hkdf_info(kind, tenant_id, scope_id);
    let info_slice = [info.as_slice()];
    let okm =
        prk.expand(&info_slice, HKDF_SHA256)
            .map_err(|_| CarbonError::UserEnvelopeFailed {
                reason: "HKDF slot subkey derivation failed",
            })?;
    let mut subkey = Zeroizing::new([0u8; 32]);
    okm.fill(subkey.as_mut())
        .map_err(|_| CarbonError::UserEnvelopeFailed {
            reason: "HKDF slot subkey fill failed",
        })?;
    Ok(subkey)
}

/// Encrypt `plaintext` into a `CUEN\x01` wire envelope bound to `(kind, tenant_id, scope_id)`.
#[cfg(feature = "edge")]
pub fn encrypt_cuen_payload(
    u_dek: &[u8; 32],
    kind: CuenPayloadKind,
    tenant_id: &str,
    scope_id: &str,
    plaintext: &[u8],
) -> CarbonResult<Vec<u8>> {
    let slot_key = derive_cuen_slot_subkey(u_dek, kind, tenant_id, scope_id)?;
    let unbound = UnboundKey::new(&AES_256_GCM, slot_key.as_ref()).map_err(|_| {
        CarbonError::UserEnvelopeFailed {
            reason: "AES-256-GCM slot key init failed",
        }
    })?;
    let aead_key = LessSafeKey::new(unbound);

    let mut nonce = [0u8; CUEN_NONCE_BYTES];
    secure_random_fill(&mut nonce).map_err(|_| CarbonError::UserEnvelopeFailed {
        reason: "CSPRNG fill failed for CUEN nonce",
    })?;
    let pt_sha = sha256_bytes(plaintext);
    let aad = build_cuen_aad(kind, tenant_id, scope_id, &nonce, &pt_sha).map_err(|_| {
        CarbonError::UserEnvelopeFailed {
            reason: "invalid CUEN AAD parameters",
        }
    })?;

    let mut ct_and_tag = plaintext.to_vec();
    aead_key
        .seal_in_place_append_tag(
            Nonce::assume_unique_for_key(nonce),
            Aad::from(&aad),
            &mut ct_and_tag,
        )
        .map_err(|_| CarbonError::UserEnvelopeFailed {
            reason: "CUEN AES-256-GCM encryption failed",
        })?;

    pack_cuen_v1(kind, &nonce, &pt_sha, &ct_and_tag).map_err(|_| CarbonError::UserEnvelopeFailed {
        reason: "failed to pack CUEN v1 envelope",
    })
}

/// Authenticate and decrypt a `CUEN\x01` wire envelope using a pre-derived `slot_subkey` (`RT-07`).
#[cfg(feature = "edge")]
pub fn decrypt_cuen_with_slot_subkey(
    slot_subkey: &[u8; 32],
    expected_kind: CuenPayloadKind,
    tenant_id: &str,
    scope_id: &str,
    wire: &[u8],
) -> CarbonResult<Zeroizing<Vec<u8>>> {
    let parsed =
        parse_cuen_v1(wire, expected_kind).map_err(|_| CarbonError::UserEnvelopeFailed {
            reason: "CUEN v1 header magic or payload slot kind mismatch",
        })?;
    let aad = build_cuen_aad(
        expected_kind,
        tenant_id,
        scope_id,
        &parsed.nonce,
        &parsed.plaintext_sha256,
    )
    .map_err(|_| CarbonError::UserEnvelopeFailed {
        reason: "invalid CUEN AAD context during decryption",
    })?;

    let unbound = UnboundKey::new(&AES_256_GCM, slot_subkey).map_err(|_| {
        CarbonError::UserEnvelopeFailed {
            reason: "AES-256-GCM slot key init failed",
        }
    })?;
    let aead_key = LessSafeKey::new(unbound);
    let mut buf = Zeroizing::new(parsed.ciphertext_and_tag.to_vec());
    let pt_len = aead_key
        .open_in_place(
            Nonce::assume_unique_for_key(parsed.nonce),
            Aad::from(&aad),
            buf.as_mut_slice(),
        )
        .map_err(|_| CarbonError::UserEnvelopeFailed {
            reason: "CUEN AES-256-GCM authentication tag verification failed",
        })?
        .len();
    buf.truncate(pt_len);
    if sha256_bytes(&buf) != parsed.plaintext_sha256 {
        return Err(CarbonError::UserEnvelopeFailed {
            reason: "CUEN inner plaintext SHA-256 mismatch",
        });
    }
    Ok(buf)
}

/// Authenticate and decrypt a `CUEN\x01` wire envelope under `u_dek`, enforcing `expected_kind`.
#[cfg(feature = "edge")]
pub fn decrypt_cuen_payload(
    u_dek: &[u8; 32],
    expected_kind: CuenPayloadKind,
    tenant_id: &str,
    scope_id: &str,
    wire: &[u8],
) -> CarbonResult<Zeroizing<Vec<u8>>> {
    let slot_subkey = derive_cuen_slot_subkey(u_dek, expected_kind, tenant_id, scope_id)?;
    decrypt_cuen_with_slot_subkey(&slot_subkey, expected_kind, tenant_id, scope_id, wire)
}

/// `CarbonEncryptedPredictPayload` is a struct serialized inside `CuenPayloadKind::PredictFeatures`
/// so neither feature column names nor feature values appear in cleartext JSON on `/predict`.
#[cfg(feature = "edge")]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CarbonEncryptedPredictPayload {
    pub feature_names: Vec<String>,
    pub rows: Vec<Vec<Option<f64>>>,
}
