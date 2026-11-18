//! Pure-Rust client envelope encryption (`CUEN\x01` + `CUDK\x01`) for `jAIn`, browser WASM,
//! and `carbon-cli` (`ADR-0007` Phase C7).
//!
//! Byte-for-byte wire-compatible with `jope-carbon::security::user_envelope` (`aws-lc-rs`):
//! `ECDH-ES` on NIST `P-256` + `HKDF-SHA256` + `AES-256-GCM` with `Zeroizing` key material.

use base64::Engine as _;
use jope_prep::cuen::{
    build_cudk_aad, build_cuen_aad, pack_cudk_v1, pack_cuen_v1, parse_cudk_v1, parse_cuen_v1,
    sha256_bytes, CuenPayloadKind, CUDK_CIPHERTEXT_TAG_BYTES, CUDK_WIRE_BYTES, CUEN_NONCE_BYTES,
    P256_UNCOMPRESSED_PUBKEY_BYTES,
};
use p256::{elliptic_curve::sec1::ToSec1Point, PublicKey, SecretKey};
use zeroize::Zeroizing;

use crate::aead::{aes256_gcm_open, aes256_gcm_seal, hkdf_sha256_32};

fn p256_ecdh_shared_x(
    secret: &SecretKey,
    peer_pub: &PublicKey,
) -> EnvelopeResult<Zeroizing<[u8; 32]>> {
    let point = (peer_pub.to_projective() * secret.to_nonzero_scalar().as_ref()).to_affine();
    let encoded = point.to_sec1_point(false);
    let Some(x_bytes) = encoded.x() else {
        return Err(EnvelopeError::AuthenticationFailed(
            "ECDH-P256 shared point at infinity",
        ));
    };
    let mut shared = Zeroizing::new([0u8; 32]);
    shared.copy_from_slice(x_bytes.as_slice());
    Ok(shared)
}

/// Domain salt for `CUDK\x01` key-wrapping `HKDF-SHA256`.
pub const CUDK_HKDF_SALT: &[u8] = b"jope.carbon.cudk.salt.v1";
/// Domain salt for `CUEN\x01` slot-subkey `HKDF-SHA256`.
pub const CUEN_HKDF_SALT: &[u8] = b"jope.carbon.cuen.salt.v1";
/// Context scope bound into `FitJmat` and `HoldoutJmat` `CUEN\x01` AAD at upload time.
pub const CAS_BLOB_CUEN_SCOPE: &str = "cas_blob_v1";

/// `EnvelopeError` is an enum describing client-side `CUEN`/`CUDK` or `ctk2` failures.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EnvelopeError {
    /// Invalid key, nonce, or wire encoding.
    InvalidMaterial(&'static str),
    /// Cryptographic authentication or signature check failed.
    AuthenticationFailed(&'static str),
}

impl std::fmt::Display for EnvelopeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidMaterial(msg) | Self::AuthenticationFailed(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for EnvelopeError {}

/// Convenience result alias for `jope-envelope`.
pub type EnvelopeResult<T> = Result<T, EnvelopeError>;

/// `ClientEnvelopeKeypair` is a struct holding a NIST P-256 `ECDH` keypair in pure Rust for
/// wrapping per-run `U_DEK` keys or unwrapping server response `CUDK\x01` envelopes.
pub struct ClientEnvelopeKeypair {
    secret: SecretKey,
    pubkey_uncompressed: [u8; P256_UNCOMPRESSED_PUBKEY_BYTES],
}

impl std::fmt::Debug for ClientEnvelopeKeypair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClientEnvelopeKeypair")
            .field("secret", &"[REDACTED]")
            .field("pubkey_b64", &self.pubkey_b64())
            .finish()
    }
}

impl ClientEnvelopeKeypair {
    /// Load a deterministic P-256 `ECDH` keypair from a 32-byte big-endian private scalar.
    pub fn from_scalar_bytes(scalar: &[u8; 32]) -> EnvelopeResult<Self> {
        let secret = SecretKey::from_slice(scalar)
            .map_err(|_| EnvelopeError::InvalidMaterial("invalid 32-byte P-256 private scalar"))?;
        let encoded = secret.public_key().to_sec1_point(false);
        let slice = encoded.as_bytes();
        if slice.len() != P256_UNCOMPRESSED_PUBKEY_BYTES || slice[0] != 0x04 {
            return Err(EnvelopeError::InvalidMaterial(
                "unexpected P-256 SEC1 uncompressed public point length",
            ));
        }
        let mut pubkey_uncompressed = [0u8; P256_UNCOMPRESSED_PUBKEY_BYTES];
        pubkey_uncompressed.copy_from_slice(slice);
        Ok(Self {
            secret,
            pubkey_uncompressed,
        })
    }

    /// Load a P-256 `ECDH` keypair from a 64-character hex scalar.
    pub fn from_hex(hex_str: &str) -> EnvelopeResult<Self> {
        let raw = Zeroizing::new(
            hex::decode(hex_str.trim())
                .map_err(|_| EnvelopeError::InvalidMaterial("invalid hex P-256 scalar"))?,
        );
        if raw.len() != 32 {
            return Err(EnvelopeError::InvalidMaterial(
                "P-256 private scalar hex must decode to 32 bytes",
            ));
        }
        let mut arr = Zeroizing::new([0u8; 32]);
        arr.copy_from_slice(&raw);
        Self::from_scalar_bytes(&arr)
    }

    /// Generate a fresh random P-256 `ECDH` keypair using OS CSPRNG (non-`wasm32` targets).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn generate() -> EnvelopeResult<Self> {
        for _ in 0..8 {
            let mut seed = Zeroizing::new([0u8; 32]);
            getrandom::fill(seed.as_mut())
                .map_err(|_| EnvelopeError::InvalidMaterial("OS CSPRNG fill failed"))?;
            if let Ok(kp) = Self::from_scalar_bytes(&seed) {
                return Ok(kp);
            }
        }
        Err(EnvelopeError::InvalidMaterial(
            "failed to sample valid P-256 private scalar",
        ))
    }

    /// Uncompressed 65-byte SEC1 public key (`0x04 || X || Y`).
    pub fn pubkey_bytes(&self) -> &[u8; P256_UNCOMPRESSED_PUBKEY_BYTES] {
        &self.pubkey_uncompressed
    }

    /// Standard Base64 encoding of the 65-byte uncompressed P-256 public point.
    pub fn pubkey_b64(&self) -> String {
        base64::engine::general_purpose::STANDARD.encode(self.pubkey_uncompressed)
    }
}

/// Decode a Base64-encoded 65-byte uncompressed NIST P-256 public key (`0x04 || X || Y`) and
/// verify that the point lies on the NIST P-256 curve.
pub fn decode_p256_pubkey_b64(
    pubkey_b64: &str,
) -> EnvelopeResult<[u8; P256_UNCOMPRESSED_PUBKEY_BYTES]> {
    let raw = base64::engine::general_purpose::STANDARD
        .decode(pubkey_b64.trim())
        .map_err(|_| EnvelopeError::InvalidMaterial("invalid base64 P-256 public key"))?;
    if raw.len() != P256_UNCOMPRESSED_PUBKEY_BYTES || raw[0] != 0x04 {
        return Err(EnvelopeError::InvalidMaterial(
            "P-256 public key must be 65-byte uncompressed SEC1 (0x04)",
        ));
    }
    let _ = PublicKey::from_sec1_bytes(&raw)
        .map_err(|_| EnvelopeError::InvalidMaterial("P-256 public point is not on curve"))?;
    let mut out = [0u8; P256_UNCOMPRESSED_PUBKEY_BYTES];
    out.copy_from_slice(&raw);
    Ok(out)
}

/// Generate a fresh 256-bit per-run Data Encryption Key (`U_DEK`) on non-`wasm32` targets.
#[cfg(not(target_arch = "wasm32"))]
pub fn generate_random_u_dek() -> EnvelopeResult<Zeroizing<[u8; 32]>> {
    let mut dek = Zeroizing::new([0u8; 32]);
    getrandom::fill(dek.as_mut())
        .map_err(|_| EnvelopeError::InvalidMaterial("OS CSPRNG fill failed for U_DEK"))?;
    Ok(dek)
}

/// Wrap `u_dek` to `recipient_pubkey` (`65` B P-256) with explicit caller-supplied ephemeral
/// entropy (`ephemeral_scalar` + `nonce`), suitable for both `wasm32` (`crypto.getRandomValues`)
/// and deterministic vector tests.
pub fn wrap_u_dek_to_pubkey_with_material(
    u_dek: &[u8; 32],
    recipient_pubkey: &[u8; P256_UNCOMPRESSED_PUBKEY_BYTES],
    tenant_id: &str,
    run_id: &str,
    ephemeral_scalar: &[u8; 32],
    nonce: &[u8; CUEN_NONCE_BYTES],
) -> EnvelopeResult<[u8; CUDK_WIRE_BYTES]> {
    let eph = ClientEnvelopeKeypair::from_scalar_bytes(ephemeral_scalar)?;
    let peer_pub = PublicKey::from_sec1_bytes(recipient_pubkey)
        .map_err(|_| EnvelopeError::InvalidMaterial("invalid recipient P-256 public key"))?;
    let aad = build_cudk_aad(
        tenant_id,
        run_id,
        recipient_pubkey,
        eph.pubkey_bytes(),
        nonce,
    )
    .map_err(|_| EnvelopeError::InvalidMaterial("invalid CUDK AAD inputs"))?;

    let shared = p256_ecdh_shared_x(&eph.secret, &peer_pub)?;
    let aead_key = hkdf_sha256_32(CUDK_HKDF_SALT, &*shared, &aad);
    let ct_and_tag = aes256_gcm_seal(&aead_key, nonce, &aad, u_dek);
    let mut wrapped_arr = [0u8; CUDK_CIPHERTEXT_TAG_BYTES];
    wrapped_arr.copy_from_slice(&ct_and_tag);
    pack_cudk_v1(eph.pubkey_bytes(), nonce, &wrapped_arr)
        .map_err(|_| EnvelopeError::InvalidMaterial("failed to pack CUDK v1 envelope"))
}

/// Wrap `u_dek` to `recipient_pubkey` using OS CSPRNG (non-`wasm32` targets).
#[cfg(not(target_arch = "wasm32"))]
pub fn wrap_u_dek_to_pubkey(
    u_dek: &[u8; 32],
    recipient_pubkey: &[u8; P256_UNCOMPRESSED_PUBKEY_BYTES],
    tenant_id: &str,
    run_id: &str,
) -> EnvelopeResult<[u8; CUDK_WIRE_BYTES]> {
    let mut scalar = Zeroizing::new([0u8; 32]);
    for _ in 0..8 {
        getrandom::fill(scalar.as_mut())
            .map_err(|_| EnvelopeError::InvalidMaterial("CSPRNG fill failed"))?;
        if SecretKey::from_slice(&*scalar).is_ok() {
            break;
        }
    }
    let mut nonce = [0u8; CUEN_NONCE_BYTES];
    getrandom::fill(&mut nonce)
        .map_err(|_| EnvelopeError::InvalidMaterial("CSPRNG nonce fill failed"))?;
    wrap_u_dek_to_pubkey_with_material(u_dek, recipient_pubkey, tenant_id, run_id, &scalar, &nonce)
}

/// Unwrap a `133`-byte `CUDK\x01` envelope using `keypair`.
pub fn unwrap_u_dek_with_keypair(
    keypair: &ClientEnvelopeKeypair,
    cudk_wire: &[u8],
    tenant_id: &str,
    run_id: &str,
) -> EnvelopeResult<Zeroizing<[u8; 32]>> {
    let parsed = parse_cudk_v1(cudk_wire)
        .map_err(|_| EnvelopeError::InvalidMaterial("invalid CUDK v1 wire header or length"))?;
    let peer_pub = PublicKey::from_sec1_bytes(&parsed.ephemeral_pubkey)
        .map_err(|_| EnvelopeError::InvalidMaterial("invalid ephemeral P-256 public key"))?;
    let aad = build_cudk_aad(
        tenant_id,
        run_id,
        keypair.pubkey_bytes(),
        &parsed.ephemeral_pubkey,
        &parsed.nonce,
    )
    .map_err(|_| EnvelopeError::InvalidMaterial("invalid CUDK AAD context"))?;

    let shared = p256_ecdh_shared_x(&keypair.secret, &peer_pub)?;
    let aead_key = hkdf_sha256_32(CUDK_HKDF_SALT, &*shared, &aad);
    let pt = aes256_gcm_open(&aead_key, &parsed.nonce, &aad, &parsed.wrapped_dek_and_tag).map_err(
        |_| {
            EnvelopeError::AuthenticationFailed(
                "CUDK AES-256-GCM tag verification failed (wrong key, tenant, or run_id)",
            )
        },
    )?;
    if pt.len() != 32 {
        return Err(EnvelopeError::InvalidMaterial(
            "unwrapped U_DEK is not 32 bytes",
        ));
    }
    let mut dek = Zeroizing::new([0u8; 32]);
    dek.copy_from_slice(&pt);
    Ok(dek)
}

/// Build the canonical `HKDF-SHA256` `info` byte string for a `CUEN\x01` slot subkey.
pub fn build_slot_hkdf_info(kind: CuenPayloadKind, tenant_id: &str, scope_id: &str) -> Vec<u8> {
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

/// Derive a single-slot `AES-256-GCM` subkey (`32` B) from `u_dek` (`RT-07`).
pub fn derive_cuen_slot_subkey(
    u_dek: &[u8; 32],
    kind: CuenPayloadKind,
    tenant_id: &str,
    scope_id: &str,
) -> Zeroizing<[u8; 32]> {
    let info = build_slot_hkdf_info(kind, tenant_id, scope_id);
    hkdf_sha256_32(CUEN_HKDF_SALT, u_dek, &info)
}

/// Encrypt `plaintext` into a `CUEN\x01` wire envelope with an explicit 12-byte `nonce`
/// (`wasm32`-ready; supply 12 fresh random bytes from `crypto.getRandomValues` per call).
pub fn encrypt_cuen_payload_with_nonce(
    u_dek: &[u8; 32],
    kind: CuenPayloadKind,
    tenant_id: &str,
    scope_id: &str,
    nonce: &[u8; CUEN_NONCE_BYTES],
    plaintext: &[u8],
) -> EnvelopeResult<Vec<u8>> {
    let slot_key = derive_cuen_slot_subkey(u_dek, kind, tenant_id, scope_id);
    let pt_sha = sha256_bytes(plaintext);
    let aad = build_cuen_aad(kind, tenant_id, scope_id, nonce, &pt_sha)
        .map_err(|_| EnvelopeError::InvalidMaterial("invalid CUEN AAD parameters"))?;
    let ct_and_tag = aes256_gcm_seal(&slot_key, nonce, &aad, plaintext);
    pack_cuen_v1(kind, nonce, &pt_sha, &ct_and_tag)
        .map_err(|_| EnvelopeError::InvalidMaterial("failed to pack CUEN v1 envelope"))
}

/// Encrypt `plaintext` into a `CUEN\x01` wire envelope using OS CSPRNG (non-`wasm32` targets).
#[cfg(not(target_arch = "wasm32"))]
pub fn encrypt_cuen_payload(
    u_dek: &[u8; 32],
    kind: CuenPayloadKind,
    tenant_id: &str,
    scope_id: &str,
    plaintext: &[u8],
) -> EnvelopeResult<Vec<u8>> {
    let mut nonce = [0u8; CUEN_NONCE_BYTES];
    getrandom::fill(&mut nonce)
        .map_err(|_| EnvelopeError::InvalidMaterial("CSPRNG fill failed for CUEN nonce"))?;
    encrypt_cuen_payload_with_nonce(u_dek, kind, tenant_id, scope_id, &nonce, plaintext)
}

/// Authenticate and decrypt a `CUEN\x01` wire envelope using a pre-derived `slot_subkey`.
pub fn decrypt_cuen_with_slot_subkey(
    slot_subkey: &[u8; 32],
    expected_kind: CuenPayloadKind,
    tenant_id: &str,
    scope_id: &str,
    wire: &[u8],
) -> EnvelopeResult<Zeroizing<Vec<u8>>> {
    let parsed = parse_cuen_v1(wire, expected_kind).map_err(|_| {
        EnvelopeError::InvalidMaterial("CUEN v1 header magic or payload slot kind mismatch")
    })?;
    let aad = build_cuen_aad(
        expected_kind,
        tenant_id,
        scope_id,
        &parsed.nonce,
        &parsed.plaintext_sha256,
    )
    .map_err(|_| EnvelopeError::InvalidMaterial("invalid CUEN AAD context during decryption"))?;

    let pt = aes256_gcm_open(slot_subkey, &parsed.nonce, &aad, parsed.ciphertext_and_tag).map_err(
        |_| {
            EnvelopeError::AuthenticationFailed(
                "CUEN AES-256-GCM authentication tag verification failed",
            )
        },
    )?;
    if sha256_bytes(&pt) != parsed.plaintext_sha256 {
        return Err(EnvelopeError::AuthenticationFailed(
            "CUEN inner plaintext SHA-256 mismatch",
        ));
    }
    Ok(pt)
}

/// Authenticate and decrypt a `CUEN\x01` wire envelope under `u_dek`.
pub fn decrypt_cuen_payload(
    u_dek: &[u8; 32],
    expected_kind: CuenPayloadKind,
    tenant_id: &str,
    scope_id: &str,
    wire: &[u8],
) -> EnvelopeResult<Zeroizing<Vec<u8>>> {
    let slot_subkey = derive_cuen_slot_subkey(u_dek, expected_kind, tenant_id, scope_id);
    decrypt_cuen_with_slot_subkey(&slot_subkey, expected_kind, tenant_id, scope_id, wire)
}
