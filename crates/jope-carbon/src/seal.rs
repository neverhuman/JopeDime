//! Authenticated checkpoint state sealing (`AES-256-GCM` + context-bound `HMAC-SHA256` key
//! derivation via `aws-lc-rs`, with `Zeroizing` key material).
//!
//! Ensures intermediate search state (`sealed_checkpoint.enc`) stored in S3 CAS or paused for
//! early-stop / resume cannot be inspected in cleartext, tampered with, or replayed across
//! another tenant or run.

#[cfg(feature = "edge")]
use aws_lc_rs::{
    aead::{Aad, LessSafeKey, Nonce, UnboundKey, AES_256_GCM},
    hmac,
    rand::fill as secure_random_fill,
};
#[cfg(feature = "edge")]
use zeroize::Zeroizing;

#[cfg(feature = "edge")]
use crate::error::{CarbonError, CarbonResult};

#[cfg(feature = "edge")]
const SEAL_MAGIC: &[u8; 8] = b"CCKP\x01\x00\x00\x00";
#[cfg(feature = "edge")]
const NONCE_BYTES: usize = 12;
#[cfg(feature = "edge")]
const TAG_BYTES: usize = 16;

/// `CarbonSealKey` is a struct wrapping a 32-byte master secret in `Zeroizing` memory for deriving
/// per-tenant, per-run `AES-256-GCM` checkpoint encryption keys.
#[cfg(feature = "edge")]
pub struct CarbonSealKey {
    master: Zeroizing<[u8; 32]>,
}

#[cfg(feature = "edge")]
impl CarbonSealKey {
    /// Construct a `CarbonSealKey` from a 32-byte master secret.
    pub fn from_bytes(secret: [u8; 32]) -> Self {
        Self {
            master: Zeroizing::new(secret),
        }
    }

    fn derive_run_key(&self, tenant_id: &str, run_id: &str) -> Zeroizing<[u8; 32]> {
        let prk = hmac::Key::new(hmac::HMAC_SHA256, self.master.as_ref());
        let mut ctx = hmac::Context::with_key(&prk);
        ctx.update(b"jope.carbon.checkpoint_key.v1\0");
        ctx.update((tenant_id.len() as u32).to_le_bytes().as_slice());
        ctx.update(tenant_id.as_bytes());
        ctx.update([0u8].as_slice());
        ctx.update((run_id.len() as u32).to_le_bytes().as_slice());
        ctx.update(run_id.as_bytes());
        let tag = ctx.sign();
        let mut derived = Zeroizing::new([0u8; 32]);
        derived.copy_from_slice(&tag.as_ref()[..32]);
        derived
    }
}

/// `SealedCheckpointEnvelope` is a struct for the `CCKP` v1 authenticated ciphertext wire container:
/// `8 B` magic + `12 B` random nonce + `8 B` wave index + `AES-256-GCM(plaintext || 16 B tag)`
/// authenticated over `(magic, nonce, wave_index, tenant_id, run_id)` AAD.
#[cfg(feature = "edge")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SealedCheckpointEnvelope {
    wave_index: u64,
    wire_bytes: Vec<u8>,
}

#[cfg(feature = "edge")]
impl SealedCheckpointEnvelope {
    /// Encrypt and authenticate `plaintext_state` under `(key, tenant_id, run_id, wave_index)`.
    pub fn seal(
        key: &CarbonSealKey,
        tenant_id: &str,
        run_id: &str,
        wave_index: u64,
        plaintext_state: &[u8],
    ) -> CarbonResult<Self> {
        if tenant_id.is_empty() || run_id.is_empty() {
            return Err(CarbonError::SealAuthenticationFailed {
                reason: "tenant_id and run_id must be non-empty",
            });
        }
        let derived = key.derive_run_key(tenant_id, run_id);
        let unbound = UnboundKey::new(&AES_256_GCM, derived.as_ref()).map_err(|_| {
            CarbonError::SealAuthenticationFailed {
                reason: "failed to initialize AES-256-GCM key",
            }
        })?;
        let aead_key = LessSafeKey::new(unbound);

        let mut nonce_bytes = [0u8; NONCE_BYTES];
        secure_random_fill(&mut nonce_bytes).map_err(|_| {
            CarbonError::SealAuthenticationFailed {
                reason: "CSPRNG fill failed",
            }
        })?;
        let nonce = Nonce::assume_unique_for_key(nonce_bytes);
        let aad_vec = build_aad(tenant_id, run_id, wave_index, &nonce_bytes);

        let mut in_out = plaintext_state.to_vec();
        aead_key
            .seal_in_place_append_tag(nonce, Aad::from(&aad_vec), &mut in_out)
            .map_err(|_| CarbonError::SealAuthenticationFailed {
                reason: "AES-256-GCM seal failed",
            })?;

        let mut wire_bytes = Vec::with_capacity(8 + NONCE_BYTES + 8 + in_out.len());
        wire_bytes.extend_from_slice(SEAL_MAGIC);
        wire_bytes.extend_from_slice(&nonce_bytes);
        wire_bytes.extend_from_slice(&wave_index.to_le_bytes());
        wire_bytes.extend_from_slice(&in_out);

        Ok(Self {
            wave_index,
            wire_bytes,
        })
    }

    /// Authenticate and decrypt `wire_bytes`, refusing any tamper, wrong key, or wrong `(tenant_id, run_id)`.
    pub fn open(
        key: &CarbonSealKey,
        tenant_id: &str,
        run_id: &str,
        wire_bytes: &[u8],
    ) -> CarbonResult<(u64, Zeroizing<Vec<u8>>)> {
        let min_len = 8 + NONCE_BYTES + 8 + TAG_BYTES;
        if tenant_id.is_empty() || run_id.is_empty() || wire_bytes.len() < min_len {
            return Err(CarbonError::SealAuthenticationFailed {
                reason: "envelope shorter than minimum header + GCM tag or empty context",
            });
        }
        if &wire_bytes[..8] != SEAL_MAGIC {
            return Err(CarbonError::SealAuthenticationFailed {
                reason: "envelope header magic or version mismatch",
            });
        }
        let mut nonce_bytes = [0u8; NONCE_BYTES];
        nonce_bytes.copy_from_slice(&wire_bytes[8..8 + NONCE_BYTES]);
        let mut wave_raw = [0u8; 8];
        wave_raw.copy_from_slice(&wire_bytes[8 + NONCE_BYTES..8 + NONCE_BYTES + 8]);
        let wave_index = u64::from_le_bytes(wave_raw);

        let derived = key.derive_run_key(tenant_id, run_id);
        let unbound = UnboundKey::new(&AES_256_GCM, derived.as_ref()).map_err(|_| {
            CarbonError::SealAuthenticationFailed {
                reason: "failed to initialize AES-256-GCM key",
            }
        })?;
        let aead_key = LessSafeKey::new(unbound);
        let aad_vec = build_aad(tenant_id, run_id, wave_index, &nonce_bytes);

        let mut buf = Zeroizing::new(wire_bytes[8 + NONCE_BYTES + 8..].to_vec());
        let plaintext_len = aead_key
            .open_in_place(
                Nonce::assume_unique_for_key(nonce_bytes),
                Aad::from(&aad_vec),
                buf.as_mut_slice(),
            )
            .map_err(|_| CarbonError::SealAuthenticationFailed {
                reason: "AES-256-GCM tag verification failed",
            })?
            .len();
        buf.truncate(plaintext_len);
        Ok((wave_index, buf))
    }

    /// Wave index recorded in the authenticated header.
    pub fn wave_index(&self) -> u64 {
        self.wave_index
    }

    /// Serialized `CCKP` v1 wire bytes (`header + nonce + wave_index + ciphertext + GCM tag`).
    pub fn wire_bytes(&self) -> &[u8] {
        &self.wire_bytes
    }
}

#[cfg(feature = "edge")]
fn build_aad(
    tenant_id: &str,
    run_id: &str,
    wave_index: u64,
    nonce_bytes: &[u8; NONCE_BYTES],
) -> Vec<u8> {
    let mut aad = Vec::with_capacity(64 + tenant_id.len() + run_id.len());
    aad.extend_from_slice(SEAL_MAGIC);
    aad.extend_from_slice(nonce_bytes);
    aad.extend_from_slice(&wave_index.to_le_bytes());
    aad.extend_from_slice(&(tenant_id.len() as u32).to_le_bytes());
    aad.extend_from_slice(tenant_id.as_bytes());
    aad.extend_from_slice(&(run_id.len() as u32).to_le_bytes());
    aad.extend_from_slice(run_id.as_bytes());
    aad
}
