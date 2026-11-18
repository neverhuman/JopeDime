//! Asymmetric `ECDSA-P256` (`ES256`) `ctk2.*` capability tokens minted by `app.veox.ai` (`ADR-0007`).
//!
//! `app.veox.ai` holds only its own P-256 private signing key and never holds `carbon-edge`'s
//! `auth_secret_hex`. `carbon-edge` verifies `ctk2.<claims_b64url>.<es256_sig_b64url>` against
//! pinned uncompressed SEC1 P-256 public keys (`CARBON_APP_AUTH_PUBKEYS`, supporting comma-separated
//! `[active, previous]` zero-downtime key rotation).

use base64::Engine as _;
use jope_prep::cuen::P256_UNCOMPRESSED_PUBKEY_BYTES;
use p256::ecdsa::{
    signature::{Signer, Verifier},
    Signature, SigningKey, VerifyingKey,
};
use serde::{Deserialize, Serialize};
use zeroize::Zeroizing;

use crate::envelope::{decode_p256_pubkey_b64, EnvelopeError, EnvelopeResult};

/// Wire prefix for asymmetric P-256 app-minted capability tokens (`ADR-0007`).
pub const CTK2_TOKEN_PREFIX: &str = "ctk2.";
/// Domain separation prefix signed by `ECDSA-P256-SHA256` for `ctk2.*` tokens.
pub const CTK2_SIGNING_DOMAIN: &[u8] = b"jope.carbon.cap.v2.";
/// Default maximum allowed remaining lifetime (`86,400 s` = 24h) for an app-minted `ctk2.*` token.
pub const DEFAULT_MAX_CTK2_TTL_SECS: u64 = 86_400;

/// `AppTokenTier` is an enum naming the product tiers `app.veox.ai` is permitted to mint (`free` or `pro`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AppTokenTier {
    /// Free tier (`<= 5,000` rows, `<= 64` features, `fast`/`medium`).
    Free,
    /// Pro tier (`<= 100,000` rows, `<= 512` features, `fast`..`ultra`).
    Pro,
}

impl AppTokenTier {
    /// Lowercase wire string (`"free"` or `"pro"`).
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Free => "free",
            Self::Pro => "pro",
        }
    }
}

/// `AppCapabilityClaims` is a struct serialized inside `ctk2.<claims_b64url>.<sig_b64url>`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AppCapabilityClaims {
    pub tenant_id: String,
    pub tier: AppTokenTier,
    pub exp_epoch_secs: u64,
    #[serde(default)]
    pub admin: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub iss: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kid: Option<String>,
}

/// `AppSigningKey` is a struct wrapping an `ECDSA-P256` signing key held by `app.veox.ai` to mint
/// per-user `ctk2.*` tokens without holding `carbon-edge`'s symmetric `auth_secret_hex`.
pub struct AppSigningKey {
    signing_key: SigningKey,
    pubkey_uncompressed: [u8; P256_UNCOMPRESSED_PUBKEY_BYTES],
}

impl std::fmt::Debug for AppSigningKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AppSigningKey")
            .field("signing_key", &"[REDACTED]")
            .field("pubkey_b64", &self.pubkey_b64())
            .finish()
    }
}

impl AppSigningKey {
    /// Construct from a 32-byte big-endian NIST P-256 private scalar.
    pub fn from_scalar_bytes(scalar: &[u8; 32]) -> EnvelopeResult<Self> {
        let signing_key = SigningKey::from_slice(scalar)
            .map_err(|_| EnvelopeError::InvalidMaterial("invalid 32-byte P-256 signing scalar"))?;
        let pt = signing_key.verifying_key().to_sec1_point(false);
        let bytes = pt.as_bytes();
        if bytes.len() != P256_UNCOMPRESSED_PUBKEY_BYTES || bytes[0] != 0x04 {
            return Err(EnvelopeError::InvalidMaterial(
                "unexpected P-256 verifying key SEC1 encoding",
            ));
        }
        let mut pubkey_uncompressed = [0u8; P256_UNCOMPRESSED_PUBKEY_BYTES];
        pubkey_uncompressed.copy_from_slice(bytes);
        Ok(Self {
            signing_key,
            pubkey_uncompressed,
        })
    }

    /// Construct from a 64-character hex P-256 private scalar.
    pub fn from_hex(hex_str: &str) -> EnvelopeResult<Self> {
        let raw = Zeroizing::new(
            hex::decode(hex_str.trim())
                .map_err(|_| EnvelopeError::InvalidMaterial("invalid hex P-256 signing key"))?,
        );
        if raw.len() != 32 {
            return Err(EnvelopeError::InvalidMaterial(
                "P-256 signing key hex must decode to 32 bytes",
            ));
        }
        let mut arr = Zeroizing::new([0u8; 32]);
        arr.copy_from_slice(&raw);
        Self::from_scalar_bytes(&arr)
    }

    /// Parse a PKCS#8 PEM NIST P-256 signing key.
    pub fn from_pkcs8_pem(pem_str: &str) -> EnvelopeResult<Self> {
        use p256::pkcs8::DecodePrivateKey as _;
        let secret = p256::SecretKey::from_pkcs8_pem(pem_str.trim()).map_err(|_| {
            EnvelopeError::InvalidMaterial("invalid PKCS#8 PEM NIST P-256 private key")
        })?;
        let bytes = secret.to_bytes();
        let mut arr = Zeroizing::new([0u8; 32]);
        arr.copy_from_slice(bytes.as_slice());
        Self::from_scalar_bytes(&arr)
    }

    /// Encode this signing key as a `Zeroizing<String>` PKCS#8 PEM document.
    pub fn to_pkcs8_pem(&self) -> EnvelopeResult<Zeroizing<String>> {
        use p256::pkcs8::{EncodePrivateKey as _, LineEnding};
        let scalar = self.signing_key.to_bytes();
        let secret = p256::SecretKey::from_slice(scalar.as_slice())
            .map_err(|_| EnvelopeError::InvalidMaterial("invalid P-256 scalar"))?;
        let pem = secret
            .to_pkcs8_pem(LineEnding::LF)
            .map_err(|_| EnvelopeError::InvalidMaterial("encode PKCS#8 PEM failed"))?;
        Ok(Zeroizing::new(pem.as_str().to_owned()))
    }

    /// Encode the verifying key as an SPKI PEM (`-----BEGIN PUBLIC KEY-----`).
    pub fn pubkey_spki_pem(&self) -> EnvelopeResult<String> {
        use p256::pkcs8::{EncodePublicKey as _, LineEnding};
        let vk = p256::PublicKey::from_sec1_bytes(&self.pubkey_uncompressed)
            .map_err(|_| EnvelopeError::InvalidMaterial("invalid P-256 SEC1 public key"))?;
        vk.to_public_key_pem(LineEnding::LF)
            .map_err(|_| EnvelopeError::InvalidMaterial("encode SPKI PEM failed"))
    }

    /// Generate a fresh P-256 `ctk2.*` signing keypair using OS CSPRNG (non-`wasm32` targets).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn generate() -> EnvelopeResult<(Self, Zeroizing<String>)> {
        for _ in 0..8 {
            let mut seed = Zeroizing::new([0u8; 32]);
            getrandom::fill(seed.as_mut())
                .map_err(|_| EnvelopeError::InvalidMaterial("OS CSPRNG fill failed"))?;
            if let Ok(sk) = Self::from_scalar_bytes(&seed) {
                let hex_priv = Zeroizing::new(hex::encode(*seed));
                return Ok((sk, hex_priv));
            }
        }
        Err(EnvelopeError::InvalidMaterial(
            "failed to sample valid P-256 signing scalar",
        ))
    }

    /// Uncompressed 65-byte SEC1 public key (`0x04 || X || Y`) to pin in `CARBON_APP_AUTH_PUBKEYS`.
    pub fn pubkey_bytes(&self) -> &[u8; P256_UNCOMPRESSED_PUBKEY_BYTES] {
        &self.pubkey_uncompressed
    }

    /// Base64 encoding of the 65-byte uncompressed SEC1 public key for `CARBON_APP_AUTH_PUBKEYS`.
    pub fn pubkey_b64(&self) -> String {
        base64::engine::general_purpose::STANDARD.encode(self.pubkey_uncompressed)
    }

    /// Mint a deterministic RFC 6979 `ctk2.<claims_b64url>.<es256_sig_b64url>` capability token.
    pub fn mint_token(&self, claims: &AppCapabilityClaims) -> EnvelopeResult<String> {
        if claims.admin {
            return Err(EnvelopeError::InvalidMaterial(
                "app-minted ctk2 tokens cannot assert admin=true",
            ));
        }
        validate_tenant_id(&claims.tenant_id)?;
        let json_bytes = serde_json::to_vec(claims)
            .map_err(|_| EnvelopeError::InvalidMaterial("serialize ctk2 claims failed"))?;
        let claims_b64 = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&json_bytes);
        let msg = build_ctk2_signing_message(&claims_b64);
        let sig: Signature = self.signing_key.sign(&msg);
        let sig_b64 = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(sig.to_bytes());
        Ok(format!("{CTK2_TOKEN_PREFIX}{claims_b64}.{sig_b64}"))
    }
}

/// Parse a comma-separated list of Base64 65-byte SEC1 uncompressed P-256 public keys.
pub fn parse_pinned_pubkeys_csv(
    csv: &str,
) -> EnvelopeResult<Vec<[u8; P256_UNCOMPRESSED_PUBKEY_BYTES]>> {
    let mut out = Vec::new();
    for item in csv.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        out.push(decode_p256_pubkey_b64(item)?);
    }
    if out.is_empty() {
        return Err(EnvelopeError::InvalidMaterial(
            "no valid P-256 public keys found in list",
        ));
    }
    Ok(out)
}

/// Verify a `ctk2.<claims_b64url>.<es256_sig_b64url>` token against one or more pinned `app.veox.ai`
/// P-256 public keys (`[active, previous]`), enforcing `admin == false`, valid `tenant_id`, expiry,
/// and `max_ttl_secs`.
pub fn verify_ctk2_token_with_pubkeys(
    token: &str,
    pinned_pubkeys: &[[u8; P256_UNCOMPRESSED_PUBKEY_BYTES]],
    now_epoch_secs: u64,
    max_ttl_secs: u64,
) -> EnvelopeResult<AppCapabilityClaims> {
    if pinned_pubkeys.is_empty() {
        return Err(EnvelopeError::AuthenticationFailed(
            "ctk2 app issuer public keys are not configured on this gateway",
        ));
    }
    let trimmed = token.trim();
    let Some(rest) = trimmed.strip_prefix(CTK2_TOKEN_PREFIX) else {
        return Err(EnvelopeError::InvalidMaterial("missing ctk2. prefix"));
    };
    let Some((claims_b64, sig_b64)) = rest.split_once('.') else {
        return Err(EnvelopeError::InvalidMaterial(
            "malformed ctk2 token segments",
        ));
    };
    if claims_b64.is_empty() || sig_b64.is_empty() || sig_b64.contains('.') {
        return Err(EnvelopeError::InvalidMaterial(
            "malformed ctk2 token structure",
        ));
    }
    let sig_bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(sig_b64)
        .map_err(|_| EnvelopeError::InvalidMaterial("invalid base64url ctk2 signature"))?;
    let sig = Signature::from_slice(&sig_bytes).map_err(|_| {
        EnvelopeError::InvalidMaterial("ctk2 signature must be 64-byte IEEE P1363 ECDSA-P256")
    })?;

    let msg = build_ctk2_signing_message(claims_b64);
    let mut matched = false;
    for pub_sec1 in pinned_pubkeys {
        if let Ok(vk) = VerifyingKey::from_sec1_bytes(pub_sec1) {
            if vk.verify(&msg, &sig).is_ok() {
                matched = true;
                break;
            }
        }
    }
    if !matched {
        return Err(EnvelopeError::AuthenticationFailed(
            "ctk2 ECDSA-P256 signature verification failed",
        ));
    }

    let claims_raw = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(claims_b64)
        .map_err(|_| EnvelopeError::InvalidMaterial("invalid base64url ctk2 claims"))?;
    let claims: AppCapabilityClaims = serde_json::from_slice(&claims_raw)
        .map_err(|_| EnvelopeError::InvalidMaterial("invalid ctk2 claims JSON"))?;
    if claims.admin {
        return Err(EnvelopeError::AuthenticationFailed(
            "ctk2 tokens cannot grant admin privileges",
        ));
    }
    validate_tenant_id(&claims.tenant_id)?;
    if now_epoch_secs > claims.exp_epoch_secs {
        return Err(EnvelopeError::AuthenticationFailed(
            "ctk2 capability token has expired",
        ));
    }
    if claims.exp_epoch_secs.saturating_sub(now_epoch_secs) > max_ttl_secs {
        return Err(EnvelopeError::AuthenticationFailed(
            "ctk2 capability token exceeds maximum permitted TTL",
        ));
    }
    Ok(claims)
}

/// Build the canonical domain-separated byte buffer signed by `ctk2.*` (`b"jope.carbon.cap.v2." || claims_b64`).
pub fn build_ctk2_signing_message(claims_b64: &str) -> Vec<u8> {
    let mut msg = Vec::with_capacity(CTK2_SIGNING_DOMAIN.len() + claims_b64.len());
    msg.extend_from_slice(CTK2_SIGNING_DOMAIN);
    msg.extend_from_slice(claims_b64.as_bytes());
    msg
}

fn validate_tenant_id(tenant_id: &str) -> EnvelopeResult<()> {
    if tenant_id.is_empty() || tenant_id.len() > 64 {
        return Err(EnvelopeError::InvalidMaterial(
            "tenant_id must be 1..=64 ASCII characters",
        ));
    }
    let ok = tenant_id
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-');
    if !ok || tenant_id.starts_with('-') || tenant_id.starts_with('_') {
        return Err(EnvelopeError::InvalidMaterial(
            "tenant_id contains disallowed characters",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unauthorized_issuer_admin_escalation_and_invalid_tenant_id_fail_closed() {
        let trusted_issuer = AppSigningKey::from_scalar_bytes(&[0x17u8; 32]).expect("trusted key");
        let untrusted_issuer =
            AppSigningKey::from_scalar_bytes(&[0x23u8; 32]).expect("untrusted key");
        let pinned = vec![*trusted_issuer.pubkey_bytes()];

        let alice_claims = AppCapabilityClaims {
            tenant_id: "veox_owner_alice".to_owned(),
            tier: AppTokenTier::Pro,
            exp_epoch_secs: 1_800_000_900,
            admin: false,
            iss: Some("app.veox.ai".to_owned()),
            kid: Some("k-active".to_owned()),
        };
        let forged_by_non_owner_key = untrusted_issuer
            .mint_token(&alice_claims)
            .expect("mint untrusted");
        assert!(
            verify_ctk2_token_with_pubkeys(&forged_by_non_owner_key, &pinned, 1_800_000_000, 3_600)
                .is_err(),
            "non-owner signing key must be rejected"
        );

        let valid_alice = trusted_issuer
            .mint_token(&alice_claims)
            .expect("mint alice");
        let verified = verify_ctk2_token_with_pubkeys(&valid_alice, &pinned, 1_800_000_000, 3_600)
            .expect("verify alice");
        assert_eq!(verified.tenant_id, "veox_owner_alice");
        assert_ne!(verified.tenant_id, "veox_non_owner_bob");

        // Admin escalation and path-traversal tenant_ids must be rejected at mint & verify boundaries.
        let mut esc = alice_claims.clone();
        esc.admin = true;
        assert!(trusted_issuer.mint_token(&esc).is_err());

        let mut bad_tenant = alice_claims;
        bad_tenant.tenant_id = "../veox_other".to_owned();
        assert!(trusted_issuer.mint_token(&bad_tenant).is_err());
    }
}
