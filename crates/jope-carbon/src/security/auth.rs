//! Capability-token authentication (`HMAC-SHA256`), strict injective `tenant_id` validation,
//! and bounded sliding-window edge rate limiting (`RT-02`, `RT-08`, `RT-11`).

#[cfg(feature = "edge")]
use std::collections::BTreeMap;
#[cfg(feature = "edge")]
use std::sync::{Arc, Mutex};

#[cfg(feature = "edge")]
use aws_lc_rs::hmac;
#[cfg(feature = "edge")]
use base64::Engine as _;
#[cfg(feature = "edge")]
use jope_api::{
    carbon::{CarbonAnonymousAuthResponse, CarbonRegisterFreeResponse, CarbonTierWire},
    decimal::DecimalU64,
};
#[cfg(feature = "edge")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "edge")]
use zeroize::Zeroizing;

#[cfg(feature = "edge")]
use crate::error::{CarbonError, CarbonResult};

#[cfg(feature = "edge")]
const TOKEN_PREFIX: &str = "ctk1.";
#[cfg(feature = "edge")]
const MAX_RATE_BUCKETS: usize = 65_536;

/// Strictly validate `tenant_id` (`^[A-Za-z0-9_-]{3,64}$`) so two distinct tenants can never
/// collide after path normalization (`RT-08`).
pub fn validate_strict_tenant_id(tenant_id: &str) -> CarbonResult<()> {
    let len = tenant_id.len();
    if !(3..=64).contains(&len) {
        return Err(CarbonError::CapabilityAuthFailed {
            reason: "tenant_id length must be between 3 and 64 ASCII characters",
        });
    }
    if !tenant_id
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
    {
        return Err(CarbonError::CapabilityAuthFailed {
            reason: "tenant_id must match ^[A-Za-z0-9_-]{3,64}$",
        });
    }
    Ok(())
}

/// Strictly validate `run_id` path segment (`^[A-Za-z0-9_-]{4,80}$`).
pub fn validate_strict_run_id(run_id: &str) -> CarbonResult<()> {
    let len = run_id.len();
    if !(4..=80).contains(&len)
        || !run_id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
    {
        return Err(CarbonError::CapabilityAuthFailed {
            reason: "run_id must match ^[A-Za-z0-9_-]{4,80}$",
        });
    }
    Ok(())
}

/// `CarbonCapabilityClaims` is a struct authenticated inside a `ctk1.<claims_b64>.<hmac_hex>`
/// Bearer capability token.
#[cfg(feature = "edge")]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CarbonCapabilityClaims {
    pub tenant_id: String,
    pub tier: CarbonTierWire,
    pub exp_epoch_secs: u64,
    #[serde(default)]
    pub admin: bool,
}

#[cfg(feature = "edge")]
impl CarbonCapabilityClaims {
    /// Decode the cleartext `CarbonCapabilityClaims` from a `ctk1.<claims_b64url>.<hmac_hex>`
    /// capability token on the client side without possessing the server's `HMAC-SHA256` secret.
    ///
    /// Note: Server-side authorization MUST always call [`CarbonAuthVerifier::verify_token`].
    pub fn decode_unverified(token: &str) -> CarbonResult<Self> {
        let trimmed = token.trim();
        let Some(after_prefix) = trimmed.strip_prefix(TOKEN_PREFIX) else {
            return Err(CarbonError::CapabilityAuthFailed {
                reason: "unrecognized capability token version prefix (expected ctk1.)",
            });
        };
        let Some((payload_b64, sig_hex)) = after_prefix.split_once('.') else {
            return Err(CarbonError::CapabilityAuthFailed {
                reason: "malformed capability token segments",
            });
        };
        let sig_bytes = hex::decode(sig_hex).map_err(|_| CarbonError::CapabilityAuthFailed {
            reason: "malformed capability token signature hex",
        })?;
        if sig_bytes.len() != 32 {
            return Err(CarbonError::CapabilityAuthFailed {
                reason: "capability token HMAC-SHA256 tag must be 32 bytes (64 hex chars)",
            });
        }
        let claims_bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(payload_b64)
            .map_err(|_| CarbonError::CapabilityAuthFailed {
                reason: "malformed capability token base64url payload",
            })?;
        let claims: Self = serde_json::from_slice(&claims_bytes).map_err(|_| {
            CarbonError::CapabilityAuthFailed {
                reason: "invalid capability token claims JSON",
            }
        })?;
        validate_strict_tenant_id(&claims.tenant_id)?;
        Ok(claims)
    }
}

/// `CarbonAuthVerifier` is a struct signing and constant-time verifying `HMAC-SHA256` capability
/// tokens for Free anonymous sessions and Pro/Admin tenants.
#[cfg(feature = "edge")]
#[derive(Clone)]
pub struct CarbonAuthVerifier {
    secret: Zeroizing<[u8; 32]>,
}

#[cfg(feature = "edge")]
impl CarbonAuthVerifier {
    pub fn from_secret(secret: [u8; 32]) -> Self {
        Self {
            secret: Zeroizing::new(secret),
        }
    }

    pub fn from_hex(hex_str: &str) -> CarbonResult<Self> {
        let raw = hex::decode(hex_str.trim()).map_err(|_| CarbonError::CapabilityAuthFailed {
            reason: "CARBON_AUTH_SECRET_HEX must be valid hex",
        })?;
        if raw.len() != 32 {
            return Err(CarbonError::CapabilityAuthFailed {
                reason: "CARBON_AUTH_SECRET_HEX must decode to 32 bytes",
            });
        }
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&raw);
        Ok(Self::from_secret(arr))
    }

    /// Sign `claims` into a URL-safe `ctk1.<payload_b64url>.<sig_hex>` capability token.
    pub fn mint_token(&self, claims: &CarbonCapabilityClaims) -> CarbonResult<String> {
        validate_strict_tenant_id(&claims.tenant_id)?;
        let json_bytes =
            serde_json::to_vec(claims).map_err(|_| CarbonError::CapabilityAuthFailed {
                reason: "failed to encode capability claims",
            })?;
        let payload_b64 = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&json_bytes);
        let signing_input = format!("{TOKEN_PREFIX}{payload_b64}");
        let key = hmac::Key::new(hmac::HMAC_SHA256, self.secret.as_ref());
        let tag = hmac::sign(&key, signing_input.as_bytes());
        let sig_hex = hex::encode(tag.as_ref());
        Ok(format!("{signing_input}.{sig_hex}"))
    }

    /// Mint a fresh 1-hour Free-tier capability token bound to a unique `anon_<ulid>` tenant ID.
    pub fn mint_anonymous_free(
        &self,
        now_epoch_secs: u64,
        engine_kek_pubkey_b64: Option<String>,
    ) -> CarbonResult<CarbonAnonymousAuthResponse> {
        let tenant_id = format!(
            "anon_{}",
            ulid::Ulid::new().to_string().to_ascii_lowercase()
        );
        let expires = now_epoch_secs.saturating_add(3_600);
        let claims = CarbonCapabilityClaims {
            tenant_id: tenant_id.clone(),
            tier: CarbonTierWire::Free,
            exp_epoch_secs: expires,
            admin: false,
        };
        let capability_token = self.mint_token(&claims)?;
        Ok(CarbonAnonymousAuthResponse {
            tenant_id,
            capability_token,
            tier: CarbonTierWire::Free,
            expires_epoch_secs: DecimalU64(expires),
            engine_kek_pubkey_b64,
        })
    }

    /// Mint a persistent 30-day (`2,592,000 s`) Free-tier capability token bound to a unique
    /// `free_<label>_<ulid12>` (or `free_<ulid>`) `tenant_id` for `POST /v1/carbon/auth/register`.
    pub fn mint_registered_free(
        &self,
        label: Option<&str>,
        now_epoch_secs: u64,
        engine_kek_pubkey_b64: Option<String>,
    ) -> CarbonResult<CarbonRegisterFreeResponse> {
        let raw_ulid = ulid::Ulid::new().to_string().to_ascii_lowercase();
        let tenant_id = match label.map(str::trim).filter(|s| !s.is_empty()) {
            Some(lbl) => {
                if lbl.len() > 24
                    || !lbl
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
                {
                    return Err(CarbonError::CapabilityAuthFailed {
                        reason: "registration label must be 1..=24 ASCII [A-Za-z0-9_-] characters",
                    });
                }
                let clean: String = lbl
                    .chars()
                    .map(|c| {
                        if c == '-' {
                            '_'
                        } else {
                            c.to_ascii_lowercase()
                        }
                    })
                    .collect();
                let suffix = &raw_ulid[raw_ulid.len().saturating_sub(12)..];
                format!("free_{clean}_{suffix}")
            }
            None => format!("free_{raw_ulid}"),
        };
        let expires = now_epoch_secs.saturating_add(30 * 86_400);
        let claims = CarbonCapabilityClaims {
            tenant_id: tenant_id.clone(),
            tier: CarbonTierWire::Free,
            exp_epoch_secs: expires,
            admin: false,
        };
        let capability_token = self.mint_token(&claims)?;
        Ok(CarbonRegisterFreeResponse {
            tenant_id,
            capability_token,
            tier: CarbonTierWire::Free,
            expires_epoch_secs: DecimalU64(expires),
            engine_kek_pubkey_b64,
        })
    }

    /// Verify an HTTP `Authorization: Bearer <token>` header in constant time against `now_epoch_secs`.
    pub fn verify_bearer_header(
        &self,
        auth_header: Option<&str>,
        now_epoch_secs: u64,
    ) -> CarbonResult<CarbonCapabilityClaims> {
        let Some(raw_hdr) = auth_header else {
            return Err(CarbonError::CapabilityAuthFailed {
                reason: "missing Authorization Bearer token",
            });
        };
        let trimmed = raw_hdr.trim();
        let token = trimmed
            .strip_prefix("Bearer ")
            .or_else(|| trimmed.strip_prefix("bearer "))
            .ok_or(CarbonError::CapabilityAuthFailed {
                reason: "Authorization header must use Bearer scheme",
            })?
            .trim();
        self.verify_token(token, now_epoch_secs)
    }

    /// Constant-time verify a raw `ctk1.<payload>.<sig>` token string.
    pub fn verify_token(
        &self,
        token: &str,
        now_epoch_secs: u64,
    ) -> CarbonResult<CarbonCapabilityClaims> {
        let Some(after_prefix) = token.strip_prefix(TOKEN_PREFIX) else {
            return Err(CarbonError::CapabilityAuthFailed {
                reason: "unrecognized capability token version prefix",
            });
        };
        let Some((payload_b64, sig_hex)) = after_prefix.split_once('.') else {
            return Err(CarbonError::CapabilityAuthFailed {
                reason: "malformed capability token segments",
            });
        };
        let sig_bytes = hex::decode(sig_hex).map_err(|_| CarbonError::CapabilityAuthFailed {
            reason: "malformed capability token signature hex",
        })?;
        let signing_input = format!("{TOKEN_PREFIX}{payload_b64}");
        let key = hmac::Key::new(hmac::HMAC_SHA256, self.secret.as_ref());
        hmac::verify(&key, signing_input.as_bytes(), &sig_bytes).map_err(|_| {
            CarbonError::CapabilityAuthFailed {
                reason: "capability token HMAC signature verification failed",
            }
        })?;

        let claims_bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(payload_b64)
            .map_err(|_| CarbonError::CapabilityAuthFailed {
                reason: "malformed capability token base64url payload",
            })?;
        let claims: CarbonCapabilityClaims =
            serde_json::from_slice(&claims_bytes).map_err(|_| {
                CarbonError::CapabilityAuthFailed {
                    reason: "invalid capability token claims JSON",
                }
            })?;
        validate_strict_tenant_id(&claims.tenant_id)?;
        if now_epoch_secs > claims.exp_epoch_secs {
            return Err(CarbonError::CapabilityAuthFailed {
                reason: "capability token has expired",
            });
        }
        Ok(claims)
    }
}

/// `CarbonRateLimiter` is a struct enforcing bounded fixed-window request quotas per subject/route
/// inside warm `carbon-edge` containers (`RT-11`).
#[cfg(feature = "edge")]
#[derive(Clone)]
pub struct CarbonRateLimiter {
    window_secs: u64,
    buckets: Arc<Mutex<BTreeMap<String, (u64, u32)>>>,
}

#[cfg(feature = "edge")]
impl CarbonRateLimiter {
    pub fn new(window_secs: u64) -> Self {
        Self {
            window_secs: window_secs.max(1),
            buckets: Arc::new(Mutex::new(BTreeMap::new())),
        }
    }

    /// Increment `bucket_key` in the current window and return `true` if within `max_per_window`.
    pub fn check_and_record(
        &self,
        bucket_key: &str,
        now_epoch_secs: u64,
        max_per_window: u32,
    ) -> bool {
        let slot = now_epoch_secs / self.window_secs;
        let Ok(mut guard) = self.buckets.lock() else {
            return false;
        };
        if guard.len() >= MAX_RATE_BUCKETS && !guard.contains_key(bucket_key) {
            guard.retain(|_, (s, _)| *s == slot);
            if guard.len() >= MAX_RATE_BUCKETS {
                return false;
            }
        }
        let entry = guard.entry(bucket_key.to_owned()).or_insert((slot, 0));
        if entry.0 != slot {
            *entry = (slot, 0);
        }
        if entry.1 >= max_per_window {
            return false;
        }
        entry.1 = entry.1.saturating_add(1);
        true
    }
}
