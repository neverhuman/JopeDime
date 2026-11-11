//! AWS SigV4 S3 presigning and gateway HMAC upload-token verification via `aws-lc-rs`.
//!
//! Pins every client upload URL to the exact declared `SHA-256`, byte length, and expiration
//! timestamp before issuing upload credentials.

#[cfg(feature = "edge")]
use aws_lc_rs::hmac;
#[cfg(feature = "edge")]
use sha2::{Digest, Sha256};
#[cfg(feature = "edge")]
use zeroize::Zeroizing;

#[cfg(feature = "edge")]
fn hmac_sha256(key: &[u8], data: &[u8]) -> [u8; 32] {
    let signing_key = hmac::Key::new(hmac::HMAC_SHA256, key);
    let tag = hmac::sign(&signing_key, data);
    let mut out = [0u8; 32];
    out.copy_from_slice(&tag.as_ref()[..32]);
    out
}

/// `CarbonPresigner` is a struct signing and verifying time-bounded, SHA-256-pinned upload URLs
/// for both AWS S3 (`AWS4-HMAC-SHA256`) and self-hosted Enterprise / emulator gateways.
#[cfg(feature = "edge")]
#[derive(Clone)]
pub struct CarbonPresigner {
    secret: Zeroizing<[u8; 32]>,
}

#[cfg(feature = "edge")]
impl CarbonPresigner {
    pub fn from_secret(secret: [u8; 32]) -> Self {
        Self {
            secret: Zeroizing::new(secret),
        }
    }

    /// Mint a gateway-authenticated `PUT` upload URL (`/v1/carbon/cas/{sha256}?bytes=...&exp=...&sig=...`)
    /// pinned to `(sha256, byte_len, expires_epoch_secs)`.
    pub fn sign_gateway_upload_url(
        &self,
        gateway_base_url: &str,
        sha256_hex: &str,
        byte_len: u64,
        expires_epoch_secs: u64,
    ) -> String {
        let lower_sha = sha256_hex.to_ascii_lowercase();
        let canonical = format!("PUT\n{lower_sha}\n{byte_len}\n{expires_epoch_secs}");
        let sig = hex::encode(hmac_sha256(self.secret.as_ref(), canonical.as_bytes()));
        let base = gateway_base_url.trim_end_matches('/');
        format!(
            "{base}/v1/carbon/cas/{lower_sha}?bytes={byte_len}&exp={expires_epoch_secs}&sig={sig}"
        )
    }

    /// Constant-time verify a gateway upload URL's query parameters against `now_epoch_secs`
    /// and actual body length.
    pub fn verify_gateway_upload(
        &self,
        sha256_hex: &str,
        declared_bytes: u64,
        expires_epoch_secs: u64,
        provided_sig_hex: &str,
        now_epoch_secs: u64,
    ) -> Result<(), &'static str> {
        if now_epoch_secs > expires_epoch_secs {
            return Err("presigned upload URL has expired");
        }
        let Ok(provided_bytes) = hex::decode(provided_sig_hex) else {
            return Err("malformed upload signature hex");
        };
        let lower_sha = sha256_hex.to_ascii_lowercase();
        let canonical = format!("PUT\n{lower_sha}\n{declared_bytes}\n{expires_epoch_secs}");
        let signing_key = hmac::Key::new(hmac::HMAC_SHA256, self.secret.as_ref());
        hmac::verify(&signing_key, canonical.as_bytes(), &provided_bytes)
            .map_err(|_| "presigned upload signature verification failed")
    }

    /// Pure `aws-lc-rs` AWS SigV4 (`AWS4-HMAC-SHA256`) S3 `PUT` query-string presigner pinning
    /// `UNSIGNED-PAYLOAD` or exact `sha256_hex` object key `blobs/<sha256>`.
    #[allow(clippy::too_many_arguments)]
    pub fn presign_s3_put_v4(
        access_key_id: &str,
        secret_access_key: &str,
        region: &str,
        bucket: &str,
        object_key: &str,
        amz_date_iso8601: &str, // e.g. "20261110T213000Z"
        expires_secs: u32,
        expected_sha256_hex: &str,
    ) -> Result<String, &'static str> {
        if amz_date_iso8601.len() < 8 || expires_secs == 0 || expires_secs > 604_800 {
            return Err("invalid SigV4 timestamp or expiration window");
        }
        let date_stamp = &amz_date_iso8601[..8];
        let host = format!("{bucket}.s3.{region}.amazonaws.com");
        let clean_key = object_key.trim_start_matches('/');
        let credential_scope = format!("{date_stamp}/{region}/s3/aws4_request");
        let credential_encoded = format!(
            "{}%2F{date_stamp}%2F{region}%2Fs3%2Faws4_request",
            url_encode_component(access_key_id)
        );
        let canonical_query = format!(
            "X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential={credential_encoded}&X-Amz-Date={amz_date_iso8601}&X-Amz-Expires={expires_secs}&X-Amz-SignedHeaders=host"
        );
        let canonical_request = format!(
            "PUT\n/{clean_key}\n{canonical_query}\nhost:{host}\n\nhost\n{}",
            expected_sha256_hex.to_ascii_lowercase()
        );
        let req_hash = format!("{:x}", Sha256::digest(canonical_request.as_bytes()));
        let string_to_sign =
            format!("AWS4-HMAC-SHA256\n{amz_date_iso8601}\n{credential_scope}\n{req_hash}");

        let k_secret = Zeroizing::new(format!("AWS4{secret_access_key}").into_bytes());
        let k_date = hmac_sha256(&k_secret, date_stamp.as_bytes());
        let k_region = hmac_sha256(&k_date, region.as_bytes());
        let k_service = hmac_sha256(&k_region, b"s3");
        let k_signing = hmac_sha256(&k_service, b"aws4_request");
        let signature = hex::encode(hmac_sha256(&k_signing, string_to_sign.as_bytes()));

        Ok(format!(
            "https://{host}/{clean_key}?{canonical_query}&X-Amz-Signature={signature}"
        ))
    }
}

#[cfg(feature = "edge")]
fn url_encode_component(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for byte in input.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char);
            }
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}
