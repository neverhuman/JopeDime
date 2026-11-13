//! Cryptographic security boundaries for Carbon serverless cells:
//! - [`auth`]: HMAC-SHA256 capability tokens (`ctk1.*`), strict `tenant_id` validation, and sliding-window rate limiting.
//! - [`secrets_vault`]: Runtime SigV4 `secretsmanager:GetSecretValue` secret loading into `Zeroizing` RAM with dual-key rotation.
//! - [`user_envelope`]: Zero-knowledge per-user `ECDH P-256 + HKDF-SHA256 + AES-256-GCM` (`CUEN\x01` & `CUDK\x01`) encryption.
//! - [`seal`]: Domain-separated (`CheckpointArchive` vs `ChampionModel`) `CCKP\x02` state sealing double-bound to `U_DEK`.
//! - [`sigv4`]: Pure `aws-lc-rs` AWS SigV4 (`AWS4-HMAC-SHA256`) request signing and size/SHA-pinned S3 `PUT`/`GET` presigning.

pub mod auth;
pub mod seal;
#[cfg(feature = "edge")]
pub mod secrets_vault;
pub mod sigv4;
pub mod user_envelope;
