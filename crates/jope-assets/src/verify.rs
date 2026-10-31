//! Trusted keys and the package signature.
//!
//! Release packages are signed by the root-held cosign key whose public half is
//! `docs/release/signing.pub`, compiled in by `build.rs`. Fixture packages are signed by a
//! deterministic development key derived in code, which only `dev-keys` builds trust. A key is
//! named by its id, the sha256 of its SPKI DER (distinct from the sha256 of the PEM file).
//!
//! The file carries the signature as fixed-width `r || s` with a low `S`. `jope-dist seal`
//! converts cosign's DER output and normalizes `S`; the reader refuses a high `S`, so a sealed
//! file has exactly one valid signature encoding.

use crate::{AssetError, SIGNING_DOMAIN};
use p256::ecdsa::{signature::Verifier, Signature, VerifyingKey};
use p256::pkcs8::{DecodePublicKey, EncodePublicKey};

include!(concat!(env!("OUT_DIR"), "/release_keys.rs"));

/// The label the development key is derived from; changing it changes the key.
const DEVELOPMENT_KEY_LABEL: &[u8] =
    b"jope-assets development key v1 (fixture packages only, never a release)";

/// sha256 of a key's SPKI DER.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KeyId(pub [u8; 32]);

impl std::fmt::Display for KeyId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sha256:{}", hex::encode(self.0))
    }
}

#[derive(Clone, Debug)]
/// A public verification key and the role its signer is allowed to serve.
pub struct TrustedKey {
    /// SHA-256 identity of this key's SPKI DER.
    pub id: KeyId,
    /// `release` or `development`.
    pub role: &'static str,
    key: VerifyingKey,
}

impl TrustedKey {
    /// Parse an SPKI PEM key and bind it to the supplied package-signing role.
    pub fn from_pem(pem: &str, role: &'static str) -> Result<Self, String> {
        let key = VerifyingKey::from_public_key_pem(pem.trim()).map_err(|e| e.to_string())?;
        Ok(Self {
            id: key_id(&key),
            role,
            key,
        })
    }

    /// The key as SPKI PEM.
    pub fn to_pem(&self) -> String {
        self.key
            .to_public_key_pem(p256::pkcs8::LineEnding::LF)
            .expect("P-256 keys encode as SPKI PEM")
    }
}

/// The id of a verifying key.
pub fn key_id(key: &VerifyingKey) -> KeyId {
    use sha2::{Digest, Sha256};
    let der = key.to_public_key_der().expect("P-256 keys encode as SPKI");
    KeyId(Sha256::digest(der.as_bytes()).into())
}

/// The deterministic development signing key.
pub fn development_signing_key() -> p256::ecdsa::SigningKey {
    use sha2::{Digest, Sha256};
    let seed = Sha256::digest(DEVELOPMENT_KEY_LABEL);
    p256::ecdsa::SigningKey::from_slice(&seed).expect("the seed is a valid P-256 scalar")
}

/// The development key as a [`TrustedKey`], for tools that verify fixture packages explicitly.
pub fn development_key() -> TrustedKey {
    let key = *development_signing_key().verifying_key();
    TrustedKey {
        id: key_id(&key),
        role: "development",
        key,
    }
}

/// Whether this build trusts the development key.
pub const TRUSTS_DEVELOPMENT_KEY: bool = cfg!(feature = "dev-keys");

/// The compiled-in release keys.
pub fn release_keys() -> Vec<TrustedKey> {
    RELEASE_KEYS_PEM
        .iter()
        .map(|pem| TrustedKey::from_pem(pem, "release").expect("docs/release/signing.pub parses"))
        .collect()
}

/// Every key this build trusts: the release keys, plus the development key under `dev-keys`.
pub fn trusted_keys() -> Vec<TrustedKey> {
    let mut keys = release_keys();
    if TRUSTS_DEVELOPMENT_KEY {
        keys.push(development_key());
    }
    keys
}

/// The exact bytes a package signature covers: [`SIGNING_DOMAIN`] then the 128 footer bytes.
pub fn signing_message(footer: &[u8; crate::FOOTER_LEN as usize]) -> Vec<u8> {
    let mut message = Vec::with_capacity(SIGNING_DOMAIN.len() + footer.len());
    message.extend_from_slice(SIGNING_DOMAIN);
    message.extend_from_slice(footer);
    message
}

/// Verify a fixed-width signature over `message` with the trusted key `claimed`.
pub fn verify<'k>(
    message: &[u8],
    signature: &[u8; 64],
    claimed: KeyId,
    keys: &'k [TrustedKey],
) -> Result<&'k TrustedKey, AssetError> {
    let key = keys
        .iter()
        .find(|k| k.id == claimed)
        .ok_or_else(|| AssetError::UntrustedKey(claimed.to_string()))?;
    let signature = Signature::from_slice(signature).map_err(|_| AssetError::Signature)?;
    if signature.normalize_s() != signature {
        // A high S is valid ECDSA but not the encoding a sealed file carries.
        return Err(AssetError::Signature);
    }
    key.key
        .verify(message, &signature)
        .map_err(|_| AssetError::Signature)?;
    Ok(key)
}

/// A DER signature (cosign's `sign-blob` output, after base64) as the fixed-width low-S form a
/// package carries. Go's `crypto/ecdsa` does not normalize S; `(r, n - s)` verifies the same.
pub fn normalize_der(der: &[u8]) -> Result<[u8; 64], AssetError> {
    let signature = Signature::from_der(der).map_err(|_| AssetError::Signature)?;
    let signature = signature.normalize_s();
    Ok(signature.to_bytes().into())
}

/// Sign with the development key, as `cosign sign-blob` would with the release key (DER).
#[cfg(feature = "pack")]
pub fn sign_with_development_key(message: &[u8]) -> Vec<u8> {
    use p256::ecdsa::signature::Signer;
    let signature: Signature = development_signing_key().sign(message);
    signature.to_der().as_bytes().to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use p256::ecdsa::signature::Signer;

    fn sign(message: &[u8]) -> [u8; 64] {
        let signature: Signature = development_signing_key().sign(message);
        normalize_der(signature.to_der().as_bytes()).unwrap()
    }

    #[test]
    fn development_key_signs_and_verifies() {
        let keys = [development_key()];
        let id = keys[0].id;
        let signature = sign(b"footer");
        assert_eq!(verify(b"footer", &signature, id, &keys).unwrap().id, id);
        assert!(matches!(
            verify(b"footeR", &signature, id, &keys),
            Err(AssetError::Signature)
        ));
        assert!(matches!(
            verify(b"footer", &signature, KeyId([0; 32]), &keys),
            Err(AssetError::UntrustedKey(_))
        ));
        assert!(matches!(
            verify(b"footer", &[0; 64], id, &keys),
            Err(AssetError::Signature)
        ));
    }

    #[test]
    fn high_s_is_normalized_on_seal_and_refused_in_a_file() {
        let keys = [development_key()];
        let id = keys[0].id;
        let low: Signature = development_signing_key().sign(b"m");
        let low = low.normalize_s();
        let (r, s) = low.split_scalars();
        let high = Signature::from_scalars(r, -*s).unwrap();
        // cosign may emit either; sealing normalizes to the low form.
        assert_eq!(
            normalize_der(high.to_der().as_bytes()).unwrap(),
            <[u8; 64]>::from(low.to_bytes())
        );
        let raw_high: [u8; 64] = high.to_bytes().into();
        assert!(matches!(
            verify(b"m", &raw_high, id, &keys),
            Err(AssetError::Signature)
        ));
    }

    #[test]
    fn pem_round_trip_keeps_the_id_and_release_key_parses() {
        let key = development_key();
        assert_eq!(
            TrustedKey::from_pem(&key.to_pem(), "release").unwrap().id,
            key.id
        );
        let release = release_keys();
        assert_eq!(release.len(), 1);
        assert_ne!(release[0].id, key.id);
        assert_eq!(
            trusted_keys().len(),
            1 + usize::from(TRUSTS_DEVELOPMENT_KEY)
        );
    }

    #[test]
    fn the_message_is_domain_separated() {
        let footer = [7u8; 128];
        let message = signing_message(&footer);
        assert!(message.starts_with(b"jope.singlefile.manifest-signature.v1\0"));
        assert_eq!(&message[SIGNING_DOMAIN.len()..], &footer);
    }
}
