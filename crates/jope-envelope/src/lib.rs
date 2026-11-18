//! Standalone, pure-Rust, `wasm32-unknown-unknown`-compatible zero-knowledge client envelope
//! encryption (`CUEN\x01` + `CUDK\x01`) and asymmetric `ECDSA-P256` `ctk2.*` capability token
//! helpers for `jAIn`, browser clients, `app.veox.ai`, and `carbon-cli` (`ADR-0007` Phase C7).
#![forbid(unsafe_code)]

pub mod aead;
pub mod envelope;
pub mod token;

pub use envelope::{
    build_slot_hkdf_info, decode_p256_pubkey_b64, decrypt_cuen_payload,
    decrypt_cuen_with_slot_subkey, derive_cuen_slot_subkey, encrypt_cuen_payload_with_nonce,
    unwrap_u_dek_with_keypair, wrap_u_dek_to_pubkey_with_material, ClientEnvelopeKeypair,
    EnvelopeError, EnvelopeResult, CAS_BLOB_CUEN_SCOPE, CUDK_HKDF_SALT, CUEN_HKDF_SALT,
};
#[cfg(not(target_arch = "wasm32"))]
pub use envelope::{encrypt_cuen_payload, generate_random_u_dek, wrap_u_dek_to_pubkey};
pub use jope_prep::cuen::{
    CuenPayloadKind, CUDK_WIRE_BYTES, CUEN_HEADER_BYTES, CUEN_NONCE_BYTES, CUEN_TAG_BYTES,
    P256_UNCOMPRESSED_PUBKEY_BYTES,
};
pub use token::{
    build_ctk2_signing_message, parse_pinned_pubkeys_csv, verify_ctk2_token_with_pubkeys,
    AppCapabilityClaims, AppSigningKey, AppTokenTier, CTK2_SIGNING_DOMAIN, CTK2_TOKEN_PREFIX,
    DEFAULT_MAX_CTK2_TTL_SECS,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cudk_and_cuen_roundtrip_and_reject_wrong_scope() {
        let engine_kp = ClientEnvelopeKeypair::from_scalar_bytes(&[0x11u8; 32]).unwrap();
        let u_dek = [0x42u8; 32];
        let eph_scalar = [0x22u8; 32];
        let nonce = [0x33u8; 12];

        let cudk = wrap_u_dek_to_pubkey_with_material(
            &u_dek,
            engine_kp.pubkey_bytes(),
            "veox_acct_01",
            "run_01JXYZ",
            &eph_scalar,
            &nonce,
        )
        .unwrap();
        let unwrapped =
            unwrap_u_dek_with_keypair(&engine_kp, &cudk, "veox_acct_01", "run_01JXYZ").unwrap();
        assert_eq!(*unwrapped, u_dek);

        // Wrong tenant or run_id must fail closed.
        assert!(unwrap_u_dek_with_keypair(&engine_kp, &cudk, "veox_other", "run_01JXYZ").is_err());
        assert!(unwrap_u_dek_with_keypair(&engine_kp, &cudk, "veox_acct_01", "run_other").is_err());

        let payload = b"fit-jmat-binary-payload-v1";
        let cuen = encrypt_cuen_payload_with_nonce(
            &u_dek,
            CuenPayloadKind::FitJmat,
            "veox_acct_01",
            CAS_BLOB_CUEN_SCOPE,
            &[0x55u8; 12],
            payload,
        )
        .unwrap();
        let pt = decrypt_cuen_payload(
            &u_dek,
            CuenPayloadKind::FitJmat,
            "veox_acct_01",
            CAS_BLOB_CUEN_SCOPE,
            &cuen,
        )
        .unwrap();
        assert_eq!(pt.as_slice(), payload);

        // Slot-kind swap (FitJmat vs HoldoutJmat) must fail closed.
        assert!(decrypt_cuen_payload(
            &u_dek,
            CuenPayloadKind::HoldoutJmat,
            "veox_acct_01",
            CAS_BLOB_CUEN_SCOPE,
            &cuen,
        )
        .is_err());
    }

    #[test]
    fn ctk2_mint_verify_and_dual_key_rotation_work() {
        let old_key = AppSigningKey::from_scalar_bytes(&[0x19u8; 32]).unwrap();
        let new_key = AppSigningKey::from_scalar_bytes(&[0x29u8; 32]).unwrap();
        let csv = format!("{}, {}", new_key.pubkey_b64(), old_key.pubkey_b64());
        let pinned = parse_pinned_pubkeys_csv(&csv).unwrap();

        let claims = AppCapabilityClaims {
            tenant_id: "veox_user_abc123".to_owned(),
            tier: AppTokenTier::Pro,
            exp_epoch_secs: 1_800_000_600,
            admin: false,
            iss: Some("app.veox.ai".to_owned()),
            kid: Some("veox-2026-11".to_owned()),
        };
        let tok_old = old_key.mint_token(&claims).unwrap();
        let tok_new = new_key.mint_token(&claims).unwrap();

        let v1 = verify_ctk2_token_with_pubkeys(&tok_old, &pinned, 1_800_000_000, 3_600).unwrap();
        let v2 = verify_ctk2_token_with_pubkeys(&tok_new, &pinned, 1_800_000_000, 3_600).unwrap();
        assert_eq!(v1, claims);
        assert_eq!(v2, claims);

        // Expired or excessive TTL rejected.
        assert!(verify_ctk2_token_with_pubkeys(&tok_new, &pinned, 1_800_000_601, 3_600).is_err());
        assert!(verify_ctk2_token_with_pubkeys(&tok_new, &pinned, 1_800_000_000, 300).is_err());
    }
}
