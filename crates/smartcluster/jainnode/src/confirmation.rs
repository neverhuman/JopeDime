use ed25519_dalek::SigningKey;
use jain_smartcluster_fabric::{valid_identifier, Digest32, DomainSignature, SignatureAuthority};

use crate::{NodeControlFrame, ProtocolError};

pub(crate) const OUTBOUND_CONFIRMATION_DOMAIN: &[u8] = b"jain.node.outbound-confirmation.v1\0";

impl NodeControlFrame {
    #[allow(clippy::too_many_arguments)]
    pub fn outbound_confirmation(
        session_id: String,
        outbound_sequence: u64,
        frame_digest: Digest32,
        at_ms: u64,
        issuer: String,
        key_id: String,
        signing_key: &SigningKey,
    ) -> Result<Self, ProtocolError> {
        if !valid_identifier(&session_id)
            || !valid_identifier(&issuer)
            || !valid_identifier(&key_id)
            || outbound_sequence == 0
            || frame_digest == [0; 32]
            || at_ms == 0
        {
            return Err(ProtocolError::InvalidConfiguration);
        }
        let mut frame = Self::ConfirmOutbound {
            session_id,
            outbound_sequence,
            frame_digest,
            at_ms,
            signature: DomainSignature {
                authority: SignatureAuthority::NodeSession,
                issuer: issuer.clone(),
                key_id: key_id.clone(),
                public_key: signing_key.verifying_key().to_bytes(),
                signature: Vec::new(),
            },
        };
        let signed_bytes = frame.signed_control_bytes()?;
        let Self::ConfirmOutbound { signature, .. } = &mut frame else {
            unreachable!();
        };
        *signature = DomainSignature::sign(
            SignatureAuthority::NodeSession,
            issuer,
            key_id,
            signing_key,
            &signed_bytes,
        );
        Ok(frame)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn verify_outbound_confirmation(
        &self,
        expected_session_id: &str,
        expected_issuer: &str,
        expected_public_key: Digest32,
        now_ms: u64,
        maximum_clock_skew_ms: u64,
    ) -> Result<(u64, Digest32), ProtocolError> {
        let Self::ConfirmOutbound {
            session_id,
            outbound_sequence,
            frame_digest,
            at_ms,
            signature,
        } = self
        else {
            return Err(ProtocolError::MalformedFrame);
        };
        if !valid_identifier(expected_session_id)
            || !valid_identifier(expected_issuer)
            || expected_public_key == [0; 32]
            || maximum_clock_skew_ms == 0
            || session_id != expected_session_id
            || *outbound_sequence == 0
            || *frame_digest == [0; 32]
            || *at_ms == 0
            || at_ms.abs_diff(now_ms) > maximum_clock_skew_ms
        {
            return Err(ProtocolError::InvalidOutbound);
        }
        signature
            .verify_exact(
                SignatureAuthority::NodeSession,
                expected_issuer,
                expected_public_key,
                &self.signed_control_bytes()?,
            )
            .map_err(|_| ProtocolError::InvalidOutbound)?;
        Ok((*outbound_sequence, *frame_digest))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confirmation_binds_session_sequence_digest_domain_and_time() {
        let key = SigningKey::from_bytes(&[47; 32]);
        let frame = NodeControlFrame::outbound_confirmation(
            "session-1".to_owned(),
            7,
            [9; 32],
            50_000,
            "hub-issuer".to_owned(),
            "hub-key".to_owned(),
            &key,
        )
        .unwrap();
        assert_eq!(
            frame
                .verify_outbound_confirmation(
                    "session-1",
                    "hub-issuer",
                    key.verifying_key().to_bytes(),
                    50_010,
                    100,
                )
                .unwrap(),
            (7, [9; 32])
        );

        for (session_id, now_ms, public_key) in [
            ("session-2", 50_010, key.verifying_key().to_bytes()),
            ("session-1", 49_899, key.verifying_key().to_bytes()),
            ("session-1", 50_101, key.verifying_key().to_bytes()),
            ("session-1", 50_010, [8; 32]),
        ] {
            assert!(frame
                .verify_outbound_confirmation(session_id, "hub-issuer", public_key, now_ms, 100,)
                .is_err());
        }

        let mut wrong_sequence = frame.clone();
        let NodeControlFrame::ConfirmOutbound {
            outbound_sequence, ..
        } = &mut wrong_sequence
        else {
            unreachable!();
        };
        *outbound_sequence = 8;
        let mut wrong_digest = frame.clone();
        let NodeControlFrame::ConfirmOutbound { frame_digest, .. } = &mut wrong_digest else {
            unreachable!();
        };
        *frame_digest = [8; 32];
        for tampered in [wrong_sequence, wrong_digest] {
            assert!(tampered
                .verify_outbound_confirmation(
                    "session-1",
                    "hub-issuer",
                    key.verifying_key().to_bytes(),
                    50_010,
                    100,
                )
                .is_err());
        }

        let mut wrong_domain = frame;
        let canonical = wrong_domain.signed_control_bytes().unwrap();
        let NodeControlFrame::ConfirmOutbound { signature, .. } = &mut wrong_domain else {
            unreachable!();
        };
        *signature = DomainSignature::sign(
            SignatureAuthority::NodeSession,
            "hub-issuer".to_owned(),
            "hub-key".to_owned(),
            &key,
            &[
                b"wrong-domain\0".as_slice(),
                &canonical[OUTBOUND_CONFIRMATION_DOMAIN.len()..],
            ]
            .concat(),
        );
        assert!(wrong_domain
            .verify_outbound_confirmation(
                "session-1",
                "hub-issuer",
                key.verifying_key().to_bytes(),
                50_010,
                100,
            )
            .is_err());

        for (sequence, digest, at_ms) in [(0, [1; 32], 1), (1, [0; 32], 1), (1, [1; 32], 0)] {
            assert!(NodeControlFrame::outbound_confirmation(
                "session-1".to_owned(),
                sequence,
                digest,
                at_ms,
                "hub-issuer".to_owned(),
                "hub-key".to_owned(),
                &key,
            )
            .is_err());
        }
    }
}
