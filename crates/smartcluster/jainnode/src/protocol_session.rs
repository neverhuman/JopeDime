use ed25519_dalek::SigningKey;
use jain_smartcluster_fabric::{
    valid_identifier, Digest32, DomainSignature, SignatureAuthority, WorkerProvision,
};

use super::{NodeControlFrame, NodeProfile, ProtocolError, SESSION_HELLO_DOMAIN};

impl NodeControlFrame {
    #[allow(clippy::too_many_arguments)]
    pub fn session_hello(
        node_id: String,
        session_id: String,
        profile: NodeProfile,
        provision: WorkerProvision,
        qualification_digest: Digest32,
        sequence: u64,
        at_ms: u64,
        key_id: String,
        signing_key: &SigningKey,
    ) -> Result<Self, ProtocolError> {
        if !valid_identifier(&node_id)
            || !valid_identifier(&session_id)
            || !valid_identifier(&key_id)
            || qualification_digest == [0; 32]
            || sequence == 0
            || at_ms == 0
        {
            return Err(ProtocolError::InvalidConfiguration);
        }
        let mut frame = Self::SessionHello {
            node_id: node_id.clone(),
            session_id,
            profile,
            provision,
            qualification_digest,
            sequence,
            at_ms,
            signature: DomainSignature {
                authority: SignatureAuthority::NodeSession,
                issuer: node_id.clone(),
                key_id: key_id.clone(),
                public_key: signing_key.verifying_key().to_bytes(),
                signature: Vec::new(),
            },
        };
        let signed_bytes = frame.signed_session_hello_bytes()?;
        let Self::SessionHello { signature, .. } = &mut frame else {
            unreachable!();
        };
        *signature = DomainSignature::sign(
            SignatureAuthority::NodeSession,
            node_id,
            key_id,
            signing_key,
            &signed_bytes,
        );
        Ok(frame)
    }

    pub fn signed_session_hello_bytes(&self) -> Result<Vec<u8>, ProtocolError> {
        let Self::SessionHello { .. } = self else {
            return Err(ProtocolError::MalformedFrame);
        };
        let mut unsigned = self.clone();
        let Self::SessionHello { signature, .. } = &mut unsigned else {
            unreachable!();
        };
        signature.issuer.clear();
        signature.key_id.clear();
        signature.public_key = [0; 32];
        signature.signature.clear();
        let payload = postcard::to_stdvec(&unsigned).map_err(|_| ProtocolError::MalformedFrame)?;
        let mut bytes = Vec::with_capacity(SESSION_HELLO_DOMAIN.len() + payload.len());
        bytes.extend_from_slice(SESSION_HELLO_DOMAIN);
        bytes.extend_from_slice(&payload);
        Ok(bytes)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn verify_session_hello(
        &self,
        expected_node_id: &str,
        expected_session_id: &str,
        expected_profile: NodeProfile,
        expected_provision: WorkerProvision,
        expected_qualification_digest: Digest32,
        expected_key_id: &str,
        expected_public_key: Digest32,
        now_ms: u64,
        maximum_clock_skew_ms: u64,
    ) -> Result<(), ProtocolError> {
        let Self::SessionHello {
            node_id,
            session_id,
            profile,
            provision,
            qualification_digest,
            sequence,
            at_ms,
            signature,
        } = self
        else {
            return Err(ProtocolError::MalformedFrame);
        };
        if !valid_identifier(expected_node_id)
            || !valid_identifier(expected_session_id)
            || !valid_identifier(expected_key_id)
            || expected_qualification_digest == [0; 32]
            || expected_public_key == [0; 32]
            || maximum_clock_skew_ms == 0
            || node_id != expected_node_id
            || session_id != expected_session_id
            || *profile != expected_profile
            || *provision != expected_provision
            || *qualification_digest != expected_qualification_digest
            || *sequence == 0
            || *at_ms == 0
            || at_ms.abs_diff(now_ms) > maximum_clock_skew_ms
            || signature.key_id != expected_key_id
        {
            return Err(ProtocolError::SessionRevoked);
        }
        signature
            .verify_exact(
                SignatureAuthority::NodeSession,
                expected_node_id,
                expected_public_key,
                &self.signed_session_hello_bytes()?,
            )
            .map_err(|_| ProtocolError::SessionRevoked)
    }
}
