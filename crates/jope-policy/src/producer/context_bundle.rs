use super::*;

/// Private context admission payload for the separate policy child. The parent
/// authenticates all original bounded receipts before pinning these normalized
/// attestations. This transport verifier alone is not process-execution proof.
#[cfg(feature = "context-producer")]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualifiedContextBundle {
    pub artifact: Vec<u8>,
    pub qualification: Vec<u8>,
    pub reference_manifest: Vec<u8>,
    pub context_execution: Vec<u8>,
    pub case_executions: Vec<Vec<u8>>,
    pub pins: crate::context_producer::AdmissionPins,
}
#[cfg(feature = "context-producer")]
impl QualifiedContextBundle {
    /// Verify the bundle against its carried pins. Those pins are only as trustworthy as the
    /// channel that delivered the bundle: a controller that authenticated them independently
    /// (for example `cpu_context_bridge::load_qualified`) may call this; a worker must use
    /// [`Self::admit_for_request`].
    pub fn verify(&self) -> Result<crate::context_producer::VerifiedCpuContext> {
        crate::context_producer::verify_qualified(
            &self.artifact,
            &self.qualification,
            &self.reference_manifest,
            &self.context_execution,
            &self.case_executions,
            &self.pins,
        )
    }

    /// Policy-worker admission of a qualified CPU context. The bundle's pins must be exactly the
    /// admission the controller issued in the authenticated request
    /// ([`Request::context_admission_sha256`]); only then is the bundle verified against them.
    /// The final context must then be the frozen assembly of the qualified components under the
    /// accepted transforms, bit-identical to the actual policy input, and deliverable to the
    /// accepted policy for this request's scope.
    pub fn admit_for_request(
        &self,
        request: &Request,
        transforms: &crate::context_producer::FrozenTransforms,
        policy_input: &[u8],
    ) -> Result<crate::context_producer::ContextHandoff> {
        request.validate()?;
        let issued = request.context_admission_sha256.as_deref().ok_or_else(|| {
            anyhow::anyhow!("request carries no controller-issued context admission pin")
        })?;
        ensure!(
            self.pins.sha256()? == issued,
            "context bundle pins are not the controller-issued admission of this request"
        );
        let verified = self.verify()?;
        let context = verified.replay_assembly(transforms)?;
        ensure!(
            context.to_le_bytes() == policy_input,
            "qualified context differs from the actual policy input"
        );
        let handoff = verified.handoff(
            &context,
            crate::context_producer::ContextConsumer::AcceptedPolicyProposal,
        )?;
        handoff.deliver_to_policy(request)?;
        Ok(handoff)
    }
}

#[cfg(all(test, feature = "context-producer"))]
mod bundle_tests {
    use super::*;
    use crate::context_producer::*;
    fn forged_bundle() -> QualifiedContextBundle {
        let artifact = br#"{"not":"checked yet"}"#.to_vec();
        let pins = AdmissionPins {
            source_sha256: "e".repeat(40),
            producer_sha256: "f".repeat(64),
            artifact_sha256: sha256_hex(&artifact),
            qualification_sha256: sha256_hex(b"q"),
            reference_manifest_sha256: sha256_hex(b"m"),
            context_execution_sha256: sha256_hex(b"x"),
            qualification_execution_sha256: vec![sha256_hex(b"c"); 8],
            verification_policy_sha256: verification_policy_sha256(),
            training: TrainingIdentity {
                dataset_sha256: "a".repeat(64),
                prepared_sha256: "b".repeat(64),
                training_sha256: "c".repeat(64),
                fitting_rows_sha256: "d".repeat(64),
                cut: 0,
            },
            input_sha256: "1".repeat(64),
            preprocessing_sha256: "2".repeat(64),
        };
        QualifiedContextBundle {
            artifact,
            qualification: b"q".to_vec(),
            reference_manifest: b"m".to_vec(),
            context_execution: b"x".to_vec(),
            case_executions: vec![b"c".to_vec(); 8],
            pins,
        }
    }
    fn refusal(
        bundle: &QualifiedContextBundle,
        request: &Request,
        transforms: &FrozenTransforms,
    ) -> String {
        match bundle.admit_for_request(request, transforms, &[0; 8192]) {
            Ok(_) => panic!("forged bundle admitted"),
            Err(error) => format!("{error:#}"),
        }
    }

    /// W2-T02 G-03: the worker verifies a bundle against the controller-issued admission in the
    /// request, never against the pins the bundle carries. Pre-fix reproduction (retained in the
    /// w2b-repro-prefix evidence): a bundle that chose its own pins passed every pin gate of
    /// `verify` and was refused only later, while parsing its artifact.
    #[test]
    fn worker_admission_requires_the_request_issued_pins() {
        let transforms = FrozenTransforms::synthetic_for_testing(1.0);
        let bundle = forged_bundle();
        let request = super::super::tests::request();
        // No admission issued: refused before any verification.
        let error = refusal(&bundle, &request, &transforms);
        assert!(
            error.contains("no controller-issued context admission"),
            "{error}"
        );
        // The controller issued other pins: refused, whatever the bundle claims.
        let mut other = bundle.pins.clone();
        other.input_sha256 = "9".repeat(64);
        let issued_other = Request {
            context_admission_sha256: Some(other.sha256().unwrap()),
            ..request.clone()
        };
        let error = refusal(&bundle, &issued_other, &transforms);
        assert!(
            error.contains("not the controller-issued admission"),
            "{error}"
        );
        // Pins changed after issuance (for example to name a substituted artifact): refused.
        let issued = Request {
            context_admission_sha256: Some(bundle.pins.sha256().unwrap()),
            ..request.clone()
        };
        let mut substituted = bundle.clone();
        substituted.artifact = b"{}".to_vec();
        substituted.pins.artifact_sha256 = sha256_hex(&substituted.artifact);
        let error = refusal(&substituted, &issued, &transforms);
        assert!(
            error.contains("not the controller-issued admission"),
            "{error}"
        );
        // Positive control for the gate: with the issued pins the bundle passes the pin gate and
        // full verification runs (this synthetic bundle then fails on its artifact content, so
        // nothing is admitted).
        let error = refusal(&bundle, &issued, &transforms);
        assert!(
            !error.contains("controller-issued") && error.contains("unknown field"),
            "{error}"
        );
    }
}
