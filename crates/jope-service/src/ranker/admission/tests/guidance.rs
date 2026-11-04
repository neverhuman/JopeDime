use super::*;

#[test]
fn unqualified_auto_is_heuristic_with_catalogued_warning_and_real_reason() {
    for reason in [
        GuidanceUnavailableReason::Integrity,
        GuidanceUnavailableReason::Parity,
        GuidanceUnavailableReason::Quality,
        GuidanceUnavailableReason::Capability,
        GuidanceUnavailableReason::RevealedDecision,
    ] {
        let (plan, warnings) =
            resolve_unqualified(Guidance::Auto, LandmarkMode::Full, reason).unwrap();
        assert_eq!(plan.value, Guidance::Heuristic);
        assert!(!plan.prior_only);
        assert_eq!(plan.reason, reason.message());
        assert_eq!(
            warnings.len(),
            if reason == GuidanceUnavailableReason::Integrity {
                2
            } else {
                1
            }
        );
        assert_eq!(warnings[0].code, "GUIDANCE_HEURISTIC");
        assert_eq!(warnings[0].severity, Severity::Info);
        assert!(!warnings[0].host);
        assert_eq!(warnings[0].message, plan.reason);
        if reason == GuidanceUnavailableReason::Integrity {
            assert_eq!(warnings[1].code, "MODEL_INTEGRITY_FAILED");
            assert_eq!(warnings[1].severity, Severity::Error);
        }
    }
}

#[test]
fn learned_is_refused_and_progressive_remains_unavailable_for_every_request() {
    assert_eq!(
        resolve_unqualified(
            Guidance::Learned,
            LandmarkMode::Full,
            GuidanceUnavailableReason::Integrity
        )
        .unwrap_err(),
        ErrorCode::ModelIntegrityFailed
    );
    for reason in [
        GuidanceUnavailableReason::Parity,
        GuidanceUnavailableReason::Quality,
        GuidanceUnavailableReason::Capability,
        GuidanceUnavailableReason::RevealedDecision,
    ] {
        assert_eq!(
            resolve_unqualified(Guidance::Learned, LandmarkMode::Full, reason).unwrap_err(),
            ErrorCode::CapabilityUnavailable
        );
    }
    for requested in [Guidance::Auto, Guidance::Learned, Guidance::Heuristic] {
        assert_eq!(
            resolve_unqualified(
                requested,
                LandmarkMode::Progressive,
                GuidanceUnavailableReason::Integrity
            )
            .unwrap_err(),
            ErrorCode::CapabilityUnavailable
        );
    }
    let (plan, warnings) = resolve_unqualified(
        Guidance::Heuristic,
        LandmarkMode::None,
        GuidanceUnavailableReason::Integrity,
    )
    .unwrap();
    assert_eq!(plan.value, Guidance::Heuristic);
    assert!(!plan.prior_only);
    assert!(warnings.is_empty());
}

#[test]
fn auto_without_landmarks_stays_heuristic_for_every_unqualified_reason() {
    for reason in [
        GuidanceUnavailableReason::Integrity,
        GuidanceUnavailableReason::Parity,
        GuidanceUnavailableReason::Quality,
        GuidanceUnavailableReason::Capability,
        GuidanceUnavailableReason::RevealedDecision,
    ] {
        let (plan, warnings) =
            resolve_unqualified(Guidance::Auto, LandmarkMode::None, reason).unwrap();
        assert_eq!(plan.value, Guidance::Heuristic);
        assert!(!plan.prior_only);
        assert_eq!(plan.reason, reason.message());
        assert_eq!(warnings[0].code, "GUIDANCE_HEURISTIC");
        assert_eq!(warnings[0].message, plan.reason);
        assert_eq!(
            warnings.len(),
            if reason == GuidanceUnavailableReason::Integrity {
                2
            } else {
                1
            }
        );
    }
}

#[test]
fn explicit_heuristic_never_claims_learned_qualification_or_emits_fallback_warning() {
    for reason in [
        GuidanceUnavailableReason::Integrity,
        GuidanceUnavailableReason::Parity,
        GuidanceUnavailableReason::Quality,
        GuidanceUnavailableReason::Capability,
        GuidanceUnavailableReason::RevealedDecision,
    ] {
        for landmarks in [LandmarkMode::None, LandmarkMode::Full] {
            let (plan, warnings) =
                resolve_unqualified(Guidance::Heuristic, landmarks, reason).unwrap();
            assert_eq!(plan.value, Guidance::Heuristic);
            assert!(!plan.prior_only);
            assert_eq!(plan.reason, "Heuristic guidance was requested.");
            assert!(warnings.is_empty());
        }
    }
}
