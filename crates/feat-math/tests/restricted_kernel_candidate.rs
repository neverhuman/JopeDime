#[allow(dead_code)]
#[path = "../src/kernel/restricted_candidate.rs"]
mod restricted_candidate;

use restricted_candidate::{
    artifact_digest, source_digest, CandidateArtifact, ForbiddenOperation, FuelBound, Instruction,
    KernelError, RestrictedKernel, SandboxCapability,
};

fn bounds() -> FuelBound {
    FuelBound {
        max_program_instructions: 16,
        max_source_bytes: 64,
        max_input_width: 4,
        max_rows: 8,
        max_stack: 8,
        max_fuel: 8 * (16 + 2),
    }
}

fn program() -> Vec<Instruction> {
    vec![
        Instruction::LoadFeature(0),
        Instruction::Constant(2.0),
        Instruction::Multiply,
        Instruction::Constant(1.0),
        Instruction::Add,
    ]
}

fn make_candidate(instructions: Vec<Instruction>) -> CandidateArtifact {
    CandidateArtifact {
        source: b"candidate-source".to_vec(),
        toolchain_id: "rustc-pinned".into(),
        target_id: "x86_64-unknown-linux-gnu".into(),
        instructions,
        self_issued_certificate: None,
    }
}

fn capability(candidate: &CandidateArtifact) -> SandboxCapability {
    SandboxCapability::new(
        "rustc-pinned",
        "x86_64-unknown-linux-gnu",
        source_digest(&candidate.source),
        artifact_digest(&candidate.instructions),
        bounds(),
    )
}

#[test]
fn bounded_arithmetic_scan_and_fold_are_deterministic() {
    let candidate = make_candidate(program());
    let capability = capability(&candidate);
    let kernel = RestrictedKernel::admit(candidate, &capability).unwrap();
    let rows = vec![vec![1.0], vec![2.0], vec![3.0]];
    assert_eq!(kernel.evaluate_rows(&rows).unwrap(), vec![3.0, 5.0, 7.0]);
    assert_eq!(kernel.evaluate_rows(&rows).unwrap(), vec![3.0, 5.0, 7.0]);
    assert_eq!(kernel.evaluate_sum(&rows).unwrap(), 15.0);
}

#[test]
fn caller_pinned_toolchain_target_source_and_artifact_are_required() {
    let source = make_candidate(program());
    let cap = capability(&source);
    let mut wrong = make_candidate(program());
    wrong.toolchain_id = "other".into();
    assert_eq!(
        RestrictedKernel::admit(wrong, &cap).unwrap_err(),
        KernelError::WrongToolchain
    );
    let mut wrong = make_candidate(program());
    wrong.target_id = "other-target".into();
    assert_eq!(
        RestrictedKernel::admit(wrong, &cap).unwrap_err(),
        KernelError::WrongTarget
    );
    let mut wrong = make_candidate(program());
    wrong.source.push(b'!');
    assert_eq!(
        RestrictedKernel::admit(wrong, &cap).unwrap_err(),
        KernelError::WrongSourceDigest
    );
    let wrong_artifact = SandboxCapability::new(
        "rustc-pinned",
        "x86_64-unknown-linux-gnu",
        source_digest(&source.source),
        [0; 32],
        bounds(),
    );
    assert_eq!(
        RestrictedKernel::admit(source, &wrong_artifact).unwrap_err(),
        KernelError::WrongArtifactDigest
    );
}

#[test]
fn unpinned_identity_and_self_issued_certificate_are_refused() {
    let candidate = make_candidate(program());
    let cap = SandboxCapability::new(
        "",
        "target",
        source_digest(&candidate.source),
        artifact_digest(&candidate.instructions),
        bounds(),
    );
    assert_eq!(
        RestrictedKernel::admit(candidate, &cap).unwrap_err(),
        KernelError::InvalidBinding
    );
    let mut candidate = make_candidate(program());
    candidate.self_issued_certificate = Some(vec![1]);
    let cap = capability(&candidate);
    assert_eq!(
        RestrictedKernel::admit(candidate, &cap).unwrap_err(),
        KernelError::SelfIssuedCertificate
    );
}

#[test]
fn external_state_escape_and_scientific_authority_operations_are_rejected() {
    let forbidden = [
        ForbiddenOperation::FileSystem,
        ForbiddenOperation::Network,
        ForbiddenOperation::Process,
        ForbiddenOperation::Clock,
        ForbiddenOperation::Randomness,
        ForbiddenOperation::Environment,
        ForbiddenOperation::HiddenMutableState,
        ForbiddenOperation::Recursion,
        ForbiddenOperation::Jump,
        ForbiddenOperation::UnboundedLoop,
        ForbiddenOperation::ReadOuterOrSealedSplit,
        ForbiddenOperation::ModifyEvaluator,
        ForbiddenOperation::ModifyTrustedKernel,
        ForbiddenOperation::IssueCertificate,
        ForbiddenOperation::PromoteCandidate,
    ];
    for operation in forbidden {
        let candidate = make_candidate(vec![
            Instruction::Constant(1.0),
            Instruction::Forbidden(operation),
        ]);
        let cap = capability(&candidate);
        assert_eq!(
            RestrictedKernel::admit(candidate, &cap).unwrap_err(),
            KernelError::ForbiddenInstruction(operation)
        );
    }
}

#[test]
fn resource_exhaustion_and_invalid_programs_fail_closed() {
    let candidate = make_candidate(program());
    let mut cap = capability(&candidate);
    cap.fuel.max_rows = 9;
    assert_eq!(
        RestrictedKernel::admit(candidate, &cap).unwrap_err(),
        KernelError::InvalidFuelBound
    );
    let candidate = make_candidate(program());
    let kernel = RestrictedKernel::admit(candidate.clone(), &capability(&candidate)).unwrap();
    assert_eq!(
        kernel.evaluate_rows(&vec![vec![1.0]; 9]).unwrap_err(),
        KernelError::TooManyRows
    );
    assert_eq!(
        kernel.evaluate_rows(&[vec![]]).unwrap_err(),
        KernelError::InvalidInputShape
    );
    let mut large = make_candidate(program());
    large.source = vec![0; 65];
    let cap = SandboxCapability::new(
        "rustc-pinned",
        "x86_64-unknown-linux-gnu",
        source_digest(&large.source),
        artifact_digest(&large.instructions),
        bounds(),
    );
    assert_eq!(
        RestrictedKernel::admit(large, &cap).unwrap_err(),
        KernelError::SourceTooLarge
    );
    let candidate = make_candidate(vec![
        Instruction::Constant(1.0),
        Instruction::Constant(0.0),
        Instruction::Divide,
    ]);
    let kernel = RestrictedKernel::admit(candidate.clone(), &capability(&candidate)).unwrap();
    assert_eq!(
        kernel.evaluate_rows(&[vec![]]).unwrap_err(),
        KernelError::DivisionByZero
    );
    let candidate = make_candidate(vec![
        Instruction::Constant(f64::MAX),
        Instruction::Constant(2.0),
        Instruction::Multiply,
    ]);
    let kernel = RestrictedKernel::admit(candidate.clone(), &capability(&candidate)).unwrap();
    assert_eq!(
        kernel.evaluate_rows(&[vec![]]).unwrap_err(),
        KernelError::NonFiniteValue
    );
}
