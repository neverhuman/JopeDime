use anyhow::{bail, ensure, Result};
use jope_learning::universe::{
    CellStatus, ImmutableSourceView, ObservedOutcome, OutcomeKey, OutcomeStatus, Universe,
    UniverseEvidenceVerifier, SOURCE_VIEW_SCHEMA,
};
use jope_records::wire_contract::{
    ExactF64, MeasuredEvidenceKind, MeasuredEvidenceProof, ParsedMeasuredClaim,
    VerifiedMeasuredOutcome,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    requested: Vec<OutcomeKey>,
    observations: Vec<ObservedOutcome>,
}

struct FixtureVerifier;

impl UniverseEvidenceVerifier for FixtureVerifier {
    fn verify_view(&self, view: &ImmutableSourceView) -> Result<()> {
        ensure!(view.source_receipt_sha256 != "0".repeat(64));
        ensure!(view.revocation_snapshot_sha256 != "0".repeat(64));
        Ok(())
    }

    fn verify_observation(
        &self,
        view: &ImmutableSourceView,
        observation: &ObservedOutcome,
    ) -> Result<Option<VerifiedMeasuredOutcome>> {
        ensure!(
            view.observation_locators_sha256
                .contains(&observation.source_locator_sha256),
            "observation is not in the verified view"
        );
        if let Some(bits) = &observation.metric_f64_bits {
            let claim = ParsedMeasuredClaim {
                metric_name: observation.key.scope.metric_name.clone(),
                metric: ExactF64::from_hex("/metric_f64_bits", bits)?,
                evidence_receipt_sha256: observation.evidence_receipt_sha256.clone(),
                evidence_kind: MeasuredEvidenceKind::MeasuredCpuCv,
            };
            let proof = MeasuredEvidenceProof {
                receipt_sha256: observation.evidence_receipt_sha256.clone(),
                evidence_kind: MeasuredEvidenceKind::MeasuredCpuCv,
                independent_verifier_passed: true,
            };
            return Ok(Some(VerifiedMeasuredOutcome::verify_from_claim(
                &claim, &proof,
            )?));
        }
        Ok(None)
    }
}

struct RevokedVerifier;

impl UniverseEvidenceVerifier for RevokedVerifier {
    fn verify_view(&self, _view: &ImmutableSourceView) -> Result<()> {
        bail!("source view is revoked")
    }

    fn verify_observation(
        &self,
        _view: &ImmutableSourceView,
        _observation: &ObservedOutcome,
    ) -> Result<Option<VerifiedMeasuredOutcome>> {
        bail!("must not inspect observations after revoked view")
    }
}

struct NoMeasuredProof;

impl UniverseEvidenceVerifier for NoMeasuredProof {
    fn verify_view(&self, _view: &ImmutableSourceView) -> Result<()> {
        Ok(())
    }

    fn verify_observation(
        &self,
        _view: &ImmutableSourceView,
        observation: &ObservedOutcome,
    ) -> Result<Option<VerifiedMeasuredOutcome>> {
        ensure!(observation.status != OutcomeStatus::Complete);
        Ok(None)
    }
}

fn fixture() -> Fixture {
    serde_json::from_str(include_str!("fixtures/measured-universe-v1/table.json")).unwrap()
}

fn source_view(observations: &[ObservedOutcome]) -> ImmutableSourceView {
    let mut locators: Vec<_> = observations
        .iter()
        .map(|row| row.source_locator_sha256.clone())
        .collect();
    locators.sort();
    locators.dedup();
    let mut view = ImmutableSourceView {
        schema: SOURCE_VIEW_SCHEMA.into(),
        source_commit: "1".repeat(40),
        source_receipt_sha256: "8".repeat(64),
        revocation_snapshot_sha256: "9".repeat(64),
        observation_locators_sha256: locators,
        view_sha256: String::new(),
    };
    view.seal().unwrap();
    view
}

fn build(f: &Fixture) -> Result<Universe> {
    let view = source_view(&f.observations);
    Universe::build(
        &view,
        f.requested.clone(),
        f.observations.clone(),
        &FixtureVerifier,
    )
}

fn complete_row(key: OutcomeKey, bits: &str, receipt: char, locator: char) -> ObservedOutcome {
    ObservedOutcome {
        key,
        status: OutcomeStatus::Complete,
        metric_f64_bits: Some(bits.into()),
        evidence_receipt_sha256: receipt.to_string().repeat(64),
        source_locator_sha256: locator.to_string().repeat(64),
    }
}

#[test]
fn fixture_has_exact_hand_checked_values_and_finite_support() -> Result<()> {
    let f = fixture();
    ensure!(f.requested.len() == 5 && f.observations.len() == 4);
    let universe = build(&f)?;
    let mut values: Vec<_> = universe.measured().map(|(_, bits)| bits).collect();
    values.sort_unstable();
    ensure!(values == ["3fd0000000000000", "3fe0000000000000", "3fe8000000000000"]);
    ensure!(universe.outcomes.len() == 5);
    ensure!(universe
        .outcomes
        .iter()
        .all(|cell| { cell.status != CellStatus::Complete || cell.metric_f64_bits.is_some() }));
    Ok(())
}

#[test]
fn mismatched_evaluator_or_role_cannot_join() -> Result<()> {
    let mut f = fixture();
    let requested = f.requested[0].clone();
    let mut wrong_evaluator = f.observations[0].clone();
    wrong_evaluator.key.scope.evaluator_sha256 = "a".repeat(64);
    wrong_evaluator.source_locator_sha256 = "1".repeat(64);
    let mut wrong_role = f.observations[0].clone();
    wrong_role.key.scope.role = "validation".into();
    wrong_role.source_locator_sha256 = "2".repeat(64);
    f.observations = vec![wrong_evaluator, wrong_role];
    let view = source_view(&f.observations);
    let universe = Universe::build(&view, vec![requested], f.observations, &FixtureVerifier)?;
    ensure!(universe.outcomes[0].status == CellStatus::Missing);
    ensure!(universe.join_refusals.len() == 2);
    ensure!(universe.join_refusals[0].mismatched_fields == ["evaluator_sha256"]);
    ensure!(universe.join_refusals[1].mismatched_fields == ["role"]);
    Ok(())
}

#[test]
fn equal_key_conflicting_values_are_quarantined_independent_of_arrival_order() -> Result<()> {
    let f = fixture();
    let mut rows = vec![f.observations[0].clone(), f.observations[0].clone()];
    rows[1].metric_f64_bits = Some("3fd0000000000000".into());
    rows[1].source_locator_sha256 = "1".repeat(64);
    let view = source_view(&rows);
    let requested = vec![f.requested[0].clone()];
    let first = Universe::build(&view, requested.clone(), rows.clone(), &FixtureVerifier)?;
    rows.reverse();
    let second = Universe::build(&view, requested, rows, &FixtureVerifier)?;
    ensure!(first.outcomes[0].status == CellStatus::Conflicting);
    ensure!(first.outcomes[0].metric_f64_bits.is_none());
    ensure!(first.outcomes[0].observations.len() == 2);
    ensure!(first.digest()? == second.digest()?);
    Ok(())
}

#[test]
fn compatible_backfill_preserves_each_immutable_source_locator() -> Result<()> {
    let f = fixture();
    let mut duplicate = f.observations[0].clone();
    duplicate.source_locator_sha256 = "1".repeat(64);
    let rows = vec![f.observations[0].clone(), duplicate];
    let view = source_view(&rows);
    let universe = Universe::build(&view, vec![f.requested[0].clone()], rows, &FixtureVerifier)?;
    let cell = &universe.outcomes[0];
    ensure!(cell.status == CellStatus::Complete && cell.observations.len() == 2);
    ensure!(
        cell.observations[0].source_locator_sha256 != cell.observations[1].source_locator_sha256
    );
    Ok(())
}

#[test]
fn distinct_declared_replicates_remain_separate_keys() -> Result<()> {
    let mut f = fixture();
    let mut first = f.requested[0].clone();
    first.replicate_id = Some("replicate-1".into());
    let mut second = first.clone();
    second.replicate_id = Some("replicate-2".into());
    let rows = vec![
        complete_row(first.clone(), "3fe8000000000000", '8', '1'),
        complete_row(second.clone(), "3fd0000000000000", '7', '2'),
    ];
    let view = source_view(&rows);
    f.requested = vec![first, second];
    let universe = Universe::build(&view, f.requested, rows, &FixtureVerifier)?;
    ensure!(universe.outcomes.len() == 2);
    ensure!(universe
        .outcomes
        .iter()
        .all(|cell| cell.status == CellStatus::Complete));
    Ok(())
}

#[test]
fn unavailable_outcomes_keep_status_and_requested_denominator() -> Result<()> {
    let base = fixture().requested[0].clone();
    let mut requested = Vec::new();
    let mut rows = Vec::new();
    for (idx, status) in [
        OutcomeStatus::Failed,
        OutcomeStatus::Timeout,
        OutcomeStatus::Rejected,
    ]
    .into_iter()
    .enumerate()
    {
        let mut key = base.clone();
        key.program_sha256 = format!("{:064x}", idx + 20);
        requested.push(key.clone());
        rows.push(ObservedOutcome {
            key,
            status,
            metric_f64_bits: None,
            evidence_receipt_sha256: format!("{:064x}", idx + 30),
            source_locator_sha256: format!("{:064x}", idx + 40),
        });
    }
    let view = source_view(&rows);
    let universe = Universe::build(&view, requested, rows, &FixtureVerifier)?;
    let coverage = &universe.coverage[0];
    ensure!(coverage.support_cardinality == 3 && coverage.observed == 3);
    ensure!(coverage.failed == 1 && coverage.censored == 1 && coverage.rejected == 1);
    ensure!(coverage.complete_measured == 0 && !coverage.full_support);
    ensure!(universe.measured().next().is_none());
    Ok(())
}

#[test]
fn full_and_partial_support_are_reported_per_context_and_checkpoint() -> Result<()> {
    let f = fixture();
    let universe = build(&f)?;
    ensure!(universe.coverage.len() == 3);
    let a_c = universe
        .coverage
        .iter()
        .find(|c| c.context_sha256 == "a".repeat(64) && c.checkpoint_sha256 == "c".repeat(64))
        .unwrap();
    ensure!(a_c.support_cardinality == 2 && a_c.complete_measured == 2 && a_c.full_support);
    let a_d = universe
        .coverage
        .iter()
        .find(|c| c.context_sha256 == "a".repeat(64) && c.checkpoint_sha256 == "d".repeat(64))
        .unwrap();
    ensure!(a_d.support_cardinality == 1 && a_d.full_support);
    let b_c = universe
        .coverage
        .iter()
        .find(|c| c.context_sha256 == "b".repeat(64) && c.checkpoint_sha256 == "c".repeat(64))
        .unwrap();
    ensure!(b_c.support_cardinality == 2 && b_c.unknown == 1 && b_c.missing == 1);
    ensure!(!b_c.full_support);
    Ok(())
}

#[test]
fn probability_mass_remains_unknown_without_adopted_frozen_law() -> Result<()> {
    let universe = build(&fixture())?;
    ensure!(universe.coverage.iter().all(|report| {
        report.probability_mass.is_none()
            && report
                .probability_mass_reason
                .contains("no adopted typed frozen-law DTO")
    }));
    Ok(())
}

#[test]
fn revoked_source_view_fails_before_observations_are_examined() -> Result<()> {
    let f = fixture();
    let view = source_view(&f.observations);
    let result = Universe::build(&view, f.requested, f.observations, &RevokedVerifier);
    ensure!(result.is_err());
    Ok(())
}

#[test]
fn complete_value_without_sealed_i02_authority_is_refused() -> Result<()> {
    let f = fixture();
    let view = source_view(&f.observations);
    ensure!(Universe::build(&view, f.requested, f.observations, &NoMeasuredProof).is_err());
    Ok(())
}

#[test]
fn observation_outside_authenticated_view_is_refused() -> Result<()> {
    let f = fixture();
    let mut row = f.observations[0].clone();
    row.source_locator_sha256 = "1".repeat(64);
    let view = source_view(&[]);
    ensure!(Universe::build(
        &view,
        vec![f.requested[0].clone()],
        vec![row],
        &FixtureVerifier
    )
    .is_err());
    Ok(())
}

#[test]
fn nonfinite_exact_bits_are_refused() -> Result<()> {
    let f = fixture();
    let mut row = f.observations[0].clone();
    row.metric_f64_bits = Some("7ff0000000000000".into());
    let view = source_view(&[row.clone()]);
    ensure!(Universe::build(
        &view,
        vec![f.requested[0].clone()],
        vec![row],
        &FixtureVerifier
    )
    .is_err());
    Ok(())
}
