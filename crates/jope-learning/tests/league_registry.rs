#[path = "../src/selfplay/league.rs"]
mod league;

use anyhow::Result;
use league::{
    ContextIdentity, HistoricalCost, LeagueGate, LeagueGateReason, LeagueMember, LeagueRegistry,
    MatchmakingMemory, MemberKind, PayoffCell, PayoffManifest, PayoffPurpose, Pfsp,
};

const H0: &str = "0000000000000000000000000000000000000000000000000000000000000000";
const H1: &str = "1111111111111111111111111111111111111111111111111111111111111111";
const H2: &str = "2222222222222222222222222222222222222222222222222222222222222222";
const H3: &str = "3333333333333333333333333333333333333333333333333333333333333333";
const H4: &str = "4444444444444444444444444444444444444444444444444444444444444444";
const H5: &str = "5555555555555555555555555555555555555555555555555555555555555555";
const H6: &str = "6666666666666666666666666666666666666666666666666666666666666666";
const H7: &str = "7777777777777777777777777777777777777777777777777777777777777777";

fn member(id: &str, kind: MemberKind, ordinal: usize) -> LeagueMember {
    let hashes = [H0, H1, H2, H3, H4, H5, H6, H7];
    let acceptance_receipt_sha256 = (kind == MemberKind::AcceptedCompetitor)
        .then(|| hashes[(ordinal + 4) % hashes.len()].to_owned());
    LeagueMember {
        member_id: id.into(),
        kind,
        policy_sha256: hashes[ordinal % hashes.len()].into(),
        source_manifest_sha256: H7.into(),
        protocol_sha256: H1.into(),
        support_sha256: H2.into(),
        task_distribution_sha256: H3.into(),
        historical_cost: HistoricalCost {
            evidence_sha256: H4.into(),
            cpu_seconds: 120.0 + ordinal as f64,
            wall_seconds: 30.0 + ordinal as f64,
        },
        acceptance_receipt_sha256,
    }
}

fn registry(competitors: usize) -> Result<LeagueRegistry> {
    let mut r = LeagueRegistry::new();
    r.register(member("frozen-parent", MemberKind::FrozenParent, 0))?;
    r.register(member("simple-control", MemberKind::SimpleControl, 1))?;
    for i in 0..competitors {
        r.register(member(
            &format!("accepted-{i}"),
            MemberKind::AcceptedCompetitor,
            i + 2,
        ))?;
    }
    Ok(r)
}

fn contexts() -> Vec<ContextIdentity> {
    vec![
        ContextIdentity {
            context_id: "ctx-a".into(),
            family_id: "family-a".into(),
        },
        ContextIdentity {
            context_id: "ctx-b".into(),
            family_id: "family-b".into(),
        },
    ]
}

fn ids(r: &LeagueRegistry) -> Vec<String> {
    r.members().map(|m| m.member_id.clone()).collect()
}

fn cells(r: &LeagueRegistry, ctx: &[ContextIdentity], mode: &str) -> Vec<PayoffCell> {
    let members: Vec<_> = r.members().collect();
    let mut result = Vec::new();
    for i in 0..members.len() {
        for j in (i + 1)..members.len() {
            for context in ctx {
                let (left_payoff, right_payoff) = match mode {
                    "cycle" => {
                        let l = members[i].member_id.as_str();
                        let rr = members[j].member_id.as_str();
                        match (l, rr) {
                            ("accepted-0", "accepted-1") => (2.0, 1.0),
                            ("accepted-0", "accepted-2") => (1.0, 2.0),
                            ("accepted-1", "accepted-2") => (2.0, 1.0),
                            _ => (1.0, 1.0),
                        }
                    }
                    "specialization" if context.family_id == "family-a" => {
                        if members[i].member_id == "accepted-0" {
                            (2.0, 1.0)
                        } else if members[j].member_id == "accepted-0" {
                            (1.0, 2.0)
                        } else {
                            (1.0, 1.0)
                        }
                    }
                    "specialization" => {
                        if members[i].member_id == "accepted-0" {
                            (1.0, 2.0)
                        } else if members[j].member_id == "accepted-0" {
                            (2.0, 1.0)
                        } else {
                            (1.0, 1.0)
                        }
                    }
                    _ => ((members.len() - i) as f64, (members.len() - j) as f64),
                };
                result.push(PayoffCell {
                    left_member_id: members[i].member_id.clone(),
                    right_member_id: members[j].member_id.clone(),
                    context_id: context.context_id.clone(),
                    protocol_sha256: H1.into(),
                    support_sha256: H2.into(),
                    budget_sha256: H5.into(),
                    evaluator_sha256: H6.into(),
                    left_payoff,
                    right_payoff,
                    left_cpu_seconds: 9.0 + i as f64,
                    right_cpu_seconds: 8.0 + j as f64,
                    evidence_sha256: H7.into(),
                });
            }
        }
    }
    result
}

fn manifest(r: &LeagueRegistry, purpose: PayoffPurpose, mode: &str) -> Result<PayoffManifest> {
    let ctx = contexts();
    PayoffManifest::freeze(r, purpose, ctx.clone(), &ids(r), cells(r, &ctx, mode))
}

#[test]
fn baseline_registry_freezes_parent_controls_identity_and_cost() -> Result<()> {
    let mut r = registry(0)?;
    r.validate_frozen_baseline()?;
    let original = r.member("frozen-parent").unwrap().clone();
    assert!(
        !r.register(original.clone())?,
        "exact re-registration is idempotent"
    );
    let table = manifest(&r, PayoffPurpose::FrozenBaseline, "transitive")?;
    assert_eq!(
        table.cells.len(),
        2,
        "each of two members must cover both contexts"
    );
    assert_eq!(
        table
            .members
            .iter()
            .find(|m| m.member_id == "frozen-parent")
            .unwrap()
            .historical_cost,
        original.historical_cost
    );
    assert_eq!(table.identity_sha256(&r)?, table.identity_sha256(&r)?);

    let mut changed_cost = original;
    changed_cost.historical_cost.cpu_seconds += 1.0;
    assert!(
        r.register(changed_cost).is_err(),
        "member cost cannot be rewritten under historical ID"
    );
    Ok(())
}

#[test]
fn restored_registry_cannot_rebind_map_key_to_member_identity() -> Result<()> {
    let r = registry(0)?;
    let mut encoded = serde_json::to_value(&r)?;
    encoded["members"]["frozen-parent"]["member_id"] =
        serde_json::Value::String("renamed-parent".into());
    let restored: LeagueRegistry = serde_json::from_value(encoded)?;
    assert!(restored.validate_frozen_baseline().is_err());
    Ok(())
}

#[test]
fn frozen_baseline_payoff_table_must_include_parent_and_simple_control() -> Result<()> {
    let r = registry(1)?;
    let context_roster = contexts();
    let selected = vec!["frozen-parent".to_string(), "accepted-0".to_string()];
    let selected_set: std::collections::BTreeSet<_> = selected.iter().map(String::as_str).collect();
    let selected_cells: Vec<_> = cells(&r, &context_roster, "transitive")
        .into_iter()
        .filter(|cell| {
            selected_set.contains(cell.left_member_id.as_str())
                && selected_set.contains(cell.right_member_id.as_str())
        })
        .collect();
    assert!(PayoffManifest::freeze(
        &r,
        PayoffPurpose::FrozenBaseline,
        context_roster,
        &selected,
        selected_cells
    )
    .is_err());
    Ok(())
}

#[test]
fn partial_coverage_and_changed_protocol_are_rejected() -> Result<()> {
    let r = registry(0)?;
    let complete = manifest(&r, PayoffPurpose::FrozenBaseline, "transitive")?;
    let mut partial = complete.clone();
    partial.cells.pop();
    assert!(
        partial.validate(&r).is_err(),
        "missing context pair must not join payoff table"
    );
    let mut changed_protocol = complete;
    changed_protocol.cells[0].protocol_sha256 = H0.into();
    assert!(
        changed_protocol.validate(&r).is_err(),
        "mixed protocol must not join payoff table"
    );
    Ok(())
}

#[test]
fn member_identity_and_historical_cost_are_bound_into_frozen_table() -> Result<()> {
    let r = registry(0)?;
    let mut changed = manifest(&r, PayoffPurpose::FrozenBaseline, "transitive")?;
    changed.members[0].historical_cost.wall_seconds += 99.0;
    assert!(changed.validate(&r).is_err());
    Ok(())
}

#[test]
fn qualification_manifest_cannot_enter_training_matchmaking_memory() -> Result<()> {
    let r = registry(1)?;
    let qual = manifest(&r, PayoffPurpose::Qualification, "transitive")?;
    let mut memory = MatchmakingMemory::default();
    assert!(memory.absorb(&qual, &r).is_err());
    assert!(
        memory.is_empty(),
        "rejected qualification must not append partial rows"
    );
    let train = manifest(&r, PayoffPurpose::TrainingMatchmaking, "transitive")?;
    memory.absorb(&train, &r)?;
    assert_eq!(memory.len(), train.cells.len());
    assert!(memory
        .rows()
        .iter()
        .all(|row| row.manifest_sha256 == train.identity_sha256(&r).unwrap()));
    Ok(())
}

#[test]
fn member_counts_alone_never_activate_pf_scheduling() -> Result<()> {
    for count in [2, 3, 5] {
        let r = registry(count)?;
        let table = manifest(&r, PayoffPurpose::TrainingMatchmaking, "transitive")?;
        assert_eq!(
            Pfsp::assess(&table, &r)?,
            LeagueGate::Deferred {
                reason: LeagueGateReason::NoMeasuredCycleOrSpecialization
            },
            "{count} members without a measured cycle/specialization stay deferred"
        );
    }
    let r = registry(0)?;
    let table = manifest(&r, PayoffPurpose::TrainingMatchmaking, "transitive")?;
    assert_eq!(
        Pfsp::assess(&table, &r)?,
        LeagueGate::Deferred {
            reason: LeagueGateReason::NoAcceptedCompetitors
        }
    );
    Ok(())
}

#[test]
fn measured_cycle_or_family_specialization_only_yields_separate_review_candidate() -> Result<()> {
    let r = registry(3)?;
    let table = manifest(&r, PayoffPurpose::TrainingMatchmaking, "cycle")?;
    assert_eq!(
        Pfsp::assess(&table, &r)?,
        LeagueGate::CandidateForSeparateReview {
            evidence_sha256: table.identity_sha256(&r)?,
            basis: "measured_nontransitive_cycle".into()
        }
    );

    let r = registry(2)?;
    let table = manifest(&r, PayoffPurpose::TrainingMatchmaking, "specialization")?;
    assert_eq!(
        Pfsp::assess(&table, &r)?,
        LeagueGate::CandidateForSeparateReview {
            evidence_sha256: table.identity_sha256(&r)?,
            basis: "measured_family_specialization".into()
        }
    );
    Ok(())
}

#[test]
fn qualification_and_baseline_purposes_never_enable_adaptive_league() -> Result<()> {
    let r = registry(3)?;
    let qual = manifest(&r, PayoffPurpose::Qualification, "cycle")?;
    assert_eq!(
        Pfsp::assess(&qual, &r)?,
        LeagueGate::Deferred {
            reason: LeagueGateReason::QualificationPurpose
        }
    );
    let baseline = manifest(&r, PayoffPurpose::FrozenBaseline, "cycle")?;
    assert_eq!(
        Pfsp::assess(&baseline, &r)?,
        LeagueGate::Deferred {
            reason: LeagueGateReason::FrozenBaselinePurpose
        }
    );
    Ok(())
}
