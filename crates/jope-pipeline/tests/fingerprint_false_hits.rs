#[path = "../../feat-math/src/kernel/strategy/fingerprint.rs"]
mod fingerprint;

use feat_math::GenomeIr;
use fingerprint::{
    ByteIdentity, CanonicalIdentity, FingerprintError, ProbeRelation, ProbeRole, ProbeRow,
    TrainProbeSet,
};

fn train_row(row_id: u64, x: f64) -> ProbeRow {
    ProbeRow {
        row_id,
        values: vec![x],
        role: ProbeRole::Train,
        ancestry: vec![ProbeRole::Train],
    }
}

#[test]
fn finite_probe_collision_is_only_advisory_and_does_not_prove_equal_behavior() {
    let probes = TrainProbeSet::new(
        "synthetic-training-partition-v1",
        vec![train_row(10, -1.0), train_row(11, 0.0), train_row(12, 1.0)],
    )
    .unwrap();

    // f(x)=0 and g(x)=x(x-1)(x+1) match at every probe point.
    let first = probes.fingerprint_outputs(&[0.0, 0.0, 0.0], 8).unwrap();
    let second = probes.fingerprint_outputs(&[0.0, 0.0, 0.0], 8).unwrap();
    assert_eq!(first.compare(&second), ProbeRelation::SameOnFiniteProbes);

    let held_x: f64 = 2.0;
    let held_first = 0.0;
    let held_second = held_x * (held_x - 1.0) * (held_x + 1.0);
    assert_eq!(held_first, 0.0);
    assert_eq!(held_second, 6.0);
    assert_ne!(held_first, held_second);

    // Rounded probe buckets can also collide when the raw probe outputs differ.
    let rounded_a = probes
        .fingerprint_outputs(&[1.0001, 1.0001, 1.0001], 2)
        .unwrap();
    let rounded_b = probes
        .fingerprint_outputs(&[1.0002, 1.0002, 1.0002], 2)
        .unwrap();
    assert_eq!(
        rounded_a.compare(&rounded_b),
        ProbeRelation::SameOnFiniteProbes
    );

    // The advisory API has no constructor for a measured observation or proof.
    assert_eq!(probes.len(), 3);
    assert_eq!(fingerprint::STRATEGY_ID, "advisory-fingerprint.v1");
}

#[test]
fn probe_builder_rejects_outer_sealed_unknown_and_tainted_ancestry() {
    for role in [
        ProbeRole::Validation,
        ProbeRole::OuterEvaluation,
        ProbeRole::Sealed,
        ProbeRole::Unknown,
    ] {
        let error = TrainProbeSet::new(
            "partition-v1",
            vec![ProbeRow {
                row_id: 1,
                values: vec![0.0],
                role,
                ancestry: vec![ProbeRole::Train],
            }],
        )
        .unwrap_err();
        assert_eq!(error, FingerprintError::NonTrainRole { row_id: 1, role });
    }

    let error = TrainProbeSet::new(
        "partition-v1",
        vec![ProbeRow {
            row_id: 2,
            values: vec![1.0],
            role: ProbeRole::Train,
            ancestry: vec![ProbeRole::Train, ProbeRole::Sealed],
        }],
    )
    .unwrap_err();
    assert_eq!(
        error,
        FingerprintError::NonTrainAncestor {
            row_id: 2,
            ancestor: ProbeRole::Sealed
        }
    );

    let valid = TrainProbeSet::new("partition-v1", vec![train_row(3, 2.0)]).unwrap();
    assert_eq!(
        valid
            .fingerprint_outputs(&[4.0], 4)
            .unwrap()
            .compare(&valid.fingerprint_outputs(&[4.0], 4).unwrap()),
        ProbeRelation::SameOnFiniteProbes
    );
}

#[test]
fn byte_and_canonical_identity_are_distinct_domains() {
    let source_a = "(pipe X (ridge))";
    let source_b = " (pipe X (ridge)) ";
    let bytes_a = ByteIdentity::of(source_a.as_bytes());
    let bytes_b = ByteIdentity::of(source_b.as_bytes());
    assert_ne!(bytes_a, bytes_b);

    let canonical_a = CanonicalIdentity::of_source(source_a).unwrap();
    let canonical_b = CanonicalIdentity::of_source(source_b).unwrap();
    assert_eq!(canonical_a, canonical_b);

    let parsed = GenomeIr::parse(source_a).unwrap();
    assert_eq!(canonical_a, CanonicalIdentity::of_genome(&parsed));
    assert_ne!(bytes_a.as_bytes(), canonical_a.as_bytes());
}

#[test]
fn fingerprints_are_bound_to_exact_probe_set_and_rounding() {
    let probes_a =
        TrainProbeSet::new("train-a", vec![train_row(1, 0.0), train_row(2, 1.0)]).unwrap();
    let probes_b =
        TrainProbeSet::new("train-b", vec![train_row(1, 0.0), train_row(2, 1.0)]).unwrap();
    let a = probes_a.fingerprint_outputs(&[2.0, 3.0], 2).unwrap();
    let b = probes_b.fingerprint_outputs(&[2.0, 3.0], 2).unwrap();
    assert_eq!(a.compare(&b), ProbeRelation::IncomparableProbeSets);
    let finer = probes_a.fingerprint_outputs(&[2.0, 3.0], 3).unwrap();
    assert_eq!(a.compare(&finer), ProbeRelation::IncomparableProbeSets);
    assert_ne!(a.digest(), b.digest());
}
