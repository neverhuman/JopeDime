use feat_math::kernel::ordered_axis::{OrderedInputIdentity, TaskMode};
use feat_math::kernel::realized_path_stats::{
    Config, Edit, ExperimentalProfile, PathStatsProgram, PathStatsSearch,
};
use feat_math::kernel::wire_v2::{VersionError, VersionedProgram};
use feat_math::GenomeIr;
#[test]
fn bounded_profile_typed_routes_and_reader_first_ingress() {
    let input = OrderedInputIdentity::new("group", "time");
    let child = GenomeIr::parse("(pipe X (ridge lambda_frac=0.01))").unwrap();
    let profile = ExperimentalProfile {
        path_stats: true,
        prior: 0.125,
    };
    let disabled = ExperimentalProfile {
        path_stats: false,
        prior: 0.125,
    };
    let search =
        PathStatsSearch::new(1, input.clone(), child.clone(), TaskMode::Forecasting).unwrap();
    assert!(PathStatsSearch::new(1, input.clone(), child.clone(), TaskMode::IID).is_err());
    assert!(PathStatsSearch::new(0, input.clone(), child.clone(), TaskMode::Ordered).is_err());
    assert!(PathStatsSearch::new(
        1,
        OrderedInputIdentity::new("g".repeat(3_000_000), "t"),
        child.clone(),
        TaskMode::Ordered
    )
    .is_err());
    let opaque_child = GenomeIr::parse("(pipe X (standardize) (ridge lambda_frac=0.01))").unwrap();
    assert!(
        PathStatsSearch::new(1, input.clone(), opaque_child.clone(), TaskMode::Ordered).is_err()
    );
    let opaque = VersionedProgram::PathStatsV2(
        PathStatsProgram::new(Config::new(2, 64, 4).unwrap(), input, opaque_child.clone()).unwrap(),
    );
    assert!(search.edit(&opaque, 7, Edit::Change, profile).is_err());
    let dropped = search.edit(&opaque, 7, Edit::Drop, disabled).unwrap();
    assert_eq!(
        dropped.to_bytes().unwrap(),
        VersionedProgram::LegacyV1(feat_math::kernel::Program::LegacyV1(opaque_child))
            .to_bytes()
            .unwrap()
    );
    let mut born = 0;
    for seed in 0..1024 {
        let node = search.generate(seed, profile).unwrap();
        born += usize::from(matches!(node, VersionedProgram::PathStatsV2(_)));
        let bytes = node.to_bytes().unwrap();
        assert_eq!(
            VersionedProgram::from_bytes(&bytes)
                .unwrap()
                .to_bytes()
                .unwrap(),
            bytes
        );
        assert_eq!(
            VersionedProgram::parse_lossless_text(&node.to_lossless_text().unwrap())
                .unwrap()
                .to_bytes()
                .unwrap(),
            bytes
        );
        assert_eq!(
            search.generate(seed, disabled).unwrap().to_bytes().unwrap(),
            search.base().to_bytes().unwrap()
        );
    }
    assert_eq!(born, 131);
    let grow = search.edit(&search.base(), 7, Edit::Grow, profile).unwrap();
    let change = search.edit(&grow, 7, Edit::Change, profile).unwrap();
    assert_ne!(grow.to_bytes().unwrap(), change.to_bytes().unwrap());
    assert_eq!(
        search
            .edit(&change, 7, Edit::Drop, disabled)
            .unwrap()
            .to_bytes()
            .unwrap(),
        search.base().to_bytes().unwrap()
    );
    assert!(search.edit(&grow, 7, Edit::Change, disabled).is_err());
    for zero in [0.0, -0.0] {
        let p = ExperimentalProfile {
            path_stats: true,
            prior: zero,
        };
        assert_eq!(
            ExperimentalProfile::from_bytes(&p.to_bytes().unwrap())
                .unwrap()
                .prior
                .to_bits(),
            zero.to_bits()
        );
    }
    for version in [0, 1, 3] {
        let mut b = profile.to_bytes().unwrap();
        b[7] = version;
        assert!(matches!(
            ExperimentalProfile::from_bytes(&b),
            Err(VersionError::UnsupportedVersion(_))
        ));
    }
    let mut b = profile.to_bytes().unwrap();
    b[8] = 2;
    assert!(ExperimentalProfile::from_bytes(&b).is_err());
    for prior in [-1.0, 0.25001, f64::INFINITY, f64::NAN] {
        assert!(ExperimentalProfile {
            path_stats: true,
            prior
        }
        .to_bytes()
        .is_err());
    }
}
