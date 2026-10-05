use feat_math::kernel::finite_difference::{DifferenceSearch, Edit, ExperimentalProfile};
use feat_math::kernel::ordered_axis::OrderedInputIdentity;
use feat_math::kernel::ordered_axis::TaskMode;
use feat_math::kernel::wire_v2::VersionError;
use feat_math::kernel::wire_v2::VersionedProgram;
#[test]
fn bounded_profile_generation_codec_and_gp_arms() {
    // Parseable stored programs retain opaque downstream bytes, but generation
    // must refuse whole shapes that cannot execute all three training recipes.
    let input = OrderedInputIdentity::new("g", "t");
    let unsupported =
        feat_math::GenomeIr::parse("(pipe X (standardize) (ridge lambda_frac=0.01))").unwrap();
    assert!(
        DifferenceSearch::new(1, input.clone(), unsupported.clone(), TaskMode::Forecasting)
            .is_err()
    );
    let p = ExperimentalProfile {
        difference: true,
        prior: 0.125,
    };
    let search = DifferenceSearch::new(
        1,
        OrderedInputIdentity::new("g", "t"),
        feat_math::GenomeIr::parse("(pipe X (ridge lambda_frac=0.01))").unwrap(),
        TaskMode::Forecasting,
    )
    .unwrap();
    assert!(DifferenceSearch::new(
        1,
        input.clone(),
        feat_math::GenomeIr::parse("(pipe X (ridge lambda_frac=0.01))").unwrap(),
        TaskMode::IID
    )
    .is_err());
    assert!(DifferenceSearch::new(
        1,
        OrderedInputIdentity::new("g".repeat(3_000_000), "t"),
        feat_math::GenomeIr::parse("(pipe X (ridge lambda_frac=0.01))").unwrap(),
        TaskMode::Forecasting
    )
    .is_err());
    for version in [0, 1, 3] {
        let mut bytes = p.to_bytes().unwrap();
        bytes[7] = version;
        assert!(matches!(
            ExperimentalProfile::from_bytes(&bytes),
            Err(VersionError::UnsupportedVersion(_))
        ));
    }
    let mut flags = p.to_bytes().unwrap();
    flags[8] = 2;
    assert!(matches!(
        ExperimentalProfile::from_bytes(&flags),
        Err(VersionError::UnsupportedVersion(_))
    ));
    let grow = search.edit(&search.base(), 7, Edit::Grow, p).unwrap();
    let opaque = VersionedProgram::LegacyV1(feat_math::kernel::Program::LegacyV1(unsupported));
    assert!(search.edit(&opaque, 7, Edit::Grow, p).is_err());
    let feat_math::kernel::wire_v2::VersionedProgram::LegacyV1(
        feat_math::kernel::Program::LegacyV1(ref downstream),
    ) = opaque
    else {
        unreachable!()
    };
    let stored = VersionedProgram::DifferenceV2(
        feat_math::kernel::finite_difference::DifferenceProgram::new(
            feat_math::kernel::finite_difference::DifferenceConfig::new(2, 59, 1, 1.).unwrap(),
            1,
            input,
            downstream.clone(),
        )
        .unwrap(),
    );
    assert!(search.edit(&stored, 7, Edit::Change, p).is_err());
    let bytes = stored.to_bytes().unwrap();
    assert_eq!(
        VersionedProgram::from_bytes(&bytes)
            .unwrap()
            .to_bytes()
            .unwrap(),
        bytes
    );
    let change = search.edit(&grow, 7, Edit::Change, p).unwrap();
    let disabled = ExperimentalProfile {
        difference: false,
        prior: 0.,
    };
    let drop = search.edit(&change, 7, Edit::Drop, disabled).unwrap();
    assert_eq!(
        search
            .edit(&stored, 7, Edit::Drop, disabled)
            .unwrap()
            .to_bytes()
            .unwrap(),
        opaque.to_bytes().unwrap()
    );
    assert_eq!(drop.to_bytes().unwrap(), search.base().to_bytes().unwrap());
    assert_ne!(grow.to_bytes().unwrap(), change.to_bytes().unwrap());
    let mut present = 0;
    for seed in 0..1024 {
        let x = search.generate(seed, p).unwrap();
        present += usize::from(matches!(x, VersionedProgram::DifferenceV2(_)));
        let bytes = x.to_bytes().unwrap();
        assert_eq!(
            VersionedProgram::from_bytes(&bytes)
                .unwrap()
                .to_bytes()
                .unwrap(),
            bytes
        );
        let text = x.to_lossless_text().unwrap();
        assert_eq!(
            VersionedProgram::parse_lossless_text(&text)
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
    assert_eq!(present, 131);
    assert_ne!(p.identity().unwrap(), disabled.identity().unwrap());
    for prior in [0., -0., 0.125, 0.25] {
        let x = ExperimentalProfile {
            difference: true,
            prior,
        };
        assert_eq!(
            ExperimentalProfile::from_bytes(&x.to_bytes().unwrap())
                .unwrap()
                .prior
                .to_bits(),
            prior.to_bits()
        );
    }
    for prior in [f64::NAN, f64::INFINITY, -0.01, 0.250001] {
        assert!(ExperimentalProfile {
            difference: true,
            prior
        }
        .validate()
        .is_err());
    }
    for n in 0..17 {
        assert!(ExperimentalProfile::from_bytes(&p.to_bytes().unwrap()[..n]).is_err());
    }
    assert!(search.edit(&grow, 7, Edit::Change, disabled).is_err());
    println!(
        "{} seeded programs; {} wire59 generated; grow/change/drop and profile codec PASS",
        1024, present
    );
}
