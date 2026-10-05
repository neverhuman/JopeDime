use feat_math::kernel::ordered_axis::{FoldBoundary, GroupKey, OrderedInputIdentity, TaskMode};
use feat_math::kernel::timescale_nabla::{Config, TimeGradientState, TIME_GAP_REJECTED};
#[test]
fn maximum_width_checkpoint_and_gap_rejections_preserve_state() {
    let input = OrderedInputIdentity::new("group", "time");
    let fold = FoldBoundary::new("maximum-width", 1, 2).unwrap();
    let mut state = TimeGradientState::new(
        Config::new(2, 75, 512, 0.125).unwrap(),
        TaskMode::Forecasting,
        &input,
        fold.clone(),
    )
    .unwrap();
    let group = GroupKey::categorical("a").unwrap();
    assert!(state.push(group.clone(), 0., &[]).is_err());
    let first = state.push(group.clone(), 0., &vec![2.; 512]).unwrap();
    assert_eq!(first.values.len(), 512);
    assert_eq!(first.values, vec![0.; 512]);
    let bytes = state.to_bytes().unwrap();
    let mut restored =
        TimeGradientState::from_bytes(&bytes, TaskMode::Forecasting, &input, fold).unwrap();
    assert_eq!(restored.to_bytes().unwrap(), bytes);
    for t in [0., -1., 0.0625, f64::NAN, f64::INFINITY] {
        let rejected = restored.push(group.clone(), t, &vec![5.; 512]).unwrap_err();
        assert_eq!(rejected.faults, TIME_GAP_REJECTED);
        assert_eq!(restored.to_bytes().unwrap(), bytes);
    }
    let output = restored.push(group, 0.5, &vec![5.; 512]).unwrap();
    assert_eq!(output.values, vec![6.; 512]);
    assert_eq!(output.faults, vec![0; 512]);
    restored.finish().unwrap();
    use feat_math::kernel::timescale_nabla::{
        Edit, ExperimentalProfile, OrderedRow, TimeGradientFittedCpp, TimeGradientModel,
        TimeGradientProgram, TimeGradientSearch,
    };
    use feat_math::kernel::wire_v2::{VersionedModel, VersionedProgram};
    let child = feat_math::GenomeIr::parse("(pipe X (ridge lambda_frac=0.01))").unwrap();
    for width in 1..=512 {
        let p = TimeGradientProgram::new(
            Config::new(2, 75, width, 0.125).unwrap(),
            input.clone(),
            child.clone(),
        )
        .unwrap();
        let bytes = p.to_bytes().unwrap();
        assert_eq!(
            TimeGradientProgram::from_bytes(&bytes)
                .unwrap()
                .to_bytes()
                .unwrap(),
            bytes
        );
        let routed = VersionedProgram::from_bytes(&bytes).unwrap();
        assert_eq!(routed.to_bytes().unwrap(), bytes);
        assert_eq!(
            VersionedProgram::parse_lossless_text(&routed.to_lossless_text().unwrap())
                .unwrap()
                .to_bytes()
                .unwrap(),
            bytes
        );
        if width == 256 || width == 512 {
            let search =
                TimeGradientSearch::new(width, input.clone(), child.clone(), TaskMode::Forecasting)
                    .unwrap();
            let grown = search
                .edit(
                    &search.base(),
                    7,
                    Edit::Grow,
                    ExperimentalProfile {
                        time_gradient: true,
                        prior: 0.125,
                    },
                )
                .unwrap();
            assert_eq!(
                VersionedProgram::from_bytes(&grown.to_bytes().unwrap())
                    .unwrap()
                    .to_bytes()
                    .unwrap(),
                grown.to_bytes().unwrap()
            );
            let rows = [
                OrderedRow {
                    group: GroupKey::numeric(0.).unwrap(),
                    timestamp: 0.,
                    values: vec![2.; width],
                },
                OrderedRow {
                    group: GroupKey::numeric(0.).unwrap(),
                    timestamp: 0.5,
                    values: vec![5.; width],
                },
            ];
            let model = p
                .fit_training(
                    &rows[..1],
                    &[1.],
                    TaskMode::Forecasting,
                    FoldBoundary::new("wide", 1, 2).unwrap(),
                )
                .unwrap();
            let json = model.to_json().unwrap();
            assert_eq!(
                TimeGradientModel::from_json(&json)
                    .unwrap()
                    .to_json()
                    .unwrap(),
                json
            );
            let mut public = VersionedModel::from_json(&json).unwrap();
            let fitted = public.emit_time_gradient_fitted_cpp17().unwrap();
            assert_eq!(
                TimeGradientFittedCpp::from_codec_json(&fitted.codec)
                    .unwrap()
                    .source,
                fitted.source
            );
            let expected = model.clone().predict(&rows[1..]).unwrap();
            assert_eq!(
                public
                    .predict_ordered(&rows[1..])
                    .unwrap()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }
    for width in [0, 513, u64::MAX] {
        let p = TimeGradientProgram::new(
            Config::new(2, 75, 1, 0.125).unwrap(),
            input.clone(),
            child.clone(),
        )
        .unwrap();
        let mut bytes = p.to_bytes().unwrap();
        bytes[26..34].copy_from_slice(&width.to_le_bytes());
        assert!(VersionedProgram::from_bytes(&bytes).is_err());
    }
    assert!(Config::new(2, 75, 513, 0.125).is_err());
}
