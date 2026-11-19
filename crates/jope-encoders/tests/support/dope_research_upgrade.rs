//! Deterministic unit and integration checks for opt-in `Ultra` 629-D, `PacketFeaturesV1` 196-D,
//! and `UltraPacket859` 859-D APIs, verifying isolation from serving `Dope512`.

use super::{DopeDevice, DopeEncoder, WEIGHTS};
use dope_core::mlc_v3::{MlcV3Encoding, ObliviousTree, TargetBundle, MLC_V3_GATE_KEYS_14};
use starforge::dope_encoder::{
    build_research_conditioning, extract_packet_v1_from_dml3_bytes,
    extract_packet_v1_from_encoding, prepare_table_with_ids, BundleScope, LandmarkBundle, PrepTask,
    ResearchConditioningLevel, DOPE_COL_LANDMARK_POOL_34_DIM, DOPE_RESEARCH_VIEW_629_DIM,
    DOPE_ULTRA_PACKET_859_DIM,
};
use std::collections::BTreeMap;

fn build_synthetic_encoding(level: &str, low_bound: f64, high_bound: f64) -> MlcV3Encoding {
    let mut gates = BTreeMap::new();
    for (idx, name) in MLC_V3_GATE_KEYS_14.iter().enumerate() {
        gates.insert((*name).to_string(), 0.04 * (idx as f64 + 1.5));
    }
    MlcV3Encoding {
        level: level.to_string(),
        task: "regression".to_string(),
        code_width: 8192,
        n_train: 160,
        p_use: 3,
        gates: gates.clone(),
        task_latent: (0..128).map(|i| ((i as f32) - 50.0) * 0.008).collect(),
        dist_latent: (0..128).map(|i| ((i as f32) - 20.0) * 0.006).collect(),
        knot_grid: vec![0.05, 0.25, 0.5, 0.75, 0.95],
        col_knots: vec![
            vec![-1.8, -0.4, 0.0, 0.5, 1.9],
            vec![-1.2, -0.3, 0.1, 0.7, 1.5],
            vec![0.0, 0.0, 1.0, 1.0, 1.0],
        ],
        is_binary_col: vec![false, false, true],
        binary_probs: vec![0.0, 0.0, 0.35],
        binary_low_high: vec![[0.0, 0.0], [0.0, 0.0], [0.0, 1.0]],
        discrete_support: BTreeMap::new(),
        chol_factor: vec![vec![0.7, 0.1], vec![0.25, 0.65], vec![0.15, 0.35]],
        diag_res_std: vec![0.62, 0.71, 0.88],
        regime_weights: vec![0.4, 0.6],
        regime_means: vec![vec![-0.3, 0.1, 0.0], vec![0.3, -0.1, 0.2]],
        regime_scales: vec![vec![0.9, 1.0, 1.0], vec![1.1, 0.95, 1.0]],
        target_bundle: TargetBundle {
            task_kind: "regression".to_string(),
            gates,
            pairs: vec![(0, 1), (1, 2)],
            include_v3_primitives: true,
            w_basis: vec![
                0.05, 0.7, -0.25, 0.15, 0.3, 0.1, -0.1, 0.12, 0.04, 0.08, 0.22, -0.18,
            ],
            res_net_mean: 0.09,
            neural_w: 0.42,
            trees: vec![ObliviousTree {
                features: vec![0, 1],
                thresholds: vec![-0.1, 0.3],
                leaf_values: vec![-0.6, -0.1, 0.2, 0.8],
            }],
            residual_std: 0.38,
            noise_scale: 0.27,
            class_prior: 0.5,
            y_min: low_bound,
            y_max: high_bound,
            k_classes: 1,
            class_priors: Vec::new(),
            u_basis: Vec::new(),
            v_basis: Vec::new(),
            mc_trees: Vec::new(),
        },
    }
}

#[test]
fn research_view_629_enforces_level_and_matches_dope512_prefix_on_high() {
    let encoder = DopeEncoder::from_bytes(WEIGHTS, &DopeDevice::Cpu).unwrap();
    let x = (0..24)
        .map(|i| vec![(i as f64) * 0.25, ((i % 5) as f64) - 2.0])
        .collect::<Vec<_>>();
    let y = x
        .iter()
        .map(|r| 1.5 * r[0] - 0.4 * r[1])
        .collect::<Vec<_>>();
    let prepared = prepare_table_with_ids(&x, &y, PrepTask::Regression, (0..24).collect()).unwrap();
    let zero_bundle = LandmarkBundle::zero();
    let high_cond =
        build_research_conditioning(&prepared, &zero_bundle, ResearchConditioningLevel::High)
            .unwrap();
    let dope512 = encoder.encode_conditioning(&high_cond).unwrap();
    let view629 = encoder
        .encode_research_view_629(&high_cond, ResearchConditioningLevel::High)
        .unwrap();
    assert_eq!(view629.features629.len(), DOPE_RESEARCH_VIEW_629_DIM);
    assert_eq!(view629.col_landmarks34.len(), DOPE_COL_LANDMARK_POOL_34_DIM);
    assert_eq!(
        &view629.features629[..512],
        dope512.to_vec().unwrap().as_slice()
    );
    assert!(encoder
        .encode_research_view_629(&high_cond, ResearchConditioningLevel::Ultra)
        .is_err());
    assert!(
        build_research_conditioning(&prepared, &zero_bundle, ResearchConditioningLevel::Ultra)
            .is_err()
    );
}

#[test]
fn research_view_629_ultra_requires_all_23_landmarks_and_activates_tail() {
    let encoder = DopeEncoder::from_bytes(WEIGHTS, &DopeDevice::Cpu).unwrap();
    let x = (0..32)
        .map(|i| vec![(i as f64 * 0.19).sin(), (i as f64) / 16.0, ((i % 4) as f64)])
        .collect::<Vec<_>>();
    let y = x
        .iter()
        .map(|r| 0.8 * r[0] + r[1] * r[1] - 0.1 * r[2])
        .collect::<Vec<_>>();
    let prepared = prepare_table_with_ids(&x, &y, PrepTask::Regression, (0..32).collect()).unwrap();
    let values = [0.35f32; 23];
    let full_bundle = LandmarkBundle::from_squashed(
        &values,
        &[true; 23],
        BundleScope {
            subsample_n: None,
            source_digest: "f".repeat(64),
            provenance: None,
        },
    );
    let high_cond =
        build_research_conditioning(&prepared, &full_bundle, ResearchConditioningLevel::High)
            .unwrap();
    let ultra_cond =
        build_research_conditioning(&prepared, &full_bundle, ResearchConditioningLevel::Ultra)
            .unwrap();
    let high_view = encoder
        .encode_research_view_629(&high_cond, ResearchConditioningLevel::High)
        .unwrap();
    let ultra_view = encoder
        .encode_research_view_629(&ultra_cond, ResearchConditioningLevel::Ultra)
        .unwrap();
    assert_eq!(ultra_view.features629.len(), DOPE_RESEARCH_VIEW_629_DIM);
    assert_ne!(high_view.features629, ultra_view.features629);
    assert!(ultra_view.features629[512..].iter().any(|v| v.abs() > 1e-6));

    let mut tampered = ultra_cond.clone();
    tampered.bundle.mask[22] = false;
    assert!(encoder
        .encode_research_view_629(&tampered, ResearchConditioningLevel::Ultra)
        .is_err());
}

#[test]
fn packet_v1_summary_is_extrema_free_and_distinguishes_target_ablation() {
    let enc_a = build_synthetic_encoding("ultra", -10.0, 25.0);
    let enc_b = build_synthetic_encoding("ultra", -9.0e6, 4.5e7);
    let pkt_a = extract_packet_v1_from_encoding(&enc_a, true).unwrap();
    let pkt_b = extract_packet_v1_from_encoding(&enc_b, true).unwrap();
    let pkt_notarget = extract_packet_v1_from_encoding(&enc_a, false).unwrap();
    assert_eq!(pkt_a.global28.len(), 28);
    assert_eq!(pkt_a.features196.len(), 196);
    assert_eq!(pkt_a.features196, pkt_b.features196);
    assert_eq!(pkt_a.global28, pkt_b.global28);
    assert_eq!(pkt_a.digest, pkt_b.digest);
    assert_ne!(pkt_a.features196, pkt_notarget.features196);
    assert!(extract_packet_v1_from_dml3_bytes(b"short", true).is_err());
}

#[test]
fn ultra_packet_859_assembles_exact_layout_and_refuses_notarget_packet() {
    let encoder = DopeEncoder::from_bytes(WEIGHTS, &DopeDevice::Cpu).unwrap();
    let x = (0..28)
        .map(|i| vec![(i as f64) * 0.15, ((i % 3) as f64)])
        .collect::<Vec<_>>();
    let y = x
        .iter()
        .map(|r| 2.0 * r[0] + 0.5 * r[1])
        .collect::<Vec<_>>();
    let prepared = prepare_table_with_ids(&x, &y, PrepTask::Regression, (0..28).collect()).unwrap();
    let full_bundle = LandmarkBundle::from_squashed(
        &[0.42f32; 23],
        &[true; 23],
        BundleScope {
            subsample_n: None,
            source_digest: "e".repeat(64),
            provenance: None,
        },
    );
    let ultra_cond =
        build_research_conditioning(&prepared, &full_bundle, ResearchConditioningLevel::Ultra)
            .unwrap();
    let view = encoder
        .encode_research_view_629(&ultra_cond, ResearchConditioningLevel::Ultra)
        .unwrap();
    let enc = build_synthetic_encoding("ultra", -5.0, 5.0);
    let pkt = extract_packet_v1_from_encoding(&enc, true).unwrap();
    let up859 = encoder.encode_ultra_packet_859(&ultra_cond, &pkt).unwrap();
    assert_eq!(up859.features859.len(), DOPE_ULTRA_PACKET_859_DIM);
    assert_eq!(&up859.features859[..629], view.features629.as_slice());
    assert_eq!(
        &up859.features859[629..663],
        view.col_landmarks34.as_slice()
    );
    assert_eq!(&up859.features859[663..859], pkt.features196.as_slice());

    let pkt_notarget = extract_packet_v1_from_encoding(&enc, false).unwrap();
    assert!(encoder
        .encode_ultra_packet_859(&ultra_cond, &pkt_notarget)
        .is_err());
}
