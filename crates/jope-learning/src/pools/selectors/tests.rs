use super::super::pool::{Member, Pool, Status, POOL_SCHEMA};
use super::*;

#[test]
fn cholesky_solves_a_small_system() {
    // [[4,2],[2,3]] x = [2,1] -> x = [0.5, 0]
    let x = cholesky_solve(vec![4.0, 2.0, 2.0, 3.0], vec![2.0, 1.0], 2).unwrap();
    assert!((x[0] - 0.5).abs() < 1e-12 && x[1].abs() < 1e-12);
}

#[test]
fn contextual_surrogate_qhat_learns_context_program_interactions_and_screens_funnel() {
    // Construct 4 training pools where contexts with z > 0 reward token 1 and contexts with z < 0 reward token 2.
    // Marginal popularity of token 1 and token 2 is identical (0.5 each), so context-blind selectors tie at 0.5,
    // while ContextualSurrogateQHat learns the bilinear z * phi interaction and ranks the held-out pool accurately.
    let mk_hex = |b: u8| format!("{b:02x}").repeat(32);
    let p_pos_a = mk_hex(0x11);
    let p_neg_a = mk_hex(0x22);
    let p_pos_b = mk_hex(0x33);
    let p_neg_b = mk_hex(0x44);

    let mk_pool = |cid: u8, parent: &str, pos_wins: bool, use_b: bool| -> Pool {
        let (p1, p2) = if use_b {
            (p_pos_b.clone(), p_neg_b.clone())
        } else {
            (p_pos_a.clone(), p_neg_a.clone())
        };
        let (q1, q2) = if pos_wins { (0.9, 0.1) } else { (0.1, 0.9) };
        Pool {
            schema: POOL_SCHEMA.into(),
            context_id: format!(
                "{}{}",
                if cid % 2 == 1 { "ff" } else { "00" },
                &mk_hex(cid)[2..]
            ),
            parent_id: parent.into(),
            role: "train".into(),
            protocol_sha256: mk_hex(0xaa),
            incumbent_q: Some(0.0),
            members: vec![
                Member {
                    program_sha256: p1,
                    status: Status::Complete,
                    q: Some(q1),
                    tokens: Some(vec![1, 1, 0]),
                },
                Member {
                    program_sha256: p2,
                    status: Status::Complete,
                    q: Some(q2),
                    tokens: Some(vec![2, 2, 0]),
                },
            ],
        }
    };

    let all_pools = vec![
        mk_pool(1, "p1", true, false),
        mk_pool(2, "p2", false, false),
        mk_pool(3, "p3", true, true),
        mk_pool(4, "p4", false, true),
        mk_pool(5, "p5", true, true),
    ];
    let prog_feats = token_bag_features(&all_pools).unwrap();
    let ctx_feats = Features::context_id_descriptors(&all_pools, 1).unwrap();

    let train_refs = vec![&all_pools[0], &all_pools[1], &all_pools[2], &all_pools[3]];
    let qhat =
        ContextualSurrogateQHat::fit(&train_refs, &ctx_feats, &prog_feats, 2, 2.0, 1e-2).unwrap();
    let (bm, pm) = qhat.selected_multipliers();
    assert!(
        [1.0, 10.0, 100.0, 1000.0].contains(&bm)
            && [1.0, 10.0, 100.0, 1000.0, 10_000.0, 100_000.0].contains(&pm)
            && pm >= bm
            && qhat
                .inner_cv_loss()
                .is_some_and(|v| v.is_finite() && v >= 0.0),
        "expected inner-CV selected multipliers in grid with pm >= bm, got ({bm}, {pm})"
    );
    assert!(qhat
        .rule()
        .contains(&format!("base_mult={bm}, proj_mult={pm}")));
    let single =
        ContextualSurrogateQHat::fit(&[&all_pools[0]], &ctx_feats, &prog_feats, 2, 2.0, 1e-2);
    assert!(single.is_err());

    let test_view = all_pools[4].view();
    let scores = qhat.scores(&test_view).unwrap();
    assert!(
        scores[0] > scores[1],
        "expected positive-z context to rank token-1 program higher: {scores:?}"
    );

    let funnel = pre_screen_funnel(&test_view, &qhat, 64, 1).unwrap();
    assert_eq!(funnel.selected_program_sha256, vec![p_pos_b.clone()]);

    // Verify Strength-of-Schedule (SoS) target calibration: when `p_pos_a` beats `p_neg_a` in two pools
    // (`t(p_pos_a) = 1.0`, `t(p_neg_a) = 0.0`), a program `p_pos_b` that ties `0.5` against elite `p_pos_a`
    // receives a strictly higher SoS-calibrated target than `p_neg_b` tying `0.5` against weak `p_neg_a`,
    // and non-tied pools with nonzero `ctx_offset` preserve their exact 1.0 winner-loser gap without clamping.
    let sos_pools = [
        mk_pool(10, "sp1", true, false),
        mk_pool(11, "sp2", true, false),
        Pool {
            schema: POOL_SCHEMA.into(),
            context_id: mk_hex(12),
            parent_id: "sp3".into(),
            role: "train".into(),
            protocol_sha256: mk_hex(0xaa),
            incumbent_q: Some(0.0),
            members: vec![
                Member {
                    program_sha256: p_pos_a.clone(),
                    status: Status::Complete,
                    q: Some(0.85),
                    tokens: None,
                },
                Member {
                    program_sha256: p_pos_b.clone(),
                    status: Status::Complete,
                    q: Some(0.85),
                    tokens: None,
                },
            ],
        },
        Pool {
            schema: POOL_SCHEMA.into(),
            context_id: mk_hex(13),
            parent_id: "sp4".into(),
            role: "train".into(),
            protocol_sha256: mk_hex(0xaa),
            incumbent_q: Some(0.0),
            members: vec![
                Member {
                    program_sha256: p_neg_a.clone(),
                    status: Status::Complete,
                    q: Some(0.15),
                    tokens: None,
                },
                Member {
                    program_sha256: p_neg_b.clone(),
                    status: Status::Complete,
                    q: Some(0.15),
                    tokens: None,
                },
            ],
        },
        Pool {
            schema: POOL_SCHEMA.into(),
            context_id: mk_hex(15),
            parent_id: "sp6".into(),
            role: "train".into(),
            protocol_sha256: mk_hex(0xaa),
            incumbent_q: Some(0.0),
            members: vec![
                Member {
                    program_sha256: p_pos_a.clone(),
                    status: Status::Complete,
                    q: Some(0.95),
                    tokens: None,
                },
                Member {
                    program_sha256: p_pos_b.clone(),
                    status: Status::Complete,
                    q: Some(0.75),
                    tokens: None,
                },
            ],
        },
    ];
    let raw_sos: Vec<(&Pool, Vec<f64>)> = sos_pools.iter().map(|p| (p, rank_targets(p))).collect();
    let calibrated = ContextualSurrogateQHat::compute_sos_targets(&raw_sos, 2.0);
    assert!(
            calibrated[2][1] > 0.5 && calibrated[3][1] < 0.5 && calibrated[2][1] > calibrated[3][1],
            "expected SoS calibration to boost target against elite opponent ({}) above weak opponent ({})",
            calibrated[2][1],
            calibrated[3][1]
        );
    assert!(
        (calibrated[2][0] - calibrated[2][1]).abs() < 1e-12,
        "expected equal-q members in the same context pool to retain identical calibrated targets"
    );
    assert!(
            calibrated[4][0] > 1.0 && (calibrated[4][0] - calibrated[4][1] - 1.0).abs() < 1e-12,
            "expected non-tied pool with positive ctx_offset to preserve exact 1.0 winner-loser gap: {:?}",
            calibrated[4]
        );
    let single_pool = Pool {
        schema: POOL_SCHEMA.into(),
        context_id: mk_hex(14),
        parent_id: "sp5".into(),
        role: "train".into(),
        protocol_sha256: mk_hex(0xaa),
        incumbent_q: Some(0.0),
        members: vec![Member {
            program_sha256: mk_hex(0x99),
            status: Status::Complete,
            q: Some(0.5),
            tokens: None,
        }],
    };
    let single_cal = ContextualSurrogateQHat::compute_sos_targets(
        &[(&single_pool, rank_targets(&single_pool))],
        2.0,
    );
    assert_eq!(single_cal, vec![vec![0.5]]);
    let seen_single_pool = Pool {
        schema: POOL_SCHEMA.into(),
        context_id: mk_hex(16),
        parent_id: "sp7".into(),
        role: "train".into(),
        protocol_sha256: mk_hex(0xaa),
        incumbent_q: Some(0.0),
        members: vec![Member {
            program_sha256: p_pos_a.clone(),
            status: Status::Complete,
            q: Some(0.5),
            tokens: None,
        }],
    };
    let seen_single_cal = ContextualSurrogateQHat::compute_sos_targets(
        &[
            (&sos_pools[0], rank_targets(&sos_pools[0])),
            (&seen_single_pool, rank_targets(&seen_single_pool)),
        ],
        2.0,
    );
    assert!(
        (seen_single_cal[1][0] - (1.0 + 0.5 * 2.0) / (1.0 + 2.0)).abs() < 1e-12,
        "expected single-member pool with out-of-parent history to equal mu_p=2/3, got {}",
        seen_single_cal[1][0]
    );
}

#[test]
fn pre_screen_funnel_deduplicates_before_proposal_pool_truncation() {
    struct FixedScores(Vec<f64>);
    impl Selector for FixedScores {
        fn name(&self) -> &str {
            "fixed"
        }
        fn class(&self) -> SelectorClass {
            SelectorClass::Candidate
        }
        fn rule(&self) -> String {
            "fixed".into()
        }
        fn scores(&self, _view: &PoolView<'_>) -> Result<Vec<f64>> {
            Ok(self.0.clone())
        }
    }

    let mk_hex = |b: u8| format!("{b:02x}").repeat(32);
    let p1 = mk_hex(0x01);
    let p2 = mk_hex(0x02);
    let p3 = mk_hex(0x03);
    // Feed 3 duplicate entries of p1 first, followed by p2 (highest score) and p3.
    // With proposal_pool_k = 2 and eval_budget_b = 2, deduplication BEFORE .take(proposal_pool_k)
    // must retain p1 and p2 rather than starving the proposal pool to 1 distinct program.
    let programs = vec![
        p1.as_str(),
        p1.as_str(),
        p1.as_str(),
        p2.as_str(),
        p3.as_str(),
    ];
    let view = PoolView {
        context_id: &mk_hex(0xaa),
        parent_id: "parent",
        incumbent_q: None,
        programs,
        tokens: vec![None; 5],
    };
    let selector = FixedScores(vec![0.2, 0.2, 0.2, 0.9, 0.5]);
    let funnel = pre_screen_funnel(&view, &selector, 2, 2).unwrap();
    assert_eq!(funnel.proposal_pool_k, 2);
    assert_eq!(funnel.eval_budget_b, 2);
    assert_eq!(funnel.selected_program_sha256, vec![p2, p1]);
}
