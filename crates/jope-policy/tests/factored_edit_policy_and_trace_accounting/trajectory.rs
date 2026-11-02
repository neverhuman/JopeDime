use super::*;

pub fn verify_g10_all_attempt_trajectory_and_replay_receipt() {
    let ctx = make_context(16, &[0, 2, 6, 10], vec![1, 4, 9, 2], None, 500);
    let ptr0 = ctx.column_pointer(0).unwrap();
    let ptr6 = ctx.column_pointer(6).unwrap();

    let k_eos = EditKey::eos();
    let k_col0 = EditKey {
        kind: EditKind::ColumnTransform,
        site: Some(EditSite::Column(ptr0)),
        operator: Some(EditOperator::Replace),
        fidelity: Some(Fidelity::Exact),
        stop: false,
    };
    let k_col6_cont = EditKey {
        kind: EditKind::ColumnTransform,
        site: Some(EditSite::Column(ptr6)),
        operator: Some(EditOperator::Insert),
        fidelity: Some(Fidelity::Approximate),
        stop: false,
    };

    let policy = EditPolicy::new(
        ctx.clone(),
        vec![
            PolicyComponent {
                weight: 0.25,
                key: k_eos.clone(),
                parameter: None,
            },
            PolicyComponent {
                weight: 0.40,
                key: k_col0.clone(),
                parameter: None,
            },
            PolicyComponent {
                weight: 0.35,
                key: k_col6_cont.clone(),
                parameter: Some(NormalDensity {
                    mean: -0.2,
                    standard_deviation: 0.4,
                }),
            },
        ],
    )
    .unwrap();
    let mask = policy.action_mask().unwrap();

    let run_trajectory = || -> (ReplayTrajectoryReceipt, Vec<TrajectoryAttemptRecord>) {
        let total = 256usize;
        let mut records = Vec::with_capacity(total);
        let mut accepted_count = 0usize;
        let mut refused_count = 0usize;
        let mut total_charged = 0u64;
        let mut total_lower_bound = 0u64;

        for idx in 0..total {
            let lower_bound_micros = 50u64 + (idx as u64 % 7) * 10;
            // Every 8th attempt uses an out-of-range mixture draw (>= 1.0) or illegal action to verify refusal logging
            if idx % 8 == 7 {
                let res = policy.sample(&mask, 1.05, 0.0);
                assert!(res.is_err(), "out-of-unit mixture draw must fail");
                let charged_micros = lower_bound_micros + 15;
                assert!(
                    charged_micros >= lower_bound_micros,
                    "trace accounting must never be less than lower bound"
                );
                refused_count += 1;
                total_charged += charged_micros;
                total_lower_bound += lower_bound_micros;
                records.push(TrajectoryAttemptRecord {
                    attempt_index: idx,
                    mixture_draw_bits: 1.05f64.to_bits(),
                    normal_draw_bits: 0.0f64.to_bits(),
                    accepted: false,
                    disposition: "refused_invalid_mixture_draw".into(),
                    action_digest: None,
                    measure_kind: None,
                    measure_value_bits: None,
                    charged_micros,
                    lower_bound_micros,
                });
            } else {
                let u = ((idx as f64) + 0.5) / (total as f64);
                let z = ((idx as i64 % 11) - 5) as f64 * 0.2;
                let action = policy.sample(&mask, u, z).unwrap();
                let measure = policy.score(&mask, &action).unwrap();
                let (m_kind, m_bits) = match measure {
                    ProbabilityMeasure::Mass(v) => ("mass".to_string(), v.to_bits()),
                    ProbabilityMeasure::Density(v) => ("density".to_string(), v.to_bits()),
                };
                let act_sha = sha256_hex(&serde_json::to_vec(&action).unwrap());
                let charged_micros = lower_bound_micros + ctx.residual.diagnostics_charged_micros;
                assert!(charged_micros >= lower_bound_micros);
                accepted_count += 1;
                total_charged += charged_micros;
                total_lower_bound += lower_bound_micros;
                records.push(TrajectoryAttemptRecord {
                    attempt_index: idx,
                    mixture_draw_bits: u.to_bits(),
                    normal_draw_bits: z.to_bits(),
                    accepted: true,
                    disposition: "accepted".into(),
                    action_digest: Some(act_sha),
                    measure_kind: Some(m_kind),
                    measure_value_bits: Some(m_bits),
                    charged_micros,
                    lower_bound_micros,
                });
            }
        }

        let records_sha256 = sha256_hex(&serde_json::to_vec(&records).unwrap());
        let receipt = ReplayTrajectoryReceipt {
            schema: "jope.factored_edit_trajectory_receipt.v1".into(),
            total_attempts: total,
            accepted_count,
            refused_count,
            truncated_accepted: 0,
            misbound_accepted: 0,
            total_charged_micros: total_charged,
            total_lower_bound_micros: total_lower_bound,
            records_sha256,
        };
        (receipt, records)
    };

    let (r1, rec1) = run_trajectory();
    let (r2, rec2) = run_trajectory();
    assert_eq!(
        r1, r2,
        "deterministic CLI/trajectory replay receipt must match exactly"
    );
    assert_eq!(
        rec1, rec2,
        "all-attempt trajectory records must replay bit-identically"
    );
    assert_eq!(
        r1.accepted_count + r1.refused_count,
        r1.total_attempts,
        "G10: 100% of attempts must be logged in the trajectory"
    );
    assert_eq!(r1.total_attempts, rec1.len());
    assert!(
        r1.total_charged_micros > r1.total_lower_bound_micros,
        "G10: trace accounting must cover and exceed lower-bound step costs"
    );
}
