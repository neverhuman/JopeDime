macro_rules! projection_and_vocabulary_ignore_audit_data {
    ($make_job:expr) => {
        #[test]
        fn projection_and_vocabulary_ignore_audit_data() {
            for task in ["regression", "binary", "multi_label"] {
                let (raw, mut job) = fixture(task);
                let p = prepare(&raw, &job, 42).unwrap();
                let poisoned = std::str::from_utf8(&raw)
                    .unwrap()
                    .lines()
                    .enumerate()
                    .map(|(i, line)| {
                        if p.fold.audit.contains(&i) {
                            let mut row = line.split(',').map(str::to_owned).collect::<Vec<_>>();
                            row[0] = "999999999".into();
                            row[3] = "987654321".into();
                            row.join(",") + "\n"
                        } else {
                            line.to_owned() + "\n"
                        }
                    })
                    .collect::<String>()
                    .into_bytes();
                job.sha256 = hash(&poisoned);
                job.bytes = poisoned.len();
                let q = prepare(&poisoned, &job, 42).unwrap();
                assert_eq!(p.columns, q.columns);
                assert_eq!(p.labels, q.labels);
                assert_eq!(p.fit.rows, q.fit.rows);
                assert_eq!(p.fit_y, q.fit_y);
                assert_eq!(p.selection.rows, q.selection.rows);
                assert_eq!(p.selection_y, q.selection_y);
            }

            let make = |poison: bool| {
                let f = fold(100, "wide", 42, false);
                (0..100)
                    .map(|i| {
                        let mut row = (0..70)
                            .map(|j| {
                                if poison && f.audit.contains(&i) && j == 69 {
                                    "10000000".into()
                                } else {
                                    format!("{}", (i * (j + 1)) % 101)
                                }
                            })
                            .collect::<Vec<_>>();
                        row.push(i.to_string());
                        row.join(",") + "\n"
                    })
                    .collect::<String>()
                    .into_bytes()
            };
            let raw = make(false);
            let mut job = ($make_job)(&raw);
            let p = prepare(&raw, &job, 42).unwrap();
            let raw = make(true);
            job.sha256 = hash(&raw);
            job.bytes = raw.len();
            let q = prepare(&raw, &job, 42).unwrap();
            assert_eq!(p.columns.len(), 64);
            assert_eq!(p.columns, q.columns);
            assert_eq!(p.fit.rows, q.fit.rows);
            assert_eq!(
                p.fit.rows.len() + p.selection.rows.len() + p.audit.rows.len(),
                100
            );
        }
    };
}

#[allow(unused_macros)]
macro_rules! classification_and_regression_replay_and_audit_isolation {
    () => {
        #[test]
        fn classification_and_regression_replay_and_audit_isolation() {
            for task in ["regression", "binary", "multi_label"] {
                let (raw, job) = fixture(task);
                let mut p = prepare(&raw, &job, 42).unwrap();
                let a = evaluate(&p, &job, 42, 600).unwrap();
                assert_eq!(a["search"]["candidate_attempts"], 12);
                assert_eq!(a["champion"]["serialized_model_replay_exact"], true);
                assert!(a["search"]["trace"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|t| t["origin"] == "mutate"));
                for y in &mut p.audit_y {
                    *y += 10000.0;
                }
                let b = evaluate(&p, &job, 42, 600).unwrap();
                assert_eq!(a["champion"]["model_sha256"], b["champion"]["model_sha256"]);
                assert_eq!(a["champion"]["dsl"], b["champion"]["dsl"]);
                assert_eq!(
                    a["champion"]["audit_predictions_sha256"],
                    b["champion"]["audit_predictions_sha256"]
                );
                assert_eq!(
                    a["baseline_after_selection"]["audit_predictions_sha256"],
                    b["baseline_after_selection"]["audit_predictions_sha256"]
                );
                let baseline_predictions: Vec<f64> = serde_json::from_value(
                    a["baseline_after_selection"]["audit_predictions"].clone(),
                )
                .unwrap();
                assert_eq!(
                    hash(
                        &baseline_predictions
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect::<Vec<_>>()
                    ),
                    a["baseline_after_selection"]["audit_predictions_sha256"]
                );
                assert_eq!(
                    a["baseline_after_selection"]["audit_used_for_selection"],
                    false
                );
                assert_ne!(
                    a["champion"]["audit_metrics"],
                    b["champion"]["audit_metrics"]
                );
            }
        }
    };
}
