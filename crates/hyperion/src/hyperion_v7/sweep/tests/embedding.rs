#[test]
fn embedding_only_matches_full_scatter_and_exact_patch_order() -> Result<()> {
    use super::run_composite_embedding_sweep;
    let input = grid(17, 7);
    for fixed in [false, true] {
        for (batch, chunks) in [(1, 1), (7, 5), (1024, 1)] {
            let plan = SweepPlan {
                sweeps: 3,
                scales: vec![4, 7, 11, 15],
                batch,
                seed: 20_260_623,
                target_chunks_per_sweep: chunks,
                fixed_perms: fixed.then(|| identity_perms(17, 7, 3)),
            };
            let full_model = EmbeddingTrace::default();
            let embedding_model = EmbeddingTrace {
                forbid_confidence: true,
                ..Default::default()
            };
            let full = run_composite_sweep(
                &full_model,
                &input,
                crate::hyperion_v7::sweep::PatchInput {
                    pad_u8: 128,
                    quant_scale: 3.,
                    max_patch: 24,
                },
                &plan,
                &NullSink,
                |_| {},
            )?;
            let emb = run_composite_embedding_sweep(
                &embedding_model,
                &input,
                128,
                3.,
                24,
                &plan,
                &NullSink,
            )?;
            assert_eq!(emb.cellmap, full.cellmap);
            assert_eq!(emb.row_emb, full.row_emb);
            assert_eq!(emb.dataset_vec, full.dataset_vec);
            assert_eq!(
                (emb.h, emb.w, emb.cancelled, emb.advanced),
                (full.h, full.w, full.cancelled, full.advanced)
            );
            assert_eq!(
                *embedding_model.calls.lock().unwrap(),
                *full_model.calls.lock().unwrap()
            );
        }
    }
    Ok(())
}

#[test]
fn embedding_only_preserves_cooperative_stopping() -> Result<()> {
    use super::run_composite_embedding_sweep;
    struct Stop {
        cancel: bool,
    }
    impl crate::progress::Sink for Stop {
        fn emit(&self, _: &crate::progress::Event) {}
        fn is_cancelled(&self) -> bool {
            self.cancel
        }
        fn take_advance_request(&self, _: u8) -> bool {
            !self.cancel
        }
    }
    let plan = SweepPlan {
        sweeps: 3,
        scales: vec![4, 7],
        batch: 1,
        seed: 19,
        target_chunks_per_sweep: 5,
        fixed_perms: None,
    };
    for cancel in [false, true] {
        let input = grid(17, 7);
        let full_model = EmbeddingTrace::default();
        let embedding_model = EmbeddingTrace {
            forbid_confidence: true,
            ..Default::default()
        };
        let full = run_composite_sweep(
            &full_model,
            &input,
            crate::hyperion_v7::sweep::PatchInput {
                pad_u8: 128,
                quant_scale: 3.,
                max_patch: 24,
            },
            &plan,
            &Stop { cancel },
            |_| {},
        )?;
        let emb = run_composite_embedding_sweep(
            &embedding_model,
            &input,
            128,
            3.,
            24,
            &plan,
            &Stop { cancel },
        )?;
        assert_eq!((emb.cancelled, emb.advanced), (cancel, !cancel));
        assert_eq!(emb.cellmap, full.cellmap);
        assert_eq!(emb.row_emb, full.row_emb);
        assert_eq!(emb.dataset_vec, full.dataset_vec);
        assert_eq!(embedding_model.calls.lock().unwrap().len(), 1);
        assert_eq!(
            *embedding_model.calls.lock().unwrap(),
            *full_model.calls.lock().unwrap()
        );
    }
    Ok(())
}

#[test]
fn embedding_only_refuses_invalid_embedding_shape() {
    use super::run_composite_embedding_sweep;
    struct Invalid;
    impl PatchModel for Invalid {
        fn forward_batch(&self, _: &[f32], _: &[f32], _: usize, _: usize) -> Result<PatchOutputs> {
            anyhow::bail!("confidence branch must not run")
        }
        fn forward_emb_batch(&self, _: &[f32], _: &[f32], _: usize, _: usize) -> Result<Vec<f32>> {
            Ok(vec![])
        }
        fn emb_dim(&self) -> usize {
            512
        }
    }
    let plan = SweepPlan {
        sweeps: 1,
        scales: vec![4],
        batch: 1,
        seed: 17,
        target_chunks_per_sweep: 1,
        fixed_perms: None,
    };
    let err = run_composite_embedding_sweep(&Invalid, &grid(9, 6), 128, 3., 24, &plan, &NullSink)
        .err()
        .unwrap();
    assert!(err.to_string().contains("emb floats"));
}
