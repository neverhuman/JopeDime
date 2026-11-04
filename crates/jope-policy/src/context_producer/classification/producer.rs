use super::*;

impl CpuProducer {
    pub fn produce_classification(
        &self,
        input_bytes: &[u8],
        source_sha40: &str,
        producer_sha256: &str,
    ) -> Result<ClassificationArtifact> {
        ensure!(
            input_bytes.len() <= 128 << 20,
            "classification input exceeds128MiB"
        );
        validate_hex(source_sha40, 40)?;
        validate_hex(producer_sha256, 64)?;
        let input: ClassificationInput = serde_json::from_slice(input_bytes)?;
        // One canonical input encoding makes worker/parent replay exact.
        ensure!(
            serde_json::to_vec(&input)? == input_bytes,
            "classification input must be canonical typed JSON"
        );
        let prepared = input.prepare()?;
        // Input scope of this encode's products: the feature rows' JSON digest.
        let scope = input_scope(&prepared.rows)?;
        let n = prepared.rows.len();
        let width = input.features.len();
        let mut columns = vec![0f32; n * width];
        for (i, row) in prepared.rows.iter().enumerate() {
            for j in 0..width {
                columns[j * n + i] = row[j] as f32;
            }
        }
        ensure!(
            columns.iter().all(|v| v.is_finite()),
            "classification feature conversion exceeds FP32"
        );
        let matrix = Matrix::new(columns, n, width);
        let bounds = fit_quantile_bounds(&matrix, self.settings.q_low, self.settings.q_high);
        let surface = CellSurface::from_quantized_rows(
            quantize_u8_with_bounds(&matrix, &bounds, self.settings.pad_u8),
            n,
            &scope,
        )?;
        let sweep = run_composite_embedding_sweep(
            &self.hyperion,
            surface.grid(),
            self.settings.pad_u8,
            self.settings.quant_scale,
            self.settings.max_patch,
            &SweepPlan {
                sweeps: 3,
                scales: vec![4, 7, 11, 15],
                batch: 1024,
                seed: self.settings.seed,
                target_chunks_per_sweep: 1,
                fixed_perms: None,
            },
            &NullSink,
        )?;
        ensure!(
            !sweep.cancelled && sweep.row_emb.len() == n * RowEmbeddings::WIDTH,
            "incomplete classification feature sweep"
        );
        let row_embeddings = RowEmbeddings::from_hyperion_sweep(sweep.row_emb, &surface)?;
        let pooled = HyperionSignature::pool(&row_embeddings)?;
        let row_encodings = row_embeddings.rows().map(<[f32]>::to_vec).collect();
        let neutral = prepare_jope_foundation_rows(&neutral_foundation_rows(&prepared.rows))?;
        ensure!(
            neutral.input.context_y.iter().all(|v| *v == 0.),
            "neutral foundation labels changed"
        );
        let foundation = RawFoundation::from_foundation_encoder(
            self.foundation.encode_standardized(&neutral.input)?.vector,
            &scope,
        )?;
        // Same frozen named transformations and arithmetic as the regression assembly's first
        // two blocks; the Target block is not produced here.
        let hyperion = self.transforms.signature_block(&pooled)?;
        let projected_foundation = self
            .transforms
            .projection_block(&self.transforms.project(&foundation)?)?;
        let artifact = ClassificationArtifact {
            schema: CLASSIFICATION_SCHEMA.into(),
            feature_policy: CLASSIFICATION_FEATURE_POLICY.into(),
            backend: BACKEND.into(),
            task: input.task,
            classes: input.classes.clone(),
            source_sha40: source_sha40.into(),
            producer_sha256: producer_sha256.into(),
            input_sha256: sha(input_bytes),
            training: input.identity.clone(),
            original_labels_sha256: sha(&serde_json::to_vec(
                &input.rows.iter().map(|r| &r.label).collect::<Vec<_>>(),
            )?),
            preprocessing_sha256: sha(&serde_json::to_vec(&prepared.preprocessing)?),
            preprocessing: prepared.preprocessing,
            feature_rows_sha256: sha(&serde_json::to_vec(&prepared.rows)?),
            reference_encoder: self.identity.clone(),
            hyperion,
            projected_foundation,
            row_encodings,
            native_classification_parity_claimed: false,
            accepted_regression_policy_compatible: false,
            final_test_targets_used: false,
        };
        artifact.validate_for_input(&input)?;
        Ok(artifact)
    }
}
