pub struct CpuProducer {
    hyperion: V7Model,
    settings: hyperion::hyperion::HyperionSweepSettings,
    foundation: StarlightFoundationEncoder,
    target: FusedTargetEncoder,
    transforms: FrozenTransforms,
    identity: EncoderIdentity,
}
impl CpuProducer {
    pub fn load(files: &EncoderFiles) -> Result<Self> {
        Self::load_on_device(files, Device::Cpu)
    }

    fn load_on_device(files: &EncoderFiles, device: Device) -> Result<Self> {
        ensure!(
            files.hyperion_weights.sha256 == HYPERION_WEIGHTS
                && files.hyperion_manifest.sha256 == HYPERION_MANIFEST
                && files.frozen_transforms.sha256 == TRANSFORMS_SHA256,
            "encoder lineage differs from accepted component assets"
        );
        // The model is parsed from exactly the authenticated bytes; the weights path is not
        // reopened, so a replacement after authentication (A->B->A) cannot reach inference.
        let weights = files.hyperion_weights.read(512 << 20)?;
        let settings = verified_hyperion_settings(&files.hyperion_manifest)?;
        let hyperion = if device.is_cpu() {
            V7Model::load_cpu_from_bytes(&weights, &settings)?
        } else {
            #[cfg(feature = "inference-cuda")]
            {
                V7Model::load_cuda_from_bytes(&weights, &settings, device.clone())?
            }
            #[cfg(not(feature = "inference-cuda"))]
            {
                anyhow::bail!("CUDA inference feature required")
            }
        };
        drop(weights);
        ensure!(
            files.component_bundle.is_absolute()
                && files.component_bundle.canonicalize()? == files.component_bundle,
            "component bundle must be physical and absolute"
        );
        let bundle = LoadedModelBundle::load(&files.component_bundle)?;
        ensure!(bundle.target_seed() == 17, "frozen target seed changed");
        // Derived from the loaded (pinned) bundle, then held to the release decision: the
        // producer can only ever report the accepted identity.
        let identity = EncoderIdentity {
            hyperion_weights: HYPERION_WEIGHTS.into(),
            hyperion_manifest: HYPERION_MANIFEST.into(),
            component_bundle: sha(&serde_json::to_vec(&bundle.manifest)?),
            foundation_weights: bundle
                .manifest
                .encoders
                .get("foundation")
                .context("foundation identity absent")?
                .sha256
                .clone(),
            target_weights: bundle
                .manifest
                .encoders
                .get("target")
                .context("target identity absent")?
                .sha256
                .clone(),
            frozen_transforms: TRANSFORMS_SHA256.into(),
            assembly: ASSEMBLY_SHA256.into(),
            recipe: ENCODER_RECIPE.into(),
        };
        identity.require_accepted()?;
        Ok(Self {
            hyperion,
            settings,
            foundation: StarlightFoundationEncoder::load_bundle(
                &bundle,
                device.clone(),
                DType::F32,
            )?,
            target: FusedTargetEncoder::load_on_device(&bundle, device)?,
            transforms: FrozenTransforms::load(&files.frozen_transforms)?,
            identity,
        })
    }
    pub fn identity(&self) -> &EncoderIdentity {
        &self.identity
    }

    /// The only ordinary producer entry point consumes original training rows.
    /// It neither receives holdout rows/labels nor opens a corpus path.
    pub fn produce(
        &self,
        input_bytes: &[u8],
        source_sha: &str,
        producer_sha: &str,
    ) -> Result<CpuContextArtifact> {
        ensure!(
            input_bytes.len() <= 128 << 20,
            "training payload exceeds 128 MiB"
        );
        validate_hex(source_sha, 40)?;
        validate_hex(producer_sha, 64)?;
        let input: TrainingInput = serde_json::from_slice(input_bytes)?;
        let prepared = input.prepare()?;
        let components = self.encode_prepared(&prepared.rows)?;
        let context = self.transforms.assemble(&components)?;
        let artifact = CpuContextArtifact {
            schema: SCHEMA.into(),
            backend: BACKEND.into(),
            qualification: "unqualified".into(),
            source_sha256: source_sha.into(),
            producer_sha256: producer_sha.into(),
            input_sha256: sha(input_bytes),
            training: input.identity.clone(),
            selected_training_rows_sha256: sha(&serde_json::to_vec(
                &input.rows.iter().map(|r| &r.id).collect::<Vec<_>>(),
            )?),
            original_labels_sha256: sha(&serde_json::to_vec(
                &input.rows.iter().map(|r| &r.label).collect::<Vec<_>>(),
            )?),
            preprocessing_sha256: sha(&serde_json::to_vec(&prepared.preprocessing)?),
            // The components' input scope is exactly the prepared rows' JSON digest.
            prepared_rows_sha256: components.scope().into(),
            preprocessing: prepared.preprocessing,
            encoder: self.identity.clone(),
            components: components.to_vectors(),
            vector: context.values().to_vec(),
            final_test_targets_used: false,
        };
        artifact.validate()?;
        Ok(artifact)
    }

    /// Qualification seam only: rows are the frozen native train preprocessing
    /// result (features followed by one regression target), never a corpus root.
    /// This returns typed components, not an admitted context or policy binding.
    pub fn encode_prepared(&self, rows: &[Vec<f64>]) -> Result<Components> {
        self.encode_timed(rows, false).map(|v| v.0)
    }

    fn encode_timed(
        &self,
        rows: &[Vec<f64>],
        profile: bool,
    ) -> Result<(Components, BTreeMap<String, f64>, serde_json::Value)> {
        let mut times = BTreeMap::new();
        // Input scope of every product below; computed before the timed stages.
        let scope = input_scope(rows)?;
        if profile {
            self.hyperion.device().synchronize()?;
        }
        let started = std::time::Instant::now();
        let width = validate_rows(rows)? - 1;
        let n = rows.len();
        let mut columns = vec![0f32; n * width];
        for (i, row) in rows.iter().enumerate() {
            for j in 0..width {
                columns[j * n + i] = row[j] as f32;
            }
        }
        ensure!(
            columns.iter().all(|v| v.is_finite()),
            "feature conversion exceeds FP32"
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
            "incomplete Hyperion sweep"
        );
        let row_embeddings = RowEmbeddings::from_hyperion_sweep(sweep.row_emb, &surface)?;
        let hyperion = HyperionSignature::pool(&row_embeddings)?;
        if profile {
            self.hyperion.device().synchronize()?;
        }
        times.insert("hyperion".into(), started.elapsed().as_secs_f64());
        let started = std::time::Instant::now();
        let mut foundation = prepare_jope_foundation_rows(rows)?;
        foundation.input.context_y = standardize_context_labels(&foundation.input.context_y);
        let foundation_vector = RawFoundation::from_foundation_encoder(
            self.foundation
                .encode_standardized(&foundation.input)?
                .vector,
            &scope,
        )?;
        if profile {
            self.foundation.device().synchronize()?;
        }
        times.insert("foundation".into(), started.elapsed().as_secs_f64());
        let started = std::time::Instant::now();
        let targets = minmax_targets(&rows.iter().map(|r| r[width]).collect::<Vec<_>>());
        let windows = FrozenTargetWindows::native(n, TargetTask::Regression, 17)?;
        let sampling = serde_json::json!({"hyperion_rows":n,"hyperion_features":width,"hyperion_sweeps":3,"hyperion_scales":[4,7,11,15],"hyperion_batch":1024,"target_chunks_per_sweep":1,
            "foundation_context_rows":foundation.context_indices.len(),"foundation_query_rows":foundation.query_indices.len(),
            "target_neural_windows":windows.neural.iter().map(Vec::len).collect::<Vec<_>>(),"target_classical_windows":windows.classical.iter().map(Vec::len).collect::<Vec<_>>()});
        let target = TargetBlock::from_target_encoder(
            self.target
                .encode_with_windows(&targets, TargetTask::Regression, &windows)?
                .vector,
            &scope,
        )?;
        if profile {
            self.target.device().synchronize()?;
        }
        times.insert("target".into(), started.elapsed().as_secs_f64());
        let result = Components {
            hyperion,
            foundation: foundation_vector,
            target,
        };
        result.validate()?;
        Ok((result, times, sampling))
    }
    /// Named transformation: the frozen assembly of one encode's typed products.
    pub fn assemble(&self, components: &Components) -> Result<FinalContext> {
        self.transforms.assemble(components)
    }
}
