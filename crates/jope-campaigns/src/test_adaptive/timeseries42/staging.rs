fn stage_campaign(
    root: &Path,
    specs: &[DatasetSpec],
    release_binary_sha256: &str,
    corpus_eval_sha256: &str,
) -> Result<(StagingReceipt, String)> {
    let path = root.join("staging.v1.json");
    if path.is_file() {
        let bytes = fs::read(&path)?;
        let staging: StagingReceipt = serde_json::from_slice(&bytes)?;
        validate_staging_receipt(&staging, specs, release_binary_sha256, corpus_eval_sha256)?;
        if bytes != canonical_json(&staging)? {
            bail!("invalid TimeSeries42 staging receipt");
        }
        verify_existing_staging_assets(&staging)?;
        return Ok((staging, sha256_bytes(&bytes)));
    }
    let release_binary = std::env::current_exe()?;
    if sha256_file(&release_binary)? != release_binary_sha256 {
        bail!("release binary changed before TimeSeries42 staging");
    }
    let corpus_eval = Path::new(corpus_eval_str());
    if sha256_file(corpus_eval)? != corpus_eval_sha256 {
        bail!("corpus_eval changed before TimeSeries42 staging");
    }
    let marker_source = root.join("node-marker.v1");
    immutable_or_equal(&marker_source, NODE_MARKER_BYTES)?;
    let mut nodes = Vec::new();
    let controller = controller_hostname()?;
    for node in NODES {
        let local_node = is_local_scq_node(node, &controller);
        let hostname = if local_node {
            controller.clone()
        } else {
            String::from_utf8(
                scq_batch::ssh_output(node, &["hostname", "-s"], "TimeSeries42 node hostname")?
                    .stdout,
            )?
            .trim()
            .to_string()
        };
        if hostname.is_empty() {
            bail!("TimeSeries42 node {node} reported an empty hostname");
        }
        let bundle_root = remote_bundle_root(release_binary_sha256);
        let runner_path = format!("{bundle_root}/jope-dime-regressor");
        let corpus_eval_path = format!("{bundle_root}/corpus_eval");
        let marker_path = node_marker_path(&bundle_root, &hostname);
        if local_node {
            copy_local_checked(&release_binary, Path::new(&runner_path), true)?;
            copy_local_checked(corpus_eval, Path::new(&corpus_eval_path), true)?;
            copy_local_checked(&marker_source, Path::new(&marker_path), false)?;
        } else {
            copy_remote_checked(node, &release_binary, &runner_path, true)?;
            copy_remote_checked(node, corpus_eval, &corpus_eval_path, true)?;
            copy_remote_checked(node, &marker_source, &marker_path, false)?;
        }
        let mut datasets = Vec::with_capacity(specs.len());
        for spec in specs {
            let remote_root = remote_dataset_root(release_binary_sha256, &spec.dataset_hash);
            let train_path = format!("{remote_root}/train.csv");
            let test_path = format!("{remote_root}/test.csv");
            let meta_path = format!("{remote_root}/meta.json");
            if local_node {
                copy_local_checked(Path::new(&spec.train_path), Path::new(&train_path), false)?;
                copy_local_checked(Path::new(&spec.test_path), Path::new(&test_path), false)?;
                copy_local_checked(Path::new(&spec.meta_path), Path::new(&meta_path), false)?;
            } else {
                copy_remote_checked(node, Path::new(&spec.train_path), &train_path, false)?;
                copy_remote_checked(node, Path::new(&spec.test_path), &test_path, false)?;
                copy_remote_checked(node, Path::new(&spec.meta_path), &meta_path, false)?;
            }
            datasets.push(StagedDataset {
                dataset_hash: spec.dataset_hash.clone(),
                train_path,
                train_sha256: spec.train_sha256.clone(),
                test_path,
                test_sha256: spec.test_sha256.clone(),
                meta_path,
                meta_sha256: spec.meta_sha256.clone(),
            });
        }
        nodes.push(StagedNode {
            node: node.to_string(),
            hostname,
            bundle_root,
            runner_path,
            runner_sha256: release_binary_sha256.to_string(),
            corpus_eval_path,
            corpus_eval_sha256: corpus_eval_sha256.to_string(),
            datasets,
        });
    }
    let staging = StagingReceipt {
        schema: STAGING_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        release_binary_sha256: release_binary_sha256.to_string(),
        corpus_eval_sha256: corpus_eval_sha256.to_string(),
        nodes,
    };
    validate_staging_receipt(&staging, specs, release_binary_sha256, corpus_eval_sha256)?;
    verify_staging_assets(&staging)?;
    let sha = write_json(&path, &staging)?;
    Ok((staging, sha))
}

fn verify_existing_staging_assets(staging: &StagingReceipt) -> Result<()> {
    if env_truthy(SKIP_RESUME_STAGING_VERIFY_ENV) {
        eprintln!(
            "TimeSeries42 resume staging verification skipped by {SKIP_RESUME_STAGING_VERIFY_ENV}=1"
        );
        return Ok(());
    }
    if env_truthy(FULL_RESUME_STAGING_VERIFY_ENV) {
        return verify_staging_assets(staging);
    }
    verify_representative_staging_assets(staging)
}

fn validate_staging_receipt(
    staging: &StagingReceipt,
    specs: &[DatasetSpec],
    release_binary_sha256: &str,
    corpus_eval_sha256: &str,
) -> Result<()> {
    if staging.schema != STAGING_SCHEMA
        || !staging.test_adaptive
        || staging.benchmark_valid
        || staging.release_binary_sha256 != release_binary_sha256
        || staging.corpus_eval_sha256 != corpus_eval_sha256
        || staging.nodes.len() != NODES.len()
    {
        bail!("invalid TimeSeries42 staging receipt");
    }
    for (index, node) in staging.nodes.iter().enumerate() {
        if node.node != NODES[index]
            || node.hostname.is_empty()
            || node.runner_sha256 != release_binary_sha256
            || node.corpus_eval_sha256 != corpus_eval_sha256
            || node.datasets.len() != specs.len()
        {
            bail!("invalid TimeSeries42 staging node {}", node.node);
        }
        for spec in specs {
            let staged = node
                .datasets
                .iter()
                .find(|candidate| candidate.dataset_hash == spec.dataset_hash)
                .with_context(|| format!("staging omits {} on {}", spec.dataset_hash, node.node))?;
            if staged.train_sha256 != spec.train_sha256
                || staged.test_sha256 != spec.test_sha256
                || staged.meta_sha256 != spec.meta_sha256
            {
                bail!("invalid TimeSeries42 staged dataset {}", spec.dataset_hash);
            }
        }
    }
    Ok(())
}

fn validate_staged_path(
    node: &StagedNode,
    path: &str,
    sha256: &str,
    executable: bool,
) -> Result<()> {
    let mode = staging_file_mode(executable);
    if is_controller_node(&node.node)? {
        validate_local_checked(Path::new(path), sha256, mode)
    } else {
        scq_batch::validate_remote_immutable(&node.node, path, sha256, mode)
    }
    .with_context(|| format!("verify TimeSeries42 staged path {}:{path}", node.node))
}

fn verify_staging_assets(staging: &StagingReceipt) -> Result<()> {
    let marker_sha256 = sha256_bytes(NODE_MARKER_BYTES);
    for node in &staging.nodes {
        validate_staged_path(node, &node.runner_path, &node.runner_sha256, true)?;
        validate_staged_path(node, &node.corpus_eval_path, &node.corpus_eval_sha256, true)?;
        validate_staged_path(
            node,
            &node_marker_path(&node.bundle_root, &node.hostname),
            &marker_sha256,
            false,
        )?;
        for dataset in &node.datasets {
            validate_staged_path(node, &dataset.train_path, &dataset.train_sha256, false)?;
            validate_staged_path(node, &dataset.test_path, &dataset.test_sha256, false)?;
            validate_staged_path(node, &dataset.meta_path, &dataset.meta_sha256, false)?;
        }
    }
    Ok(())
}

fn verify_representative_staging_assets(staging: &StagingReceipt) -> Result<()> {
    let marker_sha256 = sha256_bytes(NODE_MARKER_BYTES);
    let errors = Arc::new(Mutex::new(Vec::<String>::new()));
    thread::scope(|scope| {
        for node in &staging.nodes {
            let errors = Arc::clone(&errors);
            let marker_sha256 = marker_sha256.clone();
            scope.spawn(move || {
                let result = (|| -> Result<()> {
                    validate_staged_path(node, &node.runner_path, &node.runner_sha256, true)?;
                    validate_staged_path(
                        node,
                        &node.corpus_eval_path,
                        &node.corpus_eval_sha256,
                        true,
                    )?;
                    validate_staged_path(
                        node,
                        &node_marker_path(&node.bundle_root, &node.hostname),
                        &marker_sha256,
                        false,
                    )?;
                    let dataset = node
                        .datasets
                        .first()
                        .with_context(|| format!("staging node {} has no datasets", node.node))?;
                    validate_staged_path(node, &dataset.train_path, &dataset.train_sha256, false)?;
                    validate_staged_path(node, &dataset.test_path, &dataset.test_sha256, false)?;
                    validate_staged_path(node, &dataset.meta_path, &dataset.meta_sha256, false)?;
                    Ok(())
                })();
                if let Err(error) = result {
                    errors
                        .lock()
                        .unwrap()
                        .push(format!("{}: {error:#}", node.node));
                }
            });
        }
    });
    let errors = Arc::try_unwrap(errors)
        .expect("TimeSeries42 staging verification error references remain")
        .into_inner()
        .unwrap();
    if !errors.is_empty() {
        bail!(
            "TimeSeries42 representative staging verification failed: {}",
            errors.join("; ")
        );
    }
    eprintln!(
        "TimeSeries42 representative resume staging verification passed; set {FULL_RESUME_STAGING_VERIFY_ENV}=1 for full re-hash"
    );
    Ok(())
}

fn staged_dataset<'a>(
    staging: &'a StagingReceipt,
    node: &str,
    dataset: &str,
) -> Result<&'a StagedDataset> {
    staging
        .nodes
        .iter()
        .find(|record| record.node == node)
        .and_then(|record| {
            record
                .datasets
                .iter()
                .find(|candidate| candidate.dataset_hash == dataset)
        })
        .with_context(|| format!("staging lacks {node}/{dataset}"))
}

fn staged_node<'a>(staging: &'a StagingReceipt, node: &str) -> Result<&'a StagedNode> {
    staging
        .nodes
        .iter()
        .find(|record| record.node == node)
        .with_context(|| format!("staging lacks node {node}"))
}

fn label_authority(
    generation: usize,
    seed_sha256: &str,
    release_sha256: &str,
    campaign_sha256: &str,
) -> LabelAuthority {
    LabelAuthority {
        score_tier: "test_exact".to_string(),
        metric: "negative_mase".to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        selection_bias: Some("test-selected".to_string()),
        generation: format!("strict-wave-{generation:06}"),
        frozen_seed_sha256: seed_sha256.to_string(),
        model_checkpoint_sha256: release_sha256.to_string(),
        hall_of_fame_sha256: campaign_sha256.to_string(),
    }
}

fn corpus_manifest(
    spec: &DatasetSpec,
    members: &[Member],
    paths: (&str, &str, &str),
    authority: LabelAuthority,
    protocol_sha256: Option<&str>,
    execution_binding: Option<(&str, &str, &str)>,
) -> Result<CorpusShardManifest> {
    let (schema, reduction) = match spec.reduction()? {
        ReductionVersion::V1 => (CORPUS_SHARD_SCHEMA, None),
        version => (
            CORPUS_SHARD_SCHEMA_V2,
            Some(CorpusReductionSpec {
                version: version.as_str().to_string(),
            }),
        ),
    };
    let checkpoint_sha256 = protocol_sha256.map(|_| authority.model_checkpoint_sha256.clone());
    let (run_id, campaign_sha256, release_binary_sha256) = execution_binding
        .map(|(run_id, campaign_sha256, release_binary_sha256)| {
            (
                Some(run_id.to_string()),
                Some(campaign_sha256.to_string()),
                Some(release_binary_sha256.to_string()),
            )
        })
        .unwrap_or((None, None, None));
    Ok(CorpusShardManifest {
        schema: schema.to_string(),
        run_id,
        campaign_sha256,
        release_binary_sha256,
        comparator_lock_sha256: execution_binding.map(|_| spec.comparator_lock_sha256.clone()),
        task: "timeseries".to_string(),
        dataset_hash: spec.dataset_hash.clone(),
        train_csv: paths.0.to_string(),
        test_csv: paths.1.to_string(),
        train_sha256: spec.train_sha256.clone(),
        test_sha256: spec.test_sha256.clone(),
        checkpoint_epoch: checkpoint_sha256
            .as_ref()
            .map(|digest| format!("checkpoint-{digest}")),
        checkpoint_sha256,
        catalog_sha256: protocol_sha256.map(|_| authority.frozen_seed_sha256.clone()),
        protocol_sha256: protocol_sha256.map(str::to_string),
        generated_at_ms: protocol_sha256.map(|_| 0),
        label_authority: authority,
        meta_json: paths.2.to_string(),
        meta_sha256: spec.meta_sha256.clone(),
        horizon: spec.horizon,
        seasonality: spec.seasonality,
        frequency: spec.frequency.clone(),
        target_is_final_column: true,
        known_future_covariates: spec.known_future_covariates.clone(),
        past_only_covariates: Vec::new(),
        reduction,
        candidates: members
            .iter()
            .map(|member| ShardCandidate {
                semantic_sha256: member.semantic_sha256.clone(),
                canonical_genome: member.canonical_genome.clone(),
            })
            .collect(),
    })
}

fn dataset_protocol_sha256(spec: &DatasetSpec) -> Result<String> {
    let path = Path::new(coverage_comparator_root_str())
        .join("timeseries")
        .join(&spec.dataset_hash)
        .join("comparator-lock.v2.json");
    let lock: Value = load_json(&path)?;
    let protocol = lock
        .get("protocol_sha256")
        .and_then(Value::as_str)
        .context("timeseries comparator lacks protocol_sha256")?;
    if lock
        .pointer("/protocol/protocol_sha256")
        .and_then(Value::as_str)
        != Some(protocol)
    {
        bail!("timeseries comparator protocol drift at {}", path.display());
    }
    Ok(protocol.to_string())
}

fn validate_with_corpus_eval(manifest: &Path) -> Result<()> {
    let output = Command::new(corpus_eval_str())
        .args(["validate-manifest", "--manifest"])
        .arg(manifest)
        .output()
        .with_context(|| format!("run corpus_eval validate-manifest {}", manifest.display()))?;
    if !output.status.success() {
        bail!(
            "corpus_eval rejected {}: {}",
            manifest.display(),
            String::from_utf8_lossy(if output.stderr.is_empty() {
                &output.stdout
            } else {
                &output.stderr
            })
            .trim()
        );
    }
    Ok(())
}
