use super::*;

pub(crate) fn admit(output: &Path) -> Result<()> {
    let workspace = workspace_root();
    let compiler_output = Command::new("g++").arg("--version").output()?;
    if !compiler_output.status.success() {
        bail!("g++ --version failed");
    }
    let compiler = String::from_utf8(compiler_output.stdout)?;
    if !compiler.lines().next().unwrap_or_default().contains("13.3") {
        bail!("fitted C++ admission requires g++ 13.3, found {compiler:?}");
    }
    let commands = [
        (
            "default_workspace_check",
            vec!["check", "--workspace", "--all-targets"],
            None,
        ),
        (
            "all_feature_workspace_check",
            vec!["check", "--workspace", "--all-targets", "--all-features"],
            None,
        ),
        (
            "legacy_freeze_golden_champion_token_portable_parity",
            vec![
                "test",
                "-p",
                "feat-math",
                "--test",
                "legacy_v2_freeze",
                "--test",
                "golden",
                "--test",
                "champion_parity",
                "--test",
                "tokens",
                "--test",
                "portable_parity",
            ],
            None,
        ),
        ("gpu_backend_parity", vec!["test", "-p", "battle-gpu"], None),
        (
            "normal_full_registry_bit_parity",
            vec![
                "test",
                "-p",
                "feat-math",
                "--test",
                "advanced_v3_fitted_cpp",
                "--",
                "--test-threads=1",
            ],
            None,
        ),
        (
            "asan_full_registry_bit_parity",
            vec![
                "test",
                "-p",
                "feat-math",
                "--test",
                "advanced_v3_fitted_cpp",
                "--",
                "--test-threads=1",
            ],
            Some(("JOPE_FITTED_CPP_SANITIZER", "address")),
        ),
        (
            "ubsan_nonrecovering_full_registry_bit_parity",
            vec![
                "test",
                "-p",
                "feat-math",
                "--test",
                "advanced_v3_fitted_cpp",
                "--",
                "--test-threads=1",
            ],
            Some(("JOPE_FITTED_CPP_SANITIZER", "undefined")),
        ),
        (
            "codec_malformed_once_compiled_dynamic_runtime",
            vec!["test", "-p", "feat-math", "fitted_cpp_runtime"],
            None,
        ),
        (
            "mutation_crossover_100k",
            vec![
                "test",
                "-p",
                "feat-math",
                "one_hundred_thousand_typed_children_compile_without_retry",
            ],
            None,
        ),
    ];
    let mut gates = BTreeMap::new();
    for (name, args, environment) in commands {
        let started = Instant::now();
        let mut command = Command::new("cargo");
        command.current_dir(&workspace).args(&args);
        if let Some((key, value)) = environment {
            command.env(key, value);
        }
        let result = command
            .output()
            .with_context(|| format!("launch cargo gate {name}"))?;
        let mut combined = result.stdout;
        combined.extend_from_slice(&result.stderr);
        if !result.status.success() {
            bail!(
                "fitted C++ admission gate {name} failed:\n{}",
                String::from_utf8_lossy(&combined)
            );
        }
        gates.insert(
            name.to_string(),
            GateReceipt {
                passed: true,
                command: format!(
                    "{}cargo {}",
                    environment.map_or(String::new(), |(key, value)| format!("{key}={value} ")),
                    args.join(" ")
                ),
                elapsed_ms: started.elapsed().as_millis(),
                output_sha256: sha256_bytes(&combined),
            },
        );
    }
    let performance = performance_gate()?;
    let smoke = smoke_gate()?;
    let release = std::env::current_exe()?;
    let receipt = AdmissionReceipt {
        schema: "jope-dime.fitted-cpp-v3-admission.v1".to_string(),
        admitted: true,
        test_adaptive: true,
        benchmark_valid: false,
        compiler: compiler.lines().next().unwrap_or_default().to_string(),
        compiler_flags: [
            "-std=c++17",
            "-O3",
            "-march=znver2",
            "-ffp-contract=off",
            "-fno-fast-math",
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
        release_sha256: sha256_file(&release)?,
        source_sha256: source_tree_hash()?,
        gates,
        performance,
        smoke,
        created_at_unix_ms: unix_ms(),
    };
    atomic_immutable_write(output, &json_bytes(&receipt)?)
}

pub(super) fn performance_gate() -> Result<PerformanceReceipt> {
    let registry = paths::jope_registry();
    validate_dataset_hashes(&registry)?;
    let root = registry.join(DATASET);
    let train = read_regression_csv(&root.join("train.csv"))?;
    let test = read_regression_csv(&root.join("test.csv"))?;
    let genome = GenomeIr::parse(NEAR_WINNER)?;
    let training = campaign_training(&train.x)?;
    let model = fit_campaign_candidate(&genome, &training, &train.y)?;
    let legacy = fit_legacy_numeric_bound(&genome, &train.x, &train.y)?;
    let codec = encode_fitted_cpp17(&model)?;
    validate_fitted_cpp_codec(&codec.bytes)?;
    let encoded = EncodedTable::from_numeric_rows(&test.x)?;
    let runtime_root = workspace_root().join("target/fitted-cpp-admission-runtime");
    fs::create_dir_all(&runtime_root)?;
    let runtime = FittedCppRuntime::new(&runtime_root)?;
    let cpp = runtime.load(&codec.bytes)?;
    let raw = numeric_raw_table(&test.x)?;
    if prediction_bits(&legacy.predict_table(&raw)?) != prediction_bits(&model.predict_table(&raw)?)
    {
        bail!("V3 campaign fit is not bit-identical to the frozen legacy near-winner fit");
    }
    let mut rust_times = Vec::with_capacity(30);
    let mut cpp_times = Vec::with_capacity(30);
    let mut expected = Vec::new();
    let mut actual = Vec::new();
    for _ in 0..30 {
        let started = Instant::now();
        expected = model.predict_table(&raw)?;
        rust_times.push(started.elapsed().as_nanos());
        let started = Instant::now();
        actual = cpp.predict(&encoded)?;
        cpp_times.push(started.elapsed().as_nanos());
    }
    let parity = prediction_bits(&expected) == prediction_bits(&actual);
    if !parity {
        let (index, rust, cpp) = first_prediction_mismatch(&expected, &actual)
            .context("30-run fitted C++ performance gate lost bit parity without a mismatch")?;
        bail!(
            "30-run fitted C++ performance gate lost bit parity at prediction {index}: Rust {rust:?} ({:#018x}), C++ {cpp:?} ({:#018x})",
            rust.to_bits(),
            cpp.to_bits()
        );
    }
    rust_times.sort_unstable();
    cpp_times.sort_unstable();
    let rust_median = rust_times[15];
    let cpp_median = cpp_times[15];
    if cpp_median > rust_median.saturating_mul(4).saturating_add(1_000_000) {
        bail!(
            "fitted C++ prediction median {cpp_median}ns exceeds correctness-first envelope against Rust {rust_median}ns"
        );
    }
    drop(cpp);
    Ok(PerformanceReceipt {
        runs: 30,
        rust_median_ns: rust_median,
        cpp_median_ns: cpp_median,
        bit_parity: parity,
        codec_sha256: codec.sha256,
    })
}

pub(super) fn smoke_gate() -> Result<SmokeReceipt> {
    let started = Instant::now();
    let registry = paths::jope_registry();
    validate_dataset_hashes(&registry)?;
    let dataset_root = registry.join(DATASET);
    let test = read_regression_csv(&dataset_root.join("test.csv"))?;
    let release = std::env::current_exe()?;
    let release_sha256 = sha256_file(&release)?;
    let root = workspace_root().join("target").join(format!(
        "golden42-v3-smoke-{}-{}",
        std::process::id(),
        unix_ms()
    ));
    fs::create_dir_all(&root)?;
    let result = (|| -> Result<SmokeReceipt> {
        wait_for_resources(&root, "admission-smoke", 0, 0, WORKERS, 0, 0)?;
        let test_features = root.join("test-features.v1.json");
        atomic_immutable_write(&test_features, &json_bytes(&test.x)?)?;
        let cpu_ids = choose_cpus(WORKERS)?;
        let config = V3EvolutionConfig::default();
        let population = initial_population(&config, &v3_seed_sources());
        if population.len() != WORKERS {
            bail!("V3 smoke population is not 80 candidates");
        }
        let mut workers = spawn_workers(
            &release,
            &registry,
            &root,
            "admission-smoke",
            &dataset_root.join("train.csv"),
            &test_features,
            &cpu_ids,
        )?;
        let smoke_result = (|| -> Result<SmokeReceipt> {
            write_worker_registry(&root, &workers)?;
            let (_sender, receiver) = worker_channel(&mut workers)?;
            for (slot, candidate) in population.iter().enumerate() {
                let request = WorkerRequest {
                    schema: "jope-dime.golden42-v3-worker-request.v1".to_string(),
                    generation: 0,
                    slot,
                    candidate: candidate.clone(),
                    context_sha256: context_hash(&release_sha256, candidate),
                };
                serde_json::to_writer(&mut workers[slot].stdin, &request)?;
                workers[slot].stdin.write_all(b"\n")?;
                workers[slot].stdin.flush()?;
            }
            let mut completion_order = Vec::with_capacity(WORKERS);
            let mut worker_ids = BTreeSet::new();
            while completion_order.len() < WORKERS {
                match receiver.recv_timeout(Duration::from_secs(1)) {
                    Ok((worker_id, response)) => {
                        let response = response?;
                        if response.worker_id != worker_id
                            || response.slot >= population.len()
                            || response.generation != 0
                            || response.semantic_hash != population[response.slot].semantic_hash
                            || response.context_sha256
                                != context_hash(&release_sha256, &population[response.slot])
                            || response.predictions.len() != test.y.len()
                            || response.prediction_sha256 != prediction_hash(&response.predictions)
                        {
                            bail!("V3 smoke worker returned an invalid response");
                        }
                        validate_codec_artifact(
                            &root,
                            &response.codec_file,
                            &response.codec_sha256,
                        )?;
                        let rmse = raw_rmse(&test.y, &response.predictions)?;
                        worker_ids.insert(worker_id);
                        completion_order.push(V3ScoredCandidate {
                            candidate: population[response.slot].clone(),
                            rmse,
                            novelty: 1.0,
                        });
                    }
                    Err(RecvTimeoutError::Timeout) => {
                        if started.elapsed() > Duration::from_secs(3_600) {
                            bail!("V3 80-candidate smoke generation timed out");
                        }
                        for worker in &mut workers {
                            if let Some(status) = worker.child.try_wait()? {
                                bail!("V3 smoke worker {} exited early with {status}", worker.id);
                            }
                        }
                    }
                    Err(RecvTimeoutError::Disconnected) => {
                        bail!("V3 smoke worker response channels disconnected")
                    }
                }
            }
            let next_completion = next_generation(&config, 1, &completion_order, &[0; 4])?;
            let mut ranked = completion_order.clone();
            ranked.sort_by(scored_order);
            let next_ranked = next_generation(&config, 1, &ranked, &[0; 4])?;
            if next_completion != next_ranked {
                bail!("V3 next generation changed with 80-worker completion order");
            }
            let ranking_sha256 = sha256_bytes(
                ranked
                    .iter()
                    .flat_map(|item| {
                        [
                            item.candidate.semantic_hash.as_bytes(),
                            &item.rmse.to_bits().to_le_bytes(),
                        ]
                        .concat()
                    })
                    .collect::<Vec<_>>()
                    .as_slice(),
            );
            let best_rmse = ranked
                .iter()
                .map(|item| item.rmse)
                .min_by(f64::total_cmp)
                .context("empty V3 smoke ranking")?;
            Ok(SmokeReceipt {
                candidates: completion_order.len(),
                worker_processes: workers.len(),
                distinct_worker_ids: worker_ids.len(),
                cpp_only: true,
                ranking_sha256,
                next_population_sha256: sha256_bytes(&json_bytes(&next_ranked)?),
                completion_order_independent: true,
                best_rmse,
                elapsed_ms: started.elapsed().as_millis(),
            })
        })();
        terminate_workers(&mut workers);
        smoke_result
    })();
    let cleanup = fs::remove_dir_all(&root);
    match (result, cleanup) {
        (Ok(receipt), Ok(())) => Ok(receipt),
        (Err(error), _) => Err(error),
        (Ok(_), Err(error)) => Err(error.into()),
    }
}
