use super::*;

pub(super) fn campaign_root(run_id: &str, dataset: &str) -> PathBuf {
    run_root(run_id).join(dataset)
}

pub(super) fn generation_root(root: &Path, generation: usize) -> PathBuf {
    root.join("generations")
        .join(format!("generation-{generation:06}"))
}

pub(super) fn immutable_or_equal(path: &Path, bytes: &[u8]) -> Result<()> {
    if path.exists() {
        if fs::read(path)? != bytes {
            bail!("immutable artifact collision at {}", path.display());
        }
        return Ok(());
    }
    immutable_write(path, bytes)
}

pub(super) fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<String> {
    let bytes = canonical_json(value)?;
    immutable_or_equal(path, &bytes)?;
    Ok(sha256_bytes(&bytes))
}

pub(super) fn load_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    serde_json::from_slice(&fs::read(path)?).with_context(|| format!("parse {}", path.display()))
}

pub(super) fn read_cohort() -> Result<(FrozenCohort, String)> {
    let manifest: FrozenCohort =
        serde_json::from_slice(COHORT_BYTES).context("parse frozen Binary42 cohort")?;
    if manifest.schema != "JopeBinary42FrozenCohort.v1"
        || !manifest.test_adaptive
        || manifest.benchmark_valid
        || manifest.track_id != "tabular-binary-v1"
        || manifest.datasets.len() != 42
        || manifest
            .datasets
            .iter()
            .enumerate()
            .any(|(order, dataset)| dataset.order != order)
        || manifest
            .datasets
            .iter()
            .filter(|dataset| dataset.leader_auc == 1.0)
            .count()
            != 6
        || manifest.datasets.iter().any(|dataset| {
            if dataset.leader_auc.to_bits() == 1.0f64.to_bits() {
                dataset.win_policy != "exact_auc_1_tie"
            } else {
                dataset.win_policy != "strictly_greater_auc"
            }
        })
    {
        bail!("frozen Binary42 cohort contract mismatch");
    }
    let ids = manifest
        .datasets
        .iter()
        .map(|dataset| dataset.id.as_str())
        .collect::<BTreeSet<_>>();
    if ids.len() != 42 {
        bail!("frozen Binary42 cohort contains duplicate datasets");
    }
    let ilpd = manifest
        .datasets
        .iter()
        .find(|dataset| dataset.id == ILPD)
        .context("frozen Binary42 cohort omits ILPD")?;
    if ilpd.train_rows != 485
        || ilpd.test_rows != 98
        || ilpd.features != 10
        || ilpd.train_sha256 != "f2a030306747738f28cb60c9f2f3fe1591724427ee63cb9d0bf8fb43b75cc114"
        || ilpd.test_sha256 != "0b86467f1692851213f7adbf6194e91c2eda39ff3e340a4ef81805b53e388aac"
        || ilpd.leader_method != "pytorch_tabular_fttransformer_epochs5"
        || ilpd.leader_auc.to_bits() != 0.7681159420289855f64.to_bits()
        || ilpd.leader_report_sha256
            != "5018f76223d4445f87960a076c9766bfe4140bb43d8b8bf9ae68ebb5ac85bad4"
    {
        bail!("frozen ILPD binding differs from the pilot contract");
    }
    Ok((manifest, sha256_bytes(COHORT_BYTES)))
}

pub(super) fn selected_dataset(dataset: &str) -> Result<(FrozenDataset, String)> {
    if dataset != ILPD {
        bail!("the Binary42 pilot is frozen to ILPD ({ILPD}); no other dataset may launch");
    }
    let (cohort, sha256) = read_cohort()?;
    Ok((
        cohort
            .datasets
            .into_iter()
            .find(|record| record.id == dataset)
            .context("selected dataset is absent from Binary42")?,
        sha256,
    ))
}

pub(super) fn cohort_dataset(dataset: &str) -> Result<(FrozenDataset, String)> {
    let (cohort, sha256) = read_cohort()?;
    Ok((
        cohort
            .datasets
            .into_iter()
            .find(|record| record.id == dataset)
            .with_context(|| format!("dataset {dataset} is absent from frozen Binary42"))?,
        sha256,
    ))
}

pub(super) fn admission_order(cohort: &FrozenCohort) -> Vec<FrozenDataset> {
    let mut datasets = cohort.datasets.clone();
    datasets.sort_by(|left, right| {
        let left_cost = (left.train_rows as u128) * (left.features as u128);
        let right_cost = (right.train_rows as u128) * (right.features as u128);
        right_cost
            .cmp(&left_cost)
            .then_with(|| left.order.cmp(&right.order))
    });
    datasets
}

pub(super) fn verify_dataset_binding(spec: &FrozenDataset) -> Result<()> {
    let train_path = Path::new(&spec.train_path);
    let test_path = Path::new(&spec.test_path);
    let report_path = Path::new(&spec.leader_report_path);
    if sha256_file(train_path)? != spec.train_sha256
        || sha256_file(test_path)? != spec.test_sha256
        || sha256_file(report_path)? != spec.leader_report_sha256
    {
        bail!(
            "Binary42 dataset or leader report hash mismatch for {}",
            spec.id
        );
    }
    let train = read_binary_csv(train_path)?;
    let test = read_binary_csv(test_path)?;
    if train.x.len() != spec.train_rows
        || test.x.len() != spec.test_rows
        || train.features != spec.features
        || test.features != spec.features
    {
        bail!("Binary42 dataset shape mismatch for {}", spec.id);
    }
    Ok(())
}

pub(super) fn seed_population(features: usize) -> Result<Vec<MemberRecord>> {
    seed_population_with_winners(features, &[])
}

pub(super) fn winner_seed_snapshot_sha256(sources: &[WinnerSeedSource]) -> Result<String> {
    Ok(sha256_bytes(&canonical_json(&sources)?))
}

pub(super) fn seed_population_with_winners(
    features: usize,
    winner_sources: &[WinnerSeedSource],
) -> Result<Vec<MemberRecord>> {
    if sha256_bytes(BINARY_SEEDS) != SEED_BINARY_SHA256 {
        bail!("canonical binary seed hash mismatch");
    }
    if winner_sources.len() > POPULATION {
        bail!("Binary42 confirmed winner seed bank exceeds the population");
    }
    let base = test_adaptive_seed_population(BINARY_SEEDS, features)?;
    let mut seen = BTreeSet::new();
    let mut genomes = Vec::with_capacity(POPULATION);
    for source in winner_sources {
        if sha256_bytes(source.canonical_genome.as_bytes()) != source.semantic_sha256 {
            bail!("Binary42 winner seed semantic identity changed");
        }
        if !seen.insert(source.semantic_sha256.clone()) {
            bail!("Binary42 confirmed winner seed bank contains duplicate semantics");
        }
        genomes.push(source.canonical_genome.clone());
    }
    for member in base {
        if seen.insert(member.semantic_sha256) {
            genomes.push(member.canonical_genome);
        }
        if genomes.len() == POPULATION {
            break;
        }
    }
    if genomes.len() != POPULATION {
        bail!("Binary42 winner-augmented seed bank produced a short population");
    }
    let bytes = format!("{}\n", genomes.join("\n")).into_bytes();
    let members = test_adaptive_seed_population(&bytes, features)?
        .into_iter()
        .map(MemberRecord::from_binary)
        .collect::<Result<Vec<_>>>()?;
    if winner_sources
        .iter()
        .zip(&members)
        .any(|(source, member)| source.semantic_sha256 != member.semantic_sha256)
    {
        bail!("Binary42 confirmed winners are not first in generation zero");
    }
    Ok(members)
}

pub(super) fn rank_slots(left: &LogicalSlot, right: &LogicalSlot) -> Ordering {
    let left_score = left.score.as_ref().expect("ranked slot has score");
    let right_score = right.score.as_ref().expect("ranked slot has score");
    right_score
        .auc
        .total_cmp(&left_score.auc)
        .then_with(|| left_score.logloss.total_cmp(&right_score.logloss))
        .then_with(|| left.member.complexity.cmp(&right.member.complexity))
        .then_with(|| {
            left.member
                .semantic_sha256
                .cmp(&right.member.semantic_sha256)
        })
}

pub(super) fn strict_win(
    candidate_auc: f64,
    leader_auc: f64,
    generation_zero_best_auc: f64,
) -> bool {
    candidate_auc > leader_auc && candidate_auc > generation_zero_best_auc
}

pub(super) fn policy_win(
    win_policy: &str,
    candidate_auc: f64,
    leader_auc: f64,
    generation_zero_best_auc: f64,
) -> bool {
    match win_policy {
        "strictly_greater_auc" => strict_win(candidate_auc, leader_auc, generation_zero_best_auc),
        "exact_auc_1_tie" => candidate_auc.to_bits() == 1.0f64.to_bits(),
        _ => false,
    }
}

pub(super) fn physical_hostname(node: &str) -> Result<&'static str> {
    NODES
        .iter()
        .position(|candidate| *candidate == node)
        .map(|index| PHYSICAL_HOSTNAMES[index])
        .with_context(|| format!("unknown Binary42 SCQ node {node}"))
}

#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn cohort_win(candidate_auc: f64, leader_auc: f64) -> bool {
    if leader_auc == 1.0 {
        candidate_auc == 1.0
    } else {
        candidate_auc > leader_auc
    }
}

#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn qualifying_winner(
    generation: usize,
    ranked: &[LogicalSlot],
    seed_semantics: &BTreeSet<String>,
    leader_auc: f64,
    generation_zero_best_auc: f64,
) -> Option<QualifyingWinner> {
    qualifying_winner_for_policy(
        generation,
        ranked,
        seed_semantics,
        "strictly_greater_auc",
        leader_auc,
        generation_zero_best_auc,
    )
}

pub(super) fn qualifying_winner_for_policy(
    generation: usize,
    ranked: &[LogicalSlot],
    seed_semantics: &BTreeSet<String>,
    win_policy: &str,
    leader_auc: f64,
    generation_zero_best_auc: f64,
) -> Option<QualifyingWinner> {
    if generation == 0 {
        return None;
    }
    ranked.iter().find_map(|slot| {
        let score = slot.score.as_ref()?;
        (!seed_semantics.contains(&slot.member.semantic_sha256)
            && matches!(
                slot.member.ancestry.operation.as_str(),
                "mutation" | "crossover" | "elite"
            )
            && policy_win(win_policy, score.auc, leader_auc, generation_zero_best_auc))
        .then(|| QualifyingWinner {
            generation,
            slot: slot.slot,
            semantic_sha256: slot.member.semantic_sha256.clone(),
            auc: score.auc,
            logloss: score.logloss,
            complexity: slot.member.complexity,
        })
    })
}

#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn shard_members(members: &[MemberRecord]) -> Result<Vec<Vec<MemberRecord>>> {
    shard_members_with_sizes(members, &[SHARD_SIZE; 4])
}

pub(super) fn shard_members_with_sizes(
    members: &[MemberRecord],
    shard_sizes: &[usize],
) -> Result<Vec<Vec<MemberRecord>>> {
    if members.len() != POPULATION {
        bail!("Binary42 generation requires exactly {POPULATION} members");
    }
    if shard_sizes.len() != NODES.len() || shard_sizes.iter().sum::<usize>() != POPULATION {
        bail!("Binary42 shard sizes do not account for the 60-member population");
    }
    let mut offset = 0;
    let mut shards = Vec::with_capacity(shard_sizes.len());
    for size in shard_sizes {
        shards.push(members[offset..offset + size].to_vec());
        offset += size;
    }
    Ok(shards)
}

pub(super) fn campaign_shard_sizes(config: &CampaignConfig) -> Vec<usize> {
    if config.shard_sizes.is_empty() {
        vec![SHARD_SIZE; NODES.len()]
    } else {
        config.shard_sizes.clone()
    }
}

pub(super) fn shard_start(shard_sizes: &[usize], shard: usize) -> Result<usize> {
    if shard >= shard_sizes.len() {
        bail!("Binary42 shard index is outside the configured partition");
    }
    Ok(shard_sizes[..shard].iter().sum())
}

pub(super) fn assigned_node_for_slot(shard_sizes: &[usize], slot: usize) -> Result<&'static str> {
    let mut end = 0;
    for (index, size) in shard_sizes.iter().enumerate() {
        end += size;
        if slot < end {
            return Ok(NODES[index]);
        }
    }
    bail!("Binary42 slot {slot} is outside the configured shard partition")
}

pub(super) fn generation_cap_reached(config: &CampaignConfig, generation: usize) -> bool {
    config
        .generation_cap
        .is_some_and(|cap| generation + 1 >= cap)
}

pub(super) fn failover_node(shard: usize, attempt: usize) -> Result<&'static str> {
    if shard >= NODES.len() || attempt > 2 {
        bail!("Binary42 failover index is outside the immutable policy");
    }
    Ok(NODES[(shard + attempt) % NODES.len()])
}

pub(super) fn binary_prediction_bytes(predictions: &[f64]) -> Vec<u8> {
    let mut bytes = b"prediction\n".to_vec();
    for prediction in predictions {
        bytes.extend_from_slice(format!("{prediction:.17}\n").as_bytes());
    }
    bytes
}

pub(super) fn binary_prediction_roundtrip_bytes(predictions: &[f64]) -> Vec<u8> {
    let mut bytes = b"prediction\n".to_vec();
    for prediction in predictions {
        bytes.extend_from_slice(format!("{prediction:?}\n").as_bytes());
    }
    bytes
}

pub(super) fn parse_prediction_bytes(bytes: &[u8]) -> Result<Vec<f64>> {
    let text = std::str::from_utf8(bytes).context("prediction object is not UTF-8")?;
    let mut lines = text.lines();
    if lines.next() != Some("prediction") {
        bail!("prediction object has the wrong header");
    }
    lines
        .enumerate()
        .map(|(index, line)| {
            let value = line
                .parse::<f64>()
                .with_context(|| format!("prediction row {} is not numeric", index + 1))?;
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                bail!("prediction row {} is outside [0, 1]", index + 1);
            }
            Ok(value)
        })
        .collect()
}

pub(super) fn candidate_file(generation: usize, slot: usize) -> String {
    format!("candidate-g{generation:06}-s{slot:02}.receipt.v1.json")
}

pub(super) fn prediction_file(generation: usize, slot: usize) -> String {
    format!("candidate-g{generation:06}-s{slot:02}.predictions.csv")
}

pub(super) fn expected_job_id(manifest: &ShardManifest) -> Result<String> {
    let mut binding = manifest.clone();
    binding.job_id.clear();
    Ok(format!(
        "jope-b42-{}",
        &sha256_bytes(&canonical_json(&binding)?)[..24]
    ))
}

pub(super) fn validate_shard_manifest(manifest: &ShardManifest) -> Result<()> {
    let expected_slots = if manifest.kind == "generation" {
        if manifest.cohort_mode.as_deref() == Some("all") {
            *ALL_SHARD_SIZES
                .get(manifest.shard)
                .context("Binary42 all-dataset shard index is invalid")?
        } else {
            SHARD_SIZE
        }
    } else if manifest.kind == "confirmation" {
        1
    } else {
        bail!("unknown Binary42 shard kind");
    };
    let expected_cpus = expected_slots;
    let hostname = String::from_utf8(
        scq_batch::command_output(
            {
                let mut command = Command::new("hostname");
                command.arg("-s");
                command
            },
            "hostname",
        )?
        .stdout,
    )?
    .trim()
    .to_string();
    let expected_hostname = physical_hostname(&manifest.node)?;
    let (dataset, _) = cohort_dataset(&manifest.dataset)?;
    if manifest.schema != SHARD_SCHEMA
        || !manifest.test_adaptive
        || manifest.benchmark_valid
        || (manifest.cohort_mode.is_none() && manifest.dataset != ILPD)
        || (manifest.cohort_mode.is_some() && manifest.cohort_mode.as_deref() != Some("all"))
        || !NODES.contains(&manifest.node.as_str())
        || hostname != expected_hostname
        || manifest.slots.len() != expected_slots
        || manifest.cpus_reserved != expected_cpus
        || manifest.gpus_per_task != 0
        || manifest.candidate_threads != 1
        || manifest.job_id != expected_job_id(manifest)?
        || current_binary_sha256()? != manifest.release_binary_sha256
    {
        bail!("Binary42 shard identity or resource contract mismatch");
    }
    if manifest.train_sha256 != dataset.train_sha256
        || manifest.test_sha256 != dataset.test_sha256
        || manifest.train_rows != dataset.train_rows
        || manifest.test_rows != dataset.test_rows
        || manifest.features != dataset.features
        || (manifest.cohort_mode.as_deref() == Some("all")
            && manifest.scq_binary_path.as_deref() != Some(all_scq_binary_str()))
        || (manifest.cohort_mode.is_none() && manifest.scq_binary_path.is_some())
    {
        bail!("Binary42 shard dataset binding mismatch");
    }
    let mut slot_ids = BTreeSet::new();
    let mut semantics = BTreeSet::new();
    for slot in &manifest.slots {
        if slot.generation != manifest.generation
            || slot.member.semantic_sha256 != sha256_bytes(slot.member.canonical_genome.as_bytes())
            || !slot_ids.insert(slot.slot)
            || !semantics.insert(slot.member.semantic_sha256.clone())
        {
            bail!("Binary42 shard contains an invalid or duplicate slot");
        }
    }
    if manifest.kind == "confirmation" && manifest.slots.iter().any(|slot| !slot.fresh) {
        bail!("Binary42 confirmations must be independently fresh");
    }
    Ok(())
}

pub(super) fn failed_candidate_receipt(
    manifest: &ShardManifest,
    slot: &ShardSlot,
    hostname: &str,
    cpu: usize,
    runtime_seconds: f64,
    backend: Option<CandidateBackendReceipt>,
    failure: String,
) -> CandidateReceipt {
    let empty = binary_prediction_bytes(&[]);
    CandidateReceipt {
        schema: CANDIDATE_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        kind: manifest.kind.clone(),
        run_id: manifest.run_id.clone(),
        dataset: manifest.dataset.clone(),
        generation: manifest.generation,
        slot: slot.slot,
        attempt: manifest.attempt,
        semantic_sha256: slot.member.semantic_sha256.clone(),
        canonical_genome: slot.member.canonical_genome.clone(),
        complexity: slot.member.complexity,
        release_binary_sha256: manifest.release_binary_sha256.clone(),
        train_sha256: manifest.train_sha256.clone(),
        test_sha256: manifest.test_sha256.clone(),
        train_rows: manifest.train_rows,
        test_rows: manifest.test_rows,
        features: manifest.features,
        node: manifest.node.clone(),
        hostname: hostname.to_string(),
        scq_job_id: manifest.job_id.clone(),
        candidate_threads: 1,
        cpu,
        runtime_seconds,
        fit_seconds: None,
        predict_seconds: None,
        backend,
        predictions_sha256: sha256_bytes(&empty),
        score: None,
        failure: Some(failure),
    }
}

pub(super) fn cpp17_toolchain() -> Result<CandidateBackendReceipt> {
    let output = scq_batch::command_output(
        {
            let mut command = Command::new("g++");
            command.arg("--version");
            command
        },
        "Binary42 fitted C++17 compiler identity",
    )?;
    let compiler = String::from_utf8(output.stdout)?
        .lines()
        .next()
        .unwrap_or_default()
        .to_string();
    if !compiler.contains("13.3") {
        bail!("Binary42 fitted C++17 requires g++ 13.3, found {compiler:?}");
    }
    Ok(CandidateBackendReceipt {
        name: CPP17_BACKEND.to_string(),
        language_standard: CPP17_STANDARD.to_string(),
        compiler,
        compiler_flags: CPP17_FLAGS.iter().map(|flag| (*flag).to_string()).collect(),
        source_sha256: None,
        codec_sha256: None,
        compile_seconds: None,
    })
}

pub(super) fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub(super) fn validate_candidate_backend(
    receipt: &CandidateReceipt,
    all_dataset_cohort: bool,
) -> Result<()> {
    match (&receipt.backend, all_dataset_cohort) {
        (None, false) => Ok(()),
        (Some(backend), true) => {
            if backend.name != CPP17_BACKEND
                || backend.language_standard != CPP17_STANDARD
                || !backend.compiler.contains("13.3")
                || backend.compiler_flags
                    != CPP17_FLAGS
                        .iter()
                        .map(|flag| (*flag).to_string())
                        .collect::<Vec<_>>()
            {
                bail!("Binary42 fitted C++17 backend identity mismatch");
            }
            match (&receipt.score, &receipt.failure) {
                (Some(_), None)
                    if backend.source_sha256.as_deref().is_some_and(is_sha256_hex)
                        && backend.codec_sha256.as_deref().is_some_and(is_sha256_hex)
                        && backend
                            .compile_seconds
                            .is_some_and(|seconds| seconds >= 0.0) =>
                {
                    Ok(())
                }
                (None, Some(_))
                    if backend.source_sha256.is_none()
                        && backend.codec_sha256.is_none()
                        && backend.compile_seconds.is_none() =>
                {
                    Ok(())
                }
                _ => bail!("Binary42 fitted C++17 build provenance is incomplete"),
            }
        }
        _ => bail!("Binary42 candidate used the wrong execution backend"),
    }
}

pub(super) fn evaluate_worker_candidate(
    manifest: &ShardManifest,
    slot: &ShardSlot,
    train: &RegressionData,
    test: &RegressionData,
    cpu: usize,
    cpp17_toolchain: Option<&CandidateBackendReceipt>,
    runtime_root: &Path,
) -> (CandidateReceipt, Vec<u8>) {
    let started = Instant::now();
    let all_dataset_cohort = manifest.cohort_mode.as_deref() == Some("all");
    let evaluated = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if all_dataset_cohort {
            let built = evaluate_test_adaptive_binary_cpp17(
                &slot.member.canonical_genome,
                train,
                test,
                runtime_root,
            )?;
            let mut backend = cpp17_toolchain
                .context("Binary42 fitted C++17 toolchain was not initialized")?
                .clone();
            backend.source_sha256 = Some(built.source_sha256);
            backend.codec_sha256 = Some(built.codec_sha256);
            backend.compile_seconds = Some(built.compile_seconds);
            Ok((built.evaluation, Some(backend)))
        } else {
            evaluate_test_adaptive_binary(&slot.member.canonical_genome, train, test)
                .map(|evaluation| (evaluation, None))
        }
    }));
    match evaluated {
        Ok(Ok((result, backend))) => {
            let predictions = if all_dataset_cohort {
                binary_prediction_roundtrip_bytes(&result.predictions)
            } else {
                binary_prediction_bytes(&result.predictions)
            };
            (
                CandidateReceipt {
                    schema: CANDIDATE_SCHEMA.to_string(),
                    test_adaptive: true,
                    benchmark_valid: false,
                    kind: manifest.kind.clone(),
                    run_id: manifest.run_id.clone(),
                    dataset: manifest.dataset.clone(),
                    generation: manifest.generation,
                    slot: slot.slot,
                    attempt: manifest.attempt,
                    semantic_sha256: slot.member.semantic_sha256.clone(),
                    canonical_genome: slot.member.canonical_genome.clone(),
                    complexity: result.complexity,
                    release_binary_sha256: manifest.release_binary_sha256.clone(),
                    train_sha256: manifest.train_sha256.clone(),
                    test_sha256: manifest.test_sha256.clone(),
                    train_rows: manifest.train_rows,
                    test_rows: manifest.test_rows,
                    features: manifest.features,
                    node: manifest.node.clone(),
                    hostname: manifest.node.clone(),
                    scq_job_id: manifest.job_id.clone(),
                    candidate_threads: 1,
                    cpu,
                    runtime_seconds: started.elapsed().as_secs_f64(),
                    fit_seconds: Some(result.fit_seconds),
                    predict_seconds: Some(result.predict_seconds),
                    backend,
                    predictions_sha256: sha256_bytes(&predictions),
                    score: Some(CandidateScore {
                        auc: result.auc,
                        logloss: result.logloss,
                    }),
                    failure: None,
                },
                predictions,
            )
        }
        Ok(Err(error)) => {
            let predictions = binary_prediction_bytes(&[]);
            (
                failed_candidate_receipt(
                    manifest,
                    slot,
                    &manifest.node,
                    cpu,
                    started.elapsed().as_secs_f64(),
                    cpp17_toolchain.cloned(),
                    format!("{error:#}"),
                ),
                predictions,
            )
        }
        Err(_) => {
            let predictions = binary_prediction_bytes(&[]);
            (
                failed_candidate_receipt(
                    manifest,
                    slot,
                    &manifest.node,
                    cpu,
                    started.elapsed().as_secs_f64(),
                    cpp17_toolchain.cloned(),
                    "candidate panicked".to_string(),
                ),
                predictions,
            )
        }
    }
}
