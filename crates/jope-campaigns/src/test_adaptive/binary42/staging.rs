use super::*;

pub(super) fn remote_bundle_root(node: &str, release_binary_sha256: &str) -> Result<String> {
    if !NODES.contains(&node) {
        bail!("unknown Binary42 SCQ node {node}");
    }
    Ok(format!(
        "{linux_remote_root_str}/bundles/{release_binary_sha256}",
        linux_remote_root_str = linux_remote_root_str()
    ))
}

pub(super) fn controller_hostname() -> Result<String> {
    Ok(String::from_utf8(
        scq_batch::command_output(
            {
                let mut command = Command::new("hostname");
                command.arg("-s");
                command
            },
            "controller hostname",
        )?
        .stdout,
    )?
    .trim()
    .to_string())
}

pub(super) fn is_controller_node(node: &str) -> Result<bool> {
    Ok(controller_hostname()? == physical_hostname(node)?)
}

pub(super) fn validate_local_immutable(
    path: &Path,
    expected_sha256: &str,
    expected_permissions: u32,
) -> Result<()> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("local immutable path is absent: {}", path.display()))?;
    if metadata.file_type().is_symlink()
        || !metadata.file_type().is_file()
        || metadata.permissions().mode() & 0o7777 != expected_permissions
    {
        bail!(
            "local immutable mode mismatch at {}: expected regular file with permissions {expected_permissions:#o}",
            path.display()
        );
    }
    let actual_sha256 = sha256_file(path)?;
    if actual_sha256 != expected_sha256 {
        bail!(
            "local immutable hash mismatch at {}: got {actual_sha256}, expected {expected_sha256}",
            path.display()
        );
    }
    Ok(())
}

pub(super) fn copy_local_atomic(source: &Path, destination: &Path, executable: bool) -> Result<()> {
    let expected_sha256 = sha256_file(source)?;
    let expected_permissions = if executable { 0o555 } else { 0o444 };
    if destination.exists() {
        return validate_local_immutable(destination, &expected_sha256, expected_permissions);
    }
    let parent = destination
        .parent()
        .context("local staging destination has no parent")?;
    fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(
        ".jope-stage.{}.{}.{}",
        std::process::id(),
        unix_millis(),
        &expected_sha256[..12]
    ));
    let mut source_file = fs::File::open(source)?;
    let mut temporary_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    std::io::copy(&mut source_file, &mut temporary_file)?;
    temporary_file.sync_all()?;
    fs::set_permissions(&temporary, fs::Permissions::from_mode(expected_permissions))?;
    validate_local_immutable(&temporary, &expected_sha256, expected_permissions)?;
    let publish = fs::hard_link(&temporary, destination);
    fs::remove_file(&temporary)?;
    if let Err(error) = publish {
        validate_local_immutable(destination, &expected_sha256, expected_permissions)
            .with_context(|| {
                format!(
                    "exclusive local publish failed at {}: {error}",
                    destination.display()
                )
            })?;
    }
    validate_local_immutable(destination, &expected_sha256, expected_permissions)
}

pub(super) fn submit_canary(node: &str, suffix: &str) -> Result<(String, String)> {
    let id = format!("jope-b42-canary-{node}-{suffix}");
    let spec = format!(
        "[job]\nid = {id:?}\nproject = {PROJECT:?}\ncommand = \"/bin/true\"\ntier = \"normal\"\n\n[resource]\ngpus_per_task = 0\ncpus_reserved = {SHARD_SIZE}\nomp_threads = 1\nram_gb = 0\ngang_size = 1\nplacement = \"single_node\"\n\n[[dependencies.smoke]]\nname = \"Binary42 C++17 compiler available\"\ncommand = \"/usr/bin/test\"\nargs = [\"-x\", \"/usr/bin/g++\"]\ntimeout_secs = 5\n\n[preempt]\npreemptible = false\nsignal = \"SIGTERM\"\ngrace_seconds = 60\n\n[limits]\nmax_runtime_hours = 1\nmax_restarts = 0\n"
    );
    let spec_path = std::env::temp_dir().join(format!("{id}.toml"));
    {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&spec_path)?;
        file.write_all(spec.as_bytes())?;
        file.sync_all()?;
    }
    let submitted = scq_batch::command_output(
        {
            let mut command = Command::new(scq_binary_str());
            command.arg("submit").arg(&spec_path);
            command
        },
        "Binary42 pinned SCQ canary",
    );
    fs::remove_file(&spec_path)?;
    submitted?;
    let show = scq_batch::wait_for_job(&id)?;
    let job = scq_batch::scq_job(&show, &id)?;
    if job.get("state").and_then(Value::as_str) != Some("completed")
        || job.get("exit_code").and_then(Value::as_i64) != Some(0)
        || job.get("cpus").and_then(Value::as_u64) != Some(SHARD_SIZE as u64)
    {
        bail!("Binary42 pinned SCQ canary failed its node/resource contract");
    }
    Ok((id, sha256_bytes(&canonical_json(&show)?)))
}

pub(super) fn submit_canary_at(
    scq_binary: &str,
    node: &str,
    suffix: &str,
    cpus_reserved: usize,
) -> Result<(String, String)> {
    let id = format!("jope-b42-all-canary-{node}-{suffix}");
    let spec = format!(
        "[job]\nid = {id:?}\nproject = {PROJECT:?}\ncommand = \"/bin/true\"\ntier = \"normal\"\n\n[resource]\ngpus_per_task = 0\ncpus_reserved = {cpus_reserved}\nomp_threads = 1\nram_gb = 0\ngang_size = 1\nplacement = \"single_node\"\n\n[[dependencies.smoke]]\nname = \"Binary42 C++17 compiler available\"\ncommand = \"/usr/bin/test\"\nargs = [\"-x\", \"/usr/bin/g++\"]\ntimeout_secs = 5\n\n[preempt]\npreemptible = false\nsignal = \"SIGTERM\"\ngrace_seconds = 60\n\n[limits]\nmax_runtime_hours = 1\nmax_restarts = 0\n"
    );
    let spec_path = std::env::temp_dir().join(format!("{id}.toml"));
    {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&spec_path)?;
        file.write_all(spec.as_bytes())?;
        file.sync_all()?;
    }
    let submitted = scq_batch::command_output(
        {
            let mut command = Command::new(scq_binary);
            command.arg("submit").arg(&spec_path);
            command
        },
        "Binary42 all-dataset aggregate pinned SCQ canary",
    );
    fs::remove_file(&spec_path)?;
    submitted?;
    let show = wait_for_job_at(scq_binary, &id)?;
    let job = scq_batch::scq_job(&show, &id)?;
    if job.get("state").and_then(Value::as_str) != Some("completed")
        || job.get("exit_code").and_then(Value::as_i64) != Some(0)
        || job.get("cpus").and_then(Value::as_u64) != Some(cpus_reserved as u64)
    {
        bail!("Binary42 aggregate pinned SCQ canary failed its node/resource contract");
    }
    Ok((id, sha256_bytes(&canonical_json(&show)?)))
}

pub(super) fn node_command_output(
    node: &str,
    args: &[&str],
    label: &str,
) -> Result<std::process::Output> {
    if is_controller_node(node)? {
        let mut command = Command::new(args.first().context("empty local node command")?);
        command.args(&args[1..]);
        scq_batch::command_output(command, label)
    } else {
        scq_batch::ssh_output(node, args, label)
    }
}

pub(super) fn validate_all_staging(
    staging: &AllStagingReceipt,
    cohort: &FrozenCohort,
    release_binary_sha256: &str,
    scq_resolved_path: &str,
    scq_binary_sha256: &str,
    verify_remote: bool,
) -> Result<()> {
    if staging.schema != ALL_STAGING_SCHEMA
        || !staging.test_adaptive
        || staging.benchmark_valid
        || staging.release_binary_sha256 != release_binary_sha256
        || staging.scq_configured_path != all_scq_binary_str()
        || staging.scq_resolved_path != scq_resolved_path
        || staging.scq_binary_sha256 != scq_binary_sha256
        || staging.nodes.len() != NODES.len()
    {
        bail!("Binary42 all-dataset staging identity mismatch");
    }
    let budget = AllCohortConfig::fixed().node_cpu_budget_per_active_set;
    for (node_index, node) in staging.nodes.iter().enumerate() {
        let expected_budget = budget
            .get(NODES[node_index])
            .copied()
            .context("Binary42 node CPU budget is absent")?;
        if node.node != NODES[node_index]
            || node.hostname != PHYSICAL_HOSTNAMES[node_index]
            || node.logical_cpus < expected_budget
            || !node.cpp17_compiler.contains("13.3")
            || node.canary_cpus_reserved != expected_budget
            || node.binary_sha256 != release_binary_sha256
            || node.bundle_root != remote_bundle_root(&node.node, release_binary_sha256)?
            || node.datasets.len() != cohort.datasets.len()
        {
            bail!("Binary42 all-dataset staged node binding mismatch");
        }
        for (dataset_index, staged) in node.datasets.iter().enumerate() {
            let spec = &cohort.datasets[dataset_index];
            if staged.dataset != spec.id
                || staged.train_sha256 != spec.train_sha256
                || staged.test_sha256 != spec.test_sha256
            {
                bail!("Binary42 all-dataset staged data binding mismatch");
            }
            if verify_remote {
                if is_controller_node(&node.node)? {
                    validate_local_immutable(
                        Path::new(&staged.train_path),
                        &staged.train_sha256,
                        0o444,
                    )?;
                    validate_local_immutable(
                        Path::new(&staged.test_path),
                        &staged.test_sha256,
                        0o444,
                    )?;
                } else {
                    scq_batch::validate_remote_immutable(
                        &node.node,
                        &staged.train_path,
                        &staged.train_sha256,
                        0o444,
                    )?;
                    scq_batch::validate_remote_immutable(
                        &node.node,
                        &staged.test_path,
                        &staged.test_sha256,
                        0o444,
                    )?;
                }
            }
        }
        if verify_remote {
            if is_controller_node(&node.node)? {
                validate_local_immutable(Path::new(&node.binary_path), &node.binary_sha256, 0o555)?;
            } else {
                scq_batch::validate_remote_immutable(
                    &node.node,
                    &node.binary_path,
                    &node.binary_sha256,
                    0o555,
                )?;
            }
        }
    }
    Ok(())
}

pub(super) fn stage_all_campaign(
    cohort_root: &Path,
    cohort: &FrozenCohort,
    release_binary_sha256: &str,
    scq_resolved_path: &str,
    scq_binary_sha256: &str,
) -> Result<(AllStagingReceipt, String)> {
    let path = cohort_root.join("staging.v1.json");
    if path.exists() {
        let bytes = fs::read(&path)?;
        let staging: AllStagingReceipt = serde_json::from_slice(&bytes)?;
        if bytes != canonical_json(&staging)? {
            bail!("Binary42 all-dataset staging receipt is not canonical JSON");
        }
        validate_all_staging(
            &staging,
            cohort,
            release_binary_sha256,
            scq_resolved_path,
            scq_binary_sha256,
            true,
        )?;
        return Ok((staging, sha256_bytes(&bytes)));
    }
    if controller_hostname()? != "xbabe2" {
        bail!("Binary42 all-dataset controller must run on xbabe2");
    }
    let release_binary = std::env::current_exe()?;
    if sha256_file(&release_binary)? != release_binary_sha256 {
        bail!("release binary changed before Binary42 all-dataset staging");
    }
    let doctor = scq_batch::command_output(
        {
            let mut command = Command::new(all_scq_binary_str());
            command.arg("doctor");
            command
        },
        "Binary42 all-dataset SCQ doctor",
    )?;
    let doctor_output_sha256 = sha256_bytes(&[doctor.stdout, doctor.stderr].concat());
    let suffix = format!("{}-{}", &release_binary_sha256[..8], unix_millis());
    let budgets = AllCohortConfig::fixed().node_cpu_budget_per_active_set;
    let canaries = Arc::new(Mutex::new(Vec::new()));
    thread::scope(|scope| {
        for node in NODES {
            let canaries = Arc::clone(&canaries);
            let suffix = suffix.clone();
            let cpus = budgets[node];
            scope.spawn(move || {
                let mut result = None;
                for attempt in 0..3 {
                    match submit_canary_at(
                        all_scq_binary_str(),
                        node,
                        &format!("{suffix}-a{attempt}"),
                        cpus,
                    ) {
                        Ok(receipt) => {
                            result = Some(Ok(receipt));
                            break;
                        }
                        Err(error) if attempt == 2 => {
                            let _ = record_cohort_event(
                                cohort_root,
                                "aggregate_canary_failed",
                                None,
                                Some(&format!("node={node}, attempt={attempt}, error={error:#}")),
                            );
                            result = Some(Err(error));
                        }
                        Err(error) => {
                            let _ = record_cohort_event(
                                cohort_root,
                                "aggregate_canary_retry",
                                None,
                                Some(&format!("node={node}, attempt={attempt}, error={error:#}")),
                            );
                        }
                    }
                }
                canaries
                    .lock()
                    .unwrap()
                    .push((node.to_string(), result.unwrap()));
            });
        }
    });
    let mut canaries = Arc::try_unwrap(canaries)
        .expect("Binary42 all-dataset canary references remain")
        .into_inner()
        .unwrap();
    canaries.sort_by_key(|(node, _)| {
        NODES
            .iter()
            .position(|candidate| candidate == node)
            .unwrap()
    });
    let mut nodes = Vec::new();
    for (node_index, node) in NODES.iter().copied().enumerate() {
        let (canary_job_id, canary_show_sha256) = canaries
            .get_mut(node_index)
            .context("Binary42 aggregate canary result is absent")?
            .1
            .as_ref()
            .map_err(|error| anyhow::anyhow!("{error:#}"))?
            .clone();
        let hostname = String::from_utf8(
            node_command_output(node, &["hostname", "-s"], "Binary42 node hostname")?.stdout,
        )?
        .trim()
        .to_string();
        if hostname != physical_hostname(node)? {
            bail!("Binary42 node {node} reports unexpected hostname {hostname}");
        }
        let logical_cpus = String::from_utf8(
            node_command_output(node, &["nproc"], "Binary42 node CPU inventory")?.stdout,
        )?
        .trim()
        .parse::<usize>()
        .context("parse Binary42 node CPU inventory")?;
        if logical_cpus < budgets[node] {
            bail!(
                "Binary42 node {node} exposes {logical_cpus} CPUs but cohort needs {}",
                budgets[node]
            );
        }
        let cpp17_compiler = String::from_utf8(
            node_command_output(
                node,
                &["g++", "--version"],
                "Binary42 node fitted C++17 compiler identity",
            )?
            .stdout,
        )?
        .lines()
        .next()
        .unwrap_or_default()
        .to_string();
        if !cpp17_compiler.contains("13.3") {
            bail!("Binary42 node {node} requires g++ 13.3, found {cpp17_compiler:?}");
        }
        let bundle_root = remote_bundle_root(node, release_binary_sha256)?;
        let binary_path = format!("{bundle_root}/jope-dime-regressor");
        if is_controller_node(node)? {
            copy_local_atomic(&release_binary, Path::new(&binary_path), true)?;
        } else {
            scq_batch::copy_atomic(node, &release_binary, &binary_path, true)?;
        }
        let mut datasets = Vec::with_capacity(cohort.datasets.len());
        for spec in &cohort.datasets {
            let dataset_root = format!("{bundle_root}/datasets/{}", spec.id);
            let train_path = format!("{dataset_root}/train.csv");
            let test_path = format!("{dataset_root}/test.csv");
            if is_controller_node(node)? {
                copy_local_atomic(Path::new(&spec.train_path), Path::new(&train_path), false)?;
                copy_local_atomic(Path::new(&spec.test_path), Path::new(&test_path), false)?;
            } else {
                scq_batch::copy_atomic(node, Path::new(&spec.train_path), &train_path, false)?;
                scq_batch::copy_atomic(node, Path::new(&spec.test_path), &test_path, false)?;
            }
            datasets.push(AllStagedDataset {
                dataset: spec.id.clone(),
                train_path,
                train_sha256: spec.train_sha256.clone(),
                test_path,
                test_sha256: spec.test_sha256.clone(),
            });
        }
        nodes.push(AllStagedNode {
            node: node.to_string(),
            hostname,
            logical_cpus,
            cpp17_compiler,
            bundle_root,
            binary_path,
            binary_sha256: release_binary_sha256.to_string(),
            canary_job_id,
            canary_show_sha256,
            canary_cpus_reserved: budgets[node],
            datasets,
        });
    }
    let staging = AllStagingReceipt {
        schema: ALL_STAGING_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        release_binary_sha256: release_binary_sha256.to_string(),
        scq_configured_path: all_scq_binary_str().to_string(),
        scq_resolved_path: scq_resolved_path.to_string(),
        scq_binary_sha256: scq_binary_sha256.to_string(),
        doctor_output_sha256,
        nodes,
    };
    let sha256 = write_json(&path, &staging)?;
    validate_all_staging(
        &staging,
        cohort,
        release_binary_sha256,
        scq_resolved_path,
        scq_binary_sha256,
        true,
    )?;
    Ok((staging, sha256))
}

pub(super) fn stage_campaign(
    root: &Path,
    spec: &FrozenDataset,
    release_binary_sha256: &str,
    scq_binary_sha256: &str,
) -> Result<StagingReceipt> {
    let staging_path = root.join("staging").join("staging.v1.json");
    if staging_path.exists() {
        let staging: StagingReceipt = load_json(&staging_path)?;
        validate_staging(
            &staging,
            spec,
            release_binary_sha256,
            scq_binary_sha256,
            true,
        )?;
        return Ok(staging);
    }
    let release_binary = std::env::current_exe()?;
    if sha256_file(&release_binary)? != release_binary_sha256 {
        bail!("release binary changed before Binary42 staging");
    }
    let doctor = scq_batch::command_output(
        {
            let mut command = Command::new(scq_binary_str());
            command.arg("doctor");
            command
        },
        "scq doctor",
    )?;
    let doctor_output_sha256 = sha256_bytes(&[doctor.stdout, doctor.stderr].concat());
    let local_hostname = controller_hostname()?;
    let suffix = format!("{}-{}", &release_binary_sha256[..8], unix_millis());
    let mut nodes = Vec::new();
    for node in NODES {
        let mut canary = None;
        for attempt in 0..3 {
            match submit_canary(node, &format!("{suffix}-a{attempt}")) {
                Ok(receipt) => {
                    canary = Some(receipt);
                    break;
                }
                Err(error) => eprintln!(
                    "Binary42 pinned canary attempt {attempt} failed on {node}: {error:#}"
                ),
            }
        }
        let (canary_job_id, canary_show_sha256) =
            canary.with_context(|| format!("Binary42 canary retries exhausted on {node}"))?;
        let hostname = if node == local_hostname {
            local_hostname.clone()
        } else {
            String::from_utf8(
                scq_batch::ssh_output(node, &["hostname", "-s"], "Binary42 remote hostname")?
                    .stdout,
            )?
            .trim()
            .to_string()
        };
        if hostname != physical_hostname(node)? {
            bail!("Binary42 SSH target {node} reports unexpected hostname {hostname}");
        }
        let bundle_root = remote_bundle_root(node, release_binary_sha256)?;
        let binary_path = format!("{bundle_root}/jope-dime-regressor");
        let dataset_root = format!("{bundle_root}/datasets/{}", spec.id);
        let train_path = format!("{dataset_root}/train.csv");
        let test_path = format!("{dataset_root}/test.csv");
        if node == local_hostname {
            copy_local_atomic(&release_binary, Path::new(&binary_path), true)?;
            copy_local_atomic(Path::new(&spec.train_path), Path::new(&train_path), false)?;
            copy_local_atomic(Path::new(&spec.test_path), Path::new(&test_path), false)?;
        } else {
            scq_batch::copy_atomic(node, &release_binary, &binary_path, true)?;
            scq_batch::copy_atomic(node, Path::new(&spec.train_path), &train_path, false)?;
            scq_batch::copy_atomic(node, Path::new(&spec.test_path), &test_path, false)?;
        }
        nodes.push(StagedNode {
            node: node.to_string(),
            hostname,
            bundle_root,
            binary_path,
            binary_sha256: release_binary_sha256.to_string(),
            train_path,
            train_sha256: spec.train_sha256.clone(),
            test_path,
            test_sha256: spec.test_sha256.clone(),
            canary_job_id,
            canary_show_sha256,
        });
    }
    let receipt = StagingReceipt {
        schema: STAGING_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        release_binary_sha256: release_binary_sha256.to_string(),
        scq_binary_sha256: scq_binary_sha256.to_string(),
        doctor_output_sha256,
        dataset: spec.id.clone(),
        nodes,
        cohort_staging_sha256: None,
    };
    write_json(&staging_path, &receipt)?;
    validate_staging(
        &receipt,
        spec,
        release_binary_sha256,
        scq_binary_sha256,
        true,
    )?;
    Ok(receipt)
}

pub(super) fn validate_staging(
    staging: &StagingReceipt,
    spec: &FrozenDataset,
    release_binary_sha256: &str,
    scq_binary_sha256: &str,
    verify_remote: bool,
) -> Result<()> {
    if staging.schema != STAGING_SCHEMA
        || !staging.test_adaptive
        || staging.benchmark_valid
        || staging.release_binary_sha256 != release_binary_sha256
        || staging.scq_binary_sha256 != scq_binary_sha256
        || staging.dataset != spec.id
        || staging.nodes.len() != NODES.len()
    {
        bail!("Binary42 staging receipt identity mismatch");
    }
    for (index, node) in staging.nodes.iter().enumerate() {
        if node.node != NODES[index]
            || node.hostname != PHYSICAL_HOSTNAMES[index]
            || node.binary_sha256 != release_binary_sha256
            || node.train_sha256 != spec.train_sha256
            || node.test_sha256 != spec.test_sha256
            || node.bundle_root != remote_bundle_root(&node.node, release_binary_sha256)?
        {
            bail!("Binary42 staged node binding mismatch");
        }
        if verify_remote {
            if is_controller_node(&node.node)? {
                validate_local_immutable(Path::new(&node.binary_path), &node.binary_sha256, 0o555)?;
                validate_local_immutable(Path::new(&node.train_path), &node.train_sha256, 0o444)?;
                validate_local_immutable(Path::new(&node.test_path), &node.test_sha256, 0o444)?;
            } else {
                scq_batch::validate_remote_immutable(
                    &node.node,
                    &node.binary_path,
                    &node.binary_sha256,
                    0o555,
                )?;
                scq_batch::validate_remote_immutable(
                    &node.node,
                    &node.train_path,
                    &node.train_sha256,
                    0o444,
                )?;
                scq_batch::validate_remote_immutable(
                    &node.node,
                    &node.test_path,
                    &node.test_sha256,
                    0o444,
                )?;
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn shard_manifest(
    campaign: &CampaignManifest,
    staged: &StagedNode,
    generation: usize,
    shard: usize,
    attempt: usize,
    kind: &str,
    members: &[MemberRecord],
    cache: &HashMap<String, CacheEntry>,
) -> Result<ShardManifest> {
    let shard_sizes = campaign_shard_sizes(&campaign.config);
    let start = shard_start(&shard_sizes, shard)?;
    let slots = members
        .iter()
        .enumerate()
        .map(|(offset, member)| ShardSlot {
            generation,
            slot: if kind == "generation" {
                start + offset
            } else {
                offset
            },
            member: member.clone(),
            fresh: kind == "confirmation"
                || cache
                    .get(&member.semantic_sha256)
                    .is_none_or(|entry| entry.first_generation == generation),
        })
        .collect::<Vec<_>>();
    let mut manifest = ShardManifest {
        schema: SHARD_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        kind: kind.to_string(),
        run_id: campaign.run_id.clone(),
        dataset: campaign.dataset.clone(),
        generation,
        shard,
        node: staged.node.clone(),
        job_id: String::new(),
        attempt,
        release_binary_sha256: campaign.release_binary_sha256.clone(),
        train_path: staged.train_path.clone(),
        train_sha256: campaign.dataset_binding.train_sha256.clone(),
        test_path: staged.test_path.clone(),
        test_sha256: campaign.dataset_binding.test_sha256.clone(),
        train_rows: campaign.dataset_binding.train_rows,
        test_rows: campaign.dataset_binding.test_rows,
        features: campaign.dataset_binding.features,
        cpus_reserved: if kind == "generation" {
            *shard_sizes
                .get(shard)
                .context("Binary42 shard index is invalid")?
        } else {
            1
        },
        gpus_per_task: 0,
        candidate_threads: 1,
        slots,
        cohort_mode: (!campaign.config.shard_sizes.is_empty()).then(|| "all".to_string()),
        scq_binary_path: (!campaign.config.shard_sizes.is_empty())
            .then(|| all_scq_binary_str().to_string()),
    };
    manifest.job_id = expected_job_id(&manifest)?;
    Ok(manifest)
}
