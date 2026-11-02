use super::*;

pub(in crate::test_adaptive::binary42) fn validate_all_staging(
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

pub(in crate::test_adaptive::binary42) fn stage_all_campaign(
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
