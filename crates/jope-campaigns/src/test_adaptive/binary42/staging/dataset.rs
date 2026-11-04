use super::super::member_models::StagingReceipt;
use super::*;

pub(in crate::test_adaptive::binary42) fn stage_campaign(
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

pub(in crate::test_adaptive::binary42) fn validate_staging(
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
pub(in crate::test_adaptive::binary42) fn shard_manifest(
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
