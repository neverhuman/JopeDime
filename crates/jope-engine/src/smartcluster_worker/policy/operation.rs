use super::*;

pub(super) fn verify_training_closure(root: &Path, input: &InputManifest) -> Result<()> {
    if let Operation::Train { corpus, .. } = &input.operation {
        // Historical Train manifests over unprojected corpora fail closed here by
        // design: re-verifying them would open every excluded context they bind.
        relative(corpus)?;
        let path = corpus.join("index.json");
        let sha256 = input
            .input_artifacts
            .get(&path)
            .cloned()
            .context("training corpus index is not in dependency closure")?;
        let index = crate::native_genome::load_corpus_index(
            root,
            &crate::native_graph::Artifact { path, sha256 },
        )?;
        crate::corpus_projection::verify_admitted_context_closure(&index)?;
    }
    Ok(())
}

pub(super) fn validate_operation(task: &TaskManifest, input: &InputManifest) -> Result<()> {
    ensure!(
        input.operation.dataset_logical_attempt() != Some(0),
        "dataset logical attempt must be positive"
    );
    match (&task.kind, &input.operation) {
        (
            TaskKind::ResidentGpu,
            Operation::Resident | Operation::Train { .. } | Operation::Proposal { .. },
        )
        | (TaskKind::GpSearch, Operation::DirectEvaluation { .. })
        | (TaskKind::Continuation, Operation::SeededContinuation { .. })
        | (TaskKind::GpSearch, Operation::Search { seeds: None, .. })
        | (TaskKind::Continuation, Operation::Search { seeds: Some(_), .. })
        | (TaskKind::Packing, Operation::Pack { .. } | Operation::PackAndPublish { .. })
        | (TaskKind::Publication, Operation::DatasetPublication { .. })
        | (TaskKind::Verification, Operation::Verify) => {}
        _ => anyhow::bail!(
            "operation is not implemented by this fixed service; publication belongs to dataset authority"
        ),
    }
    if !matches!(input.operation, Operation::Resident | Operation::Verify) {
        ensure!(
            task.resources.scratch_bytes > 0,
            "numerical operation requires explicit scratch reservation"
        );
    }
    if let Operation::Train {
        micro,
        checkpoint_steps,
        initial_optimizer_step,
        requested_step_increment,
        target_optimizer_step,
        ..
    } = &input.operation
    {
        ensure!(
            [64, 128, 256].contains(micro)
                && *checkpoint_steps > 0
                && *requested_step_increment > 0
                && initial_optimizer_step.checked_add(*requested_step_increment)
                    == Some(*target_optimizer_step),
            "original optimizer window invalid"
        );
    }
    for path in operation_inputs(&input.operation) {
        relative(path)?;
        ensure!(
            input
                .input_artifacts
                .keys()
                .any(|bound| bound == path || bound.starts_with(path)),
            "operation input is not in dependency closure: {}",
            path.display()
        );
    }
    for path in operation_outputs(&input.operation) {
        relative(path)?;
    }
    Ok(())
}

pub(super) fn verify_required_inputs(input: &InputManifest) -> Result<()> {
    let mut required = BTreeMap::new();
    match &input.operation {
        Operation::DirectEvaluation { manifest, sha256 }
        | Operation::SeededContinuation { manifest, sha256 } => {
            let reference = crate::native_graph::Artifact {
                path: manifest.clone(),
                sha256: sha256.clone(),
            };
            required = crate::native_genome::required_inputs(&input.root, &reference)?;
            let task: crate::native_genome::GenomeTask =
                crate::native_genome::load(&input.root, &reference)?;
            ensure!(
                (task.stage == crate::native_genome::Stage::Direct)
                    == matches!(input.operation, Operation::DirectEvaluation { .. }),
                "native genome operation stage changed"
            );
        }
        Operation::DatasetPublication {
            manifest, sha256, ..
        }
        | Operation::PackAndPublish {
            manifest, sha256, ..
        } => {
            ensure!(
                file_hash(&resolve(&input.root, manifest)?)? == *sha256,
                "dataset manifest changed"
            );
            required.insert(manifest.clone(), sha256.clone());
        }
        Operation::Train {
            corpus,
            run,
            launch,
            micro,
            checkpoint_steps,
            initialize,
            replay_policy,
            requested_step_increment,
            ..
        } => {
            required = resident::input_artifacts(
                &input.root,
                &json!({"kind":"Train","corpus":corpus,"run":run,"launch":launch,"world":1,"micro":micro,"steps":requested_step_increment,"checkpoint_steps":checkpoint_steps,"resume":false,"initialize":initialize,"replay_policy":replay_policy}),
            )?;
        }
        Operation::Proposal {
            corpus,
            checkpoint,
            output,
            cuts,
            cut_start,
            shuffled,
            champion_snapshot,
            preparation_overlay,
        } => {
            required = resident::request_artifacts(
                &input.root,
                &json!({"kind":"Proposal","corpus":corpus,"checkpoint":checkpoint,"output":output,"world":1,"cuts":cuts,"cut_start":cut_start,"shuffled":shuffled}),
                &json!({"JOPE_CHAMPION_SNAPSHOT":champion_snapshot}),
            )?;
            if let Some(a) = preparation_overlay {
                required.extend(crate::preparation_overlay::dependencies(&input.root, a)?);
            }
        }
        Operation::Search { prepared, .. } => {
            let prepared_path = resolve(&input.root, prepared)?;
            let metadata: crate::prepare::Prepared =
                read_json(&prepared_path.join("prepared.json"))?;
            ensure!(
                metadata.meta_partition != "meta_test",
                "sealed-test search admission refused"
            );
            for entry in walkdir::WalkDir::new(&prepared_path).follow_links(false) {
                let entry = entry?;
                ensure!(
                    !entry.file_type().is_symlink(),
                    "prepared input symlink refused"
                );
                if entry.file_type().is_file() {
                    required.insert(
                        entry.path().strip_prefix(&input.root)?.to_owned(),
                        file_hash(entry.path())?,
                    );
                }
            }
            if let Some(storage) = metadata.storage {
                let source = if storage.source.is_absolute() {
                    storage.source
                } else {
                    input.root.join(storage.source)
                };
                let relative = source.strip_prefix(&input.root)?.to_owned();
                ensure!(
                    file_hash(&resolve(&input.root, &relative)?)? == storage.source_sha256,
                    "prepared observation source changed"
                );
                required.insert(relative, storage.source_sha256);
            }
        }
        _ => {}
    }
    ensure!(
        required
            .iter()
            .all(|(path, sha)| input.input_artifacts.get(path) == Some(sha)),
        "required numerical dependency closure incomplete"
    );
    Ok(())
}

fn operation_inputs(op: &Operation) -> Vec<&Path> {
    match op {
        Operation::DirectEvaluation { manifest, .. }
        | Operation::SeededContinuation { manifest, .. } => vec![manifest],
        Operation::DatasetPublication { manifest, .. }
        | Operation::PackAndPublish { manifest, .. } => vec![manifest],
        Operation::Search {
            prepared,
            seeds,
            vocabulary,
            ..
        } => std::iter::once(prepared.as_path())
            .chain(seeds.iter().map(PathBuf::as_path))
            .chain(vocabulary.iter().map(PathBuf::as_path))
            .collect(),
        Operation::Pack {
            contexts,
            vocabulary,
            observations,
            ..
        } => std::iter::once(contexts.as_path())
            .chain(std::iter::once(vocabulary.as_path()))
            .chain(observations.iter().map(PathBuf::as_path))
            .collect(),
        Operation::Train {
            corpus,
            initialize,
            replay_policy,
            ..
        } => vec![corpus, initialize, replay_policy],
        Operation::Proposal {
            corpus,
            checkpoint,
            champion_snapshot,
            ..
        } => vec![corpus, checkpoint, champion_snapshot],
        _ => vec![],
    }
}
fn operation_outputs(op: &Operation) -> Vec<&Path> {
    match op {
        Operation::Search { output, .. }
        | Operation::Pack { output, .. }
        | Operation::Proposal { output, .. } => vec![output],
        Operation::Train { run, launch, .. } => vec![run, launch],
        _ => vec![],
    }
}

pub(super) fn available_storage_bytes(path: &Path) -> Result<u64> {
    let stats = nix::sys::statvfs::statvfs(path).context("storage measurement unavailable")?;
    Ok(storage_bytes_from_stats(stats))
}

pub(super) fn storage_bytes(blocks_available: u64, fragment_size: u64) -> u64 {
    blocks_available.saturating_mul(fragment_size)
}

pub(super) fn storage_bytes_from_stats(stats: nix::sys::statvfs::Statvfs) -> u64 {
    storage_bytes(stats.blocks_available(), stats.fragment_size())
}

pub(super) fn admission_guards(policy: &WorkerPolicy) -> Result<()> {
    for (mountpoint, device) in &policy.storage_devices {
        let mounts = fs::read_to_string("/proc/self/mountinfo")?;
        let source = mounts
            .lines()
            .find_map(|line| {
                let (left, right) = line.split_once(" - ")?;
                (left.split_whitespace().nth(4)? == mountpoint.to_str()?)
                    .then(|| right.split_whitespace().nth(1))
                    .flatten()
            })
            .context("required storage mount absent")?;
        let actual = fs::metadata(source)?;
        let expected = fs::metadata(device)?;
        use std::os::unix::fs::FileTypeExt;
        ensure!(
            actual.file_type().is_block_device()
                && expected.file_type().is_block_device()
                && actual.rdev() == expected.rdev(),
            "storage mount device changed"
        );
    }

    let memory = fs::read_to_string("/proc/meminfo")?;
    let available: u64 = memory
        .lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))
        .context("MemAvailable absent")?
        .split_whitespace()
        .next()
        .context("MemAvailable value absent")?
        .parse()?;
    ensure!(
        available.saturating_mul(1024) >= policy.minimum_available_memory_bytes,
        "RAM reserve blocks subtask"
    );
    for (path, reserve) in &policy.storage_reserves {
        ensure!(
            available_storage_bytes(path)? >= *reserve,
            "storage reserve blocks subtask"
        );
    }
    Ok(())
}
