use super::*;

pub(in crate::test_adaptive::binary42) fn remote_bundle_root(
    node: &str,
    release_binary_sha256: &str,
) -> Result<String> {
    if !NODES.contains(&node) {
        bail!("unknown Binary42 SCQ node {node}");
    }
    Ok(format!(
        "{linux_remote_root_str}/bundles/{release_binary_sha256}",
        linux_remote_root_str = linux_remote_root_str()
    ))
}

pub(in crate::test_adaptive::binary42) fn controller_hostname() -> Result<String> {
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

pub(in crate::test_adaptive::binary42) fn is_controller_node(node: &str) -> Result<bool> {
    Ok(controller_hostname()? == physical_hostname(node)?)
}

pub(in crate::test_adaptive::binary42) fn validate_local_immutable(
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

pub(in crate::test_adaptive::binary42) fn copy_local_atomic(
    source: &Path,
    destination: &Path,
    executable: bool,
) -> Result<()> {
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

pub(in crate::test_adaptive::binary42) fn submit_canary(
    node: &str,
    suffix: &str,
) -> Result<(String, String)> {
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

pub(in crate::test_adaptive::binary42) fn submit_canary_at(
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

pub(in crate::test_adaptive::binary42) fn node_command_output(
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
