fn current_corpus_eval_sha256() -> Result<String> {
    sha256_file(Path::new(corpus_eval_str()))
}

fn remote_bundle_root(release_binary_sha256: &str) -> String {
    format!(
        "{output_root_str}/bundles/{release_binary_sha256}",
        output_root_str = output_root_str()
    )
}

fn remote_dataset_root(release_binary_sha256: &str, dataset_hash: &str) -> String {
    format!(
        "{}/datasets/{dataset_hash}",
        remote_bundle_root(release_binary_sha256)
    )
}

fn node_marker_path(bundle_root: &str, hostname: &str) -> String {
    format!("{bundle_root}/nodes/{hostname}.ready")
}

fn controller_hostname() -> Result<String> {
    Ok(String::from_utf8(
        scq_batch::command_output(
            {
                let mut command = Command::new("hostname");
                command.arg("-s");
                command
            },
            "TimeSeries42 controller hostname",
        )?
        .stdout,
    )?
    .trim()
    .to_string())
}

fn is_local_scq_node(node: &str, controller_hostname: &str) -> bool {
    node == LOCAL_SCQ_NODE || node == controller_hostname
}

fn is_controller_node(node: &str) -> Result<bool> {
    Ok(is_local_scq_node(node, &controller_hostname()?))
}

fn validate_local_checked(
    path: &Path,
    expected_sha256: &str,
    expected_permissions: u32,
) -> Result<()> {
    let metadata = fs::symlink_metadata(path).with_context(|| {
        format!(
            "local TimeSeries42 staging path is absent: {}",
            path.display()
        )
    })?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        bail!(
            "local TimeSeries42 staging path is not a regular file: {}",
            path.display()
        );
    }
    let mode = metadata.permissions().mode() & 0o7777;
    if mode != expected_permissions {
        bail!(
            "local TimeSeries42 staging mode mismatch on {}: got {mode:#o}, expected {expected_permissions:#o}",
            path.display()
        );
    }
    let actual_sha256 = sha256_file(path)?;
    if actual_sha256 != expected_sha256 {
        bail!(
            "local TimeSeries42 staging hash mismatch on {}: got {actual_sha256}, expected {expected_sha256}",
            path.display()
        );
    }
    Ok(())
}

fn copy_local_checked(source: &Path, destination: &Path, executable: bool) -> Result<()> {
    let expected_sha256 = sha256_file(source)?;
    let expected_permissions = staging_file_mode(executable);
    if destination.exists() {
        let validation =
            validate_local_checked(destination, &expected_sha256, expected_permissions);
        if validation.is_ok() {
            return Ok(());
        }
        fs::remove_file(destination).with_context(|| {
            format!(
                "remove invalid local TimeSeries42 staging file after mismatch: {:#}",
                validation.unwrap_err()
            )
        })?;
    }
    fs::create_dir_all(
        destination
            .parent()
            .context("local TimeSeries42 destination has no parent")?,
    )?;
    fs::copy(source, destination)
        .with_context(|| format!("copy {} to {}", source.display(), destination.display()))?;
    let permissions = std::os::unix::fs::PermissionsExt::from_mode(expected_permissions);
    fs::set_permissions(destination, permissions)?;
    validate_local_checked(destination, &expected_sha256, expected_permissions)
}

fn ssh_raw_output(node: &str, remote_args: &[&str], label: &str) -> Result<std::process::Output> {
    let mut command = Command::new("ssh");
    command.arg(node).arg("--").args(remote_args);
    command.output().with_context(|| format!("launch {label}"))
}

fn remote_path_exists(node: &str, path: &str) -> Result<bool> {
    let exists = ssh_raw_output(
        node,
        &["test", "-e", path],
        "probe TimeSeries42 remote path",
    )?;
    if exists.status.success() {
        return Ok(true);
    }
    if exists.status.code() != Some(1) {
        bail!(
            "TimeSeries42 remote existence probe failed on {node}:{path} ({}): {}",
            exists.status,
            String::from_utf8_lossy(&exists.stderr).trim()
        );
    }
    let symlink = ssh_raw_output(
        node,
        &["test", "-L", path],
        "probe TimeSeries42 remote symlink",
    )?;
    if symlink.status.success() {
        bail!("TimeSeries42 remote immutable path is a symbolic link: {node}:{path}");
    }
    if symlink.status.code() != Some(1) {
        bail!(
            "TimeSeries42 remote symlink probe failed on {node}:{path} ({}): {}",
            symlink.status,
            String::from_utf8_lossy(&symlink.stderr).trim()
        );
    }
    Ok(false)
}

fn staging_file_mode(executable: bool) -> u32 {
    if executable {
        0o555
    } else {
        0o444
    }
}

fn remove_remote_path(node: &str, path: &str) -> Result<()> {
    scq_batch::ssh_output(
        node,
        &["rm", "-f", "--", path],
        "remove TimeSeries42 remote staging file",
    )?;
    Ok(())
}

fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

fn upload_remote_file(node: &str, source: &Path, destination: &str) -> Result<()> {
    let remote_command = format!("cat > {}", shell_quote(destination));
    let mut child = Command::new("timeout")
        .arg("600s")
        .arg("ssh")
        .arg("-T")
        .arg(node)
        .arg(remote_command)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("start TimeSeries42 remote upload to {node}:{destination}"))?;
    {
        let mut input = child.stdin.take().context("remote upload stdin missing")?;
        let mut file =
            fs::File::open(source).with_context(|| format!("open {}", source.display()))?;
        std::io::copy(&mut file, &mut input)
            .with_context(|| format!("stream {} to {node}:{destination}", source.display()))?;
    }
    let output = child
        .wait_with_output()
        .with_context(|| format!("finish TimeSeries42 remote upload to {node}:{destination}"))?;
    if !output.status.success() {
        bail!(
            "TimeSeries42 remote upload failed for {node}:{destination} ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(())
}

fn copy_remote_checked(
    node: &str,
    source: &Path,
    destination: &str,
    executable: bool,
) -> Result<()> {
    let parent = Path::new(destination)
        .parent()
        .context("remote TimeSeries42 destination has no parent")?
        .to_string_lossy()
        .into_owned();
    let expected_sha256 = sha256_file(source)?;
    let expected_permissions = staging_file_mode(executable);
    if remote_path_exists(node, destination)? {
        let validation = scq_batch::validate_remote_immutable(
            node,
            destination,
            &expected_sha256,
            expected_permissions,
        );
        if validation.is_ok() {
            return Ok(());
        }
        let error = validation.unwrap_err();
        remove_remote_path(node, destination).with_context(|| {
            format!("remove invalid TimeSeries42 staging file after mismatch: {error:#}")
        })?;
    }
    scq_batch::ssh_output(
        node,
        &["mkdir", "-p", &parent],
        "create TimeSeries42 remote staging directory",
    )?;
    let upload_result = (|| -> Result<()> {
        upload_remote_file(node, source, destination)?;
        let mode = if executable { "0555" } else { "0444" };
        scq_batch::ssh_output(
            node,
            &["chmod", mode, "--", destination],
            "chmod TimeSeries42 staged file",
        )?;
        scq_batch::validate_remote_immutable(
            node,
            destination,
            &expected_sha256,
            expected_permissions,
        )
    })();
    if upload_result.is_err() {
        let _ = remove_remote_path(node, destination);
    }
    upload_result
}
