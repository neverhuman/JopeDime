use super::*;

/// Where a source checkout keeps the self-play unit templates.
pub(super) fn selfplay_unit_templates_in(source: &Path) -> PathBuf {
    source.join("deploy/systemd/selfplay")
}

/// Templates that lived next to the self-play units but were not bundle
/// content: `install_workspace_sync.py` installed the xbabe2 mirror units as
/// system units (`/etc/systemd/system`) from the checkout, they ran a Python
/// interpreter directly and worked out of the mirror, so the bundle boundary
/// (`validate_systemd_unit`) does not apply to them. The Python removal P1
/// (ADR-0006) archived the installer and both templates (restore with
/// `git checkout 8a55c4fef -- <path>`, docs/archive/python-archive-manifest.json);
/// the names stay excluded so a restored copy is still not bundled
/// (`host_installed_units_are_optional_and_never_rendered_into_the_bundle`).
const HOST_INSTALLED_UNITS: [&str; 2] =
    ["jope-workspace-sync.service", "jope-workspace-sync.timer"];

pub(super) fn is_host_installed_unit(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| {
            HOST_INSTALLED_UNITS
                .iter()
                .any(|unit| name == *unit || name == format!("{unit}.in"))
        })
}

/// The placeholder values a bundle's units are rendered with: the state tree
/// the services work in and the `current` bundle's binaries (the activation
/// symlink every unit follows), so a rendered unit never names a specific
/// bundle identity.
pub(super) fn unit_roots() -> jope_paths::systemd::Roots {
    jope_paths::systemd::Roots {
        home: paths::jope_home(),
        state: paths::jope_state(),
        bundle: paths::runtime_bundle_current().join("runtime/bin"),
        user: std::env::var("USER").unwrap_or_else(|_| "root".to_owned()),
    }
}

/// Per-process scratch directory the templates are rendered into before they
/// are copied into the bundle; removed once the bundle is staged.
pub(super) fn rendered_units_dir(args: &Args) -> PathBuf {
    args.output_root
        .join(format!(".units-rendered.tmp-{}", std::process::id()))
}

/// Renders `deploy/systemd/selfplay/*.in` into `<dir>/systemd/` so the bundle
/// carries concrete units (`runtime/systemd/<unit>`), never templates.
pub(super) fn render_unit_templates(source: &Path, dir: &Path) -> Result<PathBuf> {
    let templates = selfplay_unit_templates_in(source);
    ensure!(
        templates.is_dir(),
        "no self-play unit templates at {}",
        templates.display()
    );
    if dir.exists() {
        fs::remove_dir_all(dir)?;
    }
    let out = dir.join("systemd");
    fs::create_dir_all(&out)?;
    let rendered = jope_paths::systemd::render_tree(&templates, &out, &unit_roots())
        .map_err(|error| anyhow::anyhow!("render unit templates: {error}"))?;
    for unit in rendered.iter().filter(|unit| is_host_installed_unit(unit)) {
        fs::remove_file(unit)?;
    }
    Ok(out)
}

pub(super) fn sources(
    args: &Args,
    service_manifest: &ServiceManifest,
    service_manifest_path: &Path,
) -> Result<Vec<Candidate>> {
    let mut selected = Vec::new();
    for name in ["Cargo.toml", "Cargo.lock"] {
        add(
            &mut selected,
            args.source.join(name),
            format!("runtime/{name}"),
            None,
        );
    }
    collect_extension(
        &args.source.join("src"),
        &args.source,
        "rs",
        "runtime",
        None,
        &mut selected,
    )?;
    for binary in &service_manifest.binaries {
        add(
            &mut selected,
            args.source.join("target/release").join(binary),
            format!("runtime/bin/{binary}"),
            None,
        );
    }
    let rendered_root = rendered_units_dir(args);
    let rendered_units = render_unit_templates(&args.source, &rendered_root)?;
    for extension in ["service", "timer"] {
        collect_extension(
            &rendered_units,
            &rendered_root,
            extension,
            "runtime",
            None,
            &mut selected,
        )?;
    }
    for config in [
        "manifests/selfplay/scq-nodes.v3.toml",
        "manifests/selfplay/scq-projects.v3.toml",
        "manifests/selfplay/freshness-mcp-runtime.v2.json",
        "manifests/selfplay/causal-shadow.v2.json",
        "manifests/selfplay/proposal-selection-policy.v1.json",
        "manifests/dlgp-budget-first/protocol.v1.json",
    ] {
        add(
            &mut selected,
            args.source.join(config),
            format!("runtime/{config}"),
            None,
        );
    }
    add(
        &mut selected,
        service_manifest_path.to_path_buf(),
        format!("runtime/{DEFAULT_SERVICE_MANIFEST}"),
        None,
    );
    add(
        &mut selected,
        args.runner.clone(),
        "runtime/scripts/fweval_runner.sh".to_owned(),
        None,
    );
    add(
        &mut selected,
        args.scq_projects.clone(),
        "external/scq-projects.active.toml".to_owned(),
        None,
    );
    for config in [
        "prime.toml",
        "registry/operator_registry.json",
        "registry/compatibility/gen31_policy_v3_vocab.json",
        "registry/compatibility/prime_arena_suites_v1.json",
    ] {
        add(
            &mut selected,
            args.prime.join(config),
            config.to_owned(),
            None,
        );
        add(
            &mut selected,
            args.prime.join(config),
            format!("pytorch/{config}"),
            None,
        );
    }
    for (locator, target, role) in PYTORCH_ALLOWLIST {
        let source = resolve_locator(locator, &args.source, &args.prime, &args.encoding_driver)?;
        add(&mut selected, source, (*target).to_owned(), Some(*role));
    }
    add(
        &mut selected,
        args.source
            .join("manifests/selfplay/proposal-selection-policy.v1.json"),
        "pytorch/proposals/proposal-selection-policy.v1.json".to_owned(),
        None,
    );
    add(
        &mut selected,
        args.encoding_driver.join("binexp/runner.sh"),
        "pytorch/encoding/runner.sh".to_owned(),
        None,
    );
    collect_extension(
        &args.prime.join("jope"),
        &args.prime,
        "py",
        "pytorch",
        Some("pytorch_dependency_package"),
        &mut selected,
    )?;

    selected.retain(|row| row.source.is_file());
    selected.sort_by(|left, right| left.target.cmp(&right.target));
    let mut targets = BTreeSet::new();
    for row in &selected {
        ensure!(
            targets.insert(row.target.clone()),
            "duplicate bundle target {}",
            row.target
        );
        validate_python_candidate(row, &args.source, &args.prime, &args.encoding_driver)?;
        if row.target.ends_with(".service") {
            validate_systemd_unit(&fs::read_to_string(&row.source)?)
                .with_context(|| format!("validate {}", row.source.display()))?;
        }
        if row.target.ends_with(".sh") {
            let contents = fs::read_to_string(&row.source)?;
            if row.target == "pytorch/encoding/runner.sh" {
                validate_encoding_runner(&contents)
                    .with_context(|| format!("validate {}", row.source.display()))?;
            } else {
                validate_no_python_runtime(&contents)
                    .with_context(|| format!("validate {}", row.source.display()))?;
            }
        }
    }
    for binary in &service_manifest.binaries {
        ensure!(
            selected
                .iter()
                .any(|row| row.target == format!("runtime/bin/{binary}")),
            "required release binary is missing: {binary}; run cargo build --release --workspace --bins"
        );
    }
    Ok(selected)
}

pub(super) fn add(
    selected: &mut Vec<Candidate>,
    source: PathBuf,
    target: String,
    python_role: Option<&'static str>,
) {
    selected.push(Candidate {
        source,
        target,
        python_role,
    });
}

pub(super) fn collect_extension(
    directory: &Path,
    relative_root: &Path,
    extension: &str,
    target_prefix: &str,
    python_role: Option<&'static str>,
    selected: &mut Vec<Candidate>,
) -> Result<()> {
    if !directory.is_dir() {
        return Ok(());
    }
    let mut entries: Vec<_> = fs::read_dir(directory)?.collect::<std::io::Result<_>>()?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            collect_extension(
                &path,
                relative_root,
                extension,
                target_prefix,
                python_role,
                selected,
            )?;
        } else if path.extension().and_then(|value| value.to_str()) == Some(extension) {
            let relative = path.strip_prefix(relative_root)?.to_path_buf();
            add(
                selected,
                path,
                format!("{target_prefix}/{}", relative.display()),
                python_role,
            );
        }
    }
    Ok(())
}

pub(super) fn resolve_locator(
    locator: &str,
    source: &Path,
    prime: &Path,
    encoding_driver: &Path,
) -> Result<PathBuf> {
    if let Some(relative) = locator.strip_prefix("source:") {
        return Ok(source.join(relative));
    }
    if let Some(relative) = locator.strip_prefix("prime:") {
        let relative = match relative {
            "scripts/prime_policy_trainer.py" => "training/prime_policy_trainer.py",
            "scripts/ratchet_dl_update.py" => "training/ratchet_dl_update.py",
            "scripts/encode_external.py" => "encoding/encode_external.py",
            _ => relative,
        };
        return Ok(prime.join(relative));
    }
    if let Some(relative) = locator.strip_prefix("encoding:") {
        return Ok(encoding_driver.join(relative));
    }
    bail!("invalid Python allowlist locator {locator}")
}

pub(super) fn validate_python_candidate(
    candidate: &Candidate,
    source: &Path,
    prime: &Path,
    encoding_driver: &Path,
) -> Result<()> {
    if candidate
        .source
        .extension()
        .and_then(|value| value.to_str())
        != Some("py")
        && !candidate.target.ends_with(".py")
    {
        ensure!(
            candidate.python_role.is_none(),
            "non-Python file has Python role: {}",
            candidate.target
        );
        return Ok(());
    }
    let allowlisted = PYTORCH_ALLOWLIST.iter().any(|(locator, target, role)| {
        resolve_locator(locator, source, prime, encoding_driver)
            .map(|allowed| {
                allowed == candidate.source
                    && *target == candidate.target
                    && Some(*role) == candidate.python_role
            })
            .unwrap_or(false)
    });
    let dependency_package = candidate.python_role == Some("pytorch_dependency_package")
        && candidate.source.starts_with(prime.join("jope"))
        && candidate.target.starts_with("pytorch/jope/")
        && candidate.target.ends_with(".py");
    ensure!(
        allowlisted || dependency_package,
        "operational or unallowlisted Python is forbidden: {}",
        candidate.target
    );
    Ok(())
}

pub(super) fn validate_systemd_unit(contents: &str) -> Result<()> {
    for line in systemd_logical_lines(contents)? {
        let line = line.trim();
        if let Some(working_directory) = line.strip_prefix("WorkingDirectory=") {
            ensure!(
                working_directory == immutable_flywheel_root_str(),
                "systemd WorkingDirectory is not the canonical immutable flywheel root"
            );
        }
        if line.starts_with("ExecStart=")
            || line.starts_with("ExecStartPre=")
            || line.starts_with("ExecStartPost=")
        {
            if line.contains("/jope_selfplay_supply ") {
                validate_supply_pytorch_boundary(line)?;
            } else {
                validate_systemd_python_boundary(line)?;
            }
            let checkout_target = format!("{}/target/", default_source_str());
            ensure!(
                !line.contains("/target-") && !line.contains(&checkout_target),
                "systemd runtime references a mutable or split target directory"
            );
            ensure!(
                !line.contains(python_src_str()) && !line.contains(research_scripts_str()),
                "systemd runtime references a live source directory"
            );
        }
    }
    Ok(())
}

pub(super) fn systemd_logical_lines(contents: &str) -> Result<Vec<String>> {
    let mut logical = Vec::new();
    let mut continued = String::new();
    for physical in contents.lines() {
        let physical = physical.trim();
        let (fragment, continues) = physical
            .strip_suffix('\\')
            .map(|fragment| (fragment.trim_end(), true))
            .unwrap_or((physical, false));
        if !continued.is_empty() && !fragment.is_empty() {
            continued.push(' ');
        }
        continued.push_str(fragment);
        if !continues {
            logical.push(std::mem::take(&mut continued));
        }
    }
    ensure!(
        continued.is_empty(),
        "systemd unit ends with a dangling line continuation"
    );
    Ok(logical)
}

pub(super) fn validate_systemd_python_boundary(line: &str) -> Result<()> {
    let allowed_root = Path::new(active_bundle_pytorch_root_str());
    for (index, token) in line.split_ascii_whitespace().enumerate() {
        let value = token
            .split_once('=')
            .map(|(_, value)| value)
            .unwrap_or(token)
            .trim_matches(['\'', '"']);
        let path = Path::new(value);
        let basename = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(value)
            .to_ascii_lowercase();
        let is_python_interpreter = !basename.contains(".py")
            && (basename.starts_with("python") || basename.starts_with("pypy"));
        ensure!(
            !is_python_interpreter,
            "systemd may not invoke a Python interpreter directly"
        );
        if value.to_ascii_lowercase().contains(".py") {
            ensure!(
                index != 0
                    && !value.contains(':')
                    && path.is_absolute()
                    && path.starts_with(allowed_root)
                    && path.extension().and_then(|extension| extension.to_str()) == Some("py")
                    && !path.components().any(|component| {
                        matches!(
                            component,
                            std::path::Component::ParentDir | std::path::Component::CurDir
                        )
                    }),
                "Python script reference escaped the active bundle PyTorch boundary"
            );
        }
    }
    Ok(())
}

pub(super) fn validate_supply_pytorch_boundary(line: &str) -> Result<()> {
    let normalized = line.to_ascii_lowercase();
    let pytorch = active_bundle_pytorch_root_str().to_ascii_lowercase();
    ensure!(
        !normalized.contains("/usr/bin/python")
            && !normalized.contains("python3")
            && normalized.contains(&format!(
                "--proposal-generator {pytorch}/proposals/test_exact_context_proposals.py"
            ))
            && normalized.contains(&format!(
                "--proposal-legacy-generator {pytorch}/proposals/test_exact_dl_generate.py"
            ))
            && normalized.contains(&format!(
                "--proposal-common {pytorch}/shared/test_exact_common.py"
            ))
            && normalized.contains(&format!("--proposal-pythonpath {pytorch}")),
        "supply may name only the bundled PyTorch proposal boundary"
    );
    Ok(())
}
