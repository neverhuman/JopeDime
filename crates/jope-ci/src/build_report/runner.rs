use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn run_report(
    config: &Path,
    output: &Path,
    run: &Path,
    tools: &Path,
    approved: &[String],
    hooks: &Hooks<'_>,
    interrupt: Interrupt<'_>,
    receipt: &mut Dict,
    save: &dyn Fn(&Dict) -> Outcome<()>,
) -> Outcome<()> {
    let exe = std::env::current_exe()
        .and_then(|p| p.canonicalize())
        .map_err(|e| os_error(&e, Path::new("jope-ci")))?;
    let mut paths = vec![
        exe,
        tools.join("cargo-build-dirs.sh"),
        tools.join("jope-build-report.sh"),
    ];
    let units = units_dir()?;
    for name in ["jope-build-sweep.service", "jope-build-sweep.timer"] {
        // `path.is_file()`: a missing unit is recorded as missing; any other error (a unit
        // directory that may not be searched) fails the run, as the Python raised it.
        let spelled = units.join(name).to_string();
        let path = PathBuf::from(&spelled);
        if py::path_is_file(&path).map_err(|e| os_error(&e, &path))? {
            paths.push(path);
        } else {
            list_mut(receipt, "missing_unit_files").push(Value::from(spelled));
        }
    }
    // `receipt["checksums"][str(path)] = sha256(path.read_bytes())`: the file is read first, then
    // its name is the key. A name that is not UTF-8 (this executable, the tools directory or the
    // configuration reached through a directory named with such a byte) was a surrogate-escaped
    // key in the Python's receipt; here it fails the run (stricter), never a lossy key.
    for path in &paths {
        let sum = sha256_file(path)?;
        let key = receipt_text(path, "checksum key")?.to_owned();
        receipt
            .get_mut("checksums")
            .and_then(Value::as_dict_mut)
            .expect("checksums")
            .insert(key, Value::from(sum));
    }
    let config_path = config.canonicalize().map_err(|e| os_error(&e, config))?;
    let config_bytes = fs::read(&config_path).map_err(|e| os_error(&e, &config_path))?;
    let sum = sha256_file(&config_path)?;
    let key = receipt_text(&config_path, "checksum key")?.to_owned();
    receipt
        .get_mut("checksums")
        .and_then(Value::as_dict_mut)
        .expect("checksums")
        .insert(key, Value::from(sum));
    let config = pyjson::parse_bytes(&config_bytes).map_err(Failure::new)?;
    let keys_present = config.as_dict().is_some_and(|d| {
        ["cargo_roots", "stale_hours", "measurement_roots"]
            .iter()
            .all(|k| d.contains_key(k))
    });
    if !keys_present {
        return Err(Failure::new(
            "config requires cargo_roots, stale_hours and measurement_roots",
        ));
    }
    receipt.insert("config", config.clone());
    let approved_list = Value::List(approved.iter().map(|r| Value::from(r.as_str())).collect());
    let roots = config.get("cargo_roots").expect("checked");
    if approved.len() != 2
        || roots != &approved_list
        || config.get("stale_hours") != Some(&Value::Int(48))
    {
        return Err(Failure::new(
            "service requires the two approved Cargo roots and a 48-hour filter",
        ));
    }
    let measurements = config
        .get("measurement_roots")
        .and_then(Value::as_list)
        .ok_or_else(|| Failure::new("measurement_roots must be a list"))?
        .clone();
    for (index, root) in approved.iter().map(String::as_str).enumerate() {
        let result_name = format!("candidates-{index}.json");
        let stderr_name = format!("scan-{index}.stderr");
        let row: Dict = [
            ("root", Value::from(root)),
            ("output", Value::from(result_name.clone())),
            ("stderr", Value::from(stderr_name.clone())),
        ]
        .into_iter()
        .collect();
        list_mut(receipt, "scan_outputs").push(Value::Dict(row));
        let diagnostic = run.join(&stderr_name);
        fs::write(&diagnostic, "").map_err(|e| os_error(&e, &diagnostic))?;
        match (hooks.scan)(root, interrupt) {
            Ok(document) => {
                let path = run.join(&result_name);
                fs::write(&path, dumps(&document)).map_err(|e| os_error(&e, &path))?;
            }
            Err(error) => {
                let text = format!(
                    "{}\n{}",
                    error.message,
                    error.stderr.clone().unwrap_or_default()
                );
                let _ = fs::write(&diagnostic, text);
                return Err(error);
            }
        }
    }
    // Keep Cargo reports even if a separate comparison root is stale: validate every root
    // before any measurement.
    let mut seen: Vec<PathBuf> = vec![];
    for item in &measurements {
        let valid = item.as_dict().is_some_and(|d| {
            d.get("path").and_then(Value::as_str).is_some()
                && d.get("group").and_then(Value::as_str).is_some()
                && matches!(d.get("privileged"), None | Some(Value::Bool(_)))
        });
        if !valid {
            return Err(Failure::new(
                "each measurement requires string path/group and optional boolean privileged",
            ));
        }
        let (path, resolved) =
            explicit_root(item.get("path").and_then(Value::as_str).expect("checked"))?;
        if seen.iter().any(|prior| {
            resolved == *prior || resolved.starts_with(prior) || prior.starts_with(&resolved)
        }) {
            return Err(Failure::new(format!(
                "overlapping measurement root: {path}"
            )));
        }
        seen.push(resolved);
    }
    for item in &measurements {
        let root = PathBuf::from(item.get("path").and_then(Value::as_str).expect("checked"));
        let privileged = item.get("privileged") == Some(&Value::Bool(true));
        let bytes = (hooks.measure)(&root, privileged, interrupt)?;
        let mut row = item.as_dict().expect("checked").clone();
        row.insert("allocated_bytes", exact_int(u128::from(bytes)));
        row.insert("observed_at", Value::from(utc().iso));
        list_mut(receipt, "measurements").push(Value::Dict(row));
        save(receipt)?;
    }
    let filesystem = run_bounded(
        &["df".as_ref(), "-B1".as_ref(), output.as_os_str()],
        DF_TIMEOUT,
        interrupt,
    )?;
    let path = run.join("filesystem.txt");
    fs::write(&path, filesystem).map_err(|e| os_error(&e, &path))?;
    receipt.insert("status", Value::from("success"));
    receipt.insert("exit_status", Value::from(0i64));
    Ok(())
}

/// SIGTERM/SIGINT become a failed receipt ("report interrupted by signal N") instead of a
/// `running` one.
pub fn signal_interrupt() -> std::io::Result<impl Fn() -> Outcome<()>> {
    let flag = Arc::new(AtomicUsize::new(0));
    for signal in [signal_hook::consts::SIGTERM, signal_hook::consts::SIGINT] {
        signal_hook::flag::register_usize(signal, Arc::clone(&flag), signal as usize)?;
    }
    Ok(move || match flag.load(Ordering::SeqCst) {
        0 => Ok(()),
        signal => Err(Failure::new(format!(
            "report interrupted by signal {signal}"
        ))),
    })
}
