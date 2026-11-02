use super::*;

struct ProcessLock {
    _file: File,
}

impl ProcessLock {
    fn acquire(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .mode(0o600)
            .open(path)?;
        if flock::exclusive(&file, false).is_err() {
            bail!("another trainer watchdog owns {}", path.display());
        }
        Ok(Self { _file: file })
    }
}

pub(crate) fn run() -> Result<()> {
    let args = Args::parse();
    if args.explain_checkpoint {
        return explain_checkpoint(&args);
    }
    let ops = args.root.join("ops");
    fs::create_dir_all(&ops)?;
    let _lock = ProcessLock::acquire(&ops.join("trainer-watchdog.lock"))?;
    let mut lineage = LineageCache::default();
    loop {
        match cycle(&args, &mut lineage) {
            Ok(report) => {
                atomic_json(
                    &ops.join("trainer-watchdog-v1.json"),
                    &serde_json::to_value(&report)?,
                )?;
                println!("{}", serde_json::to_string(&report)?);
            }
            Err(error) => {
                // Degrade IN SCHEMA. Never swap the document shape under the
                // consumers — see CycleReport::degraded for why.
                let report = CycleReport::degraded(
                    if args.submit { "submit" } else { "audit" },
                    &format!("{error:#}"),
                );
                atomic_json(&ops.join("trainer-watchdog-v1.json"), &report)?;
                eprintln!("{}", serde_json::to_string(&report)?);
                if args.once {
                    return Err(error);
                }
            }
        }
        if args.once {
            return Ok(());
        }
        thread::sleep(Duration::from_secs(args.interval_seconds));
    }
}
