use super::*;

/// `jope-bench ablation`: compare two search profiles at an equal evaluation budget.
#[derive(clap::Args, Clone, Debug)]
pub struct AblationArgs {
    /// Profile A, the reference.
    #[arg(long, default_value = "legacy")]
    pub profile_a: String,
    /// Profile B, the candidate.
    #[arg(long, default_value = "legacy")]
    pub profile_b: String,
    /// Refinement parents per round for kernel2-baseline and kernel2-experimental (0..=4), the
    /// same for both profiles. Required when either profile is one of them; refused otherwise.
    #[arg(long)]
    pub refine_top_k: Option<usize>,
    /// Objective trials per refined parent, inside each cut's evaluation budget. Required and
    /// refused together with --refine-top-k.
    #[arg(long)]
    pub refine_trials: Option<usize>,
    /// A dataset directory (metadata.json + train.csv) or a directory of them (repeatable).
    #[arg(long = "datasets", required = true)]
    pub datasets: Vec<PathBuf>,
    /// Search seeds, comma-separated.
    #[arg(long, value_delimiter = ',', default_value = "1,2,3,4,5")]
    pub seeds: Vec<u64>,
    /// Evaluation budget of every cut's search (1..=1000).
    #[arg(long, default_value_t = 64)]
    pub evaluations: usize,
    /// Cuts per dataset (1..=5).
    #[arg(long, default_value_t = 1)]
    pub cuts: usize,
    /// Fold seed of the cut partition (the pipeline's default).
    #[arg(long, default_value_t = jope_pipeline::DEFAULT_SEED)]
    pub fold_seed: u64,
    /// Searches run concurrently; the results do not depend on it.
    #[arg(long, default_value_t = 1)]
    pub jobs: usize,
    /// Where to write the jope.bench-ablation/v1 document. Refused when its file name is
    /// .ablation-lock or .ablation-kernel2 (reserved), when it is a symbolic link or not a regular
    /// file, when it lies under a directory that holds a kernel2 comparison's evidence
    /// (.ablation-kernel2) or a comparison in progress (.ablation-lock), and, with a kernel2
    /// profile, when it lies inside --scratch.
    #[arg(long)]
    pub out: PathBuf,
    /// Keep every search's evidence under this directory (default: a temporary directory under
    /// TMPDIR, removed afterwards; with neither this nor TMPDIR the run is refused rather than use
    /// the system temporary directory). It must not already hold a search of the same run; with a
    /// kernel2 profile it must be absent or an empty directory, and no later comparison may use
    /// it. One comparison at a time holds it (.ablation-lock).
    #[arg(long)]
    pub scratch: Option<PathBuf>,
}

/// Where a run without `--scratch` makes its temporary scratch directory: `TMPDIR`, when it is
/// set and not empty. There is no fallback to the system temporary directory; without `TMPDIR`
/// the run is refused and asks for `--scratch`.
pub(crate) fn default_scratch_root(tmpdir: Option<std::ffi::OsString>) -> Result<PathBuf> {
    match tmpdir {
        Some(dir) if !dir.is_empty() => Ok(PathBuf::from(dir)),
        _ => bail!(
            "no scratch directory: pass --scratch <dir> or set TMPDIR; the harness does not fall \
             back to the system temporary directory"
        ),
    }
}

/// The comparison `args` asks for. A refinement option given alone is refused here; every other
/// refusal of the options is [`Config`]'s, which [`execute`] makes before it creates a scratch
/// directory and [`run`] makes again.
pub(crate) fn config(args: &AblationArgs) -> Result<Config> {
    let profile_a = Profile::parse(&args.profile_a)?;
    let profile_b = Profile::parse(&args.profile_b)?;
    let kernel2 = profile_a.is_kernel2() || profile_b.is_kernel2();
    let refinement = match (args.refine_top_k, args.refine_trials) {
        (None, None) => None,
        (Some(top_k), Some(trials_per_program)) => Some(Refinement {
            top_k,
            trials_per_program,
        }),
        (None, Some(_)) if kernel2 => bail!(
            "a kernel2 profile requires an explicit --refine-top-k (0..={})",
            search::ELITES
        ),
        (Some(_), None) if kernel2 => {
            bail!("a kernel2 profile requires an explicit --refine-trials")
        }
        _ => bail!(REFINEMENT_REFUSAL),
    };
    Ok(Config {
        profile_a,
        profile_b,
        datasets: args.datasets.clone(),
        seeds: args.seeds.clone(),
        evaluations: args.evaluations,
        cuts: args.cuts,
        fold_seed: args.fold_seed,
        jobs: args.jobs,
        refinement,
    })
}

/// One line of the command's paired summary.
pub(crate) fn paired_line(label: &str, c: &PairedCounts) -> String {
    format!(
        "  {label}: W/T/L {}/{}/{} over {} pairs, median M1 RMSE A {} B {}, sign test p {}, \
         same M1 program {}/{}",
        c.wins,
        c.ties,
        c.losses,
        c.pairs,
        c.median_a,
        c.median_b,
        c.sign_test_p,
        c.same_winner,
        c.pairs
    )
}

/// Run `jope-bench ablation`. Every refusal of the options, and of the `--out` destination
/// (`check_out`), is made before the scratch directory is created. [`run`] then holds an
/// exclusive `Reservation` of the scratch for the whole comparison, so a second comparison on
/// the same `--scratch` is refused while this one runs, and refuses a scratch a kernel2
/// comparison used (`check_not_kernel2`). A comparison with a kernel2 profile also needs an
/// unused `--scratch`: one created here, empty, or an existing empty directory; [`run`] refuses
/// anything else before it writes (`check_scratch_unused`) and marks the scratch for good
/// (`mark_kernel2`). The temporary scratch made without `--scratch` is always unused. After
/// [`run`] returns, and so after its reservation is released, `--out`'s directory is created
/// when missing, `--out` is checked again, and the document is written by `replace_document`:
/// a new temporary file renamed over `--out`, so no existing file is truncated.
pub fn execute(args: &AblationArgs) -> Result<()> {
    let config = config(args)?;
    config.check()?;
    check_out(&args.out, args.scratch.as_deref(), config.has_kernel2())?;
    let temporary;
    let scratch = match &args.scratch {
        Some(dir) => {
            std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
            dir.clone()
        }
        None => {
            let root = default_scratch_root(std::env::var_os("TMPDIR"))?;
            std::fs::create_dir_all(&root).with_context(|| format!("create {}", root.display()))?;
            temporary = tempfile::Builder::new()
                .prefix("jope-bench-ablation-")
                .tempdir_in(&root)
                .with_context(|| format!("create a scratch directory under {}", root.display()))?;
            temporary.path().to_path_buf()
        }
    };
    let report = run(&config, &scratch)?;
    create_document_parent(&args.out)?;
    check_out(&args.out, Some(&scratch), config.has_kernel2())?;
    let file_sha256 = replace_document(&args.out, &report.document)?;
    let body = &report.document.body;
    eprintln!(
        "ablation {} (A) vs {} (B): {} datasets x {} seeds, {} evaluations per cut: {}",
        config.profile_a.id(),
        config.profile_b.id(),
        body.datasets.len(),
        config.seeds.len(),
        config.evaluations,
        body.verdict
    );
    for (name, gate) in body.gates.all() {
        eprintln!("  {name}: {}", gate.status);
    }
    if let Some(paired) = &body.paired {
        eprintln!("{}", paired_line("paired (B against A)", &paired.overall));
        for (dataset, record) in paired.per_dataset.iter().zip(&body.datasets) {
            eprintln!("{}", paired_line(&record.id, &dataset.counts));
        }
    }
    if let Some(seconds) = &report.document.timing.paired {
        eprintln!(
            "  search seconds: A {:.3} B {:.3} (B/A {})",
            seconds.total_seconds_a,
            seconds.total_seconds_b,
            seconds
                .ratio_b_to_a
                .map_or_else(|| "undefined".to_owned(), |r| format!("{r:.3}"))
        );
    }
    eprintln!(
        "wrote {} (body_sha256 {}, file sha256 {})",
        args.out.display(),
        report.document.body_sha256,
        file_sha256
    );
    Ok(())
}
