pub fn search(
    prepared: &Path,
    cut: usize,
    output: &Path,
    evaluations: usize,
    seconds: u64,
    qualification: bool,
) -> Result<()> {
    search_seeded(
        prepared,
        cut,
        output,
        crate::gp::SearchOptions {
            evaluations,
            seconds,
            qualification,
            seeds: None,
            vocabulary: None,
        },
    )
}
/// A comparison search may write only below its explicitly bound experiment.
pub fn verify_experiment_output(output: &Path, experiment: &Path) -> Result<()> {
    let experiment = experiment.canonicalize()?;
    let manifest: serde_json::Value = read_json(&experiment.join("manifest.json"))?;
    ensure!(
        manifest["schema"] == "parallel.seed_experiment.v1"
            || manifest["schema"] == "parallel.seed_experiment.v2",
        "experiment manifest absent or unsupported"
    );
    ensure!(
        output.is_absolute()
            && output.starts_with(&experiment)
            && !output
                .components()
                .any(|c| matches!(c, std::path::Component::ParentDir)),
        "experiment output must remain inside its directory"
    );
    let ancestor = output
        .ancestors()
        .find(|p| p.exists())
        .context("output ancestor absent")?;
    for path in output
        .ancestors()
        .take_while(|p| p.starts_with(&experiment))
    {
        if let Ok(metadata) = fs::symlink_metadata(path) {
            ensure!(
                !metadata.file_type().is_symlink(),
                "experiment output symlink rejected"
            );
        }
    }
    ensure!(
        ancestor.canonicalize()?.starts_with(&experiment),
        "experiment output symlink escapes its directory"
    );
    Ok(())
}

/// Work limits, qualification mode and optional proposal inputs for a search.
#[derive(Clone, Copy)]
pub struct SearchOptions<'a> {
    pub evaluations: usize,
    pub seconds: u64,
    pub qualification: bool,
    pub seeds: Option<&'a Path>,
    pub vocabulary: Option<&'a Path>,
}

pub fn search_seeded(
    prepared: &Path,
    cut: usize,
    output: &Path,
    options: SearchOptions<'_>,
) -> Result<()> {
    search_seeded_with_rng(prepared, cut, output, options, SEED)
}

pub fn search_seeded_with_rng(
    prepared: &Path,
    cut: usize,
    output: &Path,
    options: SearchOptions<'_>,
    rng_seed: u64,
) -> Result<()> {
    let artifact_root = crate::prepare::workspace_root(prepared)?;
    search_seeded_with_rng_in_root(&artifact_root, prepared, cut, output, options, rng_seed)
}

/// Resolve the declared custody root before accepting any search or creating
/// candidate intents. CLI callers must pass their --root explicitly; a copied
/// preparation need not contain a Cargo workspace just to retain its receipts.
pub fn search_seeded_with_rng_in_root(
    artifact_root: &Path,
    prepared: &Path,
    cut: usize,
    output: &Path,
    options: SearchOptions<'_>,
    rng_seed: u64,
) -> Result<()> {
    search_inner(
        artifact_root,
        prepared,
        cut,
        output,
        options,
        rng_seed,
        None,
    )
}

/// Called only after native worker and fit authority validate the immutable genome task.
pub(crate) fn search_native_genome(
    root: &Path,
    reference: &crate::native_graph::Artifact,
    task: &crate::native_genome::GenomeTask,
    candidate: &crate::native_genome::Candidate,
    parent_row: Option<&serde_json::Value>,
) -> Result<()> {
    task.validate()?;
    search_inner(
        root,
        &resolve(root, &candidate.context.prepared)?,
        candidate.context.cut,
        &resolve(root, &task.output)?,
        crate::gp::SearchOptions {
            evaluations: task.evaluations,
            seconds: task.seconds,
            qualification: task.qualification,
            seeds: None,
            vocabulary: Some(&resolve(root, &task.vocabulary.path)?),
        },
        task.rng_seed,
        Some((reference, task, candidate, parent_row)),
    )
}

/// Metadata only: exact legacy GP initial population before fit admission or numerical work.
pub fn metadata_initial_population(rng_seed: u64) -> Result<Vec<String>> {
    let mut population = vec!["(pipe X (ridge))".to_owned()];
    let mut seen = population.iter().cloned().collect();
    fill_initial_population(&mut population, &mut seen, rng_seed, None)?;
    Ok(population)
}
/// Appends adapted bootstrap programs and returns, for each appended slot, its
/// original generator draw and generator seed. The search executes the single
/// adaptation of that original; it never adapts an already adapted program.
fn fill_initial_population(
    population: &mut Vec<String>,
    seen: &mut BTreeSet<String>,
    rng_seed: u64,
    vocab: Option<&crate::grammar::Vocab>,
) -> Result<Vec<(String, u64)>> {
    let mut seed = rng_seed;
    let mut originals = Vec::new();
    while population.len() < 80 {
        let generator_seed = seed;
        let raw = feat_math::evolve::random_canonical_pipeline(seed, 6);
        seed = seed.wrapping_add(1);
        let g = if let Some(v) = vocab {
            match v.project_proposal(&raw) {
                Ok(g) => g,
                Err(_) => continue,
            }
        } else {
            raw.clone()
        };
        ensure!(
            seed.wrapping_sub(rng_seed) < 100000,
            "unable to construct diverse grammar population"
        );
        if seen.insert(g.clone()) {
            population.push(g);
            originals.push((raw, generator_seed));
        }
    }
    Ok(originals)
}
