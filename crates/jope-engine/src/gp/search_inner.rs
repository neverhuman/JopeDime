fn search_inner(
    artifact_root: &Path,
    prepared: &Path,
    cut: usize,
    output: &Path,
    options: SearchOptions<'_>,
    rng_seed: u64,
    native: Option<(
        &crate::native_graph::Artifact,
        &crate::native_genome::GenomeTask,
        &crate::native_genome::Candidate,
        Option<&serde_json::Value>,
    )>,
) -> Result<()> {
    include!("search_inner/setup.rs");
    for i in observations.len()..evaluations {
        include!("search_inner/slot.rs");
        include!("search_inner/dispatch.rs");
        include!("search_inner/observation.rs");
    }
    include!("search_inner/finalize.rs");
}
