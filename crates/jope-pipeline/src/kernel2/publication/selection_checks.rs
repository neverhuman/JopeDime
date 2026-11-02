use super::*;

// laneD:B18-pareto-archive begin
/// The archive is published exactly when it was selected, reads under the archive reader's own
/// refusals, and is byte for byte the archive rebuilt from the re-derived observations.
pub(super) fn check_archive(
    root: &Path,
    s: &Search,
    observations: &[Kernel2Observation],
) -> Result<()> {
    let path = Path::new(super::archive::PATH);
    let Some(record) = s.engine_selection.as_ref().and_then(|e| e.archive.as_ref()) else {
        ensure!(
            std::fs::symlink_metadata(root.join(path)).is_err(),
            "an archive is published without its selection"
        );
        return Ok(());
    };
    let digest = record
        .sha256
        .as_deref()
        .context("the archive record has no digest")?;
    ensure!(
        std::fs::symlink_metadata(root.join(path)).is_ok(),
        "a selected archive is not published"
    );
    let bytes = read_file(root, path, digest)?;
    let text = std::str::from_utf8(&bytes).context("the archive is not UTF-8")?;
    let published = Archive::from_json(text).map_err(anyhow::Error::msg)?;
    ensure!(
        published.cost_version() == record.cost_version,
        "the archive's cost version disagrees with its record"
    );
    ensure!(
        super::archive::build(observations)?.to_json().as_bytes() == bytes.as_slice(),
        "the archive does not re-derive from the observations"
    );
    Ok(())
}
// laneD:B18-pareto-archive end

// laneD:B20-rewrite-simplify begin
/// The move records are published exactly when a move was selected, match their digest, and
/// re-derive (`moves::check`): which parents each round's passes read, each proposal from its
/// parent, each outcome from the measurements, and the pool refusal count. Without a selected
/// move no measurement has a move's origin.
pub(super) fn check_moves(
    root: &Path,
    r: &Receipt,
    measurements: &[Measurement],
    observations: &[Kernel2Observation],
    evaluator: &super::Kernel2Evaluator,
) -> Result<()> {
    use super::search::moves;
    let s = &r.search;
    let path = Path::new(moves::PATH);
    let Some(record) = s.engine_selection.as_ref().and_then(|e| e.moves.as_ref()) else {
        ensure!(
            std::fs::symlink_metadata(root.join(path)).is_err()
                && measurements.iter().all(|m| {
                    // laneD:B19-minimax-linearize: the linearization pass's origin too.
                    m.observation.origin != super::REWRITE
                        && m.observation.origin != super::LINEARIZE
                }),
            "move evidence is published without its selection"
        );
        return Ok(());
    };
    let digest = record
        .sha256
        .as_deref()
        .context("the moves record has no digest")?;
    let published: Vec<moves::MoveRecord> =
        serde_json::from_slice(&read_file(root, path, digest)?)?;
    let width = r
        .dataset
        .columns
        .checked_sub(1)
        .context("no feature columns")? as usize;
    moves::check(&published, s, measurements, observations, evaluator, width)
}
// laneD:B20-rewrite-simplify end

pub(super) fn check_seeding(s: &Search, measurements: &[Measurement]) -> Result<()> {
    let seeding = &s.seeding;
    let mut distinct = BTreeSet::new();
    ensure!(
        seeding.seeds.len() <= crate::search::ELITES
            && seeding.seeds.len() as u64 <= seeding.offered
            && seeding.parent_receipt_sha256.is_none(),
        "unsupported or invalid seeding record"
    );
    for seed in &seeding.seeds {
        ensure!(
            seed.source == crate::seeds::FROM_SEEDS_FILE
                && seeding.seeds_file_sha256.is_some()
                && distinct.insert(&seed.genome)
                && seed.genome != crate::search::BOOTSTRAP_GENOME
                && seed.program_sha256 == sha256_hex(seed.genome.as_bytes())
                && measurements
                    .iter()
                    .any(|m| m.observation.genome == seed.genome
                        && m.observation.origin == crate::search::INCUMBENT
                        && m.observation.generation == 0),
            "seed was not re-measured in generation zero"
        );
    }
    ensure!(
        measurements
            .first()
            .is_some_and(|m| m.observation.genome == crate::search::BOOTSTRAP_GENOME
                && m.observation.origin == crate::search::BOOTSTRAP
                && m.observation.generation == 0),
        "missing bootstrap measurement"
    );
    Ok(())
}
