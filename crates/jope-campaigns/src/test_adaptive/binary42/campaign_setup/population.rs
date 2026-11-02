use super::*;

pub(crate) fn seed_population(features: usize) -> Result<Vec<MemberRecord>> {
    seed_population_with_winners(features, &[])
}

pub(crate) fn winner_seed_snapshot_sha256(sources: &[WinnerSeedSource]) -> Result<String> {
    Ok(sha256_bytes(&canonical_json(&sources)?))
}

pub(crate) fn seed_population_with_winners(
    features: usize,
    winner_sources: &[WinnerSeedSource],
) -> Result<Vec<MemberRecord>> {
    if sha256_bytes(BINARY_SEEDS) != SEED_BINARY_SHA256 {
        bail!("canonical binary seed hash mismatch");
    }
    if winner_sources.len() > POPULATION {
        bail!("Binary42 confirmed winner seed bank exceeds the population");
    }
    let base = test_adaptive_seed_population(BINARY_SEEDS, features)?;
    let mut seen = BTreeSet::new();
    let mut genomes = Vec::with_capacity(POPULATION);
    for source in winner_sources {
        if sha256_bytes(source.canonical_genome.as_bytes()) != source.semantic_sha256 {
            bail!("Binary42 winner seed semantic identity changed");
        }
        if !seen.insert(source.semantic_sha256.clone()) {
            bail!("Binary42 confirmed winner seed bank contains duplicate semantics");
        }
        genomes.push(source.canonical_genome.clone());
    }
    for member in base {
        if seen.insert(member.semantic_sha256) {
            genomes.push(member.canonical_genome);
        }
        if genomes.len() == POPULATION {
            break;
        }
    }
    if genomes.len() != POPULATION {
        bail!("Binary42 winner-augmented seed bank produced a short population");
    }
    let bytes = format!("{}\n", genomes.join("\n")).into_bytes();
    let members = test_adaptive_seed_population(&bytes, features)?
        .into_iter()
        .map(MemberRecord::from_binary)
        .collect::<Result<Vec<_>>>()?;
    if winner_sources
        .iter()
        .zip(&members)
        .any(|(source, member)| source.semantic_sha256 != member.semantic_sha256)
    {
        bail!("Binary42 confirmed winners are not first in generation zero");
    }
    Ok(members)
}
