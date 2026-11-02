use super::*;

pub(super) fn parse_binary_population(
    bytes: &[u8],
    expected: usize,
    task: &TaskSpec,
    schema: &DataSchema,
) -> Result<Vec<BinaryMember>> {
    let text = std::str::from_utf8(bytes).context("binary generation-zero file is not UTF-8")?;
    let mut population = Vec::new();
    let mut seen = HashSet::new();
    for (line_number, line) in text.lines().enumerate() {
        let body = line.split('#').next().unwrap_or("").trim();
        if body.is_empty() {
            continue;
        }
        let genome = GenomeIr::parse(body).with_context(|| {
            format!(
                "binary generation-zero line {} does not parse",
                line_number + 1
            )
        })?;
        if !matches!(genome, GenomeIr::Pipeline(_)) {
            bail!(
                "binary generation-zero line {} is not a pipeline",
                line_number + 1
            );
        }
        let canonical = genome.canonical();
        if canonical != body {
            bail!(
                "binary generation-zero line {} is not canonical",
                line_number + 1
            );
        }
        validate_v3_genome(&genome, task, schema).with_context(|| {
            format!(
                "binary generation-zero line {} is not a valid binary genome",
                line_number + 1
            )
        })?;
        let semantic_sha256 = sha256_bytes(canonical.as_bytes());
        if !seen.insert(semantic_sha256.clone()) {
            bail!(
                "binary generation-zero line {} duplicates an earlier genome",
                line_number + 1
            );
        }
        population.push(BinaryMember {
            canonical_genome: canonical,
            semantic_sha256,
            ancestry: BinaryAncestry {
                operation: "seed".to_string(),
                parents: Vec::new(),
                retry_count: 0,
            },
        });
    }
    if population.len() != expected {
        bail!(
            "binary population requires {expected} genomes, found {}",
            population.len()
        );
    }
    Ok(population)
}
