use super::*;

pub(super) struct MineArgs {
    pub(super) state: Option<PathBuf>,
    pub(super) directory: PathBuf,
    pub(super) min_datasets: u32,
    pub(super) min_families: u32,
    pub(super) max_stages: u32,
    pub(super) families: Option<PathBuf>,
    pub(super) private_log: Option<PathBuf>,
}

pub(super) fn mine_verb(args: &MineArgs, env: &Env, policy: &Policy) -> Result<MotifTable> {
    // Every output path and every parameter is checked before the State is read, so a
    // refusal costs nothing.
    policy.check_private(&args.directory)?;
    if let Some(log) = &args.private_log {
        policy.check_private(log)?;
    }
    let families = match &args.families {
        Some(path) => Some(FamilyMap::parse(
            &std::fs::read(path).coded("invalid_family_map")?,
        )?),
        None => None,
    };
    let parameters = MineParameters {
        min_datasets: args.min_datasets,
        min_families: args.min_families,
        max_stages: args.max_stages,
        family_map_sha256: families.as_ref().map(|map| map.sha256().to_owned()),
    };
    parameters.check()?;
    let set = read_state(env, args.state.as_deref())?;
    // Nothing admitted means nothing was examined: no table is written.
    ensure!(!set.programs().is_empty(), Code("no_admitted_programs"));
    let table = mine(&set, families.as_ref(), parameters)?;
    let inputs = input_manifest(&set, &table)?;
    let table_bytes = table.to_json().coded("io")?;
    let input_bytes = serde_json::to_vec_pretty(&inputs).coded("io")?;
    write_private_bundle(policy, &args.directory, &table_bytes, &input_bytes)?;
    Ok(table)
}

/// Create the new 0700 directory and write the table and its input list into it, 0600 each.
/// If a write fails, the files of this run and then the directory are removed, so a failed
/// run never leaves a table without its inputs.
fn write_private_bundle(
    policy: &Policy,
    directory: &Path,
    table: &[u8],
    inputs: &[u8],
) -> Result<()> {
    let created = create_private_dir(policy, directory)?;
    let table_path = created.join(TABLE_FILE);
    let inputs_path = created.join(INPUTS_FILE);
    let written = write_private_file(policy, &table_path, table)
        .and_then(|()| write_private_file(policy, &inputs_path, inputs));
    if written.is_err() {
        // Best effort: a failed removal cannot be reported more usefully than the write error.
        let _ = std::fs::remove_file(&table_path);
        let _ = std::fs::remove_file(&inputs_path);
        let _ = std::fs::remove_dir(&created);
    }
    written
}

#[derive(Serialize)]
struct MineJson<'a> {
    schema: &'static str,
    ok: bool,
    table_sha256: &'a str,
    fragments: u32,
    quarantined: u32,
    stage_counts: StageCounts,
    parameters: ParametersJson,
    inputs: InputsJson,
}

#[derive(Serialize)]
struct ParametersJson {
    min_datasets: u32,
    min_families: u32,
    max_stages: u32,
    family_map: bool,
}

#[derive(Serialize)]
struct InputsJson {
    qualified: u32,
    used: u32,
    excluded_adaptive: u32,
    excluded_at_source: u32,
    skipped_form: u32,
    skipped_reserved: u32,
    skipped_slots: u32,
    skipped_unsupported: u32,
    dropped_untyped: u32,
}

pub(super) fn render_mine(table: &MotifTable, json: bool, out: &mut dyn Write) -> Result<()> {
    let body = &table.body;
    let fragments = u32::try_from(body.motifs.len())?;
    let quarantined = u32::try_from(
        body.motifs
            .iter()
            .filter(|motif| motif.lifecycle == Lifecycle::Quarantined)
            .count(),
    )?;
    if json {
        let parameters = &body.parameters;
        let inputs = &body.inputs;
        let document = MineJson {
            schema: MINE_SCHEMA,
            ok: true,
            table_sha256: &table.table_sha256,
            fragments,
            quarantined,
            stage_counts: StageCounts::of(table),
            parameters: ParametersJson {
                min_datasets: parameters.min_datasets,
                min_families: parameters.min_families,
                max_stages: parameters.max_stages,
                family_map: parameters.family_map_sha256.is_some(),
            },
            inputs: InputsJson {
                qualified: inputs.qualified,
                used: inputs.used,
                excluded_adaptive: inputs.excluded_adaptive,
                excluded_at_source: body.source.excluded_at_source,
                skipped_form: inputs.skipped_form,
                skipped_reserved: inputs.skipped_reserved,
                skipped_slots: inputs.skipped_slots,
                skipped_unsupported: inputs.skipped_unsupported,
                dropped_untyped: inputs.dropped_untyped,
            },
        };
        writeln!(out, "{}", serde_json::to_string(&document)?)?;
    } else if fragments == 0 {
        // Nothing was kept, so nothing is quarantined either; the inputs are still listed.
        writeln!(
            out,
            "Wrote a private table with no fragments; every examined input is listed."
        )?;
    } else if quarantined == fragments {
        writeln!(
            out,
            "Wrote a private table with {fragments} fragments (all quarantined)."
        )?;
    } else {
        writeln!(
            out,
            "Wrote a private table with {fragments} fragments ({quarantined} quarantined)."
        )?;
    }
    Ok(())
}
