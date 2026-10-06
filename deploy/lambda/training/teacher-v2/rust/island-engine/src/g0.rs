// Score the finite G0 batch with the production S1 arithmetic; never dispatch fits.
use super::*;
use std::io::Write;

pub(super) fn scoring_engine(
    context: String,
    sample: SampleManifest,
    projection: Vec<f32>,
) -> Result<Engine> {
    validate_sample_mode(&sample, 2)?;
    if sample.context_sha256 != context
        || projection.len() != SIGNATURE * sample.oof_rows
        || projection.iter().any(|x| !x.is_finite())
    {
        bail!("G0 context/projection mismatch")
    }
    Ok(Engine {
        inner_cv_folds: 2,
        context,
        proposal_binary: PathBuf::new(),
        all: HashMap::new(),
        islands: POLICIES
            .iter()
            .map(|policy| Island {
                policy: *policy,
                population: vec![],
                archive: vec![],
                completed: 0,
            })
            .collect(),
        seen: HashSet::new(),
        applied_results: HashSet::new(),
        active: HashSet::new(),
        active_semantics: HashSet::new(),
        pending: HashMap::new(),
        projection,
        sample,
        rng: ChaCha20Rng::seed_from_u64(0),
        next_birth: 0,
        next_draw: 0,
        proposal_seq: 0,
        new_completed: 0,
        migration_epoch: 0,
        number_re: Regex::new(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?")?,
    })
}

fn scored(e: &Engine, program: Program, record: &Value, ordinal: usize) -> Result<Value> {
    if record["context_sha256"].as_str() != Some(e.context.as_str())
        || record["program"] != serde_json::to_value(&program)?
        || record["schema"] != "jope.s1-evaluation/v1"
        || record["input_semantics"] != "unscaled_fit_rows_v1"
        || record["fold_protocol"] != S1_TWO_FOLD
        || record["inner_cv_folds"] != 2
    {
        bail!("G0 record is not the issued two-fold context/program")
    }
    let status = record["status"]
        .as_str()
        .ok_or_else(|| anyhow!("G0 status missing"))?;
    if !matches!(status, "completed" | "failed" | "timeout") {
        bail!("G0 requires terminal dispositions")
    }
    let mut row = json!({"schema":"jope.island-scored-evaluation/s1-v1", "selection_protocol":S1_TWO_FOLD,
        "context_sha256":e.context, "program":program, "program_sha256":program.program_sha256,
        "status":status, "stage":"initialization", "g0_ordinal":ordinal,
        "island_id":ordinal % ISLANDS, "origin_island":ordinal % ISLANDS,
        "proposal_seq":ordinal+1, "proposal_draw":ordinal, "born_completed":0,
        "result_key":format!("g0/{}", program.program_sha256)});
    if status == "completed" {
        let i = e.score_record(
            record,
            program,
            row["result_key"].as_str().unwrap().to_owned(),
        )?;
        for (key, value) in [
            ("fitness", json!(i.fitness)),
            ("cv_gate_passed", json!(i.cv_gate_passed)),
            ("cv_fold_wins", json!(i.cv_fold_wins)),
            ("cv_geomean_rmse_ratio", json!(i.cv_geomean_rmse_ratio)),
            ("cv_mean_log_ratio", json!(i.cv_mean_log_ratio)),
            ("cv_tail_log_ratio", json!(i.cv_tail_log_ratio)),
            ("cv_log_ratio_sd", json!(i.cv_log_ratio_sd)),
            ("test_rmse", Value::Null),
            ("test_rmse_ratio", Value::Null),
            ("complexity_open_parens", json!(i.complexity_open_parens)),
            ("full_prediction_sha256", json!(i.full_prediction_sha256)),
            (
                "semantic_signature_f32_b64",
                json!(encode_f32(&i.signature)),
            ),
            (
                "sampled_absolute_error_f32_b64",
                json!(encode_f32(&i.cases)),
            ),
            ("roots", json!(i.roots)),
            ("billed_ms", Value::Null),
            ("max_memory_used_mb", Value::Null),
        ] {
            row[key] = value;
        }
    }
    validate_history_in(&row, &e.context, S1_TWO_FOLD)?;
    Ok(row)
}

fn batch(e: &Engine, programs: &[Program], records: &[Value]) -> Result<Vec<Value>> {
    if programs.len() != 96 || records.len() != 96 {
        bail!("G0 needs exactly 96 terminal slots")
    }
    let mut unique = HashSet::new();
    programs
        .iter()
        .zip(records)
        .enumerate()
        .map(|(n, (program, record))| {
            validate_program(program)?;
            if program.birth_seq != n as u64
                || !program.parents.is_empty()
                || program.operator != "grammar_random"
                || !unique.insert(program.program_sha256.clone())
            {
                bail!("G0 is not the fresh unique 96-program prior")
            }
            scored(e, program.clone(), record, n)
        })
        .collect()
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

pub(super) fn run(args: &[String]) -> Result<()> {
    let context = arg_value(args, "--context")?;
    let sample: SampleManifest = serde_json::from_slice(&fs::read(arg_value(args, "--sample")?)?)?;
    let raw = fs::read(arg_value(args, "--projection")?)?;
    if raw.len() % 4 != 0 || sha256(&raw) != sample.projection_f32_sha256 {
        bail!("G0 projection bytes changed")
    }
    let projection = raw
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect();
    let engine = scoring_engine(context, sample, projection)?;
    let programs: Vec<Program> = serde_json::from_slice(&fs::read(arg_value(args, "--seeds")?)?)?;
    let records: Vec<Value> =
        serde_json::from_slice(&fs::read(arg_value(args, "--evaluations")?)?)?;
    // Validate the entire batch before publishing any index or seen rows.
    let rows = batch(&engine, &programs, &records)?;
    let out = PathBuf::from(arg_value(args, "--out")?);
    fs::create_dir(&out)?;
    let mut index = Vec::new();
    let mut seen = Vec::new();
    let mut eligible = HashSet::new();
    for row in &rows {
        let mut line = serde_json::to_vec(row)?;
        line.push(b'\n');
        seen.extend_from_slice(&line);
        if row["status"] == "completed" {
            index.extend_from_slice(&line);
            if row["cv_gate_passed"] == true {
                eligible.insert(row["full_prediction_sha256"].as_str().unwrap());
            }
        }
    }
    let mut gzip = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
    gzip.write_all(&index)?;
    let index = gzip.finish()?;
    write_new(&out.join("seed-index.jsonl.gz"), &index)?;
    write_new(&out.join("seen.jsonl"), &seen)?;
    write_new(
        &out.join("manifest.json"),
        &serde_json::to_vec(&json!({
        "schema":"jope.s1-g0-scored/v1", "context_sha256":engine.context, "selection_protocol":S1_TWO_FOLD,
        "terminal_slots":96, "attempt_accounting_verified":false, "completed":rows.iter().filter(|r|r["status"]=="completed").count(),
        "unique_eligible_semantics":eligible.len(), "ready":eligible.len()>=POPULATION,
        "seed_index_sha256":sha256(&index),"seen_sha256":sha256(&seen),
        "sampling_resume_qualified":false}))?,
    )?;
    File::open(&out)?.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn g0_scores_with_two_fold_arithmetic_and_refuses_substitutions() {
        let (e, r, p) = crate::tests::two_fold_score_fixture();
        let mut r = r;
        r["program"] = serde_json::to_value(&p).unwrap();
        r["context_sha256"] = json!(e.context);
        let a = scored(&e, p.clone(), &r, 0).unwrap();
        assert!(a["cv_gate_passed"].as_bool().unwrap());
        let cached: SemanticRow = serde_json::from_value(a.clone()).unwrap();
        assert!(Individual::from_semantic_in(cached, S1_TWO_FOLD).is_ok());
        let mut one_win = a.clone();
        one_win["cv_fold_wins"] = json!(1);
        assert!(
            Individual::from_semantic_in(serde_json::from_value(one_win).unwrap(), S1_TWO_FOLD)
                .is_err()
        );
        let mut timed = r.clone();
        timed["wall_us"] = json!(999999);
        assert_eq!(a, scored(&e, p.clone(), &timed, 0).unwrap());
        let mut bad = r.clone();
        bad["official_test"] = json!({"targets":[0.0]});
        assert!(scored(&e, p.clone(), &bad, 0).is_err());
        let mut bad = r;
        bad["context_sha256"] = json!("other");
        assert!(scored(&e, p.clone(), &bad, 0).is_err());
        assert!(batch(&e, &[p], &[]).is_err());
    }

    #[test]
    fn g0_keeps_failed_and_timeout_slots_and_refuses_nonterminal_batches() {
        let (e, original, _) = crate::tests::two_fold_score_fixture();
        let mut programs = Vec::new();
        let mut records = Vec::new();
        for n in 0..96 {
            let dsl = format!("(pipe X (ridge lambda_frac={}))", (n + 1) as f64 / 10000.0);
            let program = Program {
                program_sha256: sha256(dsl.as_bytes()),
                dsl,
                birth_seq: n,
                parents: vec![],
                operator: "grammar_random".to_owned(),
            };
            let mut record = original.clone();
            record["context_sha256"] = json!(e.context);
            record["program"] = serde_json::to_value(&program).unwrap();
            if n < 2 {
                record["status"] = json!(if n == 0 { "failed" } else { "timeout" });
            }
            programs.push(program);
            records.push(record);
        }
        let rows = batch(&e, &programs, &records).unwrap();
        assert_eq!(rows.len(), 96);
        assert_eq!(
            rows.iter().filter(|r| r["status"] == "completed").count(),
            94
        );
        assert!(rows[0].get("fitness").is_none());
        assert_eq!(rows[1]["status"], "timeout");
        records[0]["status"] = json!("unknown");
        assert!(batch(&e, &programs, &records).is_err());
        records[0]["status"] = json!("failed");
        programs[95] = programs[0].clone();
        assert!(batch(&e, &programs, &records).is_err());
    }
}
