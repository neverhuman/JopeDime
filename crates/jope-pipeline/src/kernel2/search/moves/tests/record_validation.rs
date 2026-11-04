use super::*;

#[test]
fn verify_refuses_records_off_the_ranked_parents_and_inexact_children() {
    let temp = tempfile::tempdir().unwrap();
    let opt = run_options(temp.path(), 40, four_parents(), simplify());
    let receipt = run::run(&opt).unwrap();
    let output = &opt.run.output;
    let originals: Vec<(PathBuf, Vec<u8>)> = [
        "receipt.json",
        "search/cut-0/search.json",
        "search/cut-0/observations.json",
        PATH,
    ]
    .iter()
    .map(|r| (output.join(r), std::fs::read(output.join(r)).unwrap()))
    .collect();
    let restore = || {
        for (file, bytes) in &originals {
            std::fs::write(file, bytes).unwrap();
        }
    };
    let engine = receipt.search.engine_selection.clone().unwrap();
    let moves = published_moves(&opt);
    let measurements = published_measurements(&opt);
    // The published count re-derives from the records.
    assert_eq!(
        engine.moves.as_ref().unwrap().refused_by_pool,
        Some(refused_by_pool(&moves))
    );
    let mut errors = Vec::new();
    // Records deleted: an unchanged, a refused or a duplicate one, whichever the run has.
    let mut deleted = 0;
    for outcome in [UNCHANGED, REFUSED, DUPLICATE] {
        let Some(at) = moves.iter().position(|m| m.outcome == outcome) else {
            continue;
        };
        let mut edited = moves.clone();
        edited.remove(at);
        let bytes = serde_json::to_vec_pretty(&edited).unwrap();
        let path = rewrite(
            output,
            &receipt,
            Some(bytes.as_slice()),
            Some(digested(&engine, &bytes)),
        );
        let label = format!("{outcome} record removed");
        errors.push(refusal_of(&path, &label, &OFF_THE_PARENTS));
        restore();
        deleted += 1;
    }
    assert!(deleted > 0, "the run has no record that measures nothing");
    // The last record of an earlier round's pass removed: that pass now stops short of its
    // parents, which only the run's last round may do.
    let last = (0..moves.len())
        .filter(|&i| moves[i].outcome != MEASURED)
        .find(|&i| {
            moves[i].round + 1 < receipt.search.generations
                && moves.get(i + 1).is_none_or(|n| n.round != moves[i].round)
        })
        .expect("an earlier round whose last record measures nothing");
    let mut edited = moves.clone();
    edited.remove(last);
    let bytes = serde_json::to_vec_pretty(&edited).unwrap();
    let path = rewrite(
        output,
        &receipt,
        Some(bytes.as_slice()),
        Some(digested(&engine, &bytes)),
    );
    errors.push(refusal_of(
        &path,
        "a round's last record removed",
        &OFF_THE_PARENTS[1..],
    ));
    restore();
    // A record for a program the pass did not read: the bootstrap, which has no expression stage,
    // so its record re-derives (nothing to rewrite) and only the parent list refuses it.
    let bootstrap = crate::search::BOOTSTRAP_GENOME;
    let proposal = propose(&evaluator(), bootstrap, 1).unwrap();
    assert!(proposal.rewrites.is_empty() && proposal.program.is_none());
    let mut edited = moves.clone();
    edited.insert(
        0,
        MoveRecord {
            id: REWRITE_SATURATE.into(),
            round: 0,
            parent_sha256: crate::sha256_hex(bootstrap.as_bytes()),
            program: None,
            outcome: UNCHANGED.into(),
            reason: None,
            measurement: None,
            rewrites: Vec::new(),
            linearization: None,
        },
    );
    let bytes = serde_json::to_vec_pretty(&edited).unwrap();
    let path = rewrite(
        output,
        &receipt,
        Some(bytes.as_slice()),
        Some(digested(&engine, &bytes)),
    );
    errors.push(refusal_of(
        &path,
        "record for an unread parent",
        &OFF_THE_PARENTS[..1],
    ));
    restore();
    // Two records of one round swapped.
    let pair = moves
        .windows(2)
        .position(|w| w[0].round == w[1].round && w[0].parent_sha256 != w[1].parent_sha256)
        .expect("a round with two records");
    let mut edited = moves.clone();
    edited.swap(pair, pair + 1);
    let bytes = serde_json::to_vec_pretty(&edited).unwrap();
    let path = rewrite(
        output,
        &receipt,
        Some(bytes.as_slice()),
        Some(digested(&engine, &bytes)),
    );
    errors.push(refusal_of(&path, "records swapped", &OFF_THE_PARENTS[..1]));
    restore();
    // The pool refusal count changed or dropped, the records unchanged.
    for (label, count) in [
        ("pool refusal count", Some(refused_by_pool(&moves) + 1)),
        ("pool refusal count dropped", None),
    ] {
        let mut record = engine.clone();
        record.moves.as_mut().unwrap().refused_by_pool = count;
        let path = rewrite(output, &receipt, None, Some(record));
        errors.push(refusal_of(
            &path,
            label,
            &["the pool refusal count does not re-derive from the move records"],
        ));
        restore();
    }
    // A rewritten child's scores or fault counts edited: it no longer scores as its parent. The
    // child is not the selected program, and under RejectHard neither edit moves its rank.
    let child = moves
        .iter()
        .filter(|m| m.outcome == MEASURED)
        .filter_map(|m| m.measurement)
        .find(|&i| measurements[i].observation.genome != receipt.search.best.observation.genome)
        .expect("a rewritten child that was not selected");
    let edits: [MeasurementEdit; 3] = [
        ("child training R2", |m| {
            m.observation.train_r2 = m.observation.train_r2.map(|v| v - 1e-9)
        }),
        ("child RMSE", |m| {
            m.observation.rmse = m.observation.rmse.map(|v| v * (1.0 + 1e-9) + 1e-12)
        }),
        ("child evaluated cells", |m| {
            m.faults.evaluated_cells += 1;
            let mut totals = feat_math::kernel::admission::FaultTotals::default();
            totals.evaluated_cells = m.faults.evaluated_cells;
            totals.domain_cells = m.faults.domain_cells;
            totals.saturation_cells = m.faults.saturation_cells;
            m.domain_fault_rate = totals.domain_rate();
            m.saturation_fault_rate = totals.saturation_rate();
        }),
    ];
    for (label, edit) in edits {
        let mut edited = measurements.clone();
        edit(&mut edited[child]);
        assert_ne!(edited, measurements, "{label}: the edit was not applied");
        let path = rewrite_measurements(output, &receipt, &edited);
        errors.push(refusal_of(
            &path,
            label,
            &["a rewritten program does not score bit for bit as its parent"],
        ));
        restore();
    }
    publication::verify_run(&path).unwrap();
    let root = output.display().to_string();
    for error in errors {
        lint(&error.replace(&root, "<run>"));
    }
}

#[test]
fn an_unstarted_record_ends_its_pass() {
    let temp = tempfile::tempdir().unwrap();
    // Every seconds bound has passed: generation 0 measures the bootstrap and the two seeds, and
    // the pass stops before its first parent, in a run that stops on its seconds bound.
    let mut opt = run_options(temp.path(), 40, four_parents(), simplify());
    let seeds = temp.path().join("two-seeds.json");
    let file = crate::seeds::SeedsFile::new(
        None,
        [ONE_TIMES, FOLDED]
            .iter()
            .map(|genome| crate::seeds::SeedGenome {
                genome: (*genome).into(),
                program_sha256: crate::sha256_hex(genome.as_bytes()),
                bank_entry_sha256: None,
                receipt_sha256: None,
            })
            .collect(),
    );
    std::fs::write(&seeds, serde_json::to_vec(&file).unwrap()).unwrap();
    opt.run.seed_genomes = Some(seeds);
    opt.run.clock = Clock::Expired;
    let receipt = run::run(&opt).unwrap();
    assert_eq!(receipt.search.stopped_by, crate::search::STOPPED_BY_SECONDS);
    assert_eq!(receipt.search.generations, 1);
    assert!(published_moves(&opt).is_empty());
    let output = &opt.run.output;
    let path = output.join("receipt.json");
    publication::verify_run(&path).unwrap();
    // The two parents round 0's pass would read, in rank order.
    let observations: Vec<Kernel2Observation> = published_measurements(&opt)
        .iter()
        .map(|m| m.derive(FaultPolicy::RejectHard).unwrap())
        .collect();
    let width = (receipt.dataset.columns - 1) as usize;
    let ranked = owned(ranked_parents(&evaluator(), &observations, REWRITE_SATURATE, 4).unwrap());
    assert_eq!(ranked.len(), 2);
    let unstarted = |(genome, sha256): &(String, String)| {
        let proposal = propose(&evaluator(), genome, width).unwrap();
        assert!(proposal.admitted, "{genome}");
        MoveRecord {
            id: REWRITE_SATURATE.into(),
            round: 0,
            parent_sha256: sha256.clone(),
            program: proposal.program,
            outcome: NOT_STARTED.into(),
            reason: None,
            measurement: None,
            rewrites: proposal.rewrites,
            linearization: None,
        }
    };
    let engine = receipt.search.engine_selection.clone().unwrap();
    // One unstarted record for the first parent: what a run whose start guard refused that
    // measurement publishes.
    let one = vec![unstarted(&ranked[0])];
    let bytes = serde_json::to_vec_pretty(&one).unwrap();
    let path = rewrite(
        output,
        &receipt,
        Some(bytes.as_slice()),
        Some(digested(&engine, &bytes)),
    );
    publication::verify_run(&path).unwrap();
    // A record after it: the pass had ended.
    let two = vec![unstarted(&ranked[0]), unstarted(&ranked[1])];
    let bytes = serde_json::to_vec_pretty(&two).unwrap();
    let path = rewrite(
        output,
        &receipt,
        Some(bytes.as_slice()),
        Some(digested(&engine, &bytes)),
    );
    let error = refusal_of(
        &path,
        "a record after an unstarted one",
        &["move records continue after their pass ended"],
    );
    lint(&error.replace(&output.display().to_string(), "<run>"));
    // The second parent alone: not the first ranked parent.
    let skipped = vec![unstarted(&ranked[1])];
    let bytes = serde_json::to_vec_pretty(&skipped).unwrap();
    let path = rewrite(
        output,
        &receipt,
        Some(bytes.as_slice()),
        Some(digested(&engine, &bytes)),
    );
    refusal_of(&path, "the first parent skipped", &OFF_THE_PARENTS[..1]);
}
