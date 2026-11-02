use super::*;

#[test]
fn archive_on_and_off_keep_every_measurement_refinement_and_the_winner_bit_for_bit() {
    let temp = tempfile::tempdir().unwrap();
    for (label, refine, trials) in [
        ("coordinate", RefineMethod::Coordinate, 4),
        ("lm", lm(), 12),
    ] {
        let run_with = |archive: bool| {
            let dir = temp.path().join(format!("{label}-{archive}"));
            let opt = opts(
                &dir,
                40,
                "kernel2-baseline",
                FaultPolicy::RejectHard,
                trials,
                selection(refine, archive),
            );
            let receipt = run::run(&opt).unwrap();
            assert_eq!(
                publication::verify_run(&opt.run.output.join("receipt.json")).unwrap(),
                receipt,
                "{label} {archive}"
            );
            (opt, receipt)
        };
        let (off, plain) = run_with(false);
        let (on, archived) = run_with(true);
        for relative in [
            "prepared/prepared.json",
            "search/cut-0/observations.json",
            "search/cut-0/refinements.json",
            "emit/model.rs",
            "emit/manifest.json",
        ] {
            assert_eq!(
                read(&off, relative),
                read(&on, relative),
                "{label} {relative}"
            );
        }
        assert_eq!(plain.search.best, archived.search.best, "{label}");
        assert_eq!(plain.artifacts, archived.artifacts, "{label}");
        // The summaries differ only by the archive record.
        assert_eq!(
            without_archive(&read(&off, "search/cut-0/search.json")),
            without_archive(&read(&on, "search/cut-0/search.json")),
            "{label}"
        );
        // So do the receipts: without the archive record and the summary's digest they are equal.
        let mut expected = archived.clone();
        expected.search.engine_selection = expected
            .search
            .engine_selection
            .filter(|e| e.refine.is_some())
            .map(|e| EngineRecord { archive: None, ..e });
        expected.search_sha256 = plain.search_sha256.clone();
        assert_eq!(expected, plain, "{label}");
        // Off: no sidecar and no record. On: the sidecar and its digest.
        assert!(!off.run.output.join(archive::PATH).exists());
        assert!(!String::from_utf8(read(&off, "search/cut-0/search.json"))
            .unwrap()
            .contains("archive"));
        assert!(!String::from_utf8(read(&off, "receipt.json"))
            .unwrap()
            .contains("archive"));
        let record = archived
            .search
            .engine_selection
            .as_ref()
            .and_then(|e| e.archive.as_ref())
            .unwrap();
        assert_eq!(record.id, FRONT_LOSS_COST);
        assert_eq!(record.cost_version, pareto::PIPELINE_COST_VERSION);
        assert_eq!(
            record.sha256.as_deref(),
            Some(crate::sha256_hex(&read(&on, archive::PATH)).as_str())
        );
    }
}

#[test]
fn every_member_is_non_dominated_and_the_winner_is_on_the_front_or_an_equal_loss_tie() {
    let temp = tempfile::tempdir().unwrap();
    for (label, profile, policy, refine, trials) in [
        (
            "baseline",
            "kernel2-baseline",
            FaultPolicy::RejectHard,
            RefineMethod::Coordinate,
            4,
        ),
        (
            "experimental",
            "kernel2-experimental",
            penalize(false),
            RefineMethod::Coordinate,
            4,
        ),
        ("lm", "kernel2-baseline", FaultPolicy::RejectHard, lm(), 12),
    ] {
        let opt = opts(
            &temp.path().join(label),
            40,
            profile,
            policy,
            trials,
            selection(refine, true),
        );
        check_run(label, &opt, policy);
    }
    // Fronts of two and three programs, the winner the costliest member.
    for (dataset, size) in [(SUPERNOVAE, 2), (BARIUM, 3)] {
        let opt = archived_on(&temp.path().join(dataset), dataset);
        let archive = check_run(dataset, &opt, FaultPolicy::RejectHard);
        assert_eq!(archive.members().len(), size, "{dataset}");
    }
}

/// Runs `opt`, then checks its published archive against its re-derived observations and its
/// winner. The ranked observations hold more programs than the front.
fn check_run(label: &str, opt: &run::Options, policy: FaultPolicy) -> Archive {
    let receipt = run::run(opt).unwrap();
    let archive = published(opt);
    let observations = derived(opt, policy);
    check_front(&archive, &observations);
    let best = receipt.search.best.derive(policy).unwrap();
    check_winner(&archive, &best);
    assert!(
        observations.iter().filter(|o| archive::ranked(o)).count() > archive.members().len(),
        "{label}"
    );
    archive
}

/// Measured programs of seven costs with set scores: the front keeps exactly the four that no
/// simpler program matches, the winner among them.
#[test]
fn a_constructed_front_keeps_exactly_the_programs_no_simpler_one_matches() {
    let e = evaluator();
    let c = data();
    let two_terms = "(pipe X (derive expr=\"sin(0.5*c0)+(0.3*c0)\") (ridge))";
    let scaled = "(pipe X (derive expr=\"sin(0.5*c0)+(0.3*c0)\") (standardize) (ridge))";
    let mixed =
        "(pipe X (map expr=\"sin(0.5*x)\") (derive expr=\"sin(0.7*c0)\" append=false) (ridge))";
    let programs = [
        ("(pipe X (ridge))", 1, 0.10),
        ("(pipe X (standardize) (ridge))", 2, 0.05),
        (SEED, 5, 0.30),
        (SEED_SINE, 6, 0.30),
        (two_terms, 10, 0.60),
        (mixed, 11, 0.55),
        (scaled, 11, 0.90),
    ];
    let observations: Vec<Kernel2Observation> = programs
        .iter()
        .map(|&(genome, expected, score)| {
            assert_eq!(cost(genome), expected, "{genome}");
            let mut o = measured(&e, genome, &c);
            assert!(o.observation.measured(), "{genome}: {:?}", o.observation);
            o.selection_score = Some(score);
            o
        })
        .collect();
    let archive = archive::build(&observations).unwrap();
    check_front(&archive, &observations);
    let kept: Vec<(u32, [u8; 32])> = archive
        .members()
        .iter()
        .map(|m| (m.complexity, m.key))
        .collect();
    assert_eq!(
        kept,
        vec![
            (1, key("(pipe X (ridge))")),
            (5, key(SEED)),
            (10, key(two_terms)),
            (11, key(scaled)),
        ]
    );
    let ranked = e.rank(&observations);
    assert_eq!(ranked[0].observation.genome, scaled);
    check_winner(&archive, ranked[0]);
}

#[test]
fn ties_and_duplicates_settle_identically_under_any_observation_order() {
    let e = evaluator();
    let c = data();
    // Equal cost (two stages and three expression nodes) with equal scores: fewer faulted cells,
    // then the lower key, keeps the place.
    let tied = [
        "(pipe X (derive expr=\"1*c0\") (ridge))",
        "(pipe X (derive expr=\"2*c0\") (ridge))",
        "(pipe X (derive expr=\"3*c0\") (ridge))",
    ];
    let mut observations: Vec<Kernel2Observation> = tied
        .iter()
        .map(|g| {
            let mut o = measured(&e, g, &c);
            o.selection_score = Some(0.5);
            o
        })
        .collect();
    assert!(tied.iter().all(|g| cost(g) == 5));
    assert_eq!(
        observations[0].faults.faulted_cells(),
        observations[1].faults.faulted_cells()
    );
    observations[2].faults.domain_cells = 3;
    observations[2].faults.evaluated_cells = observations[2].faults.evaluated_cells.max(3);
    assert!(observations[2].faults.faulted_cells() > 0);
    // A duplicate of each, and two spellings of one program with one codec key.
    observations.extend(observations.clone());
    let spelled = [
        "(pipe X (ridge lambda_frac=0.000000001))",
        "(pipe X (ridge lambda_frac=1e-9))",
    ];
    assert_ne!(spelled[0], spelled[1]);
    assert_eq!(key(spelled[0]), key(spelled[1]));
    for g in spelled {
        let mut o = measured(&e, g, &c);
        o.selection_score = Some(0.25);
        observations.push(o);
    }
    // A better, more complex program.
    let mut best = measured(&e, SEED_SINE, &c);
    best.selection_score = Some(0.75);
    observations.push(best);
    // Every entry is ranked, so each tie and duplicate above reaches the archive.
    assert!(observations.iter().all(archive::ranked));
    let reference = archive::build(&observations).unwrap();
    let json = reference.to_json();
    let keys: Vec<[u8; 32]> = reference.members().iter().map(|m| m.key).collect();
    let low = if key(tied[0]) < key(tied[1]) {
        tied[0]
    } else {
        tied[1]
    };
    assert_eq!(keys, vec![key(spelled[0]), key(low), key(SEED_SINE)]);
    let n = observations.len();
    let mut orders: Vec<Vec<usize>> = vec![(0..n).rev().collect()];
    for shift in 1..n {
        orders.push((0..n).map(|i| (i + shift) % n).collect());
    }
    orders.push((0..n).step_by(2).chain((1..n).step_by(2)).collect());
    orders.push((1..n).step_by(2).chain((0..n).step_by(2)).rev().collect());
    for order in orders {
        let shuffled: Vec<Kernel2Observation> =
            order.iter().map(|&i| observations[i].clone()).collect();
        assert_eq!(
            archive::build(&shuffled).unwrap().to_json(),
            json,
            "{order:?}"
        );
    }
}

#[test]
fn the_archive_input_is_exactly_the_observations_ranking_admits() {
    let c = data();
    for (label, e) in [
        ("reject-hard", evaluator()),
        (
            "penalize",
            evaluator().with_fault_policy(penalize(true)).unwrap(),
        ),
    ] {
        let mut wide = c.clone();
        for row in &mut wide.train_x {
            row.push(row[0]);
        }
        let mut observations = vec![
            measured(&e, "(pipe X (ridge lambda=0.00000001))", &c),
            measured(&e, SEED, &c),
            measured(&e, SEED_SINE, &c),
            // Refused before any score: parse, profile admission, a held-out Hard fault.
            measured(&e, "(broken", &c),
            measured(&e, "(pipe X (map expr=\"exp(x)\") (ridge))", &c),
            measured(
                &e,
                "(pipe X (derive expr=\"c1\" append=false) (ridge))",
                &wide,
            ),
        ];
        assert!(observations[5].faults.is_hard());
        // Measured, but with a score ranking refuses.
        for score in [f64::NAN, f64::INFINITY] {
            let mut o = measured(&e, "(pipe X (derive expr=\"2*c0\") (ridge))", &c);
            o.selection_score = Some(score);
            observations.push(o);
        }
        let ranked: Vec<Kernel2Observation> = e.rank(&observations).into_iter().cloned().collect();
        let filtered: Vec<&Kernel2Observation> =
            observations.iter().filter(|o| archive::ranked(o)).collect();
        let names = |list: Vec<&Kernel2Observation>| {
            let mut names: Vec<String> =
                list.iter().map(|o| o.observation.genome.clone()).collect();
            names.sort();
            names
        };
        assert_eq!(names(filtered), names(ranked.iter().collect()), "{label}");
        assert_eq!(ranked.len(), 3, "{label}");
        let all = archive::build(&observations).unwrap();
        assert_eq!(all, archive::build(&ranked).unwrap(), "{label}");
        // Whatever the refused observations hold, the archive is the same.
        let mut edited = observations.clone();
        for o in edited.iter_mut().skip(3) {
            o.observation.genome = "(pipe X (standardize) (ridge))".into();
            o.faults.domain_cells += 1;
        }
        assert_eq!(archive::build(&edited).unwrap(), all, "{label}");
    }
}

#[test]
fn program_cost_counts_every_stage_and_expression_node() {
    for (genome, expected) in [
        ("(pipe X (ridge))", 1),
        ("(pipe X (standardize) (ridge))", 2),
        (SEED, 5),
        (
            "(pipe X (derive expr=\"sin(0.5*c0)\" append=false) (ridge lambda_frac=0.000000001))",
            6,
        ),
        (
            "(pipe X (map expr=\"sin(0.5*x)\") (derive expr=\"sin(0.7*c0)\" append=false) (ridge))",
            11,
        ),
        ("(pipe X (derive expr=\"(-0.5)*c0\") (ridge))", 6),
        (
            // Three stages; nine nodes: +, clip, c0, the negation and its 1.0, 1.0, *, 0.3, c1.
            "(pipe X (derive expr=\"clip(c0,(-1.0),1.0)+(0.3*c1)\") (standardize) (ridge))",
            12,
        ),
    ] {
        let parsed = GenomeIr::parse(genome).unwrap();
        assert_eq!(program_cost(&parsed).unwrap(), expected, "{genome}");
        assert_eq!(cost(genome), expected, "{genome}");
    }
}
