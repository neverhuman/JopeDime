use super::*;

/// Every allowlist entry as a [`Divergence`], after checking the file's shape: known programs,
/// lanes and input rows, exact bit patterns that differ (or involve a NaN), the one allowed
/// status, a fix item, a reason, and no duplicate.
pub(crate) fn allowlist() -> Vec<Divergence> {
    let list: Allowlist = serde_json::from_str(ALLOWLIST).expect("parity_allowlist.v1.json parses");
    assert_eq!(list.schema, ALLOWLIST_SCHEMA);
    assert!(!list.note.trim().is_empty(), "the allowlist needs its note");
    let mut seen = BTreeSet::new();
    let mut divergences = Vec::new();
    for entry in list.entries {
        let label = format!("{} {} {}", entry.program, entry.language, entry.input);
        let program = PROGRAMS
            .iter()
            .find(|program| program.name == entry.program)
            .unwrap_or_else(|| panic!("{label}: unknown program"));
        assert!(
            Lane::ALL.iter().any(|lane| lane.id() == entry.language),
            "{label}: unknown language"
        );
        assert!(
            input_rows(program.inputs)
                .iter()
                .any(|row| row.id == entry.input),
            "{label}: the program is not evaluated on this input"
        );
        let runtime_bits =
            parse_bits(&entry.runtime_bits).unwrap_or_else(|| panic!("{label}: runtime_bits"));
        let emitted_bits =
            parse_bits(&entry.emitted_bits).unwrap_or_else(|| panic!("{label}: emitted_bits"));
        assert!(
            runtime_bits != emitted_bits
                || f64::from_bits(runtime_bits).is_nan()
                || f64::from_bits(emitted_bits).is_nan(),
            "{label}: identical bit patterns are not a divergence"
        );
        assert_eq!(entry.status, ALLOWLIST_STATUS, "{label}: status");
        assert!(
            entry.fix_item.starts_with('X')
                && entry.fix_item.len() > 1
                && entry.fix_item[1..].bytes().all(|b| b.is_ascii_digit()),
            "{label}: fix_item must name a plan item such as X6"
        );
        assert!(!entry.reason.trim().is_empty(), "{label}: reason");
        assert!(
            seen.insert((
                entry.program.clone(),
                entry.language.clone(),
                entry.input.clone()
            )),
            "{label}: duplicate entry"
        );
        divergences.push(Divergence {
            program: entry.program,
            language: entry.language,
            input: entry.input,
            runtime_bits,
            emitted_bits,
        });
    }
    divergences
}

/// Compare one lane against the runtime on every program and check the result against the
/// allowlist entries of that lane.
pub(crate) fn check_lane(lane: Lane) {
    let Some(version) = toolchain(lane) else {
        return;
    };
    let expected = allowlist();
    let (train_x, train_y) = training_data();
    let mut scratch = tempfile::Builder::new()
        .prefix(&format!("parity-bits-{}-", lane.id()))
        .tempdir_in(scratch_root())
        .unwrap();
    let keep = std::env::var_os("JOPE_PARITY_KEEP_SCRATCH").is_some_and(|value| value == "1");
    if keep {
        // Also keeps the directory when an assertion below panics.
        scratch.disable_cleanup(true);
        eprintln!(
            "parity_bits[{}]: artifacts and inputs kept under {}",
            lane.id(),
            scratch.path().display()
        );
    }
    let root = scratch.path();
    let train_csv = root.join("train.csv");
    write_train_csv(&train_csv, &train_x, &train_y);
    let mut features_csv = BTreeMap::new();
    let mut features_bits = BTreeMap::new();
    for inputs in [Inputs::Finite, Inputs::All, Inputs::NonFiniteColumnZero] {
        let key = input_key(inputs);
        let rows = input_rows(inputs);
        let csv = root.join(format!("features-{key}.csv"));
        let bits = root.join(format!("features-{key}.bits"));
        write_features_csv(&csv, &rows);
        write_features_bits(&bits, &rows);
        features_csv.insert(key, csv);
        features_bits.insert(key, bits);
    }
    let shared = Shared {
        train_csv,
        features_csv,
        features_bits,
        train_x,
        train_y,
    };

    let per_program = parallel_map(PROGRAMS, |program| {
        let genome = round_tripped_genome(program);
        let rows = input_rows(program.inputs);
        let features: Vec<Vec<f64>> = rows.iter().map(|row| row.values.clone()).collect();
        let reference: Vec<u64> = fit(&genome, &shared.train_x, &shared.train_y)
            .unwrap_or_else(|error| panic!("{}: runtime fit: {error:#}", program.name))
            .predict(&features)
            .unwrap()
            .into_iter()
            .map(f64::to_bits)
            .collect();
        let emitted = lane_predictions(
            lane,
            program,
            &genome,
            rows.len(),
            &shared,
            &root.join(program.name),
        );
        let mut divergences = Vec::new();
        for ((row, runtime_bits), emitted_bits) in rows.iter().zip(&reference).zip(&emitted) {
            let nan =
                f64::from_bits(*runtime_bits).is_nan() || f64::from_bits(*emitted_bits).is_nan();
            if runtime_bits != emitted_bits || nan {
                divergences.push(Divergence {
                    program: program.name.to_owned(),
                    language: lane.id().to_owned(),
                    input: row.id.clone(),
                    runtime_bits: *runtime_bits,
                    emitted_bits: *emitted_bits,
                });
            }
        }
        (rows.len(), divergences)
    });
    let compared: usize = per_program.iter().map(|(rows, _)| rows).sum();
    let found: BTreeSet<Divergence> = per_program
        .into_iter()
        .flat_map(|(_, divergences)| divergences)
        .collect();
    let listed: BTreeSet<Divergence> = expected
        .into_iter()
        .filter(|entry| entry.language == lane.id())
        .collect();
    let unexpected: Vec<&Divergence> = found.difference(&listed).collect();
    let stale: Vec<&Divergence> = listed.difference(&found).collect();
    eprintln!(
        "parity_bits[{}] {version}: {} programs, {compared} predictions, {} divergences \
         ({} allowlisted, {} unexpected, {} stale)",
        lane.id(),
        PROGRAMS.len(),
        found.len(),
        found.intersection(&listed).count(),
        unexpected.len(),
        stale.len()
    );
    let mut report = String::new();
    for divergence in &unexpected {
        report.push_str(&format!("  unexpected: {}\n", divergence.describe()));
    }
    for divergence in &stale {
        report.push_str(&format!(
            "  stale (no longer reproduces): {}\n",
            divergence.describe()
        ));
    }
    assert!(
        report.is_empty(),
        "parity_bits[{}]: the runtime and the emitted program disagree beyond the allowlist \
         (fixtures/parity_allowlist.v1.json):\n{report}",
        lane.id()
    );
    if keep {
        // `close` removes the directory whatever `disable_cleanup` says; `keep` leaves it.
        let _ = scratch.keep();
    } else {
        scratch.close().unwrap();
    }
}

/// Reparse canonical text and the typed envelope before any lane fits or emits a program.
pub(crate) fn round_tripped_genome(program: &Program) -> GenomeIr {
    let original = GenomeIr::parse(program.source)
        .unwrap_or_else(|error| panic!("{}: {error:#}", program.name));
    for expression in expressions(program) {
        let source = expression.to_source().unwrap();
        let reparsed = MathExpr::parse(&source).unwrap();
        assert_eq!(
            reparsed, expression,
            "{}: scalar tree changed",
            program.name
        );
        assert_eq!(reparsed.to_source().unwrap(), source);
    }
    let canonical = original.canonical();
    let reparsed = GenomeIr::parse(&canonical)
        .unwrap_or_else(|error| panic!("{}: canonical parse: {error:#}", program.name));
    assert_eq!(
        reparsed.canonical(),
        canonical,
        "{}: fixed point",
        program.name
    );
    let bytes = KernelProgram::import_legacy(&original).to_bytes().unwrap();
    assert_eq!(
        KernelProgram::import_legacy(&reparsed).to_bytes().unwrap(),
        bytes,
        "{}: canonical text changed the executable tree",
        program.name
    );
    let envelope = KernelProgram::import_legacy(&reparsed)
        .to_lossless_text()
        .unwrap();
    assert_eq!(
        KernelProgram::parse_lossless_text(&envelope)
            .unwrap()
            .to_bytes()
            .unwrap(),
        bytes,
        "{}: typed envelope changed the executable tree",
        program.name
    );
    reparsed
}

#[test]
fn parity_bits_programs_round_trip_before_emission() {
    for program in PROGRAMS {
        round_tripped_genome(program);
    }
}

#[test]
fn parity_bits_runtime_matches_emitted_rust() {
    check_lane(Lane::Rust);
}

#[test]
fn parity_bits_runtime_matches_emitted_python() {
    check_lane(Lane::Python);
}

#[test]
fn parity_bits_runtime_matches_fitted_cpp17() {
    check_lane(Lane::Cpp17Fitted);
}

// ------------------------------------------------------------------------------------------------
// Coverage, corpus and allowlist checks
// ------------------------------------------------------------------------------------------------
