use super::*;

/// `mathexpr_boundary_parity.rs` `rows`: training rows, targets and prediction rows.
fn recipe_rows() -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>) {
    let train: Vec<Vec<f64>> = (0..32)
        .map(|i| {
            vec![
                i as f64 / 8.0 - 2.0,
                (i % 7) as f64 / 3.0 - 1.0,
                (i % 11) as f64 / 5.0 - 1.0,
            ]
        })
        .collect();
    let targets = train
        .iter()
        .map(|r| 0.7 * r[0] - 0.2 * r[1] + 0.1 * r[2])
        .collect();
    let mut test = vec![
        vec![0.125, -0.75, 1.5],
        vec![-2.0, -1.0, -0.5],
        vec![0.0, -0.0, 0.0],
        vec![f64::from_bits(1), 1e-12, 1.0],
    ];
    for value in [1e-12_f64.to_bits() - 1, 1e-12_f64.to_bits() + 1] {
        test.push(vec![f64::from_bits(value), 0.5, 1.0]);
    }
    (train, targets, test)
}

/// `mathexpr_boundary_parity.rs` `execute(program, "g++", "-std=c++17", "-O2", false)`.
#[test]
fn the_refined_winner_matches_the_cpp17_recipe_bit_for_bit() {
    let original = GenomeIr::parse(PINNED_LM_PROGRAM).unwrap();
    let canonical = original.canonical();
    let genome = GenomeIr::parse(&canonical).unwrap();
    assert_eq!(canonical, genome.canonical());
    let bytes = KernelProgram::import_legacy(&genome).to_bytes().unwrap();
    assert_eq!(
        bytes,
        KernelProgram::from_bytes(&bytes)
            .unwrap()
            .to_bytes()
            .unwrap()
    );
    let recipe = emit_recipe_cpp17(&genome, &EmitOptions::default()).unwrap();
    let (mut train, targets, mut test) = recipe_rows();
    assert!(matches!(genome, GenomeIr::Pipeline(_)));
    // Pipeline entry points sanitize features, so a pipeline also receives non-finite cells.
    for (index, value) in [
        f64::from_bits(0x7ff8_1234_5678_9abc),
        f64::INFINITY,
        f64::NEG_INFINITY,
    ]
    .into_iter()
    .enumerate()
    {
        train[index][index] = value;
        let mut row = vec![0.75, -1.25, 1.5];
        row[index] = value;
        test.push(row);
    }
    let expected: Vec<u64> = fit_predict(&genome, &train, &targets, &test)
        .unwrap()
        .into_iter()
        .map(f64::to_bits)
        .collect();
    let scratch = tempfile::Builder::new()
        .prefix("emitted-winner-recipe-")
        .tempdir_in(scratch_root())
        .unwrap();
    let source = scratch.path().join("recipe.cpp");
    let binary = scratch.path().join("recipe");
    fs::write(&source, recipe.source + recipe_driver()).unwrap();
    run_checked(
        "emitted winner recipe: g++",
        Command::new("g++")
            .args(["-std=c++17", "-O2", "-ffp-contract=off", "-fno-fast-math"])
            .arg(&source)
            .arg("-o")
            .arg(&binary)
            .env("TMPDIR", scratch.path()),
    );
    let mut input = format!("{} {} 3\n", train.len(), test.len());
    for value in train
        .into_iter()
        .flatten()
        .chain(targets)
        .chain(test.into_iter().flatten())
    {
        input.push_str(&format!("{:016x}\n", value.to_bits()));
    }
    let mut child = Command::new(&binary)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "emitted winner recipe: status {:?}: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    let actual: Vec<u64> = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|line| u64::from_str_radix(line, 16).unwrap())
        .collect();
    assert_eq!(actual, expected, "emitted winner: g++ -std=c++17");
    scratch.close().unwrap();
}
