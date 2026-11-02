fn rows() -> Vec<Vec<f64>> {
    let mut rows = vec![vec![0.25, -0.75, 1.5], vec![-0.0, 0.0, -0.0]];
    for value in [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::MAX,
        -1e200,
        1e-300,
        f64::from_bits(1),
        -2.0,
        -1.0,
        -0.0,
        0.0,
        1e-12,
        f64::from_bits(1e-12_f64.to_bits() - 1),
        f64::from_bits(1e-12_f64.to_bits() + 1),
    ] {
        rows.push(vec![value, value, value]);
        rows.push(vec![value, 0.5, -0.5]);
    }
    rows
}

fn matrix(rows: &[Vec<f64>], language: ScalarLanguage) -> String {
    let rows = rows
        .iter()
        .map(|row| {
            let cells = row
                .iter()
                .map(|v| match language {
                    ScalarLanguage::Rust => format!("f64::from_bits({}u64)", v.to_bits()),
                    ScalarLanguage::Python => format!("m.scalar_bits_v2({})", v.to_bits()),
                    ScalarLanguage::Cpp17 => format!("f64_from_bits({}ULL)", v.to_bits()),
                })
                .collect::<Vec<_>>()
                .join(",");
            match language {
                ScalarLanguage::Rust => format!("vec![{cells}]"),
                ScalarLanguage::Python => format!("[{cells}]"),
                ScalarLanguage::Cpp17 => format!("std::vector<double>{{{cells}}}"),
            }
        })
        .collect::<Vec<_>>()
        .join(",");
    match language {
        ScalarLanguage::Rust => format!("vec![{rows}]"),
        ScalarLanguage::Python => format!("[{rows}]"),
        ScalarLanguage::Cpp17 => format!("std::vector<std::vector<double>>{{{rows}}}"),
    }
}

fn agrees(actual: f64, expected: f64) -> bool {
    if actual.is_nan() || expected.is_nan() {
        return actual.is_nan() && expected.is_nan();
    }
    if !actual.is_finite() || !expected.is_finite() || actual == 0.0 || expected == 0.0 {
        return actual.to_bits() == expected.to_bits();
    }
    (actual - expected).abs() <= ABS_TOLERANCE + REL_TOLERANCE * actual.abs().max(expected.abs())
}

/// Compiler receipts are separate from deterministic emitted program bytes.
/// Keep failed compiler output too; this never deletes prior retained evidence.
fn run_rust_compiler(command: &mut Command, directory: &Path) {
    use sha2::{Digest, Sha256};
    let output = command
        .output()
        .unwrap_or_else(|error| panic!("{command:?}: {error}"));
    fs::write(directory.join("rust-compile.stdout"), &output.stdout).unwrap();
    fs::write(directory.join("rust-compile.stderr"), &output.stderr).unwrap();
    let source = fs::read(directory.join("probe.rs")).unwrap();
    let binary = if output.status.success() {
        let bytes = fs::read(directory.join("rust-probe")).unwrap();
        Some(serde_json::json!({
            "path": "rust-probe",
            "bytes": bytes.len(),
            "sha256": format!("{:x}", Sha256::digest(&bytes)),
        }))
    } else {
        None
    };
    let receipt = serde_json::json!({
        "schema": "jope.gp-emitter-rust-compile/v1",
        "program": command.get_program().to_string_lossy(),
        "arguments": command.get_args().map(|arg| arg.to_string_lossy()).collect::<Vec<_>>(),
        "source_path": "probe.rs",
        "source_sha256": format!("{:x}", Sha256::digest(&source)),
        "success": output.status.success(),
        "exit_code": output.status.code(),
        "exit_status": output.status.to_string(),
        "binary": binary,
    });
    fs::write(
        directory.join("rust-compile.json"),
        serde_json::to_vec_pretty(&receipt).unwrap(),
    )
    .unwrap();
    assert!(
        output.status.success(),
        "{command:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn compile_case(case: usize, program: &ScalarExpressionProgram, directory: &Path) {
    let rows = rows();
    let train: Vec<Vec<f64>> = (0..16)
        .map(|i| vec![i as f64 / 8.0 - 1.0, 0.25, -0.5])
        .collect();
    let labels: Vec<f64> = (0..16).map(|i| i as f64 / 16.0).collect();
    let native = program
        .fit_training(&train, &labels, carrier(case), LAMBDA)
        .unwrap();
    let (features, faults) = native.transform(&rows).unwrap();
    let (predictions, prediction_faults) = native.predict(&rows).unwrap();
    assert_eq!(faults, prediction_faults);
    let expected: Vec<_> = features
        .iter()
        .flatten()
        .chain(&predictions)
        .copied()
        .collect();
    fs::create_dir_all(directory).unwrap();
    fs::write(directory.join("program.bin"), program.to_bytes().unwrap()).unwrap();
    fs::write(directory.join("native.json"), serde_json::to_vec(&serde_json::json!({"values_bits":expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"faults":faults})).unwrap()).unwrap();
    for language in LANGUAGES {
        let source = emit(case, program, language);
        let test_rows = matrix(&rows, language);
        let training = matrix(&train, language);
        let y = matrix(std::slice::from_ref(&labels), language);
        let observations = match language {
            ScalarLanguage::Rust => {
                fs::write(directory.join("model.rs"), &source).unwrap();
                let driver = format!("\n#[test] fn gp_probe(){{let rows={test_rows};let train={training};let y={y};let model=Model::fit(&train,&y[0]);let (f,q)=scalar_stage1_values(&rows);for v in f.into_iter().flatten().chain(model.predict(&rows)){{println!(\"V {{:016x}}\",v.to_bits());}}for v in q{{println!(\"Q {{}}\",v);}}}}\n");
                fs::write(directory.join("probe.rs"), source + &driver).unwrap();
                // Retain this run's exact source, binary and command evidence without
                // copying the standard library's full DWARF into every probe binary.
                // strip=debuginfo preserves symbol names and all numerical/codegen flags.
                run_rust_compiler(
                    Command::new("rustc")
                        .args([
                            "--test",
                            "--edition=2021",
                            "-C",
                            "opt-level=1",
                            "-C",
                            "strip=debuginfo",
                        ])
                        .arg(directory.join("probe.rs"))
                        .arg("-o")
                        .arg(directory.join("rust-probe")),
                    directory,
                );
                vec![(
                    "Rust",
                    run(Command::new(directory.join("rust-probe")).args([
                        "--exact",
                        "gp_probe",
                        "--nocapture",
                        // Pretty single-thread status otherwise prefixes the first V record.
                        "--format=terse",
                    ])),
                )]
            }
            ScalarLanguage::Python => {
                fs::write(directory.join("model.py"), source).unwrap();
                run(Command::new("python3")
                    .args(["-m", "py_compile"])
                    .arg(directory.join("model.py")));
                let driver = format!("import model as m\nimport struct\nrows={test_rows}\nmodel=m.Model().fit({training}, {y}[0])\nf,q=m.scalar_stage1_values(rows)\nfor v in [v for row in f for v in row]+model.predict(rows): print('V %016x'%struct.unpack('<Q',struct.pack('<d',v))[0])\nfor v in q: print('Q %d'%v)\n");
                fs::write(directory.join("probe.py"), driver).unwrap();
                vec![(
                    "Python",
                    run(Command::new("python3").arg(directory.join("probe.py"))),
                )]
            }
            ScalarLanguage::Cpp17 => {
                fs::write(directory.join("model.cpp"), source).unwrap();
                let driver = format!("#include \"model.cpp\"\n#include <cstdio>\nint main(){{const auto rows={test_rows};const auto y={y};const auto model=Model::fit({training},y[0]);const auto f=scalar_stage1_values(rows);auto print=[](double v){{std::uint64_t bits;std::memcpy(&bits,&v,8);std::printf(\"V %016llx\\n\",static_cast<unsigned long long>(bits));}};for(const auto& row:f.first)for(double v:row)print(v);for(double v:model.predict(rows))print(v);for(auto v:f.second)std::printf(\"Q %u\\n\",v);}}\n");
                fs::write(directory.join("probe.cpp"), driver).unwrap();
                ["g++", "clang++"]
                    .into_iter()
                    .map(|compiler| {
                        let executable = directory.join(format!("{compiler}-probe"));
                        run(Command::new(compiler)
                            .args([
                                "-std=c++17",
                                "-Wall",
                                "-Werror",
                                "-O1",
                                "-fno-fast-math",
                                "-ffp-contract=off",
                            ])
                            .arg(directory.join("probe.cpp"))
                            .arg("-o")
                            .arg(&executable));
                        (compiler, run(&mut Command::new(executable)))
                    })
                    .collect()
            }
        };
        for (implementation, observed) in observations {
            fs::write(directory.join(format!("{implementation}.out")), &observed).unwrap();
            let values: Vec<_> = observed
                .lines()
                .filter_map(|line| line.strip_prefix("V "))
                .map(|bits| f64::from_bits(u64::from_str_radix(bits, 16).unwrap()))
                .collect();
            let actual_faults: Vec<u32> = observed
                .lines()
                .filter_map(|line| line.strip_prefix("Q "))
                .map(|v| v.parse().unwrap())
                .collect();
            assert_eq!(actual_faults, faults, "case {case} {implementation} faults");
            assert_eq!(
                values.len(),
                expected.len(),
                "case {case} {implementation} value count"
            );
            for (row, (actual, expected)) in values.into_iter().zip(&expected).enumerate() {
                assert!(
                    agrees(actual, *expected),
                    "case {case} {implementation} value {row}: {actual:?} != {expected:?}"
                );
            }
        }
    }
}

fn differential(shard: usize) {
    let directory = evidence_directory(&format!("shard-{shard}-"));
    // Record the actual selected toolchain once per shard, outside source-byte comparisons.
    fs::write(
        directory.join("rustc-version.txt"),
        run(Command::new("rustc").arg("-vV")),
    )
    .unwrap();
    for (case, program) in corpus()
        .iter()
        .enumerate()
        .filter(|(case, _)| case % 8 == shard)
    {
        compile_case(case, program, &directory.join(case.to_string()));
    }
}

macro_rules! shards {
    ($($name:ident: $shard:literal),*) => {$(#[test] fn $name() { differential($shard); })*};
}
shards!(compile_0:0, compile_1:1, compile_2:2, compile_3:3,
    compile_4:4, compile_5:5, compile_6:6, compile_7:7);
