#[test]
#[ignore = "subprocess entry point, explicitly executed twice by repeated_emission"]
fn emit_child() {
    let directory = std::env::var_os("JOPE_GP_EMIT_CHILD_DIR").expect("child destination");
    let directory = Path::new(&directory);
    fs::create_dir_all(directory).unwrap();
    for (case, program) in corpus().iter().enumerate() {
        for (lang, language) in LANGUAGES.into_iter().enumerate() {
            let first = emit(case, program, language);
            assert_eq!(
                first,
                emit(case, program, language),
                "case {case}, {language:?}"
            );
            fs::write(directory.join(format!("{case}-{lang}")), first).unwrap();
        }
    }
}

fn evidence_directory(label: &str) -> std::path::PathBuf {
    let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join("gp-emitter-determinism");
    fs::create_dir_all(&root).unwrap();
    let path = tempfile::Builder::new()
        .prefix(label)
        .tempdir_in(root)
        .unwrap()
        .keep();
    println!("GP emitter evidence: {}", path.display());
    path
}

#[test]
fn repeated_emission() {
    assert_eq!(corpus().len(), 84);
    let directory = evidence_directory("processes-");
    for child in ["first", "second"] {
        let output = run(Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "emit_child", "--ignored", "--nocapture"])
            .env("JOPE_GP_EMIT_CHILD_DIR", directory.join(child)));
        assert!(
            output.contains("1 passed"),
            "child test did not execute: {output}"
        );
    }
    for (case, program) in corpus().iter().enumerate() {
        for (lang, language) in LANGUAGES.into_iter().enumerate() {
            let first = emit(case, program, language);
            assert_eq!(first, emit(case, program, language));
            for child in ["first", "second"] {
                assert_eq!(
                    first.as_bytes(),
                    fs::read(directory.join(child).join(format!("{case}-{lang}"))).unwrap(),
                    "case {case}, {language:?}, {child}"
                );
            }
        }
    }
}
