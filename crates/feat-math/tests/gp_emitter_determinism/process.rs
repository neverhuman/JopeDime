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

/// One run's evidence directory. Every run is retained by default. With
/// `JOPE_GP_EMITTER_RETAIN=failures` (set by CI, issue #2158) a run whose test
/// passed is condensed on drop into `<run>.passed.tsv` listing each file's
/// relative path, size and SHA-256, and only then removed; a failing run stays
/// whole. Earlier runs are never touched.
struct Evidence(std::path::PathBuf);

impl std::ops::Deref for Evidence {
    type Target = Path;
    fn deref(&self) -> &Path {
        &self.0
    }
}

impl Drop for Evidence {
    fn drop(&mut self) {
        if std::thread::panicking()
            || std::env::var("JOPE_GP_EMITTER_RETAIN").as_deref() != Ok("failures")
        {
            return;
        }
        let mut files = Vec::new();
        list_files(&self.0, &mut files);
        files.sort();
        let mut manifest = String::new();
        for file in files {
            use sha2::Digest;
            let bytes = fs::read(&file).unwrap();
            let relative = file.strip_prefix(&self.0).unwrap().display();
            let digest = sha2::Sha256::digest(&bytes);
            let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
            manifest += &format!("{relative}\t{}\t{hex}\n", bytes.len());
        }
        let summary = self.0.with_extension("passed.tsv");
        let partial = self.0.with_extension("passed.tsv.partial");
        fs::write(&partial, manifest).unwrap();
        fs::rename(&partial, &summary).unwrap();
        fs::remove_dir_all(&self.0).unwrap();
        println!("GP emitter evidence condensed: {}", summary.display());
    }
}

fn list_files(directory: &Path, files: &mut Vec<std::path::PathBuf>) {
    for entry in fs::read_dir(directory).unwrap() {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_dir() {
            list_files(&entry.path(), files);
        } else {
            files.push(entry.path());
        }
    }
}

fn evidence_directory(label: &str) -> Evidence {
    let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join("gp-emitter-determinism");
    fs::create_dir_all(&root).unwrap();
    let path = tempfile::Builder::new()
        .prefix(label)
        .tempdir_in(root)
        .unwrap()
        .keep();
    println!("GP emitter evidence: {}", path.display());
    Evidence(path)
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
