use super::prelude::*;
use super::types::{extension, valid_hash, Artifact, Status};
use super::{hash, LANGUAGES};

/// Resolve one supported language identifier to its pinned lexer kind.
pub fn language_kind(language: &str) -> Result<Lang> {
    match feat_math::source_stream::language_kind(language) {
        Some(kind) => Ok(kind),
        None => bail!("unknown language"),
    }
}
/// Emit all configured language artifacts and return their aggregate emission time.
pub fn emit(occurrence: &Occurrence) -> Result<(Vec<Artifact>, u64)> {
    ensure!(
        occurrence.schema == SCHEMA
            && occurrence.program.len() <= feat_math::source_stream::MAX_PROGRAM_BYTES,
        "invalid/oversized occurrence"
    );
    let started = Instant::now();
    let mut artifacts = Vec::new();
    for (language, source) in LANGUAGES
        .into_iter()
        .zip(emit_languages(&occurrence.program))
    {
        artifacts.push(match source {
            Ok(source) => {
                let count = count_tokens(&source, language_kind(language)?);
                Artifact {
                    language: language.into(),
                    sha256: Some(hash(source.as_bytes())),
                    source: Some(source),
                    code_tokens: count.code_tokens as u64,
                    doc_tokens: count.doc_tokens as u64,
                    bytes: count.bytes as u64,
                    status: Status::Pending,
                    error: None,
                }
            }
            Err(error) => Artifact {
                language: language.into(),
                source: None,
                sha256: None,
                code_tokens: 0,
                doc_tokens: 0,
                bytes: 0,
                status: Status::Failed,
                error: Some(error),
            },
        });
    }
    Ok((
        artifacts,
        u64::try_from(started.elapsed().as_nanos())
            .unwrap_or(u64::MAX)
            .max(1),
    ))
}

#[derive(Clone)]
/// Result of checking one emitted source artifact.
pub struct Check {
    /// Pass/fail state.
    pub status: Status,
    /// Compiler or interpreter executable used.
    pub tool: String,
    /// Bounded diagnostic text and process result.
    pub diagnostic: String,
    /// Check duration in nanoseconds.
    pub elapsed_ns: u64,
}
/// GNU timeout bounds the compiler process group. No shell or generated-code execution.
pub fn check(root: &Path, artifact: &Artifact, timeout_seconds: u64) -> Result<Check> {
    ensure!(
        (1..=300).contains(&timeout_seconds),
        "compiler timeout outside 1..300 seconds"
    );
    let digest = artifact.sha256.as_ref().context("no source digest")?;
    ensure!(valid_hash(digest), "invalid source digest");
    let source = root
        .join("sources")
        .join(format!("{digest}.{}", extension(&artifact.language)?));
    ensure!(
        hash(&fs::read(&source)?) == *digest,
        "source changed before compilation"
    );
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let scratch = root.join("checks").join(format!(
        "{digest}-{}-{stamp}-{}",
        artifact.language,
        std::process::id()
    ));
    fs::create_dir(&scratch)?;
    let diagnostics = scratch.join("diagnostic.txt");
    let (tool, mut args): (&str, Vec<String>) = match artifact.language.as_str() {
        "rust" => ("rustc", vec!["--edition=2021".into(), "--crate-name=gp_source".into(), "--crate-type=lib".into(), "--emit=metadata".into(), "-o".into(), scratch.join("checked.rmeta").to_string_lossy().into_owned()]),
        "python" => ("python3", vec!["-I".into(), "-c".into(), "import pathlib,sys; compile(pathlib.Path(sys.argv[1]).read_bytes(), sys.argv[1], 'exec')".into()]),
        "cpp17" => ("g++", vec!["-std=c++17".into(), "-fsyntax-only".into()]),
        "javascript" => ("node", vec!["--check".into()]),
        _ => bail!("unknown language"),
    };
    let variable = match artifact.language.as_str() {
        "rust" => "JOPE_GP_RUSTC",
        "python" => "JOPE_GP_PYTHON",
        "cpp17" => "JOPE_GP_CXX",
        "javascript" => "JOPE_GP_NODE",
        _ => unreachable!(),
    };
    let tool = std::env::var(variable).unwrap_or_else(|_| tool.into());
    args.push(source.to_string_lossy().into_owned());
    let version_path = scratch.join("tool-version.txt");
    let _ = Command::new("prlimit")
        .args([
            "--fsize=65536",
            "--",
            "timeout",
            "--signal=KILL",
            "2s",
            &tool,
            "--version",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::from(File::create(&version_path)?))
        .stderr(Stdio::null())
        .status();
    let mut version = String::new();
    File::open(version_path)?
        .take(65536)
        .read_to_string(&mut version)?;
    let started = Instant::now();
    let status = Command::new("prlimit")
        .args([
            "--fsize=4194304",
            "--as=8589934592",
            &format!("--cpu={timeout_seconds}"),
            "--",
            "timeout",
            "--signal=KILL",
            &format!("{timeout_seconds}s"),
            &tool,
        ])
        .args(&args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::from(File::create(&diagnostics)?))
        .status();
    let (passed, error) = match status {
        Ok(status) => (status.success(), format!("exit={status}")),
        Err(error) => (false, format!("compiler launch failed: {error}")),
    };
    let mut bytes = Vec::new();
    File::open(&diagnostics)?
        .take(65536)
        .read_to_end(&mut bytes)?;
    let mut diagnostic = String::from_utf8_lossy(&bytes).into_owned();
    diagnostic.push_str(&format!("\n{error}"));
    Ok(Check {
        status: if passed {
            Status::Passed
        } else {
            Status::Failed
        },
        tool: format!(
            "{tool}; version={version:?}; diagnostic_file={diagnostics:?}; validation=syntax-and-types-v1; output_file_limit=4194304; address_space_limit=8589934592; cpu_seconds_limit={timeout_seconds}; timeout={timeout_seconds}s; argv={args:?}"
        ),
        diagnostic,
        elapsed_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
    })
}
