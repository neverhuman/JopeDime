use super::super::tests::{emitted, fixture};
use super::super::{qualification_plan, verify_qualification, FittedRustArtifact, Task};
use super::*;
use std::path::PathBuf;
use std::process::Command;

fn which(name: &str) -> PathBuf {
    let path = std::env::var_os("PATH").expect("PATH");
    std::env::split_paths(&path)
        .map(|dir| dir.join(name))
        .find(|candidate| candidate.is_file())
        .map(|found| fs::canonicalize(found).unwrap())
        .unwrap_or_else(|| panic!("required tool {name} unavailable"))
}
fn pin(path: &Path) -> Artifact {
    Artifact {
        path: path.to_owned(),
        sha256: digest(&fs::read(path).unwrap()),
    }
}
/// A labelled change applied to a copy of an installed package directory.
type Mutation = (&'static str, Box<dyn Fn(&Path)>);
struct Qualified {
    bundle: PublicBundle,
    plan: QualificationPlan,
    result: QualificationResult,
    built: BTreeMap<String, Vec<u8>>,
    executed: BTreeMap<String, String>,
}
/// Execute the private plan exactly as the qualification route does and keep the
/// digests its execution produced for native shipping outputs.
fn qualify(root: &Path, artifact: &FittedRustArtifact, backend: Backend) -> Qualified {
    let bundle = emitted(artifact, backend);
    let rows = vec![
        vec![Some("3.25".into()), Some("NA".into())],
        vec![None, None],
        vec![Some("-1".into()), Some("unseen".into())],
    ];
    let plan = qualification_plan(artifact, &bundle, &rows).unwrap();
    let directory = root.join(format!("execute-{backend:?}-{}", &plan.fitted_sha256[..8]));
    fs::create_dir_all(&directory).unwrap();
    for (name, text) in &plan.files {
        fs::write(directory.join(name), text).unwrap();
    }
    let mut output = Vec::new();
    for argv in &plan.commands {
        let run = Command::new(&argv[0])
            .args(&argv[1..])
            .current_dir(&directory)
            .output()
            .unwrap();
        assert!(
            run.status.success(),
            "{}",
            String::from_utf8_lossy(&run.stderr)
        );
        output = run.stdout;
    }
    let result = verify_qualification(&plan, &plan.identity().unwrap(), &output).unwrap();
    let mut built = BTreeMap::new();
    let mut executed = BTreeMap::new();
    for name in built_files(backend) {
        let bytes = fs::read(directory.join(name)).unwrap();
        executed.insert((*name).to_owned(), digest(&bytes));
        built.insert((*name).to_owned(), bytes);
    }
    Qualified {
        bundle,
        plan,
        result,
        built,
        executed,
    }
}
fn profile(q: &Qualified) -> DependencyProfile {
    let (build, serve) = DependencyProfile::expected_tools(&q.plan).unwrap();
    DependencyProfile {
        runtime: q.bundle.manifest.runtime.clone(),
        host_boundary: q.bundle.manifest.host_boundary.clone(),
        numerical_policy: q.bundle.manifest.numerical_policy.clone(),
        build: build.iter().map(|n| (n.clone(), pin(&which(n)))).collect(),
        build_commands: q.plan.commands[..q.plan.commands.len() - 1].to_vec(),
        serve: serve.iter().map(|n| (n.clone(), pin(&which(n)))).collect(),
    }
}
fn seal_default(q: &Qualified, required: bool) -> (PackageManifest, BTreeMap<String, Vec<u8>>) {
    seal(
        &q.bundle,
        &q.plan,
        &q.result,
        profile(q),
        &q.built,
        &q.executed,
        required,
    )
    .unwrap()
}
fn copy(from: &Path, to: &Path) {
    fs::create_dir(to).unwrap();
    for entry in fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        fs::copy(entry.path(), to.join(entry.file_name())).unwrap();
    }
}
fn scratch() -> tempfile::TempDir {
    let scratch = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/fitted-inference-tests");
    fs::create_dir_all(&scratch).unwrap();
    tempfile::tempdir_in(scratch).unwrap()
}
/// Loaded package must serve the qualified predictions from its installed bytes.
///
/// `-B`: serving must not write `__pycache__` into the installed directory. `load` refuses
/// any entry the package did not ship, so a bytecode cache would make the next load fail
/// (and this used to pass only on hosts that set PYTHONDONTWRITEBYTECODE).
fn serve_python(directory: &Path, module: &str) -> Vec<u8> {
    let run = Command::new("python3")
            .arg("-B")
            .arg("-c")
            .arg(format!("import json,{module};print(json.dumps({module}.predict_raw([['3.25','NA'],[None,None],['-1','unseen']]),ensure_ascii=False,allow_nan=False))"))
            .current_dir(directory)
            .output()
            .unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    run.stdout
}
#[path = "tests/identity.rs"]
mod identity;
#[path = "tests/serving.rs"]
mod serving;
#[path = "tests/tamper.rs"]
mod tamper;
