use feat_math::source_stream::{Journal, Occurrence, SCHEMA};
use jope_observatory::gp_source::{self, Check, Status, Store};
use std::{
    fs,
    io::{BufReader, Cursor, Seek},
    path::Path,
    process::{Child, Command, Stdio},
    time::Duration,
};
use tempfile::TempDir;

fn test_dir() -> TempDir {
    let root = std::env::var_os("JOPE_GP_SOURCE_TEST_ROOT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/gp-source-tests")
        });
    fs::create_dir_all(&root).unwrap();
    tempfile::tempdir_in(root).unwrap()
}

fn occurrence(id: &str, program: &str) -> Occurrence {
    Occurrence {
        schema: SCHEMA.into(),
        id: id.into(),
        engine: "fixture".into(),
        coordinate: "generation/0/slot/0".into(),
        program: program.into(),
    }
}
fn publish(store: &mut Store, id: &str, program: &str) {
    let input = occurrence(id, program);
    let (artifacts, ns) = gp_source::emit(&input).unwrap();
    store.publish(input, artifacts, ns).unwrap();
}
fn drain(store: &mut Store) {
    while let Some((id, artifact, root)) = store.next_check() {
        let check = gp_source::check(&root, &artifact, 15).unwrap();
        assert_eq!(
            check.status,
            Status::Passed,
            "{}: {}",
            artifact.language,
            check.diagnostic
        );
        store.checked(id, &artifact, check).unwrap();
    }
}
fn counts(store: &Store) -> serde_json::Value {
    store.snapshot()["total"]["counts"].clone()
}

struct Server(Child);
impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
fn start(journals: &Path, state: &Path) -> (Server, std::net::SocketAddr) {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    drop(listener);
    let child = Command::new(env!("CARGO_BIN_EXE_gp-source-stream"))
        .args([
            "--journal-dir",
            journals.to_str().unwrap(),
            "--state-dir",
            state.to_str().unwrap(),
            "--listen",
            &address.to_string(),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap();
    let mut server = Server(child);
    for _ in 0..200 {
        assert!(
            server.0.try_wait().unwrap().is_none(),
            "source service exited"
        );
        if std::net::TcpStream::connect(address).is_ok() {
            return (server, address);
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    panic!("source service did not start");
}

fn once_with_rustc(journals: &Path, state: &Path, rustc: &Path) -> serde_json::Value {
    let output = Command::new(env!("CARGO_BIN_EXE_gp-source-stream"))
        .args([
            "--journal-dir",
            journals.to_str().unwrap(),
            "--state-dir",
            state.to_str().unwrap(),
            "--compile-timeout-seconds",
            "15",
            "--once",
        ])
        .env("JOPE_GP_RUSTC", rustc)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

#[path = "gp_source/ingress.rs"]
mod ingress;
#[path = "gp_source/runtime.rs"]
mod runtime;
#[path = "gp_source/store.rs"]
mod store;
