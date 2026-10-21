#![cfg(feature = "neural")]
use jope_parallel::manifest::*;
use std::{
    fs,
    io::Write,
    os::unix::{fs::MetadataExt, net::UnixStream},
    path::Path,
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

struct Worker(Child);
impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn spawn(root: &Path, pool: &Path) -> Worker {
    Worker(
        Command::new(env!("CARGO_BIN_EXE_jope-parallel"))
            .arg("--root")
            .arg(root)
            .args(["gpu-worker", "--pool"])
            .arg(pool)
            .args(["--rank", "0"])
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap(),
    )
}

fn ready(worker: &mut Worker, pool: &Path) -> serde_json::Value {
    let start = Instant::now();
    loop {
        let state = read_json::<serde_json::Value>(&pool.join("rank-0.json")).unwrap_or_default();
        if state["status"] == "idle" && state["pid"] == worker.0.id() {
            return state;
        }
        assert!(
            start.elapsed() < Duration::from_secs(10) && worker.0.try_wait().unwrap().is_none(),
            "resident worker failed to become ready"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn exited(worker: &mut Worker) -> std::process::ExitStatus {
    let start = Instant::now();
    loop {
        if let Some(status) = worker.0.try_wait().unwrap() {
            return status;
        }
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "worker did not reject invalid connection promptly"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn replacement_worker_becomes_ready_while_parent_holds_the_gpu_lease() {
    let d = tempfile::tempdir().unwrap();
    let root = d.path();
    let pool = root.join("runs/pool");
    fs::create_dir_all(&pool).unwrap();
    let lease = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(root.join("runs/encoding-service.lock"))
        .unwrap();
    assert!(lease.try_lock().is_ok());
    let mut child = spawn(root, &pool);
    let state = ready(&mut child, &pool);
    assert_eq!(state["worker"]["process"]["pid"], child.0.id());
    assert!(state["worker"]["process"]["start_ticks"].as_u64().unwrap() > 0);
    assert_eq!(
        state["worker"]["process"]["boot_id"],
        fs::read_to_string("/proc/sys/kernel/random/boot_id")
            .unwrap()
            .trim()
    );
    assert_eq!(
        state["worker"]["socket_inode"],
        pool.join("rank-0.sock").metadata().unwrap().ino()
    );
    assert_eq!(
        state["worker"]["executable_sha256"],
        file_hash(Path::new(env!("CARGO_BIN_EXE_jope-parallel"))).unwrap()
    );
}

#[test]
fn connection_loss_before_descriptor_does_not_accept_work_and_replacement_has_new_identity() {
    let d = tempfile::tempdir().unwrap();
    let pool = d.path().join("runs/pool");
    fs::create_dir_all(&pool).unwrap();
    let mut original = spawn(d.path(), &pool);
    let before = ready(&mut original, &pool);
    drop(UnixStream::connect(pool.join("rank-0.sock")).unwrap());
    assert!(!exited(&mut original).success());
    assert!(!pool.join("tasks").exists());
    let mut replacement = spawn(d.path(), &pool);
    let after = ready(&mut replacement, &pool);
    assert_ne!(before["worker"]["process"], after["worker"]["process"]);
}

#[test]
fn descriptorless_peer_is_rejected_without_acceptance() {
    let d = tempfile::tempdir().unwrap();
    let pool = d.path().join("runs/pool");
    fs::create_dir_all(&pool).unwrap();
    let mut worker = spawn(d.path(), &pool);
    ready(&mut worker, &pool);
    let mut connection = UnixStream::connect(pool.join("rank-0.sock")).unwrap();
    connection.write_all(b"L").unwrap();
    drop(connection);
    assert!(!exited(&mut worker).success());
    assert!(!pool.join("tasks").exists());
}

#[test]
fn replacement_cannot_steal_a_surviving_workers_socket() {
    let d = tempfile::tempdir().unwrap();
    let pool = d.path().join("runs/pool");
    fs::create_dir_all(&pool).unwrap();
    let mut original = spawn(d.path(), &pool);
    let before = ready(&mut original, &pool);
    let mut competing = spawn(d.path(), &pool);
    assert!(!exited(&mut competing).success());
    assert!(original.0.try_wait().unwrap().is_none());
    assert_eq!(
        read_json::<serde_json::Value>(&pool.join("rank-0.json")).unwrap(),
        before
    );
    assert_eq!(
        pool.join("rank-0.sock").metadata().unwrap().ino(),
        before["worker"]["socket_inode"].as_u64().unwrap()
    );
}
