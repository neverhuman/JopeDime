use super::*;
#[test]
#[ignore = "subprocess fixture for abrupt controller death"]
fn parent_death_child() {
    let path = PathBuf::from(std::env::var_os("JOPE_PARENT_DEATH_TEST").unwrap());
    let mut command = Command::new("sleep");
    command.arg("60");
    bind_parent_death(&mut command);
    let mut child = command.spawn().unwrap();
    fs::write(path, child.id().to_string()).unwrap();
    child.wait().unwrap();
}
#[test]
fn sigkill_of_controller_reaps_its_rank_clients() {
    let d = tempfile::tempdir().unwrap();
    let path = d.path().join("child.pid");
    let mut parent = Command::new(std::env::current_exe().unwrap())
        .args([
            "--ignored",
            "--exact",
            "controller::lifetime_tests::parent_death_child",
        ])
        .env("JOPE_PARENT_DEATH_TEST", &path)
        .stdout(Stdio::null())
        .spawn()
        .unwrap();
    let start = Instant::now();
    while !path.is_file() {
        assert!(start.elapsed() < Duration::from_secs(10));
        std::thread::sleep(Duration::from_millis(20));
    }
    let pid = fs::read_to_string(&path).unwrap();
    parent.kill().unwrap();
    parent.wait().unwrap();
    let start = Instant::now();
    loop {
        let stat = fs::read_to_string(format!("/proc/{pid}/stat"));
        if stat.is_err()
            || stat
                .unwrap()
                .rsplit(')')
                .next()
                .unwrap()
                .trim_start()
                .starts_with('Z')
        {
            break;
        }
        if start.elapsed() > Duration::from_secs(5) {
            // SAFETY: kill takes only integers and has no memory-safety preconditions; it
            // cleans up the fixture's surviving rank before the test fails.
            unsafe {
                libc::kill(pid.parse().unwrap(), libc::SIGKILL);
            }
            panic!("rank survived its controller");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}
