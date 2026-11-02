use super::*;

/// Explicit opt-in: dependency features are not visible as this crate's cfgs.
/// Run with INT02_INTERNAL_CAS_CUTS=1 and --features jope-service/failpoints.
/// Selecting these cuts without the hooks must fail, never silently skip.
#[cfg(unix)]
fn internal_cas_sigkill_cuts() {
    for round in 1..=3 {
        let committed = round - 1;
        let bytes = format!("synthetic model fixture {round}");
        let hash = jope_assets::sha256_hex(bytes.as_bytes());
        let reference = jope_service::store::BlobRef {
            sha256: std::array::from_fn(|i| {
                u8::from_str_radix(&hash[2 * i..2 * i + 2], 16).unwrap()
            }),
            bytes: bytes.len() as u64,
        };
        for cut in [
            "after_write",
            "after_file_sync",
            "after_hash_verify",
            "after_rename",
            "after_directory_sync",
        ] {
            let point = format!("cas:{round}:{cut}");
            let home = tempfile::tempdir().unwrap();
            prepare_crash_prefix(home.path(), committed);
            let stderr = home.path().join("child.stderr");
            let mut child = ChildGuard(
                Command::new(std::env::current_exe().unwrap())
                    .args(["--exact", "crash_child", "--ignored", "--nocapture"])
                    .env("INT02_CRASH_HOME", home.path())
                    .env("INT02_CRASH_POINT", &point)
                    .env("INT02_CAS_COMMITTED", committed.to_string())
                    .env("JOPE_STORE_FAILPOINT", cut)
                    .stdout(Stdio::null())
                    .stderr(Stdio::from(fs::File::create(&stderr).unwrap()))
                    .spawn()
                    .unwrap(),
            );
            let marker = format!("JOPE_STORE_CRASH:{cut}");
            let deadline = Instant::now() + Duration::from_secs(15);
            loop {
                let output = fs::read_to_string(&stderr).unwrap();
                if output.lines().any(|line| line == marker) {
                    break;
                }
                assert!(child.0.try_wait().unwrap().is_none(), "{point}: child exited before hook; build with --features jope-service/failpoints: {output}");
                assert!(
                    Instant::now() < deadline,
                    "{point}: hook not reached: {output}"
                );
                std::thread::sleep(Duration::from_millis(10));
            }
            child.0.kill().unwrap();
            let status = child.0.wait().unwrap();
            assert!(!status.success(), "{point}");
            #[cfg(unix)]
            {
                use std::os::unix::process::ExitStatusExt;
                assert_eq!(
                    status.signal(),
                    Some(9),
                    "{point}: must be supervisor SIGKILL, not abort"
                );
            }
            let renamed = matches!(cut, "after_rename" | "after_directory_sync");
            let staged = fs::read_dir(home.path().join("tmp"))
                .unwrap()
                .map(|entry| entry.unwrap().path())
                .collect::<Vec<_>>();
            assert_eq!(staged.len(), usize::from(!renamed), "{point}");
            if let Some(staged) = staged.first() {
                assert_eq!(fs::read(staged).unwrap(), bytes.as_bytes(), "{point}");
            }
            {
                let store = Store::open(home.path()).unwrap();
                assert_eq!(store.verify_blob(&reference).is_ok(), renamed, "{point}");
                if renamed {
                    assert_eq!(
                        store.read_blob(&reference).unwrap(),
                        bytes.as_bytes(),
                        "{point}"
                    );
                }
                assert!(
                    fs::read_dir(home.path().join("tmp"))
                        .unwrap()
                        .next()
                        .is_none(),
                    "{point}: staging not recovered"
                );
            }
            verify_crash_continuation(home.path(), committed, &point);
            eprintln!("INT02 CAS round={round} committed={committed} hook={cut} signal=9 renamed={renamed} staging_recovered=true");
        }
    }
}

#[cfg(not(unix))]
fn internal_cas_sigkill_cuts() {
    panic!("internal CAS cuts require native Unix signals");
}

#[test]
fn r1_sigkill_at_public_blob_and_journal_boundaries_recovers_atomic_records_and_sequences() {
    for (round, boundary) in (1..=3).flat_map(|round| {
        [
            "before_blob_publish",
            "after_blob_publish",
            "before_journal_commit",
            "after_journal_commit",
        ]
        .map(|boundary| (round, boundary))
    }) {
        let point = format!("{round}:{boundary}");
        let home = tempfile::tempdir().unwrap();
        prepare_crash_fixture(home.path());
        let mut child = ChildGuard(
            Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "crash_child", "--ignored", "--nocapture"])
                .env("INT02_CRASH_HOME", home.path())
                .env("INT02_CRASH_POINT", &point)
                .stdout(Stdio::null())
                .spawn()
                .unwrap(),
        );
        let deadline = Instant::now() + Duration::from_secs(15);
        while !fs::read_to_string(home.path().join("ready")).is_ok_and(|marker| marker == point) {
            assert!(
                child.0.try_wait().unwrap().is_none(),
                "child exited: {point}"
            );
            assert!(Instant::now() < deadline, "child did not reach {point}");
            std::thread::sleep(Duration::from_millis(10));
        }
        child.0.kill().unwrap();
        let status = child.0.wait().unwrap();
        assert!(!status.success());
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            assert_eq!(status.signal(), Some(9), "{point}");
        }
        let committed = round - u64::from(boundary != "after_journal_commit");
        verify_crash_continuation(home.path(), committed, &point);
    }
    if std::env::var("INT02_INTERNAL_CAS_CUTS").as_deref() == Ok("1") {
        internal_cas_sigkill_cuts();
    } else {
        eprintln!("INT02 internal CAS cuts not selected; require INT02_INTERNAL_CAS_CUTS=1 and --features jope-service/failpoints");
    }
}

#[test]
#[cfg(all(feature = "mcp", any(target_os = "linux", target_os = "macos")))]
fn live_terminal_attach_and_mcp_reconnect_preserve_the_same_run_projection() {
    live_driver::run();
}
