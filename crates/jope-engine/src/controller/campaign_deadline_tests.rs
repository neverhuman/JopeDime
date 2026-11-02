#[cfg(test)]
mod campaign_deadline_tests {
    use super::*;

    #[test]
    fn auxiliary_stop_reaps_only_owned_rank() {
        let mut command = Command::new("/bin/sleep");
        command.arg("30");
        // SAFETY: the hook runs between fork and exec and only calls async-signal-safe setpgid;
        // io::Error::last_os_error reads errno without allocating.
        unsafe {
            command.pre_exec(|| {
                if libc::setpgid(0, 0) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let child = command.spawn().unwrap();
        let pid = child.id() as i32;
        let signal = AtomicBool::new(false);
        let result = (|| -> Result<()> {
            let _owned = Ranks(vec![child]);
            check_cancel(Some(&signal))?;
            signal.store(true, Ordering::Relaxed);
            check_cancel(Some(&signal))
        })();
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("auxiliary training stopped"));
        // SAFETY: kill with signal 0 sends nothing and has no memory-safety preconditions; it
        // only probes whether the reaped rank's pid still exists.
        assert_eq!(unsafe { libc::kill(pid, 0) }, -1);
        assert_eq!(
            std::io::Error::last_os_error().raw_os_error(),
            Some(libc::ESRCH)
        );
    }

    #[test]
    fn expired_active_wait_reaps_owned_child_and_preserves_saved_work() {
        let saved = tempfile::tempdir().unwrap();
        let checkpoint = saved.path().join("checkpoint.json");
        fs::write(&checkpoint, b"durable checkpoint").unwrap();
        let mut command = Command::new("/bin/sleep");
        command.arg("30");
        // SAFETY: the hook runs between fork and exec and only calls async-signal-safe setpgid;
        // io::Error::last_os_error reads errno without allocating.
        unsafe {
            command.pre_exec(|| {
                if libc::setpgid(0, 0) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let child = command.spawn().unwrap();
        let pid = child.id() as i32;
        let result = (|| -> Result<()> {
            let _owned = Ranks(vec![child]);
            // Expire after the rank is alive, as happens during a polling loop.
            let _deadline = crate::campaign::DeadlineGuard::new(crate::telemetry::now() + 1);
            loop {
                crate::campaign::check_deadline()?;
                std::thread::sleep(Duration::from_millis(10));
            }
        })();
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("fixed campaign deadline"));
        // SAFETY: kill with signal 0 sends nothing and has no memory-safety preconditions; it
        // only probes whether the reaped rank's pid still exists.
        assert_eq!(unsafe { libc::kill(pid, 0) }, -1);
        assert_eq!(
            std::io::Error::last_os_error().raw_os_error(),
            Some(libc::ESRCH)
        );
        assert_eq!(fs::read(checkpoint).unwrap(), b"durable checkpoint");
        crate::campaign::check_deadline().unwrap();
    }
}
