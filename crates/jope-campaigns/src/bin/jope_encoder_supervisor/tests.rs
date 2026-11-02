#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_names_cannot_escape_the_job_directory() {
        assert!(validate_simple_name("output.f32le").is_ok());
        assert!(validate_simple_name("../output.f32le").is_err());
        assert!(validate_simple_name("nested/output.f32le").is_err());
        assert!(validate_simple_name("/tmp/output.f32le").is_err());
    }

    #[test]
    fn landlock_abi_is_a_capability_gate() {
        match LandlockRuleset::new() {
            Ok(ruleset) => assert!(ruleset.abi >= 3),
            Err(error) => assert!(error.to_string().contains("Landlock")),
        }
    }

    #[test]
    fn landlock_denies_test_v1_checkout_and_undeclared_bytes() {
        let temporary = tempfile::tempdir().unwrap();
        let allowed = temporary.path().join("job-input");
        fs::create_dir(&allowed).unwrap();
        fs::write(allowed.join("train.csv"), b"train").unwrap();
        let denied = [
            temporary.path().join("test-bytes/test.csv"),
            temporary.path().join("expanded-v1/cache.bin"),
            temporary.path().join("jope-dime/source.py"),
            temporary.path().join("undeclared/metadata.json"),
        ];
        for path in &denied {
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, b"denied").unwrap();
        }
        let ruleset = LandlockRuleset::new().expect("Landlock is a required capability");
        let read_execute = ACCESS_READ_FILE | ACCESS_READ_DIR | ACCESS_EXECUTE;
        for path in ["/usr", "/lib", "/lib64", "/etc/ld.so.cache"] {
            let path = Path::new(path);
            if path.exists() {
                ruleset
                    .allow_path(&fs::canonicalize(path).unwrap(), read_execute)
                    .unwrap();
            }
        }
        ruleset
            .allow_path(&allowed, ACCESS_READ_FILE | ACCESS_READ_DIR)
            .unwrap();
        let ruleset_fd = ruleset.fd.as_raw_fd();
        let mut command = Command::new("/bin/sh");
        command
            .arg("-c")
            .arg(
                "/usr/bin/cat \"$1\" && ! /usr/bin/cat \"$2\" && ! /usr/bin/cat \"$3\" && ! /usr/bin/cat \"$4\" && ! /usr/bin/cat \"$5\"",
            )
            .arg("landlock-test")
            .arg(allowed.join("train.csv"))
            .args(&denied);
        // SAFETY: the closure makes only async-signal-safe prctl/landlock syscalls and reads errno,
        // never allocating or locking; `ruleset`, which owns ruleset_fd, outlives status().
        unsafe {
            command.pre_exec(move || {
                if libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                if libc::syscall(libc::SYS_landlock_restrict_self, ruleset_fd, 0u32) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        assert!(command.status().unwrap().success());
    }
}
