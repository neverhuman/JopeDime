//! Per-process CPU time and resident memory sampling.

use std::io;

/// Cumulative CPU time in seconds and resident memory in bytes for `pid`.
pub fn sample(pid: u32) -> io::Result<(f64, u64)> {
    platform::sample(pid)
}

#[cfg(target_os = "linux")]
#[allow(unsafe_code)]
mod platform {
    use std::{fs, io};

    pub(super) fn sample(pid: u32) -> io::Result<(f64, u64)> {
        let stat = fs::read_to_string(format!("/proc/{pid}/stat"))?;
        let fields = stat
            .rsplit_once(')')
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "malformed proc stat"))?
            .1
            .split_whitespace()
            .collect::<Vec<_>>();
        let user_ticks = parse_field(&fields, 11)?;
        let system_ticks = parse_field(&fields, 12)?;
        // SAFETY: sysconf has no pointer arguments or memory-safety preconditions.
        let ticks_per_second = unsafe { libc::sysconf(libc::_SC_CLK_TCK) };
        if ticks_per_second <= 0 {
            return Err(io::Error::last_os_error());
        }

        let statm = fs::read_to_string(format!("/proc/{pid}/statm"))?;
        let resident_pages = statm
            .split_whitespace()
            .nth(1)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "malformed proc statm"))?
            .parse::<u64>()
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        // SAFETY: sysconf has no pointer arguments or memory-safety preconditions.
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if page_size <= 0 {
            return Err(io::Error::last_os_error());
        }
        let rss = resident_pages
            .checked_mul(page_size as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "RSS overflow"))?;

        let total_ticks = user_ticks
            .checked_add(system_ticks)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "CPU time overflow"))?;
        Ok((total_ticks as f64 / ticks_per_second as f64, rss))
    }

    fn parse_field(fields: &[&str], index: usize) -> io::Result<u64> {
        fields
            .get(index)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing proc stat field"))?
            .parse()
            .map_err(|error: std::num::ParseIntError| {
                io::Error::new(io::ErrorKind::InvalidData, error)
            })
    }
}

#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
mod platform {
    use std::{ffi::c_void, io};

    const RUSAGE_INFO_V2: libc::c_int = 2;

    #[repr(C)]
    struct RusageInfoV2 {
        uuid: [u8; 16],
        user_time: u64,
        system_time: u64,
        remaining: [u64; 16],
    }

    #[link(name = "proc")]
    unsafe extern "C" {
        fn proc_pid_rusage(
            pid: libc::c_int,
            flavor: libc::c_int,
            buffer: *mut c_void,
        ) -> libc::c_int;
    }

    pub(super) fn sample(pid: u32) -> io::Result<(f64, u64)> {
        let mut usage = RusageInfoV2 {
            uuid: [0; 16],
            user_time: 0,
            system_time: 0,
            remaining: [0; 16],
        };
        // SAFETY: `usage` has the rusage_info_v2 C layout and sufficient size; proc_pid_rusage
        // writes that structure for the requested process and flavor.
        let result = unsafe {
            proc_pid_rusage(
                pid as libc::c_int,
                RUSAGE_INFO_V2,
                (&mut usage as *mut RusageInfoV2).cast(),
            )
        };
        if result != 0 {
            return Err(io::Error::last_os_error());
        }
        let resident_size = usage.remaining[4];
        Ok((
            (usage.user_time + usage.system_time) as f64 / 1_000_000_000.0,
            resident_size,
        ))
    }
}

#[cfg(windows)]
#[allow(unsafe_code)]
mod platform {
    use std::{io, mem::size_of};
    use windows_sys::Win32::{
        Foundation::{CloseHandle, FILETIME, HANDLE},
        System::{
            ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS},
            Threading::{GetProcessTimes, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION},
        },
    };

    struct ProcessHandle(HANDLE);

    impl Drop for ProcessHandle {
        fn drop(&mut self) {
            // SAFETY: the handle was returned by OpenProcess and is owned by this guard.
            unsafe { CloseHandle(self.0) };
        }
    }

    pub(super) fn sample(pid: u32) -> io::Result<(f64, u64)> {
        // SAFETY: OpenProcess has no pointer arguments; the returned handle is checked below.
        let handle = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
        if handle.is_null() {
            return Err(io::Error::last_os_error());
        }
        let handle = ProcessHandle(handle);

        // SAFETY: every FILETIME is a plain C integer pair, for which all-zero is valid.
        let (mut creation, mut exit, mut kernel, mut user) =
            unsafe { std::mem::zeroed::<(FILETIME, FILETIME, FILETIME, FILETIME)>() };
        // SAFETY: all FILETIME outputs are writable, and the handle is valid for this call.
        if unsafe { GetProcessTimes(handle.0, &mut creation, &mut exit, &mut kernel, &mut user) }
            == 0
        {
            return Err(io::Error::last_os_error());
        }

        // SAFETY: PROCESS_MEMORY_COUNTERS is an output-only C struct; zero initializes its fields
        // and cb is set to its exact size before the API writes into it.
        let mut memory = PROCESS_MEMORY_COUNTERS {
            cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
            ..unsafe { std::mem::zeroed() }
        };
        // SAFETY: memory points to an initialized structure whose size is passed in cb.
        if unsafe {
            GetProcessMemoryInfo(
                handle.0,
                &mut memory,
                size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
            )
        } == 0
        {
            return Err(io::Error::last_os_error());
        }

        let cpu_ticks = filetime_ticks(kernel) + filetime_ticks(user);
        Ok((
            cpu_ticks as f64 / 10_000_000.0,
            memory.WorkingSetSize as u64,
        ))
    }

    fn filetime_ticks(value: FILETIME) -> u64 {
        (u64::from(value.dwHighDateTime) << 32) | u64::from(value.dwLowDateTime)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        env,
        io::{BufRead, BufReader, Write},
        process::{Child, Command, Stdio},
        time::Duration,
    };

    struct ChildOnDrop(Child);

    impl Drop for ChildOnDrop {
        fn drop(&mut self) {
            if self.0.try_wait().ok().flatten().is_none() {
                let _ = self.0.kill();
            }
        }
    }

    impl std::ops::Deref for ChildOnDrop {
        type Target = Child;

        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    impl std::ops::DerefMut for ChildOnDrop {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.0
        }
    }

    #[test]
    fn helper_process() {
        if env::var_os("JOPE_SAMPLER_HELPER").is_none() {
            return;
        }
        let stdin = std::io::stdin();
        let mut input = BufReader::new(stdin.lock());
        println!("READY");
        let _ = std::io::stdout().flush();
        let mut command = String::new();
        input.read_line(&mut command).unwrap();
        assert_eq!(command.trim(), "GO");

        let mut memory = vec![0_u8; 200 * 1024 * 1024];
        for page in memory.chunks_mut(4096) {
            page[0] = 1;
        }
        let started = std::time::Instant::now();
        let mut value = 0_u64;
        while started.elapsed() < Duration::from_secs(2) {
            value = value.wrapping_add(1);
            std::hint::black_box(value);
        }
        println!("DONE");
        let _ = std::io::stdout().flush();
        command.clear();
        input.read_line(&mut command).unwrap();
        assert_eq!(command.trim(), "STOP");
        std::hint::black_box(memory);
    }

    #[test]
    fn samples_busy_child_cpu_and_resident_memory() {
        let child = Command::new(std::env::current_exe().unwrap())
            .arg("--exact")
            .arg("sampler::tests::helper_process")
            .arg("--nocapture")
            .env("JOPE_SAMPLER_HELPER", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn helper process");
        let mut child = ChildOnDrop(child);
        let mut output = BufReader::new(child.stdout.take().unwrap());
        read_marker(&mut output, "READY");
        let before = super::sample(child.id()).expect("sample helper before workload");

        let mut input = child.stdin.take().unwrap();
        let started = std::time::Instant::now();
        writeln!(input, "GO").unwrap();
        input.flush().unwrap();
        read_marker(&mut output, "DONE");
        let after = super::sample(child.id()).expect("sample helper after workload");
        writeln!(input, "STOP").unwrap();
        drop(input);
        assert!(child.wait().unwrap().success());

        let elapsed = started.elapsed();
        assert!(elapsed >= Duration::from_secs(2));
        let cpu_cores = (after.0 - before.0) / elapsed.as_secs_f64();
        assert!(
            (0.8..=1.2).contains(&cpu_cores),
            "measured {cpu_cores:.3} CPU cores"
        );
        assert!(
            after.1 >= before.1 + 150 * 1024 * 1024,
            "RSS did not grow by at least 150 MiB: before={} after={}",
            before.1,
            after.1
        );
    }

    fn read_marker(output: &mut impl BufRead, expected: &str) {
        let mut line = String::new();
        loop {
            line.clear();
            assert_ne!(
                output.read_line(&mut line).unwrap(),
                0,
                "child exited before {expected}"
            );
            if line.trim() == expected {
                return;
            }
        }
    }
}
