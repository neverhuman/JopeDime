//! Owned process lifetime and platform containment. No foreign PIDs are killed.

use std::{
    io,
    process::{Child, Command},
};

#[cfg(unix)]
pub(crate) fn prepare(command: &mut Command) {
    use std::os::unix::process::CommandExt;
    command.process_group(0);
    #[cfg(target_os = "linux")]
    {
        let parent = std::process::id();
        // SAFETY: this closure performs only async-signal-safe syscalls between fork and exec.
        // Capturing the expected parent closes the death-before-prctl race.
        #[allow(unsafe_code)]
        unsafe {
            command.pre_exec(move || {
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                    return Err(io::Error::last_os_error());
                }
                if libc::getppid() as u32 != parent {
                    return Err(io::Error::new(
                        io::ErrorKind::Interrupted,
                        "worker parent exited",
                    ));
                }
                Ok(())
            });
        }
    }
}

#[cfg(windows)]
pub(crate) fn prepare(_: &mut Command) {}

/// An additional child-side check also protects launches outside the pool.
pub(crate) fn protect_parent() -> io::Result<()> {
    #[cfg(target_os = "linux")]
    {
        #[allow(unsafe_code)]
        // SAFETY: getppid/prctl have no pointer arguments here. Recheck after arming the signal.
        unsafe {
            let parent = libc::getppid();
            if parent == 1 {
                return Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "worker has no parent",
                ));
            }
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                return Err(io::Error::last_os_error());
            }
            if libc::getppid() != parent {
                return Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "worker parent exited",
                ));
            }
        }
    }
    Ok(())
}

pub(crate) fn kill(child: &mut Child) {
    // try_wait caches a reaped status; never signal a PID/group after it can be reused.
    if child.try_wait().ok().flatten().is_some() {
        return;
    }
    #[cfg(unix)]
    {
        #[allow(unsafe_code)]
        // SAFETY: the pool created this process group and retains the unreaped group leader.
        unsafe {
            libc::kill(-(child.id() as libc::pid_t), libc::SIGKILL);
        }
    }
    let _ = child.kill();
    let _ = child.wait();
}

#[cfg(not(windows))]
pub(crate) struct Containment;
#[cfg(not(windows))]
impl Containment {
    pub(crate) fn assign(_: &Child, _: u64) -> io::Result<Self> {
        Ok(Self)
    }
}

#[cfg(windows)]
#[allow(unsafe_code)]
mod windows {
    use super::*;
    use std::{mem::size_of, os::windows::io::AsRawHandle};
    use windows_sys::Win32::{
        Foundation::{CloseHandle, HANDLE},
        System::JobObjects::{
            AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
            SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, JOB_OBJECT_LIMIT_PROCESS_MEMORY,
        },
    };
    pub(crate) struct Containment(HANDLE);
    // SAFETY: the owned Job Object handle supports use/closure on any thread.
    unsafe impl Send for Containment {}
    impl Containment {
        pub(crate) fn assign(child: &Child, cap: u64) -> io::Result<Self> {
            let cap = usize::try_from(cap).map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidInput, "RSS cap exceeds address space")
            })?;
            // SAFETY: unnamed, non-inheritable owned Job Object; return handle checked.
            let raw = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
            if raw.is_null() {
                return Err(io::Error::last_os_error());
            }
            let job = Self(raw);
            // SAFETY: the C structure has a valid all-zero initial representation.
            let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
            info.BasicLimitInformation.LimitFlags =
                JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_PROCESS_MEMORY;
            info.ProcessMemoryLimit = cap;
            // SAFETY: info is initialized, size/layout match the requested information class;
            // process and job handles remain valid for the duration of both calls.
            if unsafe {
                SetInformationJobObject(
                    job.0,
                    JobObjectExtendedLimitInformation,
                    (&info as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(),
                    size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                )
            } == 0
                || unsafe { AssignProcessToJobObject(job.0, child.as_raw_handle().cast()) } == 0
            {
                return Err(io::Error::last_os_error());
            }
            Ok(job)
        }
    }
    impl Drop for Containment {
        fn drop(&mut self) {
            // SAFETY: this is the sole owner of the non-inherited Job Object handle.
            unsafe {
                CloseHandle(self.0);
            }
        }
    }
}
#[cfg(windows)]
pub(crate) use windows::Containment;
