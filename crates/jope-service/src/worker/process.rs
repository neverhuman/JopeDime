//! Owned process lifetime and platform containment. No foreign PIDs are killed.

// Private lifetime marker: written only after a complete response has been flushed.
pub(crate) const RETIRE_EXIT_CODE: i32 = 73;

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
    use std::{
        mem::size_of,
        os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle},
    };
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
        SetInformationJobObject, JOBOBJECT_BASIC_LIMIT_INFORMATION,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        JOB_OBJECT_LIMIT_PROCESS_MEMORY,
    };
    // OwnedHandle supplies thread-safe ownership and closes the non-inherited job on drop.
    pub(crate) struct Containment(OwnedHandle);
    impl Containment {
        pub(crate) fn assign(child: &Child, cap: u64) -> io::Result<Self> {
            let cap = usize::try_from(cap).map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidInput, "RSS cap exceeds address space")
            })?;
            // SAFETY: null attributes create an unnamed, non-inheritable owned Job Object.
            let raw = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
            if raw.is_null() {
                return Err(io::Error::last_os_error());
            }
            // SAFETY: the non-null CreateJobObjectW handle is owned exactly once from here.
            let job = Self(unsafe { OwnedHandle::from_raw_handle(raw.cast()) });
            let info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION {
                BasicLimitInformation: JOBOBJECT_BASIC_LIMIT_INFORMATION {
                    LimitFlags: JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                        | JOB_OBJECT_LIMIT_PROCESS_MEMORY,
                    ..Default::default()
                },
                ProcessMemoryLimit: cap,
                ..Default::default()
            };
            // SAFETY: the initialized C structure matches this class and size; the owned job
            // remains alive for the synchronous call, which only reads this structure.
            let configured = unsafe {
                SetInformationJobObject(
                    job.0.as_raw_handle().cast(),
                    JobObjectExtendedLimitInformation,
                    (&info as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(),
                    size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                )
            };
            if configured == 0 {
                return Err(io::Error::last_os_error());
            }
            // SAFETY: job and child are live owned kernel handles; no request has been sent.
            let assigned = unsafe {
                AssignProcessToJobObject(job.0.as_raw_handle().cast(), child.as_raw_handle().cast())
            };
            if assigned == 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(job)
        }
    }
}
#[cfg(windows)]
pub(crate) use windows::Containment;
