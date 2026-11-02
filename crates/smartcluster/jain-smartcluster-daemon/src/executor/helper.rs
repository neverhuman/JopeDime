use super::*;

pub fn run_exec_helper(arguments: &[String]) -> Result<(), ExecutorError> {
    let HelperInvocation {
        expected_parent,
        identity,
        program,
        arguments,
    } = parse_helper_arguments(arguments)?;
    nix::sys::prctl::set_pdeathsig(Signal::SIGKILL).map_err(ExecutorError::ArmParentDeathSignal)?;
    verify_helper_parent(expected_parent)?;
    setsid().map_err(ExecutorError::SetSid)?;
    // If stdin is the PTY slave, establish it as this fresh session's controlling
    // terminal. Pipes return ENOTTY and intentionally continue without one.
    let _ = rustix::process::ioctl_tiocsctty(std::io::stdin());
    raise(Signal::SIGSTOP).map_err(ExecutorError::StopSelf)?;
    nix::sys::prctl::set_pdeathsig(None::<Signal>)
        .map_err(ExecutorError::ClearParentDeathSignal)?;
    verify_helper_parent(expected_parent)?;
    if let Some((uid, gid)) = identity {
        nix::unistd::setgroups(&[]).map_err(ExecutorError::ClearSupplementaryGroups)?;
        nix::unistd::setgid(nix::unistd::Gid::from_raw(gid)).map_err(ExecutorError::SetGid)?;
        nix::unistd::setuid(nix::unistd::Uid::from_raw(uid)).map_err(ExecutorError::SetUid)?;
    }
    nix::sys::prctl::set_no_new_privs().map_err(ExecutorError::NoNewPrivileges)?;
    let error = std::process::Command::new(program).args(arguments).exec();
    Err(ExecutorError::Exec {
        program: PathBuf::from(program),
        source: error,
    })
}

pub(super) struct HelperInvocation<'a> {
    pub(super) expected_parent: Pid,
    pub(super) identity: Option<(u32, u32)>,
    pub(super) program: &'a str,
    pub(super) arguments: &'a [String],
}

pub(super) fn parse_helper_arguments(
    arguments: &[String],
) -> Result<HelperInvocation<'_>, ExecutorError> {
    let [expected_parent, uid, gid, remainder @ ..] = arguments else {
        return Err(ExecutorError::HelperArguments);
    };
    let expected_parent = expected_parent
        .parse::<i32>()
        .ok()
        .filter(|pid| *pid > 0)
        .map(Pid::from_raw)
        .ok_or(ExecutorError::HelperParent)?;
    let identity = match (uid.as_str(), gid.as_str()) {
        ("-", "-") => None,
        ("-", _) | (_, "-") => return Err(ExecutorError::HelperIdentity),
        (uid, gid) => Some((
            uid.parse().map_err(|_| ExecutorError::HelperIdentity)?,
            gid.parse().map_err(|_| ExecutorError::HelperIdentity)?,
        )),
    };
    let (program, program_arguments) = remainder
        .split_first()
        .ok_or(ExecutorError::HelperArguments)?;
    Ok(HelperInvocation {
        expected_parent,
        identity,
        program,
        arguments: program_arguments,
    })
}

fn verify_helper_parent(expected: Pid) -> Result<(), ExecutorError> {
    let actual = nix::unistd::getppid();
    if actual == expected {
        Ok(())
    } else {
        Err(ExecutorError::HelperParentChanged { expected, actual })
    }
}
