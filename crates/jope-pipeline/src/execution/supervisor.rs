use super::*;

/// Existing provisioned service only: this library never starts/stops services,
/// changes the runner fleet, or relaxes a missing isolation capability.
#[derive(Clone, Debug)]
pub struct Supervisor {
    pub unit: String,
    pub cgroup_root: PathBuf,
}
impl Supervisor {
    pub(super) fn verify(&self, experiment: &Experiment, global_deadline: u64) -> Result<()> {
        ensure!(
            self.unit.starts_with("jope-asm-")
                && self.unit.ends_with(".service")
                && self
                    .unit
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"-._".contains(&b)),
            "unexpected supervisor unit"
        );
        let props = systemd_properties(&self.unit)?;
        for (key, expected) in [
            ("Type", "exec"),
            ("ExitType", "main"),
            ("KillMode", "control-group"),
            ("SendSIGKILL", "yes"),
            ("Restart", "no"),
            ("OOMPolicy", "continue"),
            ("RemainAfterExit", "no"),
            ("ActiveState", "active"),
        ] {
            ensure!(
                props.get(key).map(String::as_str) == Some(expected),
                "supervisor {key} must be {expected}"
            );
        }
        ensure!(
            matches!(
                props.get("FinalKillSignal").map(String::as_str),
                Some("9" | "SIGKILL")
            ),
            "supervisor final signal must be SIGKILL"
        );
        ensure!(
            props.get("MainPID").and_then(|v| v.parse::<u32>().ok()) == Some(std::process::id()),
            "controller must be supervisor MainPID"
        );
        let control = props
            .get("ControlGroup")
            .context("supervisor cgroup missing")?;
        ensure!(
            self.cgroup_root == Path::new("/sys/fs/cgroup/system.slice").join(&self.unit)
                && self.cgroup_root
                    == Path::new("/sys/fs/cgroup").join(control.trim_start_matches('/')),
            "supervisor must be the expected system.slice cgroup"
        );
        ensure!(
            fs::canonicalize(&self.cgroup_root)? == self.cgroup_root
                && statfs(&self.cgroup_root)?.filesystem_type() == CGROUP2_SUPER_MAGIC,
            "real physical cgroup v2 required"
        );
        let own = fs::read_to_string("/proc/self/cgroup")?;
        ensure!(
            own.lines()
                .any(|line| line == format!("0::{control}/controller")),
            "controller must occupy delegated controller subgroup"
        );
        let runtime = parse_timespan(
            props
                .get("RuntimeMaxUSec")
                .context("runtime limit absent")?,
        )?;
        let stop = parse_timespan(props.get("TimeoutStopUSec").context("stop bound absent")?)?;
        if let Some(extra) = props.get("RuntimeRandomizedExtraUSec") {
            ensure!(
                parse_timespan(extra)? == 0,
                "randomized supervisor runtime is forbidden"
            );
        }
        ensure!(
            stop <= DRAIN_MS * 1000 && stop > 0,
            "supervisor stop grace exceeds one second"
        );
        let started: u64 = props
            .get("ActiveEnterTimestampMonotonic")
            .context("supervisor start absent")?
            .parse()?;
        let monotonic = clock_gettime(ClockId::CLOCK_MONOTONIC)?;
        let mono_us = monotonic.tv_sec() as u64 * 1_000_000 + monotonic.tv_nsec() as u64 / 1000;
        ensure!(
            mono_us >= started && runtime > mono_us - started,
            "supervisor runtime expired"
        );
        let remaining_ms = (runtime - (mono_us - started) + stop).div_ceil(1000);
        ensure!(
            now_ms()?
                .checked_add(remaining_ms)
                .is_some_and(|end| end <= global_deadline),
            "supervisor may outlive original global deadline"
        );
        let resources = &experiment.resources;
        verify_limit(&self.cgroup_root, "memory.max", resources.memory_bytes)?;
        ensure!(
            control_text(&self.cgroup_root, "memory.swap.max")? == "0",
            "swap must be zero"
        );
        verify_cpu(&self.cgroup_root, resources.cpus as u64)?;
        let cpus = parse_cpus(&control_text(&self.cgroup_root, "cpuset.cpus.effective")?)?;
        ensure!(
            !cpus.is_empty()
                && cpus.len() <= resources.cpus as usize
                && cpus.len() >= resources.workers as usize,
            "supervisor CPU allocation differs"
        );
        ensure!(
            control_text(&self.cgroup_root, "cgroup.procs")?.is_empty(),
            "supervisor inner cgroup has processes"
        );
        // systemd delegates these controllers and moves MainPID into /controller,
        // but the delegate must enable them for its own child groups. Initialize
        // only this already-qualified, empty service root; never an ancestor.
        let available = control_text(&self.cgroup_root, "cgroup.controllers")?;
        ensure!(
            ["cpu", "memory", "cpuset"]
                .iter()
                .all(|v| available.split_whitespace().any(|e| e == *v)),
            "required controllers were not delegated"
        );
        fs::write(
            self.cgroup_root.join("cgroup.subtree_control"),
            "+cpu +cpuset +memory",
        )
        .context("enable delegated fit controllers")?;
        let enabled = control_text(&self.cgroup_root, "cgroup.subtree_control")?;
        ensure!(
            ["cpu", "memory", "cpuset"]
                .iter()
                .all(|v| enabled.split_whitespace().any(|e| e == *v)),
            "required delegated controllers unavailable"
        );
        Ok(())
    }
}
