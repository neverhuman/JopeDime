use super::*;
use std::fs::File;
use std::process::Command;

pub struct SystemdUserHost;

impl SystemdUserHost {
    fn systemctl(args: &[&str]) -> Result<std::process::Output> {
        Command::new("systemctl")
            .arg("--user")
            .args(args)
            .output()
            .with_context(|| format!("run systemctl --user {}", args.join(" ")))
    }
}

impl ActivationHost for SystemdUserHost {
    fn is_enabled(&mut self, unit: &str) -> Result<bool> {
        Ok(Self::systemctl(&["is-enabled", "--quiet", unit])?
            .status
            .success())
    }

    fn is_active(&mut self, unit: &str) -> Result<bool> {
        Ok(Self::systemctl(&["is-active", "--quiet", unit])?
            .status
            .success())
    }

    fn validate_candidate(&mut self, staged_root: &Path, units: &[String]) -> Result<()> {
        let version = Command::new("systemd-analyze").arg("--version").output()?;
        ensure!(version.status.success(), "systemd-analyze --version failed");
        let output = Command::new("systemd-analyze")
            .arg("--user")
            .arg(format!("--root={}", staged_root.display()))
            .arg("verify")
            .args(units)
            .output()?;
        ensure!(
            output.status.success(),
            "candidate user-unit validation failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
        Ok(())
    }

    fn daemon_reload(&mut self) -> Result<()> {
        let output = Self::systemctl(&["daemon-reload"])?;
        ensure!(
            output.status.success(),
            "systemctl --user daemon-reload failed"
        );
        Ok(())
    }

    fn enable_timers(&mut self, timers: &[String]) -> Result<()> {
        if timers.is_empty() {
            return Ok(());
        }
        let mut args = vec!["enable", "--now"];
        args.extend(timers.iter().map(String::as_str));
        let output = Self::systemctl(&args)?;
        ensure!(output.status.success(), "enable DIME timers failed");
        Ok(())
    }

    fn try_restart_services(&mut self, services: &[String]) -> Result<()> {
        if services.is_empty() {
            return Ok(());
        }
        let mut args = vec!["try-restart"];
        args.extend(services.iter().map(String::as_str));
        let output = Self::systemctl(&args)?;
        ensure!(
            output.status.success(),
            "restart affected DIME services failed"
        );
        Ok(())
    }

    fn restore_states(&mut self, units: &[ActivationUnitPreimageV1]) -> Result<()> {
        for unit in units {
            if unit.enabled {
                let output = Self::systemctl(&["enable", &unit.unit])?;
                ensure!(output.status.success(), "restore enabled state failed");
            } else {
                let output = Self::systemctl(&["disable", &unit.unit])?;
                ensure!(output.status.success(), "restore disabled state failed");
            }
            if unit.active {
                let output = Self::systemctl(&["start", &unit.unit])?;
                ensure!(output.status.success(), "restore active state failed");
            } else {
                let output = Self::systemctl(&["stop", &unit.unit])?;
                ensure!(output.status.success(), "restore inactive state failed");
            }
        }
        Ok(())
    }
}

pub struct ActivationLock {
    _file: File,
}

impl ActivationLock {
    pub fn acquire(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .mode(0o600)
            .open(path)?;
        // Every failure, including EWOULDBLOCK or other errors, is refused below and
        // drops/closes `file`.
        ensure!(
            file.try_lock().is_ok(),
            "another DIME activation transaction owns the lock"
        );
        Ok(Self { _file: file })
    }
}
