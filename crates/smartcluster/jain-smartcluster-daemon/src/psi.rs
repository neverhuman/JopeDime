use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::io::{unix::AsyncFd, Interest};

use crate::config::PsiConfig;
use crate::overload::PressureLevel;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum PsiResource {
    Cpu,
    Memory,
    Io,
}

impl PsiResource {
    const fn file_name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu.pressure",
            Self::Memory => "memory.pressure",
            Self::Io => "io.pressure",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PsiLine {
    pub avg10: f64,
    pub avg60: f64,
    pub avg300: f64,
    pub total_us: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PsiSample {
    pub some: PsiLine,
    pub full: Option<PsiLine>,
}

impl PsiSample {
    pub fn parse(source: &str) -> Result<Self, PsiError> {
        let mut some = None;
        let mut full = None;
        for line in source.lines() {
            let mut fields = line.split_whitespace();
            let kind = fields.next().ok_or(PsiError::MalformedSample)?;
            let mut parsed = PsiLine::default();
            for field in fields {
                let (name, value) = field.split_once('=').ok_or(PsiError::MalformedSample)?;
                match name {
                    "avg10" => parsed.avg10 = parse_float(value)?,
                    "avg60" => parsed.avg60 = parse_float(value)?,
                    "avg300" => parsed.avg300 = parse_float(value)?,
                    "total" => {
                        parsed.total_us = value.parse().map_err(|_| PsiError::MalformedSample)?;
                    }
                    _ => {}
                }
            }
            match kind {
                "some" => some = Some(parsed),
                "full" => full = Some(parsed),
                _ => {}
            }
        }
        Ok(Self {
            some: some.ok_or(PsiError::MissingSome)?,
            full,
        })
    }
}

fn parse_float(value: &str) -> Result<f64, PsiError> {
    value.parse().map_err(|_| PsiError::MalformedSample)
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PsiEvent {
    pub resource: PsiResource,
    pub level: PressureLevel,
    pub sample: PsiSample,
}

pub struct PsiTrigger {
    resource: PsiResource,
    level: PressureLevel,
    file: AsyncFd<File>,
}

impl PsiTrigger {
    pub fn open(
        cgroup_path: &Path,
        resource: PsiResource,
        level: PressureLevel,
        stall_us: u64,
        window_us: u64,
    ) -> Result<Self, PsiError> {
        if stall_us == 0 || window_us == 0 || stall_us > window_us {
            return Err(PsiError::InvalidTrigger);
        }
        let path = cgroup_path.join(resource.file_name());
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|source| PsiError::Open {
                path: path.clone(),
                source,
            })?;
        write_trigger(&mut file, stall_us, window_us).map_err(|source| PsiError::Register {
            path: path.clone(),
            source,
        })?;
        file.seek(SeekFrom::Start(0))?;
        let file = AsyncFd::with_interest(file, Interest::PRIORITY)?;
        Ok(Self {
            resource,
            level,
            file,
        })
    }

    pub async fn wait(&self) -> Result<PsiEvent, PsiError> {
        let mut readiness = self.file.ready(Interest::PRIORITY).await?;
        readiness.clear_ready();
        let mut source = String::new();
        let mut file = self.file.get_ref();
        file.seek(SeekFrom::Start(0))?;
        file.read_to_string(&mut source)?;
        Ok(PsiEvent {
            resource: self.resource,
            level: self.level,
            sample: PsiSample::parse(&source)?,
        })
    }
}

fn write_trigger(writer: &mut impl Write, stall_us: u64, window_us: u64) -> std::io::Result<()> {
    // Formatting directly into the pressure file can submit incomplete trigger fragments.
    let trigger = format!("some {stall_us} {window_us}");
    writer.write_all(trigger.as_bytes())
}

pub struct PsiMonitor {
    triggers: Vec<PsiTrigger>,
}

impl PsiMonitor {
    pub fn register(cgroup_path: &Path, config: PsiConfig) -> Result<Self, PsiError> {
        let mut triggers = Vec::with_capacity(6);
        for resource in [PsiResource::Cpu, PsiResource::Memory, PsiResource::Io] {
            triggers.push(PsiTrigger::open(
                cgroup_path,
                resource,
                PressureLevel::Warning,
                config.warning_stall_us,
                config.window_us,
            )?);
            triggers.push(PsiTrigger::open(
                cgroup_path,
                resource,
                PressureLevel::Critical,
                config.critical_stall_us,
                config.window_us,
            )?);
        }
        Ok(Self { triggers })
    }

    pub fn spawn(self, sender: tokio::sync::mpsc::Sender<PsiEvent>) {
        for trigger in self.triggers {
            let sender = sender.clone();
            tokio::spawn(async move {
                while let Ok(event) = trigger.wait().await {
                    if sender.send(event).await.is_err() {
                        break;
                    }
                }
            });
        }
    }
}

#[derive(Debug, Error)]
pub enum PsiError {
    #[error("invalid PSI trigger threshold/window")]
    InvalidTrigger,
    #[error("cannot open PSI source {path}: {source}")]
    Open {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("cannot register PSI trigger at {path}: {source}")]
    Register {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("malformed PSI sample")]
    MalformedSample,
    #[error("PSI sample has no `some` line")]
    MissingSome,
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TriggerWriter {
        reject: bool,
        calls: usize,
    }

    impl Write for TriggerWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.calls += 1;
            if self.reject || bytes != b"some 20000 1000000" {
                return Err(std::io::Error::from(std::io::ErrorKind::InvalidInput));
            }
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn registration_submits_a_complete_trigger_and_propagates_refusal() {
        let mut writer = TriggerWriter {
            reject: false,
            calls: 0,
        };
        write_trigger(&mut writer, 20_000, 1_000_000).expect("complete trigger");
        assert_eq!(writer.calls, 1);
        writer.reject = true;
        assert_eq!(
            write_trigger(&mut writer, 20_000, 1_000_000)
                .unwrap_err()
                .kind(),
            std::io::ErrorKind::InvalidInput
        );
    }

    #[test]
    fn invalid_trigger_is_refused_before_opening_a_pressure_file() {
        let directory = tempfile::tempdir().unwrap();
        for (stall, window) in [(0, 100), (100, 0), (101, 100)] {
            assert!(matches!(
                PsiTrigger::open(
                    directory.path(),
                    PsiResource::Cpu,
                    PressureLevel::Warning,
                    stall,
                    window
                ),
                Err(PsiError::InvalidTrigger)
            ));
        }
    }

    #[test]
    fn parses_kernel_pressure_sample() {
        let sample = PsiSample::parse(
            "some avg10=1.25 avg60=0.50 avg300=0.10 total=991\n\
             full avg10=0.20 avg60=0.10 avg300=0.01 total=17\n",
        )
        .expect("parse");
        assert_eq!(sample.some.avg10, 1.25);
        assert_eq!(sample.some.total_us, 991);
        assert_eq!(sample.full.expect("full").total_us, 17);
    }

    #[test]
    fn missing_some_line_fails_closed() {
        assert!(matches!(
            PsiSample::parse("full avg10=0 avg60=0 avg300=0 total=0"),
            Err(PsiError::MissingSome)
        ));
    }
}
