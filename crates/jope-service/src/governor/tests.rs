use super::*;
use crate::telemetry::{GpuReading, ProcessReading};

struct Readings {
    memory: Result<u64, String>,
    disk: Result<u64, String>,
}
impl Readings {
    fn new(memory: u64, disk: u64) -> Self {
        Self {
            memory: Ok(memory),
            disk: Ok(disk),
        }
    }
}
impl Sensors for Readings {
    fn mem_available(&mut self) -> Result<u64, String> {
        self.memory.clone()
    }
    fn disk_free(&mut self) -> Result<u64, String> {
        self.disk.clone()
    }
    fn process(&mut self, _: u32) -> Result<ProcessReading, String> {
        panic!("no extra process probe")
    }
    fn logical_cpus(&mut self) -> Result<f64, String> {
        panic!("no CPU probe")
    }
    fn allowed_cpus(&mut self) -> Result<f64, String> {
        panic!("no CPU probe")
    }
    fn gpus(&mut self) -> Result<Vec<GpuReading>, String> {
        panic!("governor never probes GPU")
    }
}

#[derive(Default)]
struct Effects {
    calls: Vec<String>,
    workers: u32,
    admission: bool,
    fail_resize: bool,
    partial_resize: bool,
    fail_checkpoint: bool,
    fail_admission: bool,
}
impl Control for Effects {
    fn resource_admission(&mut self, allowed: bool) -> io::Result<()> {
        if self.fail_admission {
            return Err(io::Error::other("admission gate unavailable"));
        }
        self.admission = allowed;
        self.calls.push(format!("admit:{allowed}"));
        Ok(())
    }
    fn live_workers(&mut self, workers: NonZeroU32) -> io::Result<()> {
        if self.fail_resize {
            if self.partial_resize {
                self.workers = workers.get();
            }
            return Err(io::Error::other("resize unavailable"));
        }
        self.workers = workers.get();
        self.calls.push(format!("workers:{}", workers.get()));
        Ok(())
    }
    fn checkpoint_and_pause(&mut self, reason: ErrorCode) -> io::Result<()> {
        assert!(!self.admission, "stop admission precedes checkpoint");
        assert_eq!(reason, ErrorCode::DiskLow);
        if self.fail_checkpoint {
            return Err(io::Error::other("checkpoint unavailable"));
        }
        self.calls
            .extend(["commit".into(), "pause:DISK_LOW".into()]);
        Ok(())
    }
}
fn governor(workers: u32) -> Governor {
    Governor::new(
        NonZeroU64::new(1000).unwrap(),
        NonZeroU32::new(workers).unwrap(),
        100,
    )
}
mod admission;
mod replay;
mod resources;
