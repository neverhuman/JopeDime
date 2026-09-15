use crate::manifest::*;
use anyhow::{ensure, Result};
use serde_json::{json, Value};
use std::{
    fs,
    path::Path,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

fn command(cmd: &str, args: &[&str]) -> Value {
    match Command::new(cmd).args(args).output() {
        Ok(o) => {
            json!({"ok":o.status.success(),"stdout":String::from_utf8_lossy(&o.stdout),"stderr":String::from_utf8_lossy(&o.stderr)})
        }
        Err(e) => json!({"ok":false,"error":e.to_string()}),
    }
}
pub fn doctor() -> Value {
    json!({"schema":"parallel.doctor.v1","neural_compiled":cfg!(feature="neural"),"rust":command("rustc",&["--version"]),"gpus":command("nvidia-smi",&["--query-gpu=index,uuid,name,memory.total,utilization.gpu,memory.used,power.draw,temperature.gpu","--format=csv,noheader,nounits"]),"topology":command("nvidia-smi",&["topo","-m"]),"cpu":command("lscpu",&["-J"]),"numa":command("numactl",&["--hardware"]),"torch":command("python3",&["-c","import torch; print(torch.__version__); print(torch.__file__); print(torch.cuda.nccl.version()); print(torch._C._GLIBCXX_USE_CXX11_ABI)"]),"disk":command("df",&["-h","."]),"memory":fs::read_to_string("/proc/meminfo").unwrap_or_default()})
}
pub fn status(root: &Path) -> Result<Value> {
    let run = crate::dashboard::campaign(root);
    let hardware: Value = read_json(&root.join("runs/hardware.json")).unwrap_or_default();
    let gp: Value = read_json(&root.join("runs/gp-service.json")).unwrap_or_default();
    let neural: Value = read_json(&root.join("runs/neural-service.json")).unwrap_or_default();
    let encoding: Value = read_json(&root.join("runs/encoding-service.json")).unwrap_or_default();
    let count = |key: &str| gp[key].as_u64().unwrap_or(0);
    let total =
        count("completed_views") + count("failed_views") + count("active") + count("queued");
    Ok(
        json!({"schema":"parallel.status.v2","run":run,"hardware":hardware,
        "gp":{"active":gp["active"],"queued":gp["queued"],"completed":gp["completed_views"],"failed":gp["failed_views"],"total":total,"completed_percent":if total>0{Some(100.0*count("completed_views") as f64/total as f64)}else{None},"evaluations_per_second":gp["evaluations_per_second"],"timestamp":gp["timestamp"]},
        "neural":neural,"encoding":encoding,"unix_seconds":SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs()}),
    )
}
pub fn pilot_gate(root: &Path) -> Result<()> {
    let required = [
        "manifests/generated/audit/audit.json",
        "manifests/generated/transfer-receipt.json",
        "runs/parity/encoders.json",
        "runs/parity/policy.json",
        "runs/parity/optimizer.json",
        "runs/parity/gp.json",
        "runs/parity/distributed-1.json",
        "runs/parity/distributed-2.json",
        "runs/parity/distributed-8.json",
        "manifests/pilot.json",
    ];
    for p in required {
        ensure!(root.join(p).is_file(), "pilot prerequisite absent: {p}");
    }
    let audit: AuditReport = read_json(&root.join(required[0]))?;
    ensure!(audit.complete, "provenance inventory is unresolved");
    anyhow::bail!("pilot controller has not yet been qualified; no training started")
}
