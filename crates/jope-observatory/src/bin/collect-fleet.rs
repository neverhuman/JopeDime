//! Read-only probes of the four explicitly allowed fleet nodes. No process argv.
use anyhow::{bail, Context, Result};
use clap::Parser;
use jope_observatory::now;
use serde_json::{json, Value};
use std::{
    fs,
    io::Read,
    path::PathBuf,
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    output_dir: PathBuf,
    #[arg(long, default_value = "1")]
    cycles: u32,
    #[arg(long)]
    continuous: bool,
    #[arg(long, default_value = "30")]
    interval_seconds: u64,
}
#[derive(Clone, Copy)]
struct Node {
    host: &'static str,
    target: &'static str,
    port: u16,
}
const NODES: [Node; 4] = [
    Node {
        host: "xbabe1",
        target: "ubuntu@162.218.217.123",
        port: 696,
    },
    Node {
        host: "xbabe2",
        target: "ubuntu@162.218.217.123",
        port: 6969,
    },
    Node {
        host: "xbabe3",
        target: "ubuntu@162.218.217.123",
        port: 6970,
    },
    Node {
        host: "xbabe4",
        target: "xbabe4",
        port: 0,
    },
];
fn query(node: Node, command: &str) -> Result<String> {
    // Only fixed source-code command strings reach the remote shell.
    let local = fs::read_to_string("/etc/hostname").is_ok_and(|s| s.trim() == node.host);
    let mut c = if local {
        let mut cmd = Command::new("/bin/sh");
        cmd.args(["-c", &format!("LC_ALL=C {command}")]);
        cmd
    } else {
        let mut cmd = Command::new("/usr/bin/ssh");
        cmd.args([
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            "-o",
            "ServerAliveInterval=5",
            "-o",
            "ServerAliveCountMax=1",
        ]);
        if node.host == "xbabe1" {
            cmd.args(["-o", "HostKeyAlias=xbabe1"]);
        }
        if node.port > 0 {
            cmd.args(["-p", &node.port.to_string()]);
        }
        cmd.arg(node.target).arg(format!("LC_ALL=C {command}"));
        cmd
    };
    let mut child = c
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()?;
    let out = child.stdout.take().context("probe output")?;
    let reader = thread::spawn(move || {
        let mut b = vec![];
        out.take(16_385).read_to_end(&mut b).map(|_| b)
    });
    let start = Instant::now();
    loop {
        if child.try_wait()?.is_some() {
            break;
        }
        if start.elapsed() > Duration::from_secs(12) {
            child.kill()?;
            let _ = child.wait();
            bail!("probe timeout");
        }
        thread::sleep(Duration::from_millis(50));
    }
    let status = child.wait()?;
    let b = reader.join().map_err(|_| anyhow::anyhow!("reader"))??;
    if !status.success() || b.len() > 16_384 {
        bail!("probe unavailable");
    }
    Ok(String::from_utf8(b)?)
}
fn numbers(s: &str) -> Vec<f64> {
    s.split_whitespace()
        .filter_map(|v| v.parse().ok())
        .collect()
}
fn probe(node: Node) -> Value {
    let at = now();
    let os = query(node, "uname -s");
    let Ok(os) = os else {
        return json!({"schema":"jope.fleet-node-observation/v1","host":node.host,"observed_at":at,"status":"unavailable","cpu_count":null,"load":null,"gpus":null,"workload_scope":"Node probe unavailable; no idle/learning inference"});
    };
    let mac = os.trim() == "Darwin";
    if (node.host == "xbabe4" && !mac) || (node.host != "xbabe4" && os.trim() != "Linux") {
        return json!({"schema":"jope.fleet-node-observation/v1","host":node.host,"observed_at":at,"status":"unavailable","workload_scope":"OS identity mismatch; no further probes executed"});
    }
    let cores = query(
        node,
        if mac {
            "sysctl -n hw.logicalcpu"
        } else {
            "getconf _NPROCESSORS_ONLN"
        },
    )
    .ok()
    .and_then(|s| s.trim().parse::<u64>().ok());
    let load = query(
        node,
        if mac {
            "sysctl -n vm.loadavg"
        } else {
            "cat /proc/loadavg"
        },
    )
    .ok()
    .map(|s| numbers(&s).into_iter().take(3).collect::<Vec<_>>());
    let (total, available) = if mac {
        (
            query(node, "sysctl -n hw.memsize")
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok()),
            None,
        )
    } else {
        let s = query(node, "cat /proc/meminfo").unwrap_or_default();
        let field = |key: &str| {
            s.lines()
                .find(|l| l.starts_with(key))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse::<u64>().ok())
                .map(|v| v * 1024)
        };
        (field("MemTotal:"), field("MemAvailable:"))
    };
    let disk = query(node, "df -Pk /").ok().and_then(|s| {
        s.lines()
            .nth(1)
            .and_then(|l| l.split_whitespace().nth(3))
            .and_then(|v| v.parse::<u64>().ok())
            .map(|v| v * 1024)
    });
    let gpus = if mac {
        None
    } else {
        query(node,"nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits").ok().map(|s|s.lines().filter_map(|line|{
        let fields:Vec<_>=line.split(',').map(str::trim).collect();if fields.len()!=4{return None;}
        let n=|i:usize|fields[i].parse::<f64>().ok().filter(|n|n.is_finite()&&*n>=0.);
        Some(json!({"name":fields[0],"utilization_percent":n(1),"memory_used_bytes":n(2).map(|v|(v*1_048_576.)as u64),"memory_total_bytes":n(3).map(|v|(v*1_048_576.)as u64)}))
    }).collect::<Vec<_>>())
    };
    json!({"schema":"jope.fleet-node-observation/v1","host":node.host,"observed_at":at,"status":"observed","cpu_count":cores,"load":load,"memory_total_bytes":total,"memory_available_bytes":available,"disk_available_bytes":disk,"gpus":gpus,"workload_scope":"Read-only host-wide resource counters. No argv or environment read; GPU use is not attributed to JopeDime learning. macOS available memory and non-NVIDIA GPU utilization remain unknown."})
}
fn main() -> Result<()> {
    let args = Args::parse();
    if args.cycles == 0
        || args.cycles > 2880
        || args.interval_seconds < 15
        || args.interval_seconds > 3600
    {
        bail!("finite cycles1..2880 and interval15..3600 required");
    }
    fs::create_dir_all(&args.output_dir)?;
    let mut cycle = 0u64;
    loop {
        let jobs: NodeJobs = NODES
            .into_iter()
            .map(|n| thread::spawn(move || (n.host, probe(n))))
            .collect();
        for job in jobs {
            let (host, value) = job
                .join()
                .map_err(|_| anyhow::anyhow!("node probe failed"))?;
            let p = args.output_dir.join(format!("{host}.json"));
            let temp = p.with_extension(format!("{}.new", std::process::id()));
            fs::write(&temp, serde_json::to_vec(&value)?)?;
            fs::rename(&temp, &p)?;
            println!("{} {} {}", host, value["status"], value["observed_at"]);
        }
        cycle += 1;
        if !args.continuous && cycle >= u64::from(args.cycles) {
            break;
        }
        thread::sleep(Duration::from_secs(args.interval_seconds));
    }
    Ok(())
}
type NodeJobs = Vec<thread::JoinHandle<(&'static str, Value)>>;
