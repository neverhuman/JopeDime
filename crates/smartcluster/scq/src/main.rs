use std::{
    collections::BTreeSet,
    io::{self, Write},
    path::PathBuf,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{anyhow, Context, Result};
use clap::{Args, Parser, Subcommand};
use crossterm::{
    cursor,
    event::{self, Event as TerminalEvent, KeyCode},
    execute,
    terminal::{self, ClearType, EnterAlternateScreen, LeaveAlternateScreen},
};
use jain_smartcluster::{
    AttachRequest, CancelMode, CancelRequest, Client, ControlMessage, DaemonStatus, Environment,
    EventFilter, EventOffset, JainWorkload, JobEvent, JobId, JobPage, JobRecord, JobState,
    ListFilter, ListRequest, SendControlRequest, SubscribeRequest, Timestamp, Workload,
};
use scq::{
    parse_scq_toml_v0, ScqTomlDependencies, ScqTomlJob, ScqTomlLimits, ScqTomlPreemption,
    ScqTomlResource, ScqTomlSpec,
};
use tokio::io::AsyncReadExt;

#[path = "main/interactive.rs"]
mod interactive;
#[path = "main/output.rs"]
mod output;
#[path = "main/submission.rs"]
mod submission;

use interactive::{run_attach, run_tui, stream_events};
use output::{print_event, print_job, print_jobs, print_status};
use submission::submit_spec;

#[derive(Debug, Parser)]
#[command(name = "scq", version, about = "Jain single-node workload client")]
struct Cli {
    #[arg(
        long,
        env = "SCQ_SOCKET",
        default_value = "/run/jain-smartcluster/scqd.sock"
    )]
    socket: PathBuf,
    #[arg(long, global = true)]
    json: bool,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Doctor,
    Ping,
    Check {
        file: String,
    },
    Submit(Box<SubmitArgs>),
    Ls(ListArgs),
    Ps(ListArgs),
    Show {
        id: JobId,
    },
    Logs {
        id: JobId,
        #[arg(short = 'f', long)]
        follow: bool,
        #[arg(long, default_value_t = 0)]
        offset: u64,
    },
    Events {
        #[arg(short = 'f', long)]
        follow: bool,
        #[arg(long, default_value_t = 0)]
        offset: u64,
    },
    Attach {
        id: JobId,
        #[arg(long, default_value_t = 0)]
        offset: u64,
        #[arg(long)]
        read_only: bool,
    },
    Wait {
        id: JobId,
    },
    Cancel {
        id: JobId,
        #[arg(long)]
        checkpoint: bool,
        #[arg(long, default_value_t = 30)]
        grace_seconds: u64,
    },
    Pause {
        id: JobId,
    },
    Resume {
        id: JobId,
    },
    /// Alias for `pause`.
    Hold {
        id: JobId,
    },
    /// Alias for `resume`.
    Release {
        id: JobId,
    },
    Advance {
        id: JobId,
    },
    Nodes,
    Top,
    Tui,
    Snapshot,
}

#[derive(Debug, Args)]
struct SubmitArgs {
    /// TOML file or `-`; omit when using inline flags and `-- CMD ARGS`.
    file: Option<String>,
    #[arg(long)]
    id: Option<String>,
    #[arg(long)]
    project: Option<String>,
    #[arg(long, default_value = "normal")]
    tier: String,
    #[arg(long, default_value_t = 0)]
    gpus: u16,
    #[arg(long)]
    gpu_type: Option<String>,
    #[arg(long, default_value_t = 1)]
    cpus: u32,
    #[arg(long, default_value_t = 0)]
    omp: u32,
    #[arg(long, default_value_t = 0)]
    ram: u64,
    #[arg(long)]
    workdir: Option<String>,
    #[arg(long)]
    venv: Option<String>,
    /// Submit a shell-free allowlisted Jain worker workload instead of an argv program.
    #[arg(long, value_name = "TEMPLATE")]
    jain_template: Option<String>,
    #[arg(long)]
    no_preempt: bool,
    #[arg(long)]
    attach: bool,
    #[arg(last = true)]
    argv: Vec<String>,
}

#[derive(Debug, Args, Default)]
struct ListArgs {
    #[arg(long)]
    project: Option<String>,
    #[arg(long, default_value_t = 100)]
    limit: u32,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let client = Client::new(cli.socket);
    dispatch(&client, cli.command, cli.json).await
}

async fn dispatch(client: &Client, command: Command, json: bool) -> Result<()> {
    match command {
        Command::Doctor | Command::Ping | Command::Nodes | Command::Snapshot => {
            print_status(&client.status().await?, json)?;
        }
        Command::Check { file } => {
            let input = read_input(&file).await?;
            let spec = parse_scq_toml_v0(&input, effective_uid(), now())?;
            if json {
                println!("{}", serde_json::to_string_pretty(&spec)?);
            } else {
                println!("valid {} / {}", spec.project, spec.idempotency_key);
            }
        }
        Command::Submit(args) => {
            let attach = args.attach;
            let spec = submit_spec(*args).await?;
            let job = client.submit(spec).await?;
            print_job(&job, json)?;
            if attach {
                run_attach(client, job.id, EventOffset::ZERO, false).await?;
            }
        }
        Command::Ls(args) => {
            let page = client.list(list_request(args, false)?).await?;
            print_jobs(&page, json)?;
        }
        Command::Ps(args) => {
            let page = client.list(list_request(args, true)?).await?;
            print_jobs(&page, json)?;
        }
        Command::Show { id } => {
            let job = client
                .get(id)
                .await?
                .ok_or_else(|| anyhow!("job {id} was not found"))?;
            print_job(&job, json)?;
        }
        Command::Wait { id } => {
            let job = client.wait(id).await?;
            print_job(&job, json)?;
        }
        Command::Logs { id, follow, offset } => {
            stream_events(
                client,
                SubscribeRequest {
                    offset: EventOffset::new(offset),
                    filter: EventFilter {
                        job_id: Some(id),
                        ..EventFilter::default()
                    },
                },
                follow,
                json,
            )
            .await?;
        }
        Command::Events { follow, offset } => {
            stream_events(
                client,
                SubscribeRequest {
                    offset: EventOffset::new(offset),
                    ..SubscribeRequest::default()
                },
                follow,
                json,
            )
            .await?;
        }
        Command::Attach {
            id,
            offset,
            read_only,
        } => run_attach(client, id, EventOffset::new(offset), read_only).await?,
        Command::Cancel {
            id,
            checkpoint,
            grace_seconds,
        } => {
            let mode = if checkpoint {
                CancelMode::CheckpointThenCancel {
                    checkpoint_deadline_ms: grace_seconds.max(1).saturating_mul(1000),
                }
            } else {
                CancelMode::Graceful {
                    grace_ms: grace_seconds.max(1).saturating_mul(1000),
                }
            };
            print_job(
                &client.cancel(CancelRequest { job_id: id, mode }).await?,
                json,
            )?;
        }
        Command::Pause { id } | Command::Hold { id } => {
            print_job(&client.pause(id).await?, json)?;
        }
        Command::Resume { id } | Command::Release { id } => {
            print_job(&client.resume(id).await?, json)?;
        }
        Command::Advance { id } => {
            let ack = client
                .send_control(SendControlRequest {
                    job_id: id,
                    control: ControlMessage::Advance,
                })
                .await?;
            if json {
                println!("{}", serde_json::to_string_pretty(&ack)?);
            } else {
                println!("accepted at {}", ack.accepted_at);
            }
        }
        Command::Top => {
            print_status(&client.status().await?, json)?;
            print_jobs(&client.list(ListRequest::default()).await?, json)?;
        }
        Command::Tui => run_tui(client).await?,
    }
    Ok(())
}

fn list_request(args: ListArgs, active_only: bool) -> Result<ListRequest> {
    let states = if active_only {
        BTreeSet::from([
            JobState::Queued,
            JobState::Admitted,
            JobState::Starting,
            JobState::Running,
            JobState::Paused,
            JobState::Suspended,
            JobState::RetryWait,
        ])
    } else {
        BTreeSet::new()
    };
    Ok(ListRequest {
        filter: ListFilter {
            project: args
                .project
                .map(jain_smartcluster::ProjectId::new)
                .transpose()?,
            states,
            ..ListFilter::default()
        },
        limit: args.limit,
        ..ListRequest::default()
    })
}

async fn read_input(path: &str) -> Result<String> {
    if path == "-" {
        let mut input = String::new();
        tokio::io::stdin().read_to_string(&mut input).await?;
        Ok(input)
    } else {
        tokio::fs::read_to_string(path)
            .await
            .with_context(|| format!("read {path}"))
    }
}

fn effective_uid() -> u32 {
    nix::unistd::geteuid().as_raw()
}

fn now() -> Timestamp {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    Timestamp::from_unix_millis(u64::try_from(millis).unwrap_or(u64::MAX))
}
