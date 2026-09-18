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

async fn submit_spec(args: SubmitArgs) -> Result<jain_smartcluster::JobSpec> {
    let uid = effective_uid();
    if let Some(file) = args.file {
        if !args.argv.is_empty() || args.project.is_some() || args.jain_template.is_some() {
            return Err(anyhow!(
                "a TOML file cannot be combined with inline workload flags"
            ));
        }
        return Ok(parse_scq_toml_v0(&read_input(&file).await?, uid, now())?);
    }
    let project = args
        .project
        .ok_or_else(|| anyhow!("inline submit requires --project"))?;
    if let Some(template) = args.jain_template {
        let template = parse_jain_template(&template)?;
        let worker_arguments = args.argv;
        let spec = ScqTomlSpec {
            job: ScqTomlJob {
                id: args.id,
                project,
                tier: args.tier,
                command: "jain-worker".to_owned(),
                args: Vec::new(),
                workdir: None,
                venv: None,
            },
            resource: ScqTomlResource {
                gpus_per_task: args.gpus,
                gpu_type: args.gpu_type,
                cpus_reserved: args.cpus,
                omp_threads: args.omp,
                dataloader_workers: 0,
                ram_gb: args.ram,
                gang_size: 1,
                placement: "single_node".into(),
                node: None,
            },
            env: Default::default(),
            preempt: ScqTomlPreemption {
                preemptible: !args.no_preempt,
                ..ScqTomlPreemption::default()
            },
            limits: ScqTomlLimits::default(),
            dependencies: ScqTomlDependencies::default(),
        };
        let mut job = spec.into_job_spec(uid, now())?;
        job.workload = Workload::Jain {
            template,
            arguments: worker_arguments,
            environment: Environment::default(),
        };
        return Ok(job);
    }
    let mut argv = args.argv.into_iter();
    let command = argv
        .next()
        .ok_or_else(|| anyhow!("inline submit requires `-- CMD ARGS`"))?;
    let spec = ScqTomlSpec {
        job: ScqTomlJob {
            id: args.id,
            project,
            tier: args.tier,
            command,
            args: argv.collect(),
            workdir: args.workdir,
            venv: args.venv,
        },
        resource: ScqTomlResource {
            gpus_per_task: args.gpus,
            gpu_type: args.gpu_type,
            cpus_reserved: args.cpus,
            omp_threads: args.omp,
            dataloader_workers: 0,
            ram_gb: args.ram,
            gang_size: 1,
            placement: "single_node".into(),
            node: None,
        },
        env: Default::default(),
        preempt: ScqTomlPreemption {
            preemptible: !args.no_preempt,
            ..ScqTomlPreemption::default()
        },
        limits: ScqTomlLimits::default(),
        dependencies: ScqTomlDependencies::default(),
    };
    Ok(spec.into_job_spec(uid, now())?)
}

fn parse_jain_template(value: &str) -> Result<JainWorkload> {
    match value.trim().to_ascii_lowercase().as_str() {
        "dataset-profile" => Ok(JainWorkload::DatasetProfile),
        "feature-expansion" => Ok(JainWorkload::FeatureExpansion),
        "training" => Ok(JainWorkload::Training),
        "report-generation" => Ok(JainWorkload::ReportGeneration),
        "prediction" => Ok(JainWorkload::Prediction),
        "export" => Ok(JainWorkload::Export),
        "chat" => Ok(JainWorkload::Chat),
        "research" => Ok(JainWorkload::Research),
        "managed-browser" => Ok(JainWorkload::ManagedBrowser),
        "jope-encode" => Ok(JainWorkload::JopeEncode),
        "jope-propose" => Ok(JainWorkload::JopePropose),
        "lime-run" => Ok(JainWorkload::LimeRun),
        "lime-batch" => Ok(JainWorkload::LimeBatch),
        other => Err(anyhow!("unknown Jain workload template {other:?}")),
    }
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

async fn stream_events(
    client: &Client,
    request: SubscribeRequest,
    follow: bool,
    json: bool,
) -> Result<()> {
    let mut events = client.subscribe(request).await?;
    loop {
        let next = if follow {
            events.recv().await
        } else {
            tokio::time::timeout(Duration::from_millis(200), events.recv())
                .await
                .ok()
                .flatten()
        };
        let Some(event) = next else { break };
        print_event(&event?, json)?;
    }
    Ok(())
}

async fn run_attach(
    client: &Client,
    job_id: JobId,
    offset: EventOffset,
    read_only: bool,
) -> Result<()> {
    let attachment = client
        .attach(AttachRequest {
            job_id,
            offset,
            read_only,
        })
        .await?;
    let mut events = attachment.events;
    if !read_only {
        let sender = attachment.sender;
        tokio::spawn(async move {
            let mut stdin = tokio::io::stdin();
            let mut buffer = vec![0_u8; 4096];
            loop {
                match stdin.read(&mut buffer).await {
                    Ok(0) => {
                        let _ = sender.close_stdin().await;
                        break;
                    }
                    Ok(length) => {
                        if sender.stdin(buffer[..length].to_vec()).await.is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        });
    }
    while let Some(event) = events.recv().await {
        print_event(&event?, false)?;
    }
    Ok(())
}

fn print_event(record: &jain_smartcluster::EventRecord, json: bool) -> Result<()> {
    if json {
        println!("{}", serde_json::to_string(record)?);
        return Ok(());
    }
    match &record.event {
        JobEvent::Stdout(bytes) | JobEvent::Pty(bytes) => {
            io::stdout().write_all(bytes)?;
            io::stdout().flush()?;
        }
        JobEvent::Stderr(bytes) => {
            io::stderr().write_all(bytes)?;
            io::stderr().flush()?;
        }
        other => println!("{} {} {:?}", record.offset, record.job_id, other),
    }
    Ok(())
}

fn print_job(job: &JobRecord, json: bool) -> Result<()> {
    if json {
        println!("{}", serde_json::to_string_pretty(job)?);
    } else {
        println!(
            "{}\t{:?}\t{}\t{}",
            job.id, job.state, job.spec.project, job.spec.tenant
        );
    }
    Ok(())
}

fn print_jobs(page: &JobPage, json: bool) -> Result<()> {
    if json {
        println!("{}", serde_json::to_string_pretty(page)?);
    } else {
        println!("ID\tSTATE\tPROJECT\tTENANT");
        for job in &page.jobs {
            print_job(job, false)?;
        }
        if let Some(cursor) = &page.next_cursor {
            println!("next cursor: {cursor}");
        }
    }
    Ok(())
}

fn print_status(status: &DaemonStatus, json: bool) -> Result<()> {
    if json {
        println!("{}", serde_json::to_string_pretty(status)?);
    } else {
        println!(
            "scqd {} protocol {} mode {:?} queued={} running={}",
            status.daemon_version,
            status.protocol_version,
            status.mode,
            status.queued_jobs,
            status.running_jobs
        );
        println!(
            "capacity cpu={}m memory={} scratch={} pressure={:?}/{:?}/{:?}",
            status.effective_capacity.cpu_millicores,
            status.effective_capacity.memory_bytes,
            status.effective_capacity.scratch_bytes,
            status.pressure.cpu,
            status.pressure.memory,
            status.pressure.io
        );
        for gpu in &status.gpus {
            println!(
                "{}\t{}\t{}\t{} / {} bytes\t{}",
                gpu.id,
                gpu.uuid,
                gpu.name,
                gpu.allocatable_vram_bytes,
                gpu.total_vram_bytes,
                if gpu.healthy { "healthy" } else { "unhealthy" }
            );
        }
    }
    Ok(())
}

struct TerminalGuard;

impl TerminalGuard {
    fn enter() -> Result<Self> {
        terminal::enable_raw_mode()?;
        execute!(io::stdout(), EnterAlternateScreen, cursor::Hide)?;
        Ok(Self)
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = execute!(io::stdout(), cursor::Show, LeaveAlternateScreen);
        let _ = terminal::disable_raw_mode();
    }
}

async fn run_tui(client: &Client) -> Result<()> {
    let _guard = TerminalGuard::enter()?;
    loop {
        let status = client.status().await?;
        let jobs = client.list(ListRequest::default()).await?;
        execute!(
            io::stdout(),
            cursor::MoveTo(0, 0),
            terminal::Clear(ClearType::All)
        )?;
        println!(
            "SCQ {:?} | queued {} | running {} | q quit",
            status.mode, status.queued_jobs, status.running_jobs
        );
        println!("ID                         STATE          PROJECT");
        for job in jobs.jobs.iter().take(30) {
            println!(
                "{:<26} {:<14} {}",
                job.id,
                format!("{:?}", job.state),
                job.spec.project
            );
        }
        io::stdout().flush()?;
        if event::poll(Duration::from_secs(1))?
            && matches!(event::read()?, TerminalEvent::Key(key) if key.code == KeyCode::Char('q'))
        {
            break;
        }
    }
    Ok(())
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
