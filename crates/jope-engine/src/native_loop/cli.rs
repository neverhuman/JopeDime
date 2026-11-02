//! The `jope-native-loop` command line: its parsing, dispatch and admission code.
//!
//! Durable eligibility controller; native Graph/SmartCluster retain dispatch authority.
//!
//! `bind`, `bind-parent-decision`, `tick` and `serve` keep the fixed `jope.native_loop.v1`
//! contract and require the loop arguments. `plan`, `status`, `verify` and `cancel` operate
//! on the versioned `jope.native_bounded_run.v1` contract (W7-T02) and take none of them:
//! status and verify are read-only and never start numerical work. `census collect` observes
//! only the local host and `census merge` combines pinned per-host documents (W7-T01); neither
//! allocates or reserves anything. `coordinate` drives a planned run through SmartCluster
//! (directly, or through each host's `agent` over the pinned transport) and `participant` is
//! the rank worker SmartCluster runs (W7-T05/T07); none of them uses SSH. `contract author`,
//! `grant sign` and `grant verify` author a contract and sign and check its grants (W7-D6);
//! the grant-authority key is only ever read from an owner-only file.
//!
//! [`main`] takes the host's observation provider from its caller and nothing else: `census
//! collect` and `participant` observe the host only through it. The production binary
//! (`src/bin/jope-native-loop.rs`) passes [`HostSources::system`]; no argument or environment
//! variable changes the sources. The test-only fixture executable
//! (`tests/support/native_loop_fixture.rs`, the example target `jope-native-loop-fixture`)
//! runs this same code with the checked-in fixture host. Cargo's default test target selection
//! compiles it; the DEFAULT `cargo build` and `cargo install` do not produce it, and an explicit
//! `cargo build --example`, `--examples` or `--all-targets`, or `cargo install --example`, can.
//! Production stays on the real host because its entrypoints hardwire [`HostSources::system`]
//! and the product target is the `jope-native-loop` binary, not because Cargo cannot build the
//! fixture.
use super::{bounded, bounded::collect::HostSources, Binding, Loop};
use crate::native_graph::{Artifact, Graph};
use anyhow::{bail, ensure, Context, Result};
use clap::{Args as ClapArgs, Parser, Subcommand};
use jain_smartcluster::{Client, ProjectId, TenantId};
use std::{fs, io::Read, os::unix::fs::OpenOptionsExt, path::PathBuf, time::Duration};
mod commands;
mod types;

use commands::bounded_command;
use types::{Args, Command};

/// Run `jope-native-loop` on this process's arguments, observing the host through `sources`.
pub fn main(sources: &HostSources) -> Result<()> {
    let args = Args::parse();
    if matches!(
        args.command,
        Command::Plan { .. }
            | Command::Preflight { .. }
            | Command::Status { .. }
            | Command::Verify { .. }
            | Command::Cancel { .. }
            | Command::Census { .. }
            | Command::Contract { .. }
            | Command::Grant { .. }
            | Command::Participant { .. }
            | Command::Coordinate { .. }
            | Command::Agent { .. }
    ) {
        ensure!(
            !args.legacy.any(),
            "bounded-run operations take no fixed-loop arguments; the frozen plan is the only input"
        );
        if let Command::Participant {
            plan,
            plan_sha256,
            launch_record,
            outbox,
            coordinator,
            participant_key,
            installation,
        } = args.command
        {
            let args = bounded::participant::ParticipantArgs {
                plan,
                plan_sha256,
                launch_record,
                outbox,
                coordinator,
                participant_key,
                installation,
                job_arguments: std::env::args_os().skip(1).collect(),
            };
            match bounded::participant::run(&args, sources) {
                Ok(code) => std::process::exit(code),
                Err(e) => {
                    // The rank's supervisor keeps its outbox; record why nothing started.
                    bounded::participant::record_refusal(&args.outbox, &e);
                    return Err(e);
                }
            }
        }
        return bounded_command(args.command, sources);
    }
    let a = args.legacy.complete().context(
        "bind/bind-parent-decision/tick/serve require --root, --state, --graph-state, --worker-state, --config, --config-sha256, --socket, --tenant and --project",
    )?;
    let controller = Loop::open(
        &a.root,
        &a.state,
        &a.graph_state,
        &Artifact {
            path: a.config,
            sha256: a.config_sha256,
        },
    )?;
    let graph = Graph::open(
        &a.root,
        &a.graph_state,
        &a.worker_state,
        TenantId::new(a.tenant)?,
        ProjectId::new(a.project)?,
    )?;
    let client = Client::new(a.socket);
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?
        .block_on(async {
            match args.command {
                Command::Bind { binding } => {
                    let f = fs::OpenOptions::new()
                        .read(true)
                        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
                        .open(binding)?;
                    ensure!(f.metadata()?.is_file(), "regular binding required");
                    let mut b = Vec::new();
                    f.take(1024 * 1024 + 1).read_to_end(&mut b)?;
                    ensure!(b.len() <= 1024 * 1024, "binding size");
                    controller.bind(&serde_json::from_slice::<Binding>(&b)?)?;
                }
                Command::BindParentDecision { decision, sha256 } => controller
                    .bind_parent_decision(&Artifact {
                        path: decision,
                        sha256,
                    })?,
                Command::Tick => println!(
                    "{}",
                    serde_json::to_string(&controller.tick(&graph, &client).await?)?
                ),
                Command::Serve => loop {
                    let progress = controller.tick(&graph, &client).await?;
                    println!("{}", serde_json::to_string(&progress)?);
                    if progress.complete {
                        break;
                    }
                    tokio::time::sleep(Duration::from_secs(1)).await;
                },
                _ => bail!("not a fixed-loop operation"),
            };
            Ok(())
        })
}
