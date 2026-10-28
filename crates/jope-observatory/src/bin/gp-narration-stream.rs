//! Inspect, follow, replay and serve retained evidence-backed search commentary.
use anyhow::{ensure, Result};
use clap::{Parser, Subcommand};
use jope_narrator::Cursor;
use jope_observatory::narration::{self, Index, Limits, MAX_PAGE_RECORDS};
use std::{
    io::{self, Write},
    net::SocketAddr,
    path::PathBuf,
    time::Duration,
};

#[derive(Parser)]
#[command(about = "Read-only search commentary derived from program edits and measurements")]
struct Args {
    /// One producer journal. No network request can change this path.
    #[arg(long)]
    journal: PathBuf,
    /// Expected run identity, also required when the journal is initially empty.
    #[arg(long)]
    run_id: String,
    #[arg(long, default_value_t = 1_000_000)]
    max_records: usize,
    #[arg(long, default_value_t = 1_073_741_824)]
    max_journal_bytes: u64,
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    /// Validate the complete retained prefix and print its boundary/status.
    Verify {
        /// Also require loss-free capture closure, separately from prefix integrity.
        #[arg(long)]
        require_complete: bool,
    },
    /// Print a finite page as JSONL (or terminal-safe text).
    Replay {
        #[arg(long, default_value = "0")]
        after: String,
        #[arg(long, default_value_t = 64)]
        limit: usize,
        #[arg(long)]
        text: bool,
    },
    /// Follow appended records; Ctrl-C stops without changing the producer.
    Follow {
        #[arg(long, default_value = "0")]
        after: String,
        #[arg(long)]
        text: bool,
    },
    /// Serve resumable WebSocket, finite HTTP pages and actual journal status.
    Serve {
        #[arg(long, default_value = "127.0.0.1:9877")]
        listen: SocketAddr,
        #[arg(long, default_value_t = 32)]
        max_clients: usize,
    },
    /// Read-only finite MCP tools over newline-delimited stdio JSON-RPC.
    Mcp,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let mut index = Index::open(
        &args.journal,
        &args.run_id,
        Limits {
            max_records: args.max_records,
            max_journal_bytes: args.max_journal_bytes,
        },
    )?;
    match args.command {
        Command::Verify { require_complete } => {
            let status = index.status()?;
            ensure!(
                !status.partial_tail,
                "journal has uncommitted trailing bytes; capture status is {}",
                status.capture.status
            );
            serde_json::to_writer(io::stdout().lock(), &status)?;
            println!();
            ensure!(
                !require_complete || status.capture.status == "complete",
                "capture is {}; complete capture required",
                status.capture.status
            );
        }
        Command::Replay { after, limit, text } => {
            narration::sequence(&after)?;
            let cursor = Cursor {
                run_id: args.run_id,
                sequence: after,
            };
            let page =
                narration::print_page(&mut index, &cursor, limit, text, &mut io::stdout().lock())?;
            eprintln!(
                "next={} more={} partial_tail={}",
                page.next.sequence, page.more, page.partial_tail
            );
        }
        Command::Follow { after, text } => {
            narration::sequence(&after)?;
            let mut cursor = Cursor {
                run_id: args.run_id,
                sequence: after,
            };
            let mut stop = Box::pin(tokio::signal::ctrl_c());
            loop {
                let page = narration::print_page(
                    &mut index,
                    &cursor,
                    MAX_PAGE_RECORDS,
                    text,
                    &mut io::stdout().lock(),
                )?;
                cursor = page.next;
                if !page.more {
                    let status = index.status()?;
                    if status.capture.status != "capturing"
                        && cursor.sequence == status.last_sequence
                    {
                        ensure!(
                            status.capture.status == "complete",
                            "capture closed with status {}",
                            status.capture.status
                        );
                        break;
                    }
                }
                tokio::select! {
                    _ = &mut stop => break,
                    _ = tokio::time::sleep(if page.more {Duration::ZERO} else {Duration::from_millis(100)}) => {},
                }
            }
        }
        Command::Serve {
            listen,
            max_clients,
        } => {
            narration::validate_listen(listen)?;
            narration::serve(index, listen, max_clients).await?;
        }
        Command::Mcp => {
            narration::mcp(
                &mut index,
                &mut io::stdin().lock(),
                &mut io::stdout().lock(),
            )?;
        }
    }
    io::stdout().flush()?;
    Ok(())
}
