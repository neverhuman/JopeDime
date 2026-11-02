use super::*;

pub(super) async fn stream_events(
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

pub(super) async fn run_attach(
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

pub(super) async fn run_tui(client: &Client) -> Result<()> {
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
