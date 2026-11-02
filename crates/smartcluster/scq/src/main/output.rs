use super::*;

pub(crate) fn print_event(record: &jain_smartcluster::EventRecord, json: bool) -> Result<()> {
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

pub(crate) fn print_job(job: &JobRecord, json: bool) -> Result<()> {
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

pub(crate) fn print_jobs(page: &JobPage, json: bool) -> Result<()> {
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

pub(crate) fn print_status(status: &DaemonStatus, json: bool) -> Result<()> {
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
