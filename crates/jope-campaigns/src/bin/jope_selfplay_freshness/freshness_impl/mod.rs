use super::*;

mod mutation;
mod runtime;
mod state;
mod status;

#[cfg(test)]
#[path = "../tests.rs"]
mod tests;

pub(crate) fn run() -> Result<()> {
    let args = Args::parse();
    let audit_only =
        args.audit_only || std::env::var("JOPE_FRESHNESS_AUDIT_ONLY").as_deref() == Ok("1");
    let socket_path = args
        .socket
        .unwrap_or_else(|| args.root.join("ops/freshness.sock"));
    if args.once {
        let mut supervisor = Supervisor::open(args.root, args.scq)?;
        let status = supervisor.cycle(audit_only, args.auto_retire_stale)?;
        println!("{}", serde_json::to_string(&status)?);
        return Ok(());
    }
    runtime::serve(
        args.root,
        args.scq,
        socket_path,
        args.interval,
        audit_only,
        args.auto_retire_stale,
    )
}
