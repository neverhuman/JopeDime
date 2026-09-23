//! Private child process of the qualified CPU controller; not a public launcher.
fn main() -> anyhow::Result<()> {
    use jope_policy::context_producer::worker::{self, WorkerInvocation, WORKER_VERSION};
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args == ["--version"] {
        println!("{WORKER_VERSION}");
        return Ok(());
    }
    worker::execute(&WorkerInvocation::parse(&args)?)
}
