//! Durable eligibility controller; native Graph/SmartCluster retain dispatch authority.
//!
//! The command line is [`jope_parallel::native_loop::cli`]. This production entrypoint
//! observes the real host: it passes [`HostSources::system`] (`/etc/machine-id`, the kernel boot
//! id and host name, `getifaddrs` and `nvidia-smi`) and nothing else, so `census collect` and
//! `participant` always observe this host, and no argument or environment variable substitutes
//! an identity source.
use jope_parallel::native_loop::{bounded::collect::HostSources, cli};

fn main() -> anyhow::Result<()> {
    cli::main(&HostSources::system())
}
