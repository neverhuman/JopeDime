//! `jope-selfplay`: the flywheel control plane as one binary (argv[0]
//! dispatch; see `jope_campaigns::dispatch`).

fn main() -> anyhow::Result<()> {
    jope_selfplay::TABLE.main()
}
