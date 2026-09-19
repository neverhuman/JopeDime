//! `jope-campaigns`: the benchmark campaigns as one binary (argv[0] dispatch;
//! see `crate::dispatch`).

fn main() -> anyhow::Result<()> {
    jope_campaigns::TABLE.main()
}
