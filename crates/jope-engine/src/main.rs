//! `jope-parallel`: the search/eval engine binary. The command line lives in
//! `jope_parallel::cli` so the `jope` umbrella can carry it unchanged.

fn main() -> anyhow::Result<()> {
    jope_parallel::cli::main()
}
