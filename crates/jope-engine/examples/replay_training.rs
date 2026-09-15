//! Direct ranks for an explicitly leased, frozen replay experiment.
#[cfg(feature = "neural")]
fn main() -> anyhow::Result<()> {
    use anyhow::{ensure, Context};
    use std::{env, path::Path};
    let a: Vec<String> = env::args().collect();
    ensure!(
        a.len() == 10,
        "usage: replay_training ROOT CORPUS RUN LAUNCH RANK STEPS CHECKPOINT POLICY RESUME"
    );
    let resume = a[9].parse::<bool>().context("resume")?;
    jope_parallel::training::rank_replay(
        Path::new(&a[1]),
        Path::new(&a[2]),
        Path::new(&a[3]),
        Path::new(&a[4]),
        a[5].parse()?,
        8,
        a[6].parse()?,
        resume,
        if resume { None } else { Some(Path::new(&a[7])) },
        Path::new(&a[8]),
    )
}
#[cfg(not(feature = "neural"))]
fn main() {
    panic!("replay training requires the neural feature");
}
