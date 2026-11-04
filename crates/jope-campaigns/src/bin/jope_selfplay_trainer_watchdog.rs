#[path = "common/flock.rs"]
mod flock;
#[path = "jope_selfplay_trainer_watchdog/implementation.rs"]
mod implementation;

fn main() -> anyhow::Result<()> {
    implementation::run()
}
