#[path = "common/flock.rs"]
mod flock;
mod implementation;

fn main() -> anyhow::Result<()> {
    implementation::run()
}
