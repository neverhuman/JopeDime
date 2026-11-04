use super::*;

fn digest(value: &str) -> Sha256Digest {
    Sha256Digest::of_bytes(value.as_bytes())
}

mod dataset;
mod panels;
mod promotion;
mod proposal;
