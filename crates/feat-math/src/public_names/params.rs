//! Plain-language names and purposes for tunable model parameters.

use super::PublicName;

mod catalog;
#[cfg(test)]
mod keys;
#[cfg(test)]
mod tests;

pub use catalog::PARAMETERS;
