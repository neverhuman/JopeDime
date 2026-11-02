//! Small Python built-in semantics the verifiers depend on: `str.strip`, hashing equality for set
//! membership, `in`, iteration, `Path.read_text`, `PurePosixPath.parts`, `os.path.realpath` and
//! `Path.resolve()`, the `re` classes of a `str` pattern, subprocess output read with `text=True`,
//! and the text of the `OSError` and `CalledProcessError` a tool printed.
//!
//! Numbers are the exception: text is read as a number by one strict grammar ([`decimal`],
//! [`int`], [`decimal_int`], [`float`]) that accepts a subset of what Python's `int()` and
//! `float()` accepted, never more (see [`decimal`]).

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::pyjson::{self, Value};
use crate::{Error, Result};

mod numbers;
pub use numbers::*;

mod values;
pub use values::*;

mod io;
pub use io::*;

mod utility;
pub use utility::*;

mod path;
pub use path::*;

mod float;
pub use float::*;

#[cfg(test)]
#[path = "py/tests.rs"]
mod tests;
