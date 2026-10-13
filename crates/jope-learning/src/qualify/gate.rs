//! The `JOPE_SEAL_GATE_ENABLED` switch.
//!
//! Only the exact value `1` turns the gate on. Unset, empty and `0` are off. Every other value,
//! including `true`, ` 1` and bytes that are not UTF-8, is [`Gate::Unrecognized`] and also off,
//! so a mistyped value can never open the gate.

use std::ffi::OsStr;
use std::fmt;

use serde::Serialize;

/// The environment variable that switches the seal and champion gates on.
pub const FLAG: &str = "JOPE_SEAL_GATE_ENABLED";

/// The state of [`FLAG`]. Callers read it once and pass it to each request, so no request
/// consults the process environment on its own.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(tag = "state", content = "value", rename_all = "snake_case")]
pub enum Gate {
    /// Unset, empty or `0`. The default.
    #[default]
    Off,
    /// Any value other than `0` or `1`, kept lossily for the refusal message. Off.
    Unrecognized(String),
    /// Exactly `1`.
    On,
}

impl Gate {
    /// The gate for a raw flag value; `None` is an unset variable.
    #[must_use]
    pub fn from_value(value: Option<&OsStr>) -> Self {
        match value {
            None => Self::Off,
            Some(raw) if raw.is_empty() || raw == "0" => Self::Off,
            Some(raw) if raw == "1" => Self::On,
            Some(raw) => Self::Unrecognized(raw.to_string_lossy().into_owned()),
        }
    }

    /// The gate as the process environment sets it now.
    #[must_use]
    pub fn from_env() -> Self {
        Self::from_value(std::env::var_os(FLAG).as_deref())
    }

    /// Whether the gate is on. Only [`Gate::On`] is.
    #[must_use]
    pub fn is_on(&self) -> bool {
        matches!(self, Self::On)
    }
}

impl fmt::Display for Gate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Off => f.write_str("off"),
            Self::Unrecognized(raw) => write!(f, "{raw:?}, which is not 1 and counts as off"),
            Self::On => f.write_str("on"),
        }
    }
}
