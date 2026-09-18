//! The one place feat-math knows how to invoke a C++ compiler (ADR-0002 §4; plan blocker B28).
//!
//! Every compiler call site goes through here — the fitted model images
//! (`advanced_v3::fitted_cpp_runtime`), the emitted-C++ gates in `transpile`, and the integration
//! tests — so a single environment contract applies everywhere (`build.rs`, which cannot use the
//! crate, mirrors the same three variables for the embedded fitted runtime):
//!
//! * `CXX` — the compiler binary (default `g++`).
//! * `JOPE_CPP_STD` — the default language standard, `17` or `20` (`c++17`/`c++20` accepted).
//!   Sites that deliberately pin a standard (the C++17 / C++20 parity matrix) ignore it.
//! * `JOPE_CXX_MARCH` — an explicit `-march=` value. Emitted programs get no `-march` unless it is
//!   set; the fitted runtime and model images default to the portable `x86-64-v2` baseline. With
//!   `-ffp-contract=off -fno-fast-math` the ISA only changes speed, never the produced bits (the CI
//!   `-march` matrix proves it).
//!
//! The fitted codec identity hashes the flag string (`runtime_flags_sha256`), so a codec encoded
//! under one environment decodes only under the same one; historical node codecs were produced
//! with `-std=c++17 -O3 -march=znver2 -ffp-contract=off -fno-fast-math` — set
//! `JOPE_CXX_MARCH=znver2` to reproduce that identity.

use std::process::Command;

/// The language standard for a compiler invocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CxxStd {
    Cxx17,
    Cxx20,
}

impl CxxStd {
    /// The `-std=` flag for this standard.
    pub fn flag(self) -> &'static str {
        match self {
            Self::Cxx17 => "-std=c++17",
            Self::Cxx20 => "-std=c++20",
        }
    }

    /// Parses `17`, `20`, `c++17`, `c++20` (case-insensitive, surrounding whitespace ignored).
    pub fn parse(value: &str) -> Option<Self> {
        let value = value.trim().to_ascii_lowercase();
        match value.trim_start_matches("c++").trim_start_matches("gnu++") {
            "17" => Some(Self::Cxx17),
            "20" => Some(Self::Cxx20),
            _ => None,
        }
    }
}

/// The portable ISA baseline used for the fitted runtime and model images when `JOPE_CXX_MARCH`
/// is unset (the historical value on the node was `znver2`).
pub const FITTED_DEFAULT_MARCH: &str = "x86-64-v2";

fn env_non_empty(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

/// The C++ compiler binary: `CXX`, or `g++`.
pub fn compiler() -> String {
    env_non_empty("CXX").unwrap_or_else(|| "g++".to_owned())
}

/// The default standard: `JOPE_CPP_STD`, or C++17. An unparseable value is an error, never a
/// silent fallback.
pub fn default_standard() -> Result<CxxStd, String> {
    standard_from(env_non_empty("JOPE_CPP_STD").as_deref())
}

/// `default_standard` over an explicit `JOPE_CPP_STD` value.
pub fn standard_from(value: Option<&str>) -> Result<CxxStd, String> {
    match value {
        None => Ok(CxxStd::Cxx17),
        Some(value) => CxxStd::parse(value)
            .ok_or_else(|| format!("JOPE_CPP_STD must be 17 or 20, got {value:?}")),
    }
}

/// The `-march=` flag: `JOPE_CXX_MARCH` when set, otherwise `default` (if any).
pub fn march_flag(default: Option<&str>) -> Option<String> {
    march_from(env_non_empty("JOPE_CXX_MARCH").as_deref(), default)
}

/// `march_flag` over an explicit `JOPE_CXX_MARCH` value.
pub fn march_from(value: Option<&str>, default: Option<&str>) -> Option<String> {
    value.or(default).map(|value| format!("-march={value}"))
}

/// The flags every fitted model image is compiled with (hashed into the codec identity).
pub fn fitted_flags() -> Result<Vec<String>, String> {
    let mut flags = vec![default_standard()?.flag().to_owned(), "-O3".to_owned()];
    flags.extend(march_flag(Some(FITTED_DEFAULT_MARCH)));
    flags.extend(["-ffp-contract=off", "-fno-fast-math"].map(str::to_owned));
    Ok(flags)
}

/// A compiler command for an emitted program: `CXX`, the requested standard (or the environment
/// default), and `-march` only when `JOPE_CXX_MARCH` is set. Callers add their own optimisation,
/// warning and sanitizer flags.
pub fn command(standard: Option<CxxStd>) -> Command {
    let standard = match standard {
        Some(standard) => standard,
        None => default_standard().unwrap_or_else(|error| panic!("{error}")),
    };
    let mut command = Command::new(compiler());
    command.arg(standard.flag());
    if let Some(march) = march_flag(None) {
        command.arg(march);
    }
    command
}

/// The first line of `CXX --version`.
pub fn version() -> std::io::Result<String> {
    let output = Command::new(compiler()).arg("--version").output()?;
    if !output.status.success() {
        return Err(std::io::Error::other(format!(
            "{} --version failed: {}",
            compiler(),
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()
        .unwrap_or_default()
        .to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standards_parse_and_flag() {
        assert_eq!(CxxStd::parse("17"), Some(CxxStd::Cxx17));
        assert_eq!(CxxStd::parse(" c++20 "), Some(CxxStd::Cxx20));
        assert_eq!(CxxStd::parse("C++17"), Some(CxxStd::Cxx17));
        assert_eq!(CxxStd::parse("23"), None);
        assert_eq!(CxxStd::Cxx20.flag(), "-std=c++20");
    }

    #[test]
    fn environment_values_resolve_or_fail_closed() {
        assert_eq!(standard_from(None), Ok(CxxStd::Cxx17));
        assert_eq!(standard_from(Some("20")), Ok(CxxStd::Cxx20));
        assert!(standard_from(Some("23"))
            .unwrap_err()
            .contains("JOPE_CPP_STD"));
        assert_eq!(march_from(None, None), None);
        assert_eq!(
            march_from(None, Some("x86-64-v2")).as_deref(),
            Some("-march=x86-64-v2")
        );
        assert_eq!(
            march_from(Some("znver2"), Some("x86-64-v2")).as_deref(),
            Some("-march=znver2")
        );
    }

    #[test]
    fn default_fitted_flags_are_the_portable_baseline() {
        // Only meaningful when the environment does not override; CI runs the matrix explicitly.
        if std::env::var_os("JOPE_CPP_STD").is_none()
            && std::env::var_os("JOPE_CXX_MARCH").is_none()
        {
            assert_eq!(
                fitted_flags().unwrap(),
                [
                    "-std=c++17",
                    "-O3",
                    "-march=x86-64-v2",
                    "-ffp-contract=off",
                    "-fno-fast-math"
                ]
            );
        }
    }
}
