//! Join a home-relative path only when `HOME` is absolute.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

/// Join `relative` onto `home`.
///
/// A relative or empty `home` is not a home directory. `None` means `HOME` is unset.
pub(crate) fn absolute_home_path(home: Option<&OsStr>, relative: &[&str]) -> Option<PathBuf> {
    let home = home?;
    if !Path::new(home).is_absolute() {
        return None;
    }
    let mut path = PathBuf::from(home);
    for part in relative {
        path.push(part);
    }
    Some(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsStr;

    #[test]
    fn a_relative_home_does_not_name_a_carbon_tool() {
        assert_eq!(
            absolute_home_path(Some(OsStr::new("/users/alice")), &[".local", "bin", "aws"]),
            Some(PathBuf::from("/users/alice/.local/bin/aws"))
        );
        assert_eq!(
            absolute_home_path(Some(OsStr::new("relative/home")), &[".local", "bin", "aws"]),
            None
        );
        assert_eq!(
            absolute_home_path(Some(OsStr::new("")), &[".local", "zig", "zig"]),
            None
        );
        assert_eq!(absolute_home_path(None, &[".local", "bin", "aws"]), None);
    }
}
