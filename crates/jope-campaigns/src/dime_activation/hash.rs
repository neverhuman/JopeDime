use super::*;

pub(crate) fn sha256_file(path: &Path) -> Result<String> {
    crate::selfplay_runtime::sha256_file(path)
}
