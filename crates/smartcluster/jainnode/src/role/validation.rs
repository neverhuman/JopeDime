use super::*;

pub(super) fn validate_envelope_file(path: &Path, file: &File) -> Result<(), RoleError> {
    let opened = file.metadata().map_err(RoleError::Io)?;
    let named = fs::symlink_metadata(path).map_err(RoleError::Io)?;
    if !opened.is_file()
        || named.file_type().is_symlink()
        || opened.dev() != named.dev()
        || opened.ino() != named.ino()
        || opened.uid() != Uid::effective().as_raw()
        || opened.nlink() != 1
        || opened.mode() & 0o777 != 0o600
        || opened.len() == 0
        || opened.len() > MAX_ROLE_ENVELOPE_BYTES
        || path.canonicalize().map_err(RoleError::Io)? != path
    {
        return Err(RoleError::InvalidEnvelope);
    }
    validate_physical_components(path)
}

pub(super) fn read_signing_key(path: &Path) -> Result<SigningKey, RoleError> {
    validate_absolute_path(path)?;
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(nix::libc::O_NOFOLLOW | nix::libc::O_CLOEXEC)
        .open(path)
        .map_err(RoleError::Io)?;
    let opened = file.metadata().map_err(RoleError::Io)?;
    let named = fs::symlink_metadata(path).map_err(RoleError::Io)?;
    if !opened.is_file()
        || named.file_type().is_symlink()
        || opened.dev() != named.dev()
        || opened.ino() != named.ino()
        || opened.uid() != Uid::effective().as_raw()
        || opened.nlink() != 1
        || opened.mode() & 0o777 != 0o600
        || opened.len() != 32
        || path.canonicalize().map_err(RoleError::Io)? != path
    {
        return Err(RoleError::InvalidEnvelope);
    }
    validate_physical_components(path)?;
    let mut secret = [0_u8; 32];
    file.read_exact(&mut secret).map_err(RoleError::Io)?;
    let signing_key = SigningKey::from_bytes(&secret);
    secret.fill(0);
    Ok(signing_key)
}

pub(super) fn validate_state_root(path: &Path) -> Result<(), RoleError> {
    validate_absolute_path(path)?;
    let metadata = fs::symlink_metadata(path).map_err(RoleError::Io)?;
    if !metadata.is_dir()
        || metadata.file_type().is_symlink()
        || metadata.uid() != Uid::effective().as_raw()
        || metadata.mode() & 0o777 != 0o700
        || path.canonicalize().map_err(RoleError::Io)? != path
    {
        return Err(RoleError::InvalidEnvelope);
    }
    validate_physical_components(path)
}

pub(super) fn validate_bound_socket(path: &Path) -> Result<(), RoleError> {
    validate_absolute_path(path)?;
    let metadata = fs::symlink_metadata(path).map_err(RoleError::Io)?;
    let owner = metadata.uid();
    if !metadata.file_type().is_socket()
        || (owner != 0 && owner != Uid::effective().as_raw())
        || metadata.mode() & 0o002 != 0
        || path.canonicalize().map_err(RoleError::Io)? != path
    {
        return Err(RoleError::InvalidEnvelope);
    }
    validate_physical_components(path)
}

pub(super) fn validate_physical_components(path: &Path) -> Result<(), RoleError> {
    let mut current = PathBuf::from("/");
    for component in path.components().skip(1) {
        current.push(component.as_os_str());
        if fs::symlink_metadata(&current)
            .map_err(RoleError::Io)?
            .file_type()
            .is_symlink()
        {
            return Err(RoleError::InvalidEnvelope);
        }
    }
    Ok(())
}

pub(super) fn validate_absolute_path(path: &Path) -> Result<(), RoleError> {
    if !valid_config_path(path) {
        return Err(RoleError::InvalidEnvelope);
    }
    Ok(())
}

pub(super) fn valid_config_path(path: &Path) -> bool {
    path.is_absolute()
        && path
            .components()
            .all(|part| !matches!(part, Component::CurDir | Component::ParentDir))
}

pub(super) fn valid_image_digest(value: &str) -> bool {
    value.len() == 71
        && value.starts_with("sha256:")
        && value[7..].bytes().any(|byte| byte != b'0')
        && value[7..]
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

pub(super) fn parse_digest(value: &str) -> Result<Digest32, RoleError> {
    parse_hex(value)
}

pub(super) fn parse_hex<const N: usize>(value: &str) -> Result<[u8; N], RoleError> {
    if value.len() != N * 2
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return Err(RoleError::InvalidEnvelope);
    }
    let mut digest = [0; N];
    for (index, chunk) in value.as_bytes().chunks_exact(2).enumerate() {
        let text = std::str::from_utf8(chunk).map_err(|_| RoleError::InvalidEnvelope)?;
        digest[index] = u8::from_str_radix(text, 16).map_err(|_| RoleError::InvalidEnvelope)?;
    }
    Ok(digest)
}
