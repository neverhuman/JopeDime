use super::*;

pub(super) fn prepare_parent(parent: &Path) -> Result<(), ServerStartupError> {
    if !parent.exists() {
        fs::create_dir_all(parent).map_err(|source| ServerStartupError::Parent {
            path: parent.to_path_buf(),
            source,
        })?;
        fs::set_permissions(parent, fs::Permissions::from_mode(0o750)).map_err(|source| {
            ServerStartupError::Permissions {
                path: parent.to_path_buf(),
                source,
            }
        })?;
    }
    let metadata = fs::metadata(parent).map_err(|source| ServerStartupError::Parent {
        path: parent.to_path_buf(),
        source,
    })?;
    if !metadata.is_dir() || metadata.mode() & 0o002 != 0 {
        return Err(ServerStartupError::InsecureParent(parent.to_path_buf()));
    }
    Ok(())
}

pub(super) fn remove_abandoned_socket(path: &Path) -> Result<(), ServerStartupError> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(source) => {
            return Err(ServerStartupError::Inspect {
                path: path.to_path_buf(),
                source,
            });
        }
    };
    if !metadata.file_type().is_socket() {
        return Err(ServerStartupError::NonSocket(path.to_path_buf()));
    }
    match std::os::unix::net::UnixStream::connect(path) {
        Ok(_) => return Err(ServerStartupError::AlreadyRunning(path.to_path_buf())),
        Err(error)
            if matches!(
                error.kind(),
                std::io::ErrorKind::ConnectionRefused | std::io::ErrorKind::NotFound
            ) => {}
        Err(source) => {
            return Err(ServerStartupError::Inspect {
                path: path.to_path_buf(),
                source,
            });
        }
    }
    fs::remove_file(path).map_err(|source| ServerStartupError::Remove {
        path: path.to_path_buf(),
        source,
    })
}
