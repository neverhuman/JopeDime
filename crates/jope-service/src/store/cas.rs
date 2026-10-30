use super::{boundary, BlobRef, Result, StoreError};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

static STAGING_ID: AtomicU64 = AtomicU64::new(0);

pub(super) fn sync_dir(path: &Path) -> Result<()> {
    #[cfg(windows)]
    let file = {
        use std::os::windows::fs::OpenOptionsExt;
        OpenOptions::new()
            .read(true)
            .custom_flags(0x02000000)
            .open(path)?
    };
    #[cfg(not(windows))]
    let file = File::open(path)?;
    file.sync_all()?;
    Ok(())
}

pub(super) fn directory(path: &Path) -> Result<()> {
    if !path.exists() {
        let mut builder = fs::DirBuilder::new();
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        match builder.create(path) {
            Ok(()) => {
                sync_dir(path)?;
                if let Some(parent) = path.parent() {
                    sync_dir(parent)?;
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error.into()),
        }
    }
    if !fs::symlink_metadata(path)?.file_type().is_dir() {
        return Err(StoreError::Integrity(
            "store directory is not a real directory",
        ));
    }
    Ok(())
}

fn hash_file(path: &Path) -> Result<BlobRef> {
    if !fs::symlink_metadata(path)?.file_type().is_file() {
        return Err(StoreError::Integrity("blob is not a regular file"));
    }
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut bytes = 0_u64;
    let mut buffer = [0_u8; 8192];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
        bytes = bytes
            .checked_add(count as u64)
            .ok_or(StoreError::Invalid("blob too large"))?;
    }
    Ok(BlobRef {
        sha256: hash.finalize().into(),
        bytes,
    })
}

pub(super) struct Cas {
    pub home: PathBuf,
}

impl Cas {
    pub fn layout(&self) -> Result<()> {
        directory(&self.home.join("tmp"))?;
        directory(&self.home.join("cas"))?;
        directory(&self.home.join("cas/sha256"))
    }

    pub fn path(&self, reference: &BlobRef) -> PathBuf {
        let name: String = reference
            .sha256
            .iter()
            .map(|v| format!("{v:02x}"))
            .collect();
        self.home
            .join("cas/sha256")
            .join(&name[..2])
            .join(&name[2..4])
            .join(&name[4..])
    }

    pub fn verify(&self, reference: &BlobRef) -> Result<()> {
        let path = self.path(reference);
        for parent in [
            self.home.join("cas"),
            self.home.join("cas/sha256"),
            path.parent().unwrap().parent().unwrap().to_path_buf(),
            path.parent().unwrap().to_path_buf(),
        ] {
            if !fs::symlink_metadata(parent)?.file_type().is_dir() {
                return Err(StoreError::Integrity("CAS parent is not a real directory"));
            }
        }
        if hash_file(&path)? != *reference {
            return Err(StoreError::Integrity("blob hash/size mismatch"));
        }
        Ok(())
    }

    pub fn publish(&self, mut input: impl Read) -> Result<BlobRef> {
        let (temporary, mut file) = loop {
            let id = STAGING_ID.fetch_add(1, Ordering::Relaxed);
            let path = self
                .home
                .join("tmp")
                .join(format!("blob-{}-{id}.part", std::process::id()));
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(file) => break (path, file),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error.into()),
            }
        };
        let mut hash = Sha256::new();
        let mut bytes = 0_u64;
        let mut buffer = [0_u8; 8192];
        loop {
            let count = input.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            file.write_all(&buffer[..count])?;
            hash.update(&buffer[..count]);
            bytes = bytes
                .checked_add(count as u64)
                .ok_or(StoreError::Invalid("blob too large"))?;
        }
        boundary("after_write");
        file.sync_all()?;
        boundary("after_file_sync");
        let reference = BlobRef {
            sha256: hash.finalize().into(),
            bytes,
        };
        drop(file);
        if hash_file(&temporary)? != reference {
            return Err(StoreError::Integrity("staged blob hash/size mismatch"));
        }
        boundary("after_hash_verify");
        let path = self.path(&reference);
        let parent = path.parent().unwrap();
        directory(parent.parent().unwrap())?;
        directory(parent)?;
        if path.symlink_metadata().is_ok() {
            self.verify(&reference)?;
            fs::remove_file(&temporary)?;
        } else {
            fs::rename(&temporary, &path)?;
        }
        boundary("after_rename");
        sync_dir(parent)?;
        boundary("after_directory_sync");
        Ok(reference)
    }

    pub fn clear_staging(&self) -> Result<()> {
        let path = self.home.join("tmp");
        if !fs::symlink_metadata(&path)?.file_type().is_dir() {
            return Err(StoreError::Integrity("tmp is not a real directory"));
        }
        for item in fs::read_dir(&path)? {
            let item = item?;
            if item.file_type()?.is_dir() {
                fs::remove_dir_all(item.path())?;
            } else {
                fs::remove_file(item.path())?;
            }
        }
        sync_dir(&path)
    }
}
