use super::*;
use crate::blobs::same_file;
use crate::hash::digest_file;

/// Re-hash every blob (or check sizes with `quick`) and check every `by-id/` link.
pub fn verify_cache(cache: &Cache, quick: bool) -> VerifyReport {
    let blobs = BlobDir::new(cache.blobs_dir()).verify(quick);
    let mut report = VerifyReport {
        blobs: blobs.blobs,
        errors: blobs.errors,
        ..Default::default()
    };
    if let Ok(rd) = std::fs::read_dir(cache.registry_dir()) {
        let mut dirs: Vec<PathBuf> = rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_dir())
            .collect();
        dirs.sort();
        for d in dirs {
            report.datasets += 1;
            let name = d
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            let row = match read_manifest(&d.join("manifest.json")) {
                Ok(r) => r,
                Err(e) => {
                    report.errors.push(format!("{name}: {e:#}"));
                    continue;
                }
            };
            if row.dataset_hash != name {
                report.errors.push(format!(
                    "{name}: manifest dataset_hash is {}",
                    row.dataset_hash
                ));
            }
            for (file, f) in row.files.entries() {
                let p = d.join(file);
                let blob = cache.blob_path(&f.sha256);
                if !p.is_file() {
                    report.errors.push(format!("{name}/{file}: missing"));
                } else if !same_file(&p, &blob) {
                    match digest_file(&p) {
                        Ok(dg) if dg.sha256 == f.sha256 => {}
                        Ok(dg) => report.errors.push(format!(
                            "{name}/{file}: hashes to {} (manifest {})",
                            dg.sha256, f.sha256
                        )),
                        Err(e) => report.errors.push(format!("{name}/{file}: {e}")),
                    }
                }
            }
        }
    }
    report
}
