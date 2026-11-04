use super::*;

/// The two explicit uploads beside the model.
#[derive(Clone, Copy)]
pub enum Upload {
    Report,
    Code,
}

const MAX_REPORT: u64 = 16 * 1024 * 1024;
const MAX_CODE: u64 = 8 * 1024 * 1024;

fn check_upload(kind: Upload, bytes: &[u8]) -> Result<()> {
    match kind {
        Upload::Report => {
            anyhow::ensure!(bytes.starts_with(b"%PDF-"), "the report must be a PDF file");
            let tail = &bytes[bytes.len().saturating_sub(1024)..];
            anyhow::ensure!(
                tail.windows(5).any(|w| w == b"%%EOF"),
                "the report PDF looks truncated"
            );
        }
        Upload::Code => {
            let mut archive = zip::ZipArchive::new(io::Cursor::new(bytes))
                .context("generated code must be a ZIP archive")?;
            anyhow::ensure!(
                !archive.is_empty() && archive.len() <= 256,
                "the code archive must hold 1 to 256 files"
            );
            for n in 0..archive.len() {
                let file = archive.by_index(n)?;
                let name = file.name()?.to_string();
                anyhow::ensure!(
                    !file.is_dir() && !name.contains(['/', '\\']) && !name.starts_with('.'),
                    "the code archive must hold flat files only"
                );
            }
        }
    }
    Ok(())
}

/// Upload a run's PDF report or ZIP of generated code to its connected record, explicitly.
pub fn publish_file(run: &str, kind: Upload, file: &Path) -> Result<String> {
    use jope_client::Transport;
    let (limit, route, header, field, label) = match kind {
        Upload::Report => (
            MAX_REPORT,
            "report",
            "x-report-sha256",
            "reportSha256",
            "report",
        ),
        Upload::Code => (MAX_CODE, "code", "x-code-sha256", "codeSha256", "code"),
    };
    let mut bytes = Vec::new();
    fs::File::open(file)
        .with_context(|| format!("open {}", file.display()))?
        .take(limit + 1)
        .read_to_end(&mut bytes)?;
    anyhow::ensure!(
        !bytes.is_empty() && bytes.len() as u64 <= limit,
        "the {label} is empty or larger than {} MiB",
        limit / 1024 / 1024
    );
    check_upload(kind, &bytes)?;
    let id: jope_api::ids::RunId = run.parse()?;
    let local = local_client()?;
    let snapshot = local.run_status(&id)?;
    let version = local.system_info()?.version;
    sync_snapshot(&snapshot, &version)?;
    let hash = jope_runtime::sha256_hex(&bytes);
    let _lock = connection_lock()?;
    let c = connection()?;
    binding(&c, run)?;
    let record: SyncRecord = serde_json::from_slice(&read_private(&record_path(&c, run)?)?)?;
    let cloud = record.cloud_id.context("run is not synced")?;
    anyhow::ensure!(uuid(&cloud), "invalid cloud run");
    let client = Client::connected(&c)?;
    let media = match kind {
        Upload::Report => "application/pdf",
        Upload::Code => "application/zip",
    };
    let mut request = client.wire(
        "POST",
        &format!("/api/cli/runs/{cloud}/{route}"),
        bytes,
        media,
    )?;
    request.headers_mut().insert(header, hash.parse()?);
    let mut response = client.agent.run(request)?;
    let status = response.status().as_u16();
    let mut body = Vec::new();
    response
        .body_mut()
        .as_reader()
        .take(MAX_JSON + 1)
        .read_to_end(&mut body)?;
    anyhow::ensure!(
        status == 200,
        "{label} upload refused (HTTP {status}); an existing different {label} is immutable"
    );
    let value: Value = serde_json::from_slice(&body)?;
    anyhow::ensure!(value[field] == hash, "uploaded {label} digest differs");
    Ok(format!(
        "Uploaded {label} · SHA-256 {hash}\n{}/runs/{cloud}",
        c.origin
    ))
}
