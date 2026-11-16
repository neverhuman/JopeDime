//! Passive regional `/carbon/{asset}` signed release mirror & privacy-preserving download telemetry.
//!
//! Extracted from `edge.rs` so gateway routing and run state machines remain focused while keeping
//! strict traversal refusal, release-asset allowlisting, and daily-HMAC IP pseudonymization.

use crate::{
    cells::{
        edge::{CarbonGatewayRequest, CarbonGatewayResponse},
        engine_invoker::now_epoch_ms,
    },
    security::{auth::CarbonRateLimiter, sigv4::CarbonPresigner},
    storage::cas::sha256_lower_hex,
};

pub(crate) fn execute_passive_download_redirect(
    region: &str,
    presigner: &CarbonPresigner,
    rate_limiter: &CarbonRateLimiter,
    method: &str,
    clean_path: &str,
    req_id: &str,
    req: &CarbonGatewayRequest,
) -> CarbonGatewayResponse {
    let raw_path = req.header_ci("x-carbon-raw-path").unwrap_or(&req.path);
    let raw_lc = raw_path.to_ascii_lowercase();
    if raw_path.contains("..")
        || raw_lc.contains("%2e")
        || raw_lc.contains("%2f")
        || raw_lc.contains("%5c")
        || raw_path.contains('\\')
        || raw_path.bytes().any(|b| b <= 0x20 || b == 0x7f)
    {
        return download_refused_400(
            "download path contains disallowed traversal or encoded characters",
        );
    }

    let default_pub_tag = std::env::var("CARBON_PUBLISHED_INSTALLER_TAG")
        .ok()
        .map(|v| v.trim().to_owned())
        .filter(|v| is_valid_release_tag(v) && v != "latest")
        .unwrap_or_else(|| "carbon-cli-v1.1.0".to_owned());

    let (tag, asset, target_url) = match clean_path {
        "/carbon/install.sh" => (
            default_pub_tag.clone(),
            "install.sh".to_owned(),
            format!(
                "https://github.com/neverhuman/JopeDime/releases/download/{default_pub_tag}/install.sh"
            ),
        ),
        "/carbon/install.ps1" => (
            default_pub_tag.clone(),
            "install.ps1".to_owned(),
            format!(
                "https://github.com/neverhuman/JopeDime/releases/download/{default_pub_tag}/install.ps1"
            ),
        ),
        "/carbon/carbon-clouds.json" => (
            "latest".to_owned(),
            "carbon-clouds.json".to_owned(),
            "https://github.com/neverhuman/JopeDime/releases/latest/download/carbon-clouds.json"
                .to_owned(),
        ),
        p if p.starts_with("/carbon/releases/") => {
            let rem = &p["/carbon/releases/".len()..];
            let Some((tag_part, asset_part)) = rem.split_once('/') else {
                return download_refused_400("expected /carbon/releases/{tag}/{asset}");
            };
            if asset_part.contains('/')
                || !is_valid_release_tag(tag_part)
                || !is_valid_release_asset(asset_part)
            {
                return download_refused_400(
                    "requested release tag or asset is not on the Carbon download allowlist",
                );
            }
            let url = if tag_part == "latest" {
                format!(
                    "https://github.com/neverhuman/JopeDime/releases/latest/download/{asset_part}"
                )
            } else {
                format!(
                    "https://github.com/neverhuman/JopeDime/releases/download/{tag_part}/{asset_part}"
                )
            };
            (tag_part.to_owned(), asset_part.to_owned(), url)
        }
        _ => return download_refused_400("unknown Carbon download route"),
    };

    let now_ms = now_epoch_ms();
    let now_secs = now_ms / 1_000;
    let source_ip = req.trusted_caller_ip("unknown");
    if !rate_limiter.check_and_record(&format!("dl:{source_ip}"), now_secs, 240) {
        return CarbonGatewayResponse::error_json(
            429,
            "rate_limited",
            "download redirect rate limit exceeded (max 240/min per IP)",
        )
        .with_header("cache-control", "private, no-store");
    }
    let ip_day_hmac = presigner.daily_ip_pseudonym_hex(now_secs, source_ip);
    let mut client_hdr_names: Vec<&str> = req
        .headers
        .keys()
        .map(String::as_str)
        .filter(|k| {
            !k.eq_ignore_ascii_case("x-carbon-trusted-source-ip")
                && !k.eq_ignore_ascii_case("x-carbon-raw-path")
                && !k.eq_ignore_ascii_case("x-carbon-via-apigw-v2")
        })
        .collect();
    client_hdr_names.sort_unstable();
    let header_names_hash =
        sha256_lower_hex(client_hdr_names.join(",").as_bytes())[..16].to_owned();
    let ua = sanitize_download_log_field(req.header_ci("user-agent"), 256);
    let accept_lang = sanitize_download_log_field(req.header_ci("accept-language"), 128);
    let sec_ch_ua = sanitize_download_log_field(req.header_ci("sec-ch-ua"), 128);
    let sec_ch_platform = sanitize_download_log_field(req.header_ci("sec-ch-ua-platform"), 64);
    let sec_ch_arch = sanitize_download_log_field(req.header_ci("sec-ch-ua-arch"), 64);
    let viewer_country = sanitize_download_log_field(
        req.header_ci("cloudfront-viewer-country")
            .or_else(|| req.header_ci("cf-ipcountry")),
        16,
    );
    let viewer_asn = sanitize_download_log_field(req.header_ci("cloudfront-viewer-asn"), 32);
    let tls_version = sanitize_download_log_field(
        req.header_ci("x-amzn-tls-version")
            .or_else(|| req.header_ci("cloudfront-viewer-tls")),
        64,
    );
    let dnt =
        matches!(req.header_ci("dnt"), Some("1")) || matches!(req.header_ci("sec-gpc"), Some("1"));
    let (inf_os, inf_arch, inf_libc, inf_archive) =
        infer_download_platform(&asset, ua.as_deref().unwrap_or(""));

    let telemetry_event = serde_json::json!({
        "schema": "carbon.download.v1",
        "ts_ms": now_ms,
        "request_id": req_id,
        "region": region,
        "method": method,
        "path": clean_path,
        "tag": tag,
        "asset": asset,
        "target_url": target_url,
        "source_ip": source_ip,
        "ip_day_hmac": ip_day_hmac,
        "header_names_hash": header_names_hash,
        "user_agent": ua,
        "accept_language": accept_lang,
        "sec_ch_ua": sec_ch_ua,
        "sec_ch_ua_platform": sec_ch_platform,
        "sec_ch_arch": sec_ch_arch,
        "viewer_country": viewer_country,
        "viewer_asn": viewer_asn,
        "tls_version": tls_version,
        "dnt": dnt,
        "inferred_os": inf_os,
        "inferred_arch": inf_arch,
        "inferred_libc": inf_libc,
        "inferred_archive": inf_archive,
    });
    if let Ok(line) = crate::error::to_json_string(&telemetry_event) {
        eprintln!("{line}");
    }

    CarbonGatewayResponse::bytes_ok(302, "text/plain; charset=utf-8", b"")
        .with_header("location", target_url)
        .with_header("cache-control", "private, no-store")
}

fn download_refused_400(message: &str) -> CarbonGatewayResponse {
    CarbonGatewayResponse::error_json(400, "invalid_download_target", message)
        .with_header("cache-control", "private, no-store")
}

pub(crate) fn is_valid_release_tag(tag: &str) -> bool {
    if tag == "latest" {
        return true;
    }
    if tag.is_empty() || tag.len() > 64 || tag.contains("..") {
        return false;
    }
    let semver_part = if let Some(rest) = tag.strip_prefix("carbon-cli-v") {
        rest
    } else if let Some(rest) = tag.strip_prefix('v') {
        rest
    } else {
        return false;
    };
    let (core, prerelease) = match semver_part.split_once('-') {
        Some((c, pre)) => (c, Some(pre)),
        None => (semver_part, None),
    };
    let mut nums = core.split('.');
    let (Some(maj), Some(min), Some(pat), None) =
        (nums.next(), nums.next(), nums.next(), nums.next())
    else {
        return false;
    };
    let digits_ok = [maj, min, pat].iter().all(|part| {
        !part.is_empty() && part.len() <= 8 && part.bytes().all(|b| b.is_ascii_digit())
    });
    if !digits_ok {
        return false;
    }
    if let Some(pre) = prerelease {
        if pre.is_empty()
            || !pre
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
        {
            return false;
        }
    }
    true
}

pub(crate) fn is_valid_release_asset(asset: &str) -> bool {
    if asset.is_empty()
        || asset.len() > 128
        || asset.contains("..")
        || !asset
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
    {
        return false;
    }
    if matches!(
        asset,
        "SHA256SUMS.txt"
            | "SHA256SUMS.txt.sig"
            | "carbon-clouds.json"
            | "install.sh"
            | "install.ps1"
            | "carbon-linux-x86_64-musl"
            | "carbon-linux-aarch64-musl"
            | "carbon-linux-armv7-musleabihf"
            | "carbon-darwin-arm64"
            | "carbon-darwin-x86_64"
            | "carbon-darwin-universal"
    ) {
        return true;
    }
    let has_os_prefix = asset.starts_with("carbon-linux-")
        || asset.starts_with("carbon-darwin-")
        || asset.starts_with("carbon-windows-");
    let has_archive_suffix = asset.ends_with(".tar.zst")
        || asset.ends_with(".tar.xz")
        || asset.ends_with(".tar.gz")
        || asset.ends_with(".zip")
        || asset.ends_with(".exe");
    has_os_prefix && has_archive_suffix
}

fn sanitize_download_log_field(raw: Option<&str>, max_chars: usize) -> Option<String> {
    let text = raw?.trim();
    if text.is_empty() {
        return None;
    }
    let redacted = crate::observability::redact_secrets(text);
    let cleaned: String = redacted
        .chars()
        .filter(|c| !c.is_control())
        .take(max_chars)
        .collect();
    if cleaned.is_empty() {
        None
    } else {
        Some(cleaned)
    }
}

fn infer_download_platform(
    asset: &str,
    user_agent: &str,
) -> (&'static str, &'static str, &'static str, &'static str) {
    let archive = if asset.ends_with(".tar.zst") {
        "tar.zst"
    } else if asset.ends_with(".tar.xz") {
        "tar.xz"
    } else if asset.ends_with(".tar.gz") {
        "tar.gz"
    } else if asset.ends_with(".zip") {
        "zip"
    } else if asset.ends_with(".exe") {
        "exe"
    } else if asset.ends_with(".ps1") {
        "ps1"
    } else if asset.ends_with(".sh") {
        "sh"
    } else {
        "meta"
    };

    let ua_lc = user_agent.to_ascii_lowercase();
    let os = if asset.starts_with("carbon-linux-") || ua_lc.contains("linux") {
        "linux"
    } else if asset.starts_with("carbon-darwin-")
        || ua_lc.contains("darwin")
        || ua_lc.contains("macintosh")
        || ua_lc.contains("mac os")
    {
        "macos"
    } else if asset.starts_with("carbon-windows-")
        || asset.ends_with(".ps1")
        || ua_lc.contains("windows")
        || ua_lc.contains("powershell")
    {
        "windows"
    } else {
        "unknown"
    };

    let arch = if asset.contains("arm64")
        || asset.contains("aarch64")
        || ua_lc.contains("aarch64")
        || ua_lc.contains("arm64")
    {
        "arm64"
    } else if asset.contains("armv7") || ua_lc.contains("armv7") {
        "armv7"
    } else if asset.contains("x86_64")
        || asset.contains("amd64")
        || ua_lc.contains("x86_64")
        || ua_lc.contains("win64")
        || ua_lc.contains("x64")
    {
        "x86_64"
    } else if asset.contains("universal") {
        "universal"
    } else {
        "unknown"
    };

    let libc = if asset.contains("musleabihf") {
        "musleabihf"
    } else if asset.contains("musl") {
        "musl"
    } else if asset.contains("msvc") || os == "windows" {
        "msvc"
    } else if os == "macos" {
        "libsystem"
    } else if os == "linux" {
        "gnu_or_musl"
    } else {
        "unknown"
    };

    (os, arch, libc, archive)
}
