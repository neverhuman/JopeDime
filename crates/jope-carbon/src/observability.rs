//! Zero-dependency structured observability, secret redaction, request correlation IDs,
//! `NO_COLOR`/`DO_NOT_TRACK` handling, `BrokenPipe`-safe stdout, and semantic CLI exit codes.

use std::io::{self, IsTerminal, Write};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::Arc;

use serde::Serialize;

pub const EXIT_OK: u8 = 0;
pub const EXIT_USAGE: u8 = 2;
pub const EXIT_DATA_VALIDATION: u8 = 3;
pub const EXIT_QUOTA_TIER: u8 = 4;
pub const EXIT_NETWORK: u8 = 5;
pub const EXIT_CRYPTO_INTEGRITY: u8 = 6;
pub const EXIT_SERVER_RUN: u8 = 7;
pub const EXIT_AUTH_EXPIRED: u8 = 9;
pub const EXIT_INTERRUPTED: u8 = 130;

/// `CarbonLogLevel` controls verbosity on `stderr`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum CarbonLogLevel {
    Quiet = 0,
    Info = 1,
    Debug = 2,
    Trace = 3,
}

/// `CarbonLogFormat` controls human text vs single-line JSON diagnostic output on `stderr`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum CarbonLogFormat {
    Text = 0,
    Json = 1,
}

/// `CarbonColorMode` controls ANSI color output (`auto`, `always`, `never`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum CarbonColorMode {
    Auto = 0,
    Always = 1,
    Never = 2,
}

static LOG_LEVEL: AtomicU8 = AtomicU8::new(CarbonLogLevel::Info as u8);
static LOG_FORMAT: AtomicU8 = AtomicU8::new(CarbonLogFormat::Text as u8);
static COLOR_ENABLED: AtomicBool = AtomicBool::new(false);
static SIGINT_RECEIVED: AtomicBool = AtomicBool::new(false);

/// Return `true` when `DO_NOT_TRACK` or `CARBON_NO_TELEMETRY` is set to a truthy value (`!= "0"`).
pub fn telemetry_minimized() -> bool {
    for var in ["DO_NOT_TRACK", "CARBON_NO_TELEMETRY"] {
        if let Ok(val) = std::env::var(var) {
            let trimmed = val.trim();
            if !trimmed.is_empty() && trimmed != "0" && !trimmed.eq_ignore_ascii_case("false") {
                return true;
            }
        }
    }
    false
}

/// Initialize global CLI logging, format, and ANSI color configuration.
pub fn init_cli_observability(
    verbose_count: u8,
    quiet: bool,
    log_format: CarbonLogFormat,
    color_mode: CarbonColorMode,
) {
    let mut level = if quiet {
        CarbonLogLevel::Quiet
    } else {
        match verbose_count {
            0 => CarbonLogLevel::Info,
            1 => CarbonLogLevel::Debug,
            _ => CarbonLogLevel::Trace,
        }
    };
    if !quiet && verbose_count == 0 {
        if let Ok(raw) = std::env::var("CARBON_LOG") {
            match raw.trim().to_ascii_lowercase().as_str() {
                "off" | "quiet" | "error" => level = CarbonLogLevel::Quiet,
                "info" | "warn" => level = CarbonLogLevel::Info,
                "debug" => level = CarbonLogLevel::Debug,
                "trace" => level = CarbonLogLevel::Trace,
                _ => {}
            }
        }
    }
    LOG_LEVEL.store(level as u8, Ordering::Relaxed);
    LOG_FORMAT.store(log_format as u8, Ordering::Relaxed);

    let use_color = match color_mode {
        CarbonColorMode::Always => true,
        CarbonColorMode::Never => false,
        CarbonColorMode::Auto => {
            let no_color = std::env::var_os("NO_COLOR").is_some_and(|v| !v.is_empty());
            let term_dumb = std::env::var("TERM")
                .map(|t| t.eq_ignore_ascii_case("dumb"))
                .unwrap_or(false);
            let force_color = std::env::var("CLICOLOR_FORCE")
                .map(|v| !v.trim().is_empty() && v.trim() != "0")
                .unwrap_or(false);
            if no_color || term_dumb {
                false
            } else if force_color {
                true
            } else {
                io::stderr().is_terminal()
            }
        }
    };
    COLOR_ENABLED.store(use_color, Ordering::Relaxed);
}

pub fn active_log_level() -> CarbonLogLevel {
    match LOG_LEVEL.load(Ordering::Relaxed) {
        0 => CarbonLogLevel::Quiet,
        2 => CarbonLogLevel::Debug,
        3 => CarbonLogLevel::Trace,
        _ => CarbonLogLevel::Info,
    }
}

pub fn active_log_format() -> CarbonLogFormat {
    match LOG_FORMAT.load(Ordering::Relaxed) {
        1 => CarbonLogFormat::Json,
        _ => CarbonLogFormat::Text,
    }
}

pub fn color_enabled() -> bool {
    COLOR_ENABLED.load(Ordering::Relaxed)
}

pub fn stderr_is_tty_interactive() -> bool {
    active_log_level() >= CarbonLogLevel::Info
        && active_log_format() == CarbonLogFormat::Text
        && io::stderr().is_terminal()
}

/// Alias for [`stderr_is_tty_interactive`] used by CLI progress spinners.
pub fn stderr_is_tty() -> bool {
    stderr_is_tty_interactive()
}

/// Return `true` once a cooperative `SIGINT` (`Ctrl-C`) signal has been observed.
pub fn sigint_received() -> bool {
    SIGINT_RECEIVED.load(Ordering::Relaxed)
}

/// Emit a warning event at `Info` visibility (`stderr` text or `carbon.log.v1` JSON), scrubbing secrets.
pub fn log_warn(message: &str) {
    log_event(
        CarbonLogLevel::Info,
        "warning",
        &format!("warning: {message}"),
    );
}

/// Emit a structured or human-friendly diagnostic event to `stderr`, automatically redacting secrets.
pub fn log_event(level: CarbonLogLevel, event: &str, message: &str) {
    if level > active_log_level() {
        return;
    }
    let clean_msg = redact_secrets(message);
    if active_log_format() == CarbonLogFormat::Json {
        let lvl = match level {
            CarbonLogLevel::Quiet => "error",
            CarbonLogLevel::Info => "info",
            CarbonLogLevel::Debug => "debug",
            CarbonLogLevel::Trace => "trace",
        };
        let payload = serde_json::json!({
            "schema": "carbon.log.v1",
            "ts_ms": crate::cells::engine_invoker::now_epoch_ms(),
            "level": lvl,
            "event": event,
            "message": clean_msg,
        });
        let _ = writeln!(io::stderr(), "{payload}");
    } else {
        let prefix = match (level, color_enabled()) {
            (CarbonLogLevel::Info, true) => "\x1b[36m[carbon]\x1b[0m",
            (CarbonLogLevel::Debug, true) => "\x1b[34m[carbon:debug]\x1b[0m",
            (CarbonLogLevel::Trace, true) => "\x1b[90m[carbon:trace]\x1b[0m",
            (CarbonLogLevel::Debug, false) => "[carbon:debug]",
            (CarbonLogLevel::Trace, false) => "[carbon:trace]",
            _ => "[carbon]",
        };
        let _ = writeln!(io::stderr(), "{prefix} {clean_msg}");
    }
}

/// Sanitize an incoming `x-carbon-request-id` header or mint a fresh `req_<ulid>` correlation ID.
pub fn generate_or_sanitize_request_id(incoming: Option<&str>) -> String {
    if let Some(raw) = incoming {
        let trimmed = raw.trim();
        if (8..=64).contains(&trimmed.len())
            && trimmed
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
        {
            return trimmed.to_owned();
        }
    }
    format!("req_{}", ulid::Ulid::new())
}

/// Scrub capability tokens (`ctk1.*`), `Bearer` credentials, presigned S3/gateway query strings,
/// and 64-hex secret values from diagnostic or error strings before logging or returning to callers.
pub fn redact_secrets(input: &str) -> String {
    let mut out = mask_presigned_urls(input);
    out = mask_ctk1_tokens(&out);
    out = mask_bearer_tokens(&out);
    mask_labeled_hex_secrets(&out)
}

fn mask_presigned_urls(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(pos) = rest
        .find("https://")
        .into_iter()
        .chain(rest.find("http://"))
        .min()
    {
        out.push_str(&rest[..pos]);
        let after = &rest[pos..];
        let end = after
            .find(|c: char| {
                c.is_whitespace() || matches!(c, '"' | '\'' | ')' | ']' | '>' | ',' | ';')
            })
            .unwrap_or(after.len());
        let url_tok = &after[..end];
        if let Some((base, query)) = url_tok.split_once('?') {
            let ql = query.to_ascii_lowercase();
            if ql.contains("x-amz-signature=")
                || ql.contains("x-amz-credential=")
                || ql.contains("x-amz-security-token=")
                || (ql.contains("sig=") && ql.contains("exp="))
            {
                out.push_str(base);
                out.push_str("?[REDACTED_PRESIGNED_QUERY]");
            } else {
                out.push_str(url_tok);
            }
        } else {
            out.push_str(url_tok);
        }
        rest = &after[end..];
    }
    out.push_str(rest);
    out
}

fn mask_ctk1_tokens(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(pos) = rest.find("ctk1.") {
        out.push_str(&rest[..pos]);
        let after = &rest[pos + 5..];
        let tok_len = after
            .bytes()
            .take_while(|b| b.is_ascii_alphanumeric() || matches!(*b, b'-' | b'_' | b'.'))
            .count();
        if tok_len >= 8 {
            out.push_str("ctk1.[REDACTED]");
            rest = &after[tok_len..];
        } else {
            out.push_str("ctk1.");
            rest = after;
        }
    }
    out.push_str(rest);
    out
}

fn mask_bearer_tokens(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(pos) = find_ascii_case_insensitive(rest, "bearer ") {
        let prefix_end = pos + 7;
        out.push_str(&rest[..prefix_end]);
        let after = &rest[prefix_end..];
        let trimmed = after.trim_start_matches(' ');
        if let Some(stripped) = trimmed.strip_prefix("[REDACTED]") {
            out.push_str("[REDACTED]");
            rest = stripped;
            continue;
        }
        let tok_len = trimmed
            .bytes()
            .take_while(|b| {
                b.is_ascii_alphanumeric() || matches!(*b, b'-' | b'_' | b'.' | b'=' | b'+' | b'/')
            })
            .count();
        if tok_len >= 8 {
            out.push_str("[REDACTED]");
            rest = &trimmed[tok_len..];
        } else {
            rest = after;
        }
    }
    out.push_str(rest);
    out
}

fn mask_labeled_hex_secrets(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i].is_ascii_hexdigit() {
            let mut j = i;
            while j < bytes.len() && bytes[j].is_ascii_hexdigit() {
                j += 1;
            }
            if j - i == 64 {
                let lookback_start = i.saturating_sub(40);
                let ctx = s.get(lookback_start..i).unwrap_or("").to_ascii_lowercase();
                if ctx.contains("secret")
                    || ctx.contains("kek")
                    || ctx.contains("dek")
                    || ctx.contains("token")
                {
                    out.push_str("[REDACTED_HEX_SECRET]");
                    i = j;
                    continue;
                }
            }
        }
        if let Some(ch) = s[i..].chars().next() {
            out.push(ch);
            i += ch.len_utf8();
        } else {
            break;
        }
    }
    out
}

fn find_ascii_case_insensitive(haystack: &str, needle_lower: &str) -> Option<usize> {
    let nlen = needle_lower.len();
    if nlen == 0 || haystack.len() < nlen {
        return None;
    }
    let hb = haystack.as_bytes();
    let nb = needle_lower.as_bytes();
    (0..=(hb.len() - nlen)).find(|&idx| {
        hb[idx..idx + nlen]
            .iter()
            .zip(nb.iter())
            .all(|(a, b)| a.to_ascii_lowercase() == *b)
    })
}

/// Write a single line to `stdout`, returning `Err(true)` on `BrokenPipe` (`EPIPE`) so callers
/// piped into `head`/`jq` can exit `0` without panicking.
pub fn write_stdout_line(line: &str) -> Result<(), bool> {
    let mut stdout = io::stdout().lock();
    match writeln!(stdout, "{line}").and_then(|()| stdout.flush()) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == io::ErrorKind::BrokenPipe => Err(true),
        Err(_) => Err(false),
    }
}

/// Classify a CLI failure message into `(exit_code, error_kind, optional_repair_hint)`.
pub fn classify_cli_error(message: &str) -> (u8, &'static str, Option<&'static str>) {
    let lc = message.to_ascii_lowercase();
    if lc.contains("interrupted by user") || lc.contains("sigint") {
        return (
            EXIT_INTERRUPTED,
            "interrupted",
            Some("Run was interrupted; if unshredded, inspect or resume with `carbon resume --run latest`."),
        );
    }
    if lc.contains("unauthorized")
        || lc.contains("http 401")
        || lc.contains("expired")
        || lc.contains("capability token")
    {
        return (
            EXIT_AUTH_EXPIRED,
            "auth_error",
            Some("Run `carbon whoami` or `carbon register` to inspect or refresh your capability token."),
        );
    }
    if lc.contains("tier_limit")
        || lc.contains("tier_escalation")
        || lc.contains("quota")
        || lc.contains("http 403")
        || lc.contains("http 413")
        || lc.contains("rate_limited")
        || lc.contains("http 429")
    {
        return (
            EXIT_QUOTA_TIER,
            "quota_or_tier_limit",
            Some("Check dataset/predict limits with `carbon inspect <CSV>` or upgrade tier (`--tier pro` / `carbon serve`)."),
        );
    }
    if lc.contains("sha-256")
        || lc.contains("sha256")
        || lc.contains("ecdsa")
        || lc.contains("signature")
        || lc.contains("cuen")
        || lc.contains("cudk")
        || lc.contains("kek")
        || lc.contains("tamper")
    {
        return (
            EXIT_CRYPTO_INTEGRITY,
            "crypto_integrity_error",
            Some("Run `carbon self-verify` and `carbon doctor` to verify binary signatures and regional KEK pins."),
        );
    }
    if lc.contains("http 500")
        || lc.contains("http 502")
        || lc.contains("http 503")
        || lc.contains("http 504")
        || lc.contains("run failed")
    {
        return (
            EXIT_SERVER_RUN,
            "server_execution_error",
            Some("Check cell health with `carbon doctor` and retry; correlation ID (`x-carbon-request-id`) is included above."),
        );
    }
    if lc.contains("connect")
        || lc.contains("tls")
        || lc.contains("certificate")
        || lc.contains("timeout")
        || lc.contains("timed out")
        || lc.contains("dns")
        || lc.contains("unreachable")
        || lc.contains("insecure cleartext http")
        || lc.contains("insecure redirect")
    {
        return (
            EXIT_NETWORK,
            "network_or_tls_error",
            Some("Run `carbon doctor` to check regional connectivity; behind a corporate proxy, set `CARBON_CA_BUNDLE=/path/to/ca.pem`."),
        );
    }
    if lc.contains("csv")
        || lc.contains("column")
        || lc.contains("target")
        || lc.contains("nan")
        || lc.contains("infinite")
        || lc.contains("inf ")
        || lc.contains("utf-8")
        || lc.contains("objective")
        || lc.contains("fitness")
        || lc.contains("row")
    {
        return (
            EXIT_DATA_VALIDATION,
            "data_validation_error",
            Some("Run `carbon inspect <FILE>` to check headers, missing values, and numeric columns, or `carbon fitness list` for valid objectives."),
        );
    }
    (EXIT_USAGE, "cli_error", None)
}

#[derive(Serialize)]
struct CarbonJsonErrorEnvelope<'a> {
    schema: &'static str,
    error: CarbonJsonErrorBody<'a>,
}

#[derive(Serialize)]
struct CarbonJsonErrorBody<'a> {
    kind: &'a str,
    exit_code: u8,
    message: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    hint: Option<&'a str>,
}

/// Render a redacted CLI error (and emit `carbon.error.v1` JSON to `stdout` when `json_mode` is
/// active) and return its semantic `ExitCode`.
pub fn emit_cli_error(json_mode: bool, raw_message: &str) -> ExitCode {
    let clean = redact_secrets(raw_message);
    let (code, kind, hint) = classify_cli_error(&clean);
    if json_mode {
        let env = CarbonJsonErrorEnvelope {
            schema: "carbon.error.v1",
            error: CarbonJsonErrorBody {
                kind,
                exit_code: code,
                message: &clean,
                hint,
            },
        };
        if let Ok(text) = serde_json::to_string_pretty(&env) {
            let _ = write_stdout_line(&text);
        }
    } else {
        let err_tag = if color_enabled() {
            "\x1b[31merror\x1b[0m"
        } else {
            "error"
        };
        let _ = writeln!(io::stderr(), "{err_tag} [{kind}]: {clean}");
        if let Some(h) = hint {
            let hint_tag = if color_enabled() {
                "\x1b[33mhint\x1b[0m"
            } else {
                "hint"
            };
            let _ = writeln!(io::stderr(), "  {hint_tag}: {h}");
        }
    }
    ExitCode::from(code)
}

#[cfg(all(unix, not(feature = "serve")))]
#[allow(unsafe_code)]
mod thin_cli_signal {
    use std::sync::atomic::Ordering;

    type SigHandler = extern "C" fn(i32);

    extern "C" {
        fn signal(signum: i32, handler: SigHandler) -> usize;
    }

    extern "C" fn on_termination_signal(_sig: i32) {
        super::SIGINT_RECEIVED.store(true, Ordering::SeqCst);
    }

    pub(super) fn register_once() {
        const SIGINT_NUM: i32 = 2;
        const SIGTERM_NUM: i32 = 15;
        // SAFETY: `on_termination_signal` performs only an async-signal-safe `AtomicBool::store`.
        unsafe {
            let _ = signal(SIGINT_NUM, on_termination_signal);
            let _ = signal(SIGTERM_NUM, on_termination_signal);
        }
    }
}

#[cfg(all(windows, not(feature = "serve")))]
#[allow(unsafe_code)]
mod thin_cli_signal {
    use std::sync::atomic::Ordering;

    type ConsoleHandler = unsafe extern "system" fn(u32) -> i32;

    #[link(name = "kernel32")]
    extern "system" {
        fn SetConsoleCtrlHandler(handler_routine: Option<ConsoleHandler>, add: i32) -> i32;
    }

    unsafe extern "system" fn on_ctrl_event(_ctrl_type: u32) -> i32 {
        super::SIGINT_RECEIVED.store(true, Ordering::SeqCst);
        1
    }

    pub(super) fn register_once() {
        // SAFETY: `on_ctrl_event` performs only an atomic store and returns TRUE (1) so cooperative
        // loops can shred or persist handles cleanly before exiting.
        unsafe {
            let _ = SetConsoleCtrlHandler(Some(on_ctrl_event), 1);
        }
    }
}

/// Install a cooperative Ctrl-C (`SIGINT` / `SIGTERM`) flag across both thin (`--features cli`)
/// and full (`--features serve`) builds.
pub fn install_sigint_flag() -> Arc<AtomicBool> {
    let flag = Arc::new(AtomicBool::new(false));
    #[cfg(feature = "serve")]
    {
        let flag_bg = Arc::clone(&flag);
        let _ = std::thread::Builder::new()
            .name("carbon-sigint-watch".to_owned())
            .spawn(move || {
                if let Ok(rt) = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    rt.block_on(async move {
                        if tokio::signal::ctrl_c().await.is_ok() {
                            SIGINT_RECEIVED.store(true, Ordering::SeqCst);
                            flag_bg.store(true, Ordering::SeqCst);
                        }
                    });
                }
            });
    }
    #[cfg(all(any(unix, windows), not(feature = "serve")))]
    {
        static ONCE: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        ONCE.get_or_init(thin_cli_signal::register_once);
    }
    flag
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redact_secrets_scrubs_presigned_urls_tokens_and_hex_secrets() {
        let fake_cred = format!("{}{}", "AKIA", "000000000000TEST");
        let fake_sig = "de".repeat(4);
        let fake_ctk = format!("ctk1.{}.{}", "eyJ0ZW5hbnRfaWQiOiJhY21lIn0", "12".repeat(8));
        let fake_hex64 = "ab".repeat(32);
        let dirty = format!(
            "PUT https://bucket.s3.us-east-2.amazonaws.com/cell-v1/blobs/abc?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential={fake_cred}&X-Amz-Signature={fake_sig} failed with Authorization: Bearer {fake_ctk} and dek_hex={fake_hex64}"
        );
        let scrubbed = redact_secrets(&dirty);
        assert!(
            !scrubbed.contains(&fake_cred),
            "leaked AWS credential: {scrubbed}"
        );
        assert!(
            !scrubbed.contains(&fake_sig),
            "leaked S3 signature: {scrubbed}"
        );
        assert!(
            scrubbed.contains("?[REDACTED_PRESIGNED_QUERY]"),
            "missing presigned marker: {scrubbed}"
        );
        assert!(
            !scrubbed.contains("eyJ0ZW5hbnRfaWQiOiJhY21lIn0"),
            "leaked ctk1 payload: {scrubbed}"
        );
        assert!(
            !scrubbed.contains(&fake_hex64),
            "leaked 64-hex secret: {scrubbed}"
        );
    }

    #[test]
    fn sanitize_request_id_accepts_safe_ids_and_replaces_unsafe_ones() {
        assert_eq!(
            generate_or_sanitize_request_id(Some("req_01JCCARB0N_ok-1.2")),
            "req_01JCCARB0N_ok-1.2"
        );
        let generated = generate_or_sanitize_request_id(Some("bad\r\nheader"));
        assert!(generated.starts_with("req_"));
    }

    #[test]
    fn classify_cli_error_prioritizes_http_and_transport_over_target_substring() {
        let (code, kind, _) = classify_cli_error("HTTP 503 Service Unavailable on target endpoint");
        assert_eq!(code, EXIT_SERVER_RUN);
        assert_eq!(kind, "server_execution_error");

        let (code_net, kind_net, _) =
            classify_cli_error("refusing insecure cleartext HTTP endpoint 'http://example.com'");
        assert_eq!(code_net, EXIT_NETWORK);
        assert_eq!(kind_net, "network_or_tls_error");
    }
}
