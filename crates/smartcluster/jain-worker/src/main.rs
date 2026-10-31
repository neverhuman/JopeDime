//! Closed, typed workload handler launcher.
//!
//! The daemon invokes this binary with a template enum and typed, bounded
//! arguments. Template names currently classify requests only; this binary
//! validates the request and emits progress plus an application receipt, but
//! does not execute the named workload. There is no arbitrary executable path,
//! shell payload, or request-controlled working directory. Downstream services
//! can consume the structured launcher receipt and durable daemon events.

use std::{collections::BTreeMap, env, io, os::fd::RawFd, str::FromStr};

use jain_smartcluster_core::{
    AttemptNumber, FenceToken, FencedAttempt, JainWorkload, JobId, ProtocolVersion,
    StructuredProgress, Validate, WorkerReport, WorkerReportFrame,
};
use nix::fcntl::{fcntl, FcntlArg};
use nix::sys::socket::{send, MsgFlags};
use serde::Serialize;

const MAX_FRAME_BYTES: usize = 1024 * 1024;
const HANDLER_MANIFEST_VERSION: &str = "jain-smartcluster-handlers-v1";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let template = parse_template(&env::args().nth(1).ok_or("missing typed Jain workload")?)?;
    validate_environment()?;
    let attempt = fenced_attempt_from_environment()?;
    let fd = parse_control_fd(&env::var("SCQ_CONTROL_FD")?)?;

    let mut sequence = 1;
    send_report(fd, &attempt, sequence, WorkerReport::Ready)?;
    sequence += 1;

    let arguments = env::args().skip(2).collect::<Vec<_>>();
    validate_arguments(&arguments)?;
    send_report(
        fd,
        &attempt,
        sequence,
        WorkerReport::Progress(StructuredProgress {
            phase: handler_name(template).to_owned(),
            completed: 0,
            total: Some(1),
            message: Some("allowlisted handler started".to_owned()),
            fields: BTreeMap::from([(
                "handler_manifest".to_owned(),
                HANDLER_MANIFEST_VERSION.to_owned(),
            )]),
        }),
    )?;
    sequence += 1;

    let receipt = handler_receipt(template, arguments.len())?;
    send_report(
        fd,
        &attempt,
        sequence,
        WorkerReport::Application {
            name: "jain.handler.receipt.v1".to_owned(),
            payload: receipt,
        },
    )?;
    sequence += 1;
    send_report(
        fd,
        &attempt,
        sequence,
        WorkerReport::Progress(StructuredProgress {
            phase: handler_name(template).to_owned(),
            completed: 1,
            total: Some(1),
            message: Some("allowlisted handler completed".to_owned()),
            fields: BTreeMap::new(),
        }),
    )?;
    Ok(())
}

fn parse_template(value: &str) -> Result<JainWorkload, io::Error> {
    let template = match value {
        "dataset-profile" => JainWorkload::DatasetProfile,
        "feature-expansion" => JainWorkload::FeatureExpansion,
        "training" => JainWorkload::Training,
        "report-generation" => JainWorkload::ReportGeneration,
        "prediction" => JainWorkload::Prediction,
        "export" => JainWorkload::Export,
        "chat" => JainWorkload::Chat,
        "research" => JainWorkload::Research,
        "managed-browser" => JainWorkload::ManagedBrowser,
        "jope-encode" => JainWorkload::JopeEncode,
        "jope-propose" => JainWorkload::JopePropose,
        "lime-run" => JainWorkload::LimeRun,
        "lime-batch" => JainWorkload::LimeBatch,
        other => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unknown Jain workload template: {other}"),
            ))
        }
    };
    Ok(template)
}

fn handler_name(template: JainWorkload) -> &'static str {
    match template {
        JainWorkload::DatasetProfile => "dataset-profile",
        JainWorkload::FeatureExpansion => "feature-expansion",
        JainWorkload::Training => "training",
        JainWorkload::ReportGeneration => "report-generation",
        JainWorkload::Prediction => "prediction",
        JainWorkload::Export => "export",
        JainWorkload::Chat => "chat",
        JainWorkload::Research => "research",
        JainWorkload::ManagedBrowser => "managed-browser",
        JainWorkload::JopeEncode => "jope-encode",
        JainWorkload::JopePropose => "jope-propose",
        JainWorkload::LimeRun => "lime-run",
        JainWorkload::LimeBatch => "lime-batch",
    }
}

fn validate_arguments(arguments: &[String]) -> Result<(), io::Error> {
    if arguments.len() > 256 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "handler argument count exceeds the worker limit",
        ));
    }
    if arguments.iter().any(|argument| {
        argument.is_empty() || argument.len() > 16 * 1024 || argument.contains('\0')
    }) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "handler arguments are empty, oversized, or contain NUL",
        ));
    }
    Ok(())
}

/// Keep the worker's inherited environment deliberately small.  The daemon
/// already strips the environment; this second boundary protects the worker
/// if it is invoked by an unexpected launcher.
fn validate_environment() -> Result<(), io::Error> {
    const ALLOWED: &[&str] = &[
        "PATH",
        "CUDA_VISIBLE_DEVICES",
        "SCQ_CONTROL_FD",
        "SCQ_JOB_ID",
        "SCQ_ATTEMPT_NUMBER",
        "SCQ_ATTEMPT_FENCE",
        "JAIN_ARTIFACT_ROOT",
        "JAIN_SESSION_ID",
        "JAIN_RUN_ID",
    ];
    for (name, _) in env::vars() {
        if !ALLOWED.contains(&name.as_str()) {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                format!("environment variable is not allowlisted: {name}"),
            ));
        }
    }
    Ok(())
}

fn handler_receipt(template: JainWorkload, argument_count: usize) -> Result<Vec<u8>, io::Error> {
    serde_json::to_vec(&serde_json::json!({
        "schema_version": "jain.handler.receipt/v1",
        "handler_manifest": HANDLER_MANIFEST_VERSION,
        "template": handler_name(template),
        "argument_count": argument_count,
        "status": "completed",
    }))
    .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

fn parse_control_fd(value: &str) -> Result<RawFd, io::Error> {
    let fd = value.parse::<RawFd>().map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("invalid SCQ_CONTROL_FD: {error}"),
        )
    })?;
    if fd < 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "SCQ_CONTROL_FD must be nonnegative",
        ));
    }
    fcntl(fd, FcntlArg::F_GETFD).map_err(errno_io)?;
    Ok(fd)
}

fn fenced_attempt_from_environment() -> Result<FencedAttempt, Box<dyn std::error::Error>> {
    let job_id = JobId::from_str(&env::var("SCQ_JOB_ID")?)?;
    let number = AttemptNumber::new(env::var("SCQ_ATTEMPT_NUMBER")?.parse()?)?;
    let fence = FenceToken::new(env::var("SCQ_ATTEMPT_FENCE")?.parse()?)?;
    Ok(FencedAttempt::new(job_id, number, fence))
}

fn send_report(
    fd: RawFd,
    attempt: &FencedAttempt,
    sequence: u64,
    report: WorkerReport,
) -> io::Result<()> {
    report
        .validate()
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    send_frame(
        fd,
        &WorkerReportFrame {
            version: ProtocolVersion::CURRENT,
            attempt: *attempt,
            sequence,
            report,
        },
    )
}

fn send_frame<T: Serialize>(fd: RawFd, value: &T) -> io::Result<()> {
    let payload = postcard::to_stdvec(value)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    if payload.len() > MAX_FRAME_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "control frame exceeds worker limit",
        ));
    }
    let length = u32::try_from(payload.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "control frame too large"))?;
    let mut frame = Vec::with_capacity(4 + payload.len());
    frame.extend_from_slice(&length.to_be_bytes());
    frame.extend_from_slice(&payload);
    let mut bytes = frame.as_slice();
    while !bytes.is_empty() {
        match send(fd, bytes, MsgFlags::MSG_NOSIGNAL) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "control socket stopped accepting bytes",
                ))
            }
            Ok(written) => bytes = &bytes[written..],
            Err(nix::errno::Errno::EINTR) => {}
            Err(error) => return Err(errno_io(error)),
        }
    }
    Ok(())
}

fn errno_io(error: nix::errno::Errno) -> io::Error {
    io::Error::from_raw_os_error(error as i32)
}

#[cfg(test)]
mod tests {
    use std::io::Read;
    use std::os::fd::AsRawFd;
    use std::os::unix::net::UnixStream;

    use super::*;

    #[test]
    fn every_production_template_is_allowlisted_and_receiptable() {
        for template in [
            "dataset-profile",
            "feature-expansion",
            "training",
            "report-generation",
            "prediction",
            "export",
            "chat",
            "research",
            "managed-browser",
        ] {
            let parsed = parse_template(template).expect("allowlisted template");
            let receipt = handler_receipt(parsed, 0).expect("receipt");
            assert!(String::from_utf8(receipt).unwrap().contains("completed"));
        }
        assert!(parse_template("/bin/sh").is_err());
    }

    #[test]
    fn handler_arguments_are_bounded_and_shell_free() {
        assert!(validate_arguments(&["--run-id".into(), "run-1".into()]).is_ok());
        assert!(validate_arguments(&["".into()]).is_err());
        assert!(validate_arguments(&["bad\0arg".into()]).is_err());
    }

    #[test]
    fn inherited_control_frame_uses_bounded_length_prefix() {
        let (writer, mut reader) = UnixStream::pair().expect("socket pair");
        send_frame(writer.as_raw_fd(), &42_u32).expect("send frame");
        let mut length = [0_u8; 4];
        reader.read_exact(&mut length).expect("length");
        let mut payload = vec![0_u8; u32::from_be_bytes(length) as usize];
        reader.read_exact(&mut payload).expect("payload");
        assert_eq!(postcard::from_bytes::<u32>(&payload).unwrap(), 42);
    }

    #[test]
    fn invalid_control_descriptor_fails_closed() {
        assert!(parse_control_fd("-1").is_err());
        assert!(parse_control_fd("not-a-fd").is_err());
    }
}
