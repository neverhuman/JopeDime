use clap::Args;
use jope_api::{ids::RunId, RunSnapshot};
use jope_client::{transport::Request, Transport};
use std::{io::Write, str::FromStr};

#[derive(Args, Debug)]
pub struct StatusArgs {
    pub run: String,
    #[arg(long)]
    pub wait: Option<u64>,
    #[arg(long)]
    pub after_event: Option<String>,
    #[arg(long)]
    pub json: bool,
}

pub fn run<T: Transport>(args: StatusArgs, transport: &T) -> anyhow::Result<()> {
    let run = RunId::from_str(&args.run).map_err(|_| anyhow::anyhow!("invalid run identifier"))?;
    let snapshot: RunSnapshot = match (args.wait, args.after_event.as_deref()) {
        (Some(seconds), after_event) => {
            let after = after_event.unwrap_or("0");
            let request = Request::new(7, 1, &[("id", &run.to_string())])?.query(&[
                ("wait_seconds", &seconds.to_string()),
                ("after_event", after),
            ]);
            transport.execute(request)?.json()?
        }
        (None, Some(after)) => {
            let request = Request::new(7, 1, &[("id", &run.to_string())])?
                .query(&[("wait_seconds", "0"), ("after_event", after)]);
            transport.execute(request)?.json()?
        }
        (None, None) => transport.run_status(&run)?,
    };
    let stdout = std::io::stdout();
    let mut stdout = stdout.lock();
    stdout.write_all(encode_snapshot(&snapshot)?.as_bytes())?;
    Ok(())
}

fn encode_snapshot(value: &impl serde::Serialize) -> anyhow::Result<String> {
    Ok(format!("{}\n", serde_json::to_string(value)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use jope_api::decimal::DecimalU64;
    use serde_json::Value;
    #[test]
    fn status_flags_keep_decimal_cursor_exact() {
        let after = "9007199254740993";
        let value = serde_json::from_value::<DecimalU64>(Value::String(after.into())).unwrap();
        assert_eq!(value.0, 9_007_199_254_740_993);
        assert_eq!(value.0.to_string(), after);
    }

    #[test]
    fn snapshot_uses_a_single_golden_json_line() {
        let value =
            serde_json::json!({"run_id":"run_01J00000000000000000000000","state":"running"});
        assert_eq!(
            encode_snapshot(&value).unwrap(),
            "{\"run_id\":\"run_01J00000000000000000000000\",\"state\":\"running\"}\n"
        );
    }

    #[cfg(feature = "engine")]
    #[test]
    fn status_uses_the_typed_in_process_api_contract() {
        use jope_client::{InProcess, Transport};
        let client = InProcess::local(43123, &"ab".repeat(32)).unwrap();
        let run = RunId::from_str("run_01J00000000000000000000000").unwrap();
        assert!(matches!(
            client.run_status(&run),
            Err(jope_client::Error::Problem(problem))
                if problem.code == jope_api::error::ErrorCode::CapabilityUnavailable
        ));
    }
}
