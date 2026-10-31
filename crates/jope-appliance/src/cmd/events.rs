use clap::Args;
use jope_api::{decimal::DecimalU64, ids::RunId};
use jope_client::Transport;
use serde_json::Value;
use std::{io::Write, str::FromStr};

#[derive(Args, Debug)]
pub struct EventsArgs {
    pub run: String,
    #[arg(long)]
    pub after: Option<String>,
    #[arg(long)]
    pub follow: bool,
    #[arg(long = "kinds", value_delimiter = ',')]
    pub kinds: Vec<String>,
}

pub fn run<T: Transport>(args: EventsArgs, transport: &T) -> anyhow::Result<()> {
    let run = RunId::from_str(&args.run).map_err(|_| anyhow::anyhow!("invalid run identifier"))?;
    let after = args
        .after
        .as_deref()
        .map(|value| serde_json::from_value::<DecimalU64>(Value::String(value.into())))
        .transpose()?;
    let mut stdout = std::io::stdout().lock();
    if args.follow {
        for event in transport.follow_events(run, after)? {
            let value = match event? {
                jope_client::StreamEvent::Durable(value) => serde_json::to_value(value)?,
                jope_client::StreamEvent::Telemetry(value) => serde_json::to_value(value)?,
            };
            if selected(&value, &args.kinds) {
                stdout.write_all(encode_event(&value)?.as_bytes())?;
            }
        }
    } else {
        let mut cursor = after.unwrap_or(DecimalU64(0));
        loop {
            let page = serde_json::to_value(transport.run_events(&run, cursor, 256)?)?;
            let events = page["events"].as_array().cloned().unwrap_or_default();
            if events.is_empty() {
                break;
            }
            for value in &events {
                if selected(value, &args.kinds) {
                    stdout.write_all(encode_event(value)?.as_bytes())?;
                }
            }
            let next: DecimalU64 = serde_json::from_value(page["next_after"].clone())?;
            if next <= cursor || events.len() < 256 {
                break;
            }
            cursor = next;
        }
    }
    Ok(())
}

fn selected(value: &Value, kinds: &[String]) -> bool {
    kinds.is_empty()
        || value["kind"]
            .as_str()
            .is_some_and(|kind| kinds.iter().any(|filter| filter == kind))
}

fn encode_event(value: &Value) -> anyhow::Result<String> {
    Ok(format!("{}\n", serde_json::to_string(value)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn kind_filter_and_ndjson_lines_are_stable() {
        let values = [
            json!({"kind":"reasoning","body":{}}),
            json!({"kind":"warning","body":{}}),
        ];
        let selected: Vec<_> = values
            .iter()
            .filter(|value| selected(value, &["warning".into()]))
            .cloned()
            .collect();
        assert_eq!(selected, [json!({"kind":"warning","body":{}})]);
        assert_eq!(
            encode_event(&selected[0]).unwrap(),
            "{\"body\":{},\"kind\":\"warning\"}\n"
        );
    }

    #[cfg(feature = "engine")]
    #[cfg(feature = "engine")]
    #[test]
    fn events_use_the_typed_in_process_api_contract() {
        use jope_client::{InProcess, Transport};
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/int01");
        std::fs::create_dir_all(&root).unwrap();
        let directory = tempfile::tempdir_in(root).unwrap();
        let engine = std::sync::Arc::new(
            jope_service::engine::InProcessEngine::open(directory.path()).unwrap(),
        );
        let client = InProcess::with_engine(engine, 43123, &"ab".repeat(32)).unwrap();
        let run = RunId::from_str("run_01J00000000000000000000000").unwrap();
        assert!(matches!(
            client.run_events(&run, DecimalU64(0), 256),
            Err(jope_client::Error::Problem(problem))
                if problem.code == jope_api::error::ErrorCode::NotFound
        ));
    }
}
