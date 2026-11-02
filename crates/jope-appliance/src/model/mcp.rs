use super::*;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct IdArgs {
    id: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PredictArgs {
    id: String,
    csv: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ImportArgs {
    manifest: String,
    parts_hex: BTreeMap<String, String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SessionName {
    name: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SessionRecord {
    name: String,
    id: String,
    expected_head: String,
    #[serde(default)]
    label: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SessionRollback {
    name: String,
    version: u64,
    expected_head: String,
    #[serde(default)]
    label: String,
}
fn arguments<T: serde::de::DeserializeOwned>(value: Value) -> anyhow::Result<T> {
    serde_json::from_value(value).map_err(|e| Failure::input(e.to_string()))
}
fn call_tool(root: &Path, name: &str, args: Value) -> anyhow::Result<Value> {
    match name {
        "model_session_show" => {
            let args: SessionName = arguments(args)?;
            Ok(json!(sessions::read(root, &args.name)?))
        }
        "model_session_record" => {
            let args: SessionRecord = arguments(args)?;
            Ok(json!(sessions::append(
                root,
                &args.name,
                &args.expected_head,
                Some(&args.id),
                None,
                &args.label,
                MAX_FRAME as u64
            )?))
        }
        "model_session_rollback" => {
            let args: SessionRollback = arguments(args)?;
            Ok(json!(sessions::append(
                root,
                &args.name,
                &args.expected_head,
                None,
                Some(args.version),
                &args.label,
                MAX_FRAME as u64
            )?))
        }
        "model_list" => {
            if args != json!({}) {
                return Err(Failure::input("model_list accepts no arguments"));
            }
            list(root)
        }
        "model_import" => {
            let args: ImportArgs = arguments(args)?;
            let parts = args
                .parts_hex
                .into_iter()
                .map(|(k, v)| Ok((k, decode_hex(&v)?)))
                .collect::<anyhow::Result<_>>()?;
            let object = ModelObject::decode(args.manifest.into_bytes(), parts, MAX_FRAME as u64)?;
            let path = object.save(root, MAX_FRAME as u64)?;
            Ok(json!({"path":path,"inspection":object.inspect()?}))
        }
        "model_predict" => {
            let args: PredictArgs = arguments(args)?;
            if args.csv.len() > 1 << 20 {
                return Err(Failure::input("prediction CSV exceeds 1 MiB"));
            }
            let object = ModelObject::read(&object_path(root, &args.id)?, MAX_FRAME as u64)?;
            let limits = Limits {
                max_input_bytes: 1 << 20,
                max_rows: 10_000,
                ..Limits::default()
            };
            let budget = jope_runtime::budget::Budget::new(None, 1);
            // A Platt-calibrated object predicts whole; `fitted()` refuses to drop its map.
            let predictions = if let Some(contract) = object.calibrated_input()? {
                let csv = jope_runtime::table::parse_csv(args.csv.as_bytes(), &limits)?;
                let (rows, _) = contract.table(&csv)?;
                jope_runtime::api::predict_object(&object, &rows, &budget)?
            } else {
                let fitted = object.fitted()?;
                let csv = jope_runtime::table::parse_csv(args.csv.as_bytes(), &limits)?;
                let (rows, _) = jope_runtime::table::prediction_ignoring(
                    &csv,
                    fitted.schema(),
                    &fitted.header.target,
                    &fitted.header.auxiliary_columns,
                )?;
                jope_runtime::api::predict(&fitted, &rows, &budget)?
            };
            Ok(
                json!({"id":args.id,"predictions_csv":String::from_utf8(jope_runtime::api::predictions_csv(&predictions)?)?}),
            )
        }
        "model_inspect" | "model_verify" | "model_export" => {
            let args: IdArgs = arguments(args)?;
            let object = ModelObject::read(&object_path(root, &args.id)?, MAX_FRAME as u64)?;
            if name == "model_verify" {
                object.verify_executable()?;
            }
            if name == "model_export" {
                let parts: BTreeMap<_, _> = object
                    .parts()
                    .iter()
                    .map(|(k, v)| (k, encode_hex(v)))
                    .collect();
                Ok(
                    json!({"id":object.id(),"manifest":String::from_utf8(object.manifest_bytes().to_vec())?,"parts_hex":parts}),
                )
            } else {
                Ok(json!(object.inspect()?))
            }
        }
        _ => Err(Failure::input("unknown model tool")),
    }
}

/// Call the existing model-session handler from the shared MCP adapter.
#[cfg(feature = "mcp")]
pub(crate) fn call_mcp_legacy_tool(root: &Path, name: &str, args: Value) -> anyhow::Result<Value> {
    call_tool(root, name, args)
}

pub(super) fn serve(root: &Path) -> anyhow::Result<()> {
    let mut input = std::io::stdin().lock();
    let mut output = std::io::stdout().lock();
    loop {
        let mut line = Vec::new();
        let count = (&mut input)
            .take(MAX_FRAME as u64 + 1)
            .read_until(b'\n', &mut line)
            .map_err(|e| Failure::input(e.to_string()))?;
        if count == 0 {
            break;
        }
        if line.len() > MAX_FRAME {
            return Err(Failure::input("MCP frame exceeds 8 MiB"));
        }
        let request: Value = match jope_runtime::strict_json::decode(&line) {
            Ok(request) => request,
            Err(_) => {
                writeln!(
                    output,
                    "{}",
                    json!({"jsonrpc":"2.0","id":null,"error":{"code":-32700,"message":"parse error"}})
                )?;
                output.flush()?;
                continue;
            }
        };
        let id = request.get("id").cloned().unwrap_or(Value::Null);
        if request.get("jsonrpc") != Some(&json!("2.0"))
            || !request.get("method").is_some_and(Value::is_string)
            || !(id.is_null() || id.is_string() || id.is_number())
        {
            writeln!(
                output,
                "{}",
                json!({"jsonrpc":"2.0","id":null,"error":{"code":-32600,"message":"invalid request"}})
            )?;
            output.flush()?;
            continue;
        }
        let method = request["method"].as_str().unwrap_or_default();
        // Notifications never execute tools or write objects.
        if request.get("id").is_none() {
            continue;
        }
        let result = match method {
            "initialize" => Ok(
                json!({"protocolVersion":"2025-11-25","capabilities":{"tools":{"listChanged":false}},"serverInfo":{"name":"jopedime-models","version":env!("CARGO_PKG_VERSION")}}),
            ),
            "ping" => Ok(json!({})),
            "tools/list" => Ok(tools_list()),
            "tools/call" => {
                let params = &request["params"];
                let name = params["name"].as_str().unwrap_or_default();
                let args = params.get("arguments").cloned().unwrap_or(json!({}));
                let response = call_tool(root, name, args);
                Ok(match response {
                    Ok(value) => {
                        json!({"content":[{"type":"text","text":value.to_string()}],"structuredContent":value,"isError":false})
                    }
                    Err(e) => {
                        json!({"content":[{"type":"text","text":format!("{e:#}")}],"isError":true,
                        "structuredContent":{"exit_code":jope_runtime::error::kind_of(&e).exit_code()}})
                    }
                })
            }
            _ => Err((-32601, "method not found")),
        };
        let response = match result {
            Ok(result) => json!({"jsonrpc":"2.0","id":id,"result":result}),
            Err((code, message)) => {
                json!({"jsonrpc":"2.0","id":id,"error":{"code":code,"message":message}})
            }
        };
        if response.to_string().len() > MAX_FRAME {
            writeln!(
                output,
                "{}",
                json!({"jsonrpc":"2.0","id":id,"error":{"code":-32000,"message":"response exceeds 8 MiB; use the CLI for large objects"}})
            )?;
        } else {
            writeln!(output, "{response}")?;
        }
        output
            .flush()
            .map_err(|e| Failure::new(FailureKind::Output, e.to_string()))?;
    }
    Ok(())
}
