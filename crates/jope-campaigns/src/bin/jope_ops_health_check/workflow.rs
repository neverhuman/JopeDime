use super::*;

pub(crate) fn run() {
    let args = Args::parse();
    let receipt_path = args.root.join("ops/ops-health-check.v1.json");
    let (receipt, failing) = match evaluate(&args) {
        Ok(outcome) => (outcome.receipt, outcome.failing),
        Err(error) => {
            // Even our own failure keeps the schema. This receipt is a contract.
            let receipt = json!({
                "schema": RECEIPT_SCHEMA,
                "generated_at_ms": now_ms(),
                "status": "check_error",
                "fail_after_cycles": args.fail_after_cycles,
                "live_window_seconds": args.live_window_seconds,
                "unhealthy_count": 0,
                "unhealthy": [],
                "error": format!("{error:#}"),
            });
            (receipt, true)
        }
    };
    let _ = atomic_json(&receipt_path, &receipt);
    println!("{}", serde_json::to_string(&receipt).unwrap_or_default());

    if failing {
        for row in receipt["unhealthy"].as_array().into_iter().flatten() {
            if row["over_budget"] == Value::Bool(true) {
                eprintln!(
                    "FAILING {} [{}] {} cycles / {}s: {}",
                    row["file"].as_str().unwrap_or("?"),
                    row["kind"].as_str().unwrap_or("?"),
                    row["consecutive_cycles"],
                    row["failing_for_seconds"],
                    row["error"].as_str().unwrap_or(""),
                );
            }
        }
        if let Some(error) = receipt["error"].as_str() {
            eprintln!("CHECK ERROR: {error}");
        }
        if !args.audit {
            // The whole point: systemd now shows this unit as failed.
            std::process::exit(1);
        }
    }
}
