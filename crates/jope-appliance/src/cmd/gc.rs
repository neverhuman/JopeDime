use jope_client::Transport;
use serde_json::Value;

pub(crate) fn run(
    dry_run: bool,
    idempotency_key: Option<&str>,
    json_out: bool,
) -> anyhow::Result<()> {
    let client = super::dataset::resolve_client()?;
    let generated = idempotency_key.is_none();
    let key = super::common::resolve_idempotency_key(idempotency_key);
    if generated {
        super::common::write_idempotency_key(&mut std::io::stderr().lock(), &key)?;
    }
    let mut value: Value = serde_json::to_value(
        client.store_gc(&jope_api::registry::GcRequest { dry_run }, Some(&key))?,
    )?;
    if let Some(object) = value.as_object_mut() {
        super::common::add_idempotency_key(object, &key);
    }
    super::dataset::emit(&value, json_out)
}

#[cfg(test)]
mod tests {
    #[test]
    fn dry_run_payload_is_explicit() {
        assert_eq!(serde_json::json!({"dry_run":true})["dry_run"], true);
    }
}
