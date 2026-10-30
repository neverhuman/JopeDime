//! Hidden process-isolation entry. Concrete discovery evaluation is supplied by DSC-08.

pub fn run(protocol: u32) -> anyhow::Result<()> {
    // Preserve the appliance's authenticated-package boundary before processing any candidate.
    let _package = crate::bootstrap::open()?;
    jope_service::worker::serve(protocol, |_request, _matrix| {
        // Never manufacture a score while the discovery/native evaluator adapter is absent.
        Err(jope_service::worker::Failure::Unavailable)
    })?;
    Ok(())
}
