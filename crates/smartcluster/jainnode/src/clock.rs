use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) fn unix_time_ms() -> Result<u64, ()> {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| ())?;
    u64::try_from(elapsed.as_millis()).map_err(|_| ())
}
