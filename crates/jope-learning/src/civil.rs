//! Proleptic-Gregorian day arithmetic (Howard Hinnant's algorithms) so ledger paths and
//! RFC 3339 timestamps need no calendar crate.

/// Days since 1970-01-01 for a civil date.
pub fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (i64::from(m) + 9) % 12;
    let doy = (153 * mp + 2) / 5 + i64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// (year, month, day) for days since 1970-01-01.
pub fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// `(yyyy, mm)` of a unix timestamp in seconds, zero-padded for ledger paths.
pub fn year_month(ts_unix: u64) -> (String, String) {
    let (y, m, _) = civil_from_days((ts_unix / 86_400) as i64);
    (format!("{y:04}"), format!("{m:02}"))
}

/// Parse `YYYY-MM-DDTHH:MM:SS[.frac][Z|+HH:MM|-HH:MM]` (also a date-only `YYYY-MM-DD`) to unix
/// seconds. Returns `None` for anything else.
pub fn parse_rfc3339(s: &str) -> Option<u64> {
    let s = s.trim();
    let b = s.as_bytes();
    if b.len() < 10 || b[4] != b'-' || b[7] != b'-' {
        return None;
    }
    let num = |r: std::ops::Range<usize>| s.get(r).and_then(|x| x.parse::<i64>().ok());
    let (y, m, d) = (num(0..4)?, num(5..7)?, num(8..10)?);
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }
    let mut secs = days_from_civil(y, m as u32, d as u32) * 86_400;
    let mut rest = &s[10..];
    if let Some(t) = rest.strip_prefix(['T', 't', ' ']) {
        if t.len() < 8 || t.as_bytes()[2] != b':' || t.as_bytes()[5] != b':' {
            return None;
        }
        let tn = |r: std::ops::Range<usize>| t.get(r).and_then(|x| x.parse::<i64>().ok());
        let (hh, mm, ss) = (tn(0..2)?, tn(3..5)?, tn(6..8)?);
        if hh > 23 || mm > 59 || ss > 60 {
            return None;
        }
        secs += hh * 3600 + mm * 60 + ss;
        rest = &t[8..];
        if let Some(f) = rest.strip_prefix('.') {
            let digits = f.bytes().take_while(u8::is_ascii_digit).count();
            rest = &f[digits..];
        }
    }
    match rest {
        "" | "Z" | "z" => {}
        _ => {
            let sign = match rest.as_bytes()[0] {
                b'+' => 1,
                b'-' => -1,
                _ => return None,
            };
            let off = &rest[1..];
            if off.len() != 5 || off.as_bytes()[2] != b':' {
                return None;
            }
            let oh: i64 = off[0..2].parse().ok()?;
            let om: i64 = off[3..5].parse().ok()?;
            secs -= sign * (oh * 3600 + om * 60);
        }
    }
    u64::try_from(secs).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_dates() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        for days in [-1, 1, 59, 60, 365, 10_957, 19_723, 20_713, 60_000] {
            let (y, m, d) = civil_from_days(days);
            assert_eq!(days_from_civil(y, m, d), days);
        }
        assert_eq!(civil_from_days(20_713), (2026, 9, 17));
        assert_eq!(days_from_civil(2026, 9, 18), 20_714);
        assert_eq!(year_month(1_789_600_000), ("2026".into(), "09".into()));
    }

    #[test]
    fn parses_rfc3339() {
        assert_eq!(parse_rfc3339("1970-01-01T00:00:00Z"), Some(0));
        assert_eq!(
            parse_rfc3339("2026-06-30T00:40:20+00:00"),
            Some(1_782_780_020)
        );
        assert_eq!(
            parse_rfc3339("2026-06-30T02:40:20+02:00"),
            Some(1_782_780_020)
        );
        assert_eq!(
            parse_rfc3339("2026-06-30T00:40:20.123456Z"),
            Some(1_782_780_020)
        );
        assert_eq!(parse_rfc3339("2026-06-30"), Some(1_782_777_600));
        assert_eq!(parse_rfc3339("yesterday"), None);
        assert_eq!(parse_rfc3339("2026-13-01"), None);
    }
}
