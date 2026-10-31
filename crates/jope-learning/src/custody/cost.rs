//! Checked decimal accounting of captured bytes/requests, never an invoice or live price lookup.
use super::unsigned;
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::collections::BTreeMap;

#[derive(Clone, Copy)]
struct Decimal {
    coefficient: u128,
    scale: i32,
}

impl Decimal {
    fn parse(value: &Value) -> Result<Self> {
        let text = value.as_str().context("decimal price string")?;
        let (base, exponent) = match text.split_once(['e', 'E']) {
            Some((b, e)) => (b, e.parse::<i32>().context("positive finite price")?),
            None => (text, 0),
        };
        let mut digits = String::new();
        let mut dot = false;
        let mut scale = 0i32;
        for byte in base.bytes() {
            if byte == b'.' {
                ensure!(!dot, "positive finite price");
                dot = true;
            } else {
                ensure!(byte.is_ascii_digit(), "positive finite price");
                digits.push(char::from(byte));
                if dot {
                    scale = scale.checked_add(1).context("decimal overflow")?;
                }
            }
        }
        let coefficient = digits.parse::<u128>().context("positive finite price")?;
        scale = scale.checked_sub(exponent).context("decimal overflow")?;
        ensure!(
            coefficient > 0 && (-28..=28).contains(&scale),
            "positive finite price"
        );
        Ok(Self { coefficient, scale })
    }
    fn rounded(mut self) -> Result<Self> {
        let n = self.coefficient.to_string().len();
        if n > 28 {
            let power = 10u128
                .checked_pow((n - 28) as u32)
                .context("decimal overflow")?;
            let rest = self.coefficient % power;
            self.coefficient /= power;
            if rest > power / 2 || (rest == power / 2 && self.coefficient % 2 == 1) {
                self.coefficient += 1;
            }
            self.scale -= (n - 28) as i32;
            if self.coefficient.to_string().len() > 28 {
                self.coefficient /= 10;
                self.scale -= 1;
            }
        }
        Ok(self)
    }
    fn multiply(self, other: Self) -> Result<Self> {
        Self {
            coefficient: self
                .coefficient
                .checked_mul(other.coefficient)
                .context("decimal intermediate overflow")?,
            scale: self
                .scale
                .checked_add(other.scale)
                .context("decimal overflow")?,
        }
        .rounded()
    }
    fn divide(mut self, divisor: u128) -> Result<Self> {
        if self.coefficient == 0 || self.coefficient.is_multiple_of(divisor) {
            self.coefficient /= divisor;
            return Ok(self);
        }
        while (self.coefficient / divisor).to_string().len() < 28 {
            self.coefficient = self
                .coefficient
                .checked_mul(10)
                .context("decimal intermediate overflow")?;
            self.scale = self.scale.checked_add(1).context("decimal overflow")?;
        }
        let remainder = self.coefficient % divisor;
        self.coefficient /= divisor;
        if remainder > divisor / 2 || (remainder * 2 == divisor && self.coefficient % 2 == 1) {
            self.coefficient += 1;
        }
        self.rounded()
    }
    fn add(self, other: Self) -> Result<Self> {
        let scale = self.scale.max(other.scale);
        let a = self
            .coefficient
            .checked_mul(
                10u128
                    .checked_pow((scale - self.scale) as u32)
                    .context("decimal overflow")?,
            )
            .context("decimal intermediate overflow")?;
        let b = other
            .coefficient
            .checked_mul(
                10u128
                    .checked_pow((scale - other.scale) as u32)
                    .context("decimal overflow")?,
            )
            .context("decimal intermediate overflow")?;
        Self {
            coefficient: a.checked_add(b).context("decimal intermediate overflow")?,
            scale,
        }
        .rounded()
    }
    fn render(self) -> String {
        if self.coefficient == 0 && self.scale > 6 {
            return format!("0E-{}", self.scale);
        }
        let digits = self.coefficient.to_string();
        let exponent = digits.len() as i32 - 1 - self.scale;
        if exponent < -6 || self.scale < 0 {
            let tail = if digits.len() > 1 {
                format!(".{}", &digits[1..])
            } else {
                String::new()
            };
            return format!("{}{tail}E{exponent:+}", &digits[..1]);
        }
        if self.scale <= 0 {
            return format!("{digits}{}", "0".repeat((-self.scale) as usize));
        }
        let scale = self.scale as usize;
        if digits.len() > scale {
            let at = digits.len() - scale;
            format!("{}.{}", &digits[..at], &digits[at..])
        } else {
            format!("0.{}{digits}", "0".repeat(scale - digits.len()))
        }
    }
}

/// Performs the report operation and returns an error if it cannot be completed.
pub fn report(rows: &[Value], requests: &[Value], rates: &Value, as_of: &str) -> Result<Value> {
    ensure!(
        rates["region"] == "us-east-2"
            && unsigned(&rates["storage_bytes_per_gb"], "GB bytes")? == 1 << 30,
        "Ohio binary-GB rate card"
    );
    let storage_price = Decimal::parse(&rates["standard_gb_month"])?;
    let list_price = Decimal::parse(&rates["list_request"])?;
    let get_price = Decimal::parse(&rates["get_request"])?;
    ensure!(
        as_of.is_ascii()
            && as_of.len() >= 10
            && as_of.as_bytes()[4] == b'-'
            && as_of.as_bytes()[7] == b'-',
        "UTC calendar date"
    );
    ensure!(
        as_of.ends_with('Z') || as_of.ends_with("+00:00"),
        "UTC calendar date"
    );
    let year: u32 = as_of[..4].parse()?;
    let month: u32 = as_of[5..7].parse()?;
    let days = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400)) => {
            29
        }
        2 => 28,
        _ => anyhow::bail!("UTC calendar month"),
    };
    ensure!(
        year > 0
            && as_of[8..10]
                .parse::<u32>()
                .is_ok_and(|d| d > 0 && d <= days),
        "UTC calendar date"
    );
    let mut standard = 0u64;
    let mut unsupported = BTreeMap::<String, u64>::new();
    for row in rows {
        let class = row["storage_class"].as_str().context("storage class")?;
        if class == "STANDARD" {
            standard = standard
                .checked_add(unsigned(&row["bytes"], "object bytes")?)
                .context("storage byte overflow")?;
        } else {
            *unsupported.entry(class.into()).or_default() += 1;
        }
    }
    let mut counts = BTreeMap::<String, u64>::new();
    for request in requests {
        let operation = request["operation"]
            .as_str()
            .context("read-only request roster")?;
        ensure!(
            [
                "ListObjectVersions",
                "ListObjectsV2",
                "GetObject",
                "GetBucketVersioning"
            ]
            .contains(&operation),
            "read-only request roster"
        );
        *counts.entry(operation.into()).or_default() += 1;
    }
    let list = counts
        .get("ListObjectVersions")
        .copied()
        .unwrap_or_default()
        + counts.get("ListObjectsV2").copied().unwrap_or_default();
    let get = counts.get("GetObject").copied().unwrap_or_default()
        + counts
            .get("GetBucketVersioning")
            .copied()
            .unwrap_or_default();
    let storage = Decimal {
        coefficient: standard.into(),
        scale: 0,
    }
    .divide(1 << 30)?
    .multiply(storage_price)?
    .divide(days.into())?;
    let cost = list_price
        .multiply(Decimal {
            coefficient: list.into(),
            scale: 0,
        })?
        .add(get_price.multiply(Decimal {
            coefficient: get.into(),
            scale: 0,
        })?)?;
    Ok(
        json!({"standard_version_bytes":standard,"standard_storage_usd_day_estimate":storage.render(),
        "request_usd_estimate":cost.render(),"http_requests":counts,"days_in_utc_month":days,
        "storage_rate_assumption":"first 50 TiB tier; calendar-month daily normalization",
        "unsupported_storage_classes":unsupported,"all_storage_priced":unsupported.is_empty(),
        "data_transfer_usd":null,"actual_billed_usd":null,"discounts_free_tier_account_aggregate_tier":"unmeasured","qualified":false}),
    )
}
