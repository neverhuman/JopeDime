fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

fn main() {
    println!("jope {}", version());
}

#[cfg(test)]
mod tests {
    use super::version;

    #[test]
    fn version_is_semver() {
        let parts: Vec<_> = version().split('.').collect();
        assert_eq!(parts.len(), 3, "expected major.minor.patch");
        assert!(parts
            .iter()
            .all(|part| part.chars().all(|c| c.is_ascii_digit())));
    }
}
