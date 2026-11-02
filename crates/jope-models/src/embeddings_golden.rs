//! Frozen golden dataset IDs used by the hybrid embedding study.

/// The ordered 42-dataset cohort (`GOLDEN42`) used by the Python embedding
/// implementation. Preserve order because consumers emit rows in this order.
pub const GOLDEN42: [&str; 42] = [
    "019de5d76a718163",
    "047e62b52a3f4ede",
    "13b5df533e59a626",
    "17a4542ee4fb9e44",
    "1a803b27c2400dc4",
    "1cb97b1d7314d43d",
    "21fbfe8b53dbcd60",
    "24713c7f7bfa7807",
    "30ef3eaa6d2971a9",
    "3305a3ad973fb227",
    "3baa7e37fc1f7861",
    "3f781a51af45387e",
    "43b4cc3ab0ebbd97",
    "471f11cc2d38853a",
    "4f68d277d770477f",
    "542e75bc0cfdd1bf",
    "55b144de19659467",
    "55f06d147bcc2a32",
    "5f91a00bd3b66742",
    "693890aae9dc860b",
    "79e600bc5fcd5da7",
    "7af85a04de0c7018",
    "860e5558f732f638",
    "ab0239b9ef0ba28e",
    "aebf5d861a8365b4",
    "b1f909fcfceeef4b",
    "b253fbfae8460dc1",
    "b5beff4a53fa12c0",
    "b78d323566930bd7",
    "bd49e740a79bcb6c",
    "c32fa09b48fb2f27",
    "c7d815714ca52bce",
    "c8f90fc9cc5364be",
    "cf5e74b4e9594944",
    "d2daf53412ab18dc",
    "dee1fc8e9afd81c9",
    "e1c1db5260d0894e",
    "e5a3a7b3668a1cc2",
    "ee56a1015e344c11",
    "f0e7eb3d790df37d",
    "f8536dfeef471c2f",
    "fe3cc716a05d036f",
];

#[cfg(test)]
mod tests {
    use super::GOLDEN42;
    use std::collections::BTreeSet;

    #[test]
    fn golden42_matches_the_ordered_unique_cohort_contract() {
        assert_eq!(GOLDEN42.len(), 42);
        assert_eq!(GOLDEN42[0], "019de5d76a718163");
        assert_eq!(GOLDEN42[23], "ab0239b9ef0ba28e");
        assert_eq!(GOLDEN42[41], "fe3cc716a05d036f");
        assert!(GOLDEN42.windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(GOLDEN42.iter().collect::<BTreeSet<_>>().len(), 42);
    }
}
