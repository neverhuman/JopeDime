use super::allowed_cpus;

#[test]
fn allowed_cpu_list_parses_ranges_and_refuses_malformed_lists() {
    let status = "Name:\tjope\nCpus_allowed:\tff\nCpus_allowed_list:\t0-3,8,10-11\n";
    assert_eq!(
        allowed_cpus(status)
            .unwrap()
            .into_iter()
            .collect::<Vec<_>>(),
        [0, 1, 2, 3, 8, 10, 11]
    );
    for bad in ["", "3-1", "1,,2", "a", "0-70000"] {
        assert!(
            allowed_cpus(&format!("Cpus_allowed_list:\t{bad}\n")).is_err(),
            "{bad}"
        );
    }
    assert!(allowed_cpus("Name:\tjope\n").is_err());
}
