use super::*;

const LEDGER_OK: &str = r#"{"source_conventions": {"roots": {"LANE": "/x"}}, "claims": [
      {"id": "C01", "statement": "PolicyLM has 893,957 parameters", "value": "893,957", "tex": "893{,}957",
       "source": "LANE:a.json#.p", "sha256": "0000000000000000000000000000000000000000000000000000000000000000", "status": "established"},
      {"id": "C02", "statement": "Learning rate 3e-4, floor 1e-6; KL 0.004-0.011 over 8 epochs", "value": "0.0912345",
       "source": "LANE:b.json", "sha256": "1111111111111111111111111111111111111111111111111111111111111111", "status": "non-qualifying"},
      {"id": "C03", "statement": "E1 primary", "value": "pending", "source": "LANE:c.json",
       "sha256": "2222222222222222222222222222222222222222222222222222222222222222", "status": "pending"}
    ]}"#;

const WRITEUP_OK: &str = "\
# Title

## 1. Model

PolicyLM has 893,957 parameters [C01]. Section 4.4 says so (Chen et al., 2021).
The rate starts at $3\\times10^{-4}$ and stops at $10^{-6}$; KL stayed 0.004–0.011 (R23, v2) [C02].
| Arm | KL |
|---|---|
| anchored | 0.09 |

### 10.4 Pending (pre-registered or planned)

E1 is pre-registered [C03].

| Id | Result | Status |
|---|---|---|
| C03 | E1 primary | pending |

## Appendix G

`claims.json` holds 3 claims: 1 established, 0 preliminary, 1 non-qualifying and 1 pending.
Illustrative: q = log 10 ≈ 2.303.
";

const TRACE_OK: &str = "\
## Claim trace

| Id | Status | Printed | Source |
|---|---|---|---|
| C01 | established | 893,957 | LANE:a.json |
| C02 | non-qualifying | 0.004–0.011 | LANE:b.json |
| C03 | pending | — | LANE:c.json |

## Exempt numbers

| Text | Numbers | Why |
|---|---|---|
| `q = log 10 ≈ 2.303` | 10, 2.303 | illustrative arithmetic |
";

fn problems(writeup: &str, ledger: &str, trace: &str) -> Vec<String> {
    check_paper(writeup, ledger, trace).expect("parses").0
}

mod paper;
mod readme;
mod sources;
