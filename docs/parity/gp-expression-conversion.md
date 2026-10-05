# Nested expression conversion proof

This X0 test-only extension to the GP math-syntax workplan checks all 50 ordered pairs of legacy binary operators with a nested left or right operand. Readable source must preserve the AST and reach a canonical fixed point. Binary and lossless-text program envelopes must preserve executable bytes; compiled evaluation must preserve value bits and fault sets on six finite boundary rows.

Nine independent dyadic witnesses distinguish subtraction/division/power associativity, unary negation versus a negative base, and protected zero division. Each converted program executes a Rust, Python and C++17 train-and-predict recipe and must match the explicit expected bits. Missing compilers fail; no allowlist or tolerance is used. This does not qualify all operator compositions or nonfinite export inputs. Historical fixtures and runtime code are unchanged.

Acceptance: `JOPE_PARITY_STRICT=1 cargo nextest run --locked -p feat-math --test conversion_expression_matrix`.
