# E6 single-cut generation coordinator

This partial E6 prerequisite connects the shipped revision-three profile wrappers
with the existing traced evaluator and guarded training-only refinement. It is
required before shipping profile selection through `jope pipeline` and before B05.

`kernel2::search::Options::new` resolves only `shipped_kernel2_profiles`, records
the profile digest and validates a Hard-refusing fault policy. The caller provides
explicit top-k and trial limits. `run_cut` runs on the existing evaluation stack;
this slice supports one cut, one thread and one island. Unsupported concurrency
is refused. The legacy search functions, defaults and seeded streams are unchanged.

Births and offspring use E3 `random_program`, `mutate_program` and
`crossover_programs`; E0 admission owns each child's scope/fixed point. Parents
are unique canonical genomes in E4 fault-aware rank order. Every attempted
measurement, including refusal, and every refinement objective spends the same
cut allowance. The generation hook uses only training rows for optimization,
reserves child measurement, and checks cancellation/deadline before new work.
Started fits are nonpreemptive; bootstrap and incumbent measurements preserve
the existing seconds exemption. Refinement reports retain unchanged or failed
trials and any child remeasurement rather than hiding their budget cost.

The returned record binds profile/policy/budget, arms drawn (including retries),
measurements and objective/refinement counts. It is not a published v4 receipt.
Legacy v1–3 receipt, observation and emitter bytes remain unchanged; no frozen
identity input or fixture is edited or regenerated.

Acceptance: `cargo test --locked -p jope-pipeline --lib kernel2::search::tests`.
Eleven default, nonignored cases cover profile/seed/config refusal, a real cut
search through E3 births and offspring, all four expression arm draws, canonical
children, fault-aware parent order, shared trial/child accounting, deterministic
replay, expired deadlines, cancellation, Hard refusals and poisoned holdouts.

Remaining E6 gates: multiple-cut comparison, concurrent/island coordination,
CLI `--search-profile` / `--fault-policy`, opt-in v4 receipt publication and
re-derivation, and kernel2 fitted-C++ defaults. EEC clause(b), equal-budget
qualification and B05 admission are still pending. This receipt makes no live
learning-loop, promotion, full E6 closure or release qualification claim.
