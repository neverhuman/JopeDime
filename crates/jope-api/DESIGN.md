# API-01: schema skeleton before workspace registration

Implement spec §4.1–4.4, §8.4 and Appendix B as a transport-independent data
crate depending only on serde, serde_json and the existing schemars 1.2.2.
The four normative JSON examples anchor RunSpec, Plan, RunSnapshot and reasoning.
The remaining named types and summarized event bodies have explicit candidate
fields for the second COMPLEX reviewer and owner to assess before M0 freeze.
They are a schema proposal, not evidence of implemented engine capabilities.

Use one adjacent-tagged enum for durable kind/body pairs, flattened into the
public envelope. Represent telemetry with a separate envelope, so it cannot be
serialized through PublicEvent with a durable sequence. Nullable measurements
remain null; MetricValue admits the boolean catalog row. Preserve fractional
elapsed seconds separately from integer caps. IDs, counters and bytes remain
strings on the wire. The initial inline aliases reserve the API-02/03/04 module
names without editing their files; those tasks still supply validated IDs,
canonical DecimalU64 and the closed error catalog. See FOLLOW_UPS.md.

Reject importing engine/runtime types: that would pull numerical and storage
dependencies into every client and transport. Reject arbitrary JSON event
bodies: they cannot check the kind/body relationship. Objective JSON remains
opaque here because its evaluation grammar belongs to jope-runtime. Reject
adding root workspace members or CI scope in this PR: Jepson t526u expressly
assigns that subsequent step to API-06.

The t526u ruling permits the root jope-api lock entry and requires this PR first.
Run the exact root commands and disclose their pre-registration refusal. An
external proof manifest resolves the actual manifest's workspace inheritance,
references these exact source/test/example paths, and builds into this lane's
single target directory. Its reduced lock is checked against the root's pinned
package versions. This supplements the required commands; it does not make the
unregistered crate a workspace member or claim the root checks passed.

Risks: alias validation and finalized schemas remain pending API-02/03/04 plus
module integration. The engine must enforce timestamp formats, limits, source
excerpt length, state transitions and producer-side semantic constraints. Tests
exercise all event kinds, JSON preservation and schema drift; they do not claim
runtime, transport or release qualification. API-06 registration, two COMPLEX
reviewer passes and owner freeze approval remain required. Do not tag the
skeleton `v1-contract-frozen` or claim A2 closed.
