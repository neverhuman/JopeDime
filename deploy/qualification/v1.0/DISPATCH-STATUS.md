# Checkout-bound CPU dispatcher

The installed `cargo-qualification-cpu` entry point selects `cpu_gates.py` from
`--repo`, or the current working directory when no repository is specified.
It must be a regular file under that checkout; missing or escaping controllers
refuse with exit2 before import. It never falls back to the installed launcher's
neighbor, which can belong to an older checkout. Use the checkout root for the
default, or pass the explicit root through `--repo`.

Cargo's leading `qualification-cpu` token is removed once. Remaining arguments,
including repository spelling, output and jobs, are forwarded unchanged; the
selected controller retains its own parsing, source checks, job caps and exits.
This routes the intended source; it does not authenticate an untrusted checkout.

Manual CPU-only acceptance (no Cargo build or matrix):

    python3 -B deploy/qualification/v1.0/test_cargo_qualification -v
    python3 -B deploy/qualification/v1.0/cargo-qualification-cpu --repo "$CHECKOUT" --help

The explicit executable test uses paired installed/selected sentinel controllers
to check default/explicit/Cargo/equals/abbreviated arguments, original exit,
forwarding, missing source and symlink refusals. It follows the local executable
harness packaging used for schema controls. It is not a maintained hosted-CI lane
or an expansion of the existing `.py` family census. Existing controller tests and
an actual zero-execution `plan` provide separate compatibility evidence.

All proof remains CPU preparation. No signed bytes, actual release gates, GPU,
qualification lock, Q1 decision or OWNER release approval are established here.
