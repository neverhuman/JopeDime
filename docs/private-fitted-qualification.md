# Private fitted winner qualification

`benchmark report` keeps every original loss and selects the smallest complete five-fold
mean loss, with method identity breaking ties. Its `fitted_inference` section refers to
that exact selected recipe and each existing fitted fold. Unsupported backends never
change the winner, fit another model or remove its native artifact. Incomplete confirmation
has no final winner. Static method inventory is not executed parity.

Use the same immutable installed CLI, approved plan, manifest, development root and journal
used for the campaign. These commands open no sealed data:

```sh
jope benchmark export-winner --plan "$PLAN" --plan-sha256 "$PLAN_ID" \
  --manifest "$MANIFEST" --datasets "$DATASETS" --journal "$JOURNAL" \
  --dataset "$DATASET" --fold 0
jope benchmark qualify-export --prepared "$PREPARED" \
  --prepared-sha256 "$PREPARED_ID" --tools "$TOOLS" --tools-sha256 "$TOOLS_FILE_SHA256"
jope benchmark report --plan "$PLAN" --plan-sha256 "$PLAN_ID" \
  --manifest "$MANIFEST" --datasets "$DATASETS" --journal "$JOURNAL"
```

The first command returns the semantic `prepared_sha256`, private `prepared.json` path and
`public_bundle_root`. The latter contains only each validated public model bundle and its
public manifest. Do not publish the private prepared plan or worker directory: they also
contain references and unlabeled development features. A selected external winner retains
its checksum-verified native pickle and successful reload receipt. Standalone Python,
Rust, C++ and JavaScript export of that library state is explicitly unsupported.

`TOOLS` is a private JSON map using the existing campaign `Runtime` schema. Include each
command name in the prepared plans, normally `python3`, `rustc`, `g++` and `node`. Each value
pins the physical executable (`program.path`, `program.sha256`), exact observed version,
empty `argument_prefix`, complete toolchain/runtime dependency artifacts in `inputs` and
`package_files`, admitted environment, and `unavailable: null`. Use the actual toolchain
`rustc`, rather than a rustup selector. Missing tools or explicit `unavailable` reasons remain
reported as unavailable and cannot qualify. Tool inventory SHA is the hash of its file bytes;
prepared and campaign IDs are their reported semantic identities. Prepare and qualify with
matching C++ environment settings; changed emitted commands fail closed.

Qualification runs Python, Rust, the same C++ emitter under C++17 then C++20, and JavaScript.
Each actual backend must qualify, or be explicitly unsupported, before the next executes.
Compiler absence, timeout, crash or failed parity blocks subsequent execution. Each job has
one thread and at most the original 30-second fit limit; compilers and their descendants run
inside the same existing CPU/memory/swap supervisor and worker cgroup. The original campaign
window and reporting reserve are unchanged. An expired campaign cannot be granted a new
qualification allowance by repeating the command. A distinct post-panel validation stage
is not implemented by this front door.

The private child receives generated model code and unlabeled raw prediction rows. Expected
predictions and scoring labels stay in the controller. Report verification checks the exact
native fitted artifact, observed commands, tool inventory, original row/class identities,
generated source and outputs, original execution timing and cleanup, and the predeclared
numerical policy. Python/Rust/C++ use exact bits; JavaScript uses the existing 1e-12 absolute
and relative policy. Report recovery is read-only and launches no compiler or predictor.

`all_backends_qualified` is true only if all five executed qualifications pass. Operation
`ok` may also be true when unsupported outcomes are fully inventoried; it is not a claim
that an unsupported backend executed. Started work without a finished receipt has unknown
cost and status. Failed numerical verification remains `failed_parity`, not qualified.

For an ASM campaign, `fitted_export_commands` and per-backend CPU/wall are subsets of global
worker-command accounting; do not add them twice. Comparator-only fit accounting predates
these commands, so add their separate costs when reporting creation cost. Export preparation
has separately measured wall/process CPU through public materialization and verification and
is additional to worker counters. Final receipt publication remains outside that interval. Controller,
setup and repeat-invocation overhead without retained measurements remains unknown. Never
present worker subtotals as the complete creation cost or post-deadline exports as completed
inside the panel's two-hour allowance.
