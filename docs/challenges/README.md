# Engineering challenges

## Frontier regression challenge

[The master specification](frontier-regression-master-spec.md) defines a neutral challenge to produce one complete, dependency-free, safe Rust regressor through reasoning, then freeze it for Elo evaluation across thousands of unknown datasets. The final column of every complete dataset is always the target; prediction inputs contain features only.

The document includes the complete allowed-fundamentals inventory, the callable Rust interface, submission requirements, and an organizer-only red-team audit and evaluation appendix. It specifies a challenge; it does not implement an evaluator or report benchmark results.

### Issuing the prompt

1. Start a fresh participant session without repository context, existing implementations, dataset information, or prior performance results.
2. Copy only the text between `<!-- BEGIN MASTER PROMPT -->` and `<!-- END MASTER PROMPT -->`; exclude the markers and organizer appendix.
3. Settle any common resource, authoring, and scoring cards before collecting entries, then give every entrant the identical packet. The default authoring mode allows reasoning only; any compilation-only mode must be enabled uniformly.
4. Preserve the exact issued prompt bytes and Git commit ID. Freeze the participant's source before the organizer runs any numerical checks or hidden evaluation.

### Versioning

Maintain the specification here through normal Git commits and reviewed pull requests. Record the source commit and a hash of the exact participant packet for each campaign. Revisions to the interface, allowed capabilities, resources, or scoring start a new declared campaign; they do not retroactively change an issued challenge.
