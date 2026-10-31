# Jain worker launcher

`jain-worker` is the private worker process started by the Smartcluster daemon. It validates a closed workload-template name, validates a bounded argument vector and inherited environment, then sends fenced progress and an application receipt over the daemon-provided control socket.

## Current behavior

The template names classify the requested workload. This binary does **not** execute dataset profiling, training, prediction, export, browser, JOPE, or other named workload logic. It does not invoke a shell or an arbitrary executable, and it does not use a request-selected working directory. Its `jain.handler.receipt/v1` payload records the selected template and argument count. The receipt's `status: "completed"` means this launcher completed its receipt-emission path; it is not evidence that the named workload ran or produced an artifact.

The accepted template names are `dataset-profile`, `feature-expansion`, `training`, `report-generation`, `prediction`, `export`, `chat`, `research`, `managed-browser`, `jope-encode`, `jope-propose`, `lime-run`, and `lime-batch`.

## Daemon interface

The first command-line argument is the template name. Remaining arguments must be nonempty, contain no NUL byte, and be at most 16 KiB each; at most 256 are accepted. The daemon passes a connected control socket descriptor in `SCQ_CONTROL_FD` and the attempt identity in `SCQ_JOB_ID`, `SCQ_ATTEMPT_NUMBER`, and `SCQ_ATTEMPT_FENCE`. The worker validates those values and emits versioned, fenced `WorkerReportFrame` messages using a four-byte big-endian length followed by a Postcard payload. A frame may not exceed 1 MiB.

The worker accepts only this inherited environment: `PATH`, `CUDA_VISIBLE_DEVICES`, `SCQ_CONTROL_FD`, `SCQ_JOB_ID`, `SCQ_ATTEMPT_NUMBER`, `SCQ_ATTEMPT_FENCE`, `JAIN_ARTIFACT_ROOT`, `JAIN_SESSION_ID`, and `JAIN_RUN_ID`. The daemon is expected to provide a connected descriptor and a valid fenced attempt. An invalid template, environment variable, argument, descriptor, attempt identity, report, or socket write returns an error and does not produce a successful launcher receipt.

This is an internal daemon protocol, not a supported standalone workload CLI. Actual execution, artifact creation, and workload-specific completion need a separate registered handler implementation and end-to-end evidence; the launcher receipt alone proves none of those.
