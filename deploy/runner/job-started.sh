#!/usr/bin/env bash
# Root-installed, read-only hook; no checkout code or job-provided Python imports.
exec /usr/bin/python3 -I /opt/jope-runner/job-policy.py
