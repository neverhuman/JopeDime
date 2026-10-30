#!/usr/bin/env python3
"""Mock-only AG1 setup scenario for Gemini CLI."""

from mock_harness import run_golden_path


if __name__ == "__main__":
    run_golden_path("gemini_cli")
