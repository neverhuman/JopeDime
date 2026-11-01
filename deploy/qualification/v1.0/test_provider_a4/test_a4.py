"""Expose the existing A4 controls to standard Python test discovery."""

from pathlib import Path
import runpy
import unittest

A4Tests = runpy.run_path(str(Path(__file__).with_name("__main__.py")))["A4Tests"]

if __name__ == "__main__":
    unittest.main()
