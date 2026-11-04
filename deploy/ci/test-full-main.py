#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from full_main_suite_workflow import FullMainTests, FullAggregateTests  # noqa: E402
from full_main_suite_build import InputBuildTests, FittedMatrixTests, SyntheticExecutorTests, CompilerCacheTests  # noqa: E402

if __name__ == '__main__':
    unittest.main()
