import copy
import hashlib
import io
import importlib.util
import json
import os
import re
import tomllib
from pathlib import Path
import sys
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import receipt

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('full_main', ROOT / 'deploy/ci/full-main.py')
full = importlib.util.module_from_spec(spec)
spec.loader.exec_module(full)
evidence_spec = importlib.util.spec_from_file_location('aggregate_evidence', ROOT / 'deploy/ci/aggregate-evidence.py')
aggregate_evidence = importlib.util.module_from_spec(evidence_spec)
evidence_spec.loader.exec_module(aggregate_evidence)
partition_spec = importlib.util.spec_from_file_location('check_test_partition', ROOT / 'deploy/ci/check_test_partition.py')
partition = importlib.util.module_from_spec(partition_spec)
partition_spec.loader.exec_module(partition)

__all__ = ['copy', 'hashlib', 'io', 'importlib', 'json', 'os', 're', 'tomllib', 'Path', 'sys', 'subprocess', 'tarfile', 'tempfile', 'unittest', 'patch', 'receipt', 'ROOT', 'full', 'aggregate_evidence', 'partition']
