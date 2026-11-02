"""Stable filesystem and scheduler paths shared by monitor components."""

from pathlib import Path

ROOT = Path("/home/ubuntu/jope-flywheel/test_exact_gen0")
SCQ = Path("/home/ubuntu/smart_cluster/target/release/scq")
SCRIPTS = Path("/home/ubuntu/jope-dime/scripts")
PROPOSALS = ROOT / "proposals" / "dl-proposals.jsonl"
CAMPAIGN = ROOT / "campaign" / "generation-zero.json"
CORPUS = ROOT / "corpus" / "strict-test-exact.sqlite"
LOCK = ROOT / "ops" / "monitor.lock"
EVALUATOR = Path("/home/ubuntu/jope-dime/target/release/corpus_eval")
RESERVE_FLOOR = 5_000
