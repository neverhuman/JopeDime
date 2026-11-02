from pathlib import Path

SCQ = Path("/home/ubuntu/smart_cluster/target/release/scq")
ROOT = Path("/home/ubuntu/jope-flywheel/test_exact_gen0")
BACKLOG = ROOT / "ops" / "backlog.sqlite"
CORPUS = ROOT / "corpus" / "strict-test-exact.sqlite"
RECEIPTS = ROOT / "receipts"
SCQ_RUNS = Path("/home/ubuntu/smart_cluster/var/runs")
HOF_POINTER = ROOT / "hof" / "current.json"
GOLDEN42 = Path("/home/ubuntu/golden42/golden42_foundry/data/panel.json")
CATALOG_POINTER = ROOT / "catalog" / "current.json"
